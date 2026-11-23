"""One resumable entrypoint for scheduled collection and admin dataset imports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
import yaml
from filelock import FileLock
from .config import ROOT, absolute, snapshot_config
from .state import save
from .runner import execute, cancel_pending


STAGES = ("collection", "canonical", "chunk", "embedding", "retrieval")
# The coordinator owns these; a stage option may never replace them.
RESERVED = {"--config", "--mode", "--run-id", "--result", "--input", "--deleted"}


def parse_stage_options(values, mode):
    """'stage:--name[=value]' -> {stage: [arguments]}; opaque to the coordinator."""
    options = {}
    for value in values:
        stage, separator, argument = value.partition(":")
        if not separator or stage not in STAGES:
            raise ValueError(f"--stage-option must be STAGE:--OPTION with STAGE in {', '.join(STAGES)}")
        if not argument.startswith("--") or argument.split("=", 1)[0] in RESERVED or any(c.isspace() for c in argument):
            raise ValueError(f"Invalid stage option {argument!r}: use one --name or --name=value token, not a coordinator option")
        if stage == "collection" and mode != "collect":
            raise ValueError("Collection options require --mode collect")
        options.setdefault(stage, []).append(argument)
    return options


def main(argv=None, *, on_run=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "pipeline.yaml")
    parser.add_argument("--mode", choices=["collect", "join", "reset_data"], default="collect")
    parser.add_argument("--input", type=Path, help="Explicit dataset inside 01_raw; required in join/reset_data mode")
    parser.add_argument("--deleted", type=Path, help="Defaults to 01_raw/chatbot_dataset_deleted.parquet if present")
    parser.add_argument("--resume", type=Path, help="Resume a pipeline run directory")
    parser.add_argument("--resume-pending", action="store_true", help="Resume the pending run automatically (scheduler)")
    parser.add_argument("--cancel-pending", action="store_true", help="Cancel a collection run only if no stage has committed")
    parser.add_argument("--plan", action="store_true", help="Describe steps; no network, models or writes")
    parser.add_argument("--stage-option", action="append", default=[], metavar="STAGE:--OPTION[=VALUE]",
                        help="Pass a stage's own public option through unchanged, e.g. "
                             "collection:--sources=news. Stored with the run, reused on resume")
    args = parser.parse_args(argv)
    try:
        stage_options = parse_stage_options(args.stage_option, args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    if stage_options and (args.resume or args.resume_pending or args.cancel_pending):
        parser.error("--stage-option applies to a new run; a resumed run keeps its original options")
    if args.cancel_pending and (args.resume or args.resume_pending or args.input or args.deleted or args.plan):
        parser.error("--cancel-pending must be used separately")
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if args.plan:
        canonical_step = "clean batch + replacement base (next version, no merge)" if args.mode == "reset_data" else "clean batch + cumulative canonical base/deletions"
        print(f"collection (collect mode only) -> {canonical_step} -> batch chunking -> batch embedding -> Qdrant/BM25 sync")
        return
    state_root = absolute(config["state_dir"])
    state_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(state_root / ".pipeline.lock"), timeout=0):
        if args.cancel_pending:
            cancel_pending(state_root)
            return
        pending = state_root / "pending.json"
        if pending.exists():
            expected = Path(json.loads(pending.read_text(encoding="utf-8"))["run_dir"])
            if args.resume_pending and args.resume is None:
                args.resume = expected
            if args.resume is None or args.resume.resolve() != expected:
                raise RuntimeError(f"A run is pending. Resume with: python run_pipeline.py --resume \"{expected}\"")
        if args.resume:
            folder = args.resume.resolve()
            if folder.parent != state_root or not folder.name.startswith("run_"):
                raise ValueError("--resume must name a run inside the configured state directory")
            state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
            if state.get("status") == "cancelled":
                raise ValueError("This run was cancelled; start a new run")
            if state.get("status") in {"complete", "no_changes"}:
                pending.unlink(missing_ok=True)
                print(f"Pipeline already {state['status']}: {folder}", flush=True)
                return
        else:
            if args.mode in {"join", "reset_data"} and args.input is None:
                parser.error(f"--mode {args.mode} requires --input")
            if args.mode == "reset_data" and args.deleted:
                parser.error("--mode reset_data does not accept --deleted")
            if args.mode == "collect" and args.input:
                parser.error("Use --mode join with --input")
            raw_dir = absolute(config["raw_dir"])
            source = args.input.resolve() if args.input else None
            if source and (not source.is_file() or source.parent != raw_dir or source.name.endswith("_deleted.parquet")):
                raise ValueError("--input must be a non-deletion Parquet file directly inside 01_raw")
            deletion = args.deleted.resolve() if args.deleted else None
            if args.deleted and not deletion.is_file():
                raise FileNotFoundError(deletion)
            run_id = uuid4().hex
            folder = state_root / f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{run_id}"
            folder.mkdir()
            for name in ("collection", "canonical", "chunk", "embedding", "retrieval"):
                snapshot_config(absolute(config[f"{name}_config"]), folder / f"{name}.yaml", name, mode=args.mode)
            defaults = {"collection": ["wp_tracker.stage_cli"], "canonical": ["canonical_pipeline.stage_cli"],
                        "chunk": ["chunking_pipeline.stage_cli"], "embedding": ["embedding.stage_cli"],
                        "retrieval": ["retrieval.stage_cli"]}
            commands = {name: config.get("stage_commands", {}).get(name, default) for name, default in defaults.items()}
            if any(not isinstance(value, list) or not value or not all(isinstance(part, str) and part for part in value)
                   for value in commands.values()):
                raise ValueError("stage_commands entries must be nonempty module/argument lists")
            state = dict(protocol=2, commands=commands, run_id=run_id, mode=args.mode, status="running", completed={},
                         raw_source=str(source) if source else None,
                         deleted_source=str(deletion) if deletion and deletion.is_file() else None,
                         stage_options=stage_options)
            save(folder / "state.json", state)
        save(pending, {"run_dir": str(folder)})
        if on_run is not None:
            on_run(folder)
        try:
            execute(folder, state)
        except BaseException:
            print(f'Resume: python run_pipeline.py --resume "{folder}"', flush=True)
            raise
        pending.unlink(missing_ok=True)
        print(f"Pipeline {state['status']}: {folder}", flush=True)
