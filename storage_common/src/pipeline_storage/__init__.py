"""Filesystem-only storage shared by independently installed pipeline projects.

On Windows replacing a nonempty directory is not atomic. Publication uses two
same-parent renames with a durable journal and rollback. Readers may observe a
brief missing directory, but never a mixture of two runs. Call recover/publish
after a terminated writer before consuming shared outputs.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import re
from uuid import uuid4

from filelock import FileLock

from .fs import rename_with_retry


def dataset_stem(path: str | Path) -> str:
    """Remove stage suffixes while retaining source version dots."""
    name = Path(path).stem
    return re.sub(r"(?:_canonical|_chunks_\d+)$", "", name)


def batch_stem(inputs) -> str:
    return "__".join(dataset_stem(path) for path in inputs)


def select_input(folder: Path, pattern: str, *, exclude_deleted: bool = False) -> Path:
    paths = sorted(p for p in folder.glob(pattern) if p.is_file() and
                   not (exclude_deleted and p.stem.endswith("_deleted")))
    if len(paths) != 1:
        raise ValueError(f"Expected one input matching {folder / pattern}, found {len(paths)}; "
                         "select the input explicitly with --input or YAML inputs")
    return paths[0]


def artifact_dataset(shared: Path) -> str:
    """Resolve one dataset for 04_embedding/indexing; never choose an arbitrary file."""
    explicit = os.getenv("PIPELINE_DATASET")
    if explicit:
        if explicit in {".", ".."} or any(c in explicit for c in '/\\:*?"<>|'):
            raise ValueError("PIPELINE_DATASET must be a filename stem, not a path")
        return explicit
    for stage, suffix in (("03_chunks", "_chunks_384.parquet"),
                          ("04_embeddings", "_embedding_mapping.parquet"),
                          ("05_indexes", "_bm25_corpus.parquet")):
        paths = sorted((shared / stage).glob(f"*{suffix}"))
        if len(paths) > 1:
            raise ValueError("Multiple datasets found; set PIPELINE_DATASET explicitly")
        if paths:
            return paths[0].name[:-len(suffix)]
    return "dataset"


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def default_config(module_file: str, relative: str) -> Path:
    """Locate workspace configuration regardless of the terminal directory."""
    parents = Path(module_file).resolve().parents
    for parent in parents:
        candidate = parent / relative
        if (parent / "pyproject.toml").is_file() and candidate.is_file():
            return candidate
    # Wheels ship the same default template inside the Python package.
    for parent in parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Default configuration not found: {relative}; pass --config")


def workspace_root(module_file: str) -> Path:
    """Resolve an explicit workspace, a source checkout, or the caller's cwd."""
    explicit = os.getenv("CHATBOT_PROJECT_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    for parent in Path(module_file).resolve().parents:
        if (parent / "pipeline.yaml").is_file():
            return parent
    return Path.cwd().resolve()


def configuration_root(path: Path, module_file: str, project_name: str) -> Path:
    """Keep explicit config-relative paths; wheel defaults target the workspace."""
    package = Path(module_file).resolve().parent
    if path.resolve().is_relative_to(package / "config"):
        return workspace_root(module_file) / project_name
    return path.resolve().parent.parent


def run_path(archive: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return archive.resolve() / f"run_{stamp}_{uuid4().hex}"


def storage_paths(root: Path, config: dict, stage: str) -> tuple[Path, Path]:
    storage = config.get("storage", {})
    archive = resolve(root, storage.get("archive_dir", f"../archive_data/{stage}"))
    shared = resolve(root, storage.get("shared_dir", f"../shared_data/{stage}"))
    if archive == shared or archive.is_relative_to(shared) or shared.is_relative_to(archive):
        raise ValueError("Archive and shared roots must not overlap")
    if archive.is_relative_to(root.resolve()) or shared.is_relative_to(root.resolve()):
        raise ValueError("Outputs must be centralized outside the project directory")
    return archive, shared


def inventory(folder: Path) -> dict:
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f"Not a regular output directory: {folder}")
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Output symlink is not supported: {path}")
        key = path.relative_to(folder).as_posix()
        if path.is_dir():
            result[key] = None
        else:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[key] = digest.hexdigest()
    return result


@contextmanager
def archive_run(destination: Path, validate):
    """Only expose a final immutable run name after all artifacts validate."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.partial")
    staging.mkdir()
    try:
        yield staging
        validate(staging)
        inventory(staging)
        if destination.exists():
            raise FileExistsError(destination)
        rename_with_retry(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _journal(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _paths(shared: Path):
    return (shared.with_name(f".{shared.name}.publish.json"),
            shared.with_name(f".{shared.name}.staging"),
            shared.with_name(f".{shared.name}.previous"))


def _recover(shared: Path) -> None:
    journal, staging, previous = _paths(shared)
    if not journal.exists():
        if previous.exists():
            raise RuntimeError(f"Unjournaled previous output; left untouched: {previous}")
        if staging.exists():
            shutil.rmtree(staging)
        return
    payload = json.loads(journal.read_text(encoding="utf-8"))
    if payload["phase"] != "committed":
        if previous.exists():
            if shared.exists():
                shutil.rmtree(shared)
            rename_with_retry(previous, shared)
        elif not payload["existed"] and shared.exists():
            shutil.rmtree(shared)
    elif previous.exists():
        if payload["preserve_previous"]:
            # Preserve pre-refactor shared data in place beside shared stages.
            # Never silently migrate or delete historical legacy run folders.
            rename_with_retry(previous, shared.with_name(f".{shared.name}.legacy_{uuid4().hex}"))
        else:
            shutil.rmtree(previous)
    if staging.exists():
        shutil.rmtree(staging)
    journal.unlink()


def recover_publication(shared: Path) -> None:
    shared = shared.resolve()
    if not shared.parent.exists():
        return
    with FileLock(str(shared.with_name(f".{shared.name}.publish.lock")), timeout=0):
        _recover(shared)


def matching_archive(folder: Path, archive_root: Path) -> Path | None:
    """Find a committed, byte-identical archive; never trust filenames alone."""
    expected = inventory(folder)
    if not expected or not Path(archive_root).is_dir():
        return None
    for candidate in sorted(Path(archive_root).glob("run_*")):
        if candidate.is_dir() and not candidate.is_symlink():
            try:
                if inventory(candidate) == expected:
                    return candidate
            except (OSError, ValueError):
                continue
    return None


def publish(run: Path, shared: Path, validate, *, migrate_legacy: bool = False,
            previous_archive_root: Path | None = None) -> dict:
    """Publish an entire validated archive, removing stale shared artifacts.

    Complete archives survive every publication failure and may be retried.
    Legacy shared contents without our run report are retained in a hidden
    sibling backup; project-local legacy directories are never touched.
    A caller may provide previous_archive_root to suppress that backup only
    when the complete previous inventory matches a committed archive there.
    """
    run, shared = run.resolve(), shared.resolve()
    if shared.is_relative_to(run) or run.is_relative_to(shared):
        raise ValueError("Archive and shared directories must not overlap")
    validate(run)
    expected = inventory(run)
    shared.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(shared.with_name(f".{shared.name}.publish.lock")), timeout=0):
        _recover(shared)
        legacy_runs = [p for p in shared.iterdir() if p.is_dir() and
                       p.name.startswith(("run_", "bootstrap_", "previous_"))] if shared.exists() else []
        if legacy_runs and not migrate_legacy:
            raise RuntimeError("Legacy run directories are left untouched. Explicitly opt in with "
                               "--migrate-legacy-shared to retain them in a sibling backup.")
        if shared.exists() and inventory(shared) == expected:
            return {"status": "already_synced", "path": str(shared)}
        archived_previous = None
        if previous_archive_root is not None and shared.exists():
            archived_previous = matching_archive(shared, previous_archive_root)
        previous_inventory = inventory(shared) if archived_previous is not None else None
        journal, staging, previous = _paths(shared)
        try:
            shutil.copytree(run, staging)
            validate(staging)
            if inventory(staging) != expected or inventory(run) != expected:
                raise RuntimeError("Publication byte verification failed")
            payload = {"phase": "prepared", "existed": shared.exists(),
                       "preserve_previous": archived_previous is None and (bool(legacy_runs) or
                       (shared.exists() and not (shared / "run_report.json").is_file()))}
            if archived_previous is not None and (inventory(shared) != previous_inventory or
                                                  inventory(archived_previous) != previous_inventory):
                raise RuntimeError("Archived previous data changed during publication")
            _journal(journal, payload)
            if shared.exists():
                rename_with_retry(shared, previous)
            rename_with_retry(staging, shared)
            payload["phase"] = "committed"
            _journal(journal, payload)
        except BaseException:
            _recover(shared)
            raise
        _recover(shared)
    return {"status": "synced", "path": str(shared)}


def validate_json(folder: Path, names) -> None:
    for name in names:
        json.loads((folder / name).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
