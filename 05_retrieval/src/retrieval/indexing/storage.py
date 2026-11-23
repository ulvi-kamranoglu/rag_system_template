"""Archive complete local retrieval states; remote Qdrant is not transactional here."""
import argparse
from datetime import datetime, timezone
from functools import wraps
import json
from pathlib import Path, PureWindowsPath
import shutil
import os

from filelock import FileLock
import polars as pl
from pipeline_storage import (archive_run, inventory, publish, recover_publication,
                              run_path, write_json)
from retrieval import config


INDEX_SUFFIXES = ("_bm25_corpus.parquet", "_bm25_index_metadata.json",
                  "_delta_sync_metadata.json", "_qdrant_index_metadata.json",
                  "_faiss_mapping.parquet", "_faiss_index_metadata.json", ".faiss")


def prune_inactive_indexes(folder, dataset):
    """Drop old index artifacts only after the new full corpus has been built."""
    active = folder / "active_dataset.json"
    if not active.is_file() or json.loads(active.read_text(encoding="utf-8"))["dataset_name"] != dataset:
        raise ValueError("New active dataset pointer must be committed before pruning indexes")
    for path in folder.iterdir():
        if not path.is_file():
            continue
        for suffix in INDEX_SUFFIXES:
            if path.name.endswith(suffix):
                if path.name[:-len(suffix)] != dataset:
                    path.unlink()
                break


def validate_run(folder: Path):
    report = json.loads((folder / "run_report.json").read_text(encoding="utf-8"))
    actual = inventory(folder)
    files = {key: value for key, value in actual.items() if value is not None and key != "run_report.json"}
    if files != report["file_hashes"]:
        raise ValueError("Retrieval archive file inventory/hash mismatch")
    if not set(report["changed_files"]).issubset(files):
        raise ValueError("Retrieval archive is missing stage outputs")
    for name in files:
        path = folder / name
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".parquet":
            frame = pl.read_parquet(path)
            if not {"document_id", "chunk_id"}.issubset(frame.columns):
                raise ValueError(f"Invalid retrieval table: {name}")
            if frame["chunk_id"].null_count() or frame["chunk_id"].n_unique() != frame.height:
                raise ValueError(f"Invalid retrieval chunk IDs: {name}")
    for name in report["changed_files"]:
        path = folder / name
        if name.endswith("_faiss_index_metadata.json"):
            import faiss
            metadata = json.loads(path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(folder / PureWindowsPath(metadata["index_file"]).name))
            mapping = pl.read_parquet(folder / PureWindowsPath(metadata["mapping_file"]).name)
            if index.ntotal != mapping.height or index.ntotal != metadata["vectors"] or index.d != metadata["dimension"]:
                raise ValueError("Archived FAISS index/mapping mismatch")
        elif name.endswith("_bm25_index_metadata.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            corpus = pl.read_parquet(folder / PureWindowsPath(metadata["corpus_file"]).name)
            if corpus.height != metadata["documents"]:
                raise ValueError("Archived BM25 corpus/metadata mismatch")
        elif name.endswith("_delta_sync_metadata.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            corpus_name = next(n for n in report["changed_files"] if n.endswith("_bm25_corpus.parquet"))
            if pl.read_parquet(folder / corpus_name).height != metadata["bm25_corpus_chunks"]:
                raise ValueError("Archived delta corpus/metadata mismatch")
            if metadata["qdrant_after"] != metadata["bm25_corpus_chunks"]:
                raise ValueError("Archived Qdrant/BM25 delta counts differ")
        elif name.endswith("_qdrant_index_metadata.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata["final_point_count"] != metadata["indexed_points"] or metadata["final_point_count"] < 0:
                raise ValueError("Archived Qdrant index counts differ")


def _roots():
    archive, shared = config.ARCHIVE_DIR.resolve(), config.INDEXES_DIR.resolve()
    if archive == shared or archive.is_relative_to(shared) or shared.is_relative_to(archive):
        raise ValueError("Retrieval archive and shared roots must not overlap")
    package_root = Path(__file__).resolve().parents[1]
    module_root = package_root.parent.parent if package_root.parent.name == "src" else package_root
    if archive.is_relative_to(module_root) or shared.is_relative_to(module_root):
        raise ValueError("Retrieval outputs must be centralized outside the module")
    return archive, shared


def _publish(target, shared, report, migrate_legacy):
    try:
        status = publish(target, shared, validate_run, migrate_legacy=migrate_legacy)
    except Exception as exc:
        raise RuntimeError(
            f"Local retrieval archive is complete: {target}. Publication failed; do not repeat "
            f"Qdrant mutations just to retry local files. Retry: python -m retrieval.storage "
            f'--publish-run "{target}". Qdrant changes are not rolled back. {exc}'
        ) from exc
    result = {**report, "publication": status}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def retry_publication(target, *, operation=None, migrate_legacy=False):
    archive, shared = _roots()
    target = Path(target).resolve()
    if target.parent != archive or not target.name.startswith("run_"):
        raise ValueError("--publish-run must name a run in archive_data/05_indexes")
    shared.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(shared.with_name(".05_indexes.writer.lock")), timeout=0):
        recover_publication(shared)
        validate_run(target)
        report = json.loads((target / "run_report.json").read_text(encoding="utf-8"))
        if operation and report["operation"] != operation:
            raise ValueError("Archive belongs to another retrieval stage")
        current = inventory(shared) if shared.exists() else {}
        if current != inventory(target) and current != report["base_shared_inventory"]:
            raise RuntimeError("Shared retrieval state changed after this run; refusing to overwrite newer results")
        return _publish(target, shared, report, migrate_legacy)


def archived_stage(operation, output_names):
    """Run an existing stage in staging, carrying forward other local indexes."""
    def decorate(function):
        @wraps(function)
        def wrapped(argv=None):
            parser = argparse.ArgumentParser(description=f"Archived retrieval stage: {operation}")
            parser.add_argument("--publish-run", help="Retry local publication only; no Qdrant requests")
            parser.add_argument("--migrate-legacy-shared", action="store_true")
            args = parser.parse_args(argv)
            if args.publish_run:
                return retry_publication(args.publish_run, operation=operation,
                                         migrate_legacy=args.migrate_legacy_shared)
            archive, shared = _roots()
            shared.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(shared.with_name(".05_indexes.writer.lock")), timeout=0):
                recover_publication(shared)
                previous_report = shared / "run_report.json"
                if previous_report.is_file():
                    previous = json.loads(previous_report.read_text(encoding="utf-8"))
                    if "file_hashes" in previous:
                        validate_run(shared)
                base = inventory(shared) if shared.exists() else {}
                legacy = [p for p in shared.iterdir() if p.is_dir() and
                          p.name.startswith(("run_", "bootstrap_", "previous_"))] if shared.exists() else []
                if legacy and not args.migrate_legacy_shared:
                    raise RuntimeError("Legacy shared run folders require --migrate-legacy-shared before indexing")
                target = run_path(archive)
                if any(getattr(config, key).resolve().parent != shared for key in output_names):
                    raise ValueError("Retrieval stage outputs must be inside the configured shared index directory")
                expected = [getattr(config, key).name for key in output_names]
                with archive_run(target, validate_run) as staging:
                    if shared.exists():
                        shutil.copytree(shared, staging, dirs_exist_ok=True)
                    if (staging / "run_report.json").exists():
                        (staging / "run_report.json").unlink()
                    # Never let a carried-forward output hide a stage failing to write it.
                    # Delta's existing corpus is its input and is therefore retained.
                    for name in expected:
                        if operation in {"delta", "sync"} and name.endswith("_bm25_corpus.parquet"):
                            continue
                        candidate = staging / name
                        if candidate.exists():
                            candidate.unlink()
                    function(staging)
                    for name in expected:
                        if not (staging / name).is_file():
                            raise ValueError(f"Stage did not produce expected file: {name}")
                    if operation == "sync":
                        prune_inactive_indexes(staging, config.DATASET_NAME)
                    report = {"operation": operation, "dataset_name": config.DATASET_NAME,
                              "batch_dataset_name": os.getenv("PIPELINE_BATCH_DATASET"),
                              "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
                              "created_at": datetime.now(timezone.utc).isoformat(),
                              "run_dir": str(target), "archive_dir": str(archive), "shared_dir": str(shared),
                              "qdrant_collection": config.QDRANT_COLLECTION if operation in {"qdrant", "delta", "sync"} else None,
                              "qdrant_snapshot": False, "changed_files": expected,
                              "base_shared_inventory": base,
                              "file_hashes": {key: value for key, value in inventory(staging).items() if value is not None}}
                    write_json(staging / "run_report.json", report)
                return _publish(target, shared, report, args.migrate_legacy_shared)
        return wrapped
    return decorate


def main():
    parser = argparse.ArgumentParser(description="Retry archived local retrieval publication without Qdrant access")
    parser.add_argument("--publish-run", required=True)
    parser.add_argument("--migrate-legacy-shared", action="store_true")
    args = parser.parse_args()
    retry_publication(args.publish_run, migrate_legacy=args.migrate_legacy_shared)


if __name__ == "__main__":
    main()
