"""Bootstrap and full-snapshot change tracking orchestration."""
from __future__ import annotations

import logging
import json
import os
from contextlib import nullcontext
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from filelock import FileLock, Timeout
from pipeline_storage import archive_run, write_json

from .api import SOURCES, WordPressClient
from .config import Settings
from .diff import compare
from .mirror import mirror_run, validate_run
from .records import from_api, from_baseline
from .storage import commit, read_rows, recover, summarize_files, write_parquet
from .text import initial_cleaning_rejection

LOG = logging.getLogger(__name__)


def _archive(settings, run_dir, report, new, deleted):
    with archive_run(run_dir, validate_run) as staging:
        write_parquet(staging / "chatbot_dataset_new.parquet", new, "new")
        write_parquet(staging / "chatbot_dataset_deleted.parquet", deleted, "deleted")
        write_json(staging / "latest.json", report)
        write_json(staging / "run_report.json", report)


def _guard_deletions(previous: list[dict], deleted: list[dict], settings: Settings, allow_large: bool) -> None:
    if allow_large or not previous:
        return
    old_counts = Counter(row["source_type"] for row in previous)
    deleted_counts = Counter(row["source_type"] for row in deleted)
    for kind, count in deleted_counts.items():
        fraction = count / old_counts[kind]
        if fraction > settings.max_deleted_fraction:
            raise RuntimeError(
                f"Safety stop: {count}/{old_counts[kind]} ({fraction:.1%}) {kind} records disappeared. "
                "Check the WordPress API and baseline before retrying with "
                "--allow-large-deletions. No files have been written."
            )


def scan(settings: Settings, run: bool = False, allow_large_deletions: bool = False,
         client: WordPressClient | None = None, *, migrate_legacy: bool = False) -> dict:
    """A complete scan is mandatory to safely classify absent IDs as deleted."""
    if run:
        settings.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(settings.lock_file), timeout=0) if run else nullcontext():
            if run:
                recover(settings)
                previous_report = settings.reports_dir / "latest.json"
                if previous_report.is_file():
                    previous = json.loads(previous_report.read_text(encoding="utf-8"))
                    # Finish delivery before replacing the retry pointer with a new scan.
                    mirror_run(settings, Path(previous["run_dir"]), migrate_legacy=migrate_legacy)
            elif (settings.state_dir / ".pending").exists():
                raise RuntimeError("Pending transaction found. Run 'wp_tracker recover' before dry-run.")
            if not settings.baseline_file.is_file():
                raise RuntimeError(
                    "Immutable baseline is missing. Run 'python -m wp_tracker bootstrap --run' "
                    "for a new project, or 'python -m wp_tracker bootstrap --reset --run' "
                    "to rebuild it from approved imports. Back up existing state first."
                )
            baseline = read_rows(settings.baseline_file)
            previous_snapshot = read_rows(settings.snapshot_file)
            api = client or WordPressClient(settings)
            current = []
            api_counts = {}
            for source in SOURCES:
                items = api.fetch_all(source)
                api_counts[source] = len(items)
                current.extend(from_api(item, source, settings) for item in items)
            changes = compare(baseline, current)
            # Keep all IDs in the baseline/snapshot for reliable deletion checks.
            # Quality filters only determine which NEW records are exported.
            rejected = Counter()
            accepted_new = []
            for row in changes.new:
                reason = initial_cleaning_rejection(row["content"])
                if reason:
                    rejected[reason] += 1
                else:
                    accepted_new.append(row)
            _guard_deletions(baseline, changes.deleted, settings, allow_large_deletions)
            run_dir = settings.new_run_dir()
            outputs = {
                name: str(settings.output_file(name, run_dir)) if run_dir else None
                for name in ("new", "deleted")
            }
            summary = {
                "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
                "mode": "run" if run else "dry_run",
                "run_dir": str(run_dir) if run_dir else None,
                "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
                "api_counts": api_counts,
                "baseline_records": len(baseline),
                "snapshot_before": len(previous_snapshot),
                "snapshot_after": len(current),
                "new": len(accepted_new),
                "new_candidates": len(changes.new),
                "filtered_new": sum(rejected.values()),
                "filtered_new_reasons": dict(sorted(rejected.items())),
                "deleted": len(changes.deleted),
                "outputs": outputs,
                "archive_dir": str(settings.output_dir),
                "shared_dir": str(settings.shared_raw_dir),
            }
            if run:
                _archive(settings, run_dir, summary, accepted_new, changes.deleted)
                commit(settings, {
                    settings.snapshot_file: ("snapshot", current),
                    settings.reports_dir / "latest.json": summary,
                })
                try:
                    summary["mirror"] = mirror_run(settings, run_dir, migrate_legacy=migrate_legacy)
                except Exception as exc:
                    raise RuntimeError(
                        f"Local run committed at {run_dir}, but shared mirror failed: {exc}. "
                        "Local outputs are preserved. Retry with 'python -m wp_tracker sync-shared'."
                    ) from exc
            return summary
    except Timeout as exc:
        raise RuntimeError("Another tracker process is already running") from exc


def bootstrap(settings: Settings, pages: Path | None = None, posts: Path | None = None,
              run: bool = False, reset: bool = False, *, migrate_legacy: bool = False) -> dict:
    """Create or explicitly reset the immutable baseline from approved imports."""
    pages = pages or settings.import_dir / "pages.parquet"
    posts = posts or settings.import_dir / "posts.parquet"
    if run:
        settings.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(settings.lock_file), timeout=0) if run else nullcontext():
            if run:
                recover(settings)
            elif (settings.state_dir / ".pending").exists():
                raise RuntimeError("Pending transaction found. Run 'wp_tracker recover' first.")
            if (settings.snapshot_file.exists() or settings.baseline_file.exists()) and not reset:
                raise FileExistsError(
                    "Tracker already initialized. To replace the baseline from approved imports, "
                    "back up current state and pass --reset explicitly."
                )
            source_frames = {}
            rows = []
            for source, input_path in (("pages", pages), ("posts", posts)):
                input_path = Path(input_path)
                if not input_path.is_file():
                    raise FileNotFoundError(f"Baseline not found: {input_path}")
                frame = pd.read_parquet(input_path)
                source_frames[source] = len(frame)
                for original in frame.to_dict("records"):
                    rows.append(from_baseline(original, source, settings))
            # Detect duplicate IDs before writing.
            compare([], rows)
            run_dir = settings.new_run_dir("bootstrap")
            result = {
                "mode": "run" if run else "dry_run",
                "run_dir": str(run_dir) if run_dir else None,
                "outputs": {
                    name: str(settings.output_file(name, run_dir)) if run_dir else None
                    for name in ("new", "deleted")
                },
                "baseline_records": len(rows),
                "source_counts": source_frames,
                "baseline": str(settings.baseline_file),
                "snapshot": str(settings.snapshot_file),
                "reset": reset,
                "new": 0, "deleted": 0,
                "operation": "bootstrap",
                "archive_dir": str(settings.output_dir),
                "shared_dir": str(settings.shared_raw_dir),
            }
            if run:
                _archive(settings, run_dir, result, [], [])
                commit(settings, {
                    settings.baseline_file: ("snapshot", rows),
                    settings.snapshot_file: ("snapshot", rows),
                    settings.reports_dir / "latest.json": result,
                })
                try:
                    result["mirror"] = mirror_run(settings, run_dir, migrate_legacy=migrate_legacy)
                except Exception as exc:
                    raise RuntimeError(
                        f"Local run committed at {run_dir}, but shared mirror failed: {exc}. "
                        "Retry with 'python -m wp_tracker sync-shared'."
                    ) from exc
            return result
    except Timeout as exc:
        raise RuntimeError("Another tracker process is already running") from exc


def status(settings: Settings) -> dict:
    if (settings.state_dir / ".pending").exists():
        raise RuntimeError("Pending transaction found. Run 'python -m wp_tracker recover' first.")
    return summarize_files(settings)


def recover_state(settings: Settings) -> dict:
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(settings.lock_file), timeout=0):
            pending = (settings.state_dir / ".pending").is_dir()
            recover(settings)
            return {"recovered_or_cleaned": pending, **summarize_files(settings)}
    except Timeout as exc:
        raise RuntimeError("Another tracker process is already running") from exc
