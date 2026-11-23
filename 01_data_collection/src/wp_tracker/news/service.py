"""News collection with the WordPress model: fixed baseline, advancing snapshot.

State is two Parquet files plus ``latest.json`` (no SQLite, no ID allocation):
  baseline.parquet  the approved seed, what the index already contains; never changes
  snapshot.parquet  the last complete scan
A scan reads every article number from the sitemap, opens only articles that
are new to the snapshot and confirms each disappearance by URL. NEW and DELETED
are computed against the fixed baseline exactly like WordPress.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from filelock import FileLock, Timeout
import pandas as pd
from pipeline_storage import archive_run, publish, run_path, validate_json, write_json

from ..diff import compare
from ..records import FIELDS
from ..storage import commit, read_rows, recover, write_parquet
from ..text import initial_cleaning_rejection
from . import collector as http
from .schema import seed_row, validate_rows


def _clean(rows):
    """Parquet round trips turn NaT into None/NaN; keep record dictionaries uniform."""
    result = []
    for row in rows:
        item = {key: row[key] for key in FIELDS}
        for key in ("date", "modified"):
            item[key] = None if item[key] is None or pd.isna(item[key]) else pd.Timestamp(item[key])
        result.append(item)
    return result


def _validate_run(folder: Path):
    report = json.loads((folder / "run_report.json").read_text(encoding="utf-8"))
    new = read_rows(folder / "chatbot_dataset_new.parquet", "new")
    deleted = read_rows(folder / "chatbot_dataset_deleted.parquet", "deleted")
    validate_rows(_clean(new)); validate_rows(_clean(deleted))
    if len(new) != report["new"] or len(deleted) != report["deleted"]:
        raise ValueError("News report counts disagree with the archived datasets")
    validate_json(folder, ["latest.json", "run_report.json"])


def _locked(settings):
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(settings.lock_file), timeout=0)


def mirror(settings, run_dir: Path):
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != settings.archive_dir.resolve():
        raise ValueError("News run directory is outside its own archive")
    return publish(run_dir, settings.shared_dir, _validate_run)


def _state(settings):
    if not settings.baseline_file.is_file() or not settings.snapshot_file.is_file():
        raise RuntimeError("News baseline/snapshot missing; run 'news bootstrap --run' with data/import/news.parquet")
    baseline, snapshot = _clean(read_rows(settings.baseline_file)), _clean(read_rows(settings.snapshot_file))
    validate_rows(baseline); validate_rows(snapshot)
    return baseline, snapshot


def status(settings):
    initialized = settings.baseline_file.is_file() and settings.snapshot_file.is_file()
    result = {"initialized": initialized, "baseline": str(settings.baseline_file),
              "snapshot": str(settings.snapshot_file), "seed_file": str(settings.seed_file),
              "comparison": "fixed_baseline"}
    if initialized:
        result.update(baseline_records=len(read_rows(settings.baseline_file)),
                      snapshot_records=len(read_rows(settings.snapshot_file)))
    if settings.report_file.is_file():
        result["last_run"] = json.loads(settings.report_file.read_text(encoding="utf-8"))
    return result


def bootstrap(settings, seed=None, *, run=False):
    """Create baseline and snapshot from the approved, already joined dataset."""
    seed = Path(seed) if seed else settings.seed_file
    if not seed.is_file():
        raise FileNotFoundError(f"News seed not found: {seed} (place it in data/import/news.parquet)")
    rows = _clean([seed_row(row) for row in pd.read_parquet(seed).to_dict("records")])
    if not rows:
        raise ValueError("News seed must not be empty")
    validate_rows(rows)
    outcome = {"operation": "bootstrap", "seed": str(seed.resolve()), "records": len(rows),
               "baseline": str(settings.baseline_file), "run": bool(run)}
    if settings.baseline_file.exists() or settings.snapshot_file.exists():
        raise FileExistsError("News state already initialized; the fixed baseline is never replaced")
    if not run:
        return outcome
    try:
        with _locked(settings):
            if settings.baseline_file.exists():
                raise FileExistsError("News state was initialized concurrently")
            commit(settings, {settings.baseline_file: ("snapshot", rows),
                              settings.snapshot_file: ("snapshot", rows)})
    except Timeout as exc:
        raise RuntimeError("Another News operation is running") from exc
    return outcome


def _guard_deletions(baseline, deleted, settings, allow_large):
    if allow_large or not baseline:
        return
    fraction = len(deleted) / len(baseline)
    if fraction > settings.max_deleted_fraction:
        raise RuntimeError(f"Safety stop: {len(deleted)}/{len(baseline)} ({fraction:.1%}) news articles "
                           "disappeared. Check the site before retrying with --allow-large-deletions.")


def collect(settings, snapshot, *, client=None, allow_large_deletions=False):
    """Network part of a scan: the complete current article set.

    Known articles keep their snapshot row; only new numbers are opened. A known
    article absent from the sitemap stays unless its URL answers 404/410.
    """
    api = client or http.Client(settings.delay_seconds)
    try:
        by_number = {row["id"].split(":", 1)[1]: row for row in snapshot}
        listed = http.sitemap_articles(api, {n: r["url"] for n, r in by_number.items()})
        missing = [n for n in by_number if n not in listed]
        # A shrunken sitemap is a site problem, not mass deletion; stop before probing.
        if snapshot and len(missing) / len(snapshot) > settings.max_deleted_fraction and not allow_large_deletions:
            raise RuntimeError(f"Safety stop: sitemap lacks {len(missing)}/{len(snapshot)} known articles; "
                               "nothing was written")
        current, unparsed, unconfirmed, confirmed_deleted = [], [], [], []
        for number, row in by_number.items():
            if number in listed:
                current.append(row)
                continue
            verdict = http.confirm_presence(api, row["url"])
            if verdict == "deleted":
                confirmed_deleted.append(row["id"])
            else:
                current.append(row)  # present or uncertain: never delete on doubt
                if verdict == "unknown":
                    unconfirmed.append(row["id"])
        for number in sorted(set(listed) - set(by_number), key=int):
            try:
                current.append(http.fetch_article(api, listed[number]))
            except http.CollectionIncomplete as exc:
                unparsed.append({"url": listed[number], "reason": str(exc)})  # retried next scan
    finally:
        if client is None:
            api.close()
    return {"current": current, "sitemap_articles": len(listed), "confirmed_deleted": confirmed_deleted,
            "unconfirmed_missing": unconfirmed, "unparsed_new": unparsed}


def scan(settings, *, run=False, client=None, pipeline_run_id=None, allow_large_deletions=False):
    if not run:
        return {"dry_run": True, "network_requested": False, **status(settings)}
    try:
        with _locked(settings):
            recover(settings)
            if settings.report_file.is_file():
                # Finish an interrupted delivery before starting a new scan.
                mirror(settings, Path(json.loads(settings.report_file.read_text(encoding="utf-8"))["run_dir"]))
            baseline, snapshot = _state(settings)
            found = collect(settings, snapshot, client=client, allow_large_deletions=allow_large_deletions)
            current = _clean(found["current"])
            validate_rows(current)
            changes = compare(baseline, current)
            accepted_new = [row for row in changes.new if not initial_cleaning_rejection(row["content"])]
            _guard_deletions(baseline, changes.deleted, settings, allow_large_deletions)
            run_dir = run_path(settings.archive_dir)
            summary = {
                "pipeline_run_id": pipeline_run_id, "mode": "scan", "run_dir": str(run_dir),
                "scanned_at_utc": datetime.now(timezone.utc).isoformat(), "comparison": "fixed_baseline",
                "sitemap_articles": found["sitemap_articles"], "baseline_records": len(baseline),
                "snapshot_before": len(snapshot), "snapshot_after": len(current),
                "new": len(accepted_new), "new_candidates": len(changes.new),
                "filtered_new": len(changes.new) - len(accepted_new), "deleted": len(changes.deleted),
                "confirmed_deleted_this_scan": found["confirmed_deleted"],
                "unconfirmed_missing": found["unconfirmed_missing"], "unparsed_new": found["unparsed_new"],
                "shared_dir": str(settings.shared_dir),
            }
            with archive_run(run_dir, _validate_run) as staging:
                write_parquet(staging / "chatbot_dataset_new.parquet", accepted_new, "new")
                write_parquet(staging / "chatbot_dataset_deleted.parquet", changes.deleted, "deleted")
                write_json(staging / "latest.json", summary)
                write_json(staging / "run_report.json", summary)
            commit(settings, {settings.snapshot_file: ("snapshot", current), settings.report_file: summary})
            try:
                summary["mirror"] = mirror(settings, run_dir)
            except Exception as exc:
                raise RuntimeError(f"News run committed at {run_dir}; shared publish failed: {exc}. "
                                   "Retry 'news sync-shared --run'.") from exc
            return summary
    except Timeout as exc:
        raise RuntimeError("Another News operation is running") from exc


def sync_shared(settings, *, run=False):
    if not run:
        return {"dry_run": True, **status(settings)}
    try:
        with _locked(settings):
            recover(settings)
            if not settings.report_file.is_file():
                raise RuntimeError("Nothing committed to publish")
            report = json.loads(settings.report_file.read_text(encoding="utf-8"))
            return mirror(settings, Path(report["run_dir"]))
    except Timeout as exc:
        raise RuntimeError("Another News operation is running") from exc
