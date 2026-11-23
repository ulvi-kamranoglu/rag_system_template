"""Instagram weekly pipeline with WordPress-style immutable runs and shared mirrors.

This module has a separate namespace. WordPress baseline / snapshot / shared
files are never touched. Scheduled jobs use scan --run, which does not prompt.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from filelock import FileLock, Timeout
from pipeline_storage import archive_run, publish, run_path, validate_json

from ..storage import commit, read_rows, recover, write_parquet
from .browser import collect
from .schema import (
    FIELDS, assign_sequential_ids, load_seed, new_row, utc_datetime, validate_rows,
)


def _time() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            raise RuntimeError(f"Invalid JSONL at {path} line {line_no}") from exc
    return rows


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_run(folder: Path):
    report = json.loads((folder / 'run_report.json').read_text(encoding='utf-8'))
    tables = {}
    for name in ("new", "deleted", "snapshot"):
        path = folder / f"chatbot_dataset_{name}.parquet"
        rows = read_rows(path, "snapshot")
        tables[name] = rows
        if name != 'deleted':
            validate_rows(rows, report['profile'])
        if name == "deleted" and rows:
            raise ValueError("Instagram incremental scanner must never produce deletions")
        if name != "deleted" and rows and list(rows[0]) != FIELDS and set(rows[0]) != set(FIELDS):
            raise ValueError(f"Instagram {name} dataset is missing canonical columns")
    validate_json(folder, ["latest.json", "run_report.json"])
    if len(tables['new']) != report['new'] or len(tables['snapshot']) != report['snapshot_after']:
        raise ValueError('Instagram report counts differ from the archive')
    members = {row['id']: row['url'] for row in tables['snapshot']}
    if any(members.get(row['id']) != row['url'] for row in tables['new']):
        raise ValueError('Instagram NEW is not contained in SNAPSHOT')


def _archive(settings, report, new, cumulative):
    run_dir = Path(report["run_dir"])
    with archive_run(run_dir, _validate_run) as staging:
        write_parquet(staging / "chatbot_dataset_new.parquet", new, "new")
        write_parquet(staging / "chatbot_dataset_deleted.parquet", [], "deleted")
        write_parquet(staging / "chatbot_dataset_snapshot.parquet", cumulative, "snapshot")
        _write_json(staging / "latest.json", report)
        _write_json(staging / "run_report.json", report)


def _mirror(settings, report):
    # Own profile-specific shared directory so WordPress publisher can never
    # erase Instagram output during a full-tree replacement.
    run_dir = Path(report["run_dir"]).resolve()
    if run_dir.parent != settings.archive_dir.resolve():
        raise ValueError("Run directory is outside Instagram archive")
    return publish(run_dir, settings.shared_dir, _validate_run)


def _finish_previous(settings):
    previous = _read_json(settings.report_file)
    if previous:
        _mirror(settings, previous)


def _load_state(settings):
    if not settings.snapshot_file.is_file():
        raise RuntimeError(f"Instagram snapshot not initialized for {settings.username}. Run instagram bootstrap.")
    rows = read_rows(settings.snapshot_file)
    validate_rows(rows, settings.username)
    if not settings.baseline_file.is_file():
        raise RuntimeError('Instagram baseline is missing; initialize baseline and snapshot together')
    baseline = read_rows(settings.baseline_file)
    validate_rows(baseline, settings.username)
    current = {r['id']: r['url'] for r in rows}
    if any(current.get(r['id']) != r['url'] for r in baseline):
        raise ValueError('Instagram snapshot no longer contains its baseline identities')
    cfg = _read_json(settings.cutoff_file)
    cutoff = utc_datetime(cfg.get("last_real_post_utc"))
    if not cutoff:
        raise RuntimeError("Instagram cutoff state is missing or invalid; refuse to rescan blindly")
    return rows, cutoff


def _pending_read(settings) -> list[dict]:
    results = []
    for line_no, data in enumerate(_read_jsonl(settings.pending_file), 1):
        try:
            results.append(new_row(settings.username, data["url"], data["content"], data["date"]))
        except Exception as exc:
            raise RuntimeError(f"Invalid pending Instagram row at line {line_no}; restore/repair pending file") from exc
    return results


def _pending_append(settings, row):
    _append_jsonl(settings.pending_file, {
        "url": row["url"], "content": row["content"], "date": row["date"].isoformat(),
    })


def _skipped_urls(settings) -> set[str]:
    urls = set()
    for data in _read_jsonl(settings.skipped_file):
        urls.add(str(data.get("url") or ""))
    urls.discard("")
    return urls


def _naive_utc_rows(rows):
    # The existing WordPress serializer requires timezone-naive UTC datetimes.
    result = []
    for row in rows:
        item = dict(row)
        for key in ("date", "modified"):
            dt = utc_datetime(item[key])
            item[key] = dt.replace(tzinfo=None) if dt else None
        result.append(item)
    return result


def _commit_run(settings, old_rows, new, cutoff, operation, scan_info=None):
    assigned = assign_sequential_ids(settings.username, old_rows, new)
    all_rows = [*old_rows, *assigned]
    validate_rows(all_rows, settings.username)
    # The baseline is the immutable seed record. NEW is cutoff-relative: only
    # posts discovered by this run (newer than the newest collected post).
    baseline = old_rows if operation == 'bootstrap' else read_rows(settings.baseline_file)
    exported = assigned
    max_new = max((utc_datetime(r["date"]) for r in assigned), default=None)
    latest = max(cutoff, max_new) if max_new else cutoff
    settings.archive_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_path(settings.archive_dir)
    report = {
        "operation": operation, "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"), "profile": settings.username,
        "created_at_utc": _time(), "run_dir": str(run_dir), "new": len(exported), "deleted": 0,
        "discovered": len(assigned), "baseline_records": len(baseline), "comparison": "cutoff_date",
        "snapshot_before": len(old_rows), "snapshot_after": len(all_rows),
        "cutoff_before": cutoff.isoformat(), "cutoff_after": latest.isoformat(),
        "outputs": {k: str(run_dir / f"chatbot_dataset_{k}.parquet") for k in ("new", "deleted", "snapshot")},
        "shared_dir": str(settings.shared_dir), "scan": scan_info or {},
    }
    _archive(settings, report, _naive_utc_rows(exported), _naive_utc_rows(all_rows))
    updates = {
        settings.snapshot_file: ("snapshot", _naive_utc_rows(all_rows)),
        settings.cutoff_file: {"last_real_post_utc": latest.isoformat()},
        settings.report_file: report,
    }
    if operation == 'bootstrap':
        updates[settings.baseline_file] = ('snapshot', _naive_utc_rows(baseline))
    commit(settings, updates)
    try:
        report["mirror"] = _mirror(settings, report)
    except Exception as exc:
        raise RuntimeError(f"Run committed at {run_dir}; shared publish failed: {exc}. "
                           "Retry instagram sync-shared before the next scan.") from exc
    return report


def bootstrap(settings, seed: Path, *, run=False):
    if settings.snapshot_file.exists() or settings.report_file.exists() or settings.baseline_file.exists():
        raise FileExistsError("Instagram state already initialized. Do not bootstrap twice.")
    rows, cutoff, info = load_seed(seed, settings.username)
    outcome = {"operation": "bootstrap", "profile": settings.username, "seed": str(seed.resolve()),
               "records": len(rows), "cutoff_utc": cutoff.isoformat(), "details": info,
               "state": str(settings.snapshot_file), "run": bool(run)}
    if not run:
        return outcome
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(settings.lock_file), timeout=0):
            if settings.snapshot_file.exists():
                raise FileExistsError("Already initialized")
            report = _commit_run(settings, rows, [], cutoff, "bootstrap", info)
            return {**outcome, "published": report}
    except Timeout as exc:
        raise RuntimeError("Another Instagram operation is running") from exc


def status(settings):
    exists = settings.snapshot_file.is_file()
    rows = read_rows(settings.snapshot_file) if exists else []
    return {
        "profile": settings.username, "initialized": exists, "records": len(rows),
        "cutoff": _read_json(settings.cutoff_file).get("last_real_post_utc"),
        "auth_state_present": settings.auth_file.is_file(),
        "pending_records": len(_pending_read(settings)) if settings.pending_file.exists() else 0,
        "skipped_empty_captions": len(_skipped_urls(settings)),
        "shared": str(settings.shared_dir), "last_run": _read_json(settings.report_file),
    }


def sync_shared(settings):
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(settings.lock_file), timeout=0):
            recover(settings)
            report = _read_json(settings.report_file)
            if not report:
                raise RuntimeError("Nothing committed to publish")
            return _mirror(settings, report)
    except Timeout as exc:
        raise RuntimeError("Another Instagram operation is running") from exc


def scan(settings, *, run=False, headed=False, max_scrolls=None, collector=None):
    if (settings.state_dir / ".pending").exists() and not run:
        raise RuntimeError("Pending Instagram transaction found. Retry 'instagram scan --run' or 'instagram sync-shared'.")
    if not run:
        rows, cutoff = _load_state(settings)
        return {"mode": "dry_run", "profile": settings.username, "records": len(rows),
                "cutoff": cutoff.isoformat(), "action": "Run with --run to scan Instagram",
                "auth_state_present": settings.auth_file.is_file()}
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(settings.lock_file), timeout=0):
            recover(settings)
            _finish_previous(settings)
            rows, cutoff = _load_state(settings)
            # Every known post with its date, so the browser never reopens it and
            # the date boundary can use it. Skipped entries may lack a date.
            known = {r["url"]: utc_datetime(r["date"]) for r in rows}
            collected = set(known)
            skipped = _skipped_urls(settings)
            for data in _read_jsonl(settings.skipped_file):
                url = str(data.get("url") or "")
                if url and url not in known:
                    known[url] = utc_datetime(data.get("date"))
            pending = _pending_read(settings)
            new_by_url = {r["url"]: r for r in pending if r["url"] not in collected and utc_datetime(r["date"]) > cutoff}
            known.update({url: utc_datetime(row["date"]) for url, row in new_by_url.items()})

            def on_new(url, caption, date):
                row = new_row(settings.username, url, caption, date)
                if row["url"] not in known:
                    _pending_append(settings, row)
                    new_by_url[row["url"]] = row
                    known[row["url"]] = utc_datetime(row["date"])

            def on_skip(url, reason, date=None):
                if url in skipped:
                    return
                _append_jsonl(settings.skipped_file, {
                    "url": url, "reason": reason, "date": date.isoformat() if date else None,
                    "profile": settings.username,
                })
                skipped.add(url)
                known[url] = utc_datetime(date)

            runner = collector or collect
            stats = runner(
                settings, known, cutoff, on_new, headed=headed, max_scrolls=max_scrolls, on_skip=on_skip,
            )
            if not stats.get("boundary_reached"):
                raise RuntimeError("Instagram scan incomplete: no confirmed old boundary")

            additions = list(new_by_url.values())
            result = _commit_run(settings, rows, additions, cutoff, "scan", stats)
            settings.pending_file.unlink(missing_ok=True)
            return result
    except Timeout as exc:
        raise RuntimeError("Another Instagram operation is running") from exc
