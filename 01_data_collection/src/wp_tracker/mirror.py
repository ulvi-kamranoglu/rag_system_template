"""Retry the latest committed archive without contacting WordPress."""
import json
from pathlib import Path
from filelock import FileLock
from pipeline_storage import publish, validate_json
from .config import Settings
from .storage import read_rows


def validate_run(folder: Path) -> None:
    for event in ("new", "deleted"):
        path = folder / f"chatbot_dataset_{event}.parquet"
        if not path.is_file():
            raise ValueError(f"Missing required output: {path}")
        read_rows(path, event)
    validate_json(folder, ["latest.json", "run_report.json"])


def mirror_run(settings: Settings, run_dir: Path, *, migrate_legacy: bool = False) -> dict:
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != settings.output_dir.resolve() or not run_dir.name.startswith("run_"):
        raise ValueError(f"Not a collector archive: {run_dir}")
    if (settings.state_dir / ".pending").exists():
        raise RuntimeError("Pending local transaction; recover before publishing")
    return publish(run_dir, settings.shared_raw_dir, validate_run, migrate_legacy=migrate_legacy)


def sync_shared(settings: Settings, *, migrate_legacy: bool = False) -> dict:
    result = {"synced": [], "already_synced": [], "failed": []}
    if not settings.output_dir.exists():
        return result
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(settings.lock_file), timeout=0):
        if (settings.state_dir / ".pending").exists():
            raise RuntimeError("Pending local transaction; run 'python -m wp_tracker recover' first")
        report_path = settings.reports_dir / "latest.json"
        if not report_path.is_file():
            return result
        folder = Path(json.loads(report_path.read_text(encoding="utf-8"))["run_dir"])
        try:
            published = mirror_run(settings, folder, migrate_legacy=migrate_legacy)
            result[published["status"]].append(published["path"])
        except Exception as exc:
            result["failed"].append({"run_dir": str(folder), "error": str(exc)})
    return result
