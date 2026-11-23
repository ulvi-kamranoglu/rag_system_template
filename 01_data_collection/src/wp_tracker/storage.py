"""Parquet persistence with recoverable multi-file transactions."""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

from .config import Settings
from .records import FIELDS, dataset

LOG = logging.getLogger(__name__)


def read_rows(path: Path, event: str = "snapshot") -> list[dict]:
    if not path.is_file():
        return []
    frame = pd.read_parquet(path)
    if event not in {"snapshot", "new", "deleted"}:
        raise ValueError(f"Unsupported dataset type: {event}")
    required = set(FIELDS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return frame.to_dict("records")


def write_parquet(path: Path, rows: list[dict], event: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset(rows, event).to_parquet(path, index=False, engine="pyarrow")


def recover(settings: Settings) -> None:
    """Roll back an interrupted commit before reading or modifying state."""
    txn = settings.state_dir / ".pending"
    if not txn.is_dir():
        return
    manifest_path = txn / "manifest.json"
    if not manifest_path.is_file():
        shutil.rmtree(txn)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") == "committed":
        shutil.rmtree(txn)
        return
    LOG.warning("Recovering interrupted transaction; restoring previous outputs")
    for entry in manifest["files"]:
        target = Path(entry["target"])
        backup = txn / "backup" / entry["name"]
        if entry["existed"]:
            if not backup.is_file():
                raise RuntimeError(f"Transaction backup missing: {backup}")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Retain backup for repeated recovery if restoration fails midway.
            restore_tmp = target.with_name(target.name + ".recover")
            shutil.copy2(backup, restore_tmp)
            os.replace(restore_tmp, target)
        elif target.exists():
            target.unlink()
    shutil.rmtree(txn)
    # Remove a newly created, now-empty output run folder after rollback.
    output_root = settings.output_dir.resolve()
    for entry in manifest["files"]:
        target = Path(entry["target"])
        if not entry["existed"] and target.parent.parent == output_root:
            try:
                target.parent.rmdir()
            except OSError:
                pass


def _write_manifest(txn: Path, manifest: dict) -> None:
    temp = txn / "manifest.tmp"
    temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temp, txn / "manifest.json")


def commit(settings: Settings, files: dict[Path, tuple[str, list[dict]] | dict]) -> None:
    """Stage every file, backup originals, replace all, or restore on restart."""
    txn = settings.state_dir / ".pending"
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    recover(settings)
    (txn / "stage").mkdir(parents=True)
    (txn / "backup").mkdir()
    entries = []
    try:
        for index, (target, data) in enumerate(files.items()):
            name = f"file_{index}{target.suffix}"
            stage_path = txn / "stage" / name
            if isinstance(data, tuple):
                event, rows = data
                write_parquet(stage_path, rows, event)
            else:
                stage_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            existed = target.is_file()
            if existed:
                shutil.copy2(target, txn / "backup" / name)
            entries.append({"name": name, "target": str(target.resolve()), "existed": existed})
        manifest = {"phase": "prepared", "files": entries}
        _write_manifest(txn, manifest)
        for entry in entries:
            dest = Path(entry["target"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(txn / "stage" / entry["name"], dest)
        manifest["phase"] = "committed"
        _write_manifest(txn, manifest)
        # Per-run folders are immutable history; no second archive copy is needed.
        shutil.rmtree(txn)
    except Exception:
        # Recover only if a manifest exists. Before then no target was replaced.
        recover(settings)
        raise


def summarize_files(settings: Settings) -> dict:
    report_path = settings.reports_dir / "latest.json"
    latest_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    return {
        "latest_run_dir": latest_report.get("run_dir"),
        "baseline": {
            "exists": settings.baseline_file.is_file(),
            "records": len(read_rows(settings.baseline_file)),
        },
        "snapshot": {
            "exists": settings.snapshot_file.is_file(),
            "records": len(read_rows(settings.snapshot_file)),
        },
        "outputs": {
            name: {
                "path": str(settings.output_file(name)),
                "exists": settings.output_file(name).is_file(),
                "records": len(read_rows(settings.output_file(name), name)),
            }
            for name in ("new", "deleted")
        },
        "report": str(settings.reports_dir / "latest.json"),
    }
