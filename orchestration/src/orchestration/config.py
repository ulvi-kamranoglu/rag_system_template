"""Resolve workspace paths and snapshot stage configuration."""
from pathlib import Path
import yaml
from pipeline_storage import workspace_root

ROOT = workspace_root(__file__)


def absolute(value, root=ROOT):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def snapshot_config(source, target, stage, *, mode="join"):
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    original_root = source.parent.parent
    # Freeze generic locations; stages own all processing and mode semantics.
    if "project_root" in raw:
        original_root = absolute(raw["project_root"], source.parent) if raw["project_root"] else ROOT
        raw["project_root"] = str(original_root)
    for key in ("input_dir", "input_path"):
        if raw.get(key):
            raw[key] = str(absolute(raw[key], original_root))
    if raw.get("inputs"):
        raw["inputs"] = [str(absolute(value, original_root)) for value in raw["inputs"]]
    for key, value in raw.get("storage", {}).items():
        if key.endswith("_dir"):
            raw["storage"][key] = str(absolute(value, original_root))
    raw.setdefault("source_project_root", str(original_root))
    target.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return raw
