"""Immutable, numerically ordered dataset snapshots (filesystem operations only)."""
from pathlib import Path
import re
import shutil
from uuid import uuid4

from .fs import rename_with_retry


def version(name):
    match = re.fullmatch(r"(.+)_v(\d+)\.(\d+)", name)
    if not match:
        raise ValueError(f"Expected <dataset>_v<major>.<minor>, got {name!r}")
    return match[1], int(match[2]), int(match[3])


def latest(folder, suffix):
    candidates = []
    for path in Path(folder).glob(f"*{suffix}"):
        stem = path.name[:-len(suffix)]
        family, major, minor = version(stem)
        candidates.append((family, major, minor, path))
    if not candidates:
        raise ValueError(f"Place an initial versioned baseline in {folder} (expected *{suffix})")
    if len({item[0] for item in candidates}) != 1:
        raise ValueError(f"Multiple dataset families in {folder}")
    return max(candidates, key=lambda item: item[1:3])[3]


def next_name(path, suffix):
    family, major, minor = version(path.name[:-len(suffix)])
    return f"{family}_v{major}.{minor + 1}"


def commit_base(run, report):
    """Idempotent retry; never overwrite an existing version with different bytes."""
    folder = report.get("base_dir")
    if not folder:
        return
    folder = Path(folder)
    if folder.resolve() != (Path(run).parent / "base_data").resolve():
        raise ValueError("Base directory must be archive/base_data")
    folder.mkdir(parents=True, exist_ok=True)
    for name in report["base_outputs"]:
        if Path(name).name != name:
            raise ValueError("Invalid baseline filename")
        source_dir = report.get("base_source_dir", "")
        if source_dir not in {"", "base_snapshot"}:
            raise ValueError("Invalid baseline source directory")
        source, target = Path(run) / source_dir / name, folder / name
        parsed = re.fullmatch(r"(.+_v\d+\.\d+)(_.+)", Path(name).stem)
        if not parsed:
            raise ValueError(f"Invalid versioned output: {name}")
        suffix = parsed[2] + Path(name).suffix
        existing = list(folder.glob(f"*{suffix}"))
        if existing:
            newest = latest(folder, suffix)
            wanted_version = version(parsed[1])
            newest_version = version(newest.name[:-len(suffix)])
            if wanted_version[0] != newest_version[0] or wanted_version[1:] < newest_version[1:]:
                raise ValueError("Refusing to publish an older or unrelated baseline version")
        if target.exists():
            if source.read_bytes() != target.read_bytes():
                raise ValueError(f"Version already exists with different content: {target}")
            continue
        temporary = folder / f".{name}.{uuid4().hex}.partial"
        try:
            shutil.copyfile(source, temporary)
            rename_with_retry(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
