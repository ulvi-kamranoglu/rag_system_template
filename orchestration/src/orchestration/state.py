"""Atomic checkpoints and immutable stage archive lookup."""
import json
import os
import time
from uuid import uuid4


def save(path, data):
    # Windows readers (including antivirus/indexers) may briefly deny replacement.
    # Keep the previous checkpoint intact until a complete replacement succeeds.
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        for attempt in range(9):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 8:
                    raise
                time.sleep(min(0.1 * (2 ** attempt), 1.0))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


def archive_for(archive, run_id, operation=None):
    matches = []
    for path in archive.glob("run_*/run_report.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("pipeline_run_id") == run_id and (operation is None or report.get("operation") == operation):
            matches.append((path.parent, report))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple archives for the same stage/run in {archive}; inspect before resuming")
    return matches[0] if matches else None
