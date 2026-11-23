"""Pure WordPress ID comparison against a fixed baseline."""
from __future__ import annotations

from dataclasses import dataclass

from .records import FIELDS


@dataclass(frozen=True)
class Changes:
    new: list[dict]
    deleted: list[dict]


def _by_id(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        key = str(row["id"])
        if key in result:
            raise ValueError(f"Duplicate WordPress identity: {key}")
        result[key] = row
    return result


def compare(baseline: list[dict], current: list[dict]) -> Changes:
    """Compare source-qualified WordPress IDs; ignore all changes to existing records."""
    old = _by_id(baseline)
    fresh = _by_id(current)
    key_order = lambda row: (row["source_type"], int(row["id"].split(":", 1)[1]))
    return Changes(
        new=sorted(
            [{column: row[column] for column in FIELDS}
             for key, row in fresh.items() if key not in old],
            key=key_order,
        ),
        deleted=sorted(
            [{column: row[column] for column in FIELDS}
             for key, row in old.items() if key not in fresh],
            key=key_order,
        ),
    )
