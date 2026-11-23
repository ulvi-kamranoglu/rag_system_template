"""Unified, source-qualified WordPress record schema."""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .config import Settings
from .text import html_to_text

FIELDS = [
    "id", "source_type", "title", "content", "url", "date", "modified",
    "source", "language", "search_text", "metadata",
]
TYPES = {"pages": "page", "posts": "post"}


def _date(value: object, timezone: str) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(ZoneInfo(timezone))
    return timestamp.tz_convert("UTC").tz_localize(None).as_unit("ns")


def wp_date(item: Mapping, field: str, timezone: str) -> pd.Timestamp:
    utc_date = item.get(f"{field}_gmt")
    if utc_date:
        timestamp = pd.to_datetime(utc_date, utc=True, errors="coerce")
        if pd.notna(timestamp):
            return timestamp.tz_localize(None).as_unit("ns")
    return _date(item.get(field), timezone)


def _wp_id(raw: object, kind: str) -> str:
    raw_str = str(raw)
    if raw_str.startswith(f"{kind}:"):
        raw_str = raw_str[len(kind) + 1:]
    if not raw_str.isdecimal() or int(raw_str) <= 0:
        raise ValueError(f"Invalid WordPress ID for {kind}: {raw}")
    return str(int(raw_str))


def from_api(item: Mapping, source: str, settings: Settings) -> dict:
    if source not in TYPES:
        raise ValueError(f"Unknown source: {source}")
    kind = TYPES[source]
    wp_id = _wp_id(item.get("id"), kind)
    title_obj = item.get("title") or {}
    content_obj = item.get("content") or {}
    title = html_to_text(title_obj.get("rendered", ""), preserve_paragraphs=False)
    content = html_to_text(content_obj.get("rendered", ""))
    language = str(item.get("lang") or settings.default_language).lower()
    if language not in {"az", "en", "ru"}:
        language = settings.default_language
    return {
        "id": f"{kind}:{wp_id}",
        "source_type": kind,
        "title": title,
        "content": content,
        "url": str(item.get("link") or ""),
        "date": wp_date(item, "date", settings.timezone),
        "modified": wp_date(item, "modified", settings.timezone),
        "source": settings.hostname,
        "language": language,
        "search_text": content,
        "metadata": json.dumps({"wp_id": wp_id}, ensure_ascii=False, separators=(",", ":")),
    }


def from_baseline(row: Mapping, source: str, settings: Settings) -> dict:
    if source not in TYPES:
        raise ValueError(f"Unknown source: {source}")
    kind = TYPES[source]
    original_type = row.get("source_type", kind)
    if str(original_type) != kind:
        raise ValueError(f"Baseline row type mismatch: {original_type} != {kind}")
    wp_id = _wp_id(row["id"], kind)
    # The old v2 baseline already contains cleaned text. Reapplying this
    # projection is idempotent for normal text and strips leftover markup.
    title = html_to_text(_nullable(row.get("title")), preserve_paragraphs=False)
    content = html_to_text(_nullable(row.get("content")))
    language = str(_nullable(row.get("language")) or settings.default_language).lower()
    if language not in {"az", "en", "ru"}:
        language = settings.default_language
    return {
        "id": f"{kind}:{wp_id}",
        "source_type": kind,
        "title": title,
        "content": content,
        "url": str(_nullable(row.get("url")) or ""),
        "date": _baseline_date(row.get("date")),
        "modified": _baseline_date(row.get("modified")),
        "source": str(_nullable(row.get("source")) or settings.hostname),
        "language": language,
        "search_text": content,
        "metadata": json.dumps({"wp_id": wp_id}, ensure_ascii=False, separators=(",", ":")),
    }


def _nullable(value: object) -> object:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (ValueError, TypeError):
        pass
    return value


def _baseline_date(value: object) -> pd.Timestamp:
    value = _nullable(value)
    if value is None:
        return pd.NaT
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp.tz_localize(None).as_unit("ns")


def dataset(rows: list[dict], event: str = "snapshot") -> pd.DataFrame:
    if event not in {"snapshot", "new", "deleted"}:
        raise ValueError(f"Unsupported dataset type: {event}")
    columns = FIELDS
    frame = pd.DataFrame(rows, columns=columns)
    for field in ("date", "modified"):
        frame[field] = pd.to_datetime(frame[field], errors="coerce").astype("datetime64[ns]")
    return frame
