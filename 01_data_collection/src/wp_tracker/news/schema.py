"""News records in the shared 11-column WordPress format.

The identity is the site's own article number from the URL
(``/xeber/<category>/<number>-<slug>``): ``id = news:<number>`` and
``metadata.news_id = "<number>"``, just as WordPress uses ``page:<wp_id>``.
"""
import json
import re

import pandas as pd

from ..records import FIELDS
from .parsing import SOURCE, canonical_url

NUMBER_RE = re.compile(r"/([0-9]+)-[^/]*$")
ID_RE = re.compile(r"^news:([0-9]+)$")


def news_number(url: str) -> str:
    """Article number of a canonical news URL; raises if the URL has none."""
    canonical = canonical_url(url)
    match = NUMBER_RE.search(canonical or "")
    if not canonical or not match or int(match.group(1)) < 1:
        raise ValueError(f"Not a numbered news article URL: {url!r}")
    return str(int(match.group(1)))


def news_id(number: str) -> str:
    return f"news:{int(number)}"


def _timestamp(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return pd.NaT
    try:
        if pd.isna(value):
            return pd.NaT
    except (TypeError, ValueError):
        pass
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return pd.NaT if pd.isna(stamp) else stamp.tz_localize(None).as_unit("ns")


def _metadata(raw, number):
    text = "" if raw is None else str(raw)
    metadata = json.loads(text) if text.strip() else {}
    if not isinstance(metadata, dict):
        raise ValueError("News metadata must be a JSON object")
    declared = metadata.get("news_id")
    if declared not in (None, "") and str(declared).strip() != number:
        raise ValueError(f"metadata.news_id {declared!r} conflicts with URL number {number}")
    metadata["news_id"] = number
    return json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))


def article_row(url: str, title: str, content: str, published) -> dict:
    """A newly scraped article; modified equals the publication date."""
    url = canonical_url(url)
    number = news_number(url)
    stamp = _timestamp(published)
    return {"id": news_id(number), "source_type": "news", "title": title, "content": content,
            "url": url, "date": stamp, "modified": stamp, "source": SOURCE, "language": "az",
            "search_text": content, "metadata": _metadata("{}", number)}


def seed_row(row: dict) -> dict:
    """Rebuild a historical row's identity from its URL; never trust a sequential ID.

    A legacy numeric ID (1, 2, ...) is replaced. A ``news:<n>`` ID must agree
    with the URL number, otherwise the seed is rejected rather than guessed.
    """
    missing = set(FIELDS) - set(row)
    if missing:
        raise ValueError(f"News seed row is missing columns: {sorted(missing)}")
    url = canonical_url(str(row["url"] or ""))
    number = news_number(url or str(row["url"]))
    original = str(row["id"]).strip()
    match = ID_RE.fullmatch(original)
    if original.startswith("news:") and (not match or str(int(match.group(1))) != number):
        raise ValueError(f"Seed ID {original!r} conflicts with URL number {number}")
    if str(row["source_type"]).strip() != "news":
        raise ValueError(f"Seed row {original!r} is not source_type=news")
    return {"id": news_id(number), "source_type": "news", "title": str(row["title"] or ""),
            "content": str(row["content"] or ""), "url": url, "date": _timestamp(row["date"]),
            "modified": _timestamp(row["modified"]), "source": SOURCE,
            "language": str(row["language"] or "az"), "search_text": str(row["search_text"] or row["content"] or ""),
            "metadata": _metadata(row["metadata"], number)}


def validate_rows(rows: list[dict]) -> None:
    """Identity invariants for state and archives."""
    ids, urls = set(), set()
    for index, row in enumerate(rows, 1):
        if set(row) != set(FIELDS):
            raise ValueError(f"News row {index} does not use the 11-column record format")
        number = news_number(row["url"])
        if row["id"] != news_id(number) or row["source_type"] != "news":
            raise ValueError(f"News row {index}: ID {row['id']!r} does not match URL number {number}")
        if json.loads(row["metadata"]).get("news_id") != number:
            raise ValueError(f"News row {index}: metadata.news_id does not match URL")
        if row["id"] in ids or row["url"] in urls:
            raise ValueError(f"Duplicate news ID or URL: {row['id']}")
        ids.add(row["id"]); urls.add(row["url"])
