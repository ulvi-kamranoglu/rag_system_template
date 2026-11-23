"""Instagram canonical schema, legacy import, and profile-specific IDs."""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd

FIELDS = ["id", "source_type", "title", "content", "url", "date", "modified", "source", "language", "search_text", "metadata"]
POST_RE = re.compile(
    r"^/(?:[A-Za-z0-9._]{1,30}/)?(p|reel|reels)/([A-Za-z0-9_-]+)/?$"
)
PROFILE_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")
# Before the canonical ID standard, the UNEC account used unqualified
# instagram:N IDs. Seeds may still contain them; they are converted on import.
LEGACY_UNQUALIFIED_PROFILES = {"uneceduaz"}
UNQUALIFIED_ID_RE = re.compile(r"^instagram:(\d+)$")
RESEARCH_ID_RE = re.compile(r"^research:(\d+)$")
TRUE_TOKENS = {"true", "1", "yes"}
FALSE_TOKENS = {"false", "0", "no", ""}


def profile_name(profile: str) -> str:
    name = profile.strip().lower()
    if not PROFILE_RE.fullmatch(name):
        raise ValueError(f"Invalid Instagram username: {profile!r}")
    return name


def canonical_url(raw: str) -> str:
    parsed = urlparse(str(raw))
    if parsed.scheme not in ("http", "https") or (parsed.hostname or "").lower() not in {"instagram.com", "www.instagram.com"}:
        raise ValueError(f"Invalid Instagram post URL: {raw!r}")
    match = POST_RE.fullmatch(parsed.path)
    if not match:
        raise ValueError(f"Invalid Instagram post path: {raw!r}")
    kind, code = match.groups()
    return f"https://www.instagram.com/{'reel' if kind == 'reels' else kind}/{code}/"


def shortcode(url: str) -> str:
    return canonical_url(url).rstrip('/').split('/')[-1]


def utc_datetime(value) -> datetime | None:
    if value is None or value is pd.NaT or (isinstance(value, str) and not value.strip()):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime().astimezone(timezone.utc)


def is_synthetic(value) -> bool:
    """Parse date_synthetic without treating the string 'false' as true."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) == 0:
            return False
        if int(value) == 1:
            return True
        raise ValueError(f"Unusable date_synthetic value: {value!r}")
    token = str(value).strip().casefold()
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    raise ValueError(f"Unusable date_synthetic value: {value!r}")


def id_regex(profile: str) -> re.Pattern:
    """Canonical standard for every profile: instagram:<profile>:<N>."""
    return re.compile(rf"^instagram:{re.escape(profile_name(profile))}:(\d+)$")


def format_numeric_id(profile: str, number: int) -> str:
    if int(number) < 1:
        raise ValueError(f"Instagram numeric IDs must be positive, got {number}")
    return f"instagram:{profile_name(profile)}:{int(number)}"


def parse_numeric_id(profile: str, value: str) -> int:
    match = id_regex(profile).fullmatch(str(value).strip())
    if not match or int(match.group(1)) < 1:
        raise ValueError(
            f"ID {value!r} is not valid for profile {profile_name(profile)}. "
            f"Expected {format_numeric_id(profile, 1)}-style sequential IDs."
        )
    return int(match.group(1))


def max_numeric_id(rows: list[dict], profile: str) -> int:
    if not rows:
        return 0
    return max(parse_numeric_id(profile, row["id"]) for row in rows)


def normalize_seed_id(profile: str, original_id: str) -> str:
    """Convert accepted historical formats to instagram:<profile>:<N>; never invent numbers.

    Accepted: instagram:<profile>:N, research:<profile>:N, research:N and, for
    the UNEC account only, the legacy unqualified instagram:N.
    """
    original = str(original_id or "").strip()
    name = profile_name(profile)
    patterns = [id_regex(name), re.compile(rf"^research:{re.escape(name)}:(\d+)$"), RESEARCH_ID_RE]
    if name in LEGACY_UNQUALIFIED_PROFILES:
        patterns.append(UNQUALIFIED_ID_RE)
    for pattern in patterns:
        match = pattern.fullmatch(original)
        if match:
            return format_numeric_id(name, int(match.group(1)))
    raise ValueError(
        f"Unexpected Instagram ID {original!r} for profile {name}. "
        f"Expected {format_numeric_id(name, 1)}, research:N"
        + (" or legacy instagram:N" if name in LEGACY_UNQUALIFIED_PROFILES else "")
        + ". Shortcode IDs are not auto-mapped."
    )


def assign_sequential_ids(profile: str, existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    """Assign the next profile-specific numeric IDs from max(existing), not row count."""
    next_id = max_numeric_id(existing_rows, profile)
    assigned = []
    ordered = sorted(new_rows, key=lambda row: (utc_datetime(row["date"]), row["url"]))
    for row in ordered:
        next_id += 1
        item = dict(row)
        item["id"] = format_numeric_id(profile, next_id)
        item["source_type"] = "instagram"
        assigned.append(item)
    return assigned


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=FIELDS)
    for col in ("date", "modified"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None).astype("datetime64[ns]")
    return df


def load_seed(path, profile: str) -> tuple[list[dict], datetime, dict]:
    """Read the 11-column dataset (also accepts historical 12th date_synthetic)."""
    source = pd.read_parquet(path)
    missing = set(FIELDS) - set(source.columns)
    if missing:
        raise ValueError(f"Seed is missing columns: {sorted(missing)}")
    if source.empty:
        raise ValueError("Seed must contain existing Instagram records")
    results, ids, urls = [], set(), set()
    cutoff = None
    synthetic_count = 0
    normalized_ids = 0
    name = profile_name(profile)
    for idx, row in enumerate(source.to_dict("records"), 1):
        url = canonical_url(row["url"])
        if url in urls:
            raise ValueError(f"Duplicate URL in seed: {url}")
        urls.add(url)
        original_id = str(row.get("id") or "").strip()
        normalized = normalize_seed_id(name, original_id)
        if normalized != original_id:
            normalized_ids += 1
        if normalized in ids:
            raise ValueError(f"Duplicate ID in seed: {normalized}")
        ids.add(normalized)
        content = str(row.get("content") or "").strip()
        if not content:
            raise ValueError(f"Empty content in seed at row {idx}")
        date = utc_datetime(row.get("date"))
        if date is None:
            raise ValueError(f"Missing/unparseable date in seed at row {idx}; fix before bootstrap")
        if "date_synthetic" in source.columns:
            synthetic = is_synthetic(row.get("date_synthetic"))
        else:
            synthetic = False
        if synthetic:
            synthetic_count += 1
        else:
            cutoff = max(cutoff, date) if cutoff else date
        results.append({
            "id": normalized, "source_type": "instagram", "title": "", "content": content,
            "url": url, "date": date, "modified": date, "source": f"instagram:{name}",
            "language": "az", "search_text": content, "metadata": "{}",
        })
    if cutoff is None:
        raise ValueError("No verifiable dates in seed: all dates are synthetic")
    return results, cutoff, {
        "seed_records": len(results),
        "synthetic_dates": synthetic_count,
        "normalized_research_ids": normalized_ids,
        "max_numeric_id": max_numeric_id(results, name),
    }


def new_row(profile: str, url: str, caption: str, published: datetime) -> dict:
    """Build an unpublished canonical row. Sequential IDs are assigned at commit."""
    url = canonical_url(url)
    content = (caption or "").strip()
    stamp = utc_datetime(published)
    if not stamp or not content:
        raise ValueError("New Instagram record requires a real date and nonempty caption")
    return {
        "id": "", "source_type": "instagram",
        "title": "", "content": content, "url": url, "date": stamp, "modified": stamp,
        "source": f"instagram:{profile_name(profile)}", "language": "az", "search_text": content, "metadata": "{}",
    }


def validate_rows(rows: list[dict], profile: str) -> None:
    ids, urls = set(), set()
    name = profile_name(profile)
    for index, row in enumerate(rows, 1):
        if list(row) != FIELDS:
            raise ValueError(f"Canonical column/order mismatch at row {index}")
        parse_numeric_id(name, row["id"])
        if row["id"] in ids or row["url"] in urls:
            raise ValueError(f"Duplicate Instagram ID or URL at row {index}")
        ids.add(row["id"]); urls.add(row["url"])
        if row["source_type"] != "instagram" or row["source"] != f"instagram:{name}":
            raise ValueError("Incorrect Instagram source")
        if canonical_url(row["url"]) != row["url"]:
            raise ValueError("Noncanonical URL")
        if not row["content"] or row["search_text"] != row["content"]:
            raise ValueError("Missing/incorrect caption")
        if utc_datetime(row["date"]) != utc_datetime(row["modified"]):
            raise ValueError("Modified is not date")
        if not utc_datetime(row["date"]):
            raise ValueError("Missing post date")
        if row["title"] != "" or row["language"] != "az" or row["metadata"] != "{}":
            raise ValueError("Incorrect canonical constants")
