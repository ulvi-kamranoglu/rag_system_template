"""Source-aware identifier, URL and metadata normalization."""
import json
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import pandas as pd

SOURCE_ALIASES = {"pages": "page", "posts": "post", "faq": "form"}
TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def normalize_url(value: str) -> str:
    if not value.strip():
        return ""
    try:
        parts = urlsplit(value.strip())
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return value.strip()
        hostname = parts.hostname.lower()
        port = parts.port
        netloc = hostname if port is None or (parts.scheme.lower() == "https" and port == 443) or (parts.scheme.lower() == "http" and port == 80) else f"{hostname}:{port}"
        if parts.username or parts.password:
            return value.strip()
        params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.lower().startswith("utm_") and k.lower() not in TRACKING]
        return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", urlencode(params), ""))
    except ValueError:
        return value.strip()


def parse_metadata(raw: str) -> dict:
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except (TypeError, ValueError):
        pass
    raise ValueError("Metadata must be a JSON object")


WORDPRESS_TYPES = {"page", "post", "content_post"}
FRAMEWORK_URL = re.compile(r"/framework/([0-9]+)(?:[/?#]|$)")


def _metadata(row) -> dict:
    """Best effort for identity only; canonicalize_row still rejects invalid JSON."""
    try:
        return parse_metadata(str(row.get("metadata") or ""))
    except ValueError:
        return {}


def _positive(value):
    text = str(value).strip()
    return str(int(text)) if re.fullmatch(r"[0-9]+", text) and int(text) > 0 else None


def _id_parts(row):
    return [part.strip() for part in str(row["id"]).strip().split(":")]


def _legal_clause(value):
    """Normalize a hierarchical clause such as 1.2.3 or 4.2-1; None if invalid."""
    clause = str(value).strip()
    if not re.fullmatch(r"[0-9]+(?:[.-][0-9]+)*", clause):
        return None
    return re.sub(r"[0-9]+", lambda match: str(int(match.group())), clause)


def _wordpress_identity(row, kind):
    """WordPress records are rebuilt from metadata.wp_id; the ID suffix is a fallback."""
    from_id = _positive(_id_parts(row)[-1])
    declared = _metadata(row).get("wp_id")
    from_field = None
    if declared not in (None, ""):
        from_field = _positive(declared)
        if from_field is None:
            raise ValueError("wp_id must be a positive integer")
    if from_field and from_id and from_field != from_id:
        raise ValueError("wp_id conflicts with record ID")
    number = from_field or from_id
    if number is None:
        raise ValueError("Canonical ID must end with a positive integer")
    return kind, number, f"{kind}:{number}"


def _legal_identity(row):
    """Legal records are rebuilt as legal:<framework>:<clause>.

    The framework comes from the e-qanun URL and the clause from
    metadata.clause_id; ID parts are the fallback and must not contradict them.
    Using only the last number would merge clause 7 of different frameworks.
    """
    match = FRAMEWORK_URL.search(str(row.get("url") or ""))
    field_framework = _positive(match.group(1)) if match else None
    declared = _metadata(row).get("clause_id")
    field_clause = None
    if declared not in (None, ""):
        field_clause = _legal_clause(declared)
        if field_clause is None:
            raise ValueError('Legal clause_id must be numeric segments separated by "." or "-"')
    parts = _id_parts(row)
    id_framework = _positive(parts[-2]) if len(parts) >= 2 else None
    id_clause = _legal_clause(parts[-1]) if id_framework else None
    if field_framework and id_framework and field_framework != id_framework:
        raise ValueError("Legal framework in URL conflicts with record ID")
    if field_clause and id_clause and field_clause != id_clause:
        raise ValueError("clause_id conflicts with record ID")
    framework, clause = field_framework or id_framework, field_clause or id_clause
    if not framework:
        raise ValueError("Legal ID needs a framework number (URL /framework/<n> or ID <framework>:<clause>)")
    if not clause:
        raise ValueError("Legal ID needs a clause (metadata.clause_id or ID <framework>:<clause>)")
    number = f"{framework}:{clause}"
    return "legal", number, f"legal:{number}"


def canonical_identity(row):
    """Rebuild the canonical ID on every run, per source type, never inventing numbers.

    legal: legal:<framework>:<clause>; page/post/content_post: <type>:<wp_id>;
    instagram: <source>:<number>; others: <source_type>:<number>. Collectors
    assign Instagram/News numbers; canonical only normalizes their format.
    """
    kind = SOURCE_ALIASES.get(row["source_type"].strip().lower(), row["source_type"].strip().lower())
    if kind == "legal":
        return _legal_identity(row)
    if kind in WORDPRESS_TYPES:
        return _wordpress_identity(row, kind)
    number = _positive(_id_parts(row)[-1])
    if not kind or number is None:
        raise ValueError("Canonical ID must end with a positive integer")
    prefix = str(row.get("source") or "").strip() if kind == "instagram" else kind
    if not prefix or any(c.isspace() for c in prefix) or prefix.endswith(":"):
        raise ValueError("Instagram requires a nonempty source prefix without whitespace or trailing colon")
    return kind, number, f"{prefix}:{number}"


def canonicalize_row(row: pd.Series) -> pd.Series:
    kind, source_id, identifier = canonical_identity(row)
    metadata = parse_metadata(row["metadata"])
    if kind in WORDPRESS_TYPES:
        metadata["wp_id"] = source_id
    if kind == "legal":
        metadata["clause_id"] = source_id.split(":", 1)[1]
    row["id"] = identifier
    row["source_type"] = kind
    row["url"] = normalize_url(row["url"])
    language = row["language"].strip().lower()
    row["language"] = language or "az"
    if row["language"] not in {"az", "en", "ru"}:
        raise ValueError("Unsupported language")
    row["metadata"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return row
