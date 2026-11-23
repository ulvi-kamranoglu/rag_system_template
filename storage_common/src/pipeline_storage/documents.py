"""Use the same document identity in canonical, chunk and deletion records."""
import re
import math


def _text(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def document_id(row):
    explicit = row.get("document_id")
    if _text(explicit):
        return _text(explicit)
    identity = row.get("id")
    kind = row.get("source_type")
    if not _text(identity) or not _text(kind):
        raise ValueError("Deletion records require document_id or id + source_type")
    identity, kind = _text(identity), _text(kind).lower()
    kind = {"pages": "page", "posts": "post", "faq": "form"}.get(kind, kind)
    if identity.startswith(kind + ":"):
        value = identity[len(kind) + 1:].strip()
        if kind in {"page", "post"} and re.fullmatch(r"\d+", value):
            identity = f"{kind}:{int(value)}"
        return identity
    return f"{kind}:{identity}"


def deleted_ids(path):
    if path is None:
        return set()
    from pathlib import Path
    import pandas as pd
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if "document_id" not in frame.columns and not {"id", "source_type"}.issubset(frame.columns):
        raise ValueError("Deleted dataset requires document_id or id + source_type")
    return {document_id(row) for row in frame.to_dict("records")}
