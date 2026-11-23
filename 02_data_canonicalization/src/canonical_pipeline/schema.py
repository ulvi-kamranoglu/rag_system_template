"""Dataset contract and minimal type normalization."""
import pandas as pd
import math
from numbers import Real
import re

COLUMNS = ["id", "source_type", "title", "content", "url", "date", "modified", "source", "language", "search_text", "metadata"]
REQUIRED = {"id", "source_type", "content"}
TEXT_COLUMNS = ["id", "source_type", "title", "content", "url", "source", "language", "search_text", "metadata"]


def parse_datetime(value):
    """Accept ISO/datetime values and Unix epochs in s/ms/us/ns, per value.

    Already typed dates are preserved: an actual 1970 date must never be
    silently rescaled. Historical corruption needs provenance-based repair.
    """
    if isinstance(value, bool):
        return pd.NaT
    numeric = isinstance(value, Real) or (isinstance(value, str) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value.strip()))
    if numeric:
        number = float(value)
        if not math.isfinite(number):
            return pd.NaT
        magnitude = abs(number)
        unit = "s" if magnitude < 1e11 else "ms" if magnitude < 1e14 else "us" if magnitude < 1e17 else "ns"
        # Keep integer nanosecond precision; do not convert integers via float.
        epoch = int(value) if not isinstance(value, float) and "." not in str(value) else number
        return pd.to_datetime(epoch, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(value, errors="coerce", utc=True)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = df.copy()
    for name in COLUMNS:
        if name not in result:
            result[name] = None
    result = result[COLUMNS]
    for name in TEXT_COLUMNS:
        result[name] = result[name].map(lambda x: "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
    for name in ("date", "modified"):
        if pd.api.types.is_datetime64_any_dtype(result[name]):
            result[name] = pd.to_datetime(result[name], errors="coerce", utc=True)
        else:
            result[name] = pd.to_datetime(result[name].map(parse_datetime), errors="coerce", utc=True)
    return result
