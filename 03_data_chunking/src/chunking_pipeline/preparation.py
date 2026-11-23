"""Retrieval-focused preparation adapted from the original project’s retrieval corpus script."""
import html
import re
import polars as pl

LOW_VALUE_TITLES = {"example subdomain", "карта сайта", "saytın xəritəsi", "saytin xəritəsi"}
PLACEHOLDERS = {("aaa", "aaa"), ("azzz", "azzz"), ("az", "az"), ("rus", "rus")}


def normalize_whitespace(text: str) -> str:
    if text is None:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Normalize horizontal whitespace
    text = re.sub(r"[^\S\n]+", " ", text)

    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_retrieval_text(text: str | None) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.I)
    text = re.sub(r"</\s*(p|div|li|tr|h[1-6])\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    return normalize_whitespace(text)


def norm_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def prepare(df: pl.DataFrame) -> pl.DataFrame:
    required = {"id", "source_type", "title", "content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")
    records = []
    for row in df.iter_rows(named=True):
        title = clean_retrieval_text(row.get("title"))
        content = clean_retrieval_text(row.get("content"))
        title_norm, content_norm = norm_compare(title), norm_compare(content)
        same = bool(title_norm) and title_norm == content_norm
        reason = None
        if not any(ch.isalnum() for ch in content):
            reason = "content_without_alphanumeric_information"
        if title_norm in LOW_VALUE_TITLES:
            reason = "low_value_site_page"
        if (title_norm, content_norm) in PLACEHOLDERS:
            reason = "placeholder_page"
        retrieval_text = title if same else (f"Title: {title}\n\nContent:\n{content}" if title and content else title or content)
        records.append({**row, "clean_title": title, "clean_content": content,
                        "title_equals_content": same, "retrieval_text": retrieval_text,
                        "retrieval_eligible": reason is None, "exclusion_reason": reason})
    # Preserve input types and declare derived nullable columns explicitly.
    # A reason may first appear well beyond Polars' inference sample.
    schema = {**df.schema, "clean_title": pl.String, "clean_content": pl.String,
              "title_equals_content": pl.Boolean, "retrieval_text": pl.String,
              "retrieval_eligible": pl.Boolean, "exclusion_reason": pl.String}
    return pl.DataFrame(records, schema_overrides=schema) if records else df.with_columns(
        pl.lit("").alias("clean_title"), pl.lit("").alias("clean_content"),
        pl.lit(False).alias("title_equals_content"), pl.lit("").alias("retrieval_text"),
        pl.lit(False).alias("retrieval_eligible"), pl.lit(None, dtype=pl.String).alias("exclusion_reason"))
