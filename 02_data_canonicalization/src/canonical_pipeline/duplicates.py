"""Report duplicate groups and remove title+content+URL duplicates last."""
import hashlib
import re
import unicodedata
import pandas as pd


def duplicate_report(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"duplicate_id_rows": 0, "duplicate_url_rows": 0, "duplicate_content_rows": 0, "duplicate_title_content_rows": 0, "identity_conflict_groups": 0, "normalized_duplicate_content_rows": 0, "normalized_duplicate_title_content_rows": 0, "normalized_duplicate_title_content_url_rows": 0}
    content_hash = df["content"].map(lambda v: hashlib.sha256(v.encode("utf-8")).hexdigest())
    url_mask = df["url"].ne("") & df["url"].duplicated(keep=False)
    content_mask = df["content"].ne("") & content_hash.duplicated(keep=False)
    groups = df.assign(_hash=content_hash).groupby("id", dropna=False)["_hash"].nunique()
    comparison = df[["title", "content"]].copy()
    for column in comparison:
        comparison[column] = comparison[column].map(comparison_key)
    triple = comparison.assign(url=df["url"].str.strip())
    return {
        "duplicate_id_rows": int(df["id"].duplicated(keep=False).sum()),
        "duplicate_url_rows": int(url_mask.sum()),
        "duplicate_content_rows": int(content_mask.sum()),
        "duplicate_title_content_rows": int(df.duplicated(subset=["title", "content"], keep=False).sum()),
        "identity_conflict_groups": int((groups > 1).sum()),
        "normalized_duplicate_content_rows": int(comparison["content"].duplicated(keep=False).sum()),
        "normalized_duplicate_title_content_rows": int(comparison.duplicated(keep=False).sum()),
        "normalized_duplicate_title_content_url_rows": int(triple.duplicated(keep=False).sum()),
    }


def comparison_key(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Keep the first row per normalized title+content+URL combination."""
    seen, removed, keep = {}, [], []
    for index, row in df.iterrows():
        content = comparison_key(row["content"])
        # URL is already canonicalized; retain case-sensitive path/query values.
        key = (comparison_key(row["title"]), content, row["url"].strip())
        previous = seen.get(key)
        if previous is not None:
            removed.append({"row_index": int(index), "id": row["id"],
                            "kept_id": previous["id"], "kept_row_index": previous["index"],
                            "reason": "duplicate_title_content_url"})
            continue
        saved = {"id": row["id"], "index": int(index)}
        seen[key] = saved
        keep.append(index)
    return df.loc[keep].reset_index(drop=True), removed
