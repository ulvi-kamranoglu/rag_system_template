import pandas as pd
from canonical_pipeline.duplicates import duplicate_report, remove_duplicates


def test_conflicting_same_identity():
    df = pd.DataFrame({"id": ["page:1", "page:1"], "title": ["A", "A"], "url": ["a", "a"], "content": ["one", "two"]})
    report = duplicate_report(df)
    assert report["duplicate_id_rows"] == 2
    assert report["identity_conflict_groups"] == 1


def test_same_content_different_title_is_retained():
    df = pd.DataFrame({"id": ["page:1", "page:2", "post:3"],
                       "title": ["First title", "Different title", " FIRST   TITLE "],
                       "url": ["https://example.org/a"] * 3,
                       "content": ["Same content", "Same content", "same content"]})
    clean, removed = remove_duplicates(df)
    assert clean["id"].tolist() == ["page:1", "page:2"]
    assert len(removed) == 1
    assert removed[0]["id"] == "post:3"
    assert removed[0]["kept_id"] == "page:1"
    assert removed[0]["reason"] == "duplicate_title_content_url"


def test_url_difference_preserves_same_title_and_content():
    df = pd.DataFrame({"id": ["page:1", "page:2", "post:3", "post:4"],
        "title": ["Title"] * 4, "content": ["Same content"] * 4,
        "url": ["https://example.org/A", "https://example.org/a",
                "https://example.org/A?id=2", "https://example.org/A"]})
    clean, removed = remove_duplicates(df)
    assert clean["id"].tolist() == ["page:1", "page:2", "post:3"]
    assert len(removed) == 1
    assert removed[0]["id"] == "post:4"
    assert removed[0]["kept_id"] == "page:1"
    assert duplicate_report(df)["normalized_duplicate_title_content_url_rows"] == 2
