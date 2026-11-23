import json

import pandas as pd
import pytest

from wp_tracker.records import dataset, from_api, from_baseline
from wp_tracker.text import comparable, html_to_text


def _api_record(id=123, title="Hello", content="<p>Hi &amp; bye</p>", **kwargs):
    record = {
        "id": id, "title": {"rendered": title}, "content": {"rendered": content},
        "link": "https://unec.edu.az/test/", "date_gmt": "2026-10-01T01:02:03",
        "modified_gmt": "2026-10-02T01:02:03",
    }
    record.update(kwargs)
    return record


def test_api_id_and_metadata(settings):
    record = from_api(_api_record(), "pages", settings)
    assert record["id"] == "page:123"
    assert record["source_type"] == "page"
    assert record["content"] == "Hi & bye"
    assert record["search_text"] == record["content"]
    assert json.loads(record["metadata"]) == {"wp_id": "123"}
    assert record["date"] == pd.Timestamp("2026-10-01 01:02:03")


def test_api_post_with_same_wp_id(settings):
    assert from_api(_api_record(), "posts", settings)["id"] == "post:123"


def test_baseline_numeric_id_to_prefixed(settings):
    old = {"id": "123", "source_type": "page", "title": "Admission", "content": "<p>Details</p>", "language": "az"}
    transformed = from_baseline(old, "pages", settings)
    assert transformed["id"] == "page:123"
    assert transformed["search_text"] == "Details"


def test_baseline_prefixed_id(settings):
    old = {"id": "post:15", "source_type": "post", "title": "T", "content": "C"}
    assert from_baseline(old, "posts", settings)["id"] == "post:15"


def test_invalid_baseline_source_type(settings):
    with pytest.raises(ValueError, match="mismatch"):
        from_baseline({"id": 1, "source_type": "post"}, "pages", settings)


def test_invalid_wp_id(settings):
    with pytest.raises(ValueError, match="Invalid"):
        from_api(_api_record(id="oops"), "posts", settings)


def test_dates_and_empty_output_schema():
    for event in ("new", "deleted", "snapshot"):
        frame = dataset([], event)
        assert "date" in frame and "id" in frame
        assert str(frame["date"].dtype) == "datetime64[ns]"
        assert "previous_title" not in frame


def test_cleaning_equivalence():
    assert comparable("<p>Hello&nbsp; world</p>") == comparable("Hello world")
    assert html_to_text("<p>A</p><p>B</p>") == "A\n\nB"
