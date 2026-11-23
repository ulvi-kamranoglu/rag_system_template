"""Initial cleaning must never affect the set of tracked WordPress IDs."""
from __future__ import annotations

import pandas as pd
import pytest

from wp_tracker.pipeline import scan
from wp_tracker.records import from_api
from wp_tracker.storage import commit, read_rows
from wp_tracker.text import html_to_text, initial_cleaning_rejection


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", "empty_content"),
        (" \n\t ", "empty_content"),
        ("A b c d e", "short_content"),
        ("ABCDE", "short_content"),
        ("A B C D E F", None),
        ("https://unec.edu.az/info", "url_only_content"),
        ("www.unec.edu.az", "url_only_content"),
        ("https://one.az/\nhttps://two.az/", "url_only_content"),
        ("Read more: https://unec.edu.az/info", None),
        ("Useful educational content", None),
    ],
)
def test_initial_cleaning_rules(content, expected):
    assert initial_cleaning_rejection(content) == expected


def test_html_and_entities_are_cleaned_before_filtering():
    assert html_to_text("<p>Hi&nbsp; there</p>") == "Hi there"
    assert initial_cleaning_rejection(html_to_text("<p>   </p>")) == "empty_content"
    assert initial_cleaning_rejection(
        html_to_text('<a href="https://unec.edu.az">https://unec.edu.az</a>')
    ) == "url_only_content"


@pytest.fixture
def offline_storage(monkeypatch):
    """Use pickle as a local stand-in for Parquet transaction orchestration."""
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda frame, path, **kw: frame.to_pickle(path))
    monkeypatch.setattr(pd, "read_parquet", lambda path, **kw: pd.read_pickle(path))


class FakeClient:
    def __init__(self, pages, posts):
        self.sources = {"pages": pages, "posts": posts}

    def fetch_all(self, source):
        return self.sources[source]


def wp_item(number, content):
    return {
        "id": number,
        "title": {"rendered": f"Title {number}"},
        "content": {"rendered": content},
        "link": f"https://unec.edu.az/{number}/",
    }


def test_new_filter_does_not_affect_snapshot_or_deletions(settings, offline_storage):
    baseline = [from_api(wp_item(1, "Old"), "pages", settings)]
    commit(settings, {
        settings.baseline_file: ("snapshot", baseline),
        settings.snapshot_file: ("snapshot", baseline),
    })
    pages = [
        wp_item(10, "<p></p>"),
        wp_item(11, "<p>Short</p>"),
        wp_item(12, "https://unec.edu.az/info"),
        wp_item(13, "https://a.az/ https://b.az/"),
        wp_item(14, "<p>A meaningful paragraph.</p>"),
        wp_item(15, '<a href="https://unec.edu.az">https://unec.edu.az</a>'),
        wp_item(16, "Read more at https://unec.edu.az for details"),
    ]
    api = FakeClient(pages, [])
    dry = scan(settings, client=api, allow_large_deletions=True)
    assert dry["new_candidates"] == 7
    assert dry["new"] == 2
    assert dry["filtered_new"] == 5
    assert dry["filtered_new_reasons"] == {
        "empty_content": 1,
        "short_content": 1,
        "url_only_content": 3,
    }
    assert dry["deleted"] == 1
    assert not settings.output_file("new").exists()

    result = scan(settings, run=True, client=api, allow_large_deletions=True)
    assert (result["new"], result["deleted"]) == (2, 1)
    assert {row["id"] for row in read_rows(settings.snapshot_file)} == {
        f"page:{number}" for number in range(10, 17)
    }
    assert {row["id"] for row in read_rows(settings.output_file("new"))} == {
        "page:14", "page:16"
    }
    assert read_rows(settings.output_file("deleted"))[0]["id"] == "page:1"
    assert len(read_rows(settings.baseline_file)) == 1
    assert all(row["search_text"] == row["content"] for row in read_rows(settings.output_file("new")))

    # The baseline is fixed, so repeated scans retain the same clean deltas.
    assert scan(settings, client=api, allow_large_deletions=True)["new"] == 2
