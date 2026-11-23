import pytest

from wp_tracker.diff import compare


def test_new_and_deleted_by_wordpress_id(record):
    old = [record(number=1), record(number=2), record(number=3)]
    current = [record(number=1), record(number=2), record(number=4)]
    changes = compare(old, current)
    assert [row["id"] for row in changes.new] == ["page:4"]
    assert [row["id"] for row in changes.deleted] == ["page:3"]


def test_title_and_content_changes_do_not_emit_events(record):
    old = [record(number=1, title="Old title", content="Old content")]
    current = [record(number=1, title="Changed title", content="Changed body")]
    changes = compare(old, current)
    assert changes.new == []
    assert changes.deleted == []


def test_other_fields_do_not_emit_events(record):
    old = [record(number=1)]
    current = [record(number=1, url="https://example.com", language="ru")]
    assert compare(old, current) == compare(old, old)


def test_source_type_prevents_wordpress_id_collision(record):
    previous = [record(kind="post", number=5)]
    current = [record(kind="page", number=5)]
    changes = compare(previous, current)
    assert changes.new[0]["id"] == "page:5"
    assert changes.deleted[0]["id"] == "post:5"


def test_duplicate_current_id_fails(record):
    with pytest.raises(ValueError, match="Duplicate"):
        compare([], [record(number=3), record(number=3)])


def test_duplicate_baseline_id_fails(record):
    with pytest.raises(ValueError, match="Duplicate"):
        compare([record(number=3), record(number=3)], [])


def test_empty_data_sets():
    changes = compare([], [])
    assert changes.new == changes.deleted == []


def test_order_is_numeric_not_lexicographic(record):
    changes = compare([], [record(number=10), record(number=2)])
    assert [row["id"] for row in changes.new] == ["page:2", "page:10"]
