from datetime import datetime, timedelta, timezone

from retrieval.search.policy import recent_document_ids
from retrieval.search.recency import RecentDocumentIndex


def test_windows_preserve_boundaries_duplicates_and_future_exclusion():
    now = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
    rows = [
        {"document_id": "boundary", "date": now - timedelta(days=365)},
        {"document_id": "old", "date": now - timedelta(days=365, microseconds=1)},
        {"document_id": "future", "date": now + timedelta(microseconds=1)},
        {"document_id": "now", "date": now},
        {"document_id": "now", "date": now},
        {"document_id": "bad", "date": "invalid"},
        {"document_id": "page", "source_type": "page", "date": "2000-01-01", "modified": now},
        {"document_id": "post", "source_type": "post", "date": "2000-01-01", "modified": now},
    ]
    index = RecentDocumentIndex(rows)
    for days in (1, 365, 730):
        for instant in (now, now + timedelta(microseconds=2), now + timedelta(days=1)):
            assert index.document_ids(days, instant) == recent_document_ids(rows, days, instant)


def test_new_snapshot_gets_its_own_index():
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    assert RecentDocumentIndex([]).document_ids(365, now) == []
    assert RecentDocumentIndex([{"document_id": "new", "date": now}]).document_ids(365, now) == ["new"]
