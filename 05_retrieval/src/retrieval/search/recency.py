"""Exact time-window lookup over an immutable, loaded corpus snapshot."""
from bisect import bisect_left, bisect_right
from datetime import timedelta

from retrieval.search.policy import timestamp


class RecentDocumentIndex:
    def __init__(self, records):
        dated = []
        for row in records:
            date = timestamp(row)
            if date is not None:
                dated.append((date, str(row["document_id"])))
        dated.sort()
        self._dates = [date for date, _ in dated]
        self._documents = [doc for _, doc in dated]

    def document_ids(self, days, now):
        start = bisect_left(self._dates, now - timedelta(days=days))
        end = bisect_right(self._dates, now)
        return sorted(set(self._documents[start:end]))
