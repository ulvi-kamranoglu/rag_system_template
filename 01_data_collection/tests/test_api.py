import pytest
import requests

from wp_tracker.api import WordPressClient


class Response:
    def __init__(self, batch, pages=1, total=None, status=200):
        self.batch = batch
        self.headers = {"X-WP-TotalPages": str(pages), "X-WP-Total": str(total if total is not None else len(batch))}
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(str(self.status))

    def json(self):
        return self.batch


class Session:
    def __init__(self, responses):
        self.responses = responses
        self.headers = {}
        self.requests = []

    def get(self, url, params, timeout):
        self.requests.append((url, params, timeout))
        return self.responses[params["page"] - 1]


def test_full_scan_checks_page_count(settings):
    session = Session([Response([{"id": 1}], pages=2, total=2), Response([{"id": 2}], pages=2, total=2)])
    result = WordPressClient(settings, session=session).fetch_all("pages")
    assert [row["id"] for row in result] == [1, 2]
    assert all(req[1]["status"] == "publish" for req in session.requests)
    assert all(req[1]["orderby"] == "id" for req in session.requests)


def test_incomplete_scan_fails(settings):
    session = Session([Response([{"id": 1}], pages=1, total=10)])
    with pytest.raises(RuntimeError, match="Incomplete"):
        WordPressClient(settings, session=session).fetch_all("posts")


def test_duplicate_ids_fail(settings):
    session = Session([Response([{"id": 2}, {"id": 2}], total=2)])
    with pytest.raises(RuntimeError, match="Duplicate"):
        WordPressClient(settings, session=session).fetch_all("posts")


def test_http_error_propagates(settings):
    session = Session([Response([], status=500)])
    with pytest.raises(requests.HTTPError):
        WordPressClient(settings, session=session).fetch_all("posts")


def test_invalid_source(settings):
    with pytest.raises(ValueError, match="Unsupported"):
        WordPressClient(settings, session=Session([])).fetch_all("legal")


def test_wrong_response_shape(settings):
    session = Session([Response({"error": "bad"})])
    with pytest.raises(RuntimeError, match="Unexpected"):
        WordPressClient(settings, session=session).fetch_all("pages")
