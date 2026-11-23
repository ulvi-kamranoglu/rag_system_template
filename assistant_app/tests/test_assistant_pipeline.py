import pytest
from assistant_app import pipeline
from assistant_app.llm import client


@pytest.fixture(autouse=True)
def no_actual_log(monkeypatch):
    monkeypatch.setattr(pipeline, "log_turn", lambda row: None)


def test_greeting_does_not_call_ollama_or_retrieval(monkeypatch):
    def forbidden(*a, **kw): raise AssertionError("No external calls for greeting")
    monkeypatch.setattr(pipeline, "search", forbidden)
    monkeypatch.setattr(pipeline, "chat", forbidden)
    result = pipeline.answer("salam", [])
    assert result["intent"] == "greeting"
    assert result["language"] == "az"
    assert not result["show_sources"]


def test_answer_routes_retrieval_and_preserves_source_contract(monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda *a: {"intent": "unec", "language": "az", "search_query": "standalone question"})
    calls = []
    def search(query, k):
        calls.append(query)
        return [{"title": "Source", "url": "https://example.org", "date": "2026-10-25", "text": "Evidence", "source": "UNEC website"}]
    monkeypatch.setattr(pipeline, "search", search)
    monkeypatch.setattr(pipeline, "generate", lambda *a: "Grounded answer")
    result = pipeline.answer("follow-up", [])
    assert calls == ["standalone question"]
    assert result["answer"] == "Grounded answer"
    assert result["show_sources"]
    assert result["sources"] == [{"title": "Source", "url": "https://example.org", "date": "2026-10-25", "source": "UNEC website"}]


def test_not_found_does_not_stream_internal_sentinel(monkeypatch):
    def fake_chat(messages, temperature, on_token=None):
        assert messages[0]["role"] == "system"
        assert "ONLY facts" in messages[0]["content"]
        if on_token:
            on_token("NOT_")
            on_token("FOUND")
        return "NOT_FOUND"
    monkeypatch.setattr(client, "chat", fake_chat)
    tokens = []
    assert client.generate("q", [], [], "Azerbaijani", tokens.append) == "NOT_FOUND"
    assert tokens == []


def test_retrieval_error_is_reported_as_search_error(monkeypatch):
    monkeypatch.setattr(pipeline, "route", lambda *a: {"intent": "unec", "language": "az", "search_query": "question"})
    def broken(*a): raise OSError("Qdrant unavailable")
    monkeypatch.setattr(pipeline, "search", broken)
    with pytest.raises(pipeline.SearchError):
        pipeline.answer("question", [])
