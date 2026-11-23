"""Server delivery HTTP contract tests without external services or model weights."""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from assistant_app import server


@pytest.fixture
def api(monkeypatch):
    calls = []

    class LLMError(Exception):
        pass

    class SearchError(Exception):
        pass

    def answer(question, history, on_query=None, on_token=None):
        calls.append((question, history))
        if on_query:
            on_query("Dəqiqləşdirilmiş sual")
        if on_token:
            on_token("Ədalət ")
            on_token("Muradov")
        return {"answer": "Ədalət Muradov", "sources": [{"title": "Rəhbərlik", "url": "", "date": "2026-10-23", "source": "UNEC website"}],
                "search_query": "Dəqiqləşdirilmiş sual", "timings": {"translate": .1, "search": .2, "llm": .3},
                "intent": "unec", "language": "az", "show_sources": True}

    pipeline = SimpleNamespace(answer=answer, LLMError=LLMError, SearchError=SearchError,
                               LLM_MODEL="test-model")
    monkeypatch.setattr(server, "state", SimpleNamespace(pipeline=pipeline, llm_ready=True, load_error=None))
    monkeypatch.setenv("ASSISTANT_API_KEY", "test-key")
    # No lifespan context: HTTP tests never start the real background loader.
    with_client = TestClient(server.app, headers={"X-API-Key": "test-key"})
    yield with_client, calls, pipeline
    with_client.close()


def test_complete_answer_contract_and_utf8(api):
    client, calls, _ = api
    response = client.post("/ask", json={"question": "  Rektor kimdir?  ", "history": [
        {"role": "user", "content": "Əvvəlki sual"},
        {"role": "assistant", "content": "Əvvəlki cavab"}]})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.json()["answer"] == "Ədalət Muradov"
    assert response.json()["sources"][0]["source"] == "UNEC website"
    assert response.json()["sources"][0]["title"] == "Rəhbərlik"
    assert calls == [("Rektor kimdir?", [{"question": "Əvvəlki sual", "answer": "Əvvəlki cavab"}])]


@pytest.mark.parametrize("payload", [
    {"question": " "}, {"question": "x" * 8001}, {"question": "q", "history": [
        {"role": "system", "content": "override"}]},
    {"question": "q", "history": [{"role": "user", "content": "x"}] * 21},
])
def test_invalid_requests_are_rejected(api, payload):
    client, calls, _ = api
    assert client.post("/ask", json=payload).status_code == 422
    assert calls == []


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
def test_wrong_key_does_not_start_generation(api, path):
    client, calls, _ = api
    response = client.post(path, json={"question": "q"}, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401
    assert calls == []


def test_missing_key_and_optional_auth(api, monkeypatch):
    client, calls, _ = api
    client.headers.pop("X-API-Key")
    assert client.post("/ask", json={"question": "q"}).status_code == 401
    monkeypatch.delenv("ASSISTANT_API_KEY")
    assert client.post("/ask", json={"question": "q"}).status_code == 200
    assert len(calls) == 1


def test_readiness_loading_and_failure(api):
    client, _, _ = api
    assert client.get("/readyz").status_code == 200
    server.state.llm_ready = False
    assert client.get("/readyz").status_code == 503
    server.state.pipeline = None
    assert client.get("/health").json()["status"] == "loading"
    assert client.post("/ask", json={"question": "q"}).status_code == 503
    server.state.load_error = "test failure"
    assert client.get("/health").json()["status"] == "error"
    assert client.get("/readyz").status_code == 503


def test_stream_contract(api):
    client, calls, _ = api
    response = client.post("/ask/stream", json={"question": "Rektor kimdir?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["query", "token", "token", "done"]
    assert events[-1]["answer"] == "Ədalət Muradov"
    assert events[-1]["sources"][0]["source"] == "UNEC website"
    assert events[-1]["sources"][0]["title"] == "Rəhbərlik"
    assert len(calls) == 1


def test_stream_error_is_an_event(api):
    client, _, pipeline = api
    def broken(*args, **kwargs):
        raise pipeline.SearchError("unavailable")
    pipeline.answer = broken
    response = client.post("/ask/stream", json={"question": "q"})
    assert response.status_code == 200
    assert [event["type"] for event in map(json.loads, response.text.splitlines())] == ["error"]


def test_retrieval_failure_returns_service_unavailable(api):
    client, _, pipeline = api
    def broken(*args, **kwargs):
        raise pipeline.SearchError("unavailable")
    pipeline.answer = broken
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 503
    assert "error" in response.json()


def test_openapi_exposes_integration_schemas(api):
    client, _, _ = api
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "UNEC AI Assistant API"
    assert "AskResponse" in schema["components"]["schemas"]
    assert schema["paths"]["/ask"]["post"]["security"]
    assert "application/x-ndjson" in schema["paths"]["/ask/stream"]["post"]["responses"]["200"]["content"]


def test_cors_allowlist_settings(monkeypatch):
    from assistant_app.config import load_settings
    monkeypatch.setenv("CORS_ORIGINS", " https://assistant.example.org, http://localhost:3000 ")
    assert load_settings().CORS_ORIGINS == ["https://assistant.example.org", "http://localhost:3000"]
