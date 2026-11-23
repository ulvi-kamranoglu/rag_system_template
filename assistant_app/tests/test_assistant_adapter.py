from types import SimpleNamespace
import pytest
from assistant_app import retrieval_adapter as adapter


def test_adapter_maps_hits_and_refreshes_dataset_without_reloading_models(monkeypatch):
    selected = {"DATASET_NAME": "sample_v1.4", "DENSE_TOP_K": 20,
                "BM25_TOP_K": 20, "RERANKER_INPUT_TOP_K": 20}
    built = []
    def build(settings, previous):
        instance = SimpleNamespace(embedder=object() if previous is None else previous.embedder,
                                   reranker=object() if previous is None else previous.reranker,
                                   closed=False)
        def close(): instance.closed = True
        instance.close = close
        instance.search = lambda *a, **kw: [{"chunk_id": "research:1:chunk:0000", "reranker_score": .9,
            "payload": {"document_id": "research:1", "title": "Title", "chunk_text": "Grounded text",
                        "url": "https://example.org", "date": "2026-10-25T12:00:00Z", "source": "UNEC website"}}]
        built.append(instance)
        return instance
    monkeypatch.setattr(adapter, "_instance", None)
    monkeypatch.setattr(adapter, "_signature", None)
    monkeypatch.setattr(adapter, "_settings", lambda: dict(selected))
    monkeypatch.setattr(adapter, "_fingerprint", lambda s: s["DATASET_NAME"])
    monkeypatch.setattr(adapter, "_build", build)
    first = adapter.search("question", 5)[0]
    assert first["text"] == "Grounded text"
    assert first["date"] == "2026-10-25"
    assert first["source"] == "UNEC website"
    assert first["dataset_name"] == "sample_v1.4"
    adapter.search("again", 5)
    assert len(built) == 1
    selected["DATASET_NAME"] = "sample_v1.5"
    assert adapter.search("question", 5)[0]["dataset_name"] == "sample_v1.5"
    assert built[0].closed
    assert built[1].embedder is built[0].embedder
    assert built[1].reranker is built[0].reranker
    adapter.close()
    assert built[1].closed


def test_adapter_empty_question_never_loads_models(monkeypatch):
    def forbidden(): raise AssertionError("Must not load settings or models")
    monkeypatch.setattr(adapter, "_settings", forbidden)
    assert adapter.search("  ", 5) == []
    with pytest.raises(ValueError):
        adapter.search("question", 0)
