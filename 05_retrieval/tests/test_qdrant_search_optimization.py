from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PayloadSchemaType

from retrieval.backends.qdrant import QdrantVectorIndex, VECTOR_DIMENSION


def backend(client):
    index = QdrantVectorIndex.__new__(QdrantVectorIndex)
    index.client = client
    index.collection_name = "test"
    return index


def test_search_does_not_preflight_and_preserves_document_filter():
    calls = []
    def query(**kw):
        calls.append(kw)
        return SimpleNamespace(points=[SimpleNamespace(id="point", score=.8, payload={"chunk_id": "chunk"})])
    index = backend(SimpleNamespace(query_points=query))
    result = index.search(np.ones(VECTOR_DIMENSION), document_ids=["post:1"], limit=1)
    assert result == [{"id": "point", "score": .8, "payload": {"chunk_id": "chunk"}}]
    assert calls[0]["query_filter"].must[0].match.any == ["post:1"]
    assert index.search(np.ones(VECTOR_DIMENSION), document_ids=[]) == []
    assert len(calls) == 1


@pytest.mark.parametrize("status", [404, 403, 500])
def test_missing_collection_is_clear_but_other_errors_propagate(status):
    error = UnexpectedResponse(status_code=status, reason_phrase="failure", content=b'{}', headers={})
    def fail(**kw):
        raise error
    index = backend(SimpleNamespace(query_points=fail))
    if status == 404:
        with pytest.raises(RuntimeError, match="collection does not exist") as caught:
            index.search(np.ones(VECTOR_DIMENSION))
        assert caught.value.__cause__ is error
    else:
        with pytest.raises(UnexpectedResponse) as caught:
            index.search(np.ones(VECTOR_DIMENSION))
        assert caught.value is error


def test_payload_index_is_created_once_without_recreating_collection():
    schema = {}
    calls = []
    def create(**kw):
        calls.append(kw)
        schema["document_id"] = SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)
    index = backend(SimpleNamespace(get_collection=lambda name: SimpleNamespace(payload_schema=schema),
                                    create_payload_index=create))
    index.ensure_payload_indexes()
    index.ensure_payload_indexes()
    assert len(calls) == 1
    assert calls[0]["field_name"] == "document_id"
    assert calls[0]["wait"] is True


def test_conflicting_payload_index_is_not_silently_replaced():
    index = backend(SimpleNamespace(get_collection=lambda name: SimpleNamespace(
        payload_schema={"document_id": SimpleNamespace(data_type=PayloadSchemaType.INTEGER)})))
    with pytest.raises(ValueError, match="keyword"):
        index.ensure_payload_indexes()


def test_batched_document_queries_preserve_filters_order_and_empty_results():
    seen = []
    def batch(**kw):
        seen.extend(kw["requests"])
        return [SimpleNamespace(points=[SimpleNamespace(id="best", score=.7, payload={"document_id": "b"})]),
                SimpleNamespace(points=[])]
    index = backend(SimpleNamespace(query_batch_points=batch))
    result = index.search_documents(np.ones(VECTOR_DIMENSION), ["b", "a", "b"])
    assert list(result) == ["b", "a"]
    assert result["b"][0]["score"] == .7 and result["a"] == []
    assert [req.filter.must[0].match.any for req in seen] == [["b"], ["a"]]
    assert all(req.limit == 1 and req.with_payload for req in seen)
    assert index.search_documents(np.ones(VECTOR_DIMENSION), []) == {}


def test_batch_response_mismatch_fails_explicitly():
    index = backend(SimpleNamespace(query_batch_points=lambda **kw: []))
    with pytest.raises(RuntimeError, match="response count mismatch"):
        index.search_documents(np.ones(VECTOR_DIMENSION), ["doc"])
