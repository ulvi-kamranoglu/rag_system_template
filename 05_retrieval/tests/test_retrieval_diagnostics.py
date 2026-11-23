from types import SimpleNamespace
from retrieval.diagnostics import diagnose, compact


def hit(identity, source):
    return {"chunk_id": identity, "payload": {"chunk_id": identity, "source_type": source, "title": "test"}}


def test_trace_identifies_candidate_loss_without_changing_fusion():
    calls = []
    dense = [hit("old", "post"), hit("new", "research")]
    bm25 = [hit("new", "research")]
    def fuse(**kwargs):
        calls.append(kwargs)
        return [dense[0]]
    def rerank(**kwargs):
        kwargs["candidates"][0]["payload"]["title"] = "mutation"
        return kwargs["candidates"]
    retriever = SimpleNamespace(dense_search=lambda *a, **kw: dense, bm25_search=lambda *a, **kw: bm25,
                               fusion=SimpleNamespace(fuse=fuse), reranker=SimpleNamespace(rerank=rerank))
    report = diagnose(retriever, "query", dense_limit=2, bm25_limit=2,
                      reranker_input_limit=1, final_limit=1, focus_source="research")
    assert report["summary"]["dense"]["focus_ranks"] == [2]
    assert report["summary"]["bm25"]["focus_ranks"] == [1]
    assert report["summary"]["fusion"]["focus_ranks"] == []
    assert dense[0]["payload"]["title"] == "test"
    assert calls[0]["limit"] == 1
    assert len(compact(report)["stages"]) == 5


def test_empty_candidates_do_not_invoke_reranker():
    def forbidden(**kwargs):
        raise AssertionError("No candidates to rerank")
    retriever = SimpleNamespace(dense_search=lambda *a, **kw: [], bm25_search=lambda *a, **kw: [],
                               fusion=SimpleNamespace(fuse=lambda **kw: []), reranker=SimpleNamespace(rerank=forbidden))
    report = diagnose(retriever, "query", dense_limit=20, bm25_limit=20, reranker_input_limit=20, final_limit=5)
    assert report["stages"]["final"] == []
