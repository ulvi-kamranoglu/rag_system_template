"""Trace the same configured search flow used for production results."""
from copy import deepcopy
from collections import Counter


def diagnose(retriever, query, *, dense_limit, bm25_limit, reranker_input_limit,
             final_limit, focus_source=None):
    query = str(query).strip()
    if not query:
        raise ValueError("Query must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (dense_limit, bm25_limit, reranker_input_limit, final_limit)):
        raise ValueError("Search limits must be positive integers")
    trace = None
    if hasattr(retriever, "trace_search"):
        trace = retriever.trace_search(query, dense_limit=dense_limit, bm25_limit=bm25_limit,
                                       reranker_input_limit=reranker_input_limit, final_limit=final_limit)
        stages = trace["stages"]
    else:
        stages = _legacy_stages(retriever, query, dense_limit, bm25_limit, reranker_input_limit, final_limit)
    summary = {}
    for stage, results in stages.items():
        sources = Counter(str(item.get("payload", {}).get("source_type") or "unknown") for item in results)
        focus_ranks = [rank for rank, item in enumerate(results, 1)
                       if focus_source and item.get("payload", {}).get("source_type") == focus_source]
        summary[stage] = {"count": len(results), "source_type_counts": dict(sources),
                          "focus_ranks": focus_ranks}
    return {**(trace or {}), "query": query, "focus_source": focus_source,
            "limits": {**(trace or {}).get("limits", {}), "dense": dense_limit, "bm25": bm25_limit,
                       "reranker_input": reranker_input_limit, "final": final_limit},
            "summary": summary, "stages": stages,
            "note": "Missing from candidates does not mean absent from the index; scores are not factual confidence."}


def _legacy_stages(retriever, query, dense_limit, bm25_limit, reranker_input_limit, final_limit):
    dense = retriever.dense_search(query, limit=dense_limit)
    bm25 = retriever.bm25_search(query, limit=bm25_limit)
    # Keep the fusion boundary exactly the same as regular search.
    fusion = retriever.fusion.fuse(dense_results=dense, bm25_results=bm25,
                                  limit=reranker_input_limit)
    reranked = (retriever.reranker.rerank(query=query, candidates=deepcopy(fusion), limit=len(fusion))
                if fusion else [])
    return {"dense": dense, "bm25": bm25, "fusion": fusion,
              "reranked": reranked, "final": reranked[:final_limit]}


def compact(report, preview_limit=5):
    """Bound terminal output; full payloads remain available in the JSON report."""
    rows = []
    for stage, results in report["stages"].items():
        previews = []
        for rank, item in enumerate(results[:0 if stage.endswith("_raw") else preview_limit], 1):
            payload = item.get("payload", {})
            previews.append({"rank": rank, "chunk_id": item.get("chunk_id") or payload.get("chunk_id"),
                             "source_type": payload.get("source_type"), "title": str(payload.get("title") or "")[:160],
                             "score": next((item[key] for key in ("postrank_score", "reranker_score", "rrf_score", "score") if key in item), None)})
            if "freshness_bonus" in item:
                previews[-1].update({key: item.get(key) for key in
                    ("ranking_date", "ranking_date_field", "freshness_bonus", "page_bonus", "reranker_base_score")})
            if "reranker_date_context" in item:
                previews[-1]["reranker_date_context"] = item["reranker_date_context"]
        rows.append({"stage": stage, **report["summary"][stage], "preview": previews})
    return {"dataset_name": report.get("dataset_name"), "query": report["query"],
            "routing": report.get("routing"), "ranking_policy": report.get("ranking_policy"),
            "timings": report.get("timings"),
            "dropped_counts": {stage: len(values) for stage, values in report.get("dropped", {}).items()},
            "focus_source": report["focus_source"], "limits": report["limits"], "stages": rows,
            "note": report["note"]}
