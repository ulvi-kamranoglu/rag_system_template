"""One shared search flow for production results and diagnostic reports."""
from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter
from retrieval.search.policy import (route, select, recent_document_ids, postrank,
                                     reranker_candidates, RECENT_RERANKER_EXTRA_LIMIT)


def run_trace(retriever, query, *, dense_limit, bm25_limit, reranker_input_limit, final_limit):
    query = str(query).strip()
    if not query:
        raise ValueError("Query must not be empty")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (dense_limit, bm25_limit, reranker_input_limit, final_limit)):
        raise ValueError("Search limits must be positive integers")
    policy = retriever.ranking_policy
    started = perf_counter()
    enabled = policy["enabled"]
    routing = route(query)
    now = datetime.now(timezone.utc)
    multiplier = policy["candidate_multiplier"] if enabled else 1
    vector = retriever.embedder.embed_query(query)
    dense_raw = retriever.vector_index.search(query_vector=vector, limit=dense_limit * multiplier)
    bm25_raw = retriever.bm25_search(query, limit=bm25_limit * multiplier)
    dense = select(dense_raw, policy, limit=dense_limit)[0] if enabled else dense_raw
    bm25 = select(bm25_raw, policy, limit=bm25_limit)[0] if enabled else bm25_raw
    # Preserve normal RRF and add evidence from each explicitly routed branch.
    pool_limit = len(dense) + len(bm25) + 1
    fusion = retriever.fusion.fuse(dense_results=dense, bm25_results=bm25, limit=pool_limit)
    by_id = {item["chunk_id"]: {**item, "retrieval_branches": [name for name in ("dense", "bm25")
              if item.get(name + "_rank") is not None]} for item in fusion}
    recent = {}
    best_dense = {}
    def remember(hits):
        for hit in hits:
            doc = hit.get("payload", {}).get("document_id")
            if doc and (doc not in best_dense or hit["score"] > best_dense[doc]["score"]):
                best_dense[doc] = hit
    remember(dense_raw)
    if enabled and policy["recent_candidates"]:
        for days in routing["recent_windows"]:
            label = f"recent_{days}"
            recent_index = getattr(retriever, "recent_index", None)
            ids = (recent_index.document_ids(days, now) if recent_index is not None
                   else recent_document_ids(retriever.bm25_index.records, days, now))
            depth = policy[f"recent_{days}_limit"]
            hits = (retriever.vector_index.search(query_vector=vector, limit=depth * multiplier,
                                                  document_ids=ids) if ids else [])
            remember(hits)
            hits = select(hits, policy, limit=depth)[0]
            recent[label] = hits
            branch = retriever.fusion.fuse(dense_results=hits, bm25_results=[], limit=max(len(hits), 1))
            for hit in branch:
                key = hit["chunk_id"]
                if key not in by_id:
                    by_id[key] = {**hit, "dense_rank": None, "dense_score": None,
                                  "rrf_score": 0.0, "retrieval_branches": []}
                by_id[key]["rrf_score"] += hit["rrf_score"]
                by_id[key][label + "_rank"] = hit["dense_rank"]
                by_id[key]["retrieval_branches"].append(label)
    fusion_all = sorted(by_id.values(), key=lambda item: -item["rrf_score"])
    fusion_all = [{**item, "hybrid_rank": rank} for rank, item in enumerate(fusion_all, 1)]
    if enabled:
        candidates, primary, extras, dropped_candidates = reranker_candidates(
            fusion_all, policy, routing, reranker_input_limit, now)
    else:
        candidates, dropped_candidates = fusion_all[:reranker_input_limit], []
        primary, extras = candidates, []
    passages = candidates
    if enabled and policy["diversify_documents"] and policy["max_chunks_per_document"] == 1:
        passages = []
        missing = list(dict.fromkeys(str(doc) for item in candidates
                       if (doc := item.get("payload", {}).get("document_id")) and doc not in best_dense))
        batch_search = getattr(retriever.vector_index, "search_documents", None)
        looked_up = set()
        if missing and callable(batch_search):
            for hits in batch_search(query_vector=vector, document_ids=missing).values():
                remember(hits)
            looked_up.update(missing)
        for item in candidates:
            doc = item.get("payload", {}).get("document_id")
            if doc and doc not in best_dense and str(doc) not in looked_up:
                remember(retriever.vector_index.search(query_vector=vector, limit=1, document_ids=[str(doc)]))
            best = best_dense.get(doc)
            if best:
                passages.append({**item, "selection_chunk_id": item["chunk_id"],
                                 "chunk_id": best["payload"]["chunk_id"], "payload": best["payload"],
                                 "passage_selection": "dense_best", "passage_dense_score": best["score"]})
            else:
                passages.append(item)
        passages, dropped_passages = select(passages, policy)
    else:
        dropped_passages = []
    retrieved_at = perf_counter()
    reranked = retriever.reranker.rerank(query=query, candidates=deepcopy(passages), limit=len(passages)) if passages else []
    reranked_at = perf_counter()
    ranked = postrank(reranked, routing, policy, now) if enabled else reranked
    final, dropped_final = select(ranked, policy, limit=final_limit) if enabled else (ranked[:final_limit], [])
    final = [{**item, "final_rank": rank} for rank, item in enumerate(final, 1)]
    stages = {"dense_raw": dense_raw, "dense": dense, "bm25_raw": bm25_raw, "bm25": bm25,
              **recent, "fusion_all": fusion_all, "fusion_primary": primary, "recent_extras": extras,
              "fusion": candidates, "document_selected": passages, "reranked": reranked,
              "postrank": ranked, "final": final}
    return {"query": query, "routing": routing, "ranking_policy": policy,
            "timings": {"retrieval_seconds": retrieved_at - started,
                        "reranker_seconds": reranked_at - retrieved_at,
                        "total_seconds": perf_counter() - started},
            "dropped": {"candidates": dropped_candidates, "passages": dropped_passages, "final": dropped_final},
            "limits": {"dense": dense_limit, "bm25": bm25_limit,
                       "reranker_input": reranker_input_limit,
                       "recent_extra_max": RECENT_RERANKER_EXTRA_LIMIT if enabled and policy["recent_candidates"] and routing["temporal_guard"] == "default" else 0,
                       "recent_extra_count": len(extras), "final": final_limit}, "stages": stages}
