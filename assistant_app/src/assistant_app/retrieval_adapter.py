"""Map active Qdrant/BM25 results to the existing chatbot hit contract."""
from pathlib import Path
import threading

_lock = threading.RLock()
_instance = None
_signature = None


def _settings():
    from retrieval.config import load_settings
    return load_settings()


def _fingerprint(settings):
    paths = (settings["BM25_CORPUS_PATH"], settings["INDEXES_DIR"] / "run_report.json",
             settings["CONFIG_PATH"])
    stamps = []
    for path in paths:
        stat = Path(path).stat()
        stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
    return (settings["DATASET_NAME"], settings["VECTOR_BACKEND"], settings["QDRANT_URL"],
            settings["QDRANT_COLLECTION"], tuple(stamps))


def _build(settings, previous):
    from retrieval.hybrid_retriever import HybridRetriever
    return HybridRetriever(settings=settings,
                           embedder=previous.embedder if previous is not None else None,
                           reranker=previous.reranker if previous is not None else None)


def search(question, k):
    global _instance, _signature
    if not str(question).strip():
        return []
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    with _lock:
        selected = _settings()
        signature = _fingerprint(selected)
        if _instance is None or signature != _signature:
            previous = _instance
            replacement = _build(selected, previous)
            _instance, _signature = replacement, signature
            if previous is not None:
                previous.close()
        ranked = _instance.search(str(question).strip(), dense_limit=selected["DENSE_TOP_K"],
                                  bm25_limit=selected["BM25_TOP_K"],
                                  reranker_input_limit=selected["RERANKER_INPUT_TOP_K"], final_limit=k)
        hits = []
        for item in ranked:
            payload = item.get("payload", {})
            hits.append({"title": str(payload.get("title") or ""),
                         "url": str(payload.get("url") or ""),
                         "source": str(payload.get("source") or ""),
                         "date": str(payload.get("date") or payload.get("modified") or "")[:10],
                         "text": str(payload.get("chunk_text") or ""),
                         "score": float(item.get("reranker_score", 0)),
                         "chunk_id": item.get("chunk_id") or payload.get("chunk_id"),
                         "document_id": payload.get("document_id"),
                         "dataset_name": selected["DATASET_NAME"]})
        return hits


def close():
    global _instance, _signature
    with _lock:
        if _instance is not None:
            _instance.close()
        _instance, _signature = None, None
