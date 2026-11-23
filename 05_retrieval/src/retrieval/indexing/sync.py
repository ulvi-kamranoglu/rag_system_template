"""Idempotently reconcile a complete desired snapshot without recreating Qdrant."""
import json
import os
from pathlib import Path
import numpy as np
import polars as pl
from pipeline_storage import write_json
from retrieval import config
from retrieval.indexing.storage import archived_stage
from retrieval.artifacts import validate_artifacts, build_payloads
from retrieval.backends.qdrant import QdrantVectorIndex
from pipeline_storage.documents import document_id


def sync_batch(output_dir):
    """Merge batch vectors into the active corpus, or replace it for reset_data.

    Canonical's immutable complete snapshot determines document membership;
    only this batch's vectors are uploaded. Stable chunk IDs make retries safe.
    """
    report_path = Path(os.environ["PIPELINE_BATCH_REPORT"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    canonical = pl.read_parquet(report_path.parent / report["output_file"])
    full = pl.read_parquet(os.environ["PIPELINE_BASE_SNAPSHOT"])
    allowed_documents = {document_id(row) for row in full.iter_rows(named=True)}
    touched = {document_id(row) for row in canonical.iter_rows(named=True)}
    active = output_dir / "active_dataset.json"
    previous = json.loads(active.read_text(encoding="utf-8"))
    if previous.get("qdrant_collection") != config.QDRANT_COLLECTION:
        raise ValueError("Active dataset belongs to another Qdrant collection")
    previous_name = previous["dataset_name"]
    if not isinstance(previous_name, str) or any(c in previous_name for c in '/\\:*?"<>|'):
        raise ValueError("Invalid active dataset name")
    baseline = pl.read_parquet(output_dir / f"{previous_name}_bm25_corpus.parquet")
    vectors = None
    if report["output_rows"]:
        vectors = np.load(config.EMBEDDINGS_PATH, mmap_mode="r")
        mapping = pl.read_parquet(config.MAPPING_PATH)
        validate_artifacts(vectors, mapping)
        mask = np.array([str(value) in allowed_documents for value in mapping["document_id"].to_list()], dtype=bool)
        vectors = vectors[mask]
        mapping = mapping.filter(pl.Series(mask))
        payloads = build_payloads(mapping)
        incoming = mapping.drop("vector_index") if "vector_index" in mapping.columns else mapping
    else:
        incoming = baseline.head(0)
        payloads = []
    if os.environ["PIPELINE_MODE"] == "reset_data":
        desired_corpus = incoming
    else:
        retained = baseline.filter(pl.col("document_id").is_in(sorted(allowed_documents - touched)))
        desired_corpus = pl.concat([retained, incoming], how="diagonal_relaxed")
    desired = set(desired_corpus["chunk_id"].to_list())
    if len(desired) != desired_corpus.height:
        raise ValueError("Duplicate chunk IDs in merged index corpus")
    if not desired:
        raise ValueError("Refusing to publish an empty searchable corpus")
    desired_corpus.write_parquet(output_dir / config.BM25_CORPUS_PATH.name)
    with QdrantVectorIndex() as index:
        index.health_check()
        if not index.collection_exists():
            raise ValueError("Initial Qdrant collection is missing")
        parameters = index.client.get_collection(config.QDRANT_COLLECTION).config.params.vectors
        distance = getattr(parameters, "distance", "")
        if getattr(parameters, "size", None) != config.VECTOR_DIMENSION or str(getattr(distance, "value", distance)).lower() != "cosine":
            raise ValueError("Qdrant vector configuration differs from the batch")
        current = index.get_chunk_state()
        if index.count() != len(current) or not set(current).issubset(set(baseline["chunk_id"].to_list()) | desired):
            raise ValueError("Qdrant contains unexpected points outside the managed baseline")
        indexed = index.upsert(vectors, payloads, config.QDRANT_BATCH_SIZE) if payloads else 0
        removed = index.delete_chunks(sorted(set(current) - desired))
        if set(index.get_chunk_state()) != desired or index.count() != desired_corpus.height:
            raise RuntimeError("Qdrant and merged BM25 corpus differ; resume the pipeline")
    write_json(output_dir / config.BM25_METADATA_PATH.name, {
        "documents": desired_corpus.height, "index_type": "bm25", "mode": "batch_sync",
        "corpus_file": str(config.BM25_CORPUS_PATH)})
    write_json(output_dir / config.DELTA_SYNC_METADATA_PATH.name, {
        "mode": os.environ["PIPELINE_MODE"], "qdrant_before": len(current), "qdrant_after": desired_corpus.height,
        "bm25_corpus_chunks": desired_corpus.height, "upserted_chunks": indexed,
        "deleted_chunks": removed, "previous_dataset": previous_name})
    write_json(active, {"dataset_name": config.DATASET_NAME, "qdrant_collection": config.QDRANT_COLLECTION})


@archived_stage("sync", ("BM25_CORPUS_PATH", "BM25_METADATA_PATH", "DELTA_SYNC_METADATA_PATH"))
def main(output_dir):
    if os.getenv("PIPELINE_BATCH_REPORT"):
        return sync_batch(output_dir)
    vectors = np.load(config.EMBEDDINGS_PATH, mmap_mode="r")
    mapping = pl.read_parquet(config.MAPPING_PATH)
    validate_artifacts(vectors, mapping)
    payloads = build_payloads(mapping)
    desired = set(mapping["chunk_id"].to_list())
    active = output_dir / "active_dataset.json"
    previous_name = None
    if active.is_file():
        previous = json.loads(active.read_text(encoding="utf-8"))
        previous_name = previous["dataset_name"]
        if previous.get("qdrant_collection") != config.QDRANT_COLLECTION:
            raise ValueError("Active dataset belongs to another Qdrant collection")
    if previous_name is None:
        candidates = list(output_dir.glob("*_bm25_corpus.parquet"))
        if len(candidates) != 1:
            raise ValueError("An unambiguous existing BM25 baseline is required before sync")
        previous_name = candidates[0].name[:-len("_bm25_corpus.parquet")]
    if not isinstance(previous_name, str) or any(c in previous_name for c in '/\\:*?"<>|'):
        raise ValueError("Invalid active dataset name")
    baseline = pl.read_parquet(output_dir / f"{previous_name}_bm25_corpus.parquet")
    base_ids = set(baseline["chunk_id"].to_list())
    # Persist the complete desired corpus before any remote write.
    mapping.write_parquet(output_dir / config.BM25_CORPUS_PATH.name)
    with QdrantVectorIndex() as index:
        index.health_check()
        if not index.collection_exists():
            raise ValueError("Initial Qdrant index is missing; create it before incremental sync")
        information = index.client.get_collection(config.QDRANT_COLLECTION)
        parameters = information.config.params.vectors
        distance = getattr(parameters, "distance", "")
        if getattr(parameters, "size", None) != config.VECTOR_DIMENSION or str(getattr(distance, "value", distance)).lower() != "cosine":
            raise ValueError("Qdrant vector configuration differs from the snapshot")
        current = index.get_chunk_state()
        if index.count() != len(current) or not set(current).issubset(base_ids | desired):
            raise ValueError("Qdrant contains unexpected points outside the managed baseline")
        # Upsert first. Stable UUIDs make restarting an interrupted sync safe.
        indexed = index.upsert(vectors, payloads, config.QDRANT_BATCH_SIZE)
        removed = index.delete_chunks(sorted(set(current) - desired))
        final = index.get_chunk_state()
        if set(final) != desired or index.count() != mapping.height:
            raise RuntimeError("Qdrant IDs/count do not match desired snapshot; resume this pipeline")
    write_json(output_dir / config.BM25_METADATA_PATH.name, {
        "documents": mapping.height, "index_type": "bm25", "mode": "snapshot_sync",
        "corpus_file": str(config.BM25_CORPUS_PATH)})
    write_json(output_dir / config.DELTA_SYNC_METADATA_PATH.name, {
        "mode": "snapshot_sync", "qdrant_before": len(current), "qdrant_after": mapping.height,
        "bm25_corpus_chunks": mapping.height, "upserted_chunks": indexed,
        "deleted_chunks": removed, "previous_dataset": previous_name})
    write_json(active, {"dataset_name": config.DATASET_NAME,
                        "qdrant_collection": config.QDRANT_COLLECTION})


if __name__ == "__main__":
    main()
