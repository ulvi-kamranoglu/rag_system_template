"""Batch indexing preserves untouched records and safely retries remote writes."""
import json
from types import SimpleNamespace
import numpy as np
import polars as pl
import pytest
from retrieval.indexing import sync


@pytest.mark.parametrize("mode", ["collect", "join", "reset_data", "delete_only"])
def test_batch_index_membership_updates_and_retry(tmp_path, monkeypatch, mode):
    batch = tmp_path / "batch"
    batch.mkdir()
    records = [] if mode == "delete_only" else [{"id": "post:2", "source_type": "post"}, {"id": "post:4", "source_type": "post"}]
    pl.DataFrame(records, schema={"id": pl.String, "source_type": pl.String}).write_parquet(batch / "example_canonical.parquet")
    full_ids = [1] if mode == "delete_only" else [2, 4] if mode == "reset_data" else [1, 2, 4]
    full = tmp_path / "full.parquet"
    pl.DataFrame({"id": [f"post:{i}" for i in full_ids], "source_type": ["post"] * len(full_ids)}).write_parquet(full)
    report = {"run_dir": str(batch), "output_file": "example_canonical.parquet", "output_rows": len(records)}
    (batch / "run_report.json").write_text(json.dumps(report))
    monkeypatch.setenv("PIPELINE_BATCH_REPORT", str(batch / "run_report.json"))
    monkeypatch.setenv("PIPELINE_BASE_SNAPSHOT", str(full))
    monkeypatch.setenv("PIPELINE_MODE", "collect" if mode == "delete_only" else mode)
    mapping = pl.DataFrame({"chunk_id": ["updated", "new"], "document_id": ["post:2", "post:4"], "vector_index": [0, 1]})
    mapping_path = tmp_path / "example_embedding_mapping.parquet"
    vectors_path = tmp_path / "example_embeddings.npy"
    if records:
        mapping.write_parquet(mapping_path)
        np.save(vectors_path, np.ones((2, 2), dtype=np.float32))
    pl.DataFrame({"chunk_id": ["keep", "updated", "obsolete", "deleted"],
                  "document_id": ["post:1", "post:2", "post:2", "post:3"]}).write_parquet(tmp_path / "sample_v1.5_bm25_corpus.parquet")
    (tmp_path / "active_dataset.json").write_text(json.dumps({"dataset_name": "sample_v1.5", "qdrant_collection": "test"}))
    for key, value in dict(EMBEDDINGS_PATH=vectors_path, MAPPING_PATH=mapping_path, VECTOR_DIMENSION=2,
                           QDRANT_COLLECTION="test", QDRANT_BATCH_SIZE=8, DATASET_NAME="sample_v1.6",
                           BM25_CORPUS_PATH=tmp_path / "sample_v1.6_bm25_corpus.parquet",
                           BM25_METADATA_PATH=tmp_path / "sample_v1.6_bm25_index_metadata.json",
                           DELTA_SYNC_METADATA_PATH=tmp_path / "sample_v1.6_delta_sync_metadata.json").items():
        monkeypatch.setattr(sync.config, key, value)
    monkeypatch.setattr(sync, "validate_artifacts", lambda *a: None)
    monkeypatch.setattr(sync, "build_payloads", lambda frame: frame.to_dicts())
    remote = {key: "old" for key in ["keep", "updated", "obsolete", "deleted"]}
    fail_once = [True]
    upserts = []
    class Index:
        client = SimpleNamespace(get_collection=lambda name: SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=2, distance="Cosine")))))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def health_check(self): pass
        def collection_exists(self): return True
        def count(self): return len(remote)
        def get_chunk_state(self): return dict(remote)
        def upsert(self, vectors, payloads, batch_size):
            upserts.append(len(payloads))
            assert len(payloads) == 2  # Untouched vectors are never resent.
            remote.update({row["chunk_id"]: "new" for row in payloads})
            return len(payloads)
        def delete_chunks(self, ids):
            if fail_once[0]:
                fail_once[0] = False
                raise RuntimeError("simulated disconnect after upsert")
            for key in ids: remote.pop(key)
            return len(ids)
    monkeypatch.setattr(sync, "QdrantVectorIndex", Index)
    with pytest.raises(RuntimeError, match="disconnect"):
        sync.main.__wrapped__(tmp_path)
    sync.main.__wrapped__(tmp_path)
    sync.main.__wrapped__(tmp_path)  # Idempotent after successful publication too.
    desired = {"keep"} if mode == "delete_only" else {"updated", "new"} if mode == "reset_data" else {"keep", "updated", "new"}
    assert set(remote) == desired
    assert set(pl.read_parquet(tmp_path / "sample_v1.6_bm25_corpus.parquet")["chunk_id"].to_list()) == desired
    assert json.loads((tmp_path / "active_dataset.json").read_text())["dataset_name"] == "sample_v1.6"
    if mode != "reset_data": assert remote["keep"] == "old"
    if mode == "delete_only": assert not upserts
