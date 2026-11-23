from types import SimpleNamespace
import json
import numpy as np
import polars as pl
import pytest
from retrieval import sync_pipeline


@pytest.mark.parametrize("ids", [["kept", "new"], ["replacement1", "replacement2"]])
def test_sync_replaces_snapshot_and_retries_without_collection_reset(tmp_path, monkeypatch, ids):
    mapping = pl.DataFrame({"chunk_id": ids, "document_id": ["post:1", "post:3"], "vector_index": [0, 1]})
    mapping_path = tmp_path / "mapping.parquet"
    mapping.write_parquet(mapping_path)
    vector_path = tmp_path / "vectors.npy"
    np.save(vector_path, np.zeros((2, 2), dtype=np.float32))
    pl.DataFrame({"chunk_id": ["deleted", "kept"]}).write_parquet(tmp_path / "sample_v1.3_bm25_corpus.parquet")
    for key, value in dict(EMBEDDINGS_PATH=vector_path, MAPPING_PATH=mapping_path, VECTOR_DIMENSION=2,
                           QDRANT_COLLECTION="test", QDRANT_BATCH_SIZE=8, DATASET_NAME="sample_v1.4",
                           BM25_CORPUS_PATH=tmp_path / "sample_v1.4_bm25_corpus.parquet",
                           BM25_METADATA_PATH=tmp_path / "sample_v1.4_bm25_index_metadata.json",
                           DELTA_SYNC_METADATA_PATH=tmp_path / "sample_v1.4_delta_sync_metadata.json").items():
        monkeypatch.setattr(sync_pipeline.config, key, value)
    monkeypatch.setattr(sync_pipeline, "validate_artifacts", lambda *a: None)
    monkeypatch.setattr(sync_pipeline, "build_payloads", lambda frame: frame.to_dicts())
    state = {"deleted": {}, "kept": {}}
    class FakeIndex:
        client = SimpleNamespace(get_collection=lambda name: SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=2, distance="Cosine")))))
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def health_check(self): return True
        def collection_exists(self): return True
        def get_chunk_state(self): return dict(state)
        def count(self): return len(state)
        def upsert(self, vectors, payloads, batch_size):
            state.update({row["chunk_id"]: {} for row in payloads})
            return len(payloads)
        def delete_chunks(self, ids):
            for key in ids: state.pop(key)
            return len(ids)
    monkeypatch.setattr(sync_pipeline, "QdrantVectorIndex", FakeIndex)
    sync_pipeline.main.__wrapped__(tmp_path)
    assert set(state) == set(ids)
    assert json.loads((tmp_path / "active_dataset.json").read_text())["dataset_name"] == "sample_v1.4"
    sync_pipeline.main.__wrapped__(tmp_path)
    assert set(state) == set(ids)
