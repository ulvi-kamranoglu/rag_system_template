import json
from types import SimpleNamespace
import numpy as np
import polars as pl
import pytest
from embedding.incremental import reuse_vectors


def test_reuse_requires_exact_text_and_compatible_model(tmp_path, monkeypatch):
    from embedding import config, pipeline
    base = SimpleNamespace(embeddings_dir=tmp_path, metadata_output_path=tmp_path / "meta.json",
                           mapping_output_path=tmp_path / "map.parquet", embeddings_output_path=tmp_path / "vectors.npy")
    metadata = {"model": "test-model", "dimension": 2, "embedding_version": "v1", "max_length": 384, "use_fp16": False}
    base.metadata_output_path.write_text(json.dumps(metadata))
    pl.DataFrame({"chunk_id": ["one", "two"], "vector_index": [0, 1], "embedding_text": ["same", "old"]}).write_parquet(base.mapping_output_path)
    np.save(base.embeddings_output_path, np.array([[1, 0], [0, 1]], dtype=np.float32))
    monkeypatch.setattr(config, "load_settings", lambda *a, **kw: base)
    monkeypatch.setattr(pipeline, "validate_run", lambda p: None)
    settings = SimpleNamespace(config_path=tmp_path / "config.yaml", model_id="test-model", embedding_dimension=2,
                               embedding_version="v1", document_max_length=384, use_fp16=False)
    frame = pl.DataFrame({"chunk_id": ["one", "two", "new"], "embedding_text": ["same", "changed", "new text"]})
    vectors, missing = reuse_vectors(frame, settings, "base_v1.3")
    assert missing == [1, 2]
    np.testing.assert_array_equal(vectors[0], [1, 0])
    settings.model_id = "another-model"
    with pytest.raises(ValueError, match="configuration differs"):
        reuse_vectors(frame, settings, "base_v1.3")
