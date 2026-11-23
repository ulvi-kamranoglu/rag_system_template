"""Deployment safeguards for the independently delivered API checkout."""
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_local_modules_resolve_inside_delivery():
    for name in ("pipeline_storage", "wp_tracker", "canonical_pipeline", "chunking_pipeline",
                 "embedding", "retrieval", "orchestration", "assistant_app"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(ROOT), name


def test_bootstrap_refuses_existing_collection_without_writes(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import bootstrap_indexes as bootstrap
    from retrieval import config
    from retrieval.backends import qdrant
    from retrieval.indexing import sync
    calls = []

    class Index:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def health_check(self):
            return True
        def collection_exists(self):
            return True
        def ensure_collection(self):
            calls.append("create")

    monkeypatch.setattr(config, "load_settings", lambda: {
        "INDEXES_DIR": tmp_path, "VECTOR_BACKEND": "qdrant", "QDRANT_URL": "http://example.invalid:6333",
        "QDRANT_COLLECTION": "example", "DATASET_NAME": "sample_v1.0"})
    monkeypatch.setattr(bootstrap, "snapshot", lambda _: (1, {"chunk-1"}))
    monkeypatch.setattr(qdrant, "QdrantVectorIndex", Index)
    monkeypatch.setattr(sync, "main", lambda _: calls.append("sync"))
    with pytest.raises(RuntimeError, match="already exists"):
        bootstrap.main()
    assert calls == []


def test_bootstrap_creates_then_syncs_new_target(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import bootstrap_indexes as bootstrap
    from retrieval import config
    from retrieval.backends import qdrant
    from retrieval.indexing import sync
    calls = []

    class Index:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def health_check(self):
            return True
        def collection_exists(self):
            return False
        def ensure_collection(self):
            calls.append("create")

    monkeypatch.setattr(config, "load_settings", lambda: {
        "INDEXES_DIR": tmp_path, "VECTOR_BACKEND": "qdrant", "QDRANT_URL": "http://example.invalid:6333",
        "QDRANT_COLLECTION": "example", "DATASET_NAME": "sample_v1.0"})
    monkeypatch.setattr(bootstrap, "snapshot", lambda _: (1, {"chunk-1"}))
    monkeypatch.setattr(qdrant, "QdrantVectorIndex", Index)
    monkeypatch.setattr(sync, "main", lambda _: calls.append("sync"))
    bootstrap.main()
    assert calls == ["create", "sync"]
