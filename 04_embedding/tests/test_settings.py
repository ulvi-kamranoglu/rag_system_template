"""Configuration/serialization tests; no model downloads or GPU required."""
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from embedding.config import load_settings
from embedding.pipeline import run, validate_run
from pipeline_storage import inventory


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv("PIPELINE_DATASET", raising=False)
    project = tmp_path / "embedding"
    path = project / "config/config.yaml"
    path.parent.mkdir(parents=True)
    source = tmp_path / "shared_data/03_chunks/example_v1.3_chunks_256.parquet"
    source.parent.mkdir(parents=True)
    pl.DataFrame({"document_id": ["form:1"], "chunk_id": ["form:1:chunk:0000"],
                  "chunk_index": [0], "embedding_text": ["Useful university information."],
                  "embedding_tokens": [12]}).write_parquet(source)
    path.write_text(yaml.safe_dump({"input_dir": "../shared_data/03_chunks",
        "storage": {"shared_dir": "../shared_data/04_embeddings"},
        "model": {"id": "test-model", "dimension": 8, "document_max_length": 256,
                  "query_max_length": 64, "batch_size": 2, "use_fp16": True, "version": "test-v1"}}))
    return path


def test_yaml_settings_and_cwd_independence(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(config)
    assert settings.dataset_name == "example_v1.3"
    assert settings.chunks_input_path.name == "example_v1.3_chunks_256.parquet"
    assert settings.embeddings_output_path.name == "example_v1.3_embeddings.npy"
    assert settings.document_batch_size == 2
    assert settings.document_max_length == 256
    assert settings.use_fp16 is True


def test_dry_run_does_not_import_model_or_write(config, monkeypatch):
    monkeypatch.setitem(sys.modules, "embedding.embedder", None)
    settings = load_settings(config)
    report = run(config, dry_run=True)
    assert report["rows"] == 1
    assert report["dimension"] == 8
    assert not settings.embeddings_dir.exists()
    assert not settings.archive_dir.exists()


def test_fake_model_receives_yaml_settings_and_serializes_matching_outputs(config, monkeypatch):
    calls = []
    fake_module = types.ModuleType("embedding.embedder")
    class FakeEmbedder:
        device = "cpu"
        def __init__(self, settings):
            self.settings = settings
            calls.append(settings)
        def embed_texts(self, texts, batch_size):
            assert batch_size == self.settings.document_batch_size == 2
            vectors = np.zeros((len(texts), self.settings.embedding_dimension), dtype=np.float32)
            vectors[:, 0] = 1
            return vectors
    fake_module.BgeM3Embedder = FakeEmbedder
    monkeypatch.setitem(sys.modules, "embedding.embedder", fake_module)
    run(config)
    settings = calls[0]
    assert np.load(settings.embeddings_output_path).shape == (1, 8)
    assert pl.read_parquet(settings.mapping_output_path)["vector_index"].to_list() == [0]
    assert settings.metadata_output_path.is_file()
    archives = list(settings.archive_dir.glob("run_*"))
    assert len(archives) == 1
    validate_run(archives[0])
    assert inventory(archives[0]) == inventory(settings.embeddings_dir)


def test_invalid_config_fails_before_model(config):
    raw = yaml.safe_load(config.read_text())
    raw["model"]["batch_size"] = 0
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="batch_size"):
        load_settings(config)


@pytest.fixture
def fake_model(monkeypatch):
    module = types.ModuleType("embedding.embedder")
    class FakeEmbedder:
        device = "cpu"
        def __init__(self, settings):
            self.settings = settings
        def embed_texts(self, texts, batch_size):
            vectors = np.zeros((len(texts), self.settings.embedding_dimension), dtype=np.float32)
            vectors[:, 0] = 1
            return vectors
    module.BgeM3Embedder = FakeEmbedder
    monkeypatch.setitem(sys.modules, "embedding.embedder", module)


def test_history_and_complete_shared_replacement(config, fake_model):
    settings = load_settings(config)
    first = run(config)
    original = inventory(Path(first["run_dir"]))
    (settings.embeddings_dir / "obsolete.npy").write_bytes(b"stale")
    second = run(config)
    assert first["run_dir"] != second["run_dir"]
    assert inventory(Path(first["run_dir"])) == original
    assert len(list(settings.archive_dir.glob("run_*"))) == 2
    assert not (settings.embeddings_dir / "obsolete.npy").exists()
    assert inventory(Path(second["run_dir"])) == inventory(settings.embeddings_dir)


def test_failed_publication_preserves_archive_and_retries_without_model_or_input(config, fake_model, monkeypatch):
    settings = load_settings(config)
    run(config)
    before = inventory(settings.embeddings_dir)
    previous = set(settings.archive_dir.glob("run_*"))
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError("publication unavailable")
        patch.setattr("embedding.pipeline.publish", fail)
        with pytest.raises(RuntimeError, match="--publish-run"):
            run(config)
    assert inventory(settings.embeddings_dir) == before
    target, = set(settings.archive_dir.glob("run_*")) - previous
    validate_run(target)
    settings.chunks_input_path.unlink()
    monkeypatch.setitem(sys.modules, "embedding.embedder", None)
    # Publication must ignore even a newly configured unrelated input file.
    raw = yaml.safe_load(config.read_text())
    raw["input_path"] = "../missing/other_chunks_256.parquet"
    config.write_text(yaml.safe_dump(raw))
    result = run(config, publish_run=target)
    assert result["publication"]["status"] == "synced"
    assert inventory(target) == inventory(settings.embeddings_dir)
    again = run(config, publish_run=target)
    assert again["publication"]["status"] == "already_synced"


def test_archive_write_failure_leaves_shared_untouched(config, fake_model, monkeypatch):
    settings = load_settings(config)
    run(config)
    before = inventory(settings.embeddings_dir)
    archives = set(settings.archive_dir.glob("run_*"))
    def fail(*args, **kwargs):
        raise OSError("mapping write failed")
    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail)
    with pytest.raises(OSError, match="mapping write failed"):
        run(config)
    assert inventory(settings.embeddings_dir) == before
    assert set(settings.archive_dir.glob("run_*")) == archives
    assert not list(settings.archive_dir.glob("*.partial"))


def test_archive_validator_rejects_mapping_misalignment(config, fake_model):
    settings = load_settings(config)
    result = run(config)
    target = Path(result["run_dir"])
    path = target / settings.mapping_output_path.name
    pl.read_parquet(path).with_columns(pl.lit(8).alias("vector_index")).write_parquet(path)
    with pytest.raises(ValueError, match="aligned"):
        validate_run(target)


def test_overlapping_storage_and_publish_dry_run_rejected(config):
    raw = yaml.safe_load(config.read_text())
    raw["storage"]["archive_dir"] = raw["storage"]["shared_dir"]
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="overlap"):
        load_settings(config)
    with pytest.raises(ValueError, match="cannot be combined"):
        run(config, dry_run=True, publish_run="unused")
