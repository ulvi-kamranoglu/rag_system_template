import pytest
import yaml
from retrieval.config import load_settings


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    for key in ("RETRIEVAL_CONFIG", "CHATBOT_PROJECT_ROOT", "PIPELINE_DATASET", "VECTOR_BACKEND", "QDRANT_URL", "QDRANT_COLLECTION"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"project_root": str(tmp_path), "dataset_name": "sample_v1.3"}), encoding="utf-8")
    return path


def test_paths_and_compatibility(settings_file):
    settings = load_settings(settings_file)
    assert settings["MAPPING_PATH"] == settings_file.parent / "shared_data/04_embeddings/sample_v1.3_embedding_mapping.parquet"
    assert settings["ARCHIVE_DIR"] == settings_file.parent / "archive_data/05_indexes"
    assert settings["RERANKER_MODEL_ID"] == "BAAI/bge-reranker-v2-m3"
    assert settings["VECTOR_DIMENSION"] == 1024
    assert settings["RERANKER_BATCH_SIZE"] == 4


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_invalid_reranker_batch(settings_file, batch):
    raw = yaml.safe_load(settings_file.read_text())
    raw["reranker"] = {"batch_size": batch}
    settings_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        load_settings(settings_file)


def test_cli_dataset_precedes_environment(settings_file, monkeypatch):
    monkeypatch.setenv("PIPELINE_DATASET", "environment_v1.4")
    assert load_settings(settings_file, dataset="explicit_v1.5")["DATASET_NAME"] == "explicit_v1.5"


def test_ambiguous_embeddings_stop(settings_file):
    raw = yaml.safe_load(settings_file.read_text())
    raw.pop("dataset_name")
    settings_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    folder = settings_file.parent / "shared_data/04_embeddings"
    folder.mkdir(parents=True)
    for name in ("first", "second"):
        (folder / f"{name}_embedding_mapping.parquet").touch()
    with pytest.raises(ValueError, match="Multiple embedding"):
        load_settings(settings_file)


def test_invalid_numeric_setting(settings_file):
    raw = yaml.safe_load(settings_file.read_text())
    raw["qdrant"] = {"batch_size": True}
    settings_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        load_settings(settings_file)
