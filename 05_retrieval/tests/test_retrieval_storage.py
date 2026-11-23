"""Local archive/publication regressions without a Qdrant service or model."""
from pathlib import Path
import json

import polars as pl
import pytest
from pipeline_storage import inventory, write_json
from retrieval import config
from retrieval.storage import archived_stage, retry_publication, validate_run


def test_src_layout_still_rejects_outputs_inside_module(monkeypatch):
    from retrieval.indexing import storage as implementation
    source = Path(implementation.__file__).resolve()
    module = source.parents[3]
    monkeypatch.setattr(config, "ARCHIVE_DIR", module / "generated/archive")
    monkeypatch.setattr(config, "INDEXES_DIR", module / "generated/shared")
    with pytest.raises(ValueError, match="outside the module"):
        implementation._roots()


@pytest.fixture
def storage(tmp_path, monkeypatch):
    shared = tmp_path / "shared_data/05_indexes"
    monkeypatch.setattr(config, "INDEXES_DIR", shared)
    monkeypatch.setattr(config, "ARCHIVE_DIR", tmp_path / "archive_data/05_indexes")
    monkeypatch.setattr(config, "DATASET_NAME", "example")
    monkeypatch.setattr(config, "BM25_CORPUS_PATH", shared / "example_bm25_corpus.parquet")
    monkeypatch.setattr(config, "BM25_METADATA_PATH", shared / "example_bm25_index_metadata.json")
    monkeypatch.setattr(config, "QDRANT_INDEX_METADATA_PATH", shared / "example_qdrant_index_metadata.json")
    return shared


def write_bm25(folder, chunk_id="form:1:chunk:0000"):
    pl.DataFrame({"document_id": ["form:1"], "chunk_id": [chunk_id]}).write_parquet(
        folder / config.BM25_CORPUS_PATH.name)
    write_json(folder / config.BM25_METADATA_PATH.name,
               {"documents": 1, "corpus_file": str(config.BM25_CORPUS_PATH)})


@archived_stage("bm25", ("BM25_CORPUS_PATH", "BM25_METADATA_PATH"))
def bm25_stage(folder):
    write_bm25(folder)


@archived_stage("qdrant", ("QDRANT_INDEX_METADATA_PATH",))
def fake_qdrant_stage(folder):
    write_json(folder / config.QDRANT_INDEX_METADATA_PATH.name,
               {"indexed_points": 1, "final_point_count": 1})


def test_completed_states_preserve_other_stages_and_history(storage):
    first = fake_qdrant_stage([])
    archived = inventory(Path(first["run_dir"]))
    second = bm25_stage([])
    assert config.QDRANT_INDEX_METADATA_PATH.is_file()
    assert config.BM25_CORPUS_PATH.is_file()
    assert inventory(storage) == inventory(Path(second["run_dir"]))
    assert inventory(Path(first["run_dir"])) == archived
    validate_run(Path(second["run_dir"]))
    assert len(list(config.ARCHIVE_DIR.glob("run_*"))) == 2


def test_failure_before_archive_completion_does_not_replace_shared(storage):
    bm25_stage([])
    before = inventory(storage)
    @archived_stage("bm25", ("BM25_CORPUS_PATH", "BM25_METADATA_PATH"))
    def fail(folder):
        write_bm25(folder)
        raise OSError("stage failed")
    with pytest.raises(OSError, match="stage failed"):
        fail([])
    assert inventory(storage) == before
    assert len(list(config.ARCHIVE_DIR.glob("run_*"))) == 1


def test_publication_failure_retry_never_repeats_stage(storage, monkeypatch):
    bm25_stage([])
    before = inventory(storage)
    existing = set(config.ARCHIVE_DIR.glob("run_*"))
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError("copy unavailable")
        patch.setattr("retrieval.storage.publish", fail)
        with pytest.raises(RuntimeError, match="retrieval.storage"):
            fake_qdrant_stage([])
    assert inventory(storage) == before
    target, = set(config.ARCHIVE_DIR.glob("run_*")) - existing
    result = retry_publication(target)
    assert result["publication"]["status"] == "synced"
    assert inventory(target) == inventory(storage)
    assert retry_publication(target)["publication"]["status"] == "already_synced"


def test_old_retry_cannot_overwrite_newer_stage(storage):
    first = bm25_stage([])
    fake_qdrant_stage([])
    before = inventory(storage)
    with pytest.raises(RuntimeError, match="newer results"):
        retry_publication(first["run_dir"])
    assert inventory(storage) == before


def test_copied_old_output_cannot_hide_missing_new_output(storage):
    bm25_stage([])
    before = inventory(storage)
    @archived_stage("bm25", ("BM25_CORPUS_PATH", "BM25_METADATA_PATH"))
    def incomplete(folder):
        pass
    with pytest.raises(ValueError, match="expected file"):
        incomplete([])
    assert inventory(storage) == before


def test_tampered_archive_is_rejected(storage):
    result = bm25_stage([])
    target = Path(result["run_dir"])
    (target / config.BM25_METADATA_PATH.name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_run(target)


def test_sync_reads_previous_corpus_then_publishes_only_active_version(storage, monkeypatch):
    bm25_stage([])
    previous = inventory(storage)
    monkeypatch.setattr(config, "DATASET_NAME", "new")
    monkeypatch.setattr(config, "BM25_CORPUS_PATH", storage / "new_bm25_corpus.parquet")
    monkeypatch.setattr(config, "BM25_METADATA_PATH", storage / "new_bm25_index_metadata.json")
    @archived_stage("sync", ("BM25_CORPUS_PATH", "BM25_METADATA_PATH"))
    def synchronize(folder):
        assert (folder / "example_bm25_corpus.parquet").is_file()
        write_bm25(folder)
        write_json(folder / "active_dataset.json", {"dataset_name": "new"})
    result = synchronize([])
    assert not (storage / "example_bm25_corpus.parquet").exists()
    assert not (storage / "example_bm25_index_metadata.json").exists()
    assert config.BM25_CORPUS_PATH.exists()
    assert inventory(storage) == inventory(Path(result["run_dir"]))
    assert result["base_shared_inventory"] == previous
    validate_run(storage)


def test_sync_prune_failure_keeps_previous_shared_version(storage):
    bm25_stage([])
    previous = inventory(storage)
    @archived_stage("sync", ("BM25_CORPUS_PATH", "BM25_METADATA_PATH"))
    def no_active_pointer(folder):
        write_bm25(folder)
    with pytest.raises(ValueError, match="pointer"):
        no_active_pointer([])
    assert inventory(storage) == previous
