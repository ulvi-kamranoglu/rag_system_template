import json
from pathlib import Path
import pandas as pd
import pytest
import yaml
from pipeline_storage import inventory, storage_paths
from canonical_pipeline.pipeline import process
from test_pipeline import sample


def test_history_stale_removal_and_failed_processing(tmp_path):
    source = tmp_path / "input.parquet"
    sample("1").to_parquet(source)
    archive, shared = tmp_path / "archive", tmp_path / "shared"
    source_bytes = source.read_bytes()
    first = process([source], archive, shared)
    first_files = inventory(Path(first["run_dir"]))
    (shared / "obsolete").write_text("old artifact")
    second = process([source], archive, shared)
    assert inventory(shared) == inventory(Path(second["run_dir"]))
    assert inventory(Path(first["run_dir"])) == first_files
    assert source.read_bytes() == source_bytes
    assert not (shared / "obsolete").exists()
    before = inventory(shared)
    sample("2").assign(content="").to_parquet(source)
    with pytest.raises(ValueError, match="All records"):
        process([source], archive, shared)
    assert inventory(shared) == before
    assert len(list(archive.glob("run_*"))) == 2


def test_dry_run_does_not_create_destinations(tmp_path):
    source = tmp_path / "input.parquet"
    sample("1").to_parquet(source)
    before = inventory(tmp_path)
    report = process([source], tmp_path / "archive", tmp_path / "shared", True)
    assert report["run_dir"].startswith(str(tmp_path / "archive"))
    assert report["output"] == str(tmp_path / "shared/input_canonical.parquet")
    assert inventory(tmp_path) == before


def test_publish_failure_leaves_retryable_complete_archive(tmp_path, monkeypatch):
    source = tmp_path / "input.parquet"
    sample("1").to_parquet(source)
    archive, shared = tmp_path / "archive", tmp_path / "shared"
    process([source], archive, shared)
    before = inventory(shared)
    def fail(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr("canonical_pipeline.pipeline.publish", fail)
    with pytest.raises(RuntimeError, match="--publish-run"):
        process([source], archive, shared)
    assert inventory(shared) == before
    assert len(list(archive.glob("run_*"))) == 2


def test_default_input_chain_and_cwd_independence(tmp_path, monkeypatch):
    from canonical_pipeline.cli import main
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / "config/config.yaml").read_text())
    assert (project / config["input_dir"]).resolve() == project.parent / "shared_data/01_raw"
    selected = tmp_path / "chatbot_dataset_v1.2.parquet"
    monkeypatch.setattr("canonical_pipeline.cli.select_input", lambda *a, **kw: selected)
    archive, shared = storage_paths(project, config, "02_clean")
    calls = []
    def fake_process(inputs, archive_dir, shared_dir, dry_run, **kwargs):
        calls.append((inputs, archive_dir, shared_dir, dry_run))
        return {"dry_run": dry_run}
    monkeypatch.setattr("canonical_pipeline.cli.process", fake_process)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["canonical-pipeline", "--dry-run"])
    main()
    assert calls[0][1:] == (archive, shared, True)


@pytest.fixture
def batch_setup(tmp_path, monkeypatch):
    from canonical_pipeline import pipeline
    monkeypatch.setattr(pipeline, "detector", lambda: None)
    monkeypatch.setattr(pipeline, "detect_language", lambda *a: {"language": "az", "uncertain": False})
    archive, shared = tmp_path / "archive", tmp_path / "shared"
    seed = tmp_path / "sample_v1.5.parquet"
    sample("1").assign(content="Old distinct content retained for joins").to_parquet(seed)
    initial = process([seed], archive, shared)
    incoming = tmp_path / "example_filename.parquet"
    sample("2").assign(content="<p>New distinct content from upload only</p>").to_parquet(incoming)
    return pipeline, archive, shared, incoming, initial


@pytest.mark.parametrize("reset", [False, True])
def test_shared_and_run_contain_batch_while_base_contains_version(batch_setup, reset):
    pipeline, archive, shared, incoming, initial = batch_setup
    result = process([incoming], archive, shared, merge_base=not reset, reset_data=reset)
    assert result["output_file"] == "example_filename_canonical.parquet"
    assert result["base_output_file"] == "sample_v1.6_canonical.parquet"
    batch = shared / result["output_file"]
    assert set(pd.read_parquet(batch)["id"]) == {"page:2"}
    assert "<p>" not in pd.read_parquet(batch).iloc[0]["content"]
    assert batch.read_bytes() == (Path(result["run_dir"]) / result["output_file"]).read_bytes()
    full = archive / "base_data" / result["base_output_file"]
    assert set(pd.read_parquet(full)["id"]) == ({"page:2"} if reset else {"page:1", "page:2"})
    assert full.read_bytes() == (Path(result["run_dir"]) / "base_snapshot" / result["base_output_file"]).read_bytes()
    assert not (shared / "base_snapshot").exists()
    assert not list(shared.glob("sample_v*_canonical.parquet"))
    assert Path(initial["run_dir"]).is_dir()
    sample("3").assign(content="Third different document for next batch").to_parquet(incoming)
    following = process([incoming], archive, shared, merge_base=True)
    assert following["base_output_file"] == "sample_v1.7_canonical.parquet"
    assert set(pd.read_parquet(shared / following["output_file"])["id"]) == {"page:3"}
    expected = {"page:2", "page:3"} if reset else {"page:1", "page:2", "page:3"}
    assert set(pd.read_parquet(archive / "base_data" / following["base_output_file"])["id"]) == expected
    with pytest.raises(ValueError, match="older"):
        pipeline.publish_canonical(Path(result["run_dir"]), shared)


@pytest.mark.parametrize("reset", [False, True])
def test_batch_publication_retry_preserves_base_version(batch_setup, monkeypatch, reset):
    pipeline, archive, shared, incoming, _ = batch_setup
    before = inventory(shared)
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "publish", lambda *a, **kw: (_ for _ in ()).throw(OSError("offline")))
        with pytest.raises(RuntimeError, match="--publish-run"):
            process([incoming], archive, shared, merge_base=not reset, reset_data=reset)
    assert inventory(shared) == before
    target = next(path for path in archive.glob("run_*") if (path / "base_snapshot/sample_v1.6_canonical.parquet").exists())
    pipeline.publish_canonical(target, shared)
    pipeline.publish_canonical(target, shared)
    assert (shared / "example_filename_canonical.parquet").exists()
    assert len(list(archive.glob("run_*"))) == 2
    assert not (archive / "base_data/sample_v1.7_canonical.parquet").exists()


def test_reset_rejects_bad_data_and_combined_flags(batch_setup):
    _, archive, shared, incoming, _ = batch_setup
    before = inventory(shared)
    sample("2").assign(content="").to_parquet(incoming)
    with pytest.raises(ValueError, match="accepted record"):
        process([incoming], archive, shared, reset_data=True)
    assert inventory(shared) == before
    assert not (archive / "base_data/sample_v1.6_canonical.parquet").exists()
    with pytest.raises(ValueError, match="cannot be combined"):
        process([incoming], archive, shared, reset_data=True, merge_base=True)


def test_collect_deletion_only_publishes_empty_batch_and_updates_base(batch_setup, tmp_path):
    _, archive, shared, incoming, _ = batch_setup
    sample("2").head(0).to_parquet(incoming)
    deletion = tmp_path / "deleted.parquet"
    pd.DataFrame({"document_id": ["page:1"]}).to_parquet(deletion)
    result = process([incoming], archive, shared, merge_base=True, deleted_path=deletion)
    assert result["output_rows"] == 0 and result["base_output_rows"] == 0
    assert pd.read_parquet(shared / result["output_file"]).empty
    assert pd.read_parquet(archive / "base_data" / result["base_output_file"]).empty
