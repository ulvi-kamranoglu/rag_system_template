from pathlib import Path
import polars as pl
import pytest
import yaml
from pipeline_storage import inventory
from chunking_pipeline.pipeline import run
from test_core import FakeTokenizer


@pytest.fixture
def config(tmp_path, monkeypatch):
    project = tmp_path / "project"
    path = project / "config/config.yaml"
    path.parent.mkdir(parents=True)
    for name in ("alpha", "beta"):
        pl.DataFrame([{"id": name, "source_type": "form", "title": "Where?",
                       "content": "In the library. Here is a second sentence.",
                       "metadata": "{}", "url": "https://example.org"}]).write_parquet(project / f"{name}.parquet")
    cfg = {"inputs": ["alpha.parquet", "beta.parquet"], "model_id": "offline",
           "storage": {"archive_dir": "../archive", "shared_dir": "../shared"},
           "variants": [{"max_tokens": 128, "overlap_tokens": 16},
                        {"max_tokens": 256, "overlap_tokens": 32}],
           "min_content_budget": 10, "safety_margin": 4}
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr("chunking_pipeline.pipeline.AutoTokenizer.from_pretrained", lambda model: FakeTokenizer())
    return path


def test_multiple_datasets_history_stale_removal_and_dry_run(config):
    root = config.parents[2]
    before = inventory(root)
    planned = run(str(config), dry_run=True)
    assert len(planned) == 2
    assert all(path.parent == root / "shared" for path in planned)
    assert inventory(root) == before
    outputs = run(str(config))
    assert all(path.is_file() for path in outputs)
    archive, shared = root / "archive", root / "shared"
    first = next(archive.glob("run_*"))
    original = inventory(first)
    first_chunks = pl.read_parquet(outputs[0])
    assert set(first_chunks["original_id"].to_list()) == {"alpha", "beta"}
    ids = first_chunks.filter(pl.col("original_id") == "alpha")["chunk_id"].to_list()
    # Simulate the nested layout left by an earlier version.
    (shared / "old_dataset").mkdir()
    (shared / "old_dataset/stale.parquet").write_bytes(b"old")
    # Publishing only alpha on the next run removes beta completely.
    outputs = run(str(config), "alpha.parquet")
    assert len(outputs) == 2
    assert not (shared / "beta").exists()
    assert not (shared / "reports/beta").exists()
    assert not (shared / "old_dataset").exists()
    assert (shared / "alpha_retrieval_prepared.parquet").is_file()
    assert (shared / "reports/alpha_chunking_128_summary.json").is_file()
    assert pl.read_parquet(outputs[0])["chunk_id"].to_list() == ids
    assert inventory(first) == original
    assert len(list(archive.glob("run_*"))) == 2
    assert pl.read_parquet(outputs[0])["metadata"].to_list() == ["{}"]


def test_duplicate_ids_across_datasets_do_not_replace_shared(config):
    run(str(config))
    root = config.parents[2]
    before = inventory(root / "shared")
    alpha = config.parents[1] / "alpha.parquet"
    beta = config.parents[1] / "beta.parquet"
    pl.read_parquet(alpha).write_parquet(beta)
    with pytest.raises(ValueError, match="Duplicate chunk IDs across input datasets"):
        run(str(config))
    assert inventory(root / "shared") == before
    assert len(list((root / "archive").glob("run_*"))) == 1


def test_historical_nested_archive_remains_readable(config):
    import json
    import shutil
    from chunking_pipeline.pipeline import validate_run

    run(str(config), "alpha.parquet")
    shared = config.parents[2] / "shared"
    legacy = config.parents[2] / "historical_layout"
    (legacy / "alpha").mkdir(parents=True)
    for path in shared.glob("*.parquet"):
        shutil.copy2(path, legacy / "alpha" / path.name)
    shutil.copytree(shared / "reports", legacy / "reports/alpha")
    report = json.loads((shared / "run_report.json").read_text())
    report.pop("layout")
    report.pop("dataset_name")
    for file in (legacy / "alpha").glob("alpha_*"):
        file.rename(file.with_name(file.name.removeprefix("alpha_")))
    for file in (legacy / "reports/alpha").glob("alpha_*"):
        file.rename(file.with_name(file.name.removeprefix("alpha_")))
    (legacy / "run_report.json").write_text(json.dumps(report))
    before = inventory(legacy)
    validate_run(legacy)
    assert inventory(legacy) == before


def test_failed_variant_never_replaces_shared(config, monkeypatch):
    run(str(config))
    root = config.parents[2]
    before = inventory(root / "shared")
    real = __import__("chunking_pipeline.pipeline", fromlist=["build_chunks"]).build_chunks
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("Second dataset failed")
        return real(*args, **kwargs)
    monkeypatch.setattr("chunking_pipeline.pipeline.build_chunks", fail_second)
    with pytest.raises(ValueError, match="Second dataset failed"):
        run(str(config))
    assert inventory(root / "shared") == before
    assert len(list((root / "archive").glob("run_*"))) == 1


def test_publish_failure_archive_can_be_retried(config, monkeypatch):
    run(str(config))
    root = config.parents[2]
    before = inventory(root / "shared")
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError("offline")
        patch.setattr("chunking_pipeline.pipeline.publish", fail)
        with pytest.raises(RuntimeError, match="--publish-run"):
            run(str(config))
    assert inventory(root / "shared") == before
    latest = max((root / "archive").glob("run_*"), key=lambda path: path.stat().st_mtime_ns)
    run(str(config), publish_run=str(latest))
    assert inventory(root / "shared") == inventory(latest)


def test_default_input_chain_and_default_config(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((project / "config/config.yaml").read_text())
    assert (project / cfg["input_dir"]).resolve() == project.parent / "shared_data/02_clean"
    from chunking_pipeline import cli
    calls = []
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["chunk-pipeline", "--dry-run"])
    cli.main()
    assert calls == [(None, None, True, None)]


def test_chunking_ignores_legacy_base_and_never_creates_new_baselines(config):
    import json
    root = config.parents[2]
    cfg = yaml.safe_load(config.read_text())
    # Older saved configs may still carry this switch. It no longer enables merging.
    cfg["merge_base"] = True
    config.write_text(yaml.safe_dump(cfg))
    outputs = run(str(config))
    assert set(pl.read_parquet(outputs[0])["original_id"].to_list()) == {"alpha", "beta"}
    assert not (root / "archive/base_data").exists()
    assert not (root / "shared/base_data").exists()
    legacy = root / "archive/base_data"
    legacy.mkdir()
    (legacy / "old_chunks_128.parquet").write_bytes(b"historical file, never read")
    before = inventory(legacy)
    outputs = run(str(config), "alpha.parquet")
    assert set(pl.read_parquet(outputs[0])["original_id"].to_list()) == {"alpha"}
    assert inventory(legacy) == before
    report = json.loads((root / "shared/run_report.json").read_text())
    assert "base_dir" not in report and "base_outputs" not in report
