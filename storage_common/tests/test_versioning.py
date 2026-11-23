import pytest
from pipeline_storage.versioning import latest, next_name, commit_base


def test_versions_are_numeric(tmp_path):
    for name in ("sample_v1.9", "sample_v1.10", "sample_v1.2"):
        (tmp_path / f"{name}_canonical.parquet").touch()
    selected = latest(tmp_path, "_canonical.parquet")
    assert selected.name == "sample_v1.10_canonical.parquet"
    assert next_name(selected, "_canonical.parquet") == "sample_v1.11"


def test_commit_is_idempotent_but_refuses_overwrite(tmp_path):
    run = tmp_path / "run_example"
    run.mkdir()
    name = "sample_v1.4_canonical.parquet"
    source = run / name
    source.write_bytes(b"original")
    report = {"base_dir": str(tmp_path / "base_data"), "base_outputs": [name]}
    commit_base(run, report)
    commit_base(run, report)
    source.write_bytes(b"different")
    with pytest.raises(ValueError, match="different content"):
        commit_base(run, report)
