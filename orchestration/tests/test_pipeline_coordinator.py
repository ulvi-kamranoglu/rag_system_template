import json
import pandas as pd
import pytest
import yaml
from orchestration import runner as run_pipeline
from orchestration import cli, state as checkpoint
from pipeline_storage.documents import deleted_ids, document_id

def test_checkpoint_retries_windows_lock(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text('{"old": true}')
    replace = checkpoint.os.replace
    attempts = []
    def locked_once(source, destination):
        attempts.append(source)
        if len(attempts) == 1:
            assert json.loads(path.read_text()) == {"old": True}
            raise PermissionError("temporary Windows file lock")
        replace(source, destination)
    monkeypatch.setattr(checkpoint.os, "replace", locked_once)
    monkeypatch.setattr(checkpoint.time, "sleep", lambda seconds: None)
    run_pipeline.save(path, {"new": True})
    assert len(attempts) == 2
    assert json.loads(path.read_text()) == {"new": True}


def test_checkpoint_permanent_denial_keeps_previous_state(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text('{"old": true}')
    def denied(*args):
        raise PermissionError("denied")
    monkeypatch.setattr(checkpoint.os, "replace", denied)
    monkeypatch.setattr(checkpoint.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        run_pipeline.save(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}
    assert not list(tmp_path.glob(".*.tmp"))


def test_deletions_use_chunk_identity(tmp_path):
    path = tmp_path / "deleted.parquet"
    pd.DataFrame([{"id": "post: 0007", "source_type": "posts"},
                  {"id": "77", "source_type": "page"}]).to_parquet(path)
    assert deleted_ids(path) == {"post:7", "page:77"}
    assert document_id({"id": "other:8", "source_type": "page"}) == "page:other:8"


def test_invalid_deletion_schema_stops(tmp_path):
    path = tmp_path / "deleted.parquet"
    pd.DataFrame({"title": ["missing identity"]}).to_parquet(path)
    with pytest.raises(ValueError, match="requires"):
        deleted_ids(path)


def test_plan_never_starts_a_stage(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Plan must not execute anything")
    monkeypatch.setattr(cli, "execute", forbidden)
    cli.main(["--plan"])
    assert "Qdrant/BM25" in capsys.readouterr().out
    cli.main(["--plan", "--mode", "reset_data"])
    assert "no merge" in capsys.readouterr().out


def test_reset_requires_input_and_rejects_deletion_files(tmp_path):
    config = tmp_path / "pipeline.yaml"
    config.write_text(yaml.safe_dump({"state_dir": str(tmp_path / "runs")}))
    with pytest.raises(SystemExit):
        cli.main(["--config", str(config), "--mode", "reset_data"])
    with pytest.raises(SystemExit):
        cli.main(["--config", str(config), "--mode", "reset_data", "--input", "new.parquet", "--deleted", "old.parquet"])


def test_deleted_incoming_duplicate_does_not_hide_live_baseline(tmp_path, monkeypatch):
    from canonical_pipeline import pipeline
    monkeypatch.setattr(pipeline, "detector", lambda: None)
    monkeypatch.setattr(pipeline, "detect_language", lambda *a: {"language": "az", "uncertain": False, "confidence": 1})
    archive, shared = tmp_path / "archive", tmp_path / "shared"
    base = archive / "base_data"
    base.mkdir(parents=True)
    row = dict(id="post:1", source_type="post", title="Başlıq", content="Bu kifayət qədər uzun məzmundur.", url="https://example.org/page", metadata="{}")
    from canonical_pipeline.schema import standardize
    standardize(pd.DataFrame([row])).to_parquet(base / "sample_v1.3_canonical.parquet")
    incoming = tmp_path / "incoming.parquet"
    pd.DataFrame([{**row, "id": "post:2"}]).to_parquet(incoming)
    deletion = tmp_path / "deleted.parquet"
    pd.DataFrame({"document_id": ["post:2"]}).to_parquet(deletion)
    result = pipeline.process([incoming], archive, shared, True, merge_base=True, deleted_path=deletion)
    assert result["base_output_rows"] == 1
    assert result["output_rows"] == 0
    assert result["deleted_rows"] == 1
    assert result["base_output_file"] == "sample_v1.4_canonical.parquet"


