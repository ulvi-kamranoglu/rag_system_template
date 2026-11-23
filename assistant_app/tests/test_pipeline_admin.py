"""Admin auth, process isolation, durable state and safe pipeline submissions."""
import json
import os
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from filelock import FileLock
import pytest

from assistant_app import admin, pipeline_jobs as jobs, pipeline_worker as worker
from orchestration.state import save


def parquet_bytes(data=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = BytesIO()
    pq.write_table(pa.table(data or {"id": ["1"], "source_type": ["admin"], "content": ["UNEC test dataset"]}), output)
    return output.getvalue()


def upload(client, name="uploaded.parquet", content=None):
    return client.post("/admin/pipeline/join", files={"file": (name, parquet_bytes() if content is None else content, "application/octet-stream")})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    (tmp_path / "pipeline.yaml").write_text("state_dir: runs\nraw_dir: raw\n")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "new.parquet").write_bytes(b"test fixture")
    monkeypatch.setenv("ASSISTANT_ADMIN_API_KEY", "admin-secret")
    monkeypatch.setenv("ASSISTANT_API_KEY", "chat-secret")
    calls = []
    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(jobs.subprocess, "Popen", spawn)
    app = FastAPI()
    app.include_router(admin.router)
    client = TestClient(app, headers={"X-API-Key": "admin-secret"})
    yield client, tmp_path, calls
    client.close()


def test_admin_key_is_required_and_separate(setup, monkeypatch):
    client, _, calls = setup
    for key in ("", "chat-secret", "wrong"):
        assert client.post("/admin/pipeline/collect", headers={"X-API-Key": key}).status_code == 401
    monkeypatch.delenv("ASSISTANT_ADMIN_API_KEY")
    assert client.post("/admin/pipeline/collect").status_code == 503
    assert not calls


def test_collect_returns_durable_job_and_rejects_overlap(setup):
    client, root, calls = setup
    response = client.post("/admin/pipeline/collect")
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert response.json()["status"] == "running"
    assert Path(calls[0][0][0]).name == "python.exe" or Path(calls[0][0][0]).name.startswith("python")
    assert calls[0][0][-2:] == ["assistant_app.pipeline_worker", run_id]
    assert calls[0][1]["cwd"] == root
    assert client.get(f"/admin/pipeline/runs/{run_id}").json()["run_id"] == run_id
    assert client.get("/admin/pipeline/runs").json()[0]["run_id"] == run_id
    assert client.post("/admin/pipeline/collect").status_code == 409


@pytest.mark.parametrize("name", ["../outside.parquet", "..\\outside.parquet", "C:secret.parquet", "new.csv", "batch_deleted.parquet"])
def test_join_rejects_untrusted_paths(setup, name):
    client, _, calls = setup
    assert upload(client, name).status_code == 422
    assert not calls


def test_join_archives_then_replaces_raw_with_only_uploaded_file(setup):
    client, root, _ = setup
    (root / "raw" / "batch_deleted.parquet").write_bytes(b"fixture")
    content = parquet_bytes()
    response = upload(client, content=content)
    assert response.status_code == 202
    args = response.json()["arguments"]
    assert args[args.index("--input") + 1] == str((root / "raw" / "uploaded.parquet").resolve())
    assert "--deleted" not in args
    assert [p.name for p in (root / "raw").iterdir()] == ["uploaded.parquet"]
    assert (root / "raw" / "uploaded.parquet").read_bytes() == content
    assert Path(response.json()["upload_archive"]).read_bytes() == content


@pytest.mark.parametrize("content", [b"", b"not parquet"])
def test_invalid_upload_keeps_existing_raw(setup, content):
    client, root, calls = setup
    assert upload(client, content=content).status_code == 422
    assert (root / "raw" / "new.parquet").read_bytes() == b"test fixture"
    assert not calls


def test_missing_columns_keep_existing_raw(setup):
    client, root, calls = setup
    assert upload(client, content=parquet_bytes({"other": ["value"]})).status_code == 422
    assert (root / "raw" / "new.parquet").exists()
    assert not calls


def test_oversized_upload_keeps_existing_raw(setup, monkeypatch):
    client, root, calls = setup
    monkeypatch.setattr(jobs, "MAX_UPLOAD_BYTES", 1)
    assert upload(client).status_code == 413
    assert (root / "raw" / "new.parquet").exists()
    assert not calls


def test_pending_upload_does_not_replace_raw(setup):
    client, root, calls = setup
    (root / "runs").mkdir()
    save(root / "runs" / "pending.json", {"run_dir": "pending"})
    assert upload(client).status_code == 409
    assert [p.name for p in (root / "raw").iterdir()] == ["new.parquet"]
    assert not calls


def test_busy_upload_does_not_replace_raw(setup):
    client, root, _ = setup
    client.post("/admin/pipeline/collect")
    assert upload(client).status_code == 409
    assert [p.name for p in (root / "raw").iterdir()] == ["new.parquet"]


def test_join_requires_file_instead_of_old_json_body(setup):
    client, _, calls = setup
    assert client.post("/admin/pipeline/join", json={"input_filename": "new.parquet"}).status_code == 422
    assert not calls


def test_successive_archived_upload_does_not_create_redundant_legacy_backup(setup):
    from starlette.datastructures import UploadFile
    client, root, _ = setup
    (root / "runs").mkdir()
    jobs.archive_upload(UploadFile(file=BytesIO(parquet_bytes()), filename="first.parquet"))
    # The original fixture was not archived, so its safety backup must remain.
    backups = list(root.glob(".raw.legacy_*"))
    assert len(backups) == 1
    jobs.archive_upload(UploadFile(file=BytesIO(parquet_bytes()), filename="second.parquet"))
    assert list(root.glob(".raw.legacy_*")) == backups
    assert (backups[0] / "new.parquet").read_bytes() == b"test fixture"
    assert [p.name for p in (root / "raw").iterdir()] == ["second.parquet"]


def test_reset_upload_is_separate_mode_and_archived(setup):
    client, root, _ = setup
    content = parquet_bytes()
    response = client.post("/admin/pipeline/reset-data", files={"file": ("replacement.parquet", content)})
    assert response.status_code == 202
    data = response.json()
    assert data["mode"] == "reset_data"
    assert data["arguments"][data["arguments"].index("--mode") + 1] == "reset_data"
    assert data["deleted_filename"] is None
    assert [p.name for p in (root / "raw").iterdir()] == ["replacement.parquet"]
    assert Path(data["upload_archive"]).read_bytes() == content


def test_reset_requires_admin_and_refuses_pending_without_replacing_raw(setup):
    client, root, calls = setup
    assert client.post("/admin/pipeline/reset-data", headers={"X-API-Key": "chat-secret"}, files={"file": ("data.parquet", parquet_bytes())}).status_code == 401
    (root / "runs").mkdir()
    save(root / "runs" / "pending.json", {"run_dir": "pending"})
    assert client.post("/admin/pipeline/reset-data", files={"file": ("data.parquet", parquet_bytes())}).status_code == 409
    assert (root / "raw" / "new.parquet").read_bytes() == b"test fixture"
    assert not calls


def test_pending_cli_and_running_cli_block_new_requests(setup):
    client, root, calls = setup
    (root / "runs").mkdir()
    with FileLock(str(root / "runs" / ".pipeline.lock")):
        assert client.post("/admin/pipeline/collect").status_code == 409
    save(root / "runs" / "pending.json", {"run_dir": str(root / "runs" / "run_existing")})
    assert client.post("/admin/pipeline/collect").status_code == 409
    assert not calls


def test_failed_worker_resume_preserves_checkpoint(setup, monkeypatch):
    client, root, _ = setup
    first = client.post("/admin/pipeline/collect").json()
    folder = jobs.job_folder(first["run_id"])
    coordinator = root / "runs" / "run_example"
    coordinator.mkdir()
    save(coordinator / "state.json", {"run_id": "coordinator-id", "mode": "collect", "status": "running", "completed": {"collection": True}})
    save(root / "runs" / "pending.json", {"run_dir": str(coordinator)})
    def fake_main(arguments, *, on_run):
        on_run(coordinator)
        raise RuntimeError("simulated stage failure")
    monkeypatch.setattr(worker, "main", fake_main)
    assert worker.run(first["run_id"]) == 1
    failed = client.get(f"/admin/pipeline/runs/{first['run_id']}").json()
    assert failed["status"] == "failed" and failed["resumable"]
    assert failed["completed_stages"] == ["collection"]
    resumed = client.post(f"/admin/pipeline/runs/{first['run_id']}/resume")
    assert resumed.status_code == 202
    assert resumed.json()["run_id"] != first["run_id"]
    assert resumed.json()["arguments"][-2:] == ["--resume", str(coordinator)]
    assert json.loads((coordinator / "state.json").read_text())["completed"] == {"collection": True}
    assert folder.exists()


def test_worker_success_and_completed_run_cannot_resume(setup, monkeypatch):
    client, _, _ = setup
    run_id = client.post("/admin/pipeline/collect").json()["run_id"]
    monkeypatch.setattr(worker, "main", lambda *args, **kwargs: None)
    assert worker.run(run_id) == 0
    assert client.get(f"/admin/pipeline/runs/{run_id}").json()["status"] == "completed"
    assert client.post(f"/admin/pipeline/runs/{run_id}/resume").status_code == 409


def test_process_death_is_visible_without_api_memory(setup, monkeypatch):
    client, _, _ = setup
    run_id = client.post("/admin/pipeline/collect").json()["run_id"]
    monkeypatch.setattr(jobs, "alive", lambda identity: False)
    assert client.get(f"/admin/pipeline/runs/{run_id}").json()["status"] == "interrupted"
    assert client.get("/admin/pipeline/runs/not-a-run").status_code == 404


def test_logs_are_bounded(setup):
    client, _, _ = setup
    run_id = client.post("/admin/pipeline/collect").json()["run_id"]
    (jobs.job_folder(run_id) / "worker.log").write_text("x" * 70000)
    response = client.get(f"/admin/pipeline/runs/{run_id}/logs")
    assert response.status_code == 200 and len(response.content) == 65536


@pytest.fixture
def with_collection_config(setup):
    client, root, calls = setup
    config = root / "collection" / "config" / "settings.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("wordpress:\n  enabled: true\ninstagram:\n  enabled: false\n"
                      "  profiles: [uneceduaz, adalat.muradov]\nnews:\n  enabled: true\n", encoding="utf-8")
    (root / "pipeline.yaml").write_text(f"state_dir: runs\nraw_dir: raw\ncollection_config: {config.as_posix()}\n")
    return client, root, calls


def test_collect_passes_selected_sources_to_the_collection_stage(with_collection_config):
    client, _, calls = with_collection_config
    response = client.post("/admin/pipeline/collect", params=[("sources", "news"), ("sources", "wordpress/pages")])
    assert response.status_code == 202, response.text
    assert response.json()["sources"] == ["wordpress/pages", "news"]
    request = json.loads((jobs.job_folder(response.json()["run_id"]) / "request.json").read_text(encoding="utf-8"))
    assert request["arguments"][-2:] == ["--stage-option", "collection:--sources=wordpress/pages,news"]


def test_disabled_source_is_rejected_before_any_job(with_collection_config):
    client, root, calls = with_collection_config
    response = client.post("/admin/pipeline/collect", params={"sources": "instagram/uneceduaz"})
    assert response.status_code == 422 and "disabled" in response.json()["detail"]
    jobs_dir = root / "runs" / "api_jobs"
    assert not calls
    assert not jobs_dir.exists() or not any(jobs_dir.iterdir())


def test_unknown_source_is_rejected_by_swagger_choices(with_collection_config):
    client, _, calls = with_collection_config
    assert client.post("/admin/pipeline/collect", params={"sources": "twitter"}).status_code == 422
    assert not calls


def test_collect_without_selection_keeps_all_enabled_sources(with_collection_config):
    client, _, calls = with_collection_config
    response = client.post("/admin/pipeline/collect")
    assert response.status_code == 202 and response.json()["sources"] is None
    assert "--stage-option" not in json.loads((jobs.job_folder(response.json()["run_id"]) / "request.json")
                                              .read_text(encoding="utf-8"))["arguments"]


def test_sources_endpoint_lists_readiness(with_collection_config):
    client, _, _ = with_collection_config
    assert client.get("/admin/pipeline/sources").json() == [
        {"name": "wordpress/pages", "enabled": True}, {"name": "wordpress/posts", "enabled": True},
        {"name": "instagram/uneceduaz", "enabled": False}, {"name": "instagram/adalat.muradov", "enabled": False},
        {"name": "news", "enabled": True}]
