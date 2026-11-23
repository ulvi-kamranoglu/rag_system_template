"""Disk-backed admin jobs; inference and pipeline execution stay outside the API."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

from filelock import FileLock, Timeout
import psutil
import yaml

from .config import ROOT
from orchestration.state import save
from pipeline_storage import archive_run, publish, run_path


class Conflict(Exception):
    pass


class UploadTooLarge(Exception):
    pass


MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def archive_upload(upload):
    """Caller holds the submission lock; coordinator lock protects raw replacement."""
    name = upload.filename or ""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\.parquet", name) or name.endswith("_deleted.parquet"):
        raise ValueError("Use a plain .parquet filename (letters, numbers, dots, underscores or hyphens); deletion files are not join inputs")
    _, state_root, raw = configuration()
    # Uploads have a distinct archive namespace, separate from collector run reports.
    target = run_path(ROOT / "archive_data" / "01_raw" / "uploads")
    def validate(folder):
        import pyarrow.parquet as pq
        try:
            with (folder / name).open("rb") as source:
                with pq.ParquetFile(source) as parquet:
                    missing = {"id", "source_type", "content"} - set(parquet.schema_arrow.names)
                    if missing:
                        raise ValueError(f"Missing required columns: {sorted(missing)}")
                    if parquet.metadata.num_rows == 0:
                        raise ValueError("The uploaded dataset must contain at least one row")
                    for _ in parquet.iter_batches(batch_size=1024):
                        pass  # Decode row groups before replacing the current input.
        except Exception as exc:
            raise ValueError(f"Invalid Parquet dataset: {exc}") from exc
    try:
        with FileLock(str(state_root / ".pipeline.lock"), timeout=0):
            if (state_root / "pending.json").exists():
                raise Conflict("A pipeline run is pending; raw inputs were not changed")
            with archive_run(target, validate) as staging:
                size = 0
                upload.file.seek(0)
                with (staging / name).open("xb") as output:
                    while chunk := upload.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise UploadTooLarge("Upload exceeds the 512 MiB limit")
                        output.write(chunk)
            publish(target, raw, validate, migrate_legacy=True,
                    previous_archive_root=target.parent)
    except Timeout:
        raise Conflict("The pipeline or raw publication is busy; retry after it finishes")
    return name, str(target / name)


def now():
    return datetime.now(timezone.utc).isoformat()


def configuration():
    path = ROOT / "pipeline.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    def resolve(value):
        path = Path(value)
        return (path if path.is_absolute() else ROOT / path).resolve()
    return path, resolve(config["state_dir"]), resolve(config["raw_dir"])


def collection_config():
    """The collection stage configuration named by pipeline.yaml, or None."""
    path = ROOT / "pipeline.yaml"
    value = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("collection_config")
    if not value:
        return None
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def collection_sources():
    """{name: enabled} from the collection stage's own public catalog; {} if unavailable."""
    path = collection_config()
    if path is None or not path.is_file():
        return {}
    from wp_tracker.collection import catalog
    return catalog(path)


def select_sources(names):
    """Validate a requested subset with the collection stage; raises ValueError (HTTP 422)."""
    path = collection_config()
    if path is None or not path.is_file():
        raise ValueError("Collection configuration is unavailable; source selection is not possible")
    from wp_tracker.collection import select
    return select(path, names)


def jobs_root():
    return configuration()[1] / "api_jobs"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def job_folder(run_id):
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise FileNotFoundError("Unknown run")
    folder = jobs_root() / run_id
    if not (folder / "request.json").is_file():
        raise FileNotFoundError("Unknown run")
    return folder


def alive(identity):
    try:
        process = psutil.Process(identity["pid"])
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE and abs(process.create_time() - identity["created"]) < 0.01
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return False


def describe(folder):
    request = read(folder / "request.json")
    result_path = folder / "result.json"
    if result_path.exists():
        result = read(result_path)
    elif (folder / "process.json").exists():
        result = {"status": "running" if alive(read(folder / "process.json")) else "interrupted"}
    else:
        # A crashed submitter must not leave a permanent queued job.
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(request["submitted_at"])).total_seconds()
        result = {"status": "queued" if age < 30 else "interrupted"}
    data = {**request, **result}
    link = folder / "pipeline.json"
    if link.exists():
        pipeline_folder = Path(read(link)["folder"])
        pipeline = read(pipeline_folder / "state.json")
        data.update(pipeline_run_id=pipeline["run_id"], pipeline_status=pipeline["status"],
                    completed_stages=list(pipeline.get("completed", {})),
                    current_stage=pipeline.get("current_stage"),
                    dataset_name=pipeline.get("dataset_name"), resumable=False)
        pending = configuration()[1] / "pending.json"
        data["resumable"] = data["status"] in {"failed", "interrupted"} and pending.exists() and Path(read(pending)["run_dir"]).resolve() == pipeline_folder.resolve()
    else:
        data.update(pipeline_run_id=None, completed_stages=[], current_stage=None, resumable=False)
    return data


def list_jobs():
    root = jobs_root()
    folders = sorted(root.glob("*/request.json"), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
    return [describe(path.parent) for path in folders]


def raw_file(name, *, deletion=False):
    root = configuration()[2]
    if not name or Path(name).name != name or "/" in name or "\\" in name or ":" in name or not name.endswith(".parquet"):
        raise ValueError("Provide a .parquet filename directly inside the configured raw directory")
    path = (root / name).resolve()
    if path.parent != root or not path.is_file() or (not deletion and name.endswith("_deleted.parquet")):
        raise ValueError("Raw file does not exist or is not an allowed input")
    return path


def submit(mode, input_name=None, deleted_name=None, resume_id=None, upload=None, sources=None):
    config_path, state_root, _ = configuration()
    if sources and mode != "collect":
        raise ValueError("Source selection applies to collect only")
    # Validate before any job exists: a rejected selection must never leave a pending run.
    sources = select_sources(sources) if sources else None
    root = jobs_root()
    root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(root / ".submit.lock"), timeout=0):
            if any(job["status"] in {"queued", "running"} for job in list_jobs()):
                raise Conflict("A pipeline job is already running")
            # The coordinator remains authoritative against CLI/scheduler races.
            try:
                with FileLock(str(state_root / ".pipeline.lock"), timeout=0):
                    pass
            except Timeout:
                raise Conflict("The pipeline is running from another entrypoint")
            pending = state_root / "pending.json"
            args = ["--config", str(config_path)]
            resume_folder = None
            upload_archive = None
            if resume_id:
                previous = job_folder(resume_id)
                if not describe(previous)["resumable"]:
                    raise Conflict("This job cannot be resumed; inspect its status and logs")
                resume_folder = Path(read(previous / "pipeline.json")["folder"]).resolve()
                if resume_folder.parent != state_root:
                    raise Conflict("Invalid pipeline run directory")
                args += ["--resume", str(resume_folder)]
            else:
                if pending.exists():
                    raise Conflict("A pipeline run is pending; resume it before starting a new batch")
                args += ["--mode", mode]
                if sources:
                    args += ["--stage-option", "collection:--sources=" + ",".join(sources)]
                if mode in {"join", "reset_data"}:
                    if mode == "reset_data" and deleted_name:
                        raise ValueError("Reset data does not accept a deletion file")
                    if upload is not None:
                        input_name, upload_archive = archive_upload(upload)
                    args += ["--input", str(raw_file(input_name))]
                    if deleted_name:
                        args += ["--deleted", str(raw_file(deleted_name, deletion=True))]
            run_id = uuid4().hex
            folder = root / run_id
            folder.mkdir()
            save(folder / "request.json", {"run_id": run_id, "mode": mode, "submitted_at": now(),
                                           "input_filename": input_name, "deleted_filename": deleted_name,
                                           "upload_archive": upload_archive, "sources": sources,
                                           "resumed_from": resume_id, "arguments": args})
            if resume_folder:
                save(folder / "pipeline.json", {"folder": str(resume_folder)})
            options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            try:
                with (folder / "worker.log").open("ab") as output:
                    process = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "assistant_app.pipeline_worker", run_id],
                                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                               close_fds=True, **options)
                try:
                    save(folder / "process.json", {"pid": process.pid, "created": psutil.Process(process.pid).create_time()})
                except psutil.NoSuchProcess:
                    # A fast worker may already have written its result.
                    if not (folder / "result.json").exists():
                        save(folder / "result.json", {"status": "failed", "error": "Worker exited before startup", "finished_at": now()})
            except Exception as exc:
                save(folder / "result.json", {"status": "failed", "error": str(exc), "finished_at": now()})
                raise
            return describe(folder)
    except Timeout:
        raise Conflict("Another pipeline submission is in progress")
