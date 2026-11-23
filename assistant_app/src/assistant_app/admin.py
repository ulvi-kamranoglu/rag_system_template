"""Privileged, asynchronous pipeline administration."""
from pathlib import Path
from typing import Literal
from enum import Enum

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from .security import require_admin_key
from . import pipeline_jobs as jobs

router = APIRouter(prefix="/admin/pipeline", tags=["Pipeline administration"], dependencies=[Depends(require_admin_key)])


def submit(*args, **kwargs):
    try:
        return jobs.submit(*args, **kwargs)
    except jobs.Conflict as exc:
        raise HTTPException(409, str(exc))
    except jobs.UploadTooLarge as exc:
        raise HTTPException(413, str(exc))
    except FileNotFoundError:
        raise HTTPException(404, "Run not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


def _source_names():
    """Swagger choices come from the collection config at startup; submission revalidates."""
    try:
        names = list(jobs.collection_sources())
    except Exception:
        names = []
    return names or ["wordpress/pages", "wordpress/posts", "instagram/uneceduaz", "instagram/adalat.muradov", "news"]


CollectionSource = Enum("CollectionSource", {name: name for name in _source_names()}, type=str)


@router.post("/collect", status_code=202)
def collect(sources: list[CollectionSource] | None = Query(
        None, description="Optional subset of sources to collect. Empty: every source enabled in the "
                          "collection config. Selecting a disabled source returns 422.")):
    return submit("collect", sources=[source.value for source in sources] if sources else None)


@router.get("/sources")
def sources():
    """Collection sources known to the configuration and whether each is enabled."""
    try:
        return [{"name": name, "enabled": ready} for name, ready in jobs.collection_sources().items()]
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.post("/join", status_code=202)
def join(file: UploadFile = File(description="Raw Parquet dataset; replaces the shared 01_raw contents after archival")):
    try:
        return submit("join", upload=file)
    finally:
        file.file.close()


@router.post("/reset-data", status_code=202, summary="Replace the active dataset with an uploaded dataset")
def reset_data(file: UploadFile = File(description="Complete replacement Parquet dataset. Old records will not be retained; the canonical version increments.")):
    try:
        return submit("reset_data", upload=file)
    finally:
        file.file.close()


@router.get("/runs")
def runs():
    return jobs.list_jobs()


@router.get("/runs/{run_id}")
def status(run_id: str):
    try:
        return jobs.describe(jobs.job_folder(run_id))
    except FileNotFoundError:
        raise HTTPException(404, "Run not found")


@router.post("/runs/{run_id}/resume", status_code=202)
def resume(run_id: str):
    try:
        previous = jobs.describe(jobs.job_folder(run_id))
    except FileNotFoundError:
        raise HTTPException(404, "Run not found")
    return submit(previous["mode"], resume_id=run_id)


@router.get("/runs/{run_id}/logs", response_class=PlainTextResponse)
def logs(run_id: str, stage: Literal["worker", "collection", "canonical", "chunk", "embedding", "retrieval"] = "worker"):
    try:
        folder = jobs.job_folder(run_id)
    except FileNotFoundError:
        raise HTTPException(404, "Run not found")
    path = folder / "worker.log"
    if stage != "worker":
        if not (folder / "pipeline.json").exists():
            raise HTTPException(404, "Stage has not started")
        path = Path(jobs.read(folder / "pipeline.json")["folder"]) / f"{stage}.log"
    if not path.is_file():
        raise HTTPException(404, "Log not available yet")
    # A bounded tail; full per-stage logs stay on the server.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        return stream.read().decode("utf-8", errors="replace")
