"""Detached execution of a single durable admin request."""
import sys
import traceback

from orchestration.cli import main
from orchestration.state import save
from .pipeline_jobs import job_folder, read, now


def run(run_id):
    folder = job_folder(run_id)
    request = read(folder / "request.json")
    def on_run(pipeline_folder):
        save(folder / "pipeline.json", {"folder": str(pipeline_folder)})
    try:
        main(request["arguments"], on_run=on_run)
    except BaseException as exc:
        traceback.print_exc()
        save(folder / "result.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": now()})
        return 1
    save(folder / "result.json", {"status": "completed", "finished_at": now()})
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
