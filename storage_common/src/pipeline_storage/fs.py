"""Filesystem helpers that tolerate transient Windows file locks."""
import os
from pathlib import Path
import time

# On Windows, antivirus real-time scanning or the search indexer briefly opens
# freshly written files. Renaming a directory that contains such a file fails
# with "Access is denied" (WinError 5) or a sharing violation (WinError 32)
# although nothing is wrong; the same rename succeeds moments later.
RETRY_TRANSIENT_LOCKS = os.name == "nt"
RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.5, 2.0, 2.0, 2.0, 3.0, 3.0)  # about 15 seconds


def rename_with_retry(source: Path, target: Path, *, delays=None) -> Path:
    """Rename ``source`` to ``target``, retrying only transient Windows locks.

    A retry happens only while the source still exists and the target does not,
    so a rename is never repeated after it took effect and never overwrites.
    """
    source, target = Path(source), Path(target)
    schedule = RETRY_DELAYS if delays is None else delays
    attempt = 0
    while True:
        try:
            return source.rename(target)
        except PermissionError as exc:
            if (not RETRY_TRANSIENT_LOCKS or attempt >= len(schedule)
                    or not source.exists() or target.exists()):
                if RETRY_TRANSIENT_LOCKS and attempt:
                    raise PermissionError(
                        exc.errno, f"{exc.strerror}; still locked after {attempt} retries "
                        f"(about {sum(schedule[:attempt]):.0f}s). Another process, usually "
                        "antivirus scanning or the search indexer, keeps a file open. "
                        "Exclude the data folders from scanning or retry the run.",
                        str(source), None, str(target)) from exc
                raise
            time.sleep(schedule[attempt])
            attempt += 1
