"""Transient Windows file locks (antivirus/indexer) must not fail a commit or publication."""
import json
from pathlib import Path

import pytest

from pipeline_storage import archive_run, fs, publish

ORIGINAL_RENAME = Path.rename


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(fs, "RETRY_TRANSIENT_LOCKS", True)
    monkeypatch.setattr(fs, "RETRY_DELAYS", (0,) * 5)


def locked(monkeypatch, failures, *, match=None):
    """Make the next ``failures`` matching renames raise WinError-5-style errors."""
    calls = {"failed": 0, "total": 0}

    def rename(self, target):
        calls["total"] += 1
        if (match is None or match(self, Path(target))) and calls["failed"] < failures:
            calls["failed"] += 1
            raise PermissionError(13, "Access is denied", str(self))
        return ORIGINAL_RENAME(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    return calls


def test_rename_succeeds_after_transient_lock(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    calls = locked(monkeypatch, 3)
    fs.rename_with_retry(tmp_path / "a", tmp_path / "b")
    assert (tmp_path / "b").is_dir() and not (tmp_path / "a").exists()
    assert calls["failed"] == 3


def test_persistent_lock_gives_up_with_clear_message(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    locked(monkeypatch, 100)
    with pytest.raises(PermissionError, match="still locked after 5 retries.*antivirus"):
        fs.rename_with_retry(tmp_path / "a", tmp_path / "b")
    assert (tmp_path / "a").is_dir()


def test_no_retry_when_target_exists(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    calls = locked(monkeypatch, 100)
    with pytest.raises(PermissionError):
        fs.rename_with_retry(tmp_path / "a", tmp_path / "b")
    assert calls["total"] == 1


def test_no_retry_outside_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "RETRY_TRANSIENT_LOCKS", False)
    (tmp_path / "a").mkdir()
    calls = locked(monkeypatch, 1)
    with pytest.raises(PermissionError):
        fs.rename_with_retry(tmp_path / "a", tmp_path / "b")
    assert calls["total"] == 1


def write_run(folder):
    (folder / "data.txt").write_text("payload", encoding="utf-8")
    (folder / "run_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")


def test_archive_commit_survives_transient_lock(tmp_path, monkeypatch):
    destination = tmp_path / "archive" / "run_1"
    locked(monkeypatch, 2, match=lambda source, target: source.name.endswith(".partial"))
    with archive_run(destination, lambda folder: None) as staging:
        write_run(staging)
    assert (destination / "data.txt").read_text(encoding="utf-8") == "payload"
    assert not list(destination.parent.glob(".*.partial"))


def test_publication_swap_survives_transient_lock(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    first, second = tmp_path / "run_1", tmp_path / "run_2"
    for run, text in ((first, "one"), (second, "two")):
        run.mkdir(); write_run(run); (run / "data.txt").write_text(text, encoding="utf-8")
    publish(first, shared, lambda folder: None)
    locked(monkeypatch, 2, match=lambda source, target: target == shared)
    publish(second, shared, lambda folder: None)
    assert (shared / "data.txt").read_text(encoding="utf-8") == "two"
    assert not list(tmp_path.glob(".shared.*"))  # no staging/previous/journal left behind


def test_persistent_publication_lock_restores_previous(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    first, second = tmp_path / "run_1", tmp_path / "run_2"
    for run, text in ((first, "one"), (second, "two")):
        run.mkdir(); write_run(run); (run / "data.txt").write_text(text, encoding="utf-8")
    publish(first, shared, lambda folder: None)
    locked(monkeypatch, 100, match=lambda source, target: source.name == ".shared.staging")
    with pytest.raises(PermissionError, match="still locked"):
        publish(second, shared, lambda folder: None)
    assert (shared / "data.txt").read_text(encoding="utf-8") == "one"
