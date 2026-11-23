import json
from pathlib import Path
import pytest
from pipeline_storage import (archive_run, inventory, publish, recover_publication,
                              run_path, storage_paths, write_json)


def validate(folder):
    json.loads((folder / "run_report.json").read_text())
    if not (folder / "dataset.parquet").read_bytes():
        raise ValueError("Empty output")


def make_run(root, content=b"complete"):
    target = run_path(root)
    with archive_run(target, validate) as staging:
        (staging / "dataset.parquet").write_bytes(content)
        write_json(staging / "run_report.json", {"run": target.name})
    return target


def test_mapping_and_root_validation(tmp_path):
    for stage in ("01_raw", "02_clean", "03_chunks"):
        assert storage_paths(tmp_path / "project", {}, stage) == (
            tmp_path / "archive_data" / stage, tmp_path / "shared_data" / stage)
    with pytest.raises(ValueError, match="centralized"):
        storage_paths(tmp_path, {"storage": {"archive_dir": "data/output"}}, "01_raw")
    with pytest.raises(ValueError, match="overlap"):
        storage_paths(tmp_path / "project", {"storage": {"archive_dir": "../same", "shared_dir": "../same/sub"}}, "01_raw")


def test_history_latest_only_stale_removal_and_idempotence(tmp_path):
    archive, shared = tmp_path / "archive", tmp_path / "shared"
    first = make_run(archive, b"first")
    before = inventory(first)
    publish(first, shared, validate)
    (shared / "stale.txt").write_text("old artifact")
    second = make_run(archive, b"second")
    assert first != second
    publish(second, shared, validate)
    assert inventory(shared) == inventory(second)
    assert not (shared / "stale.txt").exists()
    assert not [p for p in shared.glob("run_*") if p.is_dir()]
    assert inventory(first) == before
    assert publish(second, shared, validate)["status"] == "already_synced"


@pytest.mark.parametrize("failure", ["copy", "corrupt", "rename", "journal"])
def test_failed_publication_restores_previous_complete_run(tmp_path, monkeypatch, failure):
    import pipeline_storage as storage
    first = make_run(tmp_path / "archive", b"first")
    second = make_run(tmp_path / "archive", b"second")
    shared = tmp_path / "shared"
    publish(first, shared, validate)
    old = inventory(shared)
    with monkeypatch.context() as patch:
        if failure == "copy":
            def fail(*args, **kwargs):
                raise OSError("copy failure")
            patch.setattr(storage.shutil, "copytree", fail)
        elif failure == "corrupt":
            real = storage.shutil.copytree
            def corrupt(source, target, *args, **kwargs):
                real(source, target, *args, **kwargs)
                (target / "dataset.parquet").write_bytes(b"corrupt")
            patch.setattr(storage.shutil, "copytree", corrupt)
        elif failure == "rename":
            real = Path.rename
            def fail(source, target):
                if source.name.endswith(".staging"):
                    raise OSError("rename failure")
                return real(source, target)
            patch.setattr(Path, "rename", fail)
        else:
            real = storage._journal
            def fail(path, payload):
                if payload["phase"] == "committed":
                    raise OSError("journal failure")
                return real(path, payload)
            patch.setattr(storage, "_journal", fail)
        with pytest.raises((OSError, RuntimeError)):
            publish(second, shared, validate)
    assert inventory(shared) == old
    assert second.is_dir()
    publish(second, shared, validate)
    assert inventory(shared) == inventory(second)


def test_interrupted_swap_is_recovered(tmp_path):
    import pipeline_storage as storage
    first = make_run(tmp_path / "archive", b"first")
    shared = tmp_path / "shared"
    publish(first, shared, validate)
    journal, staging, previous = storage._paths(shared)
    storage._journal(journal, {"phase": "prepared", "existed": True, "preserve_previous": False})
    shared.rename(previous)
    shared.mkdir()
    (shared / "incomplete").write_text("bad")
    recover_publication(shared)
    assert inventory(shared) == inventory(first)
    assert not journal.exists()


def test_legacy_contents_are_retained(tmp_path):
    shared = tmp_path / "shared"
    (shared / "run_legacy").mkdir(parents=True)
    (shared / "run_legacy/original").write_bytes(b"historical")
    run = make_run(tmp_path / "archive")
    with pytest.raises(RuntimeError, match="--migrate-legacy-shared"):
        publish(run, shared, validate)
    assert (shared / "run_legacy/original").read_bytes() == b"historical"
    publish(run, shared, validate, migrate_legacy=True)
    backups = list(tmp_path.glob(".shared.legacy_*"))
    assert len(backups) == 1
    assert (backups[0] / "run_legacy/original").read_bytes() == b"historical"
    assert inventory(shared) == inventory(run)


def test_failed_archive_never_gets_final_name(tmp_path):
    target = run_path(tmp_path / "archive")
    with pytest.raises(FileNotFoundError):
        with archive_run(target, validate) as staging:
            (staging / "partial").write_text("bad")
    assert not target.exists()
    assert not list(target.parent.iterdir())


@pytest.mark.parametrize("archived", [True, False])
def test_raw_backup_is_skipped_only_for_exact_archived_contents(tmp_path, archived):
    shared = tmp_path / "01_raw"
    shared.mkdir()
    (shared / "original.parquet").write_bytes(b"original")
    uploads = tmp_path / "uploads"
    old = uploads / "run_old"
    old.mkdir(parents=True)
    (old / "original.parquet").write_bytes(b"original" if archived else b"different")
    new = uploads / "run_new"
    new.mkdir()
    (new / "next.parquet").write_bytes(b"next")
    validate_raw = lambda folder: (folder / "next.parquet").read_bytes()
    publish(new, shared, validate_raw, previous_archive_root=uploads)
    backups = list(tmp_path.glob(".01_raw.legacy_*"))
    assert len(backups) == (0 if archived else 1)
    if backups:
        assert (backups[0] / "original.parquet").read_bytes() == b"original"
    assert inventory(shared) == inventory(new)
    assert (old / "original.parquet").exists()


def test_verified_raw_archive_does_not_disable_publication_rollback(tmp_path, monkeypatch):
    import pipeline_storage as storage
    shared = tmp_path / "01_raw"
    shared.mkdir()
    (shared / "original.parquet").write_bytes(b"original")
    uploads = tmp_path / "uploads"
    old = uploads / "run_old"
    old.mkdir(parents=True)
    (old / "original.parquet").write_bytes(b"original")
    new = uploads / "run_new"
    new.mkdir()
    (new / "next.parquet").write_bytes(b"next")
    rename = Path.rename
    def fail_staging(path, target):
        if path.name.endswith(".staging"):
            raise OSError("interrupted swap")
        return rename(path, target)
    monkeypatch.setattr(Path, "rename", fail_staging)
    with pytest.raises(OSError, match="interrupted"):
        publish(new, shared, lambda folder: None, previous_archive_root=uploads)
    assert inventory(shared) == inventory(old)


def test_matching_archive_rejects_extra_files_and_incomplete_runs(tmp_path):
    from pipeline_storage import matching_archive
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dataset.parquet").write_bytes(b"original")
    archive = tmp_path / "archive"
    partial = archive / ".run_1.partial"
    partial.mkdir(parents=True)
    (partial / "dataset.parquet").write_bytes(b"original")
    assert matching_archive(raw, archive) is None
    partial.rename(archive / "run_1")
    assert matching_archive(raw, archive) == archive / "run_1"
    (raw / "unarchived.txt").write_text("retain this too")
    assert matching_archive(raw, archive) is None
