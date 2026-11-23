"""Collector publication integration; generic swap faults live in storage_common."""
from pathlib import Path
import json
import pytest
from pipeline_storage import inventory
from wp_tracker import mirror
from wp_tracker.config import load_settings
from wp_tracker.pipeline import scan
from wp_tracker.storage import commit


class FakeClient:
    def fetch_all(self, source):
        return []


def test_scan_publishes_latest_only_and_dry_run_is_read_only(settings):
    commit(settings, {settings.baseline_file: ("snapshot", [])})
    before = inventory(settings.project_root)
    dry = scan(settings, client=FakeClient())
    assert str(settings.output_dir) in dry["run_dir"]
    assert dry["shared_dir"] == str(settings.shared_raw_dir)
    assert inventory(settings.project_root) == before
    first = scan(settings, run=True, client=FakeClient())
    assert inventory(settings.shared_raw_dir) == inventory(Path(first["run_dir"]))
    (settings.shared_raw_dir / "stale").write_text("old")
    second = scan(settings, run=True, client=FakeClient())
    assert first["run_dir"] != second["run_dir"]
    assert inventory(settings.shared_raw_dir) == inventory(Path(second["run_dir"]))
    assert Path(first["run_dir"]).is_dir()
    assert not [p for p in settings.shared_raw_dir.glob("run_*") if p.is_dir()]
    assert mirror.sync_shared(settings)["already_synced"] == [str(settings.shared_raw_dir.resolve())]


def test_failed_delivery_retains_archive_and_blocks_next_scan(settings, monkeypatch):
    commit(settings, {settings.baseline_file: ("snapshot", [])})
    def fail(*args, **kwargs):
        raise OSError("Shared disk unavailable")
    with monkeypatch.context() as patch:
        patch.setattr("wp_tracker.pipeline.mirror_run", fail)
        with pytest.raises(RuntimeError, match="Local run committed.*shared mirror failed"):
            scan(settings, run=True, client=FakeClient())
        report = json.loads((settings.reports_dir / "latest.json").read_text())
        archive = Path(report["run_dir"])
        assert archive.is_dir()
        snapshot = settings.snapshot_file.read_bytes()
        with pytest.raises(OSError, match="Shared disk"):
            scan(settings, run=True, client=FakeClient())
        assert settings.snapshot_file.read_bytes() == snapshot
        assert len(list(settings.output_dir.glob("run_*"))) == 1
    assert mirror.sync_shared(settings)["synced"]
    assert inventory(settings.shared_raw_dir) == inventory(archive)


def test_pending_state_blocks_sync(settings):
    settings.output_dir.mkdir(parents=True)
    (settings.state_dir / ".pending").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Pending"):
        mirror.sync_shared(settings)
    assert not settings.shared_raw_dir.exists()


def test_source_sync_offline_and_failure_result(settings, monkeypatch):
    commit(settings, {settings.baseline_file: ("snapshot", [])})
    scan(settings, run=True, client=FakeClient())
    def forbid(*args, **kwargs):
        raise AssertionError("No WordPress calls during retry")
    monkeypatch.setattr("wp_tracker.api.WordPressClient.__init__", forbid)
    assert mirror.sync_shared(settings)["already_synced"]
    settings.output_file("deleted").unlink()
    assert mirror.sync_shared(settings)["failed"]


def test_shared_config_default_and_override(tmp_path):
    root = tmp_path / "project"
    config = root / "config/settings.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    settings = load_settings(config)
    assert settings.shared_raw_dir == tmp_path / "shared_data/01_raw"
    assert settings.output_dir == tmp_path / "archive_data/01_raw"
    assert settings.reports_dir == root / "data/state"
    config.write_text("storage:\n  shared_dir: ../custom/raw\n  archive_dir: ../custom/archive\n")
    assert load_settings(config).shared_raw_dir == tmp_path / "custom/raw"
