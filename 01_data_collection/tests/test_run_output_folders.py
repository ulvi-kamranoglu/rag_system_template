"""Every successful real scan gets an independent, recoverable output directory."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from wp_tracker.pipeline import bootstrap, scan, status
from wp_tracker.records import dataset, from_api
from wp_tracker.storage import commit, read_rows


@pytest.fixture
def offline_storage(monkeypatch):
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda frame, path, **kw: frame.to_pickle(path))
    monkeypatch.setattr(pd, "read_parquet", lambda path, **kw: pd.read_pickle(path))


class FakeClient:
    def __init__(self, pages, posts):
        self.by_source = {"pages": pages, "posts": posts}

    def fetch_all(self, source):
        return self.by_source[source]


def item(number, content="Meaningful published content"):
    return {
        "id": number,
        "title": {"rendered": f"Title {number}"},
        "content": {"rendered": f"<p>{content}</p>"},
        "link": f"https://unec.edu.az/{number}/",
    }


def prepare_empty_baseline(settings):
    commit(settings, {
        settings.baseline_file: ("snapshot", []),
        settings.snapshot_file: ("snapshot", []),
    })


def test_every_scan_creates_own_folder_and_preserves_prior_results(settings, offline_storage):
    prepare_empty_baseline(settings)
    first = scan(settings, run=True, client=FakeClient([item(1)], []))
    # Use the actual returned run path, not platform-dependent string slicing.
    from pathlib import Path
    first_dir = Path(first["run_dir"])
    assert first_dir.parent == settings.output_dir
    assert first_dir.name.startswith("run_")
    assert {entry.name for entry in first_dir.iterdir()} == {
        "chatbot_dataset_new.parquet", "chatbot_dataset_deleted.parquet", "latest.json", "run_report.json"
    }
    assert [row["id"] for row in read_rows(first_dir / "chatbot_dataset_new.parquet")] == ["page:1"]

    # A dry-run never creates a new output directory and does not rewrite the report.
    dry = scan(settings, client=FakeClient([item(1), item(2)], []))
    assert dry["run_dir"].startswith(str(settings.output_dir))
    assert all(str(settings.output_dir) in path for path in dry["outputs"].values())
    assert len(list(settings.output_dir.glob("run_*"))) == 1

    second = scan(settings, run=True, client=FakeClient([item(1), item(2)], []))
    second_dir = Path(second["run_dir"])
    assert second_dir != first_dir
    assert [row["id"] for row in read_rows(first_dir / "chatbot_dataset_new.parquet")] == ["page:1"]
    assert [row["id"] for row in read_rows(second_dir / "chatbot_dataset_new.parquet")] == ["page:1", "page:2"]
    assert status(settings)["latest_run_dir"] == str(second_dir)
    assert settings.output_file("new") == second_dir / "chatbot_dataset_new.parquet"
    assert json.loads((settings.reports_dir / "latest.json").read_text(encoding="utf-8"))["run_dir"] == str(second_dir)


def test_empty_changes_still_write_both_files(settings, offline_storage):
    existing = from_api(item(5), "pages", settings)
    commit(settings, {
        settings.baseline_file: ("snapshot", [existing]),
        settings.snapshot_file: ("snapshot", [existing]),
    })
    result = scan(settings, run=True, client=FakeClient([item(5)], []))
    assert read_rows(settings.output_file("new")) == []
    assert read_rows(settings.output_file("deleted")) == []
    assert len(list(settings.output_dir.glob("run_*"))) == 1
    assert result["new"] == result["deleted"] == 0


def test_interrupted_commit_does_not_publish_partial_run(settings, offline_storage, monkeypatch):
    from pathlib import Path
    from wp_tracker import storage

    prepare_empty_baseline(settings)
    first = scan(settings, run=True, client=FakeClient([item(1)], []))
    latest = settings.reports_dir / "latest.json"
    old_report = latest.read_bytes()
    old_snapshot = settings.snapshot_file.read_bytes()
    real_replace = storage.os.replace

    def fail_second_file(src, dst):
        if Path(src).parent.name == "stage" and Path(src).name == "file_1.json":
            raise RuntimeError("Injected run failure")
        return real_replace(src, dst)

    monkeypatch.setattr(storage.os, "replace", fail_second_file)
    with pytest.raises(RuntimeError, match="Injected run failure"):
        scan(settings, run=True, client=FakeClient([item(1), item(2)], []))
    monkeypatch.setattr(storage.os, "replace", real_replace)
    assert latest.read_bytes() == old_report
    assert settings.snapshot_file.read_bytes() == old_snapshot
    assert len(list(settings.output_dir.glob("run_*"))) == 2
    # The complete archive survives a state failure, but is not published.
    assert settings.output_file("new").parent == Path(first["run_dir"])


def test_bootstrap_creates_separate_initialization_folder(settings, record, offline_storage):
    settings.import_dir.mkdir(parents=True)
    page = record(kind="page", number=1)
    post = record(kind="post", number=2)
    dataset([page]).to_parquet(settings.import_dir / "pages.parquet")
    dataset([post]).to_parquet(settings.import_dir / "posts.parquet")

    preview = bootstrap(settings)
    assert preview["run_dir"].startswith(str(settings.output_dir))
    assert not settings.output_dir.exists()

    result = bootstrap(settings, run=True)
    from pathlib import Path
    folder = Path(result["run_dir"])
    assert folder.name.startswith("run_")
    assert folder.parent == settings.output_dir
    assert set(entry.name for entry in folder.iterdir()) == {
        "chatbot_dataset_new.parquet", "chatbot_dataset_deleted.parquet", "latest.json", "run_report.json"
    }
    assert status(settings)["latest_run_dir"] == str(folder)
    assert status(settings)["outputs"]["new"]["records"] == 0
