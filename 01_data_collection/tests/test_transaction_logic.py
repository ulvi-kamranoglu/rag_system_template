"""Offline transaction and pipeline tests; stand-ins are NOT Parquet format tests."""
import pandas as pd
import pytest
from pathlib import Path

from wp_tracker.pipeline import bootstrap, scan, status
from wp_tracker.records import dataset
from wp_tracker.storage import read_rows, commit


@pytest.fixture
def offline_storage(monkeypatch):
    """Use pickle bytes only to exercise orchestration without optional pyarrow."""
    def stand_in_write(frame, path, *args, **kwargs):
        frame.to_pickle(path)
    def stand_in_read(path, *args, **kwargs):
        return pd.read_pickle(path)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", stand_in_write)
    monkeypatch.setattr(pd, "read_parquet", stand_in_read)


class FakeClient:
    def __init__(self, pages, posts, fail_posts=False):
        self.sources = {"pages": pages, "posts": posts}
        self.fail_posts = fail_posts

    def fetch_all(self, source):
        if source == "posts" and self.fail_posts:
            raise RuntimeError("API interrupted")
        return self.sources[source]


def api_item(number, title, content):
    return {
        "id": number, "title": {"rendered": title},
        "content": {"rendered": "<p>" + content + "</p>"},
        "link": f"https://unec.edu.az/{number}/",
        "date_gmt": "2026-10-01T00:00:00",
        "modified_gmt": "2026-10-01T00:00:00",
    }


def test_two_event_runs_and_history(settings, offline_storage):
    from wp_tracker.records import from_api

    first = FakeClient([api_item(2, "Page", "First")], [api_item(2, "Post", "Old")])
    baseline = [
        from_api(api_item(2, "Page", "First"), "pages", settings),
        from_api(api_item(2, "Post", "Old"), "posts", settings),
    ]
    commit(settings, {
        settings.baseline_file: ("snapshot", baseline),
        settings.snapshot_file: ("snapshot", baseline),
    })
    assert scan(settings, client=first)["new"] == 0
    assert scan(settings, run=True, client=first)["snapshot_after"] == 2
    assert status(settings)["baseline"]["records"] == 2
    assert scan(settings, run=True, client=first)["new"] == 0

    changed = FakeClient([api_item(2, "Page", "Changed")], [api_item(3, "New", "New content")])
    result = scan(settings, run=True, client=changed, allow_large_deletions=True)
    assert (result["new"], result["deleted"]) == (1, 1)
    assert read_rows(settings.output_file("deleted"))[0]["id"] == "post:2"
    assert status(settings)["baseline"]["records"] == 2

    # A later run with identical API data preserves all baseline-relative changes.
    repeated = scan(settings, run=True, client=changed, allow_large_deletions=True)
    assert (repeated["new"], repeated["deleted"]) == (1, 1)
    # Each successful scan retains its own immutable NEW and DELETED outputs.
    run_dirs = sorted(settings.output_dir.glob("run_*"))
    assert len(run_dirs) == 4
    for folder in run_dirs:
        assert (folder / "chatbot_dataset_new.parquet").is_file()
        assert (folder / "chatbot_dataset_deleted.parquet").is_file()
    assert [row["id"] for row in read_rows(
        settings.output_file("new", Path(repeated["run_dir"]))
    )] == ["post:3"]
    assert [row["id"] for row in read_rows(
        settings.output_file("deleted", Path(repeated["run_dir"]))
    )] == ["post:2"]


def test_failed_api_preserves_all_outputs(settings, offline_storage):
    before = FakeClient([api_item(1, "A", "B")], [api_item(1, "C", "D")])
    commit(settings, {settings.baseline_file: ("snapshot", []),
                      settings.snapshot_file: ("snapshot", [])})
    scan(settings, run=True, client=before)
    backup = {path: path.read_bytes() for path in (
        settings.snapshot_file,
        settings.output_file("new"),
        settings.output_file("deleted"),
    )}
    fail = FakeClient([api_item(1, "Wrong", "X")], [], fail_posts=True)
    with pytest.raises(RuntimeError, match="API interrupted"):
        scan(settings, run=True, client=fail)
    assert all(path.read_bytes() == data for path, data in backup.items())


def test_baseline_bootstrap_no_false_first_events(settings, record, offline_storage):
    settings.import_dir.mkdir(parents=True)
    p = record(kind="page", number=1)
    q = record(kind="post", number=1)
    p["id"] = q["id"] = "1"
    dataset([p]).to_parquet(settings.import_dir / "pages.parquet")
    dataset([q]).to_parquet(settings.import_dir / "posts.parquet")
    assert bootstrap(settings)["baseline_records"] == 2
    assert not settings.snapshot_file.exists()
    bootstrap(settings, run=True)
    assert {x["id"] for x in read_rows(settings.baseline_file)} == {"page:1", "post:1"}
    assert {x["id"] for x in read_rows(settings.snapshot_file)} == {"page:1", "post:1"}
    assert status(settings)["outputs"]["new"]["records"] == 0
    with pytest.raises(FileExistsError):
        bootstrap(settings, run=True)


def test_deletion_safety_prevents_persist(settings, offline_storage):
    old = FakeClient([api_item(i, "A", "Text") for i in range(1, 21)], [api_item(2, "B", "Text")])
    commit(settings, {settings.baseline_file: ("snapshot", [
        __import__("wp_tracker.records", fromlist=["from_api"]).from_api(
            row, source, settings
        ) for source, records in (("pages", old.sources["pages"]), ("posts", old.sources["posts"]))
          for row in records]),
          settings.snapshot_file: ("snapshot", [])})
    scan(settings, run=True, client=old)
    previous = settings.snapshot_file.read_bytes()
    shrunk = FakeClient([api_item(i, "A", "Text") for i in range(2, 21)], [api_item(2, "B", "Text")])
    # 1/20 = 5% -> guard is strictly greater, so deletion is permitted.
    assert scan(settings, client=shrunk)["deleted"] == 1
    shrunk_more = FakeClient([api_item(i, "A", "Text") for i in range(3, 21)], [api_item(2, "B", "Text")])
    with pytest.raises(RuntimeError, match="Safety stop"):
        scan(settings, run=True, client=shrunk_more)
    assert settings.snapshot_file.read_bytes() == previous


def test_interrupted_commit_rolls_back_all_files(settings, record, offline_storage, monkeypatch):
    from wp_tracker import storage
    a, b = settings.output_file("new"), settings.snapshot_file
    commit(settings, {a: ("new", [record(number=1)]), b: ("snapshot", [record(number=1)])})
    old = {a: a.read_bytes(), b: b.read_bytes()}
    real_replace = storage.os.replace

    def fail_second_stage(src, dst):
        from pathlib import Path
        if Path(src).parent.name == "stage" and Path(src).name == "file_1.parquet":
            raise RuntimeError("Injected commit interruption")
        real_replace(src, dst)

    monkeypatch.setattr(storage.os, "replace", fail_second_stage)
    with pytest.raises(RuntimeError, match="Injected"):
        commit(settings, {a: ("new", [record(number=2)]), b: ("snapshot", [record(number=2)])})
    monkeypatch.setattr(storage.os, "replace", real_replace)
    assert a.read_bytes() == old[a]
    assert b.read_bytes() == old[b]
    assert not (settings.state_dir / ".pending").exists()


def test_scan_requires_immutable_baseline(settings, offline_storage):
    commit(settings, {settings.snapshot_file: ("snapshot", [])})
    with pytest.raises(RuntimeError, match="Immutable baseline is missing"):
        scan(settings, client=FakeClient([], []))


def test_bootstrap_reset_restores_baseline_and_clears_events(settings, record, offline_storage):
    settings.import_dir.mkdir(parents=True)
    page = record(kind="page", number=1)
    page["id"] = "1"
    post = record(kind="post", number=2)
    post["id"] = "2"
    dataset([page]).to_parquet(settings.import_dir / "pages.parquet")
    dataset([post]).to_parquet(settings.import_dir / "posts.parquet")

    bootstrap(settings, run=True)
    original_baseline = settings.baseline_file.read_bytes()
    new_data = FakeClient([api_item(1, "Modified", "Changed"), api_item(3, "Added", "New content")],
                          [api_item(2, "Modified", "Changed")])
    scan(settings, run=True, client=new_data)
    assert status(settings)["outputs"]["new"]["records"] == 1
    assert status(settings)["baseline"]["records"] == 2
    assert status(settings)["snapshot"]["records"] == 3

    with pytest.raises(FileExistsError, match="--reset"):
        bootstrap(settings, run=True)
    assert bootstrap(settings, reset=True)["baseline_records"] == 2
    assert settings.baseline_file.read_bytes() == original_baseline

    bootstrap(settings, run=True, reset=True)
    final = status(settings)
    assert final["baseline"]["records"] == 2
    assert final["snapshot"]["records"] == 2
    assert all(value["records"] == 0 for value in final["outputs"].values())
    assert scan(settings, client=new_data)["new"] == 1
