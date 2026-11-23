import json

import pytest

pytest.importorskip("pyarrow")

from wp_tracker.pipeline import bootstrap, scan, status
from wp_tracker.records import dataset
from wp_tracker.storage import read_rows, recover, commit


class FakeClient:
    def __init__(self, pages, posts, should_fail=False):
        self.rows = {"pages": pages, "posts": posts}
        self.should_fail = should_fail

    def fetch_all(self, source):
        if source == "posts" and self.should_fail:
            raise RuntimeError("Simulated API failure")
        return self.rows[source]


def item(number, title, content, kind):
    return {
        "id": number,
        "title": {"rendered": title},
        "content": {"rendered": f"<p>{content}</p>"},
        "link": f"https://unec.edu.az/{kind}/{number}/",
        "date_gmt": "2026-10-01T00:00:00",
        "modified_gmt": "2026-10-01T00:00:00",
    }


def test_scan_two_outputs_and_baseline_relative_changes(settings):
    from wp_tracker.records import from_api

    initial = FakeClient([item(1, "Page", "One", "page")], [item(7, "Post", "Old", "post")])
    baseline = [
        from_api(item(1, "Page", "One", "page"), "pages", settings),
        from_api(item(7, "Post", "Old", "post"), "posts", settings),
    ]
    commit(settings, {
        settings.baseline_file: ("snapshot", baseline),
        settings.snapshot_file: ("snapshot", baseline),
    })
    assert scan(settings, client=initial)["new"] == 0
    assert scan(settings, run=True, client=initial)["deleted"] == 0
    assert status(settings)["baseline"]["records"] == 2

    changed = FakeClient([item(1, "Page", "Changed", "page")], [item(8, "New", "New content", "post")])
    result = scan(settings, run=True, client=changed, allow_large_deletions=True)
    assert (result["new"], result["deleted"]) == (1, 1)
    assert read_rows(settings.output_file("new"))[0]["search_text"] == "New content"
    assert read_rows(settings.output_file("deleted"))[0]["id"] == "post:7"
    assert len(read_rows(settings.baseline_file)) == 2
    repeated = scan(settings, run=True, client=changed, allow_large_deletions=True)
    assert (repeated["new"], repeated["deleted"]) == (1, 1)
    assert len(list(settings.output_dir.glob("run_*/chatbot_dataset_new.parquet"))) == 3


def test_bad_api_does_not_change_persisted_state(settings):
    initial = FakeClient([item(1, "A", "B", "page")], [item(1, "C", "D", "post")])
    commit(settings, {settings.baseline_file: ("snapshot", []),
                      settings.snapshot_file: ("snapshot", [])})
    scan(settings, run=True, client=initial)
    prior = settings.snapshot_file.read_bytes()
    failing = FakeClient([item(1, "Changed", "B", "page")], [], should_fail=True)
    with pytest.raises(RuntimeError, match="Simulated"):
        scan(settings, run=True, client=failing)
    assert settings.snapshot_file.read_bytes() == prior


def test_bootstrap_from_v2_and_status(settings, record):
    settings.import_dir.mkdir(parents=True)
    page = record(kind="page", number=123)
    page["id"] = "123"
    post = record(kind="post", number=321)
    post["id"] = "321"
    dataset([page]).to_parquet(settings.import_dir / "pages.parquet", index=False)
    dataset([post]).to_parquet(settings.import_dir / "posts.parquet", index=False)
    dry = bootstrap(settings)
    assert dry["baseline_records"] == 2
    assert not settings.snapshot_file.exists()
    actual = bootstrap(settings, run=True)
    assert actual["baseline_records"] == 2
    assert {row["id"] for row in read_rows(settings.snapshot_file)} == {"page:123", "post:321"}
    assert {row["id"] for row in read_rows(settings.baseline_file)} == {"page:123", "post:321"}
    assert status(settings)["outputs"]["new"]["records"] == 0
    with pytest.raises(FileExistsError):
        bootstrap(settings, run=True)


def test_txn_recovery_restores_old_outputs(settings, record, monkeypatch):
    from wp_tracker import storage
    a = settings.output_file("new")
    b = settings.snapshot_file
    commit(settings, {a: ("new", [record(number=1)]), b: ("snapshot", [record(number=1)])})
    old_a = a.read_bytes()
    old_b = b.read_bytes()
    original_replace = storage.os.replace

    def fail_after_first_replace(src, dst):
        from pathlib import Path
        if Path(src).parent.name == "stage" and Path(src).name == "file_1.parquet":
            raise RuntimeError("Simulated interrupted commit")
        return original_replace(src, dst)

    monkeypatch.setattr(storage.os, "replace", fail_after_first_replace)
    with pytest.raises(RuntimeError, match="Simulated"):
        commit(settings, {a: ("new", [record(number=2)]), b: ("snapshot", [record(number=2)])})
    monkeypatch.setattr(storage.os, "replace", original_replace)
    recover(settings)
    assert a.read_bytes() == old_a
    assert b.read_bytes() == old_b
