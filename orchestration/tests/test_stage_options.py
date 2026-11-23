"""Opaque per-stage options: validated shape only, stored with the run, reused on resume."""
import json

import pytest

from orchestration import runner
from orchestration.cli import parse_stage_options


def test_options_are_grouped_by_stage():
    assert parse_stage_options(["collection:--sources=news", "collection:--allow-large-deletions"], "collect") == {
        "collection": ["--sources=news", "--allow-large-deletions"]}


@pytest.mark.parametrize("value,mode,match", [
    ("--sources=news", "collect", "STAGE:--OPTION"),
    ("unknown:--x", "collect", "STAGE:--OPTION"),
    ("collection:sources=news", "collect", "Invalid stage option"),
    ("collection:--run-id=x", "collect", "Invalid stage option"),
    ("collection:--sources=a b", "collect", "Invalid stage option"),
    ("collection:--sources=news", "join", "require --mode collect"),
])
def test_invalid_options_are_rejected(value, mode, match):
    with pytest.raises(ValueError, match=match):
        parse_stage_options([value], mode)


def test_runner_appends_stored_options_to_the_stage_command(tmp_path, monkeypatch):
    commands = []
    def call(command, folder, env, label):
        commands.append(command)
        receipt = command[command.index("--result") + 1]
        (tmp_path / receipt).write_text(json.dumps({"version": 1, "stage": label, "run_id": "r", "mode": "collect",
                                                    "status": "no_changes"}), encoding="utf-8")
    monkeypatch.setattr(runner, "call", call)
    state = {"protocol": 2, "mode": "collect", "run_id": "r", "completed": {},
             "commands": {name: [f"{name}.cli"] for name in ("collection", "canonical", "chunk", "embedding", "retrieval")},
             "stage_options": {"collection": ["--sources=news"]}}
    runner.execute(tmp_path, state)
    assert commands[0][-1] == "--sources=news" and commands[0][0] == "collection.cli"
    assert state["status"] == "no_changes"
