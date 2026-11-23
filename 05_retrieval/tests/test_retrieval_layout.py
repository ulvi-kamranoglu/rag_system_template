"""Compatibility paths must share state with their canonical modules."""
import importlib
import os
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("old,new", [
    ("vector_index", "backends.qdrant"),
    ("faiss_index", "backends.faiss"),
    ("bm25_index", "backends.bm25"),
    ("hybrid_fusion", "search.fusion"),
    ("ranking_policy", "search.policy"),
    ("search_flow", "search.flow"),
    ("diagnostics", "search.diagnostics"),
    ("pipeline", "indexing.qdrant"),
    ("bm25_pipeline", "indexing.bm25"),
    ("faiss_pipeline", "indexing.faiss"),
    ("delta_sync_pipeline", "indexing.delta"),
    ("sync_pipeline", "indexing.sync"),
    ("storage", "indexing.storage"),
])
def test_legacy_import_is_the_same_module(old, new):
    assert importlib.import_module(f"retrieval.{old}") is importlib.import_module(f"retrieval.{new}")


def test_cli_configures_environment_before_loading_stage(monkeypatch, tmp_path):
    from retrieval import cli
    for key in ("RETRIEVAL_CONFIG", "CHATBOT_PROJECT_ROOT", "PIPELINE_DATASET"):
        monkeypatch.setenv(key, "before")
    observed = []
    def load(name):
        observed.append((name, os.environ["PIPELINE_DATASET"], os.environ["RETRIEVAL_CONFIG"]))
        return SimpleNamespace(main=lambda argv: observed.append(argv))
    monkeypatch.setattr(cli.importlib, "import_module", load)
    config = tmp_path / "settings.yaml"
    cli.main(["--config", str(config), "--dataset", "sample_v1.4", "--stage", "sync",
              "--publish-run", "completed-run"])
    assert observed == [("retrieval.indexing.sync", "sample_v1.4", str(config.resolve())),
                        ["--publish-run", "completed-run"]]
