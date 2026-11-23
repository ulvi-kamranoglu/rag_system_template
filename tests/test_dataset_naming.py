"""Dataset names survive the preparation and artifact boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from canonical_pipeline.pipeline import process, validate_run as validate_canonical
from chunking_pipeline.pipeline import run, validate_run as validate_chunks
from pipeline_storage import dataset_stem, artifact_dataset, select_input, inventory
from test_core import FakeTokenizer


def test_versioned_name_flows_from_raw_to_chunks(tmp_path, monkeypatch):
    source = tmp_path / "chatbot_dataset_v1.2.parquet"
    pd.DataFrame([{"id": "15", "source_type": "form", "title": "Where?",
                   "content": "In the library. Here is more information.", "metadata": "{}"}]).to_parquet(source)
    shared = tmp_path / "shared_data"
    result = process([source], tmp_path / "archive/02_clean", shared / "02_clean")
    assert Path(result["output"]).name == "chatbot_dataset_v1.2_canonical.parquet"
    validate_canonical(Path(result["run_dir"]))
    config = tmp_path / "Chunking/config/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({"input_dir": "../shared_data/02_clean", "model_id": "offline",
        "storage": {"archive_dir": "../archive/03_chunks", "shared_dir": "../shared_data/03_chunks"},
        "variants": [{"max_tokens": 384, "overlap_tokens": 64}]}))
    monkeypatch.setattr("chunking_pipeline.pipeline.AutoTokenizer.from_pretrained", lambda _: FakeTokenizer())
    outputs = run(str(config))
    assert outputs == [shared / "03_chunks/chatbot_dataset_v1.2_chunks_384.parquet"]
    validate_chunks(outputs[0].parent)
    monkeypatch.delenv("PIPELINE_DATASET", raising=False)
    assert artifact_dataset(shared) == "chatbot_dataset_v1.2"
    assert dataset_stem(outputs[0]) == "chatbot_dataset_v1.2"


def test_ambiguous_input_requires_selection(tmp_path, monkeypatch):
    for name in ("a.parquet", "b.parquet"):
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match="select the input explicitly"):
        select_input(tmp_path, "*.parquet")
    chunks = tmp_path / "03_chunks"
    chunks.mkdir()
    for name in ("a", "b"):
        (chunks / f"{name}_chunks_384.parquet").touch()
    monkeypatch.delenv("PIPELINE_DATASET", raising=False)
    with pytest.raises(ValueError, match="Multiple datasets"):
        artifact_dataset(tmp_path)
    monkeypatch.setenv("PIPELINE_DATASET", "chatbot_dataset_v1.2")
    assert artifact_dataset(tmp_path) == "chatbot_dataset_v1.2"
    monkeypatch.setenv("PIPELINE_DATASET", "../outside")
    with pytest.raises(ValueError, match="filename stem"):
        artifact_dataset(tmp_path)


def test_embedding_and_retrieval_share_versioned_paths():
    environment = {**os.environ, "PIPELINE_DATASET": "chatbot_dataset_v1.2"}
    code = """
import json
import embedding.config as e
import retrieval.config as r
import retrieval.bm25_pipeline as b
print(json.dumps([p.name for p in [e.CHUNKS_INPUT_PATH,e.EMBEDDINGS_OUTPUT_PATH,
    e.MAPPING_OUTPUT_PATH,r.EMBEDDINGS_PATH,r.MAPPING_PATH,r.BM25_CORPUS_PATH,
    b.BM25_CORPUS_PATH,r.FAISS_INDEX_PATH]]))
"""
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=True)
    names = json.loads(result.stdout)
    assert all(name.startswith("chatbot_dataset_v1.2") for name in names)
    assert names[1] == names[3] == "chatbot_dataset_v1.2_embeddings.npy"
    assert names[2] == names[4] == "chatbot_dataset_v1.2_embedding_mapping.parquet"
    assert names[5] == names[6] == "chatbot_dataset_v1.2_bm25_corpus.parquet"


def test_legacy_canonical_archive_remains_readable(tmp_path):
    source = tmp_path / "original.parquet"
    pd.DataFrame([{"id": "15", "source_type": "page", "title": "T", "content": "Some content."}]).to_parquet(source)
    report = process([source], tmp_path / "archive", tmp_path / "shared")
    folder = Path(report["run_dir"])
    (folder / report["output_file"]).rename(folder / "canonical_dataset.parquet")
    report.pop("output_file")
    (folder / "run_report.json").write_text(json.dumps(report))
    before = inventory(folder)
    validate_canonical(folder)
    assert inventory(folder) == before


def test_runner_invokes_current_python_with_utf8(tmp_path, monkeypatch):
    from orchestration import runner as run_pipeline
    from types import SimpleNamespace
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)
    environment = {"PIPELINE_DATASET": "chatbot_dataset_v1.4"}
    run_pipeline.call(["embedding.cli", "--dataset", "chatbot_dataset_v1.4"], tmp_path, environment, "embedding")
    assert calls[0][0][:4] == [sys.executable, "-X", "utf8", "-m"]
    assert calls[0][1]["env"] == environment
