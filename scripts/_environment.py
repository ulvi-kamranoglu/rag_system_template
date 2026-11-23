"""Select this delivery's source/configuration without loading model weights."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for name in ("storage_common", "01_data_collection", "02_data_canonicalization",
             "03_data_chunking", "04_embedding", "05_retrieval", "orchestration", "assistant_app"):
    sys.path.insert(0, str(ROOT / name / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("CHATBOT_PROJECT_ROOT", str(ROOT))


def snapshot(settings):
    """Validate the full local embedding/mapping/BM25 identity set, without models."""
    import json
    import numpy as np
    import polars as pl
    from retrieval.artifacts import validate_artifacts

    report_path = settings["INDEXES_DIR"] / "run_report.json"
    if report_path.is_file() and json.loads(report_path.read_text(encoding="utf-8")).get("batch_dataset_name"):
        # The active corpus is cumulative; shared embeddings intentionally
        # describe only the latest batch, not every indexed document.
        from retrieval.indexing.storage import validate_run
        validate_run(settings["INDEXES_DIR"])
        corpus = pl.read_parquet(settings["BM25_CORPUS_PATH"])
        desired = set(corpus["chunk_id"].to_list())
        if len(desired) != corpus.height:
            raise ValueError("Duplicate IDs in active BM25 corpus")
        return corpus.height, desired

    for key in ("EMBEDDINGS_PATH", "MAPPING_PATH", "EMBEDDING_METADATA_PATH",
                "BM25_CORPUS_PATH", "BM25_METADATA_PATH"):
        if not settings[key].is_file():
            raise FileNotFoundError(f"Missing {key}: {settings[key]}")
    if not (settings["INDEXES_DIR"] / "run_report.json").is_file():
        raise FileNotFoundError("Missing shared_data/05_indexes/run_report.json")
    vectors = np.load(settings["EMBEDDINGS_PATH"], mmap_mode="r")
    mapping = pl.read_parquet(settings["MAPPING_PATH"])
    validate_artifacts(vectors, mapping)
    if vectors.shape[1] != settings["VECTOR_DIMENSION"]:
        raise ValueError("Vector dimension differs from retrieval configuration")
    corpus = pl.read_parquet(settings["BM25_CORPUS_PATH"])
    desired = set(mapping["chunk_id"].to_list())
    if corpus.height != mapping.height or set(corpus["chunk_id"].to_list()) != desired:
        raise ValueError("BM25 corpus and vector mapping differ")
    metadata = json.loads(settings["EMBEDDING_METADATA_PATH"].read_text(encoding="utf-8"))
    if metadata.get("dimension") is not None and metadata["dimension"] != vectors.shape[1]:
        raise ValueError("Embedding metadata dimension mismatch")
    return mapping.height, desired
