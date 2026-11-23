"""Reuse aligned, compatible vectors for exactly unchanged embedding texts."""
import json
import numpy as np
import polars as pl


def reuse_vectors(frame, settings, dataset):
    from .config import load_settings
    from .pipeline import validate_run
    base = load_settings(settings.config_path, dataset=dataset)
    validate_run(base.embeddings_dir)
    metadata = json.loads(base.metadata_output_path.read_text(encoding="utf-8"))
    expected = {"model": settings.model_id, "dimension": settings.embedding_dimension,
                "embedding_version": settings.embedding_version,
                "max_length": settings.document_max_length, "use_fp16": settings.use_fp16}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Embedding cache configuration differs; a compatible baseline is required")
    old = pl.read_parquet(base.mapping_output_path)
    if "embedding_text" not in old.columns:
        raise ValueError("Embedding cache lacks embedding_text")
    matrix = np.load(base.embeddings_output_path, mmap_mode="r")
    by_id = {row["chunk_id"]: (row["vector_index"], row["embedding_text"])
             for row in old.iter_rows(named=True)}
    vectors = np.empty((frame.height, settings.embedding_dimension), dtype=np.float32)
    missing = []
    for position, row in enumerate(frame.iter_rows(named=True)):
        cached = by_id.get(row["chunk_id"])
        if cached is not None and cached[1] == row["embedding_text"]:
            vectors[position] = matrix[cached[0]]
        else:
            missing.append(position)
    return vectors, missing
