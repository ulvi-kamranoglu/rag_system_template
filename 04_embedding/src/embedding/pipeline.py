import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from embedding.config import load_settings
from pipeline_storage import archive_run, publish, run_path, write_json



REQUIRED_COLUMNS = {
    "document_id",
    "chunk_id",
    "chunk_index",
    "embedding_text",
}


def validate_input(
    df: pl.DataFrame,
) -> None:

    print("=" * 70)
    print("INPUT VALIDATION")
    print("=" * 70)

    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.height == 0:
        raise ValueError(
            "Input chunk dataset is empty."
        )

    null_embedding_text = (
        df
        .select(
            pl.col("embedding_text")
            .is_null()
            .sum()
        )
        .item()
    )

    if null_embedding_text > 0:
        raise ValueError(
            "embedding_text contains "
            f"{null_embedding_text} null values."
        )

    empty_embedding_text = (
        df
        .filter(
            pl.col("embedding_text")
            .cast(pl.String)
            .str.strip_chars()
            == ""
        )
        .height
    )

    if empty_embedding_text > 0:
        raise ValueError(
            "embedding_text contains "
            f"{empty_embedding_text} empty values."
        )

    duplicate_chunk_ids = (
        df
        .group_by("chunk_id")
        .len()
        .filter(
            pl.col("len") > 1
        )
        .height
    )

    if duplicate_chunk_ids > 0:
        raise ValueError(
            "Duplicate chunk_id values found: "
            f"{duplicate_chunk_ids}"
        )

    print(
        f"Rows              : {df.height:,}"
    )

    print(
        f"Columns           : {len(df.columns)}"
    )

    print(
        f"Unique chunks     : "
        f"{df['chunk_id'].n_unique():,}"
    )

    print(
        f"Unique documents  : "
        f"{df['document_id'].n_unique():,}"
    )

    print(
        "Input validation  : PASSED"
    )


def build_mapping(
    df: pl.DataFrame,
) -> pl.DataFrame:
    preferred_columns = [
        "document_id",
        "original_id",

        # Delta information
        "change_type",
        "retrieval_hash",

        # Chunk identity
        "chunk_id",
        "chunk_index",

        # Retrieval content
        "title",
        "chunk_text",
        "embedding_text",
        "embedding_tokens",

        # Backward compatibility
        "content_hash",

        # Metadata
        "url",
        "date",
        "modified",
        "source_type",
        "source",
        "language",
    ]
    available_columns = [
        column
        for column
        in preferred_columns
        if column in df.columns
    ]

    mapping = (
        df
        .select(
            available_columns
        )
        .with_row_index(
            name="vector_index",
            offset=0,
        )
    )

    # vector_index-i əvvələ keçiririk.
    ordered_columns = [
        "vector_index"
    ] + [
        column
        for column
        in mapping.columns
        if column != "vector_index"
    ]

    return mapping.select(
        ordered_columns
    )


def validate_run(folder: Path) -> None:
    """Validate the persisted set without loading a model or its source input."""
    report = json.loads((folder / "run_report.json").read_text(encoding="utf-8"))
    paths = {}
    for key, suffix in (("vectors", "_embeddings.npy"), ("mapping", "_embedding_mapping.parquet"),
                        ("metadata", "_embedding_metadata.json")):
        name = report["files"][key]
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(suffix):
            raise ValueError("Invalid embedding archive filename")
        if name != report["dataset_name"] + suffix:
            raise ValueError("Embedding artifact dataset mismatch")
        paths[key] = folder / name
    vectors = np.load(paths["vectors"], allow_pickle=False, mmap_mode="r")
    try:
        mapping = pl.read_parquet(paths["mapping"])
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        expected = (report["rows"], report["dimension"])
        if len(vectors.shape) != 2 or vectors.shape != expected or report["rows"] <= 0:
            raise ValueError("Invalid archived embedding shape")
        if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
            raise ValueError("Invalid archived embedding values")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
            raise ValueError("Archived embeddings are not normalized")
        if not (REQUIRED_COLUMNS | {"vector_index"}).issubset(mapping.columns):
            raise ValueError("Invalid archived mapping columns")
        if mapping.height != expected[0] or mapping["chunk_id"].null_count() or mapping["chunk_id"].n_unique() != mapping.height:
            raise ValueError("Invalid archived mapping identities")
        if mapping["vector_index"].to_list() != list(range(mapping.height)):
            raise ValueError("Archived mapping is not aligned with vectors")
        if (metadata["shape"] != list(expected) or metadata["rows"] != expected[0]
                or metadata["dimension"] != expected[1] or metadata["dataset_name"] != report["dataset_name"]
                or metadata["model"] != report["model"]
                or metadata["embedding_version"] != report["embedding_version"]):
            raise ValueError("Archived embedding metadata mismatch")
    finally:
        # Windows directory renames require releasing the memory-mapped file.
        vectors._mmap.close()


def publish_archive(target, settings, *, migrate_legacy=False):
    try:
        return publish(target, settings.embeddings_dir, validate_run, migrate_legacy=migrate_legacy)
    except Exception as exc:
        raise RuntimeError(
            f"Archive complete at {target}; publication failed. Retry without model inference: "
            f'python -m embedding.cli --config "{settings.config_path}" --publish-run "{target}": {exc}'
        ) from exc


def run(config_path=None, *, input_override=None, dataset=None, dry_run=False,
        publish_run=None, migrate_legacy=False):
    if publish_run:
        if dry_run or input_override:
            raise ValueError("--publish-run cannot be combined with --dry-run or --input")
        target = Path(publish_run).resolve()
        report = json.loads((target / "run_report.json").read_text(encoding="utf-8"))
        if dataset and dataset != report["dataset_name"]:
            raise ValueError("Selected dataset differs from the archive")
        settings = load_settings(config_path, dataset=report["dataset_name"], for_publication=True)
        if target.parent != settings.archive_dir or not target.name.startswith("run_"):
            raise ValueError("--publish-run must name a run in the configured embedding archive")
        report["publication"] = publish_archive(target, settings, migrate_legacy=migrate_legacy)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    settings = load_settings(config_path, input_override=input_override, dataset=dataset)
    target = run_path(settings.archive_dir)
    MODEL_ID = settings.model_id
    EMBEDDING_DIMENSION = settings.embedding_dimension
    DOCUMENT_MAX_LENGTH = settings.document_max_length
    DOCUMENT_BATCH_SIZE = settings.document_batch_size
    EMBEDDING_VERSION = settings.embedding_version
    CHUNKS_INPUT_PATH = settings.chunks_input_path
    EMBEDDINGS_OUTPUT_PATH = settings.embeddings_output_path
    MAPPING_OUTPUT_PATH = settings.mapping_output_path
    METADATA_OUTPUT_PATH = settings.metadata_output_path
    EMBEDDINGS_DIR = settings.embeddings_dir

    print()
    print("=" * 70)
    print("UNEC EMBEDDING PIPELINE")
    print("=" * 70)

    print(
        f"Input  : "
        f"{CHUNKS_INPUT_PATH}"
    )

    print(
        f"Output : "
        f"{EMBEDDINGS_DIR}"
    )

    print(
        f"Model  : "
        f"{MODEL_ID}"
    )

    print(
        f"Version: "
        f"{EMBEDDING_VERSION}"
    )

    print()

    # ========================================================
    # CHECK INPUT
    # ========================================================

    if not CHUNKS_INPUT_PATH.exists():

        raise FileNotFoundError(
            "Chunk dataset not found:\n"
            f"{CHUNKS_INPUT_PATH}"
        )

    # ========================================================
    # LOAD CHUNKS
    # ========================================================

    print("Loading chunks...")

    df = pl.read_parquet(
        CHUNKS_INPUT_PATH
    )

    validate_input(
        df
    )
    if "embedding_tokens" in df.columns and df.filter(
        pl.col("embedding_tokens").is_null() | (pl.col("embedding_tokens") > DOCUMENT_MAX_LENGTH)
    ).height:
        raise ValueError("Input token counts are missing or exceed model.document_max_length")
    if dry_run:
        report = {"dry_run": True, "rows": df.height, "dataset_name": settings.dataset_name,
                  "archive_dir": str(settings.archive_dir), "run_dir": str(target),
                  "config": str(settings.config_path), "input": str(CHUNKS_INPUT_PATH),
                  "model": MODEL_ID, "dimension": EMBEDDING_DIMENSION,
                  "batch_size": DOCUMENT_BATCH_SIZE, "use_fp16": settings.use_fp16,
                  "outputs": [str(EMBEDDINGS_OUTPUT_PATH), str(MAPPING_OUTPUT_PATH), str(METADATA_OUTPUT_PATH)]}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    # ========================================================
    # PREPARE TEXTS
    # ========================================================

    texts = (
        df[
            "embedding_text"
        ]
        .cast(pl.String)
        .to_list()
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    reuse_dataset = os.getenv("EMBEDDING_REUSE_DATASET")
    missing = list(range(df.height))
    if reuse_dataset:
        from embedding.incremental import reuse_vectors
        vectors, missing = reuse_vectors(df, settings, reuse_dataset)
    if missing:
        from embedding.embedder import BgeM3Embedder
        embedder = BgeM3Embedder(settings=settings)
    else:
        from types import SimpleNamespace
        embedder = SimpleNamespace(device="cached")

    # ========================================================
    # EMBEDDING
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)

    start_time = time.perf_counter()

    if missing:
        generated = embedder.embed_texts(
            texts=[texts[index] for index in missing],
            batch_size=(
                DOCUMENT_BATCH_SIZE
            ),
        )
        if reuse_dataset:
            vectors[missing] = generated
        else:
            vectors = generated

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if vectors.shape != (
        df.height,
        EMBEDDING_DIMENSION,
    ):

        raise RuntimeError(
            "Unexpected final embedding shape. "
            f"Expected="
            f"({df.height}, "
            f"{EMBEDDING_DIMENSION}), "
            f"found={vectors.shape}"
        )

    if not np.isfinite(
        vectors
    ).all():

        raise RuntimeError(
            "Final embedding matrix "
            "contains NaN or Inf."
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    min_norm = float(
        norms.min()
    )

    max_norm = float(
        norms.max()
    )

    mean_norm = float(
        norms.mean()
    )

    # ========================================================
    # PREPARE OUTPUT SET; do not change shared files before the archive validates.
    # ========================================================

    mapping = build_mapping(
        df
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {
        "reused_vectors": df.height - len(missing),
        "generated_vectors": len(missing),
        "run_dir": str(target),
        "archive_dir": str(settings.archive_dir),
        "dataset_name": settings.dataset_name,
        "config_file": str(settings.config_path),
        "use_fp16": settings.use_fp16,
        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "model":
            MODEL_ID,

        "embedding_version":
            EMBEDDING_VERSION,

        "dimension":
            EMBEDDING_DIMENSION,

        "max_length":
            DOCUMENT_MAX_LENGTH,

        "batch_size":
            DOCUMENT_BATCH_SIZE,

        "device":
            embedder.device,

        "rows":
            int(df.height),

        "shape": [
            int(vectors.shape[0]),
            int(vectors.shape[1]),
        ],

        "dtype":
            str(vectors.dtype),

        "elapsed_seconds":
            round(
                elapsed,
                4,
            ),

        "vectors_per_second":
            round(
                (
                    df.height / elapsed
                    if elapsed > 0
                    else 0
                ),
                4,
            ),

        "norm_min":
            min_norm,

        "norm_max":
            max_norm,

        "norm_mean":
            mean_norm,

        "input_file":
            str(
                CHUNKS_INPUT_PATH
            ),

        "embeddings_file":
            str(
                EMBEDDINGS_OUTPUT_PATH
            ),

        "mapping_file":
            str(
                MAPPING_OUTPUT_PATH
            ),
    }

    report = {"dataset_name": settings.dataset_name, "run_dir": str(target),
              "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
              "archive_dir": str(settings.archive_dir), "shared_dir": str(EMBEDDINGS_DIR),
              "input_file": str(CHUNKS_INPUT_PATH), "config_file": str(settings.config_path),
              "model": MODEL_ID, "embedding_version": EMBEDDING_VERSION,
              "rows": df.height, "dimension": EMBEDDING_DIMENSION, "dry_run": False,
              "files": {"vectors": EMBEDDINGS_OUTPUT_PATH.name,
                        "mapping": MAPPING_OUTPUT_PATH.name, "metadata": METADATA_OUTPUT_PATH.name}}
    print(f"Saving complete embedding archive: {target}")
    with archive_run(target, validate_run) as staging:
        np.save(staging / EMBEDDINGS_OUTPUT_PATH.name, vectors.astype(np.float32))
        mapping.write_parquet(staging / MAPPING_OUTPUT_PATH.name)
        write_json(staging / METADATA_OUTPUT_PATH.name, metadata)
        write_json(staging / "run_report.json", report)
    publication = publish_archive(target, settings, migrate_legacy=migrate_legacy)
    metadata["publication"] = publication

    # ========================================================
    # SUMMARY
    # ========================================================

    embeddings_mb = (
        EMBEDDINGS_OUTPUT_PATH
        .stat()
        .st_size
        / 1024
        / 1024
    )

    print()
    print("=" * 70)
    print("EMBEDDING PIPELINE FINISHED")
    print("=" * 70)

    print(
        f"Vectors       : "
        f"{vectors.shape[0]:,}"
    )

    print(
        f"Dimension     : "
        f"{vectors.shape[1]}"
    )

    print(
        f"Shape         : "
        f"{vectors.shape}"
    )

    print(
        f"Dtype         : "
        f"{vectors.dtype}"
    )

    print(
        f"Device        : "
        f"{embedder.device}"
    )

    print(
        f"Elapsed       : "
        f"{elapsed:.2f} sec"
    )

    print(
        f"Vectors/sec   : "
        f"{df.height / elapsed:.2f}"
    )

    print(
        f"Norm min      : "
        f"{min_norm:.8f}"
    )

    print(
        f"Norm max      : "
        f"{max_norm:.8f}"
    )

    print(
        f"Norm mean     : "
        f"{mean_norm:.8f}"
    )

    print(
        f"Embedding size: "
        f"{embeddings_mb:.2f} MB"
    )

    print()
    print(
        "Embeddings:"
    )

    print(
        EMBEDDINGS_OUTPUT_PATH
    )

    print(
        "Mapping:"
    )

    print(
        MAPPING_OUTPUT_PATH
    )

    print(
        "Metadata:"
    )

    print(
        METADATA_OUTPUT_PATH
    )
    print(f"Archive: {target}")
    return metadata


def main():
    from embedding.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
