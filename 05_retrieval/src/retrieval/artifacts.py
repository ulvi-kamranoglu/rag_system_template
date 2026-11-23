"""Embedding artifact validation and Qdrant/FAISS payload preparation."""
from datetime import date, datetime
import numpy as np
import polars as pl
from retrieval.config import VECTOR_DIMENSION

PAYLOAD_COLUMNS = [
    "document_id",
    "original_id",

    # Delta
    "change_type",
    "retrieval_hash",

    # Chunk
    "chunk_id",
    "chunk_index",

    # Content
    "title",
    "chunk_text",
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

def normalize_payload_value(value):

    if value is None:
        return None

    if isinstance(
        value,
        (datetime, date),
    ):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(value, np.bool_):
        return bool(value)

    return value

def validate_artifacts(
    vectors: np.ndarray,
    mapping: pl.DataFrame,
) -> None:

    print("=" * 70)
    print("ARTIFACT VALIDATION")
    print("=" * 70)

    if vectors.ndim != 2:
        raise ValueError(
            "Embedding matrix must be 2D. "
            f"Found shape={vectors.shape}"
        )

    if vectors.shape[1] != VECTOR_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension. "
            f"Expected={VECTOR_DIMENSION}, "
            f"found={vectors.shape[1]}"
        )

    if mapping.height != vectors.shape[0]:
        raise ValueError(
            "Embedding/mapping row mismatch. "
            f"Vectors={vectors.shape[0]}, "
            f"mapping={mapping.height}"
        )

    required_columns = {
        "vector_index",
        "document_id",
        "chunk_id",
    }

    missing = (
        required_columns
        - set(mapping.columns)
    )

    if missing:
        raise ValueError(
            "Mapping is missing required "
            f"columns: {sorted(missing)}"
        )

    if (
        mapping["chunk_id"]
        .null_count()
        > 0
    ):
        raise ValueError(
            "chunk_id contains null values."
        )

    if (
        mapping["document_id"]
        .null_count()
        > 0
    ):
        raise ValueError(
            "document_id contains null values."
        )

    if (
        mapping["chunk_id"]
        .n_unique()
        != mapping.height
    ):
        raise ValueError(
            "chunk_id values must be unique."
        )

    expected_indexes = list(
        range(mapping.height)
    )

    actual_indexes = (
        mapping["vector_index"]
        .cast(pl.Int64)
        .to_list()
    )

    if actual_indexes != expected_indexes:
        raise ValueError(
            "vector_index must be sequential "
            "and aligned with embeddings.npy."
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Embedding matrix contains "
            "NaN or Inf values."
        )

    print(
        f"Vectors           : "
        f"{vectors.shape[0]:,}"
    )

    print(
        f"Dimension         : "
        f"{vectors.shape[1]}"
    )

    print(
        f"Mapping rows      : "
        f"{mapping.height:,}"
    )

    print(
        f"Unique chunks     : "
        f"{mapping['chunk_id'].n_unique():,}"
    )

    print(
        f"Unique documents  : "
        f"{mapping['document_id'].n_unique():,}"
    )

    print(
        "Artifact validation: PASSED"
    )

def build_payloads(
    mapping: pl.DataFrame,
) -> list[dict]:

    available_columns = [
        column
        for column in PAYLOAD_COLUMNS
        if column in mapping.columns
    ]

    rows = (
        mapping
        .select(available_columns)
        .rows(named=True)
    )

    payloads = []

    for row in rows:

        payload = {
            key: normalize_payload_value(
                value
            )
            for key, value in row.items()
        }

        payloads.append(
            payload
        )

    return payloads
