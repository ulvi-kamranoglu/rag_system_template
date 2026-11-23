import pytest

pytestmark = pytest.mark.integration

import numpy as np
import polars as pl

from embedding.config import (
    EMBEDDING_DIMENSION,
)

from embedding.embedder import (
    BgeM3Embedder,
)

from embedding.pipeline import (
    validate_input,
    build_mapping,
)


def test_single_embedding():

    embedder = BgeM3Embedder()

    vector = embedder.embed_text(
        "Title: UNEC\n\n"
        "Content:\n"
        "UNEC haqqında test məlumatı."
    )

    assert vector.shape == (
        EMBEDDING_DIMENSION,
    )

    assert np.isfinite(
        vector
    ).all()

    norm = np.linalg.norm(
        vector
    )

    assert np.isclose(
        norm,
        1.0,
        atol=1e-5,
    )


def test_batch_embeddings():

    embedder = BgeM3Embedder()

    texts = [
        (
            "Title: UNEC bakalavr\n\n"
            "Content:\n"
            "UNEC-də bakalavr "
            "proqramları mövcuddur."
        ),
        (
            "Title: UNEC magistratura\n\n"
            "Content:\n"
            "UNEC-də magistratura "
            "proqramları mövcuddur."
        ),
        (
            "Title: UNEC tələbələr\n\n"
            "Content:\n"
            "UNEC tələbələr üçün "
            "müxtəlif xidmətlər təqdim edir."
        ),
    ]

    vectors = (
        embedder.embed_texts(
            texts
        )
    )

    assert vectors.shape == (
        3,
        EMBEDDING_DIMENSION,
    )

    assert np.isfinite(
        vectors
    ).all()

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    assert np.allclose(
        norms,
        1.0,
        atol=1e-5,
    )


def test_chunk_input_validation():

    df = pl.DataFrame(
        {
            "document_id": [
                "doc_001",
                "doc_001",
            ],

            "chunk_id": [
                "doc_001_chunk_0000",
                "doc_001_chunk_0001",
            ],

            "chunk_index": [
                0,
                1,
            ],

            "embedding_text": [
                "Title: Test\nContent: A",
                "Title: Test\nContent: B",
            ],
        }
    )

    # Error verməməlidir.
    validate_input(
        df
    )


def test_mapping():

    df = pl.DataFrame(
        {
            "document_id": [
                "doc_001",
                "doc_002",
            ],

            "change_type": [
                "NEW",
                "UPDATED",
            ],

            "retrieval_hash": [
                "hash_new_001",
                "hash_updated_002",
            ],

            "chunk_id": [
                "doc_001_chunk_0000",
                "doc_002_chunk_0000",
            ],

            "chunk_index": [
                0,
                0,
            ],

            "embedding_text": [
                "Text A",
                "Text B",
            ],
        }
    )

    mapping = build_mapping(
        df
    )

    assert mapping.height == 2

    assert (
        mapping["vector_index"]
        .to_list()
        == [0, 1]
    )

    assert (
        mapping["chunk_id"]
        .to_list()
        == [
            "doc_001_chunk_0000",
            "doc_002_chunk_0000",
        ]
    )

    assert (
        mapping["change_type"]
        .to_list()
        == [
            "NEW",
            "UPDATED",
        ]
    )

    assert (
        mapping["retrieval_hash"]
        .to_list()
        == [
            "hash_new_001",
            "hash_updated_002",
        ]
    )