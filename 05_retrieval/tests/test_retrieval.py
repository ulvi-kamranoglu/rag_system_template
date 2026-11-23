import pytest

pytestmark = pytest.mark.integration

import numpy as np

from embedding.embedder import (
    BgeM3Embedder,
)

from retrieval.vector_index import (
    QdrantVectorIndex,
)


TEST_COLLECTION = (
    "unec_chunks_integration_test"
)


@pytest.fixture(scope="module")
def embedder():

    return BgeM3Embedder()


@pytest.fixture(scope="module")
def index():

    qdrant = QdrantVectorIndex(
        collection_name=TEST_COLLECTION
    )

    qdrant.health_check()

    qdrant.recreate_collection()

    yield qdrant

    if qdrant.collection_exists():

        qdrant.client.delete_collection(
            collection_name=(
                TEST_COLLECTION
            )
        )

    qdrant.close()


def test_qdrant_health(
    index,
):

    assert (
        index.health_check()
        is True
    )


def test_upsert_and_count(
    embedder,
    index,
):

    texts = [
        (
            "Title: UNEC bakalavr\n\n"
            "Content:\n"
            "UNEC-də bakalavr "
            "proqramları mövcuddur."
        ),
        (
            "Title: UNEC bakalavr\n\n"
            "Content:\n"
            "Tələbələr müxtəlif "
            "ixtisaslarda təhsil "
            "ala bilərlər."
        ),
        (
            "Title: UNEC magistratura\n\n"
            "Content:\n"
            "UNEC-də magistratura "
            "proqramları təqdim olunur."
        ),
    ]

    vectors = embedder.embed_texts(
        texts
    )

    payloads = [
        {
            "document_id":
                "doc_test_001",

            "chunk_id":
                "doc_test_001_chunk_0000",

            "chunk_index": 0,

            "title":
                "UNEC bakalavr",

            "chunk_text":
                (
                    "UNEC-də bakalavr "
                    "proqramları mövcuddur."
                ),
        },
        {
            "document_id":
                "doc_test_001",

            "chunk_id":
                "doc_test_001_chunk_0001",

            "chunk_index": 1,

            "title":
                "UNEC bakalavr",

            "chunk_text":
                (
                    "Tələbələr müxtəlif "
                    "ixtisaslarda təhsil "
                    "ala bilərlər."
                ),
        },
        {
            "document_id":
                "doc_test_002",

            "chunk_id":
                "doc_test_002_chunk_0000",

            "chunk_index": 0,

            "title":
                "UNEC magistratura",

            "chunk_text":
                (
                    "UNEC-də magistratura "
                    "proqramları təqdim olunur."
                ),
        },
    ]

    written = index.upsert(
        vectors=vectors,
        payloads=payloads,
    )

    assert written == 3

    assert index.count() == 3


def test_semantic_search(
    embedder,
    index,
):

    query = (
        "UNEC-də magistratura "
        "proqramları varmı?"
    )

    query_vector = (
        embedder.embed_text(
            query
        )
    )

    assert query_vector.shape == (
        1024,
    )

    assert np.isfinite(
        query_vector
    ).all()

    results = index.search(
        query_vector=query_vector,
        limit=3,
    )

    assert len(results) == 3

    top_result = results[0]

    assert (
        top_result[
            "payload"
        ]["chunk_id"]
        == "doc_test_002_chunk_0000"
    )

    assert (
        top_result[
            "payload"
        ]["title"]
        == "UNEC magistratura"
    )

    assert (
        top_result["score"]
        > results[1]["score"]
    )


def test_deterministic_point_id():

    chunk_id = (
        "doc_test_001_chunk_0000"
    )

    first = (
        QdrantVectorIndex
        .make_point_id(
            chunk_id
        )
    )

    second = (
        QdrantVectorIndex
        .make_point_id(
            chunk_id
        )
    )

    assert first == second


def test_invalid_vector_dimension(
    index,
):

    invalid_vector = np.zeros(
        100,
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError
    ):

        index.search(
            query_vector=(
                invalid_vector
            ),
            limit=5,
        )