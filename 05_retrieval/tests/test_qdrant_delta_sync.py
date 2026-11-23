import pytest

pytestmark = pytest.mark.integration

import numpy as np

from retrieval.vector_index import (
    QdrantVectorIndex,
)


TEST_COLLECTION = (
    "unec_chunks_delta_test"
)

DIMENSION = 1024


# ============================================================
# HELPERS
# ============================================================

def make_vector(
    position: int,
) -> np.ndarray:

    vector = np.zeros(
        DIMENSION,
        dtype=np.float32,
    )

    vector[position] = 1.0

    return vector


# ============================================================
# FIXTURE
# ============================================================

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


# ============================================================
# FULL DELTA SCENARIO
# ============================================================

def test_qdrant_delta_lifecycle(
    index,
):

    # --------------------------------------------------------
    # INITIAL STATE
    #
    # doc_001:
    #   chunk_0000
    #
    # doc_002:
    #   chunk_0000
    #   chunk_0001
    # --------------------------------------------------------

    initial_vectors = np.stack(
        [
            make_vector(0),
            make_vector(1),
            make_vector(2),
        ]
    )

    initial_payloads = [
        {
            "document_id":
                "doc_001",

            "chunk_id":
                "doc_001_chunk_0000",

            "retrieval_hash":
                "hash_doc_001_v1",

            "title":
                "Document 1",
        },
        {
            "document_id":
                "doc_002",

            "chunk_id":
                "doc_002_chunk_0000",

            "retrieval_hash":
                "hash_doc_002_v1",

            "title":
                "Old Document 2",
        },
        {
            "document_id":
                "doc_002",

            "chunk_id":
                "doc_002_chunk_0001",

            "retrieval_hash":
                "hash_doc_002_v1",

            "title":
                "Old Document 2",
        },
    ]

    written = index.upsert(
        vectors=initial_vectors,
        payloads=initial_payloads,
    )

    assert written == 3
    assert index.count() == 3

    state = index.get_chunk_state()

    assert set(
        state.keys()
    ) == {
        "doc_001_chunk_0000",
        "doc_002_chunk_0000",
        "doc_002_chunk_0001",
    }

    # --------------------------------------------------------
    # DELTA
    #
    # doc_001 = DELETED
    # doc_002 = UPDATED
    # doc_003 = NEW
    # --------------------------------------------------------

    # DELETED
    index.delete_document(
        "doc_001"
    )

    # UPDATED
    #
    # Əvvəl doc_002-nin bütün köhnə
    # chunk-larını silirik.
    index.delete_document(
        "doc_002"
    )

    assert index.count() == 0

    # --------------------------------------------------------
    # NEW VERSION OF doc_002
    #
    # Köhnə versiyada 2 chunk var idi.
    # Yeni versiyada yalnız 1 chunk var.
    #
    # Bu test stale chunk qalmasının
    # qarşısını yoxlayır.
    # --------------------------------------------------------

    delta_vectors = np.stack(
        [
            make_vector(10),
            make_vector(20),
        ]
    )

    delta_payloads = [
        {
            "document_id":
                "doc_002",

            "chunk_id":
                "doc_002_chunk_0000",

            "change_type":
                "UPDATED",

            "retrieval_hash":
                "hash_doc_002_v2",

            "title":
                "Updated Document 2",
        },
        {
            "document_id":
                "doc_003",

            "chunk_id":
                "doc_003_chunk_0000",

            "change_type":
                "NEW",

            "retrieval_hash":
                "hash_doc_003_v1",

            "title":
                "New Document 3",
        },
    ]

    written = index.upsert(
        vectors=delta_vectors,
        payloads=delta_payloads,
    )

    assert written == 2

    # --------------------------------------------------------
    # FINAL STATE
    # --------------------------------------------------------

    assert index.count() == 2

    final_state = (
        index.get_chunk_state()
    )

    assert set(
        final_state.keys()
    ) == {
        "doc_002_chunk_0000",
        "doc_003_chunk_0000",
    }

    # doc_001 tam silinib
    assert (
        "doc_001_chunk_0000"
        not in final_state
    )

    # doc_002-nin köhnə ikinci chunk-u
    # qalmamalıdır.
    assert (
        "doc_002_chunk_0001"
        not in final_state
    )

    # UPDATED hash
    assert (
        final_state[
        "doc_002_chunk_0000"
        ]["retrieval_hash"]
        == "hash_doc_002_v2"
    )

    # document IDs
    assert (
        final_state[
            "doc_002_chunk_0000"
        ]["document_id"]
        == "doc_002"
    )

    assert (
        final_state[
            "doc_003_chunk_0000"
        ]["document_id"]
        == "doc_003"
    )