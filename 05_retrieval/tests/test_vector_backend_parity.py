import pytest

pytestmark = pytest.mark.integration

import numpy as np
import polars as pl

from embedding.embedder import (
    BgeM3Embedder,
)

from retrieval.vector_index import (
    QdrantVectorIndex,
)

from retrieval.faiss_index import (
    FaissVectorIndex,
)

from retrieval.pipeline import (
    build_payloads,
)

from retrieval.config import (
    FAISS_INDEX_PATH,
    FAISS_MAPPING_PATH,
)


def test_qdrant_faiss_dense_parity():

    query = (
        "UNEC-də magistratura "
        "proqramları varmı?"
    )

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    embedder = BgeM3Embedder()

    query_vector = (
        embedder.embed_query(
            query
        )
    )

    # ========================================================
    # QDRANT
    # ========================================================

    qdrant = QdrantVectorIndex()

    try:

        qdrant.health_check()

        qdrant_results = (
            qdrant.search(
                query_vector=query_vector,
                limit=3,
            )
        )

    finally:

        qdrant.close()

    # ========================================================
    # FAISS
    # ========================================================

    mapping = pl.read_parquet(
        FAISS_MAPPING_PATH
    )

    payloads = build_payloads(
        mapping
    )

    faiss_index = (
        FaissVectorIndex()
    )

    faiss_index.load(
        path=FAISS_INDEX_PATH,
        payloads=payloads,
    )

    faiss_results = (
        faiss_index.search(
            query_vector=query_vector,
            limit=3,
        )
    )

    # ========================================================
    # RESULT COUNT
    # ========================================================

    assert (
        len(qdrant_results)
        == len(faiss_results)
        == 3
    )

    # ========================================================
    # CHUNK RANKING
    # ========================================================

    qdrant_chunk_ids = [
        result["payload"]["chunk_id"]
        for result in qdrant_results
    ]

    faiss_chunk_ids = [
        result["payload"]["chunk_id"]
        for result in faiss_results
    ]

    assert (
        qdrant_chunk_ids
        == faiss_chunk_ids
    )

    # ========================================================
    # SCORES
    # ========================================================

    qdrant_scores = np.array(
        [
            result["score"]
            for result in qdrant_results
        ],
        dtype=np.float32,
    )

    faiss_scores = np.array(
        [
            result["score"]
            for result in faiss_results
        ],
        dtype=np.float32,
    )

    assert np.allclose(
        qdrant_scores,
        faiss_scores,
        atol=1e-5,
    )

    # ========================================================
    # EXPECTED TOP-1
    # ========================================================

    assert (
        qdrant_chunk_ids[0]
        == "doc_test_002_chunk_0000"
    )

    assert (
        faiss_chunk_ids[0]
        == "doc_test_002_chunk_0000"
    )