import pytest

pytestmark = pytest.mark.integration

from retrieval.hybrid_retriever import (
    HybridRetriever,
)


def test_full_hybrid_retrieval():

    query = (
        "UNEC-də magistratura "
        "proqramları varmı?"
    )

    retriever = HybridRetriever()

    try:

        results = retriever.search(
            query=query,
            dense_limit=3,
            bm25_limit=3,
            reranker_input_limit=3,
            final_limit=3,
        )

        assert len(results) == 3

        top = results[0]

        assert (
            top["chunk_id"]
            == "doc_test_002_chunk_0000"
        )

        assert (
            top["payload"]["title"]
            == "UNEC magistratura"
        )

        assert (
            top["dense_rank"]
            == 1
        )

        assert (
            top["bm25_rank"]
            == 1
        )

        assert (
            top["hybrid_rank"]
            == 1
        )

        assert (
            top["reranker_rank"]
            == 1
        )

        assert (
            0.0
            <= top["reranker_score"]
            <= 1.0
        )

    finally:

        retriever.close()


def test_empty_query():

    retriever = HybridRetriever()

    try:

        results = retriever.search(
            "   "
        )

        assert results == []

    finally:

        retriever.close()


def test_hybrid_candidate_search():

    retriever = HybridRetriever()

    try:

        results = (
            retriever.hybrid_search(
                query=(
                    "magistratura proqramları"
                ),
                dense_limit=3,
                bm25_limit=3,
                hybrid_limit=3,
            )
        )

        assert len(results) == 3

        assert (
            results[0]["chunk_id"]
            == "doc_test_002_chunk_0000"
        )

        assert (
            results[0]["hybrid_rank"]
            == 1
        )

    finally:

        retriever.close()