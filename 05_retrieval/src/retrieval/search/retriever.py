import polars as pl

from embedding.embedder import (
    BgeM3Embedder,
)

from retrieval.backends.bm25 import (
    BM25ChunkIndex,
)

from retrieval.search.fusion import (
    ReciprocalRankFusion,
)

from retrieval.search.reranker import (
    BgeReranker,
)

from retrieval.backends.qdrant import (
    QdrantVectorIndex,
)

from retrieval.backends.faiss import (
    FaissVectorIndex,
)

from retrieval.artifacts import (
    build_payloads,
)

from retrieval.config import (
    BM25_CORPUS_PATH,
    DENSE_TOP_K,
    BM25_TOP_K,
    HYBRID_TOP_K,
    RRF_K,
    RERANKER_INPUT_TOP_K,
    FINAL_TOP_K,
    VECTOR_BACKEND,
    FAISS_INDEX_PATH,
    FAISS_MAPPING_PATH,
    RANKING_POLICY,
)


class HybridRetriever:

    def __init__(self, settings=None, *, embedder=None, reranker=None):
        from retrieval.config import load_settings
        self.settings = settings if settings is not None else load_settings()
        self.ranking_policy = dict(self.settings["RANKING_POLICY"])

        print("=" * 70)
        print("HYBRID RETRIEVER")
        print("=" * 70)

        # EMBEDDING MODEL

        print(
            "Loading query embedder..."
        )

        self.embedder = (
            embedder if embedder is not None else BgeM3Embedder()
        )

        # DENSE VECTOR BACKEND

        print(
            f"Loading vector backend: "
            f"{self.settings['VECTOR_BACKEND']}"
        )

        self.vector_index = (
            self._load_vector_backend(self.settings)
        )

        # BM25

        print(
            "Loading BM25 corpus..."
        )

        self.bm25_index = (
            self._load_bm25_index(self.settings["BM25_CORPUS_PATH"])
        )
        from retrieval.search.recency import RecentDocumentIndex
        self.recent_index = RecentDocumentIndex(self.bm25_index.records)

        # RRF

        print(
            "Initializing RRF fusion..."
        )

        self.fusion = (
            ReciprocalRankFusion(
                k=self.settings["RRF_K"]
            )
        )

        # RERANKER

        print(
            "Loading reranker..."
        )

        self.reranker = (
            reranker if reranker is not None else BgeReranker()
        )

        print(
            "Hybrid retriever ready."
        )

        print("=" * 70)

    # BM25 LOAD

    @staticmethod
    def _load_bm25_index(
        corpus_path=None,
    ) -> BM25ChunkIndex:

        BM25_CORPUS_PATH = corpus_path if corpus_path is not None else globals()["BM25_CORPUS_PATH"]

        if not BM25_CORPUS_PATH.exists():

            raise FileNotFoundError(
                "BM25 corpus not found:\n"
                f"{BM25_CORPUS_PATH}"
            )

        df = pl.read_parquet(
            BM25_CORPUS_PATH
        )

        if df.height == 0:

            raise ValueError(
                "BM25 corpus is empty."
            )

        required_columns = {
            "document_id",
            "chunk_id",
            "title",
            "chunk_text",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:

            raise ValueError(
                "BM25 corpus is missing "
                f"columns: {sorted(missing)}"
            )

        records = (
            df
            .rows(
                named=True
            )
        )

        index = BM25ChunkIndex(
            k1=1.5,
            b=0.75,
        )

        index.build(
            records
        )

        return index

    # DENSE SEARCH

    def dense_search(
        self,
        query: str,
        limit: int = DENSE_TOP_K,
    ) -> list[dict]:

        query = str(
            query
        ).strip()

        if not query:
            return []

        query_vector = (
            self.embedder.embed_query(
                query
            )
        )

        return (
            self.vector_index.search(
                query_vector=query_vector,
                limit=limit,
            )
        )

    # BM25 SEARCH

    def bm25_search(
        self,
        query: str,
        limit: int = BM25_TOP_K,
    ) -> list[dict]:

        query = str(
            query
        ).strip()

        if not query:
            return []

        return (
            self.bm25_index.search(
                query=query,
                limit=limit,
            )
        )

    # HYBRID CANDIDATE RETRIEVAL

    def hybrid_search(
        self,
        query: str,
        dense_limit: int = DENSE_TOP_K,
        bm25_limit: int = BM25_TOP_K,
        hybrid_limit: int = HYBRID_TOP_K,
    ) -> list[dict]:

        query = str(
            query
        ).strip()

        if not query:
            return []

        # DENSE

        dense_results = (
            self.dense_search(
                query=query,
                limit=dense_limit,
            )
        )

        # BM25

        bm25_results = (
            self.bm25_search(
                query=query,
                limit=bm25_limit,
            )
        )

        # RRF

        hybrid_results = (
            self.fusion.fuse(
                dense_results=(
                    dense_results
                ),
                bm25_results=(
                    bm25_results
                ),
                limit=hybrid_limit,
            )
        )

        return hybrid_results

    # FINAL SEARCH

    def search(
        self,
        query: str,
        dense_limit: int = DENSE_TOP_K,
        bm25_limit: int = BM25_TOP_K,
        reranker_input_limit: int = RERANKER_INPUT_TOP_K,
        final_limit: int = FINAL_TOP_K,
    ) -> list[dict]:

        query = str(
            query
        ).strip()

        if not query:
            return []

        return self.trace_search(query, dense_limit=dense_limit, bm25_limit=bm25_limit,
                                 reranker_input_limit=reranker_input_limit,
                                 final_limit=final_limit)["stages"]["final"]

    def trace_search(self, query, *, dense_limit=DENSE_TOP_K, bm25_limit=BM25_TOP_K,
                     reranker_input_limit=RERANKER_INPUT_TOP_K, final_limit=FINAL_TOP_K):
        from retrieval.search.flow import run_trace
        return run_trace(self, query, dense_limit=dense_limit, bm25_limit=bm25_limit,
                         reranker_input_limit=reranker_input_limit, final_limit=final_limit)

    def _legacy_search(self, query, dense_limit=DENSE_TOP_K, bm25_limit=BM25_TOP_K,
                       reranker_input_limit=RERANKER_INPUT_TOP_K, final_limit=FINAL_TOP_K):

        if reranker_input_limit <= 0:

            raise ValueError(
                "reranker_input_limit must "
                "be greater than 0."
            )

        if final_limit <= 0:

            raise ValueError(
                "final_limit must "
                "be greater than 0."
            )

        # DENSE + BM25 + RRF

        candidates = (
            self.hybrid_search(
                query=query,
                dense_limit=dense_limit,
                bm25_limit=bm25_limit,
                hybrid_limit=(
                    reranker_input_limit
                ),
            )
        )

        if not candidates:
            return []

        # RERANKER

        final_results = (
            self.reranker.rerank(
                query=query,
                candidates=candidates,
                limit=min(
                    final_limit,
                    len(candidates),
                ),
            )
        )

        return final_results

    # INFO

    def info(self) -> dict:

        return {
            "ranking_policy": self.ranking_policy,
            "vector_backend":
                self.settings["VECTOR_BACKEND"],

            "dense_points":
                self.vector_index.count(),

            "bm25_chunks":
                self.bm25_index.count(),

            "dense_top_k":
                DENSE_TOP_K,

            "bm25_top_k":
                BM25_TOP_K,

            "hybrid_top_k":
                HYBRID_TOP_K,

            "rrf_k":
                RRF_K,

            "reranker_input_top_k":
                RERANKER_INPUT_TOP_K,

            "final_top_k":
                FINAL_TOP_K,
        }

    # CONNECTION

    def close(self) -> None:

        if self.settings["VECTOR_BACKEND"] == "qdrant":
            self.vector_index.close()

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()

    # VECTOR BACKEND LOAD

    @staticmethod
    def _load_vector_backend(settings=None):
        from retrieval.config import load_settings
        selected = settings if settings is not None else load_settings()
        VECTOR_BACKEND = selected["VECTOR_BACKEND"]
        FAISS_INDEX_PATH = selected["FAISS_INDEX_PATH"]
        FAISS_MAPPING_PATH = selected["FAISS_MAPPING_PATH"]

        # ----------------------------------------------------
        # QDRANT
        # ----------------------------------------------------

        if VECTOR_BACKEND == "qdrant":

            index = QdrantVectorIndex(url=selected["QDRANT_URL"], collection_name=selected["QDRANT_COLLECTION"])

            index.health_check()

            if not index.collection_exists():

                raise RuntimeError(
                    "Qdrant collection "
                    "does not exist."
                )
            index.ensure_payload_indexes()
            return index

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        if VECTOR_BACKEND == "faiss":

            if not FAISS_INDEX_PATH.exists():

                raise FileNotFoundError(
                    "FAISS index not found:\n"
                    f"{FAISS_INDEX_PATH}"
                )

            if not FAISS_MAPPING_PATH.exists():

                raise FileNotFoundError(
                    "FAISS mapping not found:\n"
                    f"{FAISS_MAPPING_PATH}"
                )

            mapping = pl.read_parquet(
                FAISS_MAPPING_PATH
            )

            payloads = build_payloads(
                mapping
            )

            index = FaissVectorIndex()

            index.load(
                path=FAISS_INDEX_PATH,
                payloads=payloads,
            )

            return index

        raise ValueError(
            "Unsupported VECTOR_BACKEND: "
            f"{VECTOR_BACKEND}. "
            "Supported values: "
            "'qdrant', 'faiss'."
        )
