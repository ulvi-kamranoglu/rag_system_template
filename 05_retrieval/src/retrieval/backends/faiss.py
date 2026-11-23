from typing import Any

import faiss
import numpy as np

from retrieval.config import (
    VECTOR_DIMENSION,
)


class FaissVectorIndex:

    def __init__(self):

        # Exact inner-product search.
        #
        # BGE-M3 vectors L2-normalized olduğu üçün
        # inner product ranking cosine similarity
        # ranking ilə eyni olur.

        self.index = faiss.IndexFlatIP(
            VECTOR_DIMENSION
        )

        self.payloads: list[
            dict[str, Any]
        ] = []

    # VECTOR VALIDATION

    @staticmethod
    def validate_vectors(
        vectors: np.ndarray,
    ) -> np.ndarray:

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Vectors must be a 2D matrix. "
                f"Found shape={vectors.shape}"
            )

        if (
            vectors.shape[1]
            != VECTOR_DIMENSION
        ):
            raise ValueError(
                "Unexpected vector dimension. "
                f"Expected={VECTOR_DIMENSION}, "
                f"found={vectors.shape[1]}"
            )

        if not np.isfinite(
            vectors
        ).all():
            raise ValueError(
                "Vectors contain NaN or Inf."
            )

        return np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

    # QUERY VALIDATION

    @staticmethod
    def validate_query_vector(
        query_vector: np.ndarray,
    ) -> np.ndarray:

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if query_vector.ndim != 1:
            raise ValueError(
                "Query vector must be 1D. "
                f"Found shape={query_vector.shape}"
            )

        if (
            query_vector.shape[0]
            != VECTOR_DIMENSION
        ):
            raise ValueError(
                "Unexpected query dimension. "
                f"Expected={VECTOR_DIMENSION}, "
                f"found={query_vector.shape[0]}"
            )

        if not np.isfinite(
            query_vector
        ).all():
            raise ValueError(
                "Query vector contains NaN or Inf."
            )

        return np.ascontiguousarray(
            query_vector.reshape(
                1,
                -1,
            ),
            dtype=np.float32,
        )

    # BUILD

    def build(
        self,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
    ) -> None:

        vectors = self.validate_vectors(
            vectors
        )

        if (
            vectors.shape[0]
            != len(payloads)
        ):
            raise ValueError(
                "Vector/payload count mismatch. "
                f"Vectors={vectors.shape[0]}, "
                f"payloads={len(payloads)}"
            )

        chunk_ids = set()
        clean_payloads = []

        for row_index, payload in enumerate(
            payloads
        ):

            payload = dict(
                payload
            )

            chunk_id = payload.get(
                "chunk_id"
            )

            if chunk_id is None:
                raise ValueError(
                    f"Missing chunk_id at row {row_index}."
                )

            chunk_id = str(
                chunk_id
            ).strip()

            if not chunk_id:
                raise ValueError(
                    f"Empty chunk_id at row {row_index}."
                )

            if chunk_id in chunk_ids:
                raise ValueError(
                    f"Duplicate chunk_id: {chunk_id}"
                )

            chunk_ids.add(
                chunk_id
            )

            payload[
                "chunk_id"
            ] = chunk_id

            clean_payloads.append(
                payload
            )

        # Full rebuild
        self.index = faiss.IndexFlatIP(
            VECTOR_DIMENSION
        )

        self.index.add(
            vectors
        )

        self.payloads = (
            clean_payloads
        )

    # SEARCH

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[dict]:

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        if self.count() == 0:
            return []

        query = (
            self.validate_query_vector(
                query_vector
            )
        )

        limit = min(
            limit,
            self.count(),
        )

        allowed = set(document_ids) if document_ids is not None else None
        if allowed is not None and not allowed:
            return []
        requested = self.count() if allowed is not None else limit
        scores, indexes = (
            self.index.search(
                query,
                requested,
            )
        )

        results = []

        for rank in range(requested):

            vector_index = int(
                indexes[0, rank]
            )

            if vector_index < 0:
                continue
            if allowed is not None and self.payloads[vector_index].get("document_id") not in allowed:
                continue

            results.append(
                {
                    "id":
                        vector_index,

                    "score":
                        float(
                            scores[0, rank]
                        ),

                    "payload":
                        self.payloads[
                            vector_index
                        ],
                }
            )

            if len(results) >= limit:
                break
        return results

    # COUNT

    def count(self) -> int:

        return int(
            self.index.ntotal
        )

    # INFO

    def info(self) -> dict:

        return {
            "backend":
                "faiss",

            "index_type":
                "IndexFlatIP",

            "dimension":
                VECTOR_DIMENSION,

            "count":
                self.count(),

            "metric":
                "inner_product",
        }
    # SAVE

    def save(
        self,
        path,
    ) -> None:

        faiss.write_index(
            self.index,
            str(path),
        )

    # LOAD

    def load(
        self,
        path,
        payloads: list[
            dict[str, Any]
        ],
    ) -> None:

        self.index = faiss.read_index(
            str(path)
        )

        if (
            self.index.d
            != VECTOR_DIMENSION
        ):
            raise ValueError(
                "Loaded FAISS index has "
                "unexpected dimension. "
                f"Expected={VECTOR_DIMENSION}, "
                f"found={self.index.d}"
            )

        if (
            self.index.ntotal
            != len(payloads)
        ):
            raise ValueError(
                "FAISS index/payload "
                "count mismatch. "
                f"Index={self.index.ntotal}, "
                f"payloads={len(payloads)}"
            )

        self.payloads = [
            dict(payload)
            for payload in payloads
        ]
