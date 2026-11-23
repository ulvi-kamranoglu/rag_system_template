from typing import Any


class ReciprocalRankFusion:

    def __init__(
        self,
        k: int = 60,
    ):

        if k <= 0:
            raise ValueError(
                "RRF k must be greater than 0."
            )

        self.k = k

    # FUSION

    def fuse(
        self,
        dense_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        limit: int = 20,
    ) -> list[dict]:

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        candidates: dict[str, dict] = {}

        # DENSE

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):

            payload = result.get(
                "payload",
                {},
            )

            chunk_id = payload.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            chunk_id = str(
                chunk_id
            )

            rrf_score = (
                1.0
                / (self.k + rank)
            )

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "chunk_id":
                        chunk_id,

                    "payload":
                        payload,

                    "dense_rank":
                        None,

                    "dense_score":
                        None,

                    "bm25_rank":
                        None,

                    "bm25_score":
                        None,

                    "rrf_score":
                        0.0,
                }

            candidate = (
                candidates[
                    chunk_id
                ]
            )

            candidate[
                "dense_rank"
            ] = rank

            candidate[
                "dense_score"
            ] = result.get(
                "score"
            )

            candidate[
                "rrf_score"
            ] += rrf_score

        # BM25

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            chunk_id = result.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            chunk_id = str(
                chunk_id
            )

            payload = result.get(
                "payload",
                {},
            )

            rrf_score = (
                1.0
                / (self.k + rank)
            )

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "chunk_id":
                        chunk_id,

                    "payload":
                        payload,

                    "dense_rank":
                        None,

                    "dense_score":
                        None,

                    "bm25_rank":
                        None,

                    "bm25_score":
                        None,

                    "rrf_score":
                        0.0,
                }

            candidate = (
                candidates[
                    chunk_id
                ]
            )

            candidate[
                "bm25_rank"
            ] = result.get(
                "rank",
                rank,
            )

            candidate[
                "bm25_score"
            ] = result.get(
                "score"
            )

            candidate[
                "rrf_score"
            ] += rrf_score

            # Dense-də yox idisə,
            # metadata BM25-dən gəlir.
            if not candidate[
                "payload"
            ]:

                candidate[
                    "payload"
                ] = payload

        # SORT

        fused = sorted(
            candidates.values(),
            key=lambda item: (
                item[
                    "rrf_score"
                ],
                -(
                    item[
                        "dense_rank"
                    ]
                    or 10**9
                ),
                -(
                    item[
                        "bm25_rank"
                    ]
                    or 10**9
                ),
            ),
            reverse=True,
        )

        # Final rank
        final_results = []

        for rank, result in enumerate(
            fused[:limit],
            start=1,
        ):

            item = dict(
                result
            )

            item[
                "hybrid_rank"
            ] = rank

            final_results.append(
                item
            )

        return final_results
