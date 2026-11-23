from typing import Any
from datetime import datetime, timezone
from retrieval.search.policy import ranking_date, route

import numpy as np
import torch

from FlagEmbedding import FlagReranker

from retrieval.config import (
    RERANKER_MODEL_ID,
    RERANKER_QUERY_MAX_LENGTH,
    RERANKER_PASSAGE_MAX_LENGTH,
    RERANKER_USE_FP16,
    RERANKER_DATE_CONTEXT,
    RERANKER_BATCH_SIZE,
)


class BgeReranker:

    def __init__(self):
        self.date_context = RERANKER_DATE_CONTEXT
        self.batch_size = RERANKER_BATCH_SIZE

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        print("=" * 70)
        print("BGE RERANKER")
        print("=" * 70)

        print(
            f"Model   : {RERANKER_MODEL_ID}"
        )

        print(
            f"Device  : {self.device}"
        )

        print(
            f"FP16    : {RERANKER_USE_FP16}"
        )

        print(
            f"Query max length   : "
            f"{RERANKER_QUERY_MAX_LENGTH}"
        )

        print(
            f"Passage max length : "
            f"{RERANKER_PASSAGE_MAX_LENGTH}"
        )

        self.model = FlagReranker(
            RERANKER_MODEL_ID,
            query_max_length=(
                RERANKER_QUERY_MAX_LENGTH
            ),
            passage_max_length=(
                RERANKER_PASSAGE_MAX_LENGTH
            ),
            use_fp16=(
                RERANKER_USE_FP16
            ),
            devices=[
                self.device
            ],
        )

        print(
            "Reranker loaded successfully."
        )

        print("=" * 70)

    # PASSAGE REPRESENTATION

    @staticmethod
    def build_passage(
        payload: dict[str, Any],
    ) -> str:

        title = payload.get(
            "title"
        )

        chunk_text = payload.get(
            "chunk_text"
        )

        title = (
            ""
            if title is None
            else str(title).strip()
        )

        chunk_text = (
            ""
            if chunk_text is None
            else str(chunk_text).strip()
        )

        if title and chunk_text:

            return (
                f"Title: {title}\n\n"
                f"Content:\n{chunk_text}"
            )

        if chunk_text:
            return chunk_text

        if title:
            return title

        raise ValueError(
            "Reranker candidate contains "
            "neither title nor chunk_text."
        )

    # RERANK

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        limit: int = 8,
    ) -> list[dict]:

        query = str(
            query
        ).strip()

        if not query:
            return []

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0."
            )

        if not candidates:
            return []

        pairs = []
        now = datetime.now(timezone.utc)
        use_dates = getattr(self, "date_context", False)
        current = route(query)["temporal_guard"] == "default"
        model_query = query
        if use_dates and current:
            model_query = (f"Current information as of {now.date().isoformat()}: {query}\n"
                           "Find the currently applicable answer; prefer newer relevant evidence over outdated information.")
        date_inputs = []

        for candidate in candidates:

            payload = candidate.get(
                "payload",
                {},
            )

            passage = (
                self.build_passage(
                    payload
                )
            )
            date, field = ranking_date(payload)
            valid_date = date is not None and date <= now
            date_input = None
            if use_dates:
                value = date.isoformat() if valid_date else "unknown"
                date_input = {"date": value, "field": field if valid_date else None,
                              "current_question": current}
                # Put metadata first so passage truncation cannot remove the date.
                passage = f"Document date ({field if valid_date else 'unknown'}): {value}\n\n{passage}"
            date_inputs.append(date_input)

            pairs.append(
                [
                    model_query,
                    passage,
                ]
            )

        scores = (
            self.model.compute_score(
                pairs,
                normalize=True,
                batch_size=getattr(self, "batch_size", RERANKER_BATCH_SIZE),
            )
        )

        if np.isscalar(scores):

            scores = [
                float(scores)
            ]

        else:

            scores = [
                float(score)
                for score in scores
            ]

        if len(scores) != len(
            candidates
        ):

            raise RuntimeError(
                "Reranker score count "
                "does not match candidate count."
            )

        reranked = []

        for candidate, score, date_input in zip(
            candidates,
            scores,
            date_inputs,
        ):

            item = dict(
                candidate
            )

            item[
                "reranker_score"
            ] = score
            if date_input is not None:
                item["reranker_date_context"] = date_input

            reranked.append(
                item
            )

        reranked.sort(
            key=lambda item: (
                item[
                    "reranker_score"
                ]
            ),
            reverse=True,
        )

        final_results = []

        for rank, item in enumerate(
            reranked[:limit],
            start=1,
        ):

            result = dict(
                item
            )

            result[
                "reranker_rank"
            ] = rank

            final_results.append(
                result
            )

        return final_results
