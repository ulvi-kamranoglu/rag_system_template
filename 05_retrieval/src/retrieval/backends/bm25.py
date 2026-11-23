import re
from typing import Any

import bm25s


class BM25ChunkIndex:

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):

        if k1 <= 0:
            raise ValueError(
                "k1 must be greater than 0."
            )

        if not 0 <= b <= 1:
            raise ValueError(
                "b must be between 0 and 1."
            )

        self.k1 = k1
        self.b = b

        self.records: list[
            dict[str, Any]
        ] = []

        self.index_texts: list[str] = []

        self.tokenizer = (
            bm25s.tokenization.Tokenizer(
                stemmer=None,
                stopwords=None,
                splitter=lambda text: re.findall(
                    r"(?u)\b\w+\b",
                    text,
                ),
            )
        )

        self.retriever = None

    # NORMALIZATION

    @staticmethod
    def normalize_text(
        text: str | None,
    ) -> str:

        if text is None:
            return ""

        text = str(text)

        text = text.casefold()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # DOCUMENT REPRESENTATION

    @classmethod
    def build_document_text(
        cls,
        title: str | None,
        chunk_text: str | None,
    ) -> str:

        title = cls.normalize_text(
            title
        )

        chunk_text = cls.normalize_text(
            chunk_text
        )

        # Title weight:
        # title is intentionally repeated twice.
        parts = []

        if title:

            parts.append(title)
            parts.append(title)

        if chunk_text:

            parts.append(chunk_text)

        document_text = " ".join(
            parts
        ).strip()

        if not document_text:
            raise ValueError(
                "BM25 document text "
                "cannot be empty."
            )

        return document_text

    # BUILD INDEX

    def build(
        self,
        records: list[
            dict[str, Any]
        ],
    ) -> None:

        if not records:
            raise ValueError(
                "Cannot build BM25 index "
                "from an empty corpus."
            )

        chunk_ids = set()

        clean_records = []
        index_texts = []

        for row_index, record in enumerate(
            records
        ):

            record = dict(record)

            chunk_id = record.get(
                "chunk_id"
            )

            if chunk_id is None:

                raise ValueError(
                    "Missing chunk_id at "
                    f"row {row_index}."
                )

            chunk_id = str(
                chunk_id
            ).strip()

            if not chunk_id:

                raise ValueError(
                    "Empty chunk_id at "
                    f"row {row_index}."
                )

            if chunk_id in chunk_ids:

                raise ValueError(
                    "Duplicate chunk_id: "
                    f"{chunk_id}"
                )

            chunk_ids.add(
                chunk_id
            )

            record["chunk_id"] = (
                chunk_id
            )

            document_id = record.get(
                "document_id"
            )

            if document_id is None:

                raise ValueError(
                    "Missing document_id at "
                    f"row {row_index}."
                )

            document_id = str(
                document_id
            ).strip()

            if not document_id:

                raise ValueError(
                    "Empty document_id at "
                    f"row {row_index}."
                )

            record["document_id"] = (
                document_id
            )

            bm25_text = (
                self.build_document_text(
                    title=record.get(
                        "title"
                    ),
                    chunk_text=record.get(
                        "chunk_text"
                    ),
                )
            )

            clean_records.append(
                record
            )

            index_texts.append(
                bm25_text
            )

        # TOKENIZE CORPUS

        corpus_tokens = (
            self.tokenizer.tokenize(
                index_texts,
                update_vocab=True,
                return_as="tuple",
            )
        )

        # BM25

        self.retriever = bm25s.BM25(
            k1=self.k1,
            b=self.b,
            corpus=clean_records,
        )

        self.retriever.index(
            corpus_tokens
        )

        self.records = clean_records
        self.index_texts = index_texts

    # SEARCH

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        if self.retriever is None:
            raise RuntimeError(
                "BM25 index has not "
                "been built."
            )

        if limit <= 0:
            raise ValueError(
                "limit must be "
                "greater than 0."
            )

        query = self.normalize_text(
            query
        )

        if not query:
            return []

        limit = min(
            limit,
            len(self.records),
        )

        query_tokens = (
            self.tokenizer.tokenize(
                [query],
                update_vocab=False,
                return_as="tuple",
            )
        )

        documents, scores = (
            self.retriever.retrieve(
                query_tokens=query_tokens,
                k=limit,
                sorted=True,
                return_as="tuple",
            )
        )

        results = []

        for rank in range(limit):

            document = documents[
                0,
                rank,
            ]

            score = float(
                scores[
                    0,
                    rank,
                ]
            )

            # Depending on bm25s version,
            # corpus dictionaries may come
            # back as numpy object wrappers.
            if hasattr(
                document,
                "item",
            ):

                try:
                    document = (
                        document.item()
                    )
                except ValueError:
                    pass

            if not isinstance(
                document,
                dict,
            ):

                raise RuntimeError(
                    "Unexpected BM25 "
                    "result type: "
                    f"{type(document)}"
                )

            results.append(
                {
                    "rank":
                        rank + 1,

                    "score":
                        score,

                    "chunk_id":
                        document.get(
                            "chunk_id"
                        ),

                    "document_id":
                        document.get(
                            "document_id"
                        ),

                    "payload":
                        document,
                }
            )

        return results

    # INFO

    def count(self) -> int:

        return len(
            self.records
        )

    def info(self) -> dict:

        return {
            "documents":
                self.count(),

            "k1":
                self.k1,

            "b":
                self.b,

            "title_weight":
                2,
        }
