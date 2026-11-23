import threading

import numpy as np
import torch

from FlagEmbedding import BGEM3FlagModel

from embedding.config import load_settings


class BgeM3Embedder:

    def __init__(self, settings=None):
        self.settings = settings if settings is not None else load_settings()

        # ====================================================
        # DEVICE
        # ====================================================

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        print("=" * 70)
        print("BGE-M3 EMBEDDER")
        print("=" * 70)

        print(
            f"Model     : {self.settings.model_id}"
        )

        print(
            f"Device    : {self.device}"
        )

        print(
            f"FP16      : {self.settings.use_fp16}"
        )

        print(
            f"Dimension : "
            f"{self.settings.embedding_dimension}"
        )

        print(
            f"Max length: "
            f"{self.settings.document_max_length}"
        )

        # ====================================================
        # MODEL
        # ====================================================

        try:

            self.model = BGEM3FlagModel(
                self.settings.model_id,
                use_fp16=self.settings.use_fp16,
                devices=self.device,
            )

        except TypeError:

            # FlagEmbedding versiyaları arasında
            # compatibility üçün fallback.
            self.model = BGEM3FlagModel(
                self.settings.model_id,
                use_fp16=self.settings.use_fp16,
            )

        # Eyni modelə paralel çağırışlarda
        # təhlükəsizlik üçün.
        self._model_lock = (
            threading.Lock()
        )

        print(
            "Model loaded successfully."
        )

        # Modeli ilk real batch-dən əvvəl
        # bir dəfə işə salırıq.
        self._warmup()

        print("=" * 70)

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_vectors(
        vectors: np.ndarray,
    ) -> np.ndarray:

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        norms = np.maximum(
            norms,
            1e-12,
        )

        normalized = (
            vectors
            / norms
        )

        return normalized

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_vectors(
        self,
        vectors: np.ndarray,
        expected_rows: int,
    ) -> None:

        if vectors.ndim != 2:

            raise RuntimeError(
                "Unexpected embedding "
                f"shape: {vectors.shape}"
            )

        if (
            vectors.shape[0]
            != expected_rows
        ):

            raise RuntimeError(
                "Embedding row count "
                "mismatch. "
                f"Expected={expected_rows}, "
                f"found={vectors.shape[0]}"
            )

        if (
            vectors.shape[1]
            != self.settings.embedding_dimension
        ):

            raise RuntimeError(
                "Unexpected embedding "
                "dimension. "
                f"Expected="
                f"{self.settings.embedding_dimension}, "
                f"found="
                f"{vectors.shape[1]}"
            )

        if not np.isfinite(
            vectors
        ).all():

            raise RuntimeError(
                "Embeddings contain "
                "NaN or Inf values."
            )

    # ========================================================
    # WARM-UP
    # ========================================================

    def _warmup(self) -> None:

        print(
            "Running model warm-up..."
        )

        sample = [
            (
                "Title: UNEC\n\n"
                "Content:\n"
                "Azərbaycan Dövlət "
                "İqtisad Universiteti."
            )
        ]

        with self._model_lock:

            output = (
                self.model.encode_corpus(
                    sample,
                    batch_size=1,
                    max_length=(
                        self.settings.document_max_length
                    ),
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            )

        vectors = np.asarray(
            output["dense_vecs"],
            dtype=np.float32,
        )

        self._validate_vectors(
            vectors=vectors,
            expected_rows=1,
        )

        print(
            "Warm-up OK."
        )

    # ========================================================
    # DOCUMENT / CHUNK EMBEDDING
    # ========================================================

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:

        if not texts:

            return np.empty(
                (
                    0,
                    self.settings.embedding_dimension,
                ),
                dtype=np.float32,
            )

        cleaned_texts = []

        for index, text in enumerate(
            texts
        ):

            if text is None:

                raise ValueError(
                    "Embedding text "
                    f"at index {index} "
                    "is None."
                )

            text = str(text).strip()

            if not text:

                raise ValueError(
                    "Embedding text "
                    f"at index {index} "
                    "is empty."
                )

            cleaned_texts.append(
                text
            )

        if batch_size is None:

            batch_size = (
                self.settings.document_batch_size
            )

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be "
                "greater than 0."
            )

        with self._model_lock:

            output = (
                self.model.encode_corpus(
                    cleaned_texts,
                    batch_size=batch_size,
                    max_length=(
                        self.settings.document_max_length
                    ),
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            )

            if torch.cuda.is_available():

                torch.cuda.synchronize()

        vectors = np.asarray(
            output["dense_vecs"],
            dtype=np.float32,
        )

        self._validate_vectors(
            vectors=vectors,
            expected_rows=len(
                cleaned_texts
            ),
        )

        # Qdrant cosine və ya
        # inner-product search üçün
        # explicit L2 normalization.
        vectors = (
            self._normalize_vectors(
                vectors
            )
        )

        return vectors

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def embed_queries(
        self,
        queries: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:

        if not queries:

            return np.empty(
                (
                    0,
                    self.settings.embedding_dimension,
                ),
                dtype=np.float32,
            )

        cleaned_queries = []

        for index, query in enumerate(
            queries
        ):

            if query is None:

                raise ValueError(
                    "Query at index "
                    f"{index} is None."
                )

            query = str(
                query
            ).strip()

            if not query:

                raise ValueError(
                    "Query at index "
                    f"{index} is empty."
                )

            cleaned_queries.append(
                query
            )

        if batch_size is None:

            batch_size = (
                self.settings.document_batch_size
            )

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be "
                "greater than 0."
            )

        with self._model_lock:

            output = (
                self.model.encode_queries(
                    cleaned_queries,
                    batch_size=batch_size,
                    max_length=(
                        self.settings.query_max_length
                    ),
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            )

            if torch.cuda.is_available():

                torch.cuda.synchronize()

        vectors = np.asarray(
            output["dense_vecs"],
            dtype=np.float32,
        )

        self._validate_vectors(
            vectors=vectors,
            expected_rows=len(
                cleaned_queries
            ),
        )

        vectors = (
            self._normalize_vectors(
                vectors
            )
        )

        return vectors

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:

        vectors = self.embed_queries(
            [query]
        )

        return vectors[0]

    # ========================================================
    # SINGLE TEXT
    # ========================================================

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:

        vectors = self.embed_texts(
            [text]
        )

        return vectors[0]
