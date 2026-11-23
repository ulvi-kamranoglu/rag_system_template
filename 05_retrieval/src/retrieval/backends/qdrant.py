import uuid
from typing import Any

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    PointStruct,
    PointIdsList,
    VectorParams,
    PayloadSchemaType,
    QueryRequest,
)

from retrieval.config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    VECTOR_DIMENSION,
    QDRANT_BATCH_SIZE,
)


class QdrantVectorIndex:

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION,
    ):

        self.url = url
        self.collection_name = (
            collection_name
        )

        self.client = QdrantClient(
            url=self.url
        )

    # HEALTH

    def health_check(self) -> bool:

        try:
            self.client.get_collections()
            return True

        except Exception as exc:

            raise RuntimeError(
                "Qdrant connection failed. "
                f"URL={self.url}"
            ) from exc

    # COLLECTION

    def collection_exists(self) -> bool:

        return self.client.collection_exists(
            collection_name=(
                self.collection_name
            )
        )

    def ensure_collection(self) -> None:

        if self.collection_exists():
            self.ensure_payload_indexes()
            return

        self.client.create_collection(
            collection_name=(
                self.collection_name
            ),
            vectors_config=VectorParams(
                size=VECTOR_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        self.ensure_payload_indexes()

    def ensure_payload_indexes(self) -> None:
        """Prepare document filters once, never in the per-query hot path."""
        schema = self.client.get_collection(self.collection_name).payload_schema
        field = schema.get("document_id")
        if field is not None:
            if field.data_type != PayloadSchemaType.KEYWORD:
                raise ValueError("Qdrant document_id payload index must use keyword type")
            return
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def recreate_collection(self) -> None:

        if self.collection_exists():

            self.client.delete_collection(
                collection_name=(
                    self.collection_name
                )
            )

        self.client.create_collection(
            collection_name=(
                self.collection_name
            ),
            vectors_config=VectorParams(
                size=VECTOR_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        self.ensure_payload_indexes()

    # POINT ID

    @staticmethod
    def make_point_id(
        chunk_id: str,
    ) -> str:

        if not chunk_id:

            raise ValueError(
                "chunk_id cannot be empty."
            )

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk_id,
            )
        )

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
                "Vectors must be a "
                "2-dimensional matrix. "
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
                "Vectors contain NaN "
                "or Inf values."
            )

        return vectors

    # UPSERT

    def upsert(
        self,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
        batch_size: int = QDRANT_BATCH_SIZE,
    ) -> int:

        vectors = self.validate_vectors(
            vectors
        )

        if (
            len(payloads)
            != vectors.shape[0]
        ):

            raise ValueError(
                "Payload/vector count mismatch. "
                f"Vectors={vectors.shape[0]}, "
                f"payloads={len(payloads)}"
            )

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be "
                "greater than 0."
            )

        self.ensure_collection()

        total = vectors.shape[0]

        for start in range(
            0,
            total,
            batch_size,
        ):

            end = min(
                start + batch_size,
                total,
            )

            points = []

            for row_index in range(
                start,
                end,
            ):

                payload = dict(
                    payloads[row_index]
                )

                chunk_id = payload.get(
                    "chunk_id"
                )

                if not chunk_id:

                    raise ValueError(
                        "Every payload must "
                        "contain chunk_id."
                    )

                point_id = (
                    self.make_point_id(
                        str(chunk_id)
                    )
                )

                point = PointStruct(
                    id=point_id,
                    vector=(
                        vectors[
                            row_index
                        ].tolist()
                    ),
                    payload=payload,
                )

                points.append(
                    point
                )

            self.client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=points,
                wait=True,
            )

            print(
                f"Indexed "
                f"{end:,}/{total:,}"
            )

        return total

    # DELETE DOCUMENT

    def delete_document(
        self,
        document_id: str,
    ) -> None:

        if not document_id:

            raise ValueError(
                "document_id cannot "
                "be empty."
            )

        if not self.collection_exists():
            return

        self.client.delete(
            collection_name=(
                self.collection_name
            ),
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            ),
            wait=True,
        )

    # COUNT

    def count(self) -> int:

        if not self.collection_exists():
            return 0

        result = self.client.count(
            collection_name=(
                self.collection_name
            ),
            exact=True,
        )

        return int(
            result.count
        )

    # COLLECTION INFO

    def info(self):

        if not self.collection_exists():

            return {
                "exists": False,
                "collection":
                    self.collection_name,
                "count": 0,
            }

        collection = (
            self.client.get_collection(
                collection_name=(
                    self.collection_name
                )
            )
        )

        return {
            "exists": True,
            "collection":
                self.collection_name,
            "count":
                self.count(),
            "status":
                str(collection.status),
        }
    # SEMANTIC SEARCH

    @staticmethod
    def _query_vector(query_vector):
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"Query vector must be 1D. Found shape={vector.shape}")
        if vector.shape[0] != VECTOR_DIMENSION:
            raise ValueError(f"Unexpected query vector dimension. Expected={VECTOR_DIMENSION}, found={vector.shape[0]}")
        if not np.isfinite(vector).all():
            raise ValueError("Query vector contains NaN or Inf values.")
        return vector.tolist()

    @staticmethod
    def _hits(result):
        return [{"id": str(point.id), "score": float(point.score), "payload": point.payload or {}}
                for point in result.points]

    def search_documents(self, query_vector, document_ids):
        """Keep independent per-document queries, sharing a single HTTP request."""
        vector = self._query_vector(query_vector)
        documents = list(dict.fromkeys(document_ids))
        if not documents:
            return {}
        requests = [QueryRequest(query=vector, limit=1, with_payload=True,
                    filter=Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=[doc]))]))
                    for doc in documents]
        try:
            results = self.client.query_batch_points(collection_name=self.collection_name, requests=requests)
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                raise RuntimeError(f"Qdrant collection does not exist: {self.collection_name}") from exc
            raise
        if len(results) != len(documents):
            raise RuntimeError("Qdrant document search response count mismatch")
        return {doc: self._hits(result) for doc, result in zip(documents, results)}

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

        vector = self._query_vector(query_vector)

        if document_ids is not None and not document_ids:
            return []
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                query_filter=(Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=document_ids))])
                              if document_ids is not None else None),
            )
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                raise RuntimeError(f"Qdrant collection does not exist: {self.collection_name}") from exc
            raise

        return self._hits(result)
    # CONNECTION

    def close(self) -> None:

        self.client.close()

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()

    # EXISTING CHUNK STATE

    def get_chunk_state(
        self,
        batch_size: int = 256,
    ) -> dict[str, dict]:

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if not self.collection_exists():
            return {}

        state = {}

        offset = None

        while True:

            points, next_offset = (
                self.client.scroll(
                    collection_name=(
                        self.collection_name
                    ),
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            )

            for point in points:

                payload = (
                    point.payload or {}
                )

                chunk_id = payload.get(
                    "chunk_id"
                )

                if not chunk_id:
                    continue

                chunk_id = str(
                    chunk_id
                )

                state[chunk_id] = {
                    "point_id":
                        str(point.id),

                    "chunk_id":
                        chunk_id,

                    "document_id":
                        payload.get(
                            "document_id"
                        ),

                    "retrieval_hash":
                        payload.get(
                            "retrieval_hash"
                        ),

                    "content_hash":
                        payload.get(
                            "content_hash"
                        ),
                }

            if next_offset is None:
                break

            offset = next_offset

        return state

    # DELETE POINT IDS

    def delete_point_ids(
        self,
        point_ids: list[str],
    ) -> int:

        if not point_ids:
            return 0

        if not self.collection_exists():
            return 0

        self.client.delete(
            collection_name=(
                self.collection_name
            ),
            points_selector=PointIdsList(
                points=point_ids
            ),
            wait=True,
        )

        return len(
            point_ids
        )

    # DELETE CHUNKS

    def delete_chunks(
        self,
        chunk_ids: list[str],
    ) -> int:

        if not chunk_ids:
            return 0

        point_ids = [
            self.make_point_id(
                str(chunk_id)
            )
            for chunk_id in chunk_ids
        ]

        return self.delete_point_ids(
            point_ids
        )
