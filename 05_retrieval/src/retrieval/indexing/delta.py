import json
import time
from datetime import datetime, timezone

import numpy as np
import polars as pl
from retrieval import config as retrieval_config
from retrieval.indexing.storage import archived_stage
from pipeline_storage import write_json

from retrieval.config import (
    EMBEDDINGS_PATH,
    MAPPING_PATH,
    BM25_CORPUS_PATH,
    DELETED_DOCUMENTS_PATH,
    DELTA_SYNC_METADATA_PATH,
    INDEXES_DIR,
    QDRANT_BATCH_SIZE,
)

from retrieval.artifacts import (
    validate_artifacts,
    build_payloads,
)

from retrieval.backends.qdrant import (
    QdrantVectorIndex,
)


VALID_CHANGE_TYPES = {
    "NEW",
    "UPDATED",
}


# DELTA VALIDATION

def validate_delta_mapping(
    mapping: pl.DataFrame,
) -> None:

    required_columns = {
        "document_id",
        "chunk_id",
        "change_type",
        "retrieval_hash",
    }

    missing = (
        required_columns
        - set(mapping.columns)
    )

    if missing:
        raise ValueError(
            "Delta mapping is missing "
            f"required columns: {sorted(missing)}"
        )

    if mapping.height == 0:
        raise ValueError(
            "Delta mapping is empty."
        )

    for column in required_columns:

        if (
            mapping[column]
            .null_count()
            > 0
        ):
            raise ValueError(
                f"{column} contains null values."
            )

    change_types = set(
        mapping[
            "change_type"
        ]
        .cast(pl.String)
        .str.to_uppercase()
        .to_list()
    )

    invalid_types = (
        change_types
        - VALID_CHANGE_TYPES
    )

    if invalid_types:
        raise ValueError(
            "Invalid change_type values: "
            f"{sorted(invalid_types)}"
        )

    # Bir document bütün chunk-larında
    # eyni change_type daşımalıdır.
    inconsistent_documents = (
        mapping
        .group_by(
            "document_id"
        )
        .agg(
            pl.col(
                "change_type"
            )
            .n_unique()
            .alias(
                "change_type_count"
            )
        )
        .filter(
            pl.col(
                "change_type_count"
            )
            > 1
        )
    )

    if (
        inconsistent_documents
        .height
        > 0
    ):
        raise ValueError(
            "A document cannot contain "
            "multiple change_type values."
        )


# DELETED DOCUMENTS

def load_deleted_document_ids(
) -> set[str]:

    if not (
        DELETED_DOCUMENTS_PATH
        .exists()
    ):
        return set()

    df = pl.read_parquet(
        DELETED_DOCUMENTS_PATH
    )

    if (
        "document_id"
        not in df.columns
    ):
        raise ValueError(
            "deleted_documents.parquet "
            "must contain document_id."
        )

    if (
        df["document_id"]
        .null_count()
        > 0
    ):
        raise ValueError(
            "Deleted document_id "
            "contains null values."
        )

    return {
        str(value).strip()
        for value
        in df[
            "document_id"
        ].to_list()
        if str(value).strip()
    }


# BM25 CORPUS

def build_updated_corpus(
    existing_corpus: pl.DataFrame | None,
    delta_mapping: pl.DataFrame,
    deleted_document_ids: set[str],
) -> pl.DataFrame:

    preferred_columns = [
        "document_id",
        "original_id",
        "change_type",
        "retrieval_hash",
        "chunk_id",
        "chunk_index",
        "title",
        "chunk_text",
        "embedding_tokens",
        "content_hash",
        "url",
        "date",
        "modified",
        "source_type",
        "source",
        "language",
    ]

    available_delta_columns = [
        column
        for column
        in preferred_columns
        if column
        in delta_mapping.columns
    ]

    incoming = (
        delta_mapping
        .select(
            available_delta_columns
        )
    )

    incoming_document_ids = {
        str(value)
        for value
        in delta_mapping[
            "document_id"
        ].to_list()
    }

    # Bu document-lərin köhnə chunk-ları
    # corpus-da qalmamalıdır.
    replace_document_ids = (
        incoming_document_ids
        | deleted_document_ids
    )

    if (
        existing_corpus is None
        or existing_corpus.height == 0
    ):

        result = incoming

    else:

        if (
            "document_id"
            not in existing_corpus.columns
        ):
            raise ValueError(
                "Existing BM25 corpus "
                "does not contain document_id."
            )

        if replace_document_ids:

            remaining = (
                existing_corpus
                .filter(
                    ~pl.col(
                        "document_id"
                    )
                    .cast(pl.String)
                    .is_in(
                        list(
                            replace_document_ids
                        )
                    )
                )
            )

        else:

            remaining = (
                existing_corpus
            )

        result = pl.concat(
            [
                remaining,
                incoming,
            ],
            how="diagonal_relaxed",
        )

    if (
        result.height > 0
        and "chunk_id"
        in result.columns
        and (
            result[
                "chunk_id"
            ].n_unique()
            != result.height
        )
    ):
        raise RuntimeError(
            "Duplicate chunk_id detected "
            "after BM25 corpus sync."
        )

    return result


# MAIN

@archived_stage("delta", ("BM25_CORPUS_PATH", "DELTA_SYNC_METADATA_PATH", "BM25_METADATA_PATH"))
def main(output_dir):
    INDEXES_DIR = output_dir
    BM25_CORPUS_PATH = output_dir / retrieval_config.BM25_CORPUS_PATH.name
    DELTA_SYNC_METADATA_PATH = output_dir / retrieval_config.DELTA_SYNC_METADATA_PATH.name

    print()
    print("=" * 70)
    print("UNEC DELTA RETRIEVAL SYNC")
    print("=" * 70)

    # LOAD DELTA EMBEDDINGS

    if not EMBEDDINGS_PATH.exists():

        raise FileNotFoundError(
            "Delta embeddings not found:\n"
            f"{EMBEDDINGS_PATH}"
        )

    if not MAPPING_PATH.exists():

        raise FileNotFoundError(
            "Delta embedding mapping "
            "not found:\n"
            f"{MAPPING_PATH}"
        )

    print(
        "Loading delta embeddings..."
    )

    vectors = np.load(
        EMBEDDINGS_PATH
    )

    print(
        "Loading delta mapping..."
    )

    mapping = pl.read_parquet(
        MAPPING_PATH
    )

    # Standard vector/mapping checks
    validate_artifacts(
        vectors=vectors,
        mapping=mapping,
    )

    # Delta-specific checks
    validate_delta_mapping(
        mapping
    )

    deleted_document_ids = (
        load_deleted_document_ids()
    )

    # NORMALIZE CHANGE TYPE

    mapping = (
        mapping
        .with_columns(
            pl.col(
                "change_type"
            )
            .cast(pl.String)
            .str.to_uppercase()
        )
    )

    new_document_ids = set(
        mapping
        .filter(
            pl.col(
                "change_type"
            )
            == "NEW"
        )[
            "document_id"
        ]
        .cast(pl.String)
        .to_list()
    )

    updated_document_ids = set(
        mapping
        .filter(
            pl.col(
                "change_type"
            )
            == "UPDATED"
        )[
            "document_id"
        ]
        .cast(pl.String)
        .to_list()
    )

    incoming_document_ids = (
        new_document_ids
        | updated_document_ids
    )

    conflict_ids = (
        incoming_document_ids
        & deleted_document_ids
    )

    if conflict_ids:

        raise ValueError(
            "The same document cannot be "
            "NEW/UPDATED and DELETED "
            "in the same run: "
            f"{sorted(conflict_ids)}"
        )

    payloads = build_payloads(
        mapping
    )

    # SUMMARY BEFORE APPLY

    print()
    print("=" * 70)
    print("DELTA SUMMARY")
    print("=" * 70)

    print(
        f"NEW documents     : "
        f"{len(new_document_ids):,}"
    )

    print(
        f"UPDATED documents : "
        f"{len(updated_document_ids):,}"
    )

    print(
        f"DELETED documents : "
        f"{len(deleted_document_ids):,}"
    )

    print(
        f"Incoming chunks   : "
        f"{mapping.height:,}"
    )

    # QDRANT

    # Validate and persist the merged local corpus before making remote changes.
    existing_corpus = pl.read_parquet(BM25_CORPUS_PATH) if BM25_CORPUS_PATH.exists() else None
    updated_corpus = build_updated_corpus(existing_corpus, mapping, deleted_document_ids)
    updated_corpus.write_parquet(BM25_CORPUS_PATH)

    index = QdrantVectorIndex()

    start_time = (
        time.perf_counter()
    )

    try:

        index.health_check()

        index.ensure_collection()

        before_count = (
            index.count()
        )
        expected_base_count = existing_corpus.height if existing_corpus is not None else 0
        if before_count != expected_base_count:
            raise RuntimeError("Qdrant and the existing BM25 corpus have different base counts; "
                               "select the correct base corpus before applying a delta")

        print()
        print(
            f"Qdrant before     : "
            f"{before_count:,}"
        )

        # UPDATED

        if updated_document_ids:

            print()
            print(
                "Removing old UPDATED "
                "documents from Qdrant..."
            )

            for document_id in sorted(
                updated_document_ids
            ):

                index.delete_document(
                    document_id
                )

        # DELETED

        if deleted_document_ids:

            print()
            print(
                "Removing DELETED "
                "documents from Qdrant..."
            )

            for document_id in sorted(
                deleted_document_ids
            ):

                index.delete_document(
                    document_id
                )

        # NEW + UPDATED UPSERT

        print()
        print(
            "Upserting NEW/UPDATED chunks..."
        )

        indexed_chunks = (
            index.upsert(
                vectors=vectors,
                payloads=payloads,
                batch_size=(
                    QDRANT_BATCH_SIZE
                ),
            )
        )

        after_count = (
            index.count()
        )
        if after_count != updated_corpus.height:
            raise RuntimeError("Qdrant/BM25 counts differ after delta; remote changes may already "
                               "have been applied, but local results will not be published")

        # BM25 FULL CORPUS SYNC

        print()
        print(
            "Synchronizing BM25 corpus..."
        )

        write_json(output_dir / retrieval_config.BM25_METADATA_PATH.name, {
            "created_at": datetime.now(timezone.utc).isoformat(), "index_type": "bm25",
            "mode": "delta_corpus", "documents": updated_corpus.height,
            "corpus_file": str(retrieval_config.BM25_CORPUS_PATH)})

        elapsed = (
            time.perf_counter()
            - start_time
        )

        # METADATA

        metadata = {
            "created_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "mode":
                "delta",

            "new_documents":
                len(
                    new_document_ids
                ),

            "updated_documents":
                len(
                    updated_document_ids
                ),

            "deleted_documents":
                len(
                    deleted_document_ids
                ),

            "incoming_chunks":
                int(
                    mapping.height
                ),

            "upserted_chunks":
                int(
                    indexed_chunks
                ),

            "qdrant_before":
                int(
                    before_count
                ),

            "qdrant_after":
                int(
                    after_count
                ),

            "bm25_corpus_chunks":
                int(
                    updated_corpus.height
                ),

            "elapsed_seconds":
                round(
                    elapsed,
                    4,
                ),
        }

        with open(
            DELTA_SYNC_METADATA_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # FINAL SUMMARY

        print()
        print("=" * 70)
        print("DELTA SYNC FINISHED")
        print("=" * 70)

        print(
            f"NEW documents      : "
            f"{len(new_document_ids):,}"
        )

        print(
            f"UPDATED documents  : "
            f"{len(updated_document_ids):,}"
        )

        print(
            f"DELETED documents  : "
            f"{len(deleted_document_ids):,}"
        )

        print(
            f"Upserted chunks    : "
            f"{indexed_chunks:,}"
        )

        print(
            f"Qdrant before      : "
            f"{before_count:,}"
        )

        print(
            f"Qdrant after       : "
            f"{after_count:,}"
        )

        print(
            f"BM25 corpus chunks : "
            f"{updated_corpus.height:,}"
        )

        print(
            f"Elapsed            : "
            f"{elapsed:.4f} sec"
        )

    finally:

        index.close()


if __name__ == "__main__":
    main()
