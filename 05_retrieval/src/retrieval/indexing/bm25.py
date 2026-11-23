import json
import time
from datetime import datetime, timezone

import polars as pl
from retrieval import config as retrieval_config
from retrieval.indexing.storage import archived_stage

from retrieval.config import (
    MAPPING_PATH,
    INDEXES_DIR,
    DATASET_NAME,
    BM25_CORPUS_PATH,
    BM25_METADATA_PATH,
)

from retrieval.backends.bm25 import (
    BM25ChunkIndex,
)


REQUIRED_COLUMNS = {
    "document_id",
    "chunk_id",
    "title",
    "chunk_text",
}


# VALIDATION

def validate_mapping(
    df: pl.DataFrame,
) -> None:

    print("=" * 70)
    print("BM25 INPUT VALIDATION")
    print("=" * 70)

    missing = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if df.height == 0:
        raise ValueError(
            "Mapping dataset is empty."
        )

    if (
        df["chunk_id"]
        .null_count()
        > 0
    ):
        raise ValueError(
            "chunk_id contains null values."
        )

    if (
        df["document_id"]
        .null_count()
        > 0
    ):
        raise ValueError(
            "document_id contains null values."
        )

    if (
        df["chunk_id"]
        .n_unique()
        != df.height
    ):
        raise ValueError(
            "chunk_id values must be unique."
        )

    print(
        f"Rows             : {df.height:,}"
    )

    print(
        f"Unique chunks    : "
        f"{df['chunk_id'].n_unique():,}"
    )

    print(
        f"Unique documents : "
        f"{df['document_id'].n_unique():,}"
    )

    print(
        "Validation       : PASSED"
    )


# BUILD RECORDS

def build_records(
    df: pl.DataFrame,
) -> list[dict]:

    preferred_columns = [
        "document_id",
        "original_id",
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

    available_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    return (
        df
        .select(
            available_columns
        )
        .rows(
            named=True
        )
    )


# MAIN

@archived_stage("bm25", ("BM25_CORPUS_PATH", "BM25_METADATA_PATH"))
def main(output_dir):
    INDEXES_DIR = output_dir
    BM25_CORPUS_PATH = output_dir / retrieval_config.BM25_CORPUS_PATH.name
    BM25_METADATA_PATH = output_dir / retrieval_config.BM25_METADATA_PATH.name

    print()
    print("=" * 70)
    print("UNEC BM25 INDEX PIPELINE")
    print("=" * 70)

    print(
        f"Input : {MAPPING_PATH}"
    )

    print(
        f"Output: {INDEXES_DIR}"
    )

    print()

    # INPUT

    if not MAPPING_PATH.exists():

        raise FileNotFoundError(
            "Embedding mapping not found:\n"
            f"{MAPPING_PATH}"
        )

    print(
        "Loading mapping..."
    )

    df = pl.read_parquet(
        MAPPING_PATH
    )

    validate_mapping(
        df
    )

    # RECORDS

    records = build_records(
        df
    )

    # BUILD BM25

    print()
    print("=" * 70)
    print("BUILDING BM25 INDEX")
    print("=" * 70)

    start_time = (
        time.perf_counter()
    )

    index = BM25ChunkIndex(
        k1=1.5,
        b=0.75,
    )

    index.build(
        records
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # SAVE CORPUS

    INDEXES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus_df = pl.DataFrame(
        index.records
    )

    corpus_df.write_parquet(
        BM25_CORPUS_PATH
    )

    # SAVE METADATA

    metadata = {
        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "index_type":
            "bm25",

        "level":
            "chunk",

        "documents":
            index.count(),

        "unique_documents":
            int(
                df[
                    "document_id"
                ].n_unique()
            ),

        "k1":
            index.k1,

        "b":
            index.b,

        "title_weight":
            2,

        "elapsed_seconds":
            round(
                elapsed,
                4,
            ),

        "source_mapping":
            str(
                MAPPING_PATH
            ),

        "corpus_file":
            str(
                retrieval_config.BM25_CORPUS_PATH
            ),
    }

    with open(
        BM25_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # TEST QUERY

    print()
    print("=" * 70)
    print("TEST SEARCH")
    print("=" * 70)

    test_query = (
        "magistratura proqramları"
    )

    results = index.search(
        query=test_query,
        limit=min(
            5,
            index.count(),
        ),
    )

    print(
        f"Query: {test_query}"
    )

    for result in results:

        payload = (
            result["payload"]
        )

        print(
            f"{result['rank']}. "
            f"score="
            f"{result['score']:.4f} | "
            f"chunk="
            f"{result['chunk_id']} | "
            f"title="
            f"{payload.get('title')}"
        )

    # SUMMARY

    print()
    print("=" * 70)
    print("BM25 INDEX BUILD FINISHED")
    print("=" * 70)

    print(
        f"Chunks        : "
        f"{index.count():,}"
    )

    print(
        f"Documents     : "
        f"{df['document_id'].n_unique():,}"
    )

    print(
        f"k1            : "
        f"{index.k1}"
    )

    print(
        f"b             : "
        f"{index.b}"
    )

    print(
        f"Title weight  : 2"
    )

    print(
        f"Elapsed       : "
        f"{elapsed:.4f} sec"
    )

    print()
    print(
        "Corpus:"
    )

    print(
        BM25_CORPUS_PATH
    )

    print(
        "Metadata:"
    )

    print(
        BM25_METADATA_PATH
    )


if __name__ == "__main__":
    main()
