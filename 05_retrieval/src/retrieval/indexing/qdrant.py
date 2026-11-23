"""Build the initial Qdrant index from existing embedding artifacts."""
import json
import time
from datetime import date, datetime, timezone

import numpy as np
import polars as pl
from retrieval import config as retrieval_config
from retrieval.indexing.storage import archived_stage
from retrieval.artifacts import (
    PAYLOAD_COLUMNS,
    normalize_payload_value,
    validate_artifacts,
    build_payloads,
)

from retrieval.config import (
    EMBEDDINGS_PATH,
    MAPPING_PATH,
    EMBEDDING_METADATA_PATH,
    INDEXES_DIR,
    QDRANT_INDEX_METADATA_PATH,
    QDRANT_COLLECTION,
    VECTOR_DIMENSION,
    QDRANT_BATCH_SIZE,
    INDEX_VERSION,
)

from retrieval.backends.qdrant import (
    QdrantVectorIndex,
)


# MAIN

@archived_stage("qdrant", ("QDRANT_INDEX_METADATA_PATH",))
def main(output_dir):
    INDEXES_DIR = output_dir
    QDRANT_INDEX_METADATA_PATH = output_dir / retrieval_config.QDRANT_INDEX_METADATA_PATH.name

    print()
    print("=" * 70)
    print("UNEC QDRANT INDEX PIPELINE")
    print("=" * 70)

    print(
        f"Embeddings : {EMBEDDINGS_PATH}"
    )

    print(
        f"Mapping    : {MAPPING_PATH}"
    )

    print(
        f"Collection : {QDRANT_COLLECTION}"
    )

    print(
        f"Version    : {INDEX_VERSION}"
    )

    print()

    # CHECK INPUT FILES

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found:\n"
            f"{EMBEDDINGS_PATH}"
        )

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Mapping not found:\n"
            f"{MAPPING_PATH}"
        )

    # LOAD ARTIFACTS

    print("Loading embeddings...")

    vectors = np.load(
        EMBEDDINGS_PATH
    )

    print("Loading mapping...")

    mapping = pl.read_parquet(
        MAPPING_PATH
    )

    validate_artifacts(
        vectors=vectors,
        mapping=mapping,
    )

    # OPTIONAL EMBEDDING METADATA

    embedding_metadata = {}

    if EMBEDDING_METADATA_PATH.exists():

        with open(
            EMBEDDING_METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            embedding_metadata = (
                json.load(file)
            )

        metadata_dimension = (
            embedding_metadata.get(
                "dimension"
            )
        )

        if (
            metadata_dimension is not None
            and int(metadata_dimension)
            != VECTOR_DIMENSION
        ):
            raise ValueError(
                "Embedding metadata dimension "
                "does not match retrieval "
                "configuration."
            )

    # PAYLOADS

    print()
    print("Building Qdrant payloads...")

    payloads = build_payloads(
        mapping
    )

    # QDRANT

    print()
    print("=" * 70)
    print("QDRANT")
    print("=" * 70)

    index = QdrantVectorIndex()

    index.health_check()

    old_count = index.count()

    print(
        f"Existing points : "
        f"{old_count:,}"
    )

    # İlk offline index build üçün full rebuild.
    # Beləliklə əvvəlki/stale point-lər qalmır.
    print(
        "Recreating collection..."
    )

    index.recreate_collection()

    print(
        "Collection recreated."
    )

    # INDEX

    print()
    print("=" * 70)
    print("INDEXING")
    print("=" * 70)

    start_time = time.perf_counter()

    indexed_count = index.upsert(
        vectors=vectors,
        payloads=payloads,
        batch_size=QDRANT_BATCH_SIZE,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    final_count = index.count()

    if final_count != vectors.shape[0]:
        raise RuntimeError(
            "Qdrant point count mismatch. "
            f"Expected={vectors.shape[0]}, "
            f"found={final_count}"
        )

    # SAVE LOCAL INDEX METADATA

    INDEXES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "index_version":
            INDEX_VERSION,

        "collection":
            QDRANT_COLLECTION,

        "vector_dimension":
            VECTOR_DIMENSION,

        "distance":
            "cosine",

        "batch_size":
            QDRANT_BATCH_SIZE,

        "previous_point_count":
            int(old_count),

        "indexed_points":
            int(indexed_count),

        "final_point_count":
            int(final_count),

        "elapsed_seconds":
            round(elapsed, 4),

        "points_per_second":
            round(
                (
                    indexed_count / elapsed
                    if elapsed > 0
                    else 0
                ),
                4,
            ),

        "embedding_model":
            embedding_metadata.get(
                "model"
            ),

        "embedding_version":
            embedding_metadata.get(
                "embedding_version"
            ),

        "embeddings_file":
            str(EMBEDDINGS_PATH),

        "mapping_file":
            str(MAPPING_PATH),
    }

    with open(
        QDRANT_INDEX_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # SUMMARY

    print()
    print("=" * 70)
    print("QDRANT INDEX BUILD FINISHED")
    print("=" * 70)

    print(
        f"Previous points : "
        f"{old_count:,}"
    )

    print(
        f"Indexed points  : "
        f"{indexed_count:,}"
    )

    print(
        f"Final points    : "
        f"{final_count:,}"
    )

    print(
        f"Dimension       : "
        f"{VECTOR_DIMENSION}"
    )

    print(
        f"Distance        : cosine"
    )

    print(
        f"Elapsed         : "
        f"{elapsed:.2f} sec"
    )

    if elapsed > 0:

        print(
            f"Points/sec      : "
            f"{indexed_count / elapsed:.2f}"
        )

    print()
    print(
        "Metadata:"
    )

    print(
        QDRANT_INDEX_METADATA_PATH
    )


if __name__ == "__main__":
    main()
