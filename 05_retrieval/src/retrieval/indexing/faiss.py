import json
import time
from datetime import datetime, timezone

import numpy as np
import polars as pl
from retrieval import config as retrieval_config
from retrieval.indexing.storage import archived_stage

from retrieval.config import (
    EMBEDDINGS_PATH,
    MAPPING_PATH,
    INDEXES_DIR,
    FAISS_INDEX_PATH,
    FAISS_MAPPING_PATH,
    FAISS_METADATA_PATH,
    VECTOR_DIMENSION,
)

from retrieval.artifacts import (
    validate_artifacts,
    build_payloads,
)

from retrieval.backends.faiss import (
    FaissVectorIndex,
)


@archived_stage("faiss", ("FAISS_INDEX_PATH", "FAISS_MAPPING_PATH", "FAISS_METADATA_PATH"))
def main(output_dir):
    INDEXES_DIR = output_dir
    FAISS_INDEX_PATH = output_dir / retrieval_config.FAISS_INDEX_PATH.name
    FAISS_MAPPING_PATH = output_dir / retrieval_config.FAISS_MAPPING_PATH.name
    FAISS_METADATA_PATH = output_dir / retrieval_config.FAISS_METADATA_PATH.name

    print()
    print("=" * 70)
    print("UNEC FAISS INDEX PIPELINE")
    print("=" * 70)

    print(
        f"Embeddings : {EMBEDDINGS_PATH}"
    )

    print(
        f"Mapping    : {MAPPING_PATH}"
    )

    print(
        f"Index      : {FAISS_INDEX_PATH}"
    )

    print()

    # CHECK INPUT

    if not EMBEDDINGS_PATH.exists():

        raise FileNotFoundError(
            "Embeddings not found:\n"
            f"{EMBEDDINGS_PATH}"
        )

    if not MAPPING_PATH.exists():

        raise FileNotFoundError(
            "Embedding mapping not found:\n"
            f"{MAPPING_PATH}"
        )

    # LOAD

    print(
        "Loading embeddings..."
    )

    vectors = np.load(
        EMBEDDINGS_PATH
    )

    print(
        "Loading mapping..."
    )

    mapping = pl.read_parquet(
        MAPPING_PATH
    )

    # VALIDATE

    validate_artifacts(
        vectors=vectors,
        mapping=mapping,
    )

    payloads = build_payloads(
        mapping
    )

    # BUILD FAISS

    print()
    print("=" * 70)
    print("BUILDING FAISS INDEX")
    print("=" * 70)

    start_time = (
        time.perf_counter()
    )

    index = FaissVectorIndex()

    index.build(
        vectors=vectors,
        payloads=payloads,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if (
        index.count()
        != vectors.shape[0]
    ):

        raise RuntimeError(
            "FAISS count mismatch. "
            f"Expected={vectors.shape[0]}, "
            f"found={index.count()}"
        )

    # OUTPUT DIR

    INDEXES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # SAVE INDEX

    print(
        "Saving FAISS index..."
    )

    index.save(
        FAISS_INDEX_PATH
    )

    # SAVE MAPPING

    mapping.write_parquet(
        FAISS_MAPPING_PATH
    )

    # METADATA

    index_size_mb = (
        FAISS_INDEX_PATH
        .stat()
        .st_size
        / 1024
        / 1024
    )

    metadata = {
        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "backend":
            "faiss",

        "index_type":
            "IndexFlatIP",

        "metric":
            "inner_product",

        "normalized_vectors":
            True,

        "effective_similarity":
            "cosine",

        "dimension":
            VECTOR_DIMENSION,

        "vectors":
            int(
                index.count()
            ),

        "dtype":
            str(
                vectors.dtype
            ),

        "elapsed_seconds":
            round(
                elapsed,
                6,
            ),

        "vectors_per_second":
            round(
                (
                    vectors.shape[0]
                    / elapsed
                    if elapsed > 0
                    else 0
                ),
                4,
            ),

        "index_size_mb":
            round(
                index_size_mb,
                4,
            ),

        "embeddings_file":
            str(
                EMBEDDINGS_PATH
            ),

        "mapping_file":
            str(
                retrieval_config.FAISS_MAPPING_PATH
            ),

        "index_file":
            str(
                retrieval_config.FAISS_INDEX_PATH
            ),
    }

    with open(
        FAISS_METADATA_PATH,
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
    print("FAISS INDEX BUILD FINISHED")
    print("=" * 70)

    print(
        f"Vectors      : "
        f"{index.count():,}"
    )

    print(
        f"Dimension    : "
        f"{VECTOR_DIMENSION}"
    )

    print(
        "Index type   : "
        "IndexFlatIP"
    )

    print(
        "Similarity   : "
        "cosine via normalized inner product"
    )

    print(
        f"Elapsed      : "
        f"{elapsed:.6f} sec"
    )

    print(
        f"Index size   : "
        f"{index_size_mb:.4f} MB"
    )

    print()
    print(
        "Index:"
    )

    print(
        FAISS_INDEX_PATH
    )

    print(
        "Mapping:"
    )

    print(
        FAISS_MAPPING_PATH
    )

    print(
        "Metadata:"
    )

    print(
        FAISS_METADATA_PATH
    )


if __name__ == "__main__":
    main()
