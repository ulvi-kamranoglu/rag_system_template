"""Initialize an empty Qdrant target using the included complete snapshot."""
from _environment import ROOT, snapshot


def main():
    from retrieval.config import load_settings
    from retrieval.backends.qdrant import QdrantVectorIndex
    settings = load_settings()
    import json
    report = settings["INDEXES_DIR"] / "run_report.json"
    if report.is_file() and json.loads(report.read_text(encoding="utf-8")).get("batch_dataset_name"):
        raise ValueError("This snapshot has batch-only embeddings. Restore a Qdrant backup or rebuild from the full canonical base; batch vectors cannot bootstrap the full index.")
    if settings["VECTOR_BACKEND"] != "qdrant":
        raise ValueError("Bootstrap requires the Qdrant backend")
    count, _ = snapshot(settings)
    print(f"Project: {ROOT}\nTarget: {settings['QDRANT_URL']} / {settings['QDRANT_COLLECTION']}\n"
          f"Dataset: {settings['DATASET_NAME']} ({count} vectors)", flush=True)
    with QdrantVectorIndex(url=settings["QDRANT_URL"],
                           collection_name=settings["QDRANT_COLLECTION"]) as index:
        index.health_check()
        if index.collection_exists():
            raise RuntimeError("Collection already exists. Bootstrap will not replace it; "
                               "check a restored target or choose a new empty collection.")
        # No recreate/delete operation is used. The sync stage checks all IDs
        # against its baseline and can resume via its archived publication path.
        index.ensure_collection()
    from retrieval.indexing.sync import main as sync
    sync([])
    print("Bootstrap complete. Run scripts/check_environment.py before starting the API.")


if __name__ == "__main__":
    main()
