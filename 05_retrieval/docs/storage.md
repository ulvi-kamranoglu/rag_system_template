# Index storage and publication

Qdrant full build, FAISS full build, BM25 build and Qdrant delta sync now use
the same immutable archive and journaled local publication mechanism as the
other data stages. No command is executed merely by installing or importing code.

Paths are defined in `05_retrieval/src/retrieval/config.py`:

```text
ARCHIVE_DIR = <workspace>/archive_data/05_indexes
INDEXES_DIR = <workspace>/shared_data/05_indexes
```

Each successful stage produces a complete local snapshot:

```text
archive_data/05_indexes/run_<UTC timestamp>_<uuid>/
    <dataset>_qdrant_index_metadata.json       # when present
    <dataset>.faiss                          # when present
    <dataset>_faiss_mapping.parquet           # when present
    <dataset>_faiss_index_metadata.json       # when present
    <dataset>_bm25_corpus.parquet             # when present
    <dataset>_bm25_index_metadata.json        # when present
    <dataset>_delta_sync_metadata.json        # when present
    run_report.json
```

Not all files are produced by every stage. The current shared state is copied
to private staging, that stage's output files are replaced, serialized tables
and metadata are validated, hashes are recorded, and the completed archive is
published to shared storage. A BM25 run therefore preserves existing FAISS and
Qdrant metadata files. Full snapshots use additional disk space; old archives
are never automatically pruned.

A writer lock spans preparation and publication to prevent retrieval commands
from overwriting each other's local updates. Publication uses byte verification,
a recovery journal and rollback. Windows may briefly expose a missing shared
directory between renames. See `storage_common/README.md` for the exact guarantees.

## Manual commands

Run from `Chatbot_projects` with the environment active:

```powershell
$env:PYTHONUTF8 = "1"
$env:PIPELINE_DATASET = "chatbot_dataset_v1.3"

# INITIAL full Qdrant build: recreates the selected collection.
python -m retrieval.pipeline

# Build the BM25 corpus from the selected embedding mapping.
python -m retrieval.bm25_pipeline

# Alternative local dense index.
python -m retrieval.faiss_pipeline
```

Those are actual build commands, not dry-runs. The full-build commands do not
merge incoming batches with old data. `retrieval.delta_sync_pipeline` remains
the separate update entrypoint and requires a NEW/UPDATED mapping, deletion
input and the correct complete base BM25 corpus. This storage change does not
implement the missing collection/diff integration or base-vs-delta dataset selection.

Delta builds/validates its merged corpus in staging before touching Qdrant.
It compares the existing Qdrant count with the base corpus count before document
deletion/upsert, then compares the resulting count with the merged corpus count.
Count checks are consistency checks, not proof of identical payload content.
Delta updates the BM25 corpus metadata but does not refresh a FAISS index.

## Retrying local publication

If computation/remote indexing finished and only local publication failed, the
completed archive remains. Retry it without repeating the Qdrant operations:

```powershell
python -m retrieval.storage --publish-run "C:/path/to/archive_data/05_indexes/run_<timestamp>_<uuid>"
```

Each stage's existing command also accepts `--publish-run <archive>` for an
archive of that same operation. Retry validates hashes and refuses to overwrite
shared files if another stage has published a newer/different state since the
archived run. Already-published archives are idempotent. Legacy shared run folders
require explicit `--migrate-legacy-shared` and are retained in a sibling backup.

## Qdrant boundary

These archives contain LOCAL files, not a Qdrant server snapshot or a vector backup
of its collection. Qdrant keeps vectors in its own storage. An interrupted remote
full rebuild/delta can leave remote changes even when no completed local archive
is published. Local rollback and `--publish-run` do not roll back, restore or replay
Qdrant. Do not rerun a full build to repair only local publication: it recreates
the collection. Remote collection backup/recovery must be handled separately.

Manual model/service-free storage regressions:

```powershell
python -m pytest 05_retrieval/tests/test_retrieval_storage.py -q
```
