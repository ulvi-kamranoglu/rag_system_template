# UNEC Knowledge Pipeline — Delta Update Contract

This document defines how dataset changes move through the pipeline.

## Pipeline

```text
Data Collection
      ↓
Data Cleaning
      ↓
Change Detection / Diff
      ↓
NEW / UPDATED / UNCHANGED / DELETED
      ↓
Chunking
      ↓
Embedding
      ↓
Retrieval Sync
```

---

# 1. Change Detection Ownership

Change detection belongs to the data cleaning / diff stage.

The embedding and retrieval modules must NOT decide whether a source document is new, updated or deleted.

The diff stage compares the previous clean snapshot with the new clean snapshot.

---

# 2. Document Identity

Every document must have a stable:

```text
document_id
```

The same source document must keep the same `document_id` between pipeline runs.

Example:

```text
unec.edu.az:post:5001
```

---

# 3. Retrieval Hash

Change detection should use a retrieval-oriented hash.

Recommended input:

```text
normalized_title + normalized_content
```

Example:

```text
retrieval_hash = SHA256(
    normalized_title + normalized_content
)
```

A change in either title or content means that the document is considered updated.

The website's `modified` field must NOT be the only change-detection mechanism.

---

# 4. Change Types

Each source document is classified as one of:

```text
NEW
UPDATED
UNCHANGED
DELETED
```

Rules:

```text
NEW
document_id exists in the new snapshot
but not in the previous snapshot.

UPDATED
document_id exists in both snapshots
but retrieval_hash changed.

UNCHANGED
document_id exists in both snapshots
and retrieval_hash is identical.

DELETED
document_id exists in the previous snapshot
but not in a confirmed complete new snapshot.
```

---

# 5. Deleted Document Safety

A missing document must NOT automatically be considered deleted.

DELETED status may only be produced when the new crawl is confirmed to be a complete and successful snapshot.

This prevents crawler failures or incomplete scraping from accidentally deleting valid documents from the retrieval system.

---

# 6. Cleaning / Diff Output

The cleaning / diff stage should generate:

```text
shared_data/02_clean/changed_documents.parquet
shared_data/02_clean/<dataset>_deleted_documents.parquet
```

## changed_documents.parquet

Contains only:

```text
NEW
UPDATED
```

documents.

Required fields:

```text
document_id
change_type
title
content
retrieval_hash
```

Recommended metadata:

```text
original_id
url
date
modified
source_type
source
language
```

`change_type` must contain only:

```text
NEW
UPDATED
```

UNCHANGED documents are not sent to chunking.

---

## <dataset>_deleted_documents.parquet

Contains:

```text
document_id
```

for confirmed deleted documents.

Deleted documents do not go through chunking or embedding.

---

# 7. Chunking Input

Chunking receives only:

```text
NEW
UPDATED
```

documents.

UNCHANGED documents are skipped.

DELETED documents are not chunked.

---

# 8. Chunking Output

Chunking generates:

```text
shared_data/03_chunks/<dataset>_chunks_384.parquet
```

Required fields:

```text
document_id
change_type
chunk_id
chunk_index
title
chunk_text
embedding_text
embedding_tokens
retrieval_hash
```

Recommended metadata:

```text
original_id
url
date
modified
source_type
source
language
```

`change_type` must be preserved from the document-level input.

Example:

```text
document_id              change_type   chunk_id
---------------------------------------------------------------
unec.edu.az:post:5001    NEW           unec.edu.az:post:5001_chunk_0000
unec.edu.az:post:5001    NEW           unec.edu.az:post:5001_chunk_0001
unec.edu.az:page:200     UPDATED       unec.edu.az:page:200_chunk_0000
```

---

# 9. Embedding Responsibility

Embedding does not perform change detection.

Embedding receives `<dataset>_chunks_384.parquet` and generates vectors only for the supplied chunks.

Flow:

```text
embedding_text
      ↓
BAAI/bge-m3
      ↓
1024D dense vector
```

The embedding mapping must preserve:

```text
document_id
change_type
chunk_id
retrieval_hash
```

---

# 10. Retrieval Sync Responsibility

Retrieval applies the already-decided changes.

## NEW

```text
new chunks
   ↓
Qdrant upsert
```

## UPDATED

For each updated `document_id`:

```text
delete all existing Qdrant chunks for document_id
      ↓
upsert the new chunks
```

This document-level replacement prevents stale chunks from remaining when the number of chunks changes.

## DELETED

```text
document_id
      ↓
delete all Qdrant chunks belonging to the document
```

---

# 11. BM25 Synchronization

Qdrant supports incremental updates directly.

BM25 will use a complete local retrieval corpus.

For every delta:

```text
NEW
→ add new chunks

UPDATED
→ remove old document chunks
→ add new document chunks

DELETED
→ remove document chunks
```

After the local retrieval corpus is updated:

```text
retrieval_corpus.parquet
        ↓
BM25 rebuild
```

---

# 12. Initial Full Build

For the first pipeline run there is no previous dataset.

Therefore every valid document may be treated as:

```text
NEW
```

Flow:

```text
full clean dataset
      ↓
change_type = NEW
      ↓
chunking
      ↓
embedding
      ↓
full Qdrant build
      ↓
BM25 build
```

---

# 13. Later Delta Runs

Example:

```text
Previous snapshot: 100,000 documents
New snapshot:      110,000 documents
```

Diff result may be:

```text
NEW       : 8,000
UPDATED   : 2,000
UNCHANGED : 90,000
DELETED   : 300
```

Only:

```text
8,000 NEW
+
2,000 UPDATED
```

go through chunking and embedding.

The 90,000 unchanged documents require no new embedding computation.

The 300 deleted document IDs go directly to retrieval synchronization.