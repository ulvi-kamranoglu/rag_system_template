# Chunk Dataset Contract

This document defines the data contract between:

`data_chunking` → `embedding`

## Input file

The chunking pipeline must generate:

`shared_data/03_chunks/<dataset>_chunks_384.parquet`

## Required columns

| Column | Type | Description |
|---|---|---|
| document_id | string | Stable internal document identifier |
| original_id | string | Original source identifier |
| chunk_id | string | Globally unique chunk identifier |
| chunk_index | integer | Zero-based chunk position within document |
| title | string | Clean document title |
| chunk_text | string | Content belonging to this chunk |
| embedding_text | string | Final text that will be sent to the embedding model |
| embedding_tokens | integer | Token count of embedding_text |
| content_hash | string | Hash of the source document content |
| source_type | string | Source type, e.g. page or post |
| source | string | Source domain, e.g. unec.edu.az |
| language | string | Language code, e.g. az |

## Optional columns

| Column | Type |
|---|---|
| url | string / null |
| date | string / null |
| modified | string / null |

## Rules

### chunk_id

Must be unique.

Recommended format:

`{document_id}:chunk:{chunk_index:04d}`

Example:

`post:5001:chunk:0000`

### embedding_text

The embedding project does NOT create or modify chunks.

`embedding_text` must already be completely prepared by the chunking pipeline.

Current representation:

Title: <title>

Content:
<chunk_text>

### Token limit

Current embedding configuration:

- Model: `BAAI/bge-m3`
- Maximum embedding_text length: `384` tokens
- Tokenizer: BGE-M3 tokenizer

Therefore:

`embedding_tokens <= 384`

The chunking pipeline is responsible for satisfying this constraint.

### Overlap

Current baseline chunking configuration:

- max embedding tokens: 384
- overlap: 64 tokens

Overlap is owned by the chunking pipeline, not the embedding pipeline.

### Stable ordering

Each row represents exactly one chunk.

Chunk rows must have stable:

- document_id
- chunk_id
- chunk_index

### content_hash

All chunks belonging to the same unchanged source document must carry the same `content_hash`.

This will later be used to detect:

- new documents
- unchanged documents
- updated documents

## Embedding output

The embedding pipeline reads:

`shared_data/03_chunks/<dataset>_chunks_384.parquet`

and generates:

`shared_data/04_embeddings/<dataset>_embeddings.npy`

`shared_data/04_embeddings/<dataset>_embedding_mapping.parquet`

`shared_data/04_embeddings/<dataset>_embedding_metadata.json`

The embedding pipeline does not perform:

- cleaning
- chunk creation
- overlap logic
- title/content composition