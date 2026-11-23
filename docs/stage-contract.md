# Independent data stages (contract version 1)

Each stage is a separately runnable project. It owns input selection, validation,
mode semantics, archive creation, publication to shared_data and recovery. No
stage imports orchestration. The coordinator imports no stage package, loads no
models, reads no Parquet, chooses no intermediate dataset and performs no Qdrant
requests. It sequences commands and records their completion receipts.

## Public commands

From the workspace, with the projects installed in the selected environment:

```powershell
python -m wp_tracker.stage_cli --mode collect
python -m canonical_pipeline.stage_cli --mode collect
python -m chunking_pipeline.stage_cli
python -m embedding.stage_cli
python -m retrieval.stage_cli
```

For an upload already placed in the configured raw folder:

```powershell
python -m canonical_pipeline.stage_cli --mode join --input shared_data/01_raw/example.parquet
# Or: --mode reset_data with the same --input argument.
python -m chunking_pipeline.stage_cli
python -m embedding.stage_cli
python -m retrieval.stage_cli
```

Every command accepts `--config`, `--mode`, `--run-id` and `--result`.
A stage may define further public options of its own; the coordinator passes them through unchanged with
`run_pipeline.py --stage-option <stage>:--<name>[=value]`, stores them in the run state and reuses them on
resume, without interpreting them. Collection defines `--sources` (comma-separated subset of enabled
sources, e.g. `--sources=news,instagram/uneceduaz`); the selection is pinned per batch ID.
`--config` defaults to that project's own config. Downstream stages inherit mode
and run ID from the upstream shared report when the arguments are omitted.
Collection and a new explicit raw import generate a run ID if omitted. To retry
either of those, reuse the run ID printed by the stage. A run ID identifies one
batch and mode; it must not be reused for different input. Canonicalization also
accepts `--input` and `--deleted` as explicit user selections. Relative explicit
input paths resolve from the terminal; configured paths retain project semantics.

The existing low-level commands remain available for diagnostics and manual
processing. Use `stage_cli` for the version-1 shared handoff protocol.

## Ownership and data flow

| Stage | Reads | Publishes to shared and a new archive run |
| --- | --- | --- |
| Collection | WordPress pages/posts, two Instagram profiles, News; independent source baselines and snapshots | Combined `01_raw/chatbot_dataset_new.parquet`, `chatbot_dataset_deleted.parquet`, reports |
| Canonical | Raw shared batch/deletions, latest `archive_data/02_clean/base_data` | `02_clean/<input>_canonical.parquet`, quality reports |
| Chunk | Published canonical shared batch | `03_chunks/<batch>_chunks_<size>.parquet`, prepared data and summaries |
| Embedding | Published shared chunks at its configured token size | `04_embeddings/<batch>_embeddings.npy`, mapping and metadata |
| Retrieval | Shared embeddings and canonical batch, canonical archive full snapshot, current shared index/Qdrant | Complete active `05_indexes` and Qdrant synchronization |

Canonicalization alone creates the next full version in
`archive_data/02_clean/base_data` and retains its immutable `base_snapshot` in the
canonical run. Shared clean/chunk/embedding outputs contain the batch, not the
full historical dataset. In collect/join modes canonicalization merges the batch
with the baseline; retrieval retains untouched documents and removes deleted or
replaced chunks. In reset_data, canonicalization creates the next version from
the uploaded data alone and retrieval replaces index membership. Empty resets
are rejected. No changes creates no baseline version and stops the coordinator.
A deletion-only batch publishes explicit skipped chunk/embedding reports, so old
embeddings cannot be mistaken for this batch.

The Parquet schemas remain those validated by each project (canonical schema,
chunk identifiers and embedding text, aligned vector_index/chunk_id/document_id
mapping). The protocol adds metadata; it does not rename those columns or change
vector identity. A replacement collection project must produce both raw Parquet
files with columns accepted by canonicalization, retain reliable deletion identity,
and publish the contract below. Its internal language, scraping implementation
and snapshot storage can differ.

The built-in collection's source layout and setup are described in
[collection sources](../01_data_collection/docs/COLLECTION_SOURCES.md). Source
selection and aggregation are internal collection responsibilities.

## Shared handoff and receipts

The final `run_report.json` contains the stage-specific report plus:

```json
{
  "pipeline_run_id": "batch-id",
  "run_dir": "absolute/path/to/archive/run_...",
  "contract": {
    "version": 1,
    "stage": "collection",
    "run_id": "batch-id",
    "mode": "collect",
    "status": "completed",
    "files": {"chatbot_dataset_new.parquet": "sha256", "chatbot_dataset_deleted.parquet": "sha256"}
  }
}
```

`files` is the complete recursive shared inventory excluding `run_report.json`:
file keys use forward slashes; directory keys have null values. Additional
published reports must also appear in this inventory. The archive report carries
the same contract; canonical's archive-only base_snapshot is deliberately absent
from the shared inventory. Each consumer checks stage, mode, batch identity and
hashes. Missing, ambiguous, stale or modified inputs fail rather than falling back
to an arbitrary latest file. The original report fields remain required by the
relevant consumer: canonical exposes `output_file`, `output_rows`, `run_dir`,
`base_output_file`; chunk exposes `dataset_name`, `variants`; embedding exposes
`dataset_name` and its artifact metadata. Their native validators enforce schemas.

Publication uses the existing recoverable directory replacement. The contract is
written last using file replacement; readers using the stage protocol require a
valid contract. Protocol commands acquire shared stage locks for inputs and
outputs during processing, alongside the existing native writer locks. Direct
low-level commands and external upload tools must not be run concurrently with
stage commands against the same folders.

`--result PATH` writes a separate coordinator receipt after success:

```json
{"version":1,"stage":"canonical","run_id":"batch-id","mode":"join","status":"completed","dataset_name":"sample_v1.5","archive":"path/to/run"}
```

Statuses are `completed`, `skipped` and `no_changes`. A nonzero process exit or a
missing/invalid receipt stops orchestration. A replacement command only needs to
support this CLI, receipt, and published data contract. Configure its module and
optional fixed arguments under `pipeline.yaml:stage_commands`.

## Recovery and compatibility

Reinvoke the same stage with the same run ID. The stage discovers its own archive
and retries local publication without rerunning scraping, inference or committed
Qdrant writes. Canonical records raw input hashes under its archive `.requests`
folder before processing, and refuses changed input on a retry before commit.
If Qdrant sync was interrupted before archive commit, stable chunk IDs allow the
existing reconciliation algorithm to resume safely.

`run_pipeline.py --resume` skips completed stage receipts and reinvokes the failed
command unchanged. New coordinator checkpoints use protocol 2. Completed legacy
runs remain readable; unfinished legacy runs must be completed with the previous
release before upgrading (they are never replayed under a different protocol).
Cancellation is allowed only before any stage attempt; once a command has been
attempted, resume it because it may already have committed data. Existing data is
not migrated or reindexed by installing this change. The first new pipeline run
publishes the new contract; old shared reports cannot be used directly as new
downstream handoffs until their producer publishes a contract.
