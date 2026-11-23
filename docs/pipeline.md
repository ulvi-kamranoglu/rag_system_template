# One-command data pipeline

The coordinator is `run_pipeline.py`, configured by `pipeline.yaml`. It never
rebuilds the Qdrant collection. A canonical baseline in
`archive_data/02_clean/base_data` and a matching active Qdrant + BM25 index must
already exist. Canonicalization alone maintains versioned base datasets.

For an upload named `example_filename.parquet`, all modes publish only the
cleaned batch as `shared_data/02_clean/example_filename_canonical.parquet`.
The same batch is stored at the root of its `archive_data/02_clean/run_...`.
The cumulative version (for example `chatbot_dataset_v1.6_canonical.parquet`)
is saved separately in `archive_data/02_clean/base_data`. The run also retains
an immutable `base_snapshot/` copy to support publication retries. That copy
is never exposed in shared_data.

Collect uses the same rule with `chatbot_dataset_new_canonical.parquet`.
Join/collect merge the batch into the previous base version and apply supplied
deletions. Reset creates the next version from the batch alone. The next version
name comes from the latest numeric canonical base, regardless of upload name.
Shared clean data is never used as the cumulative baseline.

`run_pipeline.py` is the compatibility CLI. Implementation is split between
`orchestration/src/orchestration/cli.py`, `runner.py`, `config.py`, and `state.py`.
`python -m orchestration` accepts the same arguments.
Each project also has an independent `stage_cli` entrypoint. Orchestration only
sequences configurable commands and checks completion receipts. It does not import
stage implementations, select intermediate data or decide publication/recovery.
See [the stage contract](stage-contract.md) for standalone commands and replacement
project requirements.

Admin starts collection and all subsequent stages:

```powershell
python run_pipeline.py --mode collect
```

Admin supplies a Parquet file directly inside `shared_data/01_raw`, skipping collection:

```powershell
python run_pipeline.py --mode join --input shared_data/01_raw/my_dataset.parquet
```

Replace the entire active dataset with a new version derived solely from an
administrator-provided input:

```powershell
python run_pipeline.py --mode reset_data --input shared_data/01_raw/replacement.parquet
```

Reset data increments the current canonical version (for example v1.5 to v1.6)
but does not merge old rows. It ignores default deletion files and rejects
`--deleted`. The canonical stage interprets `--mode reset_data` itself;
orchestration does not rewrite its processing settings. Empty or fully rejected inputs cannot
clear the active dataset. Chunking uses only the replacement corpus, embedding
generates fresh vectors, and the existing snapshot sync deletes old Qdrant IDs
not present in the replacement and replaces the BM25 corpus. Complete prior
run archives are retained. Canonical versions are retained in `archive_data/02_clean/base_data`; other stages
do not maintain baseline folders.
Use `POST /admin/pipeline/reset-data` to upload and start the same mode through
the API. Run it in a maintenance window and restart the API after completion.

In join mode, `shared_data/01_raw/chatbot_dataset_deleted.parquet` is included if
present. An explicit `--deleted <path>` selects another deletion file. Collection
mode takes the NEW and DELETED files from the published collection shared batch,
checking its run ID and inventory hashes. Deleted records require `document_id`, or `id` and
`source_type`; all generated identities match chunking. Deletions win if the same
document is also present in the incoming data. Remove or replace an old deletion
file before importing data for which those deletions should no longer apply.

The coordinator snapshots configs into
`runtime_data/pipeline_runs/run_...`, runs stages one at a time with the active
virtual environment's Python, and records per-stage logs/checkpoints there.
Stages discover inputs from shared_data themselves; canonicalization records raw
request hashes in its own archive `.requests` folder. Full canonical baselines
and publication recovery remain archive responsibilities of the owning stage.
The terminal shows short stage progress rather than all model output. A failure
stops later stages and prints the command to resume:

```powershell
python run_pipeline.py --resume "C:/path/to/runtime_data/pipeline_runs/run_..."
```

A pending run blocks new batches. Completed archives are republished instead of
reprocessed on resume, so a publication failure does not increment the version
again or repeat model inference. Do not run standalone stage writers concurrently
with the coordinator. The global lock prevents overlapping coordinator runs.
An empty NEW and DELETED batch ends as `no_changes`; a deletion-only batch
updates the canonical base and index without loading chunking/embedding models.
Its empty batch is still archived and published in 02_clean. An empty resulting chunk
corpus is currently rejected; deleting the entire searchable corpus is not supported.

`python run_pipeline.py --cancel-pending` can cancel only before any stage command
has been attempted. Once attempted, resume the pending run: the owning stage may
have committed data before a process failure. New checkpoints use coordinator
protocol 2. Finish unfinished legacy runs using the previous release before
upgrading; completed historical runs remain readable.

Canonicalization publishes the cleaned batch under its input-derived filename
and archives the full merged/replacement version separately. Chunking processes
only that batch and publishes input-derived filenames in `shared_data/03_chunks`
and a new `archive_data/03_chunks/run_...`. Embedding similarly processes only
batch chunks, publishing batch vectors/mapping/metadata to
`shared_data/04_embeddings` and `archive_data/04_embeddings/run_...`.
Neither stage creates a base_data directory or needs full baseline embeddings.

For join/collect, final synchronization preserves untouched indexed documents,
replaces all prior chunks of incoming documents, and removes documents deleted
from the cumulative canonical base. Only batch vectors are upserted. Reset
preserves no prior documents. Both paths verify the exact final chunk ID set
and count, then publish a full BM25 corpus and `active_dataset.json` under the
new cumulative version in shared_data/05_indexes and archive_data/05_indexes.
Stable IDs and retained original index files make interrupted upserts/deletions
retryable. Embeddings in shared_data describe the batch, not the full index.
After constructing the new full corpus and active pointer, sync removes old
dataset index artifacts from staging. Only the current version is archived and
published; a failed merge/prune leaves the previous shared version intact.
A fresh Qdrant installation after batch updates requires a Qdrant backup or
rebuilding embeddings from the complete canonical base; the bootstrap helper
refuses to mistake a batch for a complete vector snapshot.

Qdrant writes and local publication are not one transaction: searches during sync
may temporarily observe a mixture of versions. Run the batch in a maintenance
window when a consistent view is required.

`retrieval-search` without `--dataset` uses the active published retrieval version.
Clear a stale explicit selection first (environment overrides the active pointer):

```powershell
Remove-Item Env:PIPELINE_DATASET -ErrorAction SilentlyContinue
retrieval-search --query "UNEC-də magistratura proqramları hansılardır?" --limit 5
```

The existing collection tracker compares WordPress IDs against its fixed baseline;
it does not detect content edits to existing WordPress IDs. This pipeline retains
that behavior; supplied raw datasets can replace existing documents by identity.
Deletion scope is the records emitted by the collector, not every absent dataset row.

## One Windows scheduled task

After the manual integration test succeeds, register one task (the script is
provided but is not executed automatically):

```powershell
.\scripts\Register-PipelineTask.ps1 -At '03:00'
```

It invokes `.venv/Scripts/python.exe run_pipeline.py --mode collect --resume-pending`
daily at Windows local time, resumes a pending batch before collecting another,
and ignores overlapping triggers. This configuration runs while the registering
user is logged in; Docker/Qdrant must be running. For execution while logged out,
configure the task's account/logon settings in Task Scheduler and ensure Qdrant
is available to that account. `-Replace` explicitly updates an existing task.
The task uses Microsoft's [ScheduledTasks action](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtaskaction)
and [task registration](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/register-scheduledtask)
interfaces.

Read-only plan and local tests (run manually):

```powershell
python run_pipeline.py --plan
python -m pytest orchestration/tests/test_pipeline_coordinator.py 05_retrieval/tests/test_snapshot_sync.py 04_embedding/tests/test_incremental_embedding.py -q
```
