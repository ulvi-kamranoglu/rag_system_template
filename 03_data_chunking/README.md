# Retrieval Preparation and Chunking

Standalone paragraph- and sentence-preserving, token-aware chunking. Source
identities and metadata are preserved. Stable chunk IDs use
`source_type:id:chunk:index`; existing source prefixes are retained once.
The BGE-M3 tokenizer enforces final embedding token limits. Titles are kept
complete in the repeated prefix while at least `min_content_budget` tokens
(default 32) remain for content. There is no separate title token cap.
If a title prefix leaves less room than the configured overlap, overlap is
reduced for that document and recorded as `overlap_tokens_effective`. If the
prefix leaves less than the minimum content budget, the complete title is
chunked together with the body instead of being repeated as a prefix. Original
title metadata and all source text are preserved; final embedding limits still
apply.

## Install and run

Server installation uses the shared root environment; see
[deployment](../docs/deployment.md). The following commands are an optional
standalone development setup inside `03_data_chunking`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ../storage_common -e ".[test]"
python -m pytest -q
chunk-pipeline --dry-run
chunk-pipeline
chunk-pipeline --input C:/data/one.parquet C:/data/two.parquet
```

Default configuration is resolved from the editable source regardless of the
terminal directory. `--config C:/path/to/config/config.yaml` selects another
config and is required for a non-editable installation. Relative input/storage
paths are resolved against that config's project root. No other pipeline must
be installed in this project's environment; storage_common is a small local
Python dependency shared as source by the three projects.

## Central storage and dataset layout

```yaml
input_dir: ../shared_data/02_clean
storage:
  archive_dir: ../archive_data/03_chunks
  shared_dir: ../shared_data/03_chunks
```

With no explicit `inputs`, the only `*_canonical.parquet` file in `input_dir`
is selected; multiple candidates require an explicit selection.
`inputs` supports multiple datasets in one run. `--input` overrides that list;
the previous single `input_path` config key is also supported. Input stems must
be unique, and no input may reside inside the output/archive roots.

Every successful batch creates a unique UTC archive:

```text
archive_data/03_chunks/run_YYYYMMDD_HHMMSS_<unique>/
    run_report.json
    <dataset>_retrieval_prepared.parquet
    <dataset>_chunks_256.parquet
    <dataset>_chunks_384.parquet
    <dataset>_chunks_512.parquet
    reports/
        <dataset>_chunking_256_summary.json
        <dataset>_chunking_384_summary.json
        <dataset>_chunking_512_summary.json
```

Multiple inputs are combined into one prepared dataset and one chunk file per
variant. Source metadata and stable IDs are preserved; duplicate chunk IDs across
inputs are rejected before outputs are written. Summary counts cover all inputs.
The exact same tree is published directly into `shared_data/03_chunks`, without
a run directory or dataset-name subdirectory. Rerunning with fewer inputs or
variants removes stale shared artifacts and previous dataset subdirectories
while retaining every historical archive. No new files are
written to project-local `data/output` or `reports`. The module has no local `data`
directory; former inputs/outputs are preserved under
`archive_data/legacy_project_data/03_data_chunking/data`.

All datasets and variants must process and validate before the final archive is
visible or shared is replaced. A dry-run performs the same preparation and
chunking and shows planned destinations but writes no outputs/checkpoints.
It can populate the tokenizer's normal external cache on first use.

Publication uses validated staging, byte verification, directory renames and a
rollback journal. Windows readers can briefly see a missing shared directory,
but no partly copied run. Complete archives survive publication errors and can
be retried without loading the tokenizer:

```powershell
chunk-pipeline --publish-run "C:/path/to/archive_data/03_chunks/run_<timestamp>_<unique>"
```

Old archives with dataset-name subdirectories remain valid for retry and are
published with their original layout. New processing runs always use the flat
layout shown above; existing archives are not rewritten.

Legacy shared run folders require explicit `--migrate-legacy-shared` to preserve
the entire previous tree at a unique hidden sibling backup before publication.
No project-local legacy folders are moved. See [storage guarantees](../storage_common/README.md).

## Processing contract

Required input columns: `id`, `source_type`, `title`, `content`. Optional source
fields (`url`, `date`, `modified`, `source`, `language`, `metadata`, `search_text`)
are copied when present. Source identities must be unique across the input batch
so combined chunk IDs remain unique.
No canonical ID regeneration or cross-run ID map is introduced.

Retrieval preparation cleans HTML/shortcodes/whitespace and applies existing
eligibility rules. Chunking preserves paragraph and sentence boundaries where
possible and splits oversized sentences with a token limit. Whole trailing
sentences provide overlap when they fit. FAQ/form titles prefix the answer's
embedding text. `search_text` is preserved; retrieval uses `embedding_text`.

The default is BAAI/bge-m3, 384 maximum tokens and 64 overlap.
Additional token variants can be configured explicitly.
There is no incremental skip/append or resume checkpoint. Source documents must
remain available downstream because chunk outputs do not contain full content.
The tokenizer needs an initial download unless already cached; tests use an
offline tokenizer and exercise multiple datasets and variants.

On Windows, the package also installs `tzdata` so UTC timestamps from canonical
Parquet files can be converted to Python datetimes without a system IANA timezone
database. After updating dependencies, rerun the installation command above.
