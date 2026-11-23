# Active data

The five numbered directories contain current stage outputs. The API follows
`05_indexes/active_dataset.json` unless an explicit dataset overrides it.
The included raw example is not automatically ingested; administrators select
inputs. Update prepared outputs with the pipeline, not manual edits.
Qdrant persistent storage is external and must be provisioned separately.

The 02_clean, 03_chunks and 04_embeddings folders hold only the most recent
batch, with names derived from its raw input. Full versioned canonical datasets
are retained only in archive_data/02_clean/base_data. The active 05_indexes
corpus is cumulative for collect/join and replaced for reset_data.
After each successful index sync, 05_indexes contains only the active dataset's
index artifacts, active_dataset.json and run_report.json. The preceding corpus
is used during merge, then obsolete versions are removed from staging before
the archive and shared publication are committed.
