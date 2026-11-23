# UNEC Data Collection

The public collection stage combines three independent collectors: WordPress
pages/posts, Instagram (`uneceduaz`, `adalat.muradov`) and UNEC News. Each of the
five sources has its own immutable baseline and advancing snapshot. WordPress
and News NEW outputs compare against the fixed baseline, not the previous
snapshot. Instagram NEW contains only posts newer than the newest collected post,
with canonical IDs `instagram:<profile>:<N>`. News reads the site sitemap, uses
the URL article number as `news:<number>` and deletes only after a 404/410 check.
All bootstrap seeds live in `data/import/` (see docs/COLLECTION_SOURCES.md).

Only collection publishes the combined `chatbot_dataset_new.parquet` and
`chatbot_dataset_deleted.parquet` to `shared_data/01_raw` and an immutable
`archive_data/01_raw/run_*`. Source internals stay in source-specific state/output
folders and `archive_data/01_raw/sources`; no `01_instagram`, `01_news` or
`01_raw_unified` namespaces are required. The existing stage contract is unchanged.

See [source setup, architecture and recovery](docs/COLLECTION_SOURCES.md).
Instagram and News are disabled in the shipped configuration until their own
baseline/bootstrap and (for Instagram) browser sessions are prepared. Enabling an
uninitialized required source fails before scraping or replacing public RAW.

```powershell
python -m wp_tracker status
python -m wp_tracker scan  # Read-only readiness, no HTTP
python -m wp_tracker.stage_cli --mode collect
```

The remaining WordPress details describe its underlying collector and legacy
combined seed/bootstrap. The public scan now runs all enabled collectors.

Collect public published Pages and Posts using complete, validated WordPress API
scans. IDs are compared against an explicitly bootstrapped **fixed baseline**.
Existing IDs changing title/content do not generate events. Page and post IDs
remain distinct. The same baseline-relative NEW/DELETED records can recur in
later runs; scanning never advances the baseline.

## Install and run

For server delivery, install from the project root into the shared environment
as described in [deployment](../docs/deployment.md). The following is an optional
standalone development setup, not an additional server environment requirement:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ../storage_common -e ".[dev]"
python -m pytest -q
python -m wp_tracker status
python -m wp_tracker scan                 # Dry-run, includes planned destinations
python -m wp_tracker scan --run
```

Default configuration is located from the package source, independently of the
terminal directory. `--config C:/path/to/config/settings.yaml` overrides it.
Relative storage/import paths are resolved against that configuration's project
root (`config/..`). Archive and shared roots must be outside the project and
must not overlap. The shared storage package is an ordinary local dependency;
the module can also execute independently of the other pipeline packages.
For non-editable installs, supply `--config` explicitly.

## Central outputs

`storage.archive_dir` defaults to `../archive_data/01_raw` and
`storage.shared_dir` to `../shared_data/01_raw`. A complete run produces:

```text
archive_data/01_raw/run_YYYYMMDD_HHMMSS_<unique>/
    chatbot_dataset_new.parquet
    chatbot_dataset_deleted.parquet
    latest.json
    run_report.json
shared_data/01_raw/
    chatbot_dataset_new.parquet
    chatbot_dataset_deleted.parquet
    latest.json
    run_report.json
```

Every completed archive is retained. Shared contains one complete published run,
with stale files removed on replacement and no run subdirectories. Outputs and
reports are staged, validated, and byte-verified. Publication uses same-parent
renames, a publisher lock, and recovery journal. On Windows the swap can briefly
make the shared directory unavailable; it never exposes a partly copied run.
See [storage guarantees](../storage_common/README.md).

No new files are generated in project-local `data/output`, `data/history`, or
`data/reports`. Existing files there are left untouched. Run reports live with
the datasets. `data/state/latest.json` is the internal commit/retry pointer, not
proof of successful shared delivery. The old `data/reports/latest.json` is a
legacy report and is no longer advanced.

## Baseline and checkpoint safety

Existing `data/state/baseline.parquet` and `snapshot.parquet` remain in place.
The baseline is fixed; snapshot records the last complete API scan, including
IDs rejected by NEW quality filters. There is no incremental pagination resume
checkpoint and no cumulative event append operation in this implementation.
Append-only history is preserved through immutable archive run directories.

Only for an uninitialized project, place approved original `pages.parquet` and
`posts.parquet` in `data/import`, then:

```powershell
python -m wp_tracker bootstrap            # Preview
python -m wp_tracker bootstrap --run
```

Bootstrap archives and publishes two empty event datasets and its reports.
It refuses existing initialized state without explicit `--reset`. Back up state
before a deliberate reset; do not use a live snapshot as the original baseline.
Imports are never modified. `--pages` and `--posts` accept alternate input paths.

The collector finalizes its archive **before** updating snapshot and the retry
pointer together through the existing recoverable multi-file transaction. If
publication fails, the archive and pointer survive. The command exits with an
error. The next live scan must finish publishing that pending committed archive
before it may collect again. Retry without contacting WordPress:

```powershell
python -m wp_tracker sync-shared
python -m wp_tracker recover              # Roll back interrupted state writes
```

`sync-shared` publishes only the latest committed archive; it never replays
historical runs over the latest dataset. State-commit failures also retain the
complete archive for inspection, while the previous committed state is restored.
Dry-runs process and validate data, show proposed archive/shared destinations,
and do not create output folders, locks, or modify baseline/snapshot/reports.

## Legacy shared directories

Legacy project-local data is not automatically moved or deleted. Existing
historical run directories in shared storage block publication unless you
explicitly opt in:

```powershell
python -m wp_tracker scan --run --migrate-legacy-shared
# If the archive is already committed after a publication failure:
python -m wp_tracker sync-shared --migrate-legacy-shared
```

The entire previous shared tree is retained in a unique hidden sibling backup,
for example `shared_data/.01_raw.legacy_<uuid>`. Backups are never automatically
pruned. No migration runs merely by installing the code. Do not reset your
existing baseline to perform this storage transition.

## Event contract and safeguards

Outputs retain the 11 columns:

```text
id, source_type, title, content, url, date, modified,
source, language, search_text, metadata
```

IDs are `page:<wp_id>` / `post:<wp_id>`, and `metadata` retains `wp_id` as JSON.
`search_text` equals projected content. NEW records have IDs absent from the
baseline and normalized content that is neither empty, at most five
non-whitespace characters, nor URL-only. HTML/entities and Unicode/whitespace
are normalized while preserving paragraphs. Reports distinguish candidates,
accepted rows, and rejection reasons. Quality filters never hide snapshot IDs
or filter DELETED rows, whose fields come from the baseline.

DELETED means absent from the public published API, including draft/private/
trashed items; it does not establish permanent deletion. Both endpoints must
pass pagination totals, unique-ID and response validation before any new
archive/state is committed. A disappearance exceeding 5% of either baseline
source type stops the run. Investigate before using `--allow-large-deletions`.

## Windows scheduler

Use the workspace scheduler for the complete pipeline, not a collection-only task.
From the workspace root:

```powershell
python run_pipeline.py --mode collect
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/Register-PipelineTask.ps1 -At "03:00"
```

Stage logs and checkpoints are managed by orchestration. See
[the pipeline guide](../docs/pipeline.md) and the collection-specific
[User Guide](docs/USER_GUIDE.md) and [PowerShell commands](docs/POWERSHELL_COMMANDS.md).
