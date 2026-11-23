# UNEC Data Cleaning and Canonicalization

Read-only batch processing between Collection and Chunking. One input produces
one canonical dataset; multiple inputs are cleaned and merged into one dataset.
HTML/text cleaning, canonical identities, duplicate reporting, and validation
are preserved. Normalized title+content+URL duplicates are removed last and reported. Invalid rows are
reported and excluded; a nonempty input with all rows rejected is not published.

`date` and `modified` accept ISO strings, typed datetimes and Unix timestamps
in seconds, milliseconds, microseconds or nanoseconds. Numeric units are inferred
per value by magnitude, including mixed numeric units in a column. Invalid
values become missing. Typed dates are preserved: a suspicious 1970 date is not
automatically multiplied, since genuine historical dates must remain valid.
Already corrupted dates require an explicit correction backed by original data.

## Install and run

Server installation uses the shared root environment; see
[deployment](../docs/deployment.md). The following commands are an optional
standalone development setup inside `02_data_canonicalization`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ../storage_common -e ".[dev]"
python -m pytest -q
canonical-pipeline --dry-run
canonical-pipeline
```

`config/config.yaml` selects the only Parquet dataset in `../shared_data/01_raw`,
excluding `*_deleted.parquet`. Use `--input` (paths relative to the terminal)
when there are multiple candidates. Configure `inputs` as a
nonempty YAML list to merge multiple datasets. Raw inputs are never modified.
The default config is located from the editable source regardless of terminal
CWD; supply `--config C:/path/to/config/config.yaml` for another config or a
non-editable installation. Paths are relative to that config's project root.

## Central storage

```yaml
input_dir: ../shared_data/01_raw
storage:
  archive_dir: ../archive_data/02_clean
  shared_dir: ../shared_data/02_clean
```

Each run archives `<dataset>_canonical.parquet`, `run_report.json`, and
`reports/quality_report.json`, `reports/rejected_records.json`, and
`reports/duplicate_report.json` inside a unique UTC
`archive_data/02_clean/run_YYYYMMDD_HHMMSS_<unique>/` directory.
The same files are published directly into `shared_data/02_clean`, preserving
report filenames and structure. Shared holds only the latest complete run;
replacement removes stale artifacts. No project-local output/report is written.
The module has no local `data` directory. Its former inputs/outputs are preserved
under `archive_data/legacy_project_data/02_data_canonicalization/data`.

Validation and byte verification precede publication. Archives are immutable;
publication failure leaves a complete archive for retry. Directory replacement
is journaled and rolls back on failure. Windows readers can briefly encounter
a missing shared directory during a swap, never a partially copied dataset.
See [storage guarantees](../storage_common/README.md).

A dry-run performs cleaning, identity validation, and duplicate detection and
prints proposed archive and shared destinations. It writes no outputs or state.
To retry an archived run without reprocessing:

```powershell
canonical-pipeline --publish-run "C:/path/to/archive_data/02_clean/run_<timestamp>_<unique>"
```

Legacy historical run folders in shared storage block publication by default.
`--migrate-legacy-shared` explicitly preserves the previous shared tree at a
unique hidden sibling backup before replacing it. No project-local legacy
folder is moved, and no backup is automatically pruned.

## Output contract

Existing IDs whose namespace differs from `source_type` are retained intact;
the mismatch does not reject a row or rewrite its ID. WordPress numeric ID and
`wp_id` matching checks apply only when the identifier uses the matching source
namespace (or originally has no namespace). Other metadata, URL and language
normalization still applies.

Columns remain `id`, `source_type`, `title`, `content`, `url`, `date`, `modified`,
`source`, `language`, `search_text`, `metadata`. `search_text` equals cleaned
content. WordPress IDs use `page:<id>` / `post:<id>`; other source adapters use
source-type prefixes. Inputs cannot reside inside the output/archive directories.

There is no resume checkpoint. Date strings without
a supplied timezone use UTC. Table-heavy HTML and source identity conflicts
should be reviewed using the reports before downstream chunking.

## Text quality and deduplication

Processing order is HTML extraction, Unicode/encoding repair, content validation,
language detection, source identity validation, then deduplication across all inputs.
The existing 11-column output schema, ID rules, URL tracking removal, metadata
validation, UTC date handling, source-derived filenames, dry-run, immutable archives,
publication rollback and legacy archive retry behavior are retained.

- BeautifulSoup extracts text from malformed and encoded HTML, preserves paragraph
  and list boundaries, and removes script/style/iframe/form content as before.
- `ftfy` repairs recoverable mojibake. Unicode is normalized to NFC, Unicode spaces
  are normalized, and formatting/control characters are removed while keeping newlines.
  Literal U+FFFD characters cannot reliably be reconstructed and are reported.
- Blank, URL-only (including `www.` and multiple URLs), non-alphanumeric and too-short
  content is rejected after cleaning. An empty title alone does not reject useful content.
- `lingua-language-detector` detects `az`, `en` or `ru` from cleaned title and content,
  replacing the original language field. Low-confidence/possibly mixed cases retain
  the best supported label and are flagged with relative confidence in the text report;
  this three-language detector cannot guarantee recognition of unsupported languages.
  Rows with no detectable language are rejected rather than assigned a default.
- Deduplication compares NFC text with collapsed whitespace and case folding.
  Title, content and canonicalized URL must all match. If any of the three differs,
  both rows are retained. Text comparison uses case folding; URL paths and query
  values remain case-sensitive. Existing URL tracking-parameter removal still applies.
  Keep the first valid input row and record discarded IDs and their retained counterpart.
  URL matches alone do not cause deletion. Identity conflicts remain reported.

`quality.min_content_chars` in `config/config.yaml` defaults to **6 non-whitespace
characters**; this deliberately conservative initial value can be overridden with
`--min-content-chars 20`, for example. Short FAQ answers below the selected limit
are also removed.

Reports include `rejected_records.json` (invalid rows), `duplicate_report.json`
(before/after counts and removal mappings) and `text_quality_report.json`
(normalization, language reassignment and warnings). Row indices are zero-based
indices in the merged source inputs. `input_rows = output_rows + rejected_rows +
duplicate_removed_rows`; normalization/language warning counters describe candidates
before deduplication, not just the final output.

From `Chatbot_projects`, manually install the added dependencies and run checks:

```powershell
python -m pip install -e ./02_data_canonicalization
python -m pytest 02_data_canonicalization/tests -q
python -m canonical_pipeline.cli --input shared_data/01_raw/chatbot_dataset_v1.3.parquet --dry-run
```
# Canonical ID policy

New uploads normalize IDs using their final positive integer suffix. Instagram
uses `<source>:<number>` (for example `instagram:adalat.muradov:1` or
`instagram:uneceduaz:1`). Other types use `<source_type>:<number>`, after source
aliases such as pages/page are normalized. A foreign old prefix is replaced;
existing numeric suffixes are preserved and leading zeros normalized. Missing
Instagram source or invalid/non-numeric IDs are rejected in quality reports;
no arbitrary numbers are assigned.

IDs are rebuilt on every run from each record's own fields, per source type;
the incoming `id` is only a fallback and must not contradict those fields:

| source_type | Canonical ID | Built from |
| --- | --- | --- |
| legal | `legal:<framework>:<clause>` | URL `/framework/<n>` + `metadata.clause_id`; else ID parts |
| page, post, content_post | `<type>:<wp_id>` | `metadata.wp_id`; else the ID's numeric suffix |
| instagram | `<source>:<number>` | collector-assigned number (starts at 1) |
| news, form, research, other | `<source_type>:<number>` | collector-assigned number |

Legal keeps both parts (`legal:43163:1.1`, `legal:27030:4.2-1`): the clause is
numeric segments separated by `.` or `-`, leading zeros normalized. Using only
the last number would merge clause 7 of different frameworks.

# Rejection guard

Rejected records are reported with a `category`. Identity/metadata errors
(ID, `wp_id`, `clause_id`, Instagram source) stop the stage before anything is
archived or published; `quality.max_identity_rejections` (default 0) allows
some. Low-value content (short, URL-only, undetectable language) is allowed up
to `max(content_rejection_tolerance, max_content_rejection_ratio * input rows)`,
default `max(5, 5%)`. A dry run reports `rejection_guard` without stopping.
The stage has pinned its inputs, so resuming the same run stops again. Known
limitation: the coordinator cannot yet cancel a run whose canonical attempt
failed without committing; run a dry run before large uploads.

Join matches incoming normalized identities to their legacy baseline aliases
so replacing a prefix does not add a second copy of the same source document.
Unrelated historical IDs and live indexes are not rewritten automatically.
Different Instagram accounts need distinct source values. The per-source
collection baselines retain their native raw identity format; canonical output
is normalized at this boundary.

