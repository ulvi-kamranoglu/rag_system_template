# Collection source contract

Collection owns all source execution, state, aggregation and recovery. Orchestration
calls only `python -m wp_tracker.stage_cli --mode collect`; it knows nothing about
Instagram profiles, News, source baselines or intermediate archives.

## State and comparison

Five independent sources live under the configured collection state directory:

```text
data/state/
  wordpress/pages/{baseline.parquet,snapshot.parquet,latest.json}
  wordpress/posts/{baseline.parquet,snapshot.parquet,latest.json}
  instagram/uneceduaz/{baseline.parquet,snapshot.parquet,cutoff.json,...}
  instagram/adalat.muradov/{baseline.parquet,snapshot.parquet,cutoff.json,...}
  news/news.unec.edu.az/{baseline.parquet,snapshot.parquet,latest.json}
  collection/latest.json
```

Baselines never advance during scans. For WordPress and News, NEW is the current
snapshot minus its fixed baseline identities; a subsequent unchanged scan emits
the same baseline-relative additions. Instagram is different: NEW contains only
posts newer than the newest post already collected (`cutoff.json`), so each post
is delivered once; its baseline is the immutable seed record.
Instagram scans are partial, so their DELETED files are always empty. WordPress
and News see every record (REST API / sitemap) and produce baseline-relative
deletions under the shared `safety.max_deleted_fraction` guard.

WordPress pages and posts are separate source states even though both use the
same API client. On the first collection run, legacy combined WordPress baseline
and snapshot are partitioned by source_type into new namespaces, without changing
the originals. This requires both legacy files; use the existing WordPress
bootstrap command with approved pages/posts imports if starting from scratch.
Existing source states are preserved and never overwritten by migration.

Instagram IDs follow the canonicalization standard `instagram:<profile>:<N>`
(`instagram:uneceduaz:N`, `instagram:adalat.muradov:N`), numbered per profile
from the seed's maximum, so collection and canonical IDs are identical. The
scan uses the authenticated Playwright collector: known posts are never
reopened (their dates come from state), unknown posts are opened to read their
date, and only posts with `date > cutoff` are collected. The scan stops after
`boundary_old_posts` (default and minimum 4) consecutive non-pinned posts that
are not newer than the cutoff; Instagram allows at most three pinned posts,
which may be old. Pinned posts are still checked but never move the boundary.
Reaching the profile end counts only if the last post was old; a stall on newer
posts fails closed. A successful scan advances the cutoff to its newest post.
Unknown posts at/before cutoff are not backfilled.

News follows the WordPress model with no SQLite and no ID allocation. The ID is
the site's own article number from the URL (`/xeber/<category>/<number>-<slug>`):
`id = news:<number>`, `metadata.news_id = "<number>"`, like `page:<wp_id>`.
A scan reads `https://news.unec.edu.az/sitemap.xml` (one request for every
article; a sitemap index is followed), keeps snapshot rows for known numbers and
opens only new articles. A known article missing from the sitemap is deleted
only if its URL answers 404/410; a present page, redirect, server error or
timeout keeps it (reported as `unconfirmed_missing`). If the sitemap lacks more
than `max_deleted_fraction` of known articles, the scan stops before probing or
writing. An article that cannot be parsed is reported in `unparsed_new` and
retried on the next scan. Low-value NEW content is filtered like WordPress but
stays in the snapshot.

## Public output and internal archives

```text
shared_data/01_raw/
  chatbot_dataset_new.parquet
  chatbot_dataset_deleted.parquet
  run_report.json
archive_data/01_raw/
  run_<...>/  # same combined files and stage contract
  sources/
    wordpress/{pages,posts}/run_<...>/
    instagram/{uneceduaz,adalat.muradov}/run_<...>/
    news/news.unec.edu.az/run_<...>/
  .requests/  # input/config hashes and per-source completion checkpoints
```

Internal source publication copies are under `01_data_collection/data/output`,
not public shared_data. The full raw dataset contract remains the original
eleven columns. Aggregation normalizes timestamps to UTC naive datetime64[ns]; it preserves text, attribution and stable identities.
Duplicate source_type/id combinations and NEW/DELETED overlaps fail publication.

A required source failure leaves the previous public RAW intact. Native source
transactions may already have committed their own snapshots; they are checkpointed
and reused on retry. The combined archive commits only after all sources validate.
The public stage contract and hash inventory are written after publication, just
as before. Other pipeline stages and the API require no source-specific changes.

## Setup (not performed automatically)

The supplied YAML enables WordPress. Instagram and News are implemented but
disabled until their own datasets and sessions are ready. No ZIP credentials,
foreign machine pointers or production snapshots are imported.

Install Instagram's optional dependency and browser in the chosen environment:

```powershell
python -m pip install -e './01_data_collection[instagram]'
python -m playwright install chromium
```

All seeds go into one folder, `01_data_collection/data/import/` (gitignored).
Each must be exactly the dataset already joined into the index, because a
baseline record is never sent again as NEW:

```text
data/import/
  instagram_uneceduaz.parquet
  instagram_adalat_muradov.parquet
  news.parquet
  pages.parquet, posts.parquet   # only to rebuild WordPress from scratch
```

Bootstrap reads these default paths (`--seed` overrides). Omit `--run` first to
validate without writing state:

```powershell
python -m wp_tracker instagram bootstrap --profile uneceduaz --run
python -m wp_tracker instagram bootstrap --profile adalat.muradov --run
python -m wp_tracker instagram login --profile uneceduaz
python -m wp_tracker instagram login --profile adalat.muradov
python -m wp_tracker news bootstrap --run
```

Bootstrap creates both the fixed baseline and initial snapshot and refuses an
initialized source. Original seed files remain unchanged. News seed IDs are
rebuilt from each URL number (legacy sequential IDs are replaced; a `news:<n>`
ID or `metadata.news_id` that disagrees with the URL is rejected). Instagram seeds may use
`instagram:<profile>:N`, `research:N`, `research:<profile>:N` or, for
uneceduaz only, the legacy `instagram:N`; all are converted to
`instagram:<profile>:N`. The cutoff is the newest non-synthetic seed date.

Set `instagram.enabled: true` and `news.enabled: true` in the selected collection
YAML after setup. Configure profiles/boundary limits there. Canonicalization and
orchestration configurations do not need Instagram/News sections.

```powershell
python -m wp_tracker scan  # no HTTP, only state/readiness paths
python -m wp_tracker.stage_cli --mode collect --run-id my-batch
```

Individual `instagram scan --run` and `news scan --run` update only their internal
source outputs; use the public collection stage to deliver a downstream batch.
Because Instagram NEW is cutoff-relative, a standalone `instagram scan --run`
advances the cutoff and the next collection will not deliver those posts again;
use it only for diagnostics on a disposable state.
They must not compete with an active unified collection run. Session JSON files
and source scratch outputs are ignored by Git; all shared/archive outputs remain
trackable as requested. The original distribution ZIP is also ignored because it
contains session files.

## Recovery and verification

Repeat the printed stage command with the same run ID, or resume the pipeline.
Already committed sources are republished/validated without scraping again.
Missing or orphaned native source state fails for recovery rather than rescanning
over an unacknowledged batch. Changed config or baseline on an unfinished run is
rejected using the pinned request hashes. Source archive inventories are also
checked before the combined batch commits.

After combined archive commit, retry is local-only: no website calls or browser.
`python -m wp_tracker sync-shared` retries the last combined collection pointer.
Live selectors, authentication and server Chromium dependencies still need an
operator-run site check; offline tests do not establish live scraper readiness.

Tests cover isolated state, baseline immutability, repeated baseline-relative
NEW, confirmed WordPress/News DELETED, sitemap safety stops, partial-source failure/resume,
publication failure and unchanged downstream stage contracts. HTTP/browser
responses are mocked and production RAW/indexes are not modified.
