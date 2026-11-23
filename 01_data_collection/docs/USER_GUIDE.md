# Collector user guide

The [project README](../README.md) is the authoritative storage and event
contract, including bootstrap, validation, publication recovery, legacy
migration and scheduler operation.

Existing initialized workspaces keep their approved imports, baseline and
snapshot. Do not bootstrap/reset simply to change storage paths. Run `status`,
then `scan` to preview destinations. `scan --run` archives the complete run,
commits internal state, and publishes latest outputs. For this workspace's
legacy shared run folder, explicitly use `--migrate-legacy-shared` on the first
publication if you want the preserved sibling backup migration.

Latest downstream inputs are `../shared_data/01_raw/chatbot_dataset_new.parquet`
and `chatbot_dataset_deleted.parquet`. Run history and reports are under the
configured `archive_data/01_raw/run_*/` folders. `data/state/latest.json` identifies
the latest committed archive and retry source. Old project-local outputs and
`data/reports/latest.json` remain historical and are no longer updated.

If publication fails, use `python -m wp_tracker sync-shared`; if it reports a
pending state transaction, first run `python -m wp_tracker recover`. The next
live scan retries the last committed archive before contacting WordPress.
No pagination checkpoint exists; incomplete API scans are fetched again in full.

Repeated NEW/DELETED rows across scans are expected because comparisons use the
fixed baseline. NEW filters do not remove IDs from the snapshot. DELETED means
absent from the public published API, not necessarily permanently deleted.

For commands see [PowerShell reference](POWERSHELL_COMMANDS.md).
