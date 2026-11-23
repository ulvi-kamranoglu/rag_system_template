# Pipeline storage

One local Python package provides archive staging, byte verification, a publisher
lock, and journaled directory replacement for the three independently installed
projects. Install it into each project's virtual environment:

```powershell
python -m pip install -e ../storage_common -e .
```

`archive_run` yields a private staging folder and exposes the final unique UTC run
name only after validation. Existing final archives are never overwritten. A
completed archive remains available if publishing or a collector state commit
fails. Consumers must ignore dot-prefixed staging directories.

`publish` copies an archive to a same-parent shared staging directory, validates
it, verifies every byte using SHA-256, then replaces the shared directory as a
unit. This removes stale artifacts. Locks and publication journals live beside
the shared directory, not inside its dataset.

Windows cannot atomically replace a populated directory using one rename. Two
renames and a flushed journal provide rollback on errors and recovery after a
terminated process. Readers can briefly encounter a missing directory during a
swap, but never a partially copied directory. The next publish recovers the
journal before doing work. `recover_publication(shared_path)` is also available
for explicit recovery. This is filesystem recovery, not a distributed
transaction or a guarantee against arbitrary filesystem/power-loss corruption.

Processing and dry-run logic do not call the writer. Dry-runs may initialize a
tokenizer's normal external model cache but never write output archives, shared
datasets, or collector checkpoints.

Legacy shared `run_*`, `bootstrap_*`, and `previous_*` directories block
publication by default. Each CLI exposes `--migrate-legacy-shared` as an explicit
opt-in: retain the entire previous shared tree at a unique hidden sibling such
as `shared_data/.01_raw.legacy_<uuid>`, then publish the latest complete outputs.
No project-local legacy directory is moved, overwritten, or deleted. Backups are
not automatically pruned. This refactor does not execute a live migration.

Run package regressions using the collector environment:

```powershell
../01_data_collection/.venv/Scripts/python.exe -m pytest -q tests
```
