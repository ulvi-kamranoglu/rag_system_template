# PowerShell command reference

From 01_data_collection after activating its environment:

```powershell
python -m pip install -e ../storage_common -e ".[dev]"
python -m pytest -q
python -m wp_tracker status
python -m wp_tracker scan
python -m wp_tracker scan --run
python -m wp_tracker sync-shared
python -m wp_tracker recover
```

For an explicit first publication migration of shared legacy run directories:

```powershell
python -m wp_tracker scan --run --migrate-legacy-shared
# Or retry an already committed archive:
python -m wp_tracker sync-shared --migrate-legacy-shared
```

For a new, uninitialized project only:

```powershell
python -m wp_tracker bootstrap
python -m wp_tracker bootstrap --run
# Alternative approved import paths:
python -m wp_tracker bootstrap --pages C:/data/pages.parquet --posts C:/data/posts.parquet --run
```

`bootstrap --reset --run` deliberately replaces the fixed baseline and snapshot;
back up state first. A storage transition never requires a baseline reset.
Investigate abnormal deletions before adding `--allow-large-deletions` to scan.

Inspect centralized data and internal state:

```powershell
Get-Content .\data\state\latest.json
Get-ChildItem ..\shared_data\01_raw
Get-ChildItem ..\archive_data\01_raw -Directory -Filter 'run_*'
```

These paths show YAML defaults; custom paths come from `config/settings.yaml`.
To select a config from any working directory use its absolute path.

For scheduling the complete pipeline, run from the workspace root:

```powershell
python run_pipeline.py --mode collect
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/Register-PipelineTask.ps1 -At '03:00'
Get-ScheduledTask -TaskName 'Chatbot-Data-Pipeline'
```

See [pipeline scheduling](../../docs/pipeline.md). Existing Windows tasks are not
changed automatically when files are cleaned up.
