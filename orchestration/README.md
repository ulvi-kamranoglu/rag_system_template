# Pipeline orchestration

Coordinates installed collection, canonicalization, chunking, embedding and
retrieval stages through their CLIs. Owns checkpoints, cancellation and resume;
does not duplicate stage implementations or load inference models.

## Layout

- `src/orchestration/`: CLI, run state, configuration snapshots and stage runner.
- `tests/`: offline coordinator contract tests.
- `pyproject.toml`: package dependencies and `chatbot-pipeline` entrypoint.
- `../pipeline.yaml`: workspace-level orchestration configuration.

## Commands

```powershell
chatbot-pipeline --plan
python run_pipeline.py --mode collect
python run_pipeline.py --mode join --input shared_data/01_raw/example.parquet
python run_pipeline.py --mode reset_data --input shared_data/01_raw/replacement.parquet
python -m orchestration --resume "path/to/run"
python -m pytest orchestration/tests -q
```

Each stage now owns its own inputs, modes, publication and recovery. The
coordinator only sequences configurable `stage_commands` and validates receipts;
it does not import stage packages or choose intermediate files. Standalone usage,
replacement-stage requirements and legacy checkpoint compatibility are described
in [the independent stage contract](../docs/stage-contract.md).

All commands keep the existing collect/join/reset_data data behavior. See
[the pipeline guide](../docs/pipeline.md) for publication and recovery rules.
