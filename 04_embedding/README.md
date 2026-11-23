# UNEC Embedding

An installable package providing a BGE-M3 query/document embedder and a YAML-configured
batch pipeline. Generated files belong in shared storage, outside this module.

From `Chatbot_projects`, using the selected Python environment:

```powershell
python -m pip install -e ./storage_common -e ./04_embedding
embedding-pipeline --dry-run
embedding-pipeline
```

Install CUDA PyTorch separately using the workspace `requirements-cuda130.txt`; see [workspace setup](../README.md).
The console command is available only after installing this package. Without
reinstalling, source-tree use remains possible with `python -m embedding.cli`;
`python -m embedding.pipeline` also delegates to the same CLI.

## Configuration

`04_embedding/config/config.yaml` controls model ID, dimension, document/query token
limits, batch size, FP16, version, input directory and shared output directory.
YAML paths resolve relative to the configuration project root (`config/..`).
The default configuration is packaged with the module and found independently of
terminal CWD. A non-editable installation should use an explicit workspace config
to avoid resolving workspace storage relative to site-packages.

```powershell
embedding-pipeline --config 04_embedding/config/config.yaml --dataset chatbot_dataset_v1.3 --dry-run
embedding-pipeline --input shared_data/03_chunks/chatbot_dataset_v1.3_chunks_384.parquet
```

CLI `--input` paths resolve from the terminal CWD. Dataset selection uses CLI
`--dataset`, then `PIPELINE_DATASET`, then YAML `dataset_name`, then unique-file
discovery. An explicit input supplies its dataset stem and must agree with a
selected dataset. Multiple candidates require explicit selection.
`EMBEDDING_CONFIG` can select a YAML for both the batch pipeline and existing
`BgeM3Embedder()` query users; explicit CLI `--config` takes priority for that run.

Dry-run reads/validates the chunk input and checks provided token counts, but does
not import the embedding model, download weights, compute vectors or write outputs.
It does not independently retokenize input strings. Actual runs retain model warm-up,
vector validation, L2 normalization and the dataset-derived NPY/Parquet/JSON names.
Outputs are archived and published as one validated set, as described below.

## Archive and publication

```yaml
storage:
  archive_dir: ../archive_data/04_embeddings
  shared_dir: ../shared_data/04_embeddings
```

An actual run first writes a private staging directory and validates the serialized
vectors, mapping alignment, metadata, dimensions, finite values and L2 norms.
Only then does it become a completed immutable archive:

```text
archive_data/04_embeddings/run_<UTC timestamp>_<uuid>/
    <dataset>_embeddings.npy
    <dataset>_embedding_mapping.parquet
    <dataset>_embedding_metadata.json
    run_report.json
```

The same complete set is then published to `shared_data/04_embeddings`. Shared
storage contains the latest successful run, including removal of stale files from
older datasets; archives retain every completed run. This does not merge datasets.
Publication uses the same lock, byte verification, journal and rollback mechanism
as canonicalization/chunking. On Windows there can be a brief missing directory
between renames, but partial file sets are not intentionally published.

A failure before archive completion leaves shared outputs untouched. A publication
failure retains the completed archive and prints a retry command:

```powershell
embedding-pipeline --config 04_embedding/config/config.yaml --publish-run "C:/path/to/archive_data/04_embeddings/run_<timestamp>_<uuid>"
```

Retry validates and copies the existing archive without model loading, GPU work or
the original chunk file. It cannot be combined with `--dry-run` or `--input`.
Legacy shared run directories require explicit `--migrate-legacy-shared` and are
retained in a sibling backup. Loose legacy outputs without a run report are also
preserved in a sibling backup by the shared storage helper.
The existing filesystem recovery rules are documented in `storage_common/README.md`.
This is publication retry, not resumption of interrupted embedding computation.

For application use:

```python
from embedding.config import load_settings
from embedding.embedder import BgeM3Embedder

settings = load_settings("04_embedding/config/config.yaml")
embedder = BgeM3Embedder(settings=settings)
vector = embedder.embed_query("UNEC haqqında məlumat")
```

The legacy constants imported from `embedding.config` remain available, resolved
from default settings. New code should pass a Settings object explicitly.

Manual model-free checks, from the workspace root:

```powershell
python -m pytest 04_embedding/tests -q
```
