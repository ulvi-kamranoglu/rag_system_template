"""YAML settings with compatibility access for existing imports."""
from dataclasses import dataclass
import os
from pathlib import Path

import yaml
from pipeline_storage import (dataset_stem, resolve, artifact_dataset, storage_paths,
                              default_config, configuration_root)


@dataclass(frozen=True)
class Settings:
    config_path: Path
    project_root: Path
    dataset_name: str
    model_id: str
    embedding_dimension: int
    document_max_length: int
    query_max_length: int
    document_batch_size: int
    use_fp16: bool
    embedding_version: str
    chunks_input_path: Path
    embeddings_dir: Path
    embeddings_output_path: Path
    mapping_output_path: Path
    metadata_output_path: Path
    archive_dir: Path


def load_settings(config_path=None, *, input_override=None, dataset=None, for_publication=False) -> Settings:
    configured = config_path or os.getenv("EMBEDDING_CONFIG")
    path = (Path(configured).expanduser().resolve() if configured else
            default_config(__file__, "config/config.yaml"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Embedding config must contain a YAML mapping")
    root = configuration_root(path, __file__, "04_embedding")
    model = raw.get("model", {})
    storage = raw.get("storage", {})
    if not isinstance(model, dict) or not isinstance(storage, dict):
        raise ValueError("model and storage must be YAML mappings")
    def positive(key, default):
        value = model.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"model.{key} must be a positive integer")
        return value
    max_length = positive("document_max_length", 384)
    chunks_dir = resolve(root, raw.get("input_dir", "../shared_data/03_chunks"))
    name = dataset or os.getenv("PIPELINE_DATASET") or raw.get("dataset_name")
    explicit_input = None if for_publication else (input_override or raw.get("input_path"))
    if explicit_input:
        source = (Path(input_override).expanduser().resolve() if input_override else
                  resolve(root, explicit_input))
        inferred = dataset_stem(source)
        if name and name != inferred:
            raise ValueError("Selected dataset must match the chunk filename stem")
        name = inferred
    else:
        if not name:
            suffix = f"_chunks_{max_length}.parquet"
            candidates = sorted(chunks_dir.glob(f"*{suffix}"))
            if len(candidates) > 1:
                raise ValueError("Multiple chunk datasets found; use --input or --dataset")
            name = candidates[0].name[:-len(suffix)] if candidates else artifact_dataset(chunks_dir.parent)
        source = chunks_dir / f"{name}_chunks_{max_length}.parquet"
    if not isinstance(name, str) or not name or name in {".", ".."} or any(ch in name for ch in '/\\:*?"<>|'):
        raise ValueError("Dataset must be a filename stem, not a path")
    archive, destination = storage_paths(root, raw, "04_embeddings")
    if source.is_relative_to(destination) or source.is_relative_to(archive):
        raise ValueError("Embedding output must not contain or overwrite its input")
    fp16 = model.get("use_fp16", False)
    if not isinstance(fp16, bool):
        raise ValueError("model.use_fp16 must be true or false")
    return Settings(path, root, name, str(model.get("id", "BAAI/bge-m3")),
                    positive("dimension", 1024), max_length, positive("query_max_length", 128),
                    positive("batch_size", 4), fp16,
                    str(model.get("version", "bge-m3-dense-1024-v1")), source, destination,
                    destination / f"{name}_embeddings.npy",
                    destination / f"{name}_embedding_mapping.parquet",
                    destination / f"{name}_embedding_metadata.json", archive)


_LEGACY = {
    "DATASET_NAME": "dataset_name", "MODEL_ID": "model_id", "EMBEDDING_DIMENSION": "embedding_dimension",
    "DOCUMENT_MAX_LENGTH": "document_max_length", "QUERY_MAX_LENGTH": "query_max_length",
    "DOCUMENT_BATCH_SIZE": "document_batch_size", "USE_FP16": "use_fp16", "EMBEDDING_VERSION": "embedding_version",
    "CHUNKS_INPUT_PATH": "chunks_input_path", "EMBEDDINGS_DIR": "embeddings_dir",
    "EMBEDDINGS_OUTPUT_PATH": "embeddings_output_path", "MAPPING_OUTPUT_PATH": "mapping_output_path",
    "METADATA_OUTPUT_PATH": "metadata_output_path",
    "ARCHIVE_DIR": "archive_dir",
}


def __getattr__(name):
    if name in _LEGACY:
        return getattr(load_settings(), _LEGACY[name])
    if name in {"PROJECT_ROOT", "SHARED_DATA_DIR", "CHUNKS_DIR"}:
        settings = load_settings()
        return {"PROJECT_ROOT": settings.project_root.parent,
                "SHARED_DATA_DIR": settings.chunks_input_path.parent.parent,
                "CHUNKS_DIR": settings.chunks_input_path.parent}[name]
    raise AttributeError(name)
