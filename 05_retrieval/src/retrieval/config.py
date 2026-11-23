"""YAML configuration; preserve existing constant imports."""
import os
import json
from pathlib import Path
import yaml
from pipeline_storage import artifact_dataset, default_config, workspace_root


def load_settings(config_path=None, *, project_root=None, dataset=None):
    selected = config_path or os.getenv("RETRIEVAL_CONFIG")
    path = Path(selected).expanduser().resolve() if selected else default_config(__file__, "config/config.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Retrieval config must be a YAML mapping")
    source_root = workspace_root(__file__)
    value = project_root or os.getenv("CHATBOT_PROJECT_ROOT") or raw.get("project_root")
    root = Path(value).expanduser() if value else source_root if (source_root / "shared_data").is_dir() else Path.cwd()
    root = (root if root.is_absolute() else path.parent / root).resolve()
    def location(value):
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else root / candidate).resolve()
    sections = {}
    for key in ("storage", "qdrant", "search", "reranker"):
        sections[key] = raw.get(key, {})
        if not isinstance(sections[key], dict):
            raise ValueError(f"{key} must be a YAML mapping")
    storage = sections["storage"]
    shared = location(storage.get("shared_data_dir", "shared_data"))
    indexes = location(storage.get("shared_dir", "shared_data/05_indexes"))
    archive = location(storage.get("archive_dir", "archive_data/05_indexes"))
    embeddings = location(raw.get("input_dir", "shared_data/04_embeddings"))
    if indexes == archive or indexes.is_relative_to(archive) or archive.is_relative_to(indexes):
        raise ValueError("Archive and shared index roots must not overlap")
    if any(embeddings == p or embeddings.is_relative_to(p) for p in (archive, indexes)):
        raise ValueError("Outputs must not contain embedding inputs")
    name = dataset or os.getenv("PIPELINE_DATASET") or raw.get("dataset_name")
    if not name and (indexes / "active_dataset.json").is_file():
        name = json.loads((indexes / "active_dataset.json").read_text(encoding="utf-8"))["dataset_name"]
    if not name:
        names = [p.name[:-len("_embedding_mapping.parquet")] for p in embeddings.glob("*_embedding_mapping.parquet")]
        if len(names) > 1:
            raise ValueError("Multiple embedding datasets; select --dataset")
        name = names[0] if names else artifact_dataset(shared)
    if not isinstance(name, str) or not name or name in {".", ".."} or any(c in name for c in '/\\:*?"<>|'):
        raise ValueError("Dataset must be a filename stem")
    settings = dict(CONFIG_PATH=path, PROJECT_ROOT=root, SHARED_DATA_DIR=shared,
                    EMBEDDINGS_DIR=embeddings, INDEXES_DIR=indexes, ARCHIVE_DIR=archive,
                    CLEAN_DIR=shared / "02_clean", DATASET_NAME=name,
                    INDEX_VERSION=raw.get("index_version", "unec-dense-qdrant-v1"))
    from .ranking_policy import settings as ranking_settings
    settings["RANKING_POLICY"] = ranking_settings(raw.get("ranking"))
    defaults = {
        "qdrant": {"url": "http://127.0.0.1:6333", "collection": "unec_chunks", "vector_dimension": 1024, "batch_size": 128},
        "search": {"vector_backend": "qdrant", "dense_top_k": 20, "bm25_top_k": 20, "hybrid_top_k": 20, "rrf_k": 60, "final_top_k": 8},
        "reranker": {"model_id": "BAAI/bge-reranker-v2-m3", "use_fp16": True, "batch_size": 4, "input_top_k": 20, "query_max_length": 256, "passage_max_length": 512, "date_context": True},
    }
    aliases = {"url": "QDRANT_URL", "collection": "QDRANT_COLLECTION", "batch_size": "QDRANT_BATCH_SIZE"}
    for section, fields in defaults.items():
        for key, default in fields.items():
            constant = ("RERANKER_" + key.upper()) if section == "reranker" else aliases.get(key, key.upper())
            value = sections[section].get(key, default)
            if constant in {"QDRANT_URL", "QDRANT_COLLECTION", "VECTOR_BACKEND"}:
                value = os.getenv(constant, value)
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"{section}.{key} must be a YAML boolean")
            elif isinstance(default, int):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{section}.{key} must be a positive integer")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{section}.{key} must be a nonempty string")
            settings[constant] = value
    settings["VECTOR_BACKEND"] = settings["VECTOR_BACKEND"].strip().lower()
    if settings["VECTOR_BACKEND"] not in {"qdrant", "faiss"}:
        raise ValueError("vector_backend must be qdrant or faiss")
    for constant, folder, suffix in (
        ("EMBEDDINGS_PATH", embeddings, "_embeddings.npy"),
        ("MAPPING_PATH", embeddings, "_embedding_mapping.parquet"),
        ("EMBEDDING_METADATA_PATH", embeddings, "_embedding_metadata.json"),
        ("QDRANT_INDEX_METADATA_PATH", indexes, "_qdrant_index_metadata.json"),
        ("BM25_CORPUS_PATH", indexes, "_bm25_corpus.parquet"),
        ("BM25_METADATA_PATH", indexes, "_bm25_index_metadata.json"),
        ("DELETED_DOCUMENTS_PATH", shared / "02_clean", "_deleted_documents.parquet"),
        ("DELTA_SYNC_METADATA_PATH", indexes, "_delta_sync_metadata.json"),
        ("FAISS_INDEX_PATH", indexes, ".faiss"),
        ("FAISS_MAPPING_PATH", indexes, "_faiss_mapping.parquet"),
        ("FAISS_METADATA_PATH", indexes, "_faiss_index_metadata.json")):
        input_name = os.getenv("PIPELINE_BATCH_DATASET", name) if constant in {"EMBEDDINGS_PATH", "MAPPING_PATH", "EMBEDDING_METADATA_PATH"} else name
        if not input_name or any(c in input_name for c in '/\\:*?"<>|') or input_name in {".", ".."}:
            raise ValueError("Invalid embedding batch dataset name")
        settings[constant] = folder / f"{input_name}{suffix}"
    return settings


_settings = None


def __getattr__(name):
    global _settings
    if not name.isupper():
        raise AttributeError(name)
    if _settings is None:
        _settings = load_settings()
    if name not in _settings:
        raise AttributeError(name)
    return _settings[name]
