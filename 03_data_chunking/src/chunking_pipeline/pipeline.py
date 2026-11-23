"""Prepare and chunk complete batches before archiving and publishing."""
from pathlib import Path
import json
import os
import polars as pl
import yaml
from transformers import AutoTokenizer
from pipeline_storage import (archive_run, default_config, publish, resolve,
                              run_path, storage_paths, validate_json, write_json,
                              batch_stem, select_input, configuration_root)
from .preparation import prepare
from .chunking import build_chunks


def validate_run(folder: Path) -> None:
    validate_json(folder, ["run_report.json"])
    report = json.loads((folder / "run_report.json").read_text(encoding="utf-8"))
    # Historical archives retain their original dataset subdirectories.
    flat = report.get("layout") == "flat"
    if report.get("dataset_name") and (Path(report["dataset_name"]).name != report["dataset_name"]
                                        or report["dataset_name"] in {".", ".."}):
        raise ValueError("Invalid dataset name in chunk report")
    prefix = report.get("dataset_name", "") + "_" if flat and report.get("dataset_name") else ""
    names = [None] if flat else [dataset["name"] for dataset in report["datasets"]]
    for name in names:
        data_dir = folder if flat else folder / name
        reports_dir = folder / "reports" if flat else folder / "reports" / name
        prepared = pl.read_parquet(data_dir / f"{prefix}retrieval_prepared.parquet")
        if "retrieval_eligible" not in prepared.columns:
            raise ValueError("Invalid prepared output")
        for size in report["variants"]:
            chunks = pl.read_parquet(data_dir / f"{prefix}chunks_{size}.parquet")
            if not chunks.height or chunks["chunk_id"].n_unique() != chunks.height:
                raise ValueError("Invalid chunk identifiers")
            if chunks["embedding_tokens"].max() > size:
                raise ValueError("Chunk exceeds token limit")
            validate_json(reports_dir, [f"{prefix}chunking_{size}_summary.json"])


def run(config_path: str | None = None, input_override: str | list[str] | None = None,
        dry_run: bool = False, publish_run: str | None = None,
        *, migrate_legacy: bool = False) -> list[Path]:
    return _run(config_path, input_override, dry_run, publish_run, migrate_legacy=migrate_legacy)


def _run(config_path=None, input_override=None, dry_run=False, publish_run=None,
         *, migrate_legacy=False):
    config_file = Path(config_path).resolve() if config_path else default_config(__file__, "config/config.yaml")
    root = configuration_root(config_file, __file__, "03_data_chunking")
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    archive, shared = storage_paths(root, config, "03_chunks")
    if publish_run:
        if dry_run:
            raise ValueError("--publish-run cannot be combined with --dry-run")
        target = Path(publish_run).resolve()
        if target.parent != archive or not target.name.startswith("run_"):
            raise ValueError("--publish-run must name a run in the configured archive")
        validate_run(target)
        print(json.dumps(publish(target, shared, validate_run, migrate_legacy=migrate_legacy), indent=2))
        return sorted(shared.rglob("*chunks_*.parquet"))
    values = input_override or config.get("inputs") or config.get("input_path")
    if isinstance(values, str):
        values = [values]
    inputs = [resolve(root, value) for value in values] if values else [
                  select_input(resolve(root, config.get("input_dir", "../shared_data/02_clean")),
                               "*_canonical.parquet")]
    name = batch_stem(inputs)
    if len({path.stem for path in inputs}) != len(inputs):
        raise ValueError("Input dataset stems must be unique")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Input dataset not found: {path}")
        if path.is_relative_to(shared) or path.is_relative_to(archive):
            raise ValueError("Output must not overwrite an input directory")
    variants = config["variants"]
    sizes = [int(v["max_tokens"]) for v in variants]
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("Chunk variant sizes must be nonempty and unique")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    batches, datasets = [], []
    for path in inputs:
        original = pl.read_parquet(path)
        prepared = prepare(original)
        outputs = []
        for variant in variants:
            size, overlap = int(variant["max_tokens"]), int(variant["overlap_tokens"])
            to_chunk = prepared
            if to_chunk.filter(pl.col("retrieval_eligible")).height:
                chunks = build_chunks(to_chunk, tokenizer, size, overlap,
                                  int(config.get("safety_margin", 4)),
                                  int(config.get("min_content_budget", 32)))
            else:
                raise ValueError("No eligible documents")
            summary = {
                "max_tokens": size, "overlap_tokens": overlap,
                "input_documents": original.height,
                "eligible_documents": prepared.filter(pl.col("retrieval_eligible")).height,
                "chunk_count": chunks.height,
                "unique_chunk_ids": chunks["chunk_id"].n_unique(),
                "trimmed_tokens": int(chunks["trimmed_tokens"].sum()),
                "max_embedding_tokens": int(chunks["embedding_tokens"].max()),
            }
            outputs.append((size, chunks, summary))
        batches.append((path.stem, prepared, outputs))
        datasets.append({"name": path.stem, "input": str(path)})
    # A flat output has one file per variant, including when inputs are multiple.
    prepared = pl.concat([batch[1] for batch in batches], how="diagonal_relaxed")
    combined = []
    for index, size in enumerate(sizes):
        chunks = pl.concat([batch[2][index][1] for batch in batches], how="diagonal_relaxed")
        if chunks["chunk_id"].n_unique() != chunks.height:
            raise ValueError("Duplicate chunk IDs across input datasets; source identities must be unique")
        summaries = [batch[2][index][2] for batch in batches]
        summary = {
            "max_tokens": size,
            "overlap_tokens": summaries[0]["overlap_tokens"],
            "input_documents": sum(item["input_documents"] for item in summaries),
            "eligible_documents": sum(item["eligible_documents"] for item in summaries),
            "chunk_count": chunks.height,
            "unique_chunk_ids": chunks["chunk_id"].n_unique(),
            "trimmed_tokens": int(chunks["trimmed_tokens"].sum()),
            "max_embedding_tokens": int(chunks["embedding_tokens"].max()),
        }
        combined.append((size, chunks, summary))
    target = run_path(archive)
    report = {"run_dir": str(target), "archive_dir": str(archive),
              "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
              "shared_dir": str(shared), "datasets": datasets,
              "variants": sizes, "dry_run": dry_run, "layout": "flat", "dataset_name": name,
              "outputs": [str(shared / f"{name}_retrieval_prepared.parquet")] +
                         [str(shared / f"{name}_chunks_{size}.parquet") for size in sizes]}
    if not dry_run:
        with archive_run(target, validate_run) as staging:
            prepared.write_parquet(staging / f"{name}_retrieval_prepared.parquet")
            for size, chunks, summary in combined:
                chunks.write_parquet(staging / f"{name}_chunks_{size}.parquet")
                write_json(staging / "reports" / f"{name}_chunking_{size}_summary.json", summary)
            write_json(staging / "run_report.json", report)
        try:
            report["publication"] = publish(target, shared, validate_run, migrate_legacy=migrate_legacy)
        except Exception as exc:
            raise RuntimeError(f"Archive complete at {target}; publication failed. "
                               f"Retry with --publish-run {target}: {exc}") from exc
    print(json.dumps(report, indent=2))
    return [shared / f"{name}_chunks_{size}.parquet" for size in sizes]
