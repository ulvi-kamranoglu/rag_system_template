"""Parse settings before importing retrieval stages with constant imports."""
import argparse
import importlib
import json
import os
from pathlib import Path


def _common(parser):
    parser.add_argument("--config", help="Retrieval YAML file")
    parser.add_argument("--project-root", help="Root containing shared_data and archive_data")
    parser.add_argument("--dataset", help="Dataset filename stem")


def _configure(args):
    for key, value in (("RETRIEVAL_CONFIG", args.config),
                       ("CHATBOT_PROJECT_ROOT", args.project_root),
                       ("PIPELINE_DATASET", args.dataset)):
        if value:
            os.environ[key] = str(Path(value).expanduser().resolve()) if key != "PIPELINE_DATASET" else value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Index existing embeddings; defaults to Qdrant")
    _common(parser)
    parser.add_argument("--stage", choices=["qdrant", "bm25", "faiss", "delta", "sync"], default="qdrant")
    parser.add_argument("--publish-run", help="Retry archived local publication")
    parser.add_argument("--migrate-legacy-shared", action="store_true")
    args = parser.parse_args(argv)
    _configure(args)
    modules = {"qdrant": "indexing.qdrant", "bm25": "indexing.bm25", "faiss": "indexing.faiss", "delta": "indexing.delta", "sync": "indexing.sync"}
    forwarded = []
    if args.publish_run:
        forwarded += ["--publish-run", args.publish_run]
    if args.migrate_legacy_shared:
        forwarded.append("--migrate-legacy-shared")
    importlib.import_module(f"retrieval.{modules[args.stage]}").main(forwarded)


def search_main(argv=None):
    parser = argparse.ArgumentParser(description="Hybrid search with reranking")
    _common(parser)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--diagnose", action="store_true", help="Trace dense, BM25, fusion and reranker results")
    parser.add_argument("--output", type=Path, help="Save full diagnostics as JSON (requires --diagnose)")
    parser.add_argument("--focus-source", help="Report ranks for a source_type, e.g. research (requires --diagnose)")
    parser.add_argument("--dense-limit", type=int)
    parser.add_argument("--bm25-limit", type=int)
    parser.add_argument("--rerank-limit", type=int)
    parser.add_argument("--legacy-ranking", action="store_true", help="Disable ranking policy for a baseline comparison")
    args = parser.parse_args(argv)
    if not args.query.strip():
        parser.error("--query must not be empty")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for flag in ("dense_limit", "bm25_limit", "rerank_limit"):
        if getattr(args, flag) is not None and getattr(args, flag) < 1:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    if (args.output or args.focus_source) and not args.diagnose:
        parser.error("--output and --focus-source require --diagnose")
    if args.output and args.output.suffix.lower() != ".json":
        parser.error("--output must be a .json file")
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new report filename")
    _configure(args)
    from retrieval.search.retriever import HybridRetriever
    retriever = HybridRetriever()
    try:
        if args.legacy_ranking:
            retriever.ranking_policy["enabled"] = False
        from retrieval import config
        options = dict(dense_limit=args.dense_limit or config.DENSE_TOP_K,
                       bm25_limit=args.bm25_limit or config.BM25_TOP_K,
                       reranker_input_limit=args.rerank_limit or config.RERANKER_INPUT_TOP_K,
                       final_limit=args.limit or config.FINAL_TOP_K)
        if args.diagnose:
            from retrieval.search.diagnostics import diagnose, compact
            report = diagnose(retriever, args.query, focus_source=args.focus_source, **options)
            report.update(dataset_name=config.DATASET_NAME, vector_backend=config.VECTOR_BACKEND,
                          qdrant_collection=config.QDRANT_COLLECTION if config.VECTOR_BACKEND == "qdrant" else None)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as output:
                    json.dump(report, output, ensure_ascii=False, indent=2, default=str)
            results = compact(report)
        else:
            results = retriever.search(args.query, **options)
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        if args.output:
            print(f"Full diagnostics: {args.output.resolve()}")
    finally:
        retriever.close()
