"""Command-line entrypoint."""
import argparse
import json
import logging
from pathlib import Path
import yaml
from .pipeline import process, publish_canonical, quality_options
from pipeline_storage import default_config, resolve, storage_paths, select_input
from pipeline_storage import configuration_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize one or more Parquet datasets")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", nargs="+", help="Input Parquet paths relative to the current directory")
    parser.add_argument("--publish-run", type=Path, help="Retry a complete archive without processing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deleted", type=Path, help="Explicit deleted document Parquet; deletions win")
    parser.add_argument("--full-report", action="store_true", help="Print all duplicate removal records")
    parser.add_argument("--min-content-chars", type=int, help="Minimum non-whitespace content length")
    parser.add_argument("--migrate-legacy-shared", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config_path = Path(args.config).resolve() if args.config else default_config(__file__, "config/config.yaml")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = configuration_root(config_path, __file__, "02_data_canonicalization")
    archive, shared = storage_paths(root, config, "02_clean")
    if args.publish_run:
        run = args.publish_run.resolve()
        if args.dry_run:
            parser.error("--publish-run cannot be combined with --dry-run")
        if run.parent != archive or not run.name.startswith("run_"):
            parser.error("--publish-run must name a run in the configured archive")
        result = publish_canonical(run, shared, migrate_legacy=args.migrate_legacy_shared)
    else:
        inputs = ([Path(value).resolve() for value in args.input] if args.input else
                  [resolve(root, value) for value in config["inputs"]] if config.get("inputs") else
                  [select_input(resolve(root, config.get("input_dir", "../shared_data/01_raw")),
                                "*.parquet", exclude_deleted=True)])
        result = process(inputs, archive, shared, args.dry_run,
                         migrate_legacy=args.migrate_legacy_shared,
                         merge_base=config.get("merge_base", False),
                         reset_data=config.get("reset_data", False),
                         deleted_path=args.deleted,
                         **{**quality_options(config),
                            **({"min_content_chars": args.min_content_chars}
                               if args.min_content_chars is not None else {})})
    if not args.full_report and "duplicates" in result:
        result = dict(result)
        duplicates = dict(result["duplicates"])
        records = duplicates.pop("removed_records", [])
        duplicates["removed_records_count"] = len(records)
        duplicates["removed_records_sample"] = records[:5]
        result["duplicates"] = duplicates
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
