"""Terminal entrypoint; only actual runs load the embedding model."""
import argparse
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Generate dataset embeddings using YAML settings")
    parser.add_argument("--config", help="YAML configuration path")
    parser.add_argument("--input", help="Chunk Parquet path relative to the terminal")
    parser.add_argument("--dataset", help="Source dataset name without stage suffixes")
    parser.add_argument("--dry-run", action="store_true", help="Validate input without loading a model or writing outputs")
    parser.add_argument("--publish-run", help="Publish an existing complete archive without model inference")
    parser.add_argument("--migrate-legacy-shared", action="store_true", help="Retain old shared run folders in a sibling backup")
    args = parser.parse_args()
    if args.publish_run and (args.dry_run or args.input):
        parser.error("--publish-run cannot be combined with --dry-run or --input")
    run(args.config, input_override=args.input, dataset=args.dataset, dry_run=args.dry_run,
        publish_run=args.publish_run, migrate_legacy=args.migrate_legacy_shared)


if __name__ == "__main__":
    main()
