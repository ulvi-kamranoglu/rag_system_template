import argparse
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description="the original project standalone chunking pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--migrate-legacy-shared", action="store_true")
    parser.add_argument("--publish-run", help="Retry publication of a complete archive")
    parser.add_argument("--input", nargs="+", help="Override input Parquet path")
    args = parser.parse_args()
    run(args.config, args.input, args.dry_run, args.publish_run,
        migrate_legacy=args.migrate_legacy_shared)


if __name__ == "__main__":
    main()
