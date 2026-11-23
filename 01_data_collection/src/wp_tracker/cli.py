"""CLI for dry-run scans, live scans, baseline import, and status."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_settings
from .pipeline import bootstrap, recover_state


def main() -> None:
    parser = argparse.ArgumentParser(prog="wp-change-tracker", description="UNEC multi-source baseline/snapshot collection")
    commands = parser.add_subparsers(dest="command", required=True)
    from .instagram.cli import register as register_instagram
    from .news.cli import register as register_news
    register_instagram(commands)
    register_news(commands)
    for name in ("scan", "bootstrap", "status", "recover", "sync-shared"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default=None)
        if name in {"scan", "bootstrap", "sync-shared"}:
            sub.add_argument("--migrate-legacy-shared", action="store_true",
                             help="Preserve legacy shared run folders in a sibling backup before publication")
        if name in ("scan", "bootstrap"):
            sub.add_argument("--run", action="store_true", help="Write Parquet files; otherwise dry-run")
        if name == "scan":
            sub.add_argument("--allow-large-deletions", action="store_true", help="Explicitly override deletion safety guard")
        elif name == "bootstrap":
            sub.add_argument("--reset", action="store_true", help="Explicitly replace existing baseline, snapshot, and event outputs from approved imports")
            sub.add_argument("--pages", type=Path, default=None)
            sub.add_argument("--posts", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    try:
        if args.command in {'instagram', 'news'}:
            if args.command == 'instagram':
                from .instagram.cli import run as source_run
            else:
                from .news.cli import run as source_run
            source_run(args)
            return
        if args.command == 'scan':
            if not args.run:
                from .collection import status as collection_status
                print(json.dumps({'dry_run': True, 'network_requested': False, **collection_status(args.config)}, indent=2))
                return
            from .stage_cli import main as collect
            command = ['--config', args.config] if args.config else []
            if args.allow_large_deletions:
                command.append('--allow-large-deletions')
            collect(command)
            return
        if args.command == 'status':
            from .collection import status as collection_status
            print(json.dumps(collection_status(args.config), indent=2))
            return
        if args.command == 'sync-shared':
            from .collection import configuration
            from pipeline_storage.stage import read
            _, _, parent = configuration(args.config)
            latest = read(parent.state_dir / 'collection/latest.json')
            from .stage_cli import main as collect
            command = ['--run-id', latest['run_id']]
            if args.config:
                command += ['--config', args.config]
            collect(command)
            return
        settings = load_settings(args.config)
        if args.command == "bootstrap":
            if args.reset and any((settings.state_dir / 'wordpress' / name / 'baseline.parquet').exists()
                                  for name in ('pages', 'posts')):
                raise ValueError('Independent WordPress baselines already exist; legacy --reset would not reset them')
            result = bootstrap(settings, pages=args.pages, posts=args.posts, run=args.run, reset=args.reset,
                               migrate_legacy=args.migrate_legacy_shared)
        elif args.command == "recover":
            result = recover_state(settings)
        else:
            raise ValueError('Unsupported collection command')
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.command in {"scan", "bootstrap"} and not args.run:
        print("DRY RUN: no Parquet data or tracking report was written.")


if __name__ == "__main__":
    main()
