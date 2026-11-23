"""Standalone News CLI. Dry runs never request the website or write state."""
import json
import os
from pathlib import Path

import pandas as pd
from pipeline_storage import default_config

from .config import load_settings
from . import service
from .schema import seed_row, validate_rows


def register(commands):
    parser = commands.add_parser("news", help="UNEC News collector (sitemap, fixed baseline)")
    parser.add_argument("news_action", choices=("status", "validate", "bootstrap", "scan", "sync-shared"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=Path, help="Seed Parquet; default data/import/news.parquet")
    parser.add_argument("--run", action="store_true", help="Explicitly execute collection/state/publication")
    parser.add_argument("--allow-large-deletions", action="store_true")


def run(args):
    path = args.config or default_config(__file__, "config/settings.yaml")
    settings = load_settings(path)
    if args.news_action == "status":
        result = service.status(settings)
    elif args.news_action == "validate":
        seed = args.seed or settings.seed_file
        rows = [seed_row(row) for row in pd.read_parquet(seed).to_dict("records")]
        validate_rows(service._clean(rows))
        result = {"valid": True, "seed": str(seed), "rows": len(rows)}
    elif args.news_action == "bootstrap":
        result = service.bootstrap(settings, args.seed, run=args.run)
    elif args.news_action == "scan":
        result = service.scan(settings, run=args.run, pipeline_run_id=os.getenv('PIPELINE_RUN_ID'),
                              allow_large_deletions=args.allow_large_deletions)
    else:
        result = service.sync_shared(settings, run=args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
