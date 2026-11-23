"""CLI extension for python -m wp_tracker instagram ..."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .config import load_ig_settings
from .service import bootstrap, scan, status, sync_shared
from .browser import prepare_auth


def default_seed(config: Path, profile: str) -> Path:
    """All seeds live in the collection import folder: instagram_<profile>.parquet ('.' -> '_')."""
    from ..config import load_settings
    return load_settings(config).import_dir / f"instagram_{profile.replace('.', '_')}.parquet"


def run(args):
    config = Path(args.config) if args.config else Path(__file__).resolve().parents[3] / 'config/settings.yaml'
    if not config.is_file():
        # Installed wheel has the package's copy of the config.
        config = Path(__file__).resolve().parents[1] / 'config/settings.yaml'
    settings = load_ig_settings(config, args.profile)
    if args.instagram_action == 'login':
        prepare_auth(settings)
        result = {'auth_state':str(settings.auth_file), 'saved':True}
    elif args.instagram_action == 'bootstrap':
        seed = args.seed or default_seed(config, settings.username)
        if not seed.is_file():
            raise FileNotFoundError(f'Instagram seed not found: {seed}')
        result = bootstrap(settings, seed, run=args.run)
    elif args.instagram_action == 'scan':
        result = scan(settings, run=args.run, headed=args.headed, max_scrolls=args.max_scrolls)
    elif args.instagram_action == 'status':
        result = status(settings)
    elif args.instagram_action == 'sync-shared':
        result = sync_shared(settings)
    else:
        raise ValueError('Unsupported Instagram command')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def register(commands):
    parent = commands.add_parser('instagram', help='Instagram incremental collector (separate from WordPress)')
    parent.add_argument('instagram_action', choices=('bootstrap','login','scan','status','sync-shared'))
    parent.add_argument('--profile', default='uneceduaz')
    parent.add_argument('--config', default=None)
    parent.add_argument('--run', action='store_true', help='Write/publish instead of dry-run')
    parent.add_argument('--seed', type=Path, default=None, help='Seed Parquet; default data/import/instagram_<profile>.parquet')
    parent.add_argument('--headed', action='store_true', help='Visible browser for local debugging only')
    parent.add_argument('--max-scrolls', type=int, default=None)
    return parent
