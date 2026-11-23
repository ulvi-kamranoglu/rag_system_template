"""Standalone collection stage: scrape, archive and publish raw outputs."""
from pipeline_storage.stage import Stage, parser, clean_environment, stage_locks


def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument('--allow-large-deletions', action='store_true')
    cli.add_argument('--sources', help='Comma-separated subset of enabled sources, e.g. news,instagram/uneceduaz; '
                                       'default: every enabled source')
    args = cli.parse_args(argv)
    if args.mode not in (None, 'collect'):
        raise ValueError('Collection supports collect mode only')
    clean_environment()
    from .collection import collect, configuration
    path, _, settings = configuration(args.config)
    with stage_locks(settings.shared_raw_dir):
        stage = Stage('collection', args, settings.output_dir, settings.shared_raw_dir)
        print(f'[collection] batch: {stage.run_id}', flush=True)
        try:
            only = args.sources.split(',') if args.sources else None
            collect(path, stage, allow_large_deletions=args.allow_large_deletions, only=only)
        except BaseException:
            print(f'Resume: python -m wp_tracker.stage_cli --config "{path}" --run-id {stage.run_id}', flush=True)
            raise


if __name__ == '__main__':
    main()
