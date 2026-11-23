"""Standalone canonical stage: discover raw batch and own baseline versioning."""
from pathlib import Path
import yaml
from pipeline_storage import default_config, configuration_root, storage_paths, resolve, select_input
from pipeline_storage.stage import Stage, parser, clean_environment, stage_locks, upstream, read


def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument('--input', type=Path, help='Optional explicit raw upload')
    cli.add_argument('--deleted', type=Path)
    args = cli.parse_args(argv)
    if args.mode == 'reset_data' and args.deleted:
        cli.error('reset_data does not accept --deleted')
    clean_environment()
    path = Path(args.config).resolve() if args.config else default_config(__file__, 'config/config.yaml')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    root = configuration_root(path, __file__, '02_data_canonicalization')
    archive, shared = storage_paths(root, config, '02_clean')
    raw = resolve(root, config.get('input_dir', '../shared_data/01_raw'))
    from .pipeline import process, publish_canonical, quality_options
    with stage_locks(raw, shared):
        collected = (not args.input and (raw / 'run_report.json').is_file() and
                     read(raw / 'run_report.json').get('contract', {}).get('stage') == 'collection')
        source_report = upstream(raw, 'collection', args.run_id, args.mode) if args.mode == 'collect' or (args.mode is None and collected) else None
        stage = Stage('canonical', args, archive, shared, source_report)
        if stage.recover(lambda target: publish_canonical(target, shared)):
            return
        incoming = args.input.resolve() if args.input else select_input(raw, '*.parquet', exclude_deleted=True)
        if incoming.parent != raw or not incoming.is_file():
            raise ValueError('Raw input must be a file directly inside the configured raw folder')
        deletion = None if stage.mode == 'reset_data' else (args.deleted.resolve() if args.deleted else raw / 'chatbot_dataset_deleted.parquet')
        if args.deleted and not deletion.is_file():
            raise FileNotFoundError(deletion)
        stage.pin_inputs([incoming] + ([deletion] if deletion and deletion.is_file() else []))
        import polars as pl
        if not pl.read_parquet(incoming).height and (deletion is None or not deletion.exists() or not pl.read_parquet(deletion).height):
            if stage.mode == 'reset_data':
                raise ValueError('Reset data requires a nonempty replacement')
            stage.skip('No new or deleted records', status='no_changes')
            return
        report = process([incoming], archive, shared, False,
                         merge_base=stage.mode != 'reset_data', reset_data=stage.mode == 'reset_data',
                         deleted_path=deletion if deletion and deletion.is_file() else None,
                         **quality_options(config))
        report['index_dataset_name'] = report['base_output_file'][:-len('_canonical.parquet')]
        stage.finish(Path(report['run_dir']), report)


if __name__ == '__main__':
    main()
