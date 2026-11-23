"""Standalone embedding stage: select chunks using its own model configuration."""
from pathlib import Path
import yaml
from pipeline_storage import default_config, configuration_root, resolve, storage_paths
from pipeline_storage.stage import Stage, parser, clean_environment, stage_locks, upstream, child_report


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    clean_environment()
    path = Path(args.config).resolve() if args.config else default_config(__file__, 'config/config.yaml')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    root = configuration_root(path, __file__, '04_embedding')
    archive, shared = storage_paths(root, config, '04_embeddings')
    source = resolve(root, config.get('input_dir', '../shared_data/03_chunks'))
    with stage_locks(source, shared):
        report = upstream(source, 'chunk', args.run_id, args.mode)
        stage = Stage('embedding', args, archive, shared, report)
        def publish_existing(target):
            from .pipeline import run
            run(str(path), publish_run=str(target))
        if stage.recover(publish_existing):
            return
        if report['contract']['status'] != 'completed':
            stage.skip('Empty chunk batch')
            return
        size = int(config.get('model', {}).get('document_max_length', 384))
        if size not in report['variants']:
            raise ValueError('Chunk stage did not publish the configured embedding token size')
        name = report['dataset_name']
        chunks = child_report(source, f'{name}_chunks_{size}.parquet')
        from .pipeline import run
        run(str(path), input_override=str(chunks), dataset=name)
        target, output = stage.existing()
        stage.finish(target, output)


if __name__ == '__main__':
    main()
