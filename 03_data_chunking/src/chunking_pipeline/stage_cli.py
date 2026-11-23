"""Standalone chunk stage: consume the published canonical batch."""
from pathlib import Path
import yaml
from pipeline_storage import default_config, configuration_root, storage_paths, resolve
from pipeline_storage.stage import Stage, parser, clean_environment, stage_locks, upstream, child_report


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    clean_environment()
    path = Path(args.config).resolve() if args.config else default_config(__file__, 'config/config.yaml')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    root = configuration_root(path, __file__, '03_data_chunking')
    archive, shared = storage_paths(root, config, '03_chunks')
    source = resolve(root, config.get('input_dir', '../shared_data/02_clean'))
    with stage_locks(source, shared):
        report = upstream(source, 'canonical', args.run_id, args.mode)
        stage = Stage('chunk', args, archive, shared, report)
        # Import the model-dependent implementation only when it is needed.
        def publish_existing(target):
            from .pipeline import run
            run(str(path), publish_run=str(target))
        if stage.recover(publish_existing):
            return
        if report['contract']['status'] != 'completed' or report['output_rows'] == 0:
            stage.skip('Empty canonical batch')
            return
        from .pipeline import run
        run(str(path), [str(child_report(source, report['output_file']))])
        target, output = stage.existing()
        stage.finish(target, output)


if __name__ == '__main__':
    main()
