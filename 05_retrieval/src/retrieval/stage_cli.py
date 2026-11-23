"""Standalone retrieval stage: resolve published batch and canonical baseline."""
import os
from pathlib import Path
from pipeline_storage.stage import Stage, parser, clean_environment, stage_locks, upstream, child_report


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    clean_environment()
    if args.config:
        os.environ['RETRIEVAL_CONFIG'] = str(Path(args.config).resolve())
    from . import config
    settings = config.load_settings(args.config)
    clean = settings['CLEAN_DIR']
    embeddings = settings['EMBEDDINGS_DIR']
    shared = settings['INDEXES_DIR']
    with stage_locks(clean, embeddings, shared):
        canonical = upstream(clean, 'canonical', args.run_id, args.mode)
        stage = Stage('retrieval', args, settings['ARCHIVE_DIR'], shared, canonical)
        from .indexing.storage import retry_publication
        if stage.recover(lambda target: retry_publication(target, operation='sync')):
            return
        batch = upstream(embeddings, 'embedding', stage.run_id, stage.mode)
        if canonical['contract']['status'] == 'no_changes':
            raise ValueError('No changes to index; stop after canonicalization')
        canonical_file = canonical['output_file']
        batch_name = canonical_file[:-len('_canonical.parquet')]
        expected = 'skipped' if canonical['output_rows'] == 0 else 'completed'
        if batch['contract']['status'] != expected:
            raise ValueError('Embedding status differs from the canonical batch')
        if expected == 'completed' and batch['dataset_name'] != batch_name:
            raise ValueError('Embedding dataset differs from the canonical batch')
        # Canonical owns the immutable full snapshot. Shared contains batch only.
        snapshot_dir = Path(canonical['run_dir']) / 'base_snapshot'
        snapshot = child_report(snapshot_dir, canonical['base_output_file'])
        from pipeline_storage import inventory
        base = snapshot.parent.parent.parent / 'base_data' / snapshot.name
        if not base.is_file() or inventory(snapshot_dir).get(snapshot.name) != inventory(base.parent).get(base.name):
            raise ValueError('Canonical base snapshot differs from the committed baseline')
        name = canonical['base_output_file'][:-len('_canonical.parquet')]
        os.environ.update(PIPELINE_DATASET=name, PIPELINE_BATCH_DATASET=batch_name,
                          PIPELINE_BATCH_REPORT=str(clean / 'run_report.json'),
                          PIPELINE_BASE_SNAPSHOT=str(snapshot))
        config._settings = None
        from .indexing.sync import main as synchronize
        synchronize([])
        target, report = stage.existing()
        stage.finish(target, report)


if __name__ == '__main__':
    main()
