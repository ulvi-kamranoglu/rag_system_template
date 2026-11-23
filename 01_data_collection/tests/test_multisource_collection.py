"""Real source persistence and unified delivery with mocked network/browser IO."""
from pathlib import Path
from collections import Counter
import json
import pandas as pd
import pytest
import yaml
from pipeline_storage import inventory
from pipeline_storage.stage import read, upstream
from wp_tracker import collection
from wp_tracker.stage_cli import main
from wp_tracker.records import dataset


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.delenv('PIPELINE_RUN_ID', raising=False)
    project = tmp_path / 'collection'
    path = project / 'config/settings.yaml'
    path.parent.mkdir(parents=True)
    config = dict(storage=dict(archive_dir=str(tmp_path / 'archive_data/01_raw'),
                               shared_dir=str(tmp_path / 'shared_data/01_raw')),
                  safety=dict(max_deleted_fraction=1), network=dict(delay_seconds=0),
                  instagram=dict(enabled=True, profiles=['uneceduaz', 'adalat.muradov']),
                  news=dict(enabled=True, delay_seconds=0))
    path.write_text(yaml.safe_dump(config))
    _, parent, sources = collection.sources_for(path)
    calls = Counter()
    from wp_tracker.records import from_api
    def item(number):
        return dict(id=number, title=dict(rendered='University information'),
                    content=dict(rendered=f'<p>Useful university information for document {number}.</p>'),
                    link=f'https://unec.edu.az/{number}', date_gmt='2026-10-01T00:00:00', modified_gmt='2026-10-01T00:00:00')
    parent.state_dir.mkdir(parents=True)
    baseline = [from_api(item(1), 'pages', parent), from_api(item(10), 'posts', parent)]
    dataset(baseline).to_parquet(parent.baseline_file)
    dataset(baseline).to_parquet(parent.snapshot_file)
    class Client:
        session = type('Session', (), {'close': lambda self: None})()
        def __init__(self, settings): pass
        def fetch_all(self, name):
            calls['wordpress/' + name] += 1
            return [item(1), item(2)] if name == 'pages' else [item(11)]
    monkeypatch.setattr('wp_tracker.api.WordPressClient', Client)
    from wp_tracker.instagram import service as ig
    from wp_tracker.instagram.schema import new_row, frame, format_numeric_id
    for source in sources:
        if source.kind != 'instagram': continue
        settings = source.settings
        row = new_row(settings.username, 'https://www.instagram.com/p/SEED/', 'Existing caption.', pd.Timestamp('2026-10-01', tz='UTC'))
        row['id'] = format_numeric_id(settings.username, 1)
        seed = project / (settings.username + '.parquet')
        frame([row]).to_parquet(seed, index=False)
        ig.bootstrap(settings, seed, run=True)
        settings.auth_file.parent.mkdir(parents=True, exist_ok=True)
        settings.auth_file.write_text('{}')  # Synthetic test session, never used by a browser.
    def instagram(settings, known, cutoff, on_new, **kwargs):
        calls['instagram/' + settings.username] += 1
        code = 'ADDEDUNEC' if settings.username == 'uneceduaz' else 'ADDEDRECTOR'
        url = f'https://www.instagram.com/p/{code}/'
        if url not in known:
            on_new(url, 'New university caption.', pd.Timestamp('2026-11-01', tz='UTC'))
        return {'boundary_reached': True}
    monkeypatch.setattr(ig, 'collect', instagram)
    import importlib.util
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name, *a: object() if name == 'playwright.sync_api' else original(name, *a))
    from wp_tracker.news import service as news, schema
    settings = next(source.settings for source in sources if source.kind == 'news')
    seed = settings.seed_file  # the shared import folder, as in production
    seed.parent.mkdir(parents=True, exist_ok=True)
    old = schema.article_row('https://news.unec.edu.az/xeber/1-category/1-old', 'Old article',
                             'Old university information.', '2026-10-01')
    dataset([{**old, 'id': 1}]).to_parquet(seed)  # legacy sequential ID; rebuilt from the URL
    news.bootstrap(settings, run=True)
    def discover(settings, snapshot, **kwargs):
        calls['news'] += 1
        new = schema.article_row('https://news.unec.edu.az/xeber/1-category/2-new', 'New article',
                                 'Previously unknown older news.', '2026-09-01')
        return {'current': list(snapshot) + ([] if any(r['id'] == new['id'] for r in snapshot) else [new]),
                'sitemap_articles': 2, 'confirmed_deleted': [], 'unconfirmed_missing': [], 'unparsed_new': []}
    monkeypatch.setattr(news, 'collect', discover)
    return path, parent, sources, calls


def run(path, run_id):
    main(['--config', str(path), '--run-id', run_id])


def test_all_sources_fixed_baselines_and_public_contract(environment):
    path, parent, sources, calls = environment
    original_legacy = parent.baseline_file.read_bytes()
    run(path, 'first')
    report = upstream(parent.shared_raw_dir, 'collection', 'first', 'collect')
    assert report['new'] == 5 and report['deleted'] == 1
    assert set(p.name for p in parent.shared_raw_dir.iterdir()) == {'chatbot_dataset_new.parquet', 'chatbot_dataset_deleted.parquet', 'run_report.json'}
    fresh = pd.read_parquet(parent.shared_raw_dir / 'chatbot_dataset_new.parquet')
    assert set(fresh.source_type) == {'page', 'post', 'instagram', 'news'}
    assert fresh.loc[fresh.source_type == 'news', 'id'].tolist() == ['news:2']
    assert fresh.loc[fresh.source_type == 'news', 'metadata'].tolist() == ['{"news_id":"2"}']
    assert fresh.loc[fresh.source_type == 'instagram', 'id'].tolist() == ['instagram:adalat.muradov:2', 'instagram:uneceduaz:2']
    assert pd.read_parquet(parent.shared_raw_dir / 'chatbot_dataset_deleted.parquet').id.tolist() == ['post:10']
    assert all(source.settings.baseline_file.is_file() and source.settings.snapshot_file.is_file() for source in sources)
    baseline_bytes = {source.key: source.settings.baseline_file.read_bytes() for source in sources}
    run(path, 'second')
    repeated = upstream(parent.shared_raw_dir, 'collection', 'second', 'collect')
    # WordPress/News repeat baseline-relative NEW; Instagram NEW is cutoff-relative.
    assert repeated['new'] == 3 and repeated['deleted'] == 1
    again = pd.read_parquet(parent.shared_raw_dir / 'chatbot_dataset_new.parquet')
    assert 'instagram' not in set(again.source_type)
    for source in sources:
        assert source.settings.baseline_file.read_bytes() == baseline_bytes[source.key]
        assert source.archive.is_relative_to(parent.output_dir / 'sources')
    assert parent.baseline_file.read_bytes() == original_legacy
    assert calls['news'] == 2
    assert len(list(parent.output_dir.glob('run_*'))) == 2


def test_partial_failure_resumes_sources_and_preserves_shared(environment, monkeypatch):
    path, parent, sources, calls = environment
    parent.shared_raw_dir.mkdir(parents=True)
    (parent.shared_raw_dir / 'existing.txt').write_text('Previous valid publication')
    previous = inventory(parent.shared_raw_dir)
    from wp_tracker.news import service
    discover = service.collect
    fail = [True]
    def interrupted(*args, **kwargs):
        if fail[0]:
            fail[0] = False
            raise RuntimeError('News unavailable')
        return discover(*args, **kwargs)
    monkeypatch.setattr(service, 'collect', interrupted)
    with pytest.raises(RuntimeError, match='News unavailable'):
        run(path, 'recover')
    assert inventory(parent.shared_raw_dir) == previous
    assert not list(parent.output_dir.glob('run_*'))
    assert calls['wordpress/pages'] == 1 and calls['instagram/uneceduaz'] == 1
    run(path, 'recover')
    assert calls['wordpress/pages'] == 1 and calls['wordpress/posts'] == 1
    assert calls['instagram/uneceduaz'] == 1 and calls['instagram/adalat.muradov'] == 1
    before = dict(calls)
    run(path, 'recover')
    assert dict(calls) == before
    assert upstream(parent.shared_raw_dir, 'collection')['new'] == 5


def test_unified_publication_failure_never_rescans(environment, monkeypatch):
    path, parent, _, calls = environment
    original = collection.publish
    def fail(*args, **kwargs):
        raise RuntimeError('Local delivery failed')
    monkeypatch.setattr(collection, 'publish', fail)
    with pytest.raises(RuntimeError, match='delivery failed'):
        run(path, 'publication')
    before = dict(calls)
    monkeypatch.setattr(collection, 'publish', original)
    run(path, 'publication')
    assert dict(calls) == before
    upstream(parent.shared_raw_dir, 'collection', 'publication')


def test_missing_source_baseline_aborts_before_any_scan(environment):
    path, parent, sources, calls = environment
    next(source.settings.baseline_file for source in sources if source.kind == 'instagram').unlink()
    with pytest.raises(RuntimeError, match='baseline is missing'):
        run(path, 'not-ready')
    assert not calls
    assert not parent.shared_raw_dir.exists()


def test_source_settings_survive_coordinator_config_snapshot(environment, tmp_path):
    path, parent, sources, _ = environment
    from orchestration.config import snapshot_config
    target = tmp_path / 'runtime/run/canonical.yaml'
    target.parent.mkdir(parents=True)
    snapshot_config(path, target, 'collection')
    _, frozen, selected = collection.sources_for(target)
    assert frozen.project_root == parent.project_root
    assert [(s.key, s.settings.state_dir, s.archive) for s in selected] == [(s.key, s.settings.state_dir, s.archive) for s in sources]


def test_public_cli_retries_unified_batch_without_network(environment, monkeypatch):
    path, parent, _, calls = environment
    run(path, 'cli')
    before = dict(calls)
    from wp_tracker.cli import main as cli
    monkeypatch.setattr('sys.argv', ['wp-change-tracker', 'sync-shared', '--config', str(path)])
    cli()
    assert dict(calls) == before
    upstream(parent.shared_raw_dir, 'collection', 'cli')


def test_retry_refuses_changed_baseline(environment, monkeypatch):
    path, parent, sources, calls = environment
    from wp_tracker.news import service
    collect_news = service.collect
    monkeypatch.setattr(service, 'collect', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('News unavailable')))
    with pytest.raises(RuntimeError):
        run(path, 'baseline-pin')
    settings = next(s.settings for s in sources if s.key == 'wordpress/pages')
    frame = pd.read_parquet(settings.baseline_file)
    frame['title'] = 'Changed baseline'
    frame.to_parquet(settings.baseline_file, index=False)
    monkeypatch.setattr(service, 'collect', collect_news)
    before = dict(calls)
    with pytest.raises(ValueError, match='inputs changed'):
        run(path, 'baseline-pin')
    assert dict(calls) == before


def test_combined_raw_is_accepted_by_existing_canonicalization(environment, monkeypatch, tmp_path):
    path, parent, _, _ = environment
    run(path, 'canonical-input')
    from canonical_pipeline import pipeline
    monkeypatch.setattr(pipeline, 'detector', lambda: None)
    monkeypatch.setattr(pipeline, 'detect_language', lambda *text: dict(language='en', confidence=0.99, uncertain=False))
    result = pipeline.process([parent.shared_raw_dir / 'chatbot_dataset_new.parquet'],
                              tmp_path / 'canonical-archive', tmp_path / 'canonical-shared', True)
    assert result['input_rows'] == 5
    assert result['output_rows'] == 5


def test_only_selected_sources_are_collected(environment):
    path, parent, sources, calls = environment
    main(['--config', str(path), '--run-id', 'only-news', '--sources', 'news,instagram/adalat.muradov'])
    report = upstream(parent.shared_raw_dir, 'collection', 'only-news', 'collect')
    assert report['selected_sources'] == ['instagram/adalat.muradov', 'news']
    fresh = pd.read_parquet(parent.shared_raw_dir / 'chatbot_dataset_new.parquet')
    assert sorted(fresh.id) == ['instagram:adalat.muradov:2', 'news:2']
    assert calls == Counter({'news': 1, 'instagram/adalat.muradov': 1})


def test_selection_is_pinned_for_resume(environment, monkeypatch):
    path, parent, sources, calls = environment
    from wp_tracker.news import service
    monkeypatch.setattr(service, 'collect', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('News unavailable')))
    with pytest.raises(RuntimeError):
        main(['--config', str(path), '--run-id', 'pinned', '--sources', 'news'])
    with pytest.raises(ValueError, match='selection changed'):
        main(['--config', str(path), '--run-id', 'pinned', '--sources', 'news,wordpress/pages'])


@pytest.mark.parametrize('names,match', [(['twitter'], 'Unknown collection source'),
                                         (['news', 'news'], 'selected once')])
def test_invalid_selection_is_rejected(environment, names, match):
    path = environment[0]
    with pytest.raises(ValueError, match=match):
        collection.select(path, names)


def test_disabled_source_cannot_be_selected(environment):
    path = environment[0]
    config = yaml.safe_load(path.read_text())
    config['news']['enabled'] = False
    path.write_text(yaml.safe_dump(config))
    assert collection.catalog(path)['news'] is False
    with pytest.raises(ValueError, match='news is disabled'):
        collection.select(path, ['news'])
    assert collection.select(path, ['wordpress/posts', 'wordpress/pages']) == ['wordpress/pages', 'wordpress/posts']
