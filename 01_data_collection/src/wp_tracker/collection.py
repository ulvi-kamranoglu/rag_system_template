"""Source isolation and unified RAW delivery owned entirely by collection."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import os
import pandas as pd
import yaml
from filelock import FileLock
from pipeline_storage import archive_run, inventory, publish, run_path
from pipeline_storage.stage import read, write_json
from .config import load_settings
from .records import FIELDS, dataset
from .storage import read_rows, commit


@dataclass(frozen=True)
class Source:
    key: str
    kind: str
    settings: object

    @property
    def archive(self):
        return self.settings.output_dir if self.kind == 'wordpress' else self.settings.archive_dir


def comparison(source):
    """WordPress/News: NEW relative to a fixed baseline. Instagram: newer than the newest collected post."""
    return 'cutoff_date' if source.kind == 'instagram' else 'fixed_baseline'


def configuration(path):
    from pipeline_storage import default_config
    path = Path(path).resolve() if path else default_config(__file__, 'config/settings.yaml')
    return path, yaml.safe_load(path.read_text(encoding='utf-8')), load_settings(path)


def enabled(value, default):
    if not isinstance(value, dict):
        raise ValueError('Source configuration must be a mapping')
    result = value.get('enabled', default)
    if not isinstance(result, bool):
        raise ValueError('Source enabled must be a boolean')
    return result


def public_name(key):
    """Selectable name of a source: wordpress/pages, wordpress/posts, instagram/<profile>, news."""
    return 'news' if key.startswith('news/') else key


def catalog(path=None):
    """Every source the configuration knows, with whether it is enabled (ready to collect)."""
    path, config, _ = configuration(path)
    result = {}
    wordpress = config.get('wordpress', {})
    names = wordpress.get('sources', ['pages', 'posts']) if isinstance(wordpress, dict) else []
    for name in ('pages', 'posts'):
        result['wordpress/' + name] = enabled(wordpress, True) and name in names
    from .instagram.config import listed_profiles
    instagram = enabled(config.get('instagram', {}), False)
    for profile in listed_profiles(path):
        result['instagram/' + profile] = instagram
    result['news'] = enabled(config.get('news', {}), False)
    return result


def select(path, names):
    """Validate a requested subset; only enabled sources may be selected."""
    if names is None:
        return None
    names = [str(name).strip() for name in names if str(name).strip()]
    if not names:
        return None
    if len(set(names)) != len(names):
        raise ValueError('Each collection source may be selected once')
    available = catalog(path)
    for name in names:
        if name not in available:
            raise ValueError(f'Unknown collection source {name!r}; available: {", ".join(available)}')
        if not available[name]:
            raise ValueError(f'{name} is disabled in collection config; enable it after its baseline/session is ready')
    return [name for name in available if name in names]  # stable configuration order


def sources_for(path, only=None):
    selected = select(path, only)
    path, config, parent = configuration(path)
    sources = []
    wordpress = config.get('wordpress', {})
    if enabled(wordpress, True):
        names = wordpress.get('sources', ['pages', 'posts'])
        if not isinstance(names, list) or not names or len(set(names)) != len(names) or set(names) - {'pages', 'posts'}:
            raise ValueError('wordpress.sources must select unique pages/posts names')
        for name in names:
            state = parent.state_dir / 'wordpress' / name
            settings = replace(parent, output_dir=parent.output_dir / 'sources/wordpress' / name,
                               state_dir=state, reports_dir=state,
                               shared_raw_dir=parent.project_root / 'data/output/wordpress' / name)
            sources.append(Source('wordpress/' + name, 'wordpress', settings))
    if enabled(config.get('instagram', {}), False):
        from .instagram.config import listed_profiles, load_ig_settings
        profiles = listed_profiles(path)
        if len(set(profiles)) != len(profiles):
            raise ValueError('Duplicate Instagram profile')
        for profile in profiles:
            sources.append(Source('instagram/' + profile, 'instagram', load_ig_settings(path, profile)))
    if enabled(config.get('news', {}), False):
        from .news.config import load_settings as news_settings
        sources.append(Source('news/news.unec.edu.az', 'news', news_settings(path)))
    if not sources:
        raise ValueError('Enable at least one collection source')
    if selected is not None:
        sources = [source for source in sources if public_name(source.key) in selected]
    return path, parent, sources


def split_wordpress(parent, sources, *, run=False):
    """Copy legacy combined state into independent namespaces, never modify it."""
    planned = []
    for source in sources:
        if source.kind != 'wordpress':
            continue
        settings = source.settings
        if settings.baseline_file.is_file():
            if not settings.snapshot_file.is_file():
                raise ValueError(f'{source.key}: snapshot missing; restore its source state')
            continue
        if not parent.baseline_file.is_file() or not parent.snapshot_file.is_file():
            raise ValueError('WordPress baseline/snapshot missing; bootstrap legacy pages/posts first')
        kind = 'page' if source.key.endswith('/pages') else 'post'
        legacy_base, legacy_snapshot = read_rows(parent.baseline_file), read_rows(parent.snapshot_file)
        if any(r['source_type'] not in {'page', 'post'} for r in [*legacy_base, *legacy_snapshot]):
            raise ValueError('Legacy WordPress state contains an unexpected source type')
        base = [r for r in legacy_base if r['source_type'] == kind]
        snapshot = [r for r in legacy_snapshot if r['source_type'] == kind]
        planned.append({'source': source.key, 'baseline_rows': len(base), 'snapshot_rows': len(snapshot)})
        if run:
            settings.state_dir.mkdir(parents=True, exist_ok=True)
            with FileLock(str(settings.lock_file), timeout=0):
                if settings.baseline_file.exists():
                    raise ValueError('WordPress source was initialized concurrently; retry')
                commit(settings, {settings.baseline_file: ('snapshot', base),
                                  settings.snapshot_file: ('snapshot', snapshot)})
    return planned


def _preflight(parent, sources):
    split_wordpress(parent, sources)  # Validate legacy migration before HTTP.
    for source in sources:
        if source.kind == 'instagram':
            from .instagram.service import _load_state
            _load_state(source.settings)
            if not source.settings.auth_file.is_file():
                raise ValueError(f'{source.key}: login session missing; run instagram login')
            try:
                import importlib.util
                available = importlib.util.find_spec('playwright.sync_api') is not None
            except ModuleNotFoundError:
                available = False
            if not available:
                raise ValueError('Install collection[instagram] and Playwright Chromium before enabling Instagram')
        elif source.kind == 'news':
            if not source.settings.baseline_file.is_file() or not source.settings.snapshot_file.is_file():
                raise ValueError('News baseline/snapshot missing; run news bootstrap --run with data/import/news.parquet')


def _committed(source, run_id):
    """No rescanning after commit, even if the unified batch was not built yet."""
    matches = []
    for path in source.archive.glob('run_*/run_report.json'):
        report = read(path)
        if report.get('pipeline_run_id') == run_id:
            matches.append((path.parent, report))
    if len(matches) > 1:
        raise ValueError(f'{source.key}: multiple source archives for this batch')
    if not matches:
        return None
    target, report = matches[0]
    latest = source.settings.reports_dir / 'latest.json' if source.kind == 'wordpress' else source.settings.report_file
    if not latest.is_file() or Path(read(latest)['run_dir']) != target:
        raise ValueError(f'{source.key}: archive is uncommitted or no longer active; restore source state')
    if source.kind == 'wordpress':
        from .mirror import mirror_run
        mirror_run(source.settings, target)
    elif source.kind == 'news':
        from .news.service import mirror
        mirror(source.settings, target)
    else:
        from .instagram.service import _mirror
        _mirror(source.settings, report)
    return target, report


def scan_source(source, *, allow_large_deletions=False):
    if source.kind == 'wordpress':
        from .pipeline import scan
        from .api import WordPressClient
        selected = source.key.rsplit('/', 1)[1]
        client = WordPressClient(source.settings)
        class OneSource:
            def fetch_all(self, name):
                return client.fetch_all(name) if name == selected else []
        try:
            return scan(source.settings, run=True, client=OneSource(), allow_large_deletions=allow_large_deletions)
        finally:
            client.session.close()
    if source.kind == 'instagram':
        from .instagram.service import scan
        return scan(source.settings, run=True)
    from .news.service import scan
    return scan(source.settings, run=True, pipeline_run_id=os.environ['PIPELINE_RUN_ID'],
                allow_large_deletions=allow_large_deletions)


def _frame(path):
    frame = pd.read_parquet(path)
    if list(frame.columns) != FIELDS:
        raise ValueError(f'Unexpected RAW columns: {path}')
    rows = frame.to_dict('records')
    for row in rows:
        identity = row['id']
        if identity is None or pd.isna(identity) or isinstance(identity, bool):
            raise ValueError('Invalid RAW identity')
        row['id'] = str(identity)
        if not row['id'].strip() or row['source_type'] not in {'page', 'post', 'instagram', 'news'}:
            raise ValueError('Invalid RAW source identity')
        for key in ('date', 'modified'):
            value = row[key]
            if pd.isna(value):
                row[key] = None
            else:
                if isinstance(value, (int, float)):
                    raise ValueError('RAW dates must be timestamps or ISO dates')
                timestamp = pd.to_datetime(value, utc=True, errors='raise')
                row[key] = timestamp.tz_localize(None)
    return dataset(rows)


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_raw(folder):
    report = read(folder / 'run_report.json')
    for event in ('new', 'deleted'):
        path = folder / f'chatbot_dataset_{event}.parquet'
        frame = _frame(path)
        if frame.duplicated(['source_type', 'id']).any():
            raise ValueError('Duplicate RAW identities')
        if len(frame) != report[event] or _sha(path) != report['hashes'][event]:
            raise ValueError('RAW report count/hash mismatch')
        if event == 'deleted' and not frame.empty and not frame.source_type.isin(['page', 'post', 'news']).all():
            raise ValueError('Partial Instagram scans cannot infer deletions')


def _pin_selection(parent, stage, sources):
    """A resumed batch must collect exactly the sources of its first attempt."""
    selection = {'sources': [public_name(source.key) for source in sources]}
    pin = parent.output_dir / '.requests' / f'{stage.run_id}.selection.json'
    if pin.exists() and read(pin) != selection:
        raise ValueError('Source selection changed since the previous attempt; resume with the original sources')
    write_json(pin, selection)


def collect(config_path, stage, *, allow_large_deletions=False, only=None):
    """One resumable collection command, no coordinator-specific callbacks."""
    path, parent, sources = sources_for(config_path, only)
    if stage.recover(lambda target: publish(target, parent.shared_raw_dir, validate_raw)):
        target, _ = stage.existing()
        write_json(parent.state_dir / 'collection/latest.json', {'run_id': stage.run_id, 'run_dir': str(target)})
        return
    _pin_selection(parent, stage, sources)
    _preflight(parent, sources)
    split_wordpress(parent, sources, run=True)
    stage.pin_inputs([path, *(source.settings.baseline_file for source in sources)])
    checkpoint = parent.output_dir / '.requests' / f'{stage.run_id}.sources.json'
    completed = read(checkpoint) if checkpoint.exists() else {}
    outputs = []
    # The top-level stage lock is held by stage_cli. Source locks are owned by
    # the native collectors; completed archives are checked before each retry.
    for source in sources:
        if source.kind == 'wordpress':
            from .pipeline import recover_state
            if (source.settings.state_dir / '.pending').exists():
                recover_state(source.settings)
        elif source.kind == 'instagram':
            from .storage import recover
            if (source.settings.state_dir / '.pending').exists():
                with FileLock(str(source.settings.lock_file), timeout=0):
                    recover(source.settings)
        committed = _committed(source, stage.run_id)
        if not committed:
            if source.key in completed:
                raise ValueError(f'{source.key}: committed source disappeared; restore before retrying')
            scan_source(source, allow_large_deletions=allow_large_deletions)
            committed = _committed(source, stage.run_id)
            if not committed:
                raise ValueError(f'{source.key}: source failed to commit its batch')
        target, report = committed
        signature = inventory(target)
        details = {'archive': str(target), 'inventory': signature}
        if source.key in completed and completed[source.key] != details:
            raise ValueError(f'{source.key}: source archive changed during retry')
        completed[source.key] = details
        write_json(checkpoint, completed)
        outputs.append((source, target))
    new, deleted = [], []
    for source, target in outputs:
        batch = _frame(target / 'chatbot_dataset_new.parquet')
        removed = _frame(target / 'chatbot_dataset_deleted.parquet')
        allowed = {'page' if source.key.endswith('/pages') else 'post'} if source.kind == 'wordpress' else {source.kind}
        if not batch.source_type.isin(allowed).all() or not removed.source_type.isin(allowed).all():
            raise ValueError('Source output contains identities from another source')
        if source.kind == 'instagram' and not removed.empty:
            raise ValueError('Instagram deletion inference is disabled')
        new.extend(batch.to_dict('records'))
        deleted.extend(removed.to_dict('records'))
    fresh, removed = dataset(new), dataset(deleted, 'deleted')
    for frame in (fresh, removed):
        if frame.duplicated(['source_type', 'id']).any():
            raise ValueError('Conflicting identities across collection sources')
    if set(zip(fresh.source_type, fresh.id)) & set(zip(removed.source_type, removed.id)):
        raise ValueError('Same identity occurs in both NEW and DELETED')
    target = run_path(parent.output_dir)
    report = dict(pipeline_run_id=stage.run_id, operation='collect', owner='collection',
                  comparison={source.key: comparison(source) for source in sources}, run_dir=str(target), archive_dir=str(parent.output_dir),
                  shared_dir=str(parent.shared_raw_dir), new=len(fresh), deleted=len(removed),
                  selected_sources=[public_name(source.key) for source in sources],
                  sources=completed)
    with archive_run(target, validate_raw) as staging:
        fresh.sort_values(['source_type', 'id']).to_parquet(staging / 'chatbot_dataset_new.parquet', index=False)
        removed.sort_values(['source_type', 'id']).to_parquet(staging / 'chatbot_dataset_deleted.parquet', index=False)
        report['hashes'] = {event: _sha(staging / f'chatbot_dataset_{event}.parquet') for event in ('new', 'deleted')}
        for source, source_dir in outputs:
            if inventory(source_dir) != completed[source.key]['inventory']:
                raise ValueError('A source changed during unified RAW preparation')
        write_json(staging / 'run_report.json', report)
    write_json(parent.state_dir / 'collection/latest.json', {'run_id': stage.run_id, 'run_dir': str(target)})
    publish(target, parent.shared_raw_dir, validate_raw)
    stage.finish(target, report)


def status(config_path):
    path, parent, sources = sources_for(config_path)
    result = {}
    for source in sources:
        settings = source.settings
        entry = {'baseline': str(settings.baseline_file), 'snapshot': str(settings.snapshot_file),
                 'initialized': settings.baseline_file.is_file() and settings.snapshot_file.is_file(),
                 'comparison': comparison(source)}
        if source.kind == 'wordpress':
            entry['legacy_migration_available'] = parent.baseline_file.is_file() and parent.snapshot_file.is_file()
        if source.kind == 'instagram':
            entry['session_present'] = settings.auth_file.is_file()
        result[source.key] = entry
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    disabled = [name for name, default in [('wordpress', True), ('instagram', False), ('news', False)]
                if not enabled(config.get(name, {}), default)]
    return {'sources': result, 'disabled_sources': disabled,
            'shared_dir': str(parent.shared_raw_dir), 'archive_dir': str(parent.output_dir)}
