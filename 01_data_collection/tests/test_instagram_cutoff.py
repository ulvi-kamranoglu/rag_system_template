"""Instagram: canonical IDs and NEW limited to posts newer than the newest collected post."""
import json

import pandas as pd
import pytest

from wp_tracker.instagram import service
from wp_tracker.instagram.config import InstagramSettings
from wp_tracker.instagram.schema import frame, new_row, normalize_seed_id


def settings_for(tmp_path, profile='uneceduaz'):
    return InstagramSettings(root=tmp_path, username=profile, archive_dir=tmp_path / 'archive' / profile,
                             shared_dir=tmp_path / 'shared' / profile, state_dir=tmp_path / 'state' / profile,
                             auth_file=tmp_path / 'auth' / f'{profile}.json')


def seed(tmp_path, profile, rows):
    path = tmp_path / f'{profile}_seed.parquet'
    records = []
    for identifier, code, date in rows:
        row = new_row(profile, f'https://www.instagram.com/p/{code}/', f'Caption {code}.', pd.Timestamp(date, tz='UTC'))
        row['id'] = identifier
        records.append(row)
    frame(records).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize('profile,original,expected', [
    ('uneceduaz', 'instagram:5', 'instagram:uneceduaz:5'),
    ('uneceduaz', 'research:007', 'instagram:uneceduaz:7'),
    ('uneceduaz', 'instagram:uneceduaz:9', 'instagram:uneceduaz:9'),
    ('adalat.muradov', 'instagram:adalat.muradov:3', 'instagram:adalat.muradov:3'),
    ('adalat.muradov', 'research:adalat.muradov:4', 'instagram:adalat.muradov:4'),
    ('adalat.muradov', 'research:1', 'instagram:adalat.muradov:1'),
])
def test_seed_ids_follow_canonical_standard(profile, original, expected):
    assert normalize_seed_id(profile, original) == expected


@pytest.mark.parametrize('profile,original', [
    ('adalat.muradov', 'instagram:5'),          # unqualified IDs belong to the UNEC legacy only
    ('uneceduaz', 'instagram:adalat.muradov:1'),
    ('uneceduaz', 'instagram:uneceduaz:DHWRYR7OO9G'),
    ('uneceduaz', 'instagram:uneceduaz:0'),
])
def test_unexpected_seed_ids_rejected(profile, original):
    with pytest.raises(ValueError):
        normalize_seed_id(profile, original)


def scanner(posts, calls):
    def collect(settings, known, cutoff, on_new, **kwargs):
        calls.append({'cutoff': cutoff, 'known': dict(known)})
        for code, date in posts:
            url = f'https://www.instagram.com/p/{code}/'
            stamp = pd.Timestamp(date, tz='UTC').to_pydatetime()
            if url not in known and stamp > cutoff:
                on_new(url, f'Caption {code}.', stamp)
        return {'boundary_reached': True}
    return collect


def snapshot_ids(settings):
    return pd.read_parquet(settings.snapshot_file)['id'].tolist()


def test_new_is_only_posts_newer_than_the_newest_collected(tmp_path):
    settings = settings_for(tmp_path)
    service.bootstrap(settings, seed(tmp_path, 'uneceduaz', [('instagram:1', 'A', '2026-10-01'),
                                                              ('instagram:2', 'B', '2026-10-05')]), run=True)
    assert snapshot_ids(settings) == ['instagram:uneceduaz:1', 'instagram:uneceduaz:2']
    assert json.loads(settings.cutoff_file.read_text())['last_real_post_utc'].startswith('2026-10-05')

    calls = []
    first = service.scan(settings, run=True, collector=scanner([('C', '2026-11-01'), ('OLD', '2026-09-01')], calls))
    assert calls[0]['cutoff'] == pd.Timestamp('2026-10-05', tz='UTC')
    # Known posts are passed with their dates, so the browser never reopens them.
    assert calls[0]['known']['https://www.instagram.com/p/A/'] == pd.Timestamp('2026-10-01', tz='UTC')
    new = pd.read_parquet(first['outputs']['new'])
    assert new['id'].tolist() == ['instagram:uneceduaz:3']       # older unknown post is not backfilled
    assert first['cutoff_after'].startswith('2026-11-01')

    second = service.scan(settings, run=True, collector=scanner([('C', '2026-11-01')], calls))
    assert second['new'] == 0 and pd.read_parquet(second['outputs']['new']).empty
    assert second['cutoff_after'] == first['cutoff_after']

    third = service.scan(settings, run=True, collector=scanner([('D', '2026-11-03'), ('E', '2026-11-02')], calls))
    assert pd.read_parquet(third['outputs']['new'])['id'].tolist() == ['instagram:uneceduaz:4', 'instagram:uneceduaz:5']
    assert snapshot_ids(settings)[-2:] == ['instagram:uneceduaz:4', 'instagram:uneceduaz:5']
    assert third['comparison'] == 'cutoff_date'


def test_numbering_continues_after_gaps_in_seed(tmp_path):
    settings = settings_for(tmp_path)
    service.bootstrap(settings, seed(tmp_path, 'uneceduaz', [('instagram:1', 'A', '2026-10-01'),
                                                              ('instagram:4114', 'B', '2026-10-05')]), run=True)
    result = service.scan(settings, run=True, collector=scanner([('C', '2026-11-01')], []))
    assert pd.read_parquet(result['outputs']['new'])['id'].tolist() == ['instagram:uneceduaz:4115']


def test_new_ids_match_canonicalization(tmp_path):
    from canonical_pipeline.identity import canonical_identity
    settings = settings_for(tmp_path, 'adalat.muradov')
    service.bootstrap(settings, seed(tmp_path, 'adalat.muradov', [('research:1', 'A', '2026-10-01')]), run=True)
    result = service.scan(settings, run=True, collector=scanner([('B', '2026-11-01')], []))
    for row in pd.read_parquet(result['outputs']['new']).to_dict('records'):
        assert canonical_identity(row)[2] == row['id'] == 'instagram:adalat.muradov:2'
