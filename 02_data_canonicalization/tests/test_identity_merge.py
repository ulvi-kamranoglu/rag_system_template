"""Renaming an uploaded identity replaces its legacy counterpart on join."""
from pathlib import Path
import pandas as pd
import pytest
from canonical_pipeline import pipeline
from canonical_pipeline.schema import standardize


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'detector', lambda: None)
    monkeypatch.setattr(pipeline, 'detect_language', lambda *a: dict(language='en', confidence=1., uncertain=False))
    def row(identifier, source, title):
        return dict(id=identifier, source_type='instagram', source=source, title=title,
                    content=title + ' university information.', url='https://example.org/' + title,
                    language='en', metadata='{}')
    base = tmp_path / 'archive/base_data'
    base.mkdir(parents=True)
    standardize(pd.DataFrame([row('research:1', 'instagram:adalat.muradov', 'Old rector'),
                              row('instagram:1', 'instagram:uneceduaz', 'Untouched university')])).to_parquet(base / 'sample_v1.6_canonical.parquet')
    return tmp_path, row


def test_join_replaces_legacy_identity_without_touching_unrelated_ids(state):
    root, row = state
    incoming = root / 'rector.parquet'
    pd.DataFrame([row('research:1', 'instagram:adalat.muradov', 'New rector')]).to_parquet(incoming)
    for minor in (7, 8):
        report = pipeline.process([incoming], root / 'archive', root / 'shared', merge_base=True)
        full = pd.read_parquet(root / 'archive/base_data' / f'sample_v1.{minor}_canonical.parquet')
        assert set(full.id) == {'instagram:adalat.muradov:1', 'instagram:1'}
        assert len(full) == 2
        batch = pd.read_parquet(Path(report['run_dir']) / report['output_file'])
        assert batch.id.tolist() == ['instagram:adalat.muradov:1']


def test_normalized_deletion_removes_legacy_identity(state):
    root, _ = state
    incoming = root / 'empty.parquet'
    pd.DataFrame(columns=['id', 'source_type', 'content']).to_parquet(incoming)
    deleted = root / 'deleted.parquet'
    pd.DataFrame([dict(id='research:1', source_type='instagram', source='instagram:adalat.muradov')]).to_parquet(deleted)
    report = pipeline.process([incoming], root / 'archive', root / 'shared', merge_base=True, deleted_path=deleted)
    assert report['deleted_rows'] == 1
    full = pd.read_parquet(root / 'archive/base_data/sample_v1.7_canonical.parquet')
    assert full.id.tolist() == ['instagram:1']


def test_conflicting_upload_ids_after_normalization_are_rejected(state):
    root, row = state
    incoming = root / 'conflict.parquet'
    pd.DataFrame([row('research:1', 'instagram:adalat.muradov', 'First'),
                  row('instagram:adalat.muradov:1', 'instagram:adalat.muradov', 'Different')]).to_parquet(incoming)
    with pytest.raises(ValueError, match='Conflicting document identities'):
        pipeline.process([incoming], root / 'archive', root / 'shared', merge_base=True)
