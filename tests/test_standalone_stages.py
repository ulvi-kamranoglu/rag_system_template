"""Isolated filesystem pipeline. Models and Qdrant are replaced in memory."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml
from pipeline_storage.stage import read, upstream


class Tokenizer:
    def encode(self, text, add_special_tokens=True, truncation=False):
        result = list(text.encode('utf-8'))
        return [1] + result + [2] if add_special_tokens else result
    def decode(self, ids, **kwargs):
        return bytes(ids).decode('utf-8')
    def num_special_tokens_to_add(self, pair=False):
        return 2


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path
    shared, archive = root / 'shared_data', root / 'archive_data'
    raw = shared / '01_raw'
    raw.mkdir(parents=True)
    def configure(project, stage, **extra):
        path = root / project / 'config/config.yaml'
        path.parent.mkdir(parents=True)
        data = dict(storage=dict(archive_dir=str(archive / stage), shared_dir=str(shared / stage)), **extra)
        path.write_text(yaml.safe_dump(data))
        return path
    configs = {}
    configs['canonical'] = configure('canonical', '02_clean', input_dir=str(raw))
    configs['chunk'] = configure('chunk', '03_chunks', input_dir=str(shared / '02_clean'), model_id='offline',
                                variants=[dict(max_tokens=384, overlap_tokens=64)])
    configs['embedding'] = configure('embedding', '04_embeddings', input_dir=str(shared / '03_chunks'),
                                    model=dict(id='offline', dimension=2, document_max_length=384))
    configs['retrieval'] = configure('retrieval', '05_indexes', input_dir=str(shared / '04_embeddings'),
                                    project_root=str(root), qdrant=dict(vector_dimension=2, collection='test'))
    base = archive / '02_clean/base_data'
    base.mkdir(parents=True)
    row = dict(id='post:1', source_type='post', title='Original', content='Original university information.', url='https://example.org/1', metadata='{}')
    from canonical_pipeline.schema import standardize
    standardize(pd.DataFrame([row])).to_parquet(base / 'sample_v1.4_canonical.parquet')
    pd.DataFrame([{**row, 'id': 'post:2', 'title': 'New record', 'content': 'New university information.', 'url': 'https://example.org/2'}]).to_parquet(raw / 'example.parquet')
    from canonical_pipeline import pipeline as canonical
    monkeypatch.setattr(canonical, 'detector', lambda: None)
    monkeypatch.setattr(canonical, 'detect_language', lambda *texts: dict(language='en', confidence=0.99, uncertain=False))
    from chunking_pipeline import pipeline as chunking
    monkeypatch.setattr(chunking.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    class Embedder:
        def __init__(self, settings): self.device = 'test'
        def embed_texts(self, texts, **kwargs):
            return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(texts), 1))
    monkeypatch.setitem(sys.modules, 'embedding.embedder', SimpleNamespace(BgeM3Embedder=Embedder))
    indexes = shared / '05_indexes'
    indexes.mkdir()
    pl.DataFrame(dict(chunk_id=['old'], document_id=['post:1'])).write_parquet(indexes / 'sample_v1.4_bm25_corpus.parquet')
    (indexes / 'active_dataset.json').write_text(json.dumps(dict(dataset_name='sample_v1.4', qdrant_collection='test')))
    return root, configs


def run_stage(name, configs, run_id='batch-one', mode='join'):
    from importlib import import_module
    module = dict(canonical='canonical_pipeline', chunk='chunking_pipeline', embedding='embedding', retrieval='retrieval')[name]
    import_module(module + '.stage_cli').main(['--config', str(configs[name]), '--mode', mode, '--run-id', run_id])


@pytest.mark.parametrize('mode', ['join', 'reset_data', 'collect'])
def test_stages_run_without_coordinator_and_recover(workspace, monkeypatch, mode):
    root, configs = workspace
    raw = root / 'shared_data/01_raw'
    if mode == 'collect':
        # Collector's public file contract is sufficient; no wp_tracker imports.
        from pipeline_storage.stage import Stage
        from pipeline_storage import run_path
        args = SimpleNamespace(run_id='batch-one', mode=mode, result=None)
        stage = Stage('collection', args, root / 'archive_data/01_raw', raw)
        target = run_path(stage.archive)
        target.mkdir(parents=True)
        import shutil
        shutil.copy2(raw / 'example.parquet', target / 'example.parquet')
        stage.finish(target, {'pipeline_run_id': stage.run_id, 'run_dir': str(target)})
    run_stage('canonical', configs, mode=mode)
    clean = root / 'shared_data/02_clean'
    report = upstream(clean, 'canonical', 'batch-one', mode)
    assert pl.read_parquet(clean / report['output_file']).height == 1
    base = root / 'archive_data/02_clean/base_data/sample_v1.5_canonical.parquet'
    assert pl.read_parquet(base).height == (1 if mode == 'reset_data' else 2)
    run_stage('canonical', configs, mode=mode)
    assert not (base.parent / 'sample_v1.6_canonical.parquet').exists()
    run_stage('chunk', configs, mode=mode)
    run_stage('embedding', configs, mode=mode)
    from retrieval import config, artifacts
    monkeypatch.setattr(config, '_settings', None)
    monkeypatch.setattr(artifacts, 'VECTOR_DIMENSION', 2)
    from retrieval.indexing import sync
    remote = {'old': 'old'}
    writes = []
    class Index:
        client = SimpleNamespace(get_collection=lambda name: SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=2, distance='Cosine')))))
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def health_check(self): pass
        def collection_exists(self): return True
        def count(self): return len(remote)
        def get_chunk_state(self): return dict(remote)
        def upsert(self, vectors, payloads, size):
            writes.append(len(payloads))
            remote.update({p['chunk_id']: 'new' for p in payloads})
            return len(payloads)
        def delete_chunks(self, ids):
            for key in ids: remote.pop(key)
            return len(ids)
    monkeypatch.setattr(sync, 'QdrantVectorIndex', Index)
    from retrieval.indexing import storage
    publish = storage._publish
    first = [True]
    def fail_local_publication(*args, **kwargs):
        if first[0]:
            first[0] = False
            raise RuntimeError('Local publication interrupted after remote sync')
        return publish(*args, **kwargs)
    monkeypatch.setattr(storage, '_publish', fail_local_publication)
    with pytest.raises(RuntimeError, match='publication interrupted'):
        run_stage('retrieval', configs, mode=mode)
    run_stage('retrieval', configs, mode=mode)
    run_stage('retrieval', configs, mode=mode)
    assert len(writes) == 1  # Publication retry never replays remote writes.
    output = root / 'shared_data/05_indexes'
    assert read(output / 'active_dataset.json')['dataset_name'] == 'sample_v1.5'
    assert pl.read_parquet(output / 'sample_v1.5_bm25_corpus.parquet').height == (1 if mode == 'reset_data' else 2)
    assert ('old' in remote) == (mode != 'reset_data')
    upstream(output, 'retrieval', 'batch-one', mode)


def test_downstream_rejects_stale_or_modified_batch(workspace):
    root, configs = workspace
    run_stage('canonical', configs)
    with pytest.raises(ValueError, match='Stale'):
        run_stage('chunk', configs, run_id='wrong')
    (root / 'shared_data/02_clean/unexpected.txt').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        run_stage('chunk', configs)


def test_delete_only_batch_skips_models(workspace):
    root, configs = workspace
    raw = root / 'shared_data/01_raw'
    pd.DataFrame(columns=['id', 'source_type', 'title', 'content']).to_parquet(raw / 'example.parquet')
    pd.DataFrame({'document_id': ['post:99']}).to_parquet(raw / 'chatbot_dataset_deleted.parquet')
    run_stage('canonical', configs)
    run_stage('chunk', configs)
    run_stage('embedding', configs)
    assert upstream(root / 'shared_data/04_embeddings', 'embedding')['contract']['status'] == 'skipped'


def test_empty_reset_rejected_before_new_baseline(workspace):
    root, configs = workspace
    pd.DataFrame(columns=['id', 'content']).to_parquet(root / 'shared_data/01_raw/example.parquet')
    with pytest.raises(ValueError, match='nonempty'):
        run_stage('canonical', configs, mode='reset_data')
    assert not (root / 'archive_data/02_clean/base_data/sample_v1.5_canonical.parquet').exists()


def test_failed_canonical_attempt_pins_original_input(workspace, monkeypatch):
    root, configs = workspace
    from canonical_pipeline import pipeline
    process = pipeline.process
    monkeypatch.setattr(pipeline, 'process', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('processing interrupted')))
    with pytest.raises(RuntimeError, match='interrupted'):
        run_stage('canonical', configs)
    monkeypatch.setattr(pipeline, 'process', process)
    raw = root / 'shared_data/01_raw/example.parquet'
    pd.DataFrame({'id': ['different'], 'content': ['Different input']}).to_parquet(raw)
    with pytest.raises(ValueError, match='inputs changed'):
        run_stage('canonical', configs)


def test_no_changes_publishes_a_receipt_without_new_version(workspace):
    root, configs = workspace
    pd.DataFrame(columns=['id', 'content']).to_parquet(root / 'shared_data/01_raw/example.parquet')
    run_stage('canonical', configs)
    report = upstream(root / 'shared_data/02_clean', 'canonical')
    assert report['contract']['status'] == 'no_changes'
    assert not (root / 'archive_data/02_clean/base_data/sample_v1.5_canonical.parquet').exists()
