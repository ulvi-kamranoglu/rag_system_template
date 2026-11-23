"""The coordinator can sequence replaceable stages with no stage imports."""
import json
import pytest
import yaml
from orchestration import runner
from orchestration.config import snapshot_config


def make_state(mode='join'):
    return dict(protocol=2, run_id='run-one', mode=mode, completed={},
                commands={name: ['replacement.' + name] for name in ('collection', 'canonical', 'chunk', 'embedding', 'retrieval')})


@pytest.mark.parametrize('mode', ['collect', 'join', 'reset_data'])
def test_order_resume_and_opaque_receipts(tmp_path, monkeypatch, mode):
    state = make_state(mode)
    calls = []
    fail = [True]
    def fake(command, folder, env, label):
        calls.append(label)
        assert command[0] == 'replacement.' + label
        assert command[command.index('--mode') + 1] == mode
        assert not any(key in env for key in ('PIPELINE_BATCH_REPORT', 'PIPELINE_BASE_SNAPSHOT'))
        if label == 'embedding' and fail[0]:
            fail[0] = False
            raise RuntimeError('offline failure')
        receipt = folder / f'{label}.result.json'
        receipt.write_text(json.dumps(dict(version=1, stage=label, run_id=state['run_id'], mode=mode,
                                          status='completed', dataset_name='example_v1.6')))
    monkeypatch.setattr(runner, 'call', fake)
    with pytest.raises(RuntimeError, match='offline failure'):
        runner.execute(tmp_path, state)
    runner.execute(tmp_path, state)
    expected = ['canonical', 'chunk', 'embedding', 'embedding', 'retrieval']
    if mode == 'collect':
        expected.insert(0, 'collection')
    assert calls == expected
    assert state['status'] == 'complete'
    assert state['dataset_name'] == 'example_v1.6'
    assert not (tmp_path / 'inputs').exists()


def test_bad_receipt_stops_pipeline(tmp_path, monkeypatch):
    state = make_state()
    def fake(command, folder, env, label):
        (folder / f'{label}.result.json').write_text(json.dumps(dict(version=1, stage=label, run_id='wrong', mode='join', status='completed')))
    monkeypatch.setattr(runner, 'call', fake)
    with pytest.raises(ValueError, match='receipt'):
        runner.execute(tmp_path, state)
    assert state['completed'] == {}
    assert state['current_stage'] == 'canonical'


def test_no_changes_does_not_run_downstream(tmp_path, monkeypatch):
    state = make_state()
    calls = []
    def fake(command, folder, env, label):
        calls.append(label)
        (folder / f'{label}.result.json').write_text(json.dumps(dict(version=1, stage=label, run_id=state['run_id'], mode='join', status='no_changes')))
    monkeypatch.setattr(runner, 'call', fake)
    runner.execute(tmp_path, state)
    assert calls == ['canonical']
    assert state['status'] == 'no_changes'


def test_config_snapshot_preserves_stage_semantics(tmp_path):
    source = tmp_path / 'stage/config/config.yaml'
    source.parent.mkdir(parents=True)
    source.write_text(yaml.safe_dump(dict(merge_base=True, reset_data=False, custom='unchanged', input_dir='../shared/raw')))
    for stage in ('canonical', 'chunk', 'embedding'):
        result = snapshot_config(source, tmp_path / f'{stage}.yaml', stage, mode='reset_data')
        assert result['merge_base'] is True and result['reset_data'] is False
        assert result['custom'] == 'unchanged'
        assert result['input_dir'] == str((tmp_path / 'shared/raw').resolve())


def test_cancel_refuses_an_attempted_stage(tmp_path):
    folder = tmp_path / 'run_failed'
    folder.mkdir()
    state = make_state('collect')
    state['current_stage'] = 'collection'
    (folder / 'state.json').write_text(json.dumps(state))
    (tmp_path / 'pending.json').write_text(json.dumps({'run_dir': str(folder)}))
    with pytest.raises(ValueError, match='attempted'):
        runner.cancel_pending(tmp_path)
    state['current_stage'] = None
    (folder / 'state.json').write_text(json.dumps(state))
    runner.cancel_pending(tmp_path)
    assert not (tmp_path / 'pending.json').exists()
