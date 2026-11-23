"""Sequence external stage commands; stages own data and recovery decisions."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from .config import ROOT
from .state import save


def call(command, folder, env, label):
    print(f"[{label}] starting; log: {folder / (label + '.log')}", flush=True)
    with (folder / (label + '.log')).open('a', encoding='utf-8') as output:
        result = subprocess.run([sys.executable, '-X', 'utf8', '-m', *command],
                                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label} failed (exit {result.returncode}); see {folder / (label + '.log')}")
    print(f'[{label}] complete', flush=True)


def cancel_pending(state_root):
    pending = state_root / 'pending.json'
    if not pending.is_file():
        raise ValueError('No pending run to cancel')
    folder = Path(json.loads(pending.read_text(encoding='utf-8'))['run_dir']).resolve()
    if folder.parent != state_root.resolve() or not folder.name.startswith('run_'):
        raise ValueError('Invalid pending run directory')
    state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
    if state['completed'] or state.get('current_stage'):
        raise ValueError('A stage has already been attempted; resume it instead of cancelling')
    state.update(status='cancelled', cancelled_at=datetime.now(timezone.utc).isoformat())
    save(folder / 'state.json', state)
    pending.unlink()


def execute(folder, state):
    if state.get('protocol') != 2:
        raise ValueError('Legacy pending run uses the old coordinator protocol; finish it with the previous release before upgrading')
    state['status'] = 'running'
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    stages = ['collection', 'canonical', 'chunk', 'embedding', 'retrieval']
    if state['mode'] != 'collect':
        stages.remove('collection')
    for name in stages:
        if name in state['completed']:
            continue
        state['current_stage'] = name
        save(folder / 'state.json', state)
        receipt = folder / f'{name}.result.json'
        receipt.unlink(missing_ok=True)
        command = [*state['commands'][name], '--config', str(folder / f'{name}.yaml'),
                   '--mode', state['mode'], '--run-id', state['run_id'], '--result', str(receipt),
                   *state.get('stage_options', {}).get(name, [])]
        if name == 'canonical':
            if state.get('raw_source'):
                command += ['--input', state['raw_source']]
            if state.get('deleted_source'):
                command += ['--deleted', state['deleted_source']]
        call(command, folder, env, name)
        result = json.loads(receipt.read_text(encoding='utf-8'))
        if (result.get('version') != 1 or result.get('stage') != name or
                result.get('run_id') != state['run_id'] or result.get('mode') != state['mode'] or
                result.get('status') not in ('completed', 'skipped', 'no_changes')):
            raise ValueError(f'Invalid completion receipt from {name}')
        state['completed'][name] = result
        state['current_stage'] = None
        save(folder / 'state.json', state)
        if result['status'] == 'no_changes':
            state['status'] = 'no_changes'
            save(folder / 'state.json', state)
            return
    state.update(status='complete', dataset_name=state['completed']['retrieval'].get('dataset_name'))
    save(folder / 'state.json', state)
