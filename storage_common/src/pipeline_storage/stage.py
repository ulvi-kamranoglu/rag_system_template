"""Public stage protocol. No dependency on the pipeline coordinator."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from uuid import uuid4

from filelock import FileLock
from . import inventory, publish, run_path, recover_publication

MODES = ('collect', 'join', 'reset_data')
PRIVATE_ENV = ('PIPELINE_DATASET', 'PIPELINE_BATCH_DATASET', 'PIPELINE_BATCH_REPORT',
               'PIPELINE_BASE_SNAPSHOT', 'PIPELINE_MODE', 'PIPELINE_RUN_ID',
               'EMBEDDING_REUSE_DATASET', 'RETRIEVAL_CONFIG', 'EMBEDDING_CONFIG')


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument('--config')
    result.add_argument('--mode', choices=MODES)
    result.add_argument('--run-id', help='Idempotency/batch identifier; optional for standalone use')
    result.add_argument('--result', type=Path, help='Optional machine-readable completion receipt')
    return result


def clean_environment():
    for key in PRIVATE_ENV:
        os.environ.pop(key, None)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def files(folder):
    return {key: value for key, value in inventory(folder).items() if key != 'run_report.json'}


def child_report(folder, filename):
    """Accept only direct artifact names, never arbitrary paths from metadata."""
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or any(c in filename for c in '/\\:'):
        raise ValueError('Invalid artifact filename in stage report')
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@contextmanager
def stage_locks(*folders):
    # These locks also protect readers of an upstream stage during publication.
    from contextlib import ExitStack
    with ExitStack() as stack:
        for folder in sorted({Path(p).resolve() for p in folders}):
            folder.parent.mkdir(parents=True, exist_ok=True)
            stack.enter_context(FileLock(str(folder.with_name('.' + folder.name + '.stage.lock')), timeout=0))
            recover_publication(folder)
        yield


def upstream(folder, expected_stage, run_id=None, mode=None):
    report = read(folder / 'run_report.json')
    contract = report.get('contract', {})
    if contract.get('version') != 1 or contract.get('stage') != expected_stage:
        raise ValueError(f'{expected_stage} must publish a version-1 stage contract first')
    if run_id and contract['run_id'] != run_id:
        raise ValueError(f'Stale {expected_stage} output: batch/run ID differs')
    if mode and contract['mode'] != mode:
        raise ValueError(f'{expected_stage} mode differs')
    if files(folder) != contract['files']:
        raise ValueError(f'{expected_stage} shared artifacts changed after publication')
    return report


class Stage:
    def __init__(self, name, args, archive, shared, source=None):
        self.name, self.args = name, args
        self.archive, self.shared = Path(archive), Path(shared)
        inherited = (source or {}).get('contract', {})
        self.run_id = args.run_id or inherited.get('run_id') or uuid4().hex
        if not self.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in self.run_id):
            raise ValueError('Invalid run ID')
        self.mode = args.mode or inherited.get('mode') or ('collect' if name == 'collection' else 'join')
        os.environ.update(PIPELINE_RUN_ID=self.run_id, PIPELINE_MODE=self.mode)

    def existing(self):
        matches = []
        for path in self.archive.glob('run_*/run_report.json'):
            report = read(path)
            if report.get('pipeline_run_id') == self.run_id:
                contract = report.get('contract', {})
                if contract and (contract['mode'] != self.mode or contract['stage'] != self.name):
                    raise ValueError('Run ID already belongs to a different operation')
                matches.append((path.parent, report))
        if len(matches) > 1:
            raise ValueError('Multiple archives for this stage/run ID')
        return matches[0] if matches else None

    def recover(self, publisher):
        existing = self.existing()
        if not existing:
            return False
        target, report = existing
        contract = report.get('contract', {})
        if contract.get('files') is not None:
            archived = files(target)
            if self.name == 'canonical':
                archived = {key: value for key, value in archived.items()
                            if key != 'base_snapshot' and not key.startswith('base_snapshot/')}
            if archived != contract['files']:
                raise ValueError('Archived artifacts changed after completion')
        status = report.get('contract', {}).get('status', 'completed')
        # The payload is already published if only the receipt update was interrupted.
        if self.shared.is_dir() and files(self.shared) == files(target):
            current = read(self.shared / 'run_report.json')
            if current.get('pipeline_run_id') != self.run_id:
                raise ValueError('Shared output belongs to a newer/different run')
        elif status in ('skipped', 'no_changes'):
            publish(target, self.shared, lambda folder: read(folder / 'run_report.json'))
        else:
            publisher(target)
        self.finish(target, report, status=status)
        return True

    def pin_inputs(self, inputs):
        """Remember a request even if processing fails before an archive commits."""
        import hashlib
        signatures = {}
        for path in inputs:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            signatures[str(path.resolve())] = digest.hexdigest()
        request = {'mode': self.mode, 'inputs': signatures}
        path = self.archive / '.requests' / f'{self.run_id}.json'
        if path.exists() and read(path) != request:
            raise ValueError('Stage inputs changed since the previous attempt; restore the original inputs')
        write_json(path, request)

    def finish(self, target, report, *, status='completed'):
        contract = {'version': 1, 'stage': self.name, 'run_id': self.run_id,
                    'mode': self.mode, 'status': status, 'files': files(self.shared)}
        report = {**report, 'contract': contract}
        write_json(target / 'run_report.json', report)
        write_json(self.shared / 'run_report.json', report)
        result = {key: contract[key] for key in ('version', 'stage', 'run_id', 'mode', 'status')}
        result['dataset_name'] = report.get('index_dataset_name') or report.get('dataset_name')
        result['archive'] = str(target)
        if self.args.result:
            self.args.result.parent.mkdir(parents=True, exist_ok=True)
            write_json(self.args.result, result)
        print(json.dumps(result, ensure_ascii=False))

    def skip(self, reason, *, status='skipped'):
        target = run_path(self.archive)
        target.mkdir(parents=True)
        report = {'pipeline_run_id': self.run_id, 'run_dir': str(target), 'reason': reason,
                  'contract': {'status': status, 'stage': self.name, 'mode': self.mode}}
        write_json(target / 'run_report.json', report)
        publish(target, self.shared, lambda folder: read(folder / 'run_report.json'))
        self.finish(target, report, status=status)
