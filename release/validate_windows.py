"""Verify the complete 0.5.12 distribution with its actual Windows launcher."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'release'
REPORTS = ROOT / 'reports'
VERSION = '0.5.12'


def native_geometry_tests():
    dependencies = os.environ.get('BADMINTON_TESTING_DEPS')
    if not dependencies:
        raise RuntimeError('BADMINTON_TESTING_DEPS must identify isolated pytest dependencies')
    with tempfile.TemporaryDirectory(prefix='badminton-native-test-') as temporary:
        report_path = Path(temporary) / 'native-geometry.json'
        command = [str(ROOT / 'runtime/python.exe'), '-B',
                   str(RELEASE / 'run_native_geometry_tests.py'),
                   '--package', str(ROOT), '--pytest-deps', dependencies,
                   '--report', str(report_path)]
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                       text=True, encoding='utf-8', errors='replace', timeout=600)
        except subprocess.TimeoutExpired as error:
            def decode(value):
                return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''
            return {'status': 'failed', 'returncode': None, 'timed_out': True,
                    'stdout': decode(error.stdout), 'stderr': decode(error.stderr)}
        data = json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else {'status': 'failed'}
        data.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
        print(completed.stdout, end='')
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end='')
        return data


def main():
    if sys.platform != 'win32':
        raise RuntimeError('Run the distribution validation on Windows')
    receipt = json.loads((RELEASE / 'receipt.json').read_text(encoding='utf-8'))
    if receipt['app_version'] != VERSION or receipt['source_worktree_dirty'] is not False:
        raise RuntimeError('Expected a clean 0.5.12 package receipt')
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f'windows-{VERSION}-validation.json'
    print(f'Checking complete {VERSION} distribution', flush=True)
    subprocess.run([sys.executable, str(RELEASE / 'verify_windows_portable.py'),
                    '--package', str(ROOT), '--report', str(path)],
                   check=True, timeout=900)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['status'] != 'passed' or data['app_version'] != VERSION:
        raise RuntimeError('Unexpected package or failed Windows startup evidence')
    if (data['source_commit'] != receipt['source_commit']
            or data['source_snapshot_sha256'] != receipt['source_snapshot_sha256']):
        raise RuntimeError('Windows evidence does not match the packaged source')
    data.update(tested_distribution=f'full_v{VERSION}',
                release_archive_sha256=receipt['sha256'])
    try:
        native = native_geometry_tests()
    except Exception as error:
        native = {'status': 'failed', 'returncode': None, 'error': f'{type(error).__name__}: {error}'}
    data['native_geometry_tests'] = native
    if native.get('status') != 'passed' or native.get('returncode') != 0:
        data['status'] = 'failed'
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if data['status'] != 'passed':
        raise RuntimeError('Bundled-runtime native geometry tests failed; see Windows report')
    print('Complete Windows distribution passed startup and 61 native geometry tests', flush=True)


if __name__ == '__main__':
    main()
