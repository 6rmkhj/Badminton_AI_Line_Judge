"""Verify the complete 0.5.13 distribution with its actual Windows launcher."""
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
VERSION = '0.5.13'


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


def full_analysis_tests():
    report_path = REPORTS / 'windows-full-analysis.json'
    command = [str(ROOT / 'runtime/python.exe'), '-B',
               str(ROOT / 'app/scripts/verify_windows_analysis.py'),
               '--package', str(ROOT), '--work-dir', str(REPORTS / '한글 분석 검증'),
               '--report', str(report_path)]
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                                   text=True, encoding='utf-8', errors='replace', timeout=1800)
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
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    if sys.platform != 'win32':
        raise RuntimeError('Run the distribution validation on Windows')
    receipt = json.loads((RELEASE / 'receipt.json').read_text(encoding='utf-8'))
    if receipt['app_version'] != VERSION or receipt['source_worktree_dirty'] is not False:
        raise RuntimeError('Expected a clean 0.5.13 package receipt')
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
    try:
        full = full_analysis_tests()
    except Exception as error:
        full = {'status': 'failed', 'returncode': None, 'error': f'{type(error).__name__}: {error}'}
    data['full_analysis_tests'] = full
    if (full.get('status') != 'passed' or full.get('returncode') != 0
            or full.get('source_commit') != receipt['source_commit']
            or full.get('source_snapshot_sha256') != receipt['source_snapshot_sha256']):
        data['status'] = 'failed'
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if data['status'] != 'passed':
        raise RuntimeError('Bundled-runtime Windows validation failed; see Windows report')
    print('Complete Windows distribution passed startup, 63 native geometry tests, and 3 full video analyses', flush=True)


if __name__ == '__main__':
    main()
