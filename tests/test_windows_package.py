import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


spec = importlib.util.spec_from_file_location('portable_builder', Path(__file__).parents[1] / 'scripts/package_windows_portable.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def commit_directory(folder):
    subprocess.run(['git', 'init', '-q', str(folder)], check=True)
    subprocess.run(['git', '-C', str(folder), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(folder), '-c', 'user.name=Package Test',
                    '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'Fixture'], check=True)
    return subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()


@pytest.fixture
def package_inputs(tmp_path):
    project = tmp_path / 'source'
    for name in ('badminton/calibration.py', 'badminton/evidence.py', 'badminton/landing_recovery.py',
                 'badminton/event_sequence.py', 'badminton/assessment.py', 'configs/poc.json', 'start.py',
                 'setup-gpu-windows.bat', 'scripts/setup_gpu.py', 'scripts/package_windows_portable.py',
                 'docs/WINDOWS_PACKAGE.md', 'docs/current-validation.md'):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('current revision fixture\n', encoding='utf-8')
    (project / 'pyproject.toml').write_text('[project]\nversion = "0.5.8"\n', encoding='utf-8')
    source_commit = commit_directory(project)
    runtime = tmp_path / 'runtime'
    payload = runtime / 'payload'
    payload.mkdir(parents=True)
    entries = []
    for filename in (*builder.RUNTIME_FILES, builder.APPLICATION):
        path = payload / filename
        with zipfile.ZipFile(path, 'w') as archive:
            if filename == builder.APPLICATION:
                archive.writestr('PACKAGE_MANIFEST.json', json.dumps({'python': {'version': 'fixture'}}))
                archive.writestr('licenses/runtime.txt', 'Runtime fixture license')
            else:
                archive.writestr('runtime/' + filename + '.txt', filename)
        entries.append({'file': filename, 'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (runtime / 'PAYLOAD_CHECKSUMS.json').write_text(json.dumps(entries), encoding='utf-8')
    return project, runtime, source_commit


@pytest.mark.parametrize('checkout', [False, True])
def test_package_records_current_report_and_actual_runtime_provenance(package_inputs, tmp_path, checkout):
    project, runtime, source_commit = package_inputs
    runtime_commit = commit_directory(runtime) if checkout else None
    receipt = builder.build(runtime, tmp_path / 'output', tmp_path / 'release.zip', project,
                            validation_report='docs/current-validation.md')
    assert receipt['source_commit'] == source_commit
    assert receipt['source_worktree_dirty'] is False
    assert receipt['app_version'] == '0.5.8'
    with zipfile.ZipFile(tmp_path / 'output/payload/08-application.zip') as archive:
        manifest = json.loads(archive.read('PACKAGE_MANIFEST.json'))
        assert manifest['validation_report'] == 'app/docs/current-validation.md'
        assert manifest['runtime_base_commit'] == runtime_commit
        assert manifest['tests_run_by_package_builder'] is False
        validation = archive.read('PACKAGE_VALIDATION.txt').decode('utf-8-sig')
        assert 'app/docs/current-validation.md' in validation
        assert '986' not in validation
        assert archive.read('app/docs/current-validation.md') == (project / 'docs/current-validation.md').read_bytes()
        assert archive.read('licenses/runtime.txt') == b'Runtime fixture license'
        checksums = json.loads(archive.read('APP_FILE_CHECKSUMS.json'))
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == digest for name, digest in checksums.items())
    for name in builder.RUNTIME_FILES:
        assert (runtime / 'payload' / name).read_bytes() == (tmp_path / 'output/payload' / name).read_bytes()


@pytest.mark.parametrize('report', ['docs/missing.md', '../outside.md', '/absolute.md', 'start.py'])
def test_package_rejects_report_outside_included_docs(package_inputs, tmp_path, report):
    project, runtime, _ = package_inputs
    with pytest.raises(ValueError):
        builder.build(runtime, tmp_path / 'output', tmp_path / 'release.zip', project, validation_report=report)
    assert not (tmp_path / 'output').exists()
    assert not (tmp_path / 'release.zip').exists()


@pytest.fixture
def windows_powershell():
    if os.name != 'nt':
        pytest.skip('Requires Windows PowerShell 5.1 and cmd.exe')
    executable = shutil.which('powershell.exe')
    assert executable, 'Windows PowerShell is required for the launcher regression test'
    version = subprocess.run(
        [executable, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
         '$PSVersionTable.PSVersion.ToString()'],
        check=True, capture_output=True, timeout=30,
    ).stdout.decode('utf-8', errors='replace').strip()
    assert version.startswith('5.1.'), f'Expected Windows PowerShell 5.1, got {version}'
    return executable


@pytest.fixture
def windows_payloads(tmp_path, windows_powershell):
    # Use the generated launcher in a user-like extraction path, including
    # non-ASCII characters and spaces. Each real ZIP has a distinct file.
    folder = tmp_path / '한글 Windows package'
    payload = folder / 'payload'
    payload.mkdir(parents=True)
    entries = []
    for filename in (*builder.RUNTIME_FILES, builder.APPLICATION):
        path = payload / filename
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('extracted/' + filename + '.txt', filename)
        entries.append({'file': filename, 'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (folder / 'PAYLOAD_CHECKSUMS.json').write_text(json.dumps(entries), encoding='utf-8')
    (folder / 'START.bat').write_bytes(builder.start_batch('9.9.9'))
    environment = dict(os.environ)
    environment['PATH'] = str(Path(windows_powershell).parent) + os.pathsep + environment.get('PATH', '')
    return folder, entries, environment


def run_windows_prepare(folder, environment):
    # Supplying a line lets the failure-path PAUSE finish without a terminal.
    # Launch from cmd.exe rather than extracting and testing only the PS snippet.
    result = subprocess.run(
        [os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c', 'START.bat', '--prepare-only'],
        cwd=folder, env=environment, input=b'\r\n', stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=90,
    )
    return result.returncode, result.stdout.decode('utf-8', errors='replace')


def test_windows_powershell_51_reproduces_nested_json_array(windows_payloads, windows_powershell):
    folder, _, _ = windows_payloads
    script = (
        "$ErrorActionPreference='Stop'; "
        "$json=Get-Content -LiteralPath 'PAYLOAD_CHECKSUMS.json' -Raw -Encoding UTF8; "
        "$legacy=@($json | ConvertFrom-Json); "
        "$direct=$json | ConvertFrom-Json; "
        "[pscustomobject]@{legacy=$legacy.Count; direct=$direct.Count} | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [windows_powershell, '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', script],
        cwd=folder, check=True, capture_output=True, timeout=30,
    )
    counts = json.loads(result.stdout.decode('utf-8', errors='replace'))
    assert counts == {'legacy': 1, 'direct': 9}


def test_windows_start_prepares_all_payloads_and_reuses_verified_marker(windows_payloads):
    folder, entries, environment = windows_payloads
    code, output = run_windows_prepare(folder, environment)
    assert code == 0, output
    marker = folder / '.package-ready-9.9.9'
    manifest_hash = hashlib.sha256((folder / 'PAYLOAD_CHECKSUMS.json').read_bytes()).hexdigest()
    assert marker.read_text(encoding='ascii').strip().lower() == manifest_hash
    for entry in entries:
        extracted = folder / 'extracted' / (entry['file'] + '.txt')
        assert extracted.read_text() == entry['file']
    # A matching marker preserves extracted application data on subsequent runs.
    preserved = folder / 'extracted' / (entries[0]['file'] + '.txt')
    preserved.write_text('preserve existing extraction', encoding='utf-8')
    code, output = run_windows_prepare(folder, environment)
    assert code == 0, output
    assert preserved.read_text(encoding='utf-8') == 'preserve existing extraction'


@pytest.mark.parametrize('damage, message', [
    ('short_manifest', 'Expected nine payload archives'),
    ('missing', 'Missing'),
    ('size', 'Size mismatch'),
    ('hash', 'SHA256 mismatch'),
])
def test_windows_start_rejects_invalid_payloads_before_extraction(windows_payloads, damage, message):
    folder, entries, environment = windows_payloads
    path = folder / 'payload' / entries[-1]['file']
    if damage == 'short_manifest':
        (folder / 'PAYLOAD_CHECKSUMS.json').write_text(json.dumps(entries[:-1]), encoding='utf-8')
    elif damage == 'missing':
        path.unlink()
    elif damage == 'size':
        path.write_bytes(path.read_bytes()[:-1])
    else:
        content = bytearray(path.read_bytes())
        content[-1] ^= 1  # Preserve the size so the SHA256 gate must reject it.
        path.write_bytes(content)
    code, output = run_windows_prepare(folder, environment)
    assert code != 0, output
    assert message in output, output
    assert not (folder / '.package-ready-9.9.9').exists()
    assert not (folder / 'extracted').exists()


def test_windows_start_rechecks_hash_even_with_current_marker(windows_payloads):
    folder, entries, environment = windows_payloads
    code, output = run_windows_prepare(folder, environment)
    assert code == 0, output
    path = folder / 'payload' / entries[-1]['file']
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    code, output = run_windows_prepare(folder, environment)
    assert code != 0, output
    assert 'SHA256 mismatch' in output, output
