import hashlib
import importlib.util
import json
from pathlib import Path
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
