import importlib.util
from pathlib import Path
import stat
import zipfile

import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script('package_windows_portable')
pytorch = load_script('portable_pytorch')


def wheel(folder, name, version='1.0', *, tag='py3-none-any', requires=(), files=None):
    folder.mkdir(exist_ok=True)
    stem = name.replace('-', '_') + '-' + version
    path = folder / (stem + '-' + tag + '.whl')
    metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Python: >=3.10\n'
    metadata += ''.join(f'Requires-Dist: {requirement}\n' for requirement in requires)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(stem + '.dist-info/METADATA', metadata)
        archive.writestr(stem + '.dist-info/WHEEL', 'Wheel-Version: 1.0\nTag: ' + tag + '\n')
        archive.writestr(stem + '.dist-info/licenses/LICENSE', 'Test license')
        for member, contents in (files or {name + '/__init__.py': '# fixture'}).items():
            archive.writestr(member, contents)
    return path


def torch_wheel(folder, **kwargs):
    return wheel(folder, 'torch', pytorch.TORCH_VERSION, tag='cp312-cp312-win_amd64', **kwargs)


def prepare(folder, destination, existing=None, bundled=None):
    return pytorch.prepare_payload(folder, destination, existing if existing is not None else set(),
                                   bundled_versions=bundled or {}, safe_name=builder.safe_name,
                                   inspect_zip=builder.inspect_zip)


def test_cpu_payload_validates_target_dependencies_and_records_licenses(tmp_path):
    folder = tmp_path / 'wheels'
    torch_wheel(folder, requires=['filelock>=3', 'sympy>=1', 'setuptools; python_version >= "3.12"',
                                 'linux-only; sys_platform == "linux"', 'optional; extra == "optree"'])
    wheel(folder, 'sympy', requires=['mpmath>=1'])
    wheel(folder, 'mpmath')
    wheel(folder, 'setuptools')
    wheel(folder, 'filelock', version='9.0')  # Supplied wheel is unnecessary: the base already provides it.
    existing = {'runtime/lib/site-packages/filelock/__init__.py'}
    destination = tmp_path / pytorch.PAYLOAD_NAME
    result = prepare(folder, destination, existing, {'filelock': '4.2.4'})
    assert result['torch_version'] == '2.10.0+cpu'
    assert result['reused_dependencies'] == {'filelock': '4.2.4'}
    assert {entry['name'] for entry in result['wheels']} == {'torch', 'sympy', 'mpmath', 'setuptools'}
    with zipfile.ZipFile(destination) as archive:
        assert 'runtime/Lib/site-packages/torch/__init__.py' in archive.namelist()
        assert not any('filelock' in name for name in archive.namelist())
        for entry in result['wheels']:
            assert len(entry['sha256']) == 64
            assert all(archive.read(name) == b'Test license' for name in entry['licenses'])
    assert 'runtime/lib/site-packages/torch/__init__.py' in existing


@pytest.mark.parametrize('requirement,bundled,error', [
    ('sympy>=1', {}, 'Missing PyTorch wheel dependency'),
    ('filelock>=5', {'filelock': '4.2.4'}, 'Incompatible dependency'),
    ('setuptools; python_version >= "3.12"', {}, 'Missing PyTorch wheel dependency'),
    ('windows-only; sys_platform == "win32"', {}, 'Missing PyTorch wheel dependency'),
])
def test_missing_or_incompatible_windows_dependencies_fail_before_output(tmp_path, requirement, bundled, error):
    folder = tmp_path / 'wheels'
    torch_wheel(folder, requires=[requirement])
    destination = tmp_path / 'payload.zip'
    with pytest.raises(ValueError, match=error):
        prepare(folder, destination, bundled=bundled)
    assert not destination.exists()


@pytest.mark.parametrize('version,tag', [
    ('2.10.0', 'cp312-cp312-win_amd64'),
    ('2.10.0+cu128', 'cp312-cp312-win_amd64'),
    ('2.10.0+cpu', 'cp311-cp311-win_amd64'),
    ('2.10.0+cpu', 'cp312-cp312-manylinux_2_28_x86_64'),
    ('2.10.0+cpu', 'cp312-cp312-win_arm64'),
])
def test_cpu_version_and_windows_interpreter_are_required(tmp_path, version, tag):
    folder = tmp_path / 'wheels'
    wheel(folder, 'torch', version, tag=tag)
    with pytest.raises(ValueError):
        prepare(folder, tmp_path / 'payload.zip')


@pytest.mark.parametrize('files,existing', [
    ({'torch/THING.py': '1', 'torch/thing.py': '2'}, set()),
    ({'torch/existing.py': '1'}, {'runtime/lib/site-packages/torch/existing.py'}),
    ({'torch': 'file', 'torch/child.py': 'child'}, set()),
    ({'torch': 'file'}, {'runtime/lib/site-packages/torch/existing.py'}),
    ({'torch/child.py': 'file'}, {'runtime/lib/site-packages/torch'}),
    ({'../escape.py': 'bad'}, set()),
    ({'torch/CON.txt': 'bad'}, set()),
    ({'torch/file:stream': 'bad'}, set()),
])
def test_unsafe_or_colliding_archive_paths_leave_existing_payloads_unchanged(tmp_path, files, existing):
    folder = tmp_path / 'wheels'
    torch_wheel(folder, files=files)
    original = set(existing)
    destination = tmp_path / 'payload.zip'
    with pytest.raises(ValueError):
        prepare(folder, destination, existing)
    assert existing == original
    assert not destination.exists()


def test_wheel_symlink_members_are_rejected(tmp_path):
    folder = tmp_path / 'wheels'
    path = torch_wheel(folder)
    with zipfile.ZipFile(path, 'a') as archive:
        entry = zipfile.ZipInfo('torch/symlink')
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(entry, '../../outside')
    with pytest.raises(ValueError, match='symlink'):
        prepare(folder, tmp_path / 'payload.zip')


def test_unrelated_or_duplicate_distributions_are_rejected(tmp_path):
    folder = tmp_path / 'wheels'
    torch_wheel(folder)
    wheel(folder, 'unrelated')
    with pytest.raises(ValueError, match='Unrelated wheels'):
        prepare(folder, tmp_path / 'payload.zip')
    wheel(folder, 'unrelated', version='2.0')
    with pytest.raises(ValueError, match='Duplicate wheel distribution'):
        prepare(folder, tmp_path / 'payload.zip')


def test_wheel_data_is_relocated_before_collision_detection(tmp_path):
    folder = tmp_path / 'wheels'
    torch_wheel(folder, files={'torch-2.10.0+cpu.data/platlib/torch/model.py': 'model'})
    with pytest.raises(ValueError, match='collision'):
        prepare(folder, tmp_path / 'collision.zip', {'runtime/lib/site-packages/torch/model.py'})
    prepare(folder, tmp_path / 'payload.zip')
    with zipfile.ZipFile(tmp_path / 'payload.zip') as archive:
        assert archive.read('runtime/Lib/site-packages/torch/model.py') == b'model'


def test_existing_output_is_not_overwritten(tmp_path):
    folder = tmp_path / 'wheels'
    torch_wheel(folder)
    destination = tmp_path / 'payload.zip'
    destination.write_bytes(b'preserved')
    with pytest.raises(ValueError, match='already exists'):
        prepare(folder, destination)
    assert destination.read_bytes() == b'preserved'
