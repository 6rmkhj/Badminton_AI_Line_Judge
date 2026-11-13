"""Install plans stay inside one environment and inspection never installs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
setup = module_from_path('gpu_setup', ROOT / 'scripts/setup_gpu.py')
launcher = module_from_path('project_launcher', ROOT / 'start.py')


def test_cuda_plan_uses_one_environment_and_explicit_cuda_index(tmp_path):
    commands = setup.install_commands(tmp_path, 'cuda', onnx=True, system='Windows')
    python = str(tmp_path / '.venv/Scripts/python.exe')
    assert all(command[:3] == [python, '-m', 'pip'] for command in commands)
    torch = next(command for command in commands if any(value.startswith('torch==') for value in command))
    assert torch[torch.index('--index-url') + 1] == setup.CUDA_INDEX
    assert '--force-reinstall' in torch
    remove = next(i for i, command in enumerate(commands) if 'uninstall' in command)
    gpu = next(i for i, command in enumerate(commands) if setup.ONNX_GPU in command)
    assert remove < gpu
    assert commands[remove][-1] == 'onnxruntime'
    assert '--force-reinstall' in commands[gpu]
    assert '--editable' in commands[-2]
    assert commands[-1][-1] == 'check'
    assert not any('--user' in command or 'sudo' in command for command in commands)


def test_mps_plan_does_not_install_cuda_or_onnx_runtime(tmp_path):
    commands = setup.install_commands(tmp_path, 'mps', system='Darwin')
    assert all(command[0] == str(tmp_path / '.venv/bin/python') for command in commands)
    assert not any('--index-url' in command or 'uninstall' in command for command in commands)
    with pytest.raises(ValueError, match='CUDA'):
        setup.install_commands(tmp_path, 'mps', onnx=True)


def test_invalid_platform_stops_before_installation():
    with pytest.raises(ValueError, match='Windows/Linux'):
        setup.validate_platform('cuda', system='Darwin', machine='arm64')
    with pytest.raises(ValueError, match='Apple Silicon'):
        setup.validate_platform('mps', system='Darwin', machine='x86_64')
    with pytest.raises(ValueError, match='Apple Silicon'):
        setup.validate_platform('mps', system='Linux', machine='aarch64')
    setup.validate_platform('cuda', system='Windows', machine='AMD64')
    setup.validate_platform('mps', system='Darwin', machine='arm64')


def test_existing_shared_environment_is_rejected(tmp_path):
    environment = tmp_path / '.venv'
    environment.mkdir()
    with pytest.raises(ValueError, match='가상환경'):
        setup.validate_target(tmp_path)
    (environment / 'pyvenv.cfg').write_text('include-system-site-packages = true\n')
    with pytest.raises(ValueError, match='시스템 패키지'):
        setup.validate_target(tmp_path)
    (environment / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    setup.validate_target(tmp_path)


def test_external_environment_link_is_rejected(tmp_path):
    external = tmp_path / 'external'
    external.mkdir()
    project = tmp_path / 'project'
    project.mkdir()
    (project / '.venv').symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='링크'):
        setup.validate_target(project)


@pytest.mark.parametrize('option', ['--dry-run', '--check-only'])
def test_inspection_modes_never_create_or_install(tmp_path, monkeypatch, capsys, option):
    monkeypatch.setattr(setup, 'PROJECT', tmp_path)
    monkeypatch.setattr(setup, 'validate_platform', lambda backend: None)
    monkeypatch.setattr(setup.subprocess, 'run', lambda *a, **kw: pytest.fail('inspection must not run installer'))
    monkeypatch.setattr(setup.venv, 'EnvBuilder', lambda *a, **kw: pytest.fail('inspection must not create venv'))
    result = setup.main(['--backend', 'cuda', option])
    assert result == (0 if option == '--dry-run' else 1)
    assert not (tmp_path / '.venv').exists()
    output = capsys.readouterr()
    if option == '--dry-run':
        assert setup.CUDA_INDEX in output.out and 'torch==' in output.out
    else:
        assert '.venv가 없습니다' in output.err


def test_check_only_runs_target_gpu_smoke_not_installer(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, 'PROJECT', tmp_path)
    monkeypatch.setattr(setup, 'validate_platform', lambda backend: None)
    python = setup.environment_python(tmp_path)
    python.parent.mkdir(parents=True)
    python.write_text('')
    (tmp_path / '.venv/pyvenv.cfg').write_text('include-system-site-packages = false\n')
    monkeypatch.setattr(setup, 'verify_prefix', lambda project: None)
    calls = []
    monkeypatch.setattr(setup.subprocess, 'run', lambda command, **kw: calls.append(command) or type('Result', (), {'returncode': 1})())
    assert setup.main(['--backend', 'cuda', '--check-only']) == 1
    assert len(calls) == 1 and calls[0][0] == str(python)
    assert 'available_torch_devices' in calls[0][calls[0].index('-c') + 1]
    assert not any('pip' in command for command in calls)


def test_project_launcher_uses_prepared_env_without_symlink_recursion(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.sys, 'platform', 'linux')
    python = tmp_path / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setattr(launcher.sys, 'prefix', str(tmp_path / 'system-python'))
    assert launcher._project_python(tmp_path) is None
    (tmp_path / '.venv/pyvenv.cfg').write_text('include-system-site-packages = false\n')
    assert launcher._project_python(tmp_path) == python
    # A venv's executable commonly resolves to base Python. Prefix, not the
    # symlink target, determines whether re-execution already happened.
    monkeypatch.setattr(launcher.sys, 'prefix', str(tmp_path / '.venv'))
    assert launcher._project_python(tmp_path) is None


@pytest.mark.parametrize('prefix,base,version', [('/outside', '/base', [3, 12]),
                                              (None, None, [3, 12]),
                                              (None, '/base', [3, 10])])
def test_broken_target_interpreter_is_rejected_before_pip(tmp_path, monkeypatch, prefix, base, version):
    expected = str(tmp_path / '.venv')
    monkeypatch.setattr(setup.subprocess, 'check_output', lambda *a, **kw:
                        setup.json.dumps([prefix or expected, base or expected, version]))
    with pytest.raises(ValueError):
        setup.verify_prefix(tmp_path)
