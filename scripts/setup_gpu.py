"""Prepare this project's isolated GPU runtime; no model weights are downloaded."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import venv

PROJECT = Path(__file__).resolve().parents[1]
TORCH_VERSION = "2.10.0"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
ONNX_GPU = "onnxruntime-gpu>=1.21,<2"

# The device inventory performs actual tensor operations before marking a device
# selectable. ONNX provider availability is an additional dependency check; each
# configured model is checked again by the strict CUDA session on analysis.
CHECK_CODE = """
import importlib.metadata
import json
import sys
from badminton.inference import available_torch_devices
backend, check_onnx = sys.argv[1], sys.argv[2] == '1'
devices = available_torch_devices()
selected = [item for item in devices if item.get('selectable') and
            (item['id'].startswith('cuda:') if backend == 'cuda' else item['id'] == 'mps')]
report = {'backend': backend, 'devices': devices, 'torch': importlib.metadata.version('torch')}
if check_onnx:
    import torch
    import onnxruntime as ort
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    report['onnxruntime'] = ort.__version__
    report['onnx_providers'] = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in report['onnx_providers']:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print('ONNX CUDA provider를 사용할 수 없습니다.', file=sys.stderr)
        raise SystemExit(1)
print(json.dumps(report, ensure_ascii=False, indent=2))
if not selected:
    print('선택한 백엔드에서 실제 GPU 연산을 확인하지 못했습니다. 드라이버와 GPU 지원 PyTorch 설치를 확인하세요.', file=sys.stderr)
    raise SystemExit(1)
"""


def environment_python(project, system=None):
    system = system or platform.system()
    return Path(project) / ".venv" / ("Scripts/python.exe" if system == "Windows" else "bin/python")


def install_commands(project, backend, onnx=False, system=None):
    """Return explicit argument arrays, never shell command strings."""
    python = str(environment_python(project, system))
    pip = [python, "-m", "pip"]
    commands = [pip + ["install", "--upgrade", "pip"]]
    torch = pip + ["install", "--upgrade", "--force-reinstall", f"torch=={TORCH_VERSION}"]
    if backend == "cuda":
        torch += ["--index-url", CUDA_INDEX]
    elif backend != "mps":
        raise ValueError("backend must be cuda or mps")
    commands.append(torch)
    if onnx:
        if backend != "cuda":
            raise ValueError("ONNX GPU 설치는 CUDA에서만 지원합니다.")
        # Both packages export the same Python namespace; reinstall the GPU
        # distribution after removing any overlapping CPU distribution.
        commands += [pip + ["uninstall", "--yes", "onnxruntime"],
                     pip + ["install", "--upgrade", "--force-reinstall", ONNX_GPU]]
    # Re-resolve the application's NumPy/OpenCV/SciPy constraints after optional
    # runtime installers, which may otherwise upgrade NumPy independently.
    commands += [pip + ["install", "--editable", str(Path(project))], pip + ["check"]]
    return commands


def validate_target(project):
    """Keep package replacement inside the project's own isolated environment."""
    environment = Path(project) / ".venv"
    if environment.is_symlink():
        raise ValueError(".venv가 다른 위치를 가리키는 링크입니다. 프로젝트 내부의 독립 가상환경을 사용하세요.")
    config = environment / "pyvenv.cfg"
    if environment.exists() and not config.is_file():
        raise ValueError("기존 .venv가 올바른 가상환경이 아닙니다. 해당 폴더를 확인한 뒤 다시 실행하세요.")
    if config.is_file():
        values = dict(line.lower().strip().split("=", 1) for line in config.read_text().splitlines() if "=" in line)
        if any(key.strip() == "include-system-site-packages" and value.strip() == "true"
               for key, value in values.items()):
            raise ValueError("시스템 패키지를 공유하는 .venv에서는 GPU 패키지를 교체하지 않습니다. 독립 가상환경을 사용하세요.")


def validate_platform(backend, system=None, machine=None):
    system, machine = system or platform.system(), machine or platform.machine()
    if backend == "cuda" and system not in {"Windows", "Linux"}:
        raise ValueError("CUDA 설치는 NVIDIA GPU가 있는 Windows/Linux에서 실행하세요.")
    if backend == "mps" and (system != "Darwin" or machine.lower() not in {"arm64", "aarch64"}):
        raise ValueError("이 MPS 설치 경로는 Apple Silicon macOS용입니다.")


def check_command(project, backend, onnx=False):
    return [str(environment_python(project)), "-B", "-X", "utf8", "-c", CHECK_CODE,
            backend, "1" if onnx else "0"]


def verify_prefix(project):
    """Reject a broken environment executable before running pip uninstall."""
    command = [str(environment_python(project)), "-B", "-c",
               "import json,sys; print(json.dumps([sys.prefix,sys.base_prefix,list(sys.version_info[:2])]))"]
    prefix, base, version = json.loads(subprocess.check_output(command, text=True))
    expected = (Path(project) / ".venv").resolve()
    if Path(prefix).resolve() != expected or Path(prefix).resolve() == Path(base).resolve():
        raise ValueError(".venv Python이 프로젝트 가상환경으로 실행되지 않습니다. 환경 경로를 확인하세요.")
    if tuple(version) < (3, 11):
        raise ValueError("기존 .venv의 Python이 3.11보다 낮습니다. 지원되는 Python 가상환경을 준비하세요.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "mps"],
                        help="cuda: NVIDIA Windows/Linux, mps: Apple Silicon macOS")
    parser.add_argument("--onnx", action="store_true", help="CUDA용 ONNX Runtime도 설치/확인")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="실행할 명령만 표시; 환경을 변경하지 않음")
    action.add_argument("--check-only", action="store_true", help="기존 프로젝트 환경의 GPU 연산만 확인")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 이상이 필요합니다.")
    try:
        commands = install_commands(PROJECT, args.backend, args.onnx)
        validate_platform(args.backend)
        validate_target(PROJECT)
        python = environment_python(PROJECT)
        check = check_command(PROJECT, args.backend, args.onnx)
        if args.dry_run:
            if not python.is_file():
                create = [sys.executable, "-m", "venv", str(PROJECT / ".venv")]
                print(subprocess.list2cmdline(create) if platform.system() == "Windows" else shlex.join(create))
            for command in commands:
                print(subprocess.list2cmdline(command) if platform.system() == "Windows" else shlex.join(command))
            print("GPU 연산 확인:")
            print(subprocess.list2cmdline(check) if platform.system() == "Windows" else shlex.join(check))
            return 0
        if args.check_only and not python.is_file():
            raise ValueError("프로젝트 .venv가 없습니다. 먼저 --check-only 없이 GPU 환경을 설치하세요.")
        if not args.check_only:
            if not python.is_file():
                venv.EnvBuilder(with_pip=True).create(PROJECT / ".venv")
            verify_prefix(PROJECT)
            for command in commands:
                subprocess.run(command, cwd=PROJECT, check=True)
        else:
            verify_prefix(PROJECT)
        result = subprocess.run(check, cwd=PROJECT, check=False)
        if result.returncode:
            print("GPU 준비 확인 실패. 서버에서 GPU를 선택하기 전에 위 원인을 해결하세요.", file=sys.stderr)
            return result.returncode
        print("GPU 연산 확인 완료. python start.py --pair 실행 후 장치 설정에서 GPU를 선택하세요.")
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"GPU 환경 준비 실패: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
