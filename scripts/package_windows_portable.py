"""Build a Windows x64 package from checked runtime payloads and current sources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import tomllib
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    '01-python.zip', '02-filelock.zip', '03-numpy.zip',
    '04-opencv_python_headless.zip', '05-scipy.zip',
    '06-ffmpeg.zip', '07-ffprobe.zip', '09-runtime-notes.zip',
)
APPLICATION = '08-application.zip'
PYTORCH_PAYLOAD = '10-pytorch_cpu.zip'
MSVCP_SOURCE = 'runtime/Lib/site-packages/numpy.libs/msvcp140-263139962577ecda4cd9469ca360a746.dll'
MSVCP_SHA256 = 'a4c2229bdc2a2a630acdc095b4d86008e5c3e3bc7773174354f3da4f5beb9cde'
SOURCE_FOLDERS = {'badminton', 'configs', 'docs', 'examples', 'scripts', 'tests', 'annotations'}
SOURCE_ROOTS = {'README.md', 'pyproject.toml', 'start.py'}
DOWNLOAD_URL = ('https://github.com/6rmkhj/Badminton_AI_Line_Judge/'
                'releases/download/v{version}/badminton-windows-{version}.zip')


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def add_pytorch_msvc(runtime_base, destination, existing_files):
    """Reuse the base's Microsoft runtime under the name PyTorch imports.

    NumPy ships this redistributable with a hashed filename; torch_cpu.dll
    imports MSVCP140.dll literally. Keep its bytes and original attribution.
    """
    with zipfile.ZipFile(runtime_base / 'payload/03-numpy.zip') as archive:
        try:
            content = archive.read(MSVCP_SOURCE)
        except KeyError as error:
            raise ValueError('Base NumPy payload is missing the checked MSVCP140 runtime') from error
    if hashlib.sha256(content).hexdigest() != MSVCP_SHA256:
        raise ValueError('Base MSVCP140 runtime hash mismatch')
    target = 'runtime/Lib/site-packages/torch/lib/msvcp140.dll'
    if target.casefold() in existing_files:
        raise ValueError('PyTorch MSVCP140 runtime already exists')
    with zipfile.ZipFile(destination, 'a', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(target, content)
    existing_files.add(target.casefold())
    return {'file': target, 'sha256': MSVCP_SHA256, 'bytes': len(content),
            'source_payload': '03-numpy.zip', 'source_entry': MSVCP_SOURCE,
            'version': '14.40.33810.0', 'copyright': 'Microsoft Corporation. All rights reserved.',
            'redistribution_conditions': 'licenses/Python-LICENSE.txt: Additional Conditions for this Windows binary build'}


def reject_symlinks(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f'Symlink input/output is not accepted: {part}')


def safe_name(name):
    """Use names that extract identically on Windows and POSIX."""
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        raise ValueError(f'Unsafe archive path: {name!r}')
    path = PurePosixPath(name)
    if path.is_absolute() or any(piece in ('', '.', '..') for piece in name.rstrip('/').split('/')):
        raise ValueError(f'Unsafe archive path: {name!r}')
    for piece in path.parts:
        if ':' in piece or piece.endswith((' ', '.')):
            raise ValueError(f'Unsafe Windows archive path: {name!r}')
        if re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', piece):
            raise ValueError(f'Reserved Windows archive path: {name!r}')
    return path.as_posix().rstrip('/')


def inspect_zip(path, global_files=None):
    local = set()
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            name = safe_name(entry.filename)
            key = name.casefold()
            if key in local:
                raise ValueError(f'Duplicate/case-colliding ZIP entry: {path.name}: {name}')
            local.add(key)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f'ZIP symlink is not accepted: {path.name}: {name}')
            if global_files is not None and not entry.is_dir():
                if key in global_files:
                    raise ValueError(f'File collision across payloads: {name}')
                global_files.add(key)
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f'CRC mismatch in {path.name}: {bad}')
    return local


def source_files(project):
    result = subprocess.run(['git', '-C', str(project), 'ls-files', '-z'],
                            check=True, capture_output=True).stdout
    names = {name.decode('utf-8') for name in result.split(b'\0') if name}
    # Include the builder and its usage guide even during a first, uncommitted
    # packaging dry run. All other inputs must already be Git-tracked.
    names.update({'scripts/package_windows_portable.py', 'docs/WINDOWS_PACKAGE.md'})
    if (project / 'scripts/portable_pytorch.py').is_file():
        names.add('scripts/portable_pytorch.py')
    files, seen = {}, set()
    for name in sorted(names):
        relative = PurePosixPath(name)
        selected = (relative.parts[0] in SOURCE_FOLDERS
                    or (len(relative.parts) == 1 and
                        (name in SOURCE_ROOTS or relative.suffix.lower() in {'.bat', '.pdf'})))
        if not selected:
            continue
        safe_name(name)
        if relative.parts[0] in {'video', 'runs', 'models', '.git', '.venv'}:
            raise ValueError(f'Excluded source path selected: {name}')
        if any(part in {'__pycache__', '.pytest_cache', '.venv', '.git'} for part in relative.parts):
            continue
        path = project / name
        reject_symlinks(path)
        if not path.is_file():
            raise ValueError(f'Tracked package source is missing: {name}')
        key = name.casefold()
        if key in seen:
            raise ValueError(f'Case-colliding source file: {name}')
        seen.add(key)
        data = path.read_bytes()
        # cmd.exe entry points always use CRLF, including copied source scripts.
        if relative.suffix.lower() == '.bat':
            data = data.replace(b'\r\n', b'\n').replace(b'\r', b'\n').replace(b'\n', b'\r\n')
        files['app/' + name] = data
    for required in ('badminton/calibration.py', 'badminton/evidence.py',
                     'badminton/landing_recovery.py', 'badminton/event_sequence.py',
                     'badminton/assessment.py', 'configs/poc.json', 'start.py',
                     'setup-gpu-windows.bat', 'scripts/setup_gpu.py'):
        if 'app/' + required not in files:
            raise ValueError(f'Required latest application file missing: {required}')
    return files


def portable_start(pytorch=False):
    source = '''"""Launch the Windows A/B workspace using its prepared GPU or CPU runtime."""
from pathlib import Path
import importlib
import os
import runpy
import subprocess
import sys
import traceback

root = Path(__file__).resolve().parent
app = root / 'app'
try:
    if sys.platform != 'win32' or sys.maxsize <= 2**32:
        raise RuntimeError('Windows 64비트 전용 패키지입니다.')
    environment = app / '.venv'
    prepared_python = environment / 'Scripts' / 'python.exe'
    if prepared_python.is_file() and Path(sys.prefix).resolve() != environment.resolve():
        if not (environment / 'pyvenv.cfg').is_file():
            raise RuntimeError('app/.venv가 올바른 가상환경이 아닙니다. GPU 설치 상태를 확인하세요.')
        os.execv(str(prepared_python), [str(prepared_python), '-X', 'utf8',
                                     str(Path(__file__).resolve()), *sys.argv[1:]])
    os.environ['PATH'] = str(root / 'tools') + os.pathsep + str(root / 'runtime') + os.pathsep + os.environ.get('PATH', '')
    print('배드민턴 A/B 영상 분석을 준비합니다.', flush=True)
    for module in ('numpy', 'cv2', 'scipy', 'filelock'):
        importlib.import_module(module)
    # PYTORCH_CHECK
    options = {'capture_output': True, 'text': True, 'encoding': 'utf-8', 'errors': 'replace', 'timeout': 30, 'check': True}
    encoders = subprocess.run([str(root / 'tools/ffmpeg.exe'), '-hide_banner', '-encoders'], **options)
    if 'libx264' not in encoders.stdout:
        raise RuntimeError('미리보기 생성에 필요한 H.264 인코더가 없습니다.')
    subprocess.run([str(root / 'tools/ffprobe.exe'), '-version'], **options)
    sys.path.insert(0, str(app))
    os.chdir(app)
    sys.argv = [str(app / 'start.py'), '--pair', '--port', '0', *sys.argv[1:]]
    print('브라우저에서 장치 설정을 확인한 뒤 A/B 영상을 올리세요. 종료하려면 Ctrl+C를 누르세요.', flush=True)
    runpy.run_path(str(app / 'start.py'), run_name='__main__')
except KeyboardInterrupt:
    pass
except Exception:
    error = traceback.format_exc()
    (root / 'startup-error.log').write_text(error, encoding='utf-8')
    print('실행하지 못했습니다. 같은 폴더의 startup-error.log를 확인해 주세요.', file=sys.stderr)
    print(error, file=sys.stderr)
    raise SystemExit(1)
'''
    check = '''import torch
    with torch.inference_mode():
        value = torch.ones(2, device='cpu').sum().item()
        if value != 2:
            raise RuntimeError('PyTorch CPU 연산 확인에 실패했습니다.')
    print(f'PyTorch {torch.__version__} CPU 연산 확인 완료.', flush=True)'''
    return source.replace('# PYTORCH_CHECK', check if pytorch else '').encode('utf-8')


def batch_bytes(text):
    return text.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')


def start_batch(version, pytorch=False):
    # Validate every payload before first extraction or launch. The marker binds
    # extraction to this exact payload manifest, not just a loose version file.
    names = sorted((*RUNTIME_FILES, APPLICATION, *((PYTORCH_PAYLOAD,) if pytorch else ())))
    expected_names = ','.join("'" + name + "'" for name in names)
    count_message = 'Expected ten payload archives' if pytorch else 'Expected nine payload archives'
    prepare = (
        "$ErrorActionPreference='Stop'; "
        "$manifest='PAYLOAD_CHECKSUMS.json'; "
        # Windows PowerShell 5.1 emits a JSON array as one pipeline object.
        # Assign it first so @() cannot wrap that output in a second array.
        "$items=Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json; "
        "$items=@($items); "
        f"if($items.Count -ne {len(names)}){{throw '{count_message}'}}; "
        f"$expected=@({expected_names}); "
        "$seen=@{}; foreach($item in $items){ "
        "if($item.file -cnotin $expected -or $seen.ContainsKey($item.file)){throw 'Invalid payload name'}; "
        "$seen[$item.file]=$true; $path=Join-Path 'payload' $item.file; "
        "if(!(Test-Path -LiteralPath $path -PathType Leaf)){throw ('Missing '+$path)}; "
        "if((Get-Item -LiteralPath $path).Length -ne $item.bytes){throw ('Size mismatch: '+$path)}; "
        "if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $item.sha256){throw ('SHA256 mismatch: '+$path)} }; "
        f"$marker='.package-ready-{version}'; "
        "$digest=(Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash; "
        "if(!(Test-Path -LiteralPath $marker) -or (Get-Content -LiteralPath $marker -Raw).Trim() -ne $digest){ "
        "$i=0; foreach($item in $items){$i++; Write-Host ('[{0}/{1}] {2}' -f $i,$items.Count,$item.file); "
        "Expand-Archive -LiteralPath (Join-Path 'payload' $item.file) -DestinationPath . -Force }; "
        "Set-Content -LiteralPath $marker -Value $digest -Encoding ASCII }"
    )
    return batch_bytes(f'''@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo 배드민턴 AI {version} 구성 파일을 확인합니다.
rem Let Windows PowerShell rebuild its own module paths when called from pwsh.
set "PSModulePath="
powershell.exe -NoLogo -NoProfile -Command "{prepare}"
if errorlevel 1 goto failed
if /I "%~1"=="--prepare-only" exit /b 0
call "%~dp0RUN.bat" %*
exit /b %ERRORLEVEL%
:failed
echo 준비하지 못했습니다. ZIP을 새 폴더에 전부 압축 해제한 뒤 다시 실행하세요.
pause
exit /b 1
''')


def run_batch():
    return batch_bytes(r'''@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0portable_start.py" goto prepare
if not exist "%~dp0runtime\python.exe" goto prepare
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
"%~dp0runtime\python.exe" -X utf8 "%~dp0portable_start.py" %*
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" pause
exit /b %APP_EXIT%
:prepare
call "%~dp0START.bat" %*
exit /b %ERRORLEVEL%
''')


def gpu_batch():
    return batch_bytes(r'''@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
call "%~dp0START.bat" --prepare-only
if errorlevel 1 exit /b 1
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PATH=%~dp0tools;%~dp0runtime;%PATH%"
cd /d "%~dp0app"
"%~dp0runtime\python.exe" -X utf8 "%~dp0app\scripts\setup_gpu.py" --backend cuda %*
set "GPU_SETUP_EXIT=%ERRORLEVEL%"
if "%GPU_SETUP_EXIT%"=="0" (
  echo GPU 준비가 완료됐습니다. START.bat를 실행하고 장치 설정에서 GPU를 선택하세요.
) else (
  echo GPU 준비에 실패했습니다. 위의 오류와 NVIDIA 드라이버를 확인하세요.
)
pause
exit /b %GPU_SETUP_EXIT%
''')


def usage_text(version, validation_report='docs/WINDOWS_PACKAGE.md', pytorch=False):
    variant = ('PyTorch CPU 포함판: PyTorch 2.10.0+cpu와 필요한 의존성을 포함합니다.\n'
               'PyTorch CPU 추론·학습에 추가 다운로드가 필요하지 않습니다. CUDA는 별도 GPU 설치가 필요합니다.\n'
               if pytorch else '기본판: PyTorch는 포함하지 않습니다.\n')
    return f'''배드민턴 AI Windows 실행 패키지 {version}

{variant}

Windows 64비트용입니다. 다운로드 ZIP을 새 폴더에 전부 압축 해제한 뒤 START.bat를 실행하세요.
첫 실행에서는 구성 파일의 SHA256을 확인하고 포함된 실행 환경을 압축 해제합니다.
CPU 기본 분석은 별도 Python 설치나 패키지 다운로드 없이 실행할 수 있습니다.
기존 패키지 폴더와 섞지 말고 쓰기 가능한 새 폴더를 사용하세요.

1. START.bat 실행 후 브라우저의 장치 설정에서 CPU/GPU·처리 스레드·프레임 버퍼 예산을 확인합니다.
2. A와 B의 원본 영상 파일을 각각 선택하면 FPS·해상도·확장자·코덱 등 정보를 확인할 수 있습니다.
3. A/B 연결 분석을 누릅니다. 원본 경기 영상과 과거 대용량 실행 결과는 패키지에 포함하지 않았습니다.
4. 종료하려면 실행 콘솔에서 Ctrl+C를 누르세요. 결과와 설정은 app/runs/에 저장됩니다.

NVIDIA GPU를 쓰려면 서버를 종료한 뒤 같은 루트 폴더의 SETUP_GPU.bat를 실행하세요.
이 설치는 인터넷과 GPU를 지원하는 NVIDIA 드라이버가 필요합니다.
번들 Python으로 app/.venv를 만들고 CUDA용 PyTorch 2.10.0을 설치한 뒤 실제 GPU 연산을 확인합니다.
설치 후 START.bat를 다시 실행하면 app/.venv를 우선 사용합니다.
ONNX CUDA 모델이 필요한 경우 명령 프롬프트에서 SETUP_GPU.bat --onnx를 실행하세요.
GPU 설치 후 패키지 폴더를 옮기면 .venv의 경로가 달라질 수 있으므로 새 위치에서 GPU 환경을 다시 준비하세요.
외부 모델 가중치는 포함하지 않습니다. AMD/Intel GPU용 설치 경로는 제공하지 않습니다.
GPU 메모리 오류는 자동 CPU 전환 없이 사유를 표시합니다. 설정에서 CPU로 바꿔 다시 실행할 수 있습니다.

분석 결과에서 코트 직선 후보 탐지와 전체 코트 좌표 보정 상태를 따로 확인할 수 있습니다.
이전 검수 결과를 재사용하면 새 탐지를 실행하지 않습니다. 개선된 탐지를 확인하려면 검수 재사용을 끄고 새 분석을 실행하세요.
후보 위치·실제 접촉·IN/OUT 판정은 구분합니다. 라인이 보여도 전체 코트나 최초 접촉이 확인되지 않으면 REVIEW가 남을 수 있습니다.
이 소스의 변경 내용과 검증 범위는 app/{validation_report} 및 app/docs/WINDOWS_PACKAGE.md를 확인하세요.

문제가 생기면 startup-error.log와 실행 콘솔 오류를 확인하세요.
이 패키지의 Windows 실행 파일 자체와 GPU 실기는 Linux 빌드 환경에서 직접 실행 검증하지 못했습니다.
런타임 출처·파일별 해시는 PACKAGE_MANIFEST.json, APP_FILE_CHECKSUMS.json, PAYLOAD_CHECKSUMS.json에 있습니다.
'''.encode('utf-8-sig')


def build(runtime_base, output_dir, archive_path, project=PROJECT, *, validation_report='docs/WINDOWS_PACKAGE.md',
          pytorch_wheels=None):
    runtime_base, output_dir, archive_path = (Path(p).absolute() for p in (runtime_base, output_dir, archive_path))
    for path in (runtime_base, output_dir, archive_path):
        reject_symlinks(path)
    if output_dir == runtime_base or output_dir.is_relative_to(runtime_base) or runtime_base.is_relative_to(output_dir):
        raise ValueError('Output directory must not overlap the runtime input')
    if (output_dir == project or output_dir.is_relative_to(project)
            or project.is_relative_to(output_dir) or archive_path.is_relative_to(project)):
        raise ValueError('Package outputs must stay outside the source repository')
    if archive_path.is_relative_to(output_dir):
        raise ValueError('Final ZIP must be outside the output tree')
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError('Output directory must be absent or empty')
    if archive_path.exists():
        raise ValueError('Final ZIP already exists; choose a fresh output path')
    manifest_path = runtime_base / 'PAYLOAD_CHECKSUMS.json'
    reject_symlinks(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    expected = set(RUNTIME_FILES) | {APPLICATION}
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError('Runtime manifest must declare exactly nine payloads')
    by_name, runtime_names = {}, set()
    for entry in entries:
        name = entry.get('file')
        if name not in expected or name in by_name:
            raise ValueError(f'Unexpected/duplicate runtime payload: {name!r}')
        path = runtime_base / 'payload' / name
        reject_symlinks(path)
        if path.stat().st_size != entry.get('bytes') or sha256(path) != entry.get('sha256'):
            raise ValueError(f'Runtime SHA256/size mismatch: {name}')
        inspect_zip(path, runtime_names if name in RUNTIME_FILES else None)
        by_name[name] = entry
    version = tomllib.loads((project / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    if not re.fullmatch(r'\d+[.]\d+[.]\d+', version):
        raise ValueError('Package version must use three numeric components')
    files = source_files(project)
    validation_report = safe_name(validation_report)
    if not validation_report.startswith('docs/') or 'app/' + validation_report not in files:
        raise ValueError('Validation report must be an included, Git-tracked docs/ file')
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    source_digest = hashlib.sha256(json_bytes(source_hashes)).hexdigest()
    commit = subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(project), 'status', '--porcelain'], text=True))
    # Runtime payloads can also be supplied as an extracted download. Report a
    # revision only when the input is actually the root of that Git checkout.
    runtime_commit = None
    runtime_checkout = subprocess.run(['git', '-C', str(runtime_base), 'rev-parse', '--show-toplevel'],
                                      capture_output=True, text=True)
    if runtime_checkout.returncode == 0 and Path(runtime_checkout.stdout.strip()).resolve() == runtime_base.resolve():
        runtime_commit = subprocess.check_output(['git', '-C', str(runtime_base), 'rev-parse', 'HEAD'], text=True).strip()
    with zipfile.ZipFile(runtime_base / 'payload' / APPLICATION) as old_app:
        old_manifest = json.loads(old_app.read('PACKAGE_MANIFEST.json').decode('utf-8-sig'))
        license_names = [entry.filename for entry in old_app.infolist()
                         if entry.filename.startswith('licenses/') and not entry.is_dir()]
        if not license_names:
            raise ValueError('Runtime distribution is missing its licenses')
        for name in license_names:
            files[name] = old_app.read(name)
    provenance = old_manifest.get('reused_runtime_provenance') or {
        key: old_manifest.get(key) for key in ('python', 'windows_wheels', 'ffmpeg')}
    pytorch = pytorch_wheels is not None
    torch_metadata = None
    torch_payload = None
    if pytorch:
        python_version = (provenance.get('python') or {}).get('version', '')
        if python_version != '3.12.10':
            raise ValueError('PyTorch CPU payload requires the checked Python 3.12.10 runtime')
        bundled_versions = {}
        for name in RUNTIME_FILES:
            with zipfile.ZipFile(runtime_base / 'payload' / name) as archive:
                for entry in archive.namelist():
                    if entry.endswith('.dist-info/METADATA'):
                        metadata = BytesParser().parsebytes(archive.read(entry))
                        bundled_versions[metadata['Name']] = metadata['Version']
        spec = importlib.util.spec_from_file_location('portable_pytorch', Path(__file__).with_name('portable_pytorch.py'))
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        # Validate and create the optional payload before writing any outputs.
        with tempfile.TemporaryDirectory(prefix='badminton-pytorch-') as folder:
            temporary = Path(folder) / PYTORCH_PAYLOAD
            torch_metadata = helper.prepare_payload(
                pytorch_wheels, temporary, runtime_names, safe_name=safe_name,
                inspect_zip=inspect_zip, bundled_versions=bundled_versions)
            torch_metadata['native_runtime'] = add_pytorch_msvc(runtime_base, temporary, runtime_names)
            inspect_zip(temporary)
            torch_payload = temporary.read_bytes()
        expected.add(PYTORCH_PAYLOAD)
    files['portable_start.py'] = portable_start(pytorch)
    files['사용방법.txt'] = usage_text(version, validation_report, pytorch)
    files['PACKAGE_VALIDATION.txt'] = (
        f'패키지 검증 범위 — {version}\n\n'
        '- 빌더에서 기존 런타임 8개와 이전 앱 ZIP의 SHA256·크기·CRC를 검증했습니다.\n'
        '- 런타임 01~07·09는 바이트 그대로 재사용하며 08 앱만 최신 소스로 재생성합니다.\n'
        + ('- 10-pytorch_cpu.zip에 Windows Python 3.12용 CPU PyTorch와 의존성을 추가했습니다.\n' if pytorch else '') +
        '- 앱 파일별 해시·소스 커밋·런타임 출처를 함께 기록했습니다.\n'
        '- 이 빌더는 자동 테스트나 영상 평가를 실행하지 않습니다.\n'
        f'- 현재 소스의 테스트·영상 평가 기록은 app/{validation_report}에 있습니다.\n'
        '- Windows 실행 파일 및 GPU 실기를 Linux 환경에서 직접 실행 검증하지 못했습니다.\n'
        '- 패키지 생성은 Windows 실행 검증을 대신하지 않습니다.\n'
        '- 원본 경기 영상·과거 대용량 실행 폴더·학습된 모델은 미포함입니다.\n'
    ).encode('utf-8-sig')
    manifest = {
        'schema_version': 1, 'name': 'Badminton AI Windows portable',
        'app_version': version, 'platform': 'Windows x86_64', 'entrypoint': 'START.bat',
        'package_variant': 'pytorch-cpu' if pytorch else 'standard',
        'pytorch': torch_metadata,
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_repository': 'https://github.com/6rmkhj/Badminton_AI_Line_Judge',
        'source_commit': commit, 'source_worktree_dirty': dirty,
        'source_snapshot_sha256': source_digest, 'source_file_count': len(source_hashes),
        'source_file_hashes': source_hashes,
        'source_transformations': ['Root/source *.bat line endings normalized to CRLF'],
        'runtime_base_commit': runtime_commit,
        'runtime_base_manifest_sha256': sha256(manifest_path),
        'reused_runtime_payloads': [by_name[name] for name in RUNTIME_FILES],
        'reused_runtime_provenance': provenance,
        'preserved_runtime_license_files': license_names,
        'original_videos_included': False, 'full_analysis_runs_included': False,
        'trained_pytorch_models_included': False,
        'default_analysis': 'CPU OpenCV candidates; optional CUDA via SETUP_GPU.bat',
        'gpu_setup_interpreter': 'Bundled Python creates app/.venv; START prefers that environment',
        'independent_human_ground_truth_accuracy_measured': False,
        'windows_executable_runtime_tested_in_this_linux_environment': False,
        'validation_report': 'app/' + validation_report,
        'tests_run_by_package_builder': False,
        'download_url': None if pytorch else DOWNLOAD_URL.format(version=version),
    }
    files['PACKAGE_MANIFEST.json'] = json_bytes(manifest)
    files['APP_FILE_CHECKSUMS.json'] = json_bytes({name: hashlib.sha256(data).hexdigest()
                                                for name, data in sorted(files.items())})
    seen = set(runtime_names)
    for name in files:
        safe_name(name)
        if name.casefold() in seen:
            raise ValueError(f'Application/runtime case collision: {name}')
        seen.add(name.casefold())
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = output_dir / 'payload'
    payload.mkdir()
    for name in RUNTIME_FILES:
        shutil.copyfile(runtime_base / 'payload' / name, payload / name)
        if sha256(payload / name) != by_name[name]['sha256']:
            raise ValueError(f'Copied runtime changed: {name}')
    if torch_payload is not None:
        (payload / PYTORCH_PAYLOAD).write_bytes(torch_payload)
    with zipfile.ZipFile(payload / APPLICATION, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    inspect_zip(payload / APPLICATION)
    new_entries = [{'file': name, 'bytes': (payload / name).stat().st_size,
                    'sha256': sha256(payload / name)} for name in sorted(expected)]
    (output_dir / 'PAYLOAD_CHECKSUMS.json').write_bytes(json_bytes(new_entries))
    root_files = {'START.bat': start_batch(version, pytorch), 'RUN.bat': run_batch(),
                  'SETUP_GPU.bat': gpu_batch(), '사용방법.txt': usage_text(version, validation_report, pytorch),
                  'README.md': (f'# 배드민턴 AI Windows {version}\n\n'
                                + ('PyTorch 2.10.0+cpu 포함판입니다. CPU 추론·학습용이며 학습된 모델은 별도로 준비하세요.\n\n' if pytorch else '') +
                                'ZIP 전체를 새 폴더에 압축 해제한 뒤 `START.bat`를 실행하세요.\n\n'
                                'NVIDIA GPU 준비는 `SETUP_GPU.bat`를 실행하세요. 인터넷과 지원 드라이버가 필요합니다.\n\n'
                                '자세한 실행 방법과 검증 한계는 `사용방법.txt`에 있습니다.\n').encode('utf-8')}
    for name, data in root_files.items():
        (output_dir / name).write_bytes(data)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f'badminton-windows-{"pytorch-cpu-" if pytorch else ""}{version}'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_STORED) as archive:
        for path in sorted(output_dir.rglob('*')):
            if path.is_file():
                archive.write(path, prefix + '/' + path.relative_to(output_dir).as_posix())
    inspect_zip(archive_path)
    digest = sha256(archive_path)
    archive_path.with_suffix(archive_path.suffix + '.sha256').write_text(
        digest + '  ' + archive_path.name + '\n', encoding='ascii')
    receipt = {'archive': str(archive_path), 'bytes': archive_path.stat().st_size,
               'sha256': digest, 'output_dir': str(output_dir), 'source_commit': commit,
               'source_worktree_dirty': dirty, 'source_snapshot_sha256': source_digest,
               'app_version': version, 'runtime_payloads_reused': len(RUNTIME_FILES),
               'package_variant': manifest['package_variant'],
               'application_source_files': len(source_hashes), 'windows_runtime_executed': False}
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-base', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--zip', dest='archive_path', type=Path, required=True)
    parser.add_argument('--validation-report', default='docs/WINDOWS_PACKAGE.md',
                        help='Included docs/ report describing validation of this source revision')
    parser.add_argument('--pytorch-wheels', type=Path,
                        help='Add offline CPU PyTorch from Windows Python 3.12 wheels in this directory')
    args = parser.parse_args(argv)
    try:
        receipt = build(args.runtime_base, args.output_dir, args.archive_path,
                        validation_report=args.validation_report, pytorch_wheels=args.pytorch_wheels)
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Package build failed: {error}\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
