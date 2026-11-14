"""Exercise a real portable package with Windows PowerShell 5.1 and its launcher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


BASE_PAYLOADS = {
    '01-python.zip', '02-filelock.zip', '03-numpy.zip',
    '04-opencv_python_headless.zip', '05-scipy.zip',
    '06-ffmpeg.zip', '07-ffprobe.zip', '08-application.zip', '09-runtime-notes.zip',
}
PYTORCH_PAYLOAD = '10-pytorch_cpu.zip'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode(data):
    return data.decode('utf-8-sig', errors='replace')


def stop_tree(process):
    """Only terminate the process tree owned by this invocation."""
    if process.poll() is not None:
        return {'already_exited': True, 'returncode': process.returncode}
    stopped = subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
                             capture_output=True, timeout=30)
    process.wait(timeout=30)
    if stopped.returncode and process.returncode is None:
        raise RuntimeError('Could not stop owned launcher: ' + decode(stopped.stderr))
    return {'taskkill_returncode': stopped.returncode, 'returncode': process.returncode}


def command(package, arguments):
    # Supply the complete CreateProcess command line: cmd /s strips the outer
    # quote pair, preserving the quoted batch path with spaces/Korean text.
    cmd = Path(os.environ['SystemRoot']) / 'System32' / 'cmd.exe'
    starter = package / 'START.bat'
    return f'"{cmd}" /d /s /c ""{starter}" {subprocess.list2cmdline(arguments)}"'


def prepare(package, evidence):
    process = subprocess.Popen(command(package, ['--prepare-only']), cwd=package,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    started = time.monotonic()
    try:
        stdout, stderr = process.communicate(b'\r\n', timeout=300)
    except subprocess.TimeoutExpired:
        evidence['cleanup'] = stop_tree(process)
        stdout, stderr = process.communicate(timeout=30)
        evidence.update(stdout=decode(stdout), stderr=decode(stderr), timed_out=True)
        raise RuntimeError('START.bat --prepare-only exceeded 300 seconds')
    evidence.update(returncode=process.returncode, stdout=decode(stdout), stderr=decode(stderr),
                    elapsed_seconds=round(time.monotonic() - started, 3))
    if process.returncode:
        raise RuntimeError(f'START.bat --prepare-only failed ({process.returncode})')


def check_http(package, evidence):
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    url = f'http://127.0.0.1:{port}/pair'
    expected = (package / 'app/badminton/static/pair.html').read_bytes()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command(package, ['--no-browser', '--port', str(port)]),
                                   cwd=package, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        evidence.update(url=url, launcher_pid=process.pid)
        try:
            deadline = time.monotonic() + 120
            last_error = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'Launcher exited before HTTP startup ({process.returncode})')
                try:
                    with opener.open(url, timeout=2) as response:
                        body = response.read(len(expected) + 1)
                        status = response.status
                    if status != 200 or body != expected or b'id="game-mode"' not in body:
                        raise RuntimeError('HTTP response does not match the packaged A/B screen')
                    evidence.update(status=status, body_sha256=hashlib.sha256(body).hexdigest(),
                                    matches_packaged_pair_html=True, game_mode_present=True,
                                    launcher_dependency_checks_completed=True)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                    last_error = str(error)
                    time.sleep(.25)
            else:
                raise RuntimeError('HTTP startup exceeded 120 seconds: ' + str(last_error))
        finally:
            try:
                evidence['cleanup'] = stop_tree(process)
            finally:
                stdout.seek(0)
                stderr.seek(0)
                evidence['stdout'] = decode(stdout.read())
                evidence['stderr'] = decode(stderr.read())


def check_pytorch_cpu(package, evidence):
    """Run compiled CPU kernels and autograd in the package's own interpreter."""
    code = r'''
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import torch

runtime = Path(sys.executable).resolve().parent
if not Path(torch.__file__).resolve().is_relative_to(runtime):
    raise RuntimeError('PyTorch was imported from outside the bundled runtime')
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE
kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
kernel32.GetModuleFileNameW.restype = wintypes.DWORD
handle = kernel32.GetModuleHandleW('msvcp140.dll')
if not handle:
    raise RuntimeError('PyTorch did not load msvcp140.dll')
buffer = ctypes.create_unicode_buffer(32768)
length = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
if not length or length >= len(buffer):
    raise RuntimeError('Could not resolve the loaded msvcp140.dll path')
dll_path = Path(buffer.value).resolve()
if not dll_path.is_relative_to(runtime / 'Lib/site-packages/torch/lib'):
    raise RuntimeError('PyTorch used msvcp140.dll outside the bundled torch/lib: ' + str(dll_path))
# Import NumPy only after checking PyTorch's native dependency, so a separately
# loaded NumPy runtime cannot make this check pass accidentally.
import numpy as np

if torch.__version__ != '2.10.0+cpu' or torch.version.cuda is not None:
    raise RuntimeError('Expected the pinned CPU-only PyTorch build')
if torch.cuda.is_available():
    raise RuntimeError('CPU package unexpectedly exposes CUDA')
torch.set_num_threads(2)
torch.manual_seed(0)
sample = np.arange(12, dtype=np.float32).reshape(3, 4)
if not np.array_equal(torch.from_numpy(sample).numpy(), sample):
    raise RuntimeError('NumPy/Tensor round trip failed')
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 4, 3), torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2),
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
before = model[-1].weight.detach().clone()
loss = torch.nn.functional.cross_entropy(
    model(torch.randn(4, 3, 16, 16)), torch.tensor([0, 1, 0, 1]),
)
loss.backward()
optimizer.step()
if not torch.isfinite(loss) or torch.equal(before, model[-1].weight):
    raise RuntimeError('CPU forward/backward/optimizer check failed')
print(json.dumps({
    'version': torch.__version__, 'cuda_build': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'python': sys.executable, 'module': torch.__file__,
    'msvcp140_dll': str(dll_path),
    'numpy_roundtrip': True, 'cpu_convolution': True,
    'backward_optimizer_update': True,
}))
'''
    environment = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONPATH='', PIP_NO_INDEX='1')
    result = subprocess.run(
        [str(package / 'runtime/python.exe'), '-I', '-X', 'utf8', '-c', code],
        cwd=package, env=environment, capture_output=True, timeout=180,
    )
    evidence.update(returncode=result.returncode, stdout=decode(result.stdout), stderr=decode(result.stderr))
    if result.returncode:
        raise RuntimeError(f'Bundled PyTorch CPU check failed ({result.returncode}): {decode(result.stderr)}')
    evidence['checks'] = json.loads(decode(result.stdout))


def verify(package, report):
    if sys.platform != 'win32':
        raise RuntimeError('This verification requires Windows and Windows PowerShell 5.1')
    if (package / 'app/.venv').exists():
        raise RuntimeError('Use a fresh package without app/.venv to verify the bundled runtime')
    powershell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    version = subprocess.run([str(powershell), '-NoLogo', '-NoProfile', '-NonInteractive',
                              '-Command', '$PSVersionTable.PSVersion.ToString()'],
                             capture_output=True, timeout=30, check=True)
    report['powershell_version'] = decode(version.stdout).strip()
    if not report['powershell_version'].startswith('5.1.'):
        raise RuntimeError('Windows PowerShell 5.1 was not selected')
    manifest_path = package / 'PAYLOAD_CHECKSUMS.json'
    manifest_digest = sha256(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    names = [entry.get('file') for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
    expected_sets = (BASE_PAYLOADS, BASE_PAYLOADS | {PYTORCH_PAYLOAD})
    if not isinstance(entries, list) or len(names) != len(entries) or len(set(names)) != len(names) or set(names) not in expected_sets:
        raise RuntimeError('Expected the exact standard or PyTorch CPU payload set')
    pytorch_included = PYTORCH_PAYLOAD in names
    report.update(payload_manifest_sha256=manifest_digest,
                  payload_count=len(entries), start_bat_sha256=sha256(package / 'START.bat'), prepares=[])
    for attempt in range(2):
        evidence = {'attempt': attempt + 1}
        report['prepares'].append(evidence)
        prepare(package, evidence)
        manifest = json.loads((package / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8-sig'))
        marker = package / ('.package-ready-' + manifest['app_version'])
        evidence['marker_matches_manifest'] = marker.read_text(encoding='ascii').strip().lower() == manifest_digest
        if not evidence['marker_matches_manifest']:
            raise RuntimeError('Ready marker does not match the verified payload manifest')
    report.update(app_version=manifest['app_version'], source_commit=manifest['source_commit'],
                  package_manifest_sha256=sha256(package / 'PACKAGE_MANIFEST.json'),
                  source_snapshot_sha256=manifest['source_snapshot_sha256'])
    expected_variant = 'pytorch-cpu' if pytorch_included else 'standard'
    if manifest.get('package_variant', 'standard') != expected_variant:
        raise RuntimeError('Package variant and payload set disagree')
    report['package_variant'] = expected_variant
    required = ('portable_start.py', 'runtime/python.exe', 'tools/ffmpeg.exe', 'tools/ffprobe.exe',
                'app/badminton/static/pair.html')
    if not all((package / name).is_file() for name in required):
        raise RuntimeError('Runtime/application extraction is incomplete')
    if pytorch_included:
        report['pytorch_cpu'] = {}
        check_pytorch_cpu(package, report['pytorch_cpu'])
    report['http'] = {}
    check_http(package, report['http'])
    # portable_start imports numpy/cv2/scipy/filelock, checks FFmpeg's libx264
    # encoder and runs FFprobe before start.py can serve the verified page.
    report['runtime_checks'] = ['bundled_python', 'numpy', 'opencv', 'scipy', 'filelock',
                                'ffmpeg_libx264', 'ffprobe', 'pair_http_200']
    if pytorch_included:
        report['runtime_checks'] += ['pytorch_cpu_version', 'pytorch_numpy_roundtrip',
                                     'pytorch_cpu_convolution', 'pytorch_backward_optimizer',
                                     'pytorch_bundled_msvcp140']
    report['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = {'status': 'failed', 'package': str(args.package.resolve()),
              'scope': 'Windows launcher/CPU startup smoke; no GPU or match analysis'}
    try:
        verify(args.package.resolve(), report)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        print(report['error'], file=sys.stderr)
        for evidence in [*report.get('prepares', []), report.get('pytorch_cpu', {}), report.get('http', {})]:
            for name in ('stdout', 'stderr'):
                if evidence.get(name):
                    print(f'{name}:\n{evidence[name]}', file=sys.stderr)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'app_version', 'source_commit',
                                                  'powershell_version', 'error') if key in report},
                     ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
