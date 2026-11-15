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
    if not isinstance(entries, list) or len(entries) != 9:
        raise RuntimeError('Expected nine payload entries in the real package')
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
    required = ('portable_start.py', 'runtime/python.exe', 'tools/ffmpeg.exe', 'tools/ffprobe.exe',
                'app/badminton/static/pair.html')
    if not all((package / name).is_file() for name in required):
        raise RuntimeError('Runtime/application extraction is incomplete')
    report['http'] = {}
    check_http(package, report['http'])
    # portable_start imports numpy/cv2/scipy/filelock, checks FFmpeg's libx264
    # encoder and runs FFprobe before start.py can serve the verified page.
    report['runtime_checks'] = ['bundled_python', 'numpy', 'opencv', 'scipy', 'filelock',
                                'ffmpeg_libx264', 'ffprobe', 'pair_http_200']
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
        for evidence in [*report.get('prepares', []), report.get('http', {})]:
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
