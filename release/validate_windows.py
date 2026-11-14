"""Verify the new distribution and the exact published 0.5.10 plus START patch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'release'
REPORTS = ROOT / 'reports'
LEGACY_NAME = 'badminton-windows-0.5.10'
LEGACY_SHA = '9044da2d402ed2e799432df23f76900110a032d0004b469d159e6d6b89053177'
LEGACY_URL = ('https://github.com/6rmkhj/Badminton_AI_Line_Judge/releases/download/'
              'v0.5.10/' + LEGACY_NAME + '.zip')
PATCH = RELEASE / 'badminton-windows-0.5.10-start-fix.zip'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def smoke(package, report_name, expected_version, extra):
    path = REPORTS / report_name
    subprocess.run([sys.executable, str(RELEASE / 'verify_windows_portable.py'),
                    '--package', str(package), '--report', str(path)],
                   check=True, timeout=900)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['status'] != 'passed' or data['app_version'] != expected_version:
        raise RuntimeError('Unexpected package or failed Windows startup evidence')
    data.update(extra)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return data


def main():
    if sys.platform != 'win32':
        raise RuntimeError('Run the distribution validation on Windows')
    REPORTS.mkdir(exist_ok=True)
    receipt = json.loads((RELEASE / 'receipt.json').read_text(encoding='utf-8'))
    print('Checking complete 0.5.11 distribution', flush=True)
    new_report = smoke(ROOT, 'windows-0.5.11-validation.json', '0.5.11',
                       {'tested_distribution': 'full_v0.5.11',
                        'release_archive_sha256': receipt['sha256']})
    if new_report['source_commit'] != receipt['source_commit']:
        raise RuntimeError('Windows evidence does not match the packaged source commit')
    patch_sha = digest(PATCH)
    if PATCH.with_suffix('.zip.sha256').read_text(encoding='ascii').split()[0] != patch_sha:
        raise RuntimeError('Hotfix checksum mismatch')
    with tempfile.TemporaryDirectory(prefix='badminton-start-') as temporary:
        base = Path(temporary) / '한글 Windows package'
        base.mkdir()
        archive_path = base / (LEGACY_NAME + '.zip')
        print('Downloading and checking the original published 0.5.10 ZIP', flush=True)
        request = urllib.request.Request(LEGACY_URL, headers={'User-Agent': 'Badminton-Windows-Validation'})
        with urllib.request.urlopen(request, timeout=120) as source, archive_path.open('wb') as destination:
            shutil.copyfileobj(source, destination)
        if digest(archive_path) != LEGACY_SHA:
            raise RuntimeError('Original 0.5.10 release ZIP has changed')
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                relative = Path(entry.filename)
                if relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError('Invalid path in the old release ZIP')
            archive.extractall(base)
        old_package = base / LEGACY_NAME
        with zipfile.ZipFile(PATCH) as patch:
            starter = patch.read('START.bat')
        (old_package / 'START.bat').write_bytes(starter)
        print('Checking original 0.5.10 with only START.bat replaced', flush=True)
        smoke(old_package, 'windows-0.5.10-hotfix-validation.json', '0.5.10',
              {'tested_distribution': 'published_v0.5.10_with_start_hotfix',
               'original_archive_sha256': LEGACY_SHA,
               'applied_hotfix_zip_sha256': patch_sha})
    print('Both complete Windows distributions passed startup verification', flush=True)


if __name__ == '__main__':
    main()
