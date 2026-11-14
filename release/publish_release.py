"""Publish byte-verified assets only after Windows startup checks have passed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'release'
DIST = ROOT / 'dist'
TAG = 'v0.5.11'
REPO = '6rmkhj/Badminton_AI_Line_Judge'
PATCH_NAME = 'badminton-windows-0.5.10-start-fix.zip'
NEW_NAME = 'badminton-windows-0.5.11.zip'
REPORT_NAMES = ('windows-0.5.11-validation.json', 'windows-0.5.10-hotfix-validation.json')


def command(*args):
    result = subprocess.run(args, cwd=ROOT, text=True, encoding='utf-8',
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f'{args[0]} exited {result.returncode}')
    return result.stdout.strip()


def api(endpoint, *args):
    return json.loads(command('gh', 'api', endpoint, *args))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if os.environ.get('GH_REPO') != REPO:
        raise RuntimeError('Unexpected release repository')
    receipt = json.loads((RELEASE / 'receipt.json').read_text(encoding='utf-8'))
    source_commit = receipt['source_commit']
    tag_ref = command('git', 'ls-remote', 'origin', f'refs/tags/{TAG}^{{}}').split()
    if not tag_ref or tag_ref[0] != source_commit:
        raise RuntimeError('Annotated release tag does not match the packaged source')
    for name in (PATCH_NAME, PATCH_NAME + '.sha256'):
        shutil.copyfile(RELEASE / name, DIST / name)
    names = {NEW_NAME, NEW_NAME + '.sha256', PATCH_NAME, PATCH_NAME + '.sha256', *REPORT_NAMES}
    if {path.name for path in DIST.iterdir() if path.is_file()} != names:
        raise RuntimeError('Release asset set is incomplete or unexpected')
    expected = {name: {'size': (DIST / name).stat().st_size,
                       'digest': 'sha256:' + digest(DIST / name)} for name in names}
    if (expected[NEW_NAME]['size'] != receipt['bytes']
            or expected[NEW_NAME]['digest'] != 'sha256:' + receipt['sha256']):
        raise RuntimeError('Full package differs from the verified build')
    for name in (NEW_NAME, PATCH_NAME):
        if (DIST / (name + '.sha256')).read_text(encoding='ascii').split()[0] != digest(DIST / name):
            raise RuntimeError('Release checksum mismatch: ' + name)
    reports = [json.loads((DIST / name).read_text(encoding='utf-8')) for name in REPORT_NAMES]
    if any(report['status'] != 'passed' for report in reports):
        raise RuntimeError('Windows verification did not pass')
    if (reports[0]['source_commit'] != source_commit or reports[0]['app_version'] != '0.5.11'
            or reports[0]['release_archive_sha256'] != receipt['sha256']):
        raise RuntimeError('Windows report does not match the new package')
    if (reports[1]['app_version'] != '0.5.10'
            or reports[1]['applied_hotfix_zip_sha256'] != digest(DIST / PATCH_NAME)):
        raise RuntimeError('Windows report does not match the legacy patch')

    endpoint = f'repos/{REPO}/releases'
    existing = [item for item in api(endpoint + '?per_page=100') if item['tag_name'] == TAG]
    if len(existing) > 1:
        raise RuntimeError('Multiple releases have the requested tag')
    if not existing:
        command('gh', 'release', 'create', TAG, '--repo', REPO, '--verify-tag',
                '--target', source_commit, '--draft',
                '--title', '0.5.11 · Windows PowerShell 시작 오류 수정',
                '--notes-file', str(RELEASE / 'notes.md'),
                *(str(DIST / name) for name in sorted(names)))
        existing = [item for item in api(endpoint + '?per_page=100') if item['tag_name'] == TAG]
    if len(existing) != 1:
        raise RuntimeError('Could not locate the release draft')
    release_id = existing[0]['id']
    release = api(f'{endpoint}/{release_id}')
    if release['target_commitish'] != source_commit:
        raise RuntimeError('Existing release uses a different source commit')
    uploaded = {asset['name']: asset for asset in release['assets']}
    for name in sorted(names):
        if name in uploaded:
            asset = uploaded[name]
            if (asset['state'] != 'uploaded' or asset['size'] != expected[name]['size']
                    or asset.get('digest') != expected[name]['digest']):
                raise RuntimeError('Existing asset differs; refusing to overwrite: ' + name)
        else:
            command('gh', 'release', 'upload', TAG, '--repo', REPO, str(DIST / name))
    release = api(f'{endpoint}/{release_id}')
    uploaded = {asset['name']: asset for asset in release['assets']}
    for name, metadata in expected.items():
        asset = uploaded[name]
        if (asset['state'] != 'uploaded' or asset['size'] != metadata['size']
                or asset.get('digest') != metadata['digest']):
            raise RuntimeError('Uploaded asset failed verification: ' + name)
    published = api(f'{endpoint}/{release_id}', '--method', 'PATCH',
                    '-F', 'draft=false', '-f', 'make_latest=true')
    if published['draft']:
        raise RuntimeError('Release is still a draft')
    print(json.dumps({'url': published['html_url'], 'source_commit': source_commit,
                      'asset_count': len(names), 'uploaded_checksums_verified': True}))


if __name__ == '__main__':
    main()
