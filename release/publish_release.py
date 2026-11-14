"""Publish byte-verified assets only after Windows startup checks have passed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'release'
DIST = ROOT / 'dist'
VERSION = '0.5.13'
TAG = 'v' + VERSION
REPO = '6rmkhj/Badminton_AI_Line_Judge'
NEW_NAME = f'badminton-windows-{VERSION}.zip'
REPORT_NAME = f'windows-{VERSION}-validation.json'


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
    if receipt['app_version'] != VERSION or receipt['source_worktree_dirty'] is not False:
        raise RuntimeError('Expected a clean 0.5.13 package receipt')
    source_commit = receipt['source_commit']
    tag_ref = command('git', 'ls-remote', 'origin', f'refs/tags/{TAG}^{{}}').split()
    if not tag_ref or tag_ref[0] != source_commit:
        raise RuntimeError('Annotated release tag does not match the packaged source')
    names = {NEW_NAME, NEW_NAME + '.sha256', REPORT_NAME}
    if {path.name for path in DIST.iterdir() if path.is_file()} != names:
        raise RuntimeError('Release asset set is incomplete or unexpected')
    expected = {name: {'size': (DIST / name).stat().st_size,
                       'digest': 'sha256:' + digest(DIST / name)} for name in names}
    if (expected[NEW_NAME]['size'] != receipt['bytes']
            or expected[NEW_NAME]['digest'] != 'sha256:' + receipt['sha256']):
        raise RuntimeError('Full package differs from the verified build')
    if (DIST / (NEW_NAME + '.sha256')).read_text(encoding='ascii').split()[0] != receipt['sha256']:
        raise RuntimeError('Release checksum mismatch: ' + NEW_NAME)
    report = json.loads((DIST / REPORT_NAME).read_text(encoding='utf-8'))
    if report['status'] != 'passed':
        raise RuntimeError('Windows verification did not pass')
    if (report['source_commit'] != source_commit or report['app_version'] != VERSION
            or report['tested_distribution'] != f'full_v{VERSION}'
            or report['source_snapshot_sha256'] != receipt['source_snapshot_sha256']
            or report['release_archive_sha256'] != receipt['sha256']
            or report['start_bat_sha256'] != digest(ROOT / 'START.bat')
            or report['payload_manifest_sha256'] != digest(ROOT / 'PAYLOAD_CHECKSUMS.json')):
        raise RuntimeError('Windows report does not match the new package')
    prepares = report.get('prepares', [])
    http = report.get('http', {})
    if (not report.get('powershell_version', '').startswith('5.1.')
            or report.get('payload_count') != 9
            or len(prepares) != 2
            or any(attempt.get('attempt') != index
                   or attempt.get('returncode') != 0
                   or attempt.get('marker_matches_manifest') is not True
                   or attempt.get('timed_out', False)
                   for index, attempt in enumerate(prepares, start=1))
            or http.get('status') != 200
            or http.get('matches_packaged_pair_html') is not True
            or http.get('game_mode_present') is not True
            or http.get('launcher_dependency_checks_completed') is not True):
        raise RuntimeError('Windows PowerShell/prepare/HTTP evidence is incomplete')
    native = report.get('native_geometry_tests', {})
    if (native.get('status') != 'passed' or native.get('returncode') != 0
            or native.get('expected_tests') != 63 or native.get('collected') != 63
            or native.get('passed') != 63 or native.get('failed') != 0
            or native.get('skipped') != 0 or native.get('errors') != 0
            or native.get('timed_out', False)
            or native.get('bundled_runtime_verified') is not True
            or native.get('pytest_dependencies_isolated') is not True
            or native.get('pytest_version') != '8.4.2'
            or native.get('test_files') != ['tests/test_native_landing.py',
                                           'tests/test_native_scale.py',
                                           'tests/test_video_estimate.py']
            or set(native.get('runtime_dependencies', {})) != {'numpy', 'cv2', 'scipy', 'filelock'}):
        raise RuntimeError('Bundled-runtime native geometry evidence is incomplete')

    full = report.get('full_analysis_tests', {})
    image_io = full.get('unicode_image_io', {})
    if (full.get('status') != 'passed' or full.get('returncode') != 0
            or full.get('timed_out', False)
            or full.get('app_version') != VERSION or full.get('source_commit') != source_commit
            or full.get('source_snapshot_sha256') != receipt['source_snapshot_sha256']
            or full.get('bundled_runtime_verified') is not True
            or set(full.get('runtime_dependencies', {})) != {'numpy', 'cv2', 'scipy', 'filelock'}
            or full.get('unicode_paths', {}).get('enabled') is not True
            or image_io.get('status') != 'passed' or image_io.get('helper_roundtrip_equal') is not True):
        raise RuntimeError('Full Windows Unicode-path analysis evidence is incomplete')
    expected_cases = {('A', 'doubles'): 'IN', ('A', 'singles'): 'OUT', ('B', 'doubles'): 'REVIEW'}
    source_hashes = {
        'A': '0cc88d220b046572383d67714b608881da45e15e5165b421d46b5b31667727bb',
        'B': 'bcac92b83d0e03f45acb66fdcf3fe6535acbef0aed53449ed96cfdf5aeb090e9',
    }
    cases = full.get('cases', [])
    if len(cases) != 3 or {(case.get('view'), case.get('game')) for case in cases} != set(expected_cases):
        raise RuntimeError('Missing or duplicate source-bound Windows analysis cases')
    for case in cases:
        key = (case['view'], case['game'])
        if (case.get('status') != 'passed'
                or case.get('source_sha256') != source_hashes[case['view']]
                or case.get('display_decision') != expected_cases[key]
                or case.get('strict_decision') != 'REVIEW'
                or case.get('evidence_warnings') != []
                or case.get('stages') != ['prepare', 'calibration', 'detection', 'tracking',
                                          'contact', 'evidence', 'save']
                or (case['view'] == 'A' and (case.get('decision_kind') != 'video_estimate'
                                            or case.get('estimate_eligible') is not True))):
            raise RuntimeError('Full Windows verdict/evidence regression: ' + str(key))
        artifacts = {artifact.get('path'): artifact for artifact in case.get('artifacts', [])}
        required = {'preview.mp4', 'detections.json', 'overview.jpg', 'evidence/calibration.jpg',
                    'evidence/annotated.mp4', 'evidence/annotated-timing.json'}
        if case['view'] == 'A':
            required.add('evidence/video-estimate.jpg')
        if (not required.issubset(artifacts)
                or any(not artifact.get('sha256') or artifact.get('bytes', 0) <= 0
                       for artifact in artifacts.values())
                or any(artifact.get('frame_count', 0) <= 0 for name, artifact in artifacts.items()
                       if name and name.endswith('.mp4'))
                or artifacts['evidence/annotated.mp4'].get('timestamp_alignment_max_error_seconds', 1) > .0001):
            raise RuntimeError('Real Windows evidence files were not fully verified: ' + str(key))

    endpoint = f'repos/{REPO}/releases'
    existing = [item for item in api(endpoint + '?per_page=100') if item['tag_name'] == TAG]
    if len(existing) > 1:
        raise RuntimeError('Multiple releases have the requested tag')
    if not existing:
        command('gh', 'release', 'create', TAG, '--repo', REPO, '--verify-tag',
                '--target', source_commit, '--draft',
                '--title', '0.5.13 · Windows 한글 경로 근거 저장 오류 수정',
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
    if set(uploaded) - names:
        raise RuntimeError('Existing release contains unexpected assets')
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
    if set(uploaded) != names:
        raise RuntimeError('Uploaded release asset set is incomplete or unexpected')
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
