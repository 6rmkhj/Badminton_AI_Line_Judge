"""Restore the exact approved 0.5.13 Git objects and publish their refs atomically."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

REPOSITORY = '6rmkhj/Badminton_AI_Line_Judge'
BOOTSTRAP_REF = 'refs/heads/downloads/publish-0.5.13-transfer'
TRANSFER = Path(__file__).resolve().parent
EXPECTED = {'source': {'head': '29a2679b95a3ea6185818086307d394abe563eb1',
            'parent': '823e986649ecf023150d9806a54c2e3a1372e6be',
            'tree': 'a287f8a7d583e6e719750889e32d304ac59c0f4d',
            'ref': 'refs/heads/codex/court-line-calibration',
            'bytes': 181149,
            'sha256': 'b5403217f9fa393c92aee6aaef463dff064298fb4ef1e04591128bb88be783b1',
            'thin': False},
 'payload': {'head': 'c298c0e4d277713d901b74a4e2a216432393d430',
             'parent': '30cf899797df4b2543d530a831a825f9ae026d3d',
             'tree': '74fd818c1742266a92c1751dc1792c08f236b9fa',
             'ref': 'refs/heads/downloads/windows-portable-0.5.13',
             'base': '30cf899797df4b2543d530a831a825f9ae026d3d',
             'bytes': 274074,
             'sha256': '06dcd22b8f62d8e3c120b0002249dbbee2e585e9eeaba533cd59bbd9f61d3a76',
             'thin': True},
 'tag': {'name': 'v0.5.13',
         'object': 'deaf7a41c841d4ca0ff598ea909f2cbb3d2368d5',
         'sha256': 'e0b07bee585e0d18c38e0fbae38cede795438585e78739a71a1d8560ab17ac32'}}


def git(*arguments, input_bytes=None):
    result = subprocess.run(['git', *arguments], input=input_bytes,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=600, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace').strip()
                           or f'git {arguments[0]} exited {result.returncode}')
    return result.stdout


def text(*arguments):
    return git(*arguments).decode('utf-8').strip()


def verify_inputs():
    manifest = json.loads((TRANSFER / 'manifest.json').read_text(encoding='utf-8'))
    if set(manifest) != set(EXPECTED):
        raise RuntimeError('Unexpected transfer manifest sections')
    for kind in ('source', 'payload'):
        expected = EXPECTED[kind]
        for key in ('head', 'parent', 'bytes', 'sha256', 'thin') + (('base',) if 'base' in expected else ()):
            if manifest[kind].get(key) != expected[key]:
                raise RuntimeError(f'Unexpected {kind} manifest field: {key}')
        if type(manifest[kind]['thin']) is not bool or type(manifest[kind]['bytes']) is not int:
            raise RuntimeError('Invalid transfer manifest value types')
        path = TRANSFER / f'{kind}.pack'
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != expected['bytes'] or digest != expected['sha256']:
            raise RuntimeError(f'{kind} pack size or SHA256 mismatch')
    if manifest['tag'] != EXPECTED['tag']:
        raise RuntimeError('Unexpected annotated tag manifest')
    tag = (TRANSFER / 'source-tag.txt').read_bytes()
    if hashlib.sha256(tag).hexdigest() != EXPECTED['tag']['sha256']:
        raise RuntimeError('Annotated tag SHA256 mismatch')
    prefix = (f"object {EXPECTED['source']['head']}\ntype commit\n"
              f"tag {EXPECTED['tag']['name']}\n").encode('ascii')
    if not tag.startswith(prefix):
        raise RuntimeError('Annotated tag does not identify the approved source')
    return tag


def remote_refs():
    refs = [EXPECTED['source']['ref'], 'refs/tags/' + EXPECTED['tag']['name'],
            EXPECTED['payload']['ref']]
    result = {}
    for line in text('ls-remote', '--refs', 'origin', *refs).splitlines():
        oid, ref = line.split()
        if ref not in refs or ref in result:
            raise RuntimeError('Unexpected remote reference response')
        result[ref] = oid
    return result


def guard_remote(refs):
    source = EXPECTED['source']
    payload = EXPECTED['payload']
    tag = EXPECTED['tag']
    if refs.get(source['ref']) not in (source['parent'], source['head']):
        raise RuntimeError('Source branch moved; refusing to overwrite it')
    if refs.get(payload['ref']) not in (None, payload['head']):
        raise RuntimeError('Payload branch already points to another commit')
    if refs.get('refs/tags/' + tag['name']) not in (None, tag['object']):
        raise RuntimeError('Release tag already points to another object')


def verify_commit(kind):
    expected = EXPECTED[kind]
    if text('cat-file', '-t', expected['head']) != 'commit':
        raise RuntimeError(f'{kind} head is not a commit')
    headers = git('cat-file', '-p', expected['head']).split(b'\n\n', 1)[0].splitlines()
    trees = [line[5:].decode('ascii') for line in headers if line.startswith(b'tree ')]
    parents = [line[7:].decode('ascii') for line in headers if line.startswith(b'parent ')]
    if trees != [expected['tree']] or parents != [expected['parent']]:
        raise RuntimeError(f'{kind} commit tree or parent mismatch')
    if text('cat-file', '-t', expected['tree']) != 'tree':
        raise RuntimeError(f'{kind} tree is unavailable')
    if text('cat-file', '-t', expected['parent']) != 'commit':
        raise RuntimeError(f'{kind} parent is unavailable')


def main():
    if os.environ.get('GITHUB_REPOSITORY') != REPOSITORY:
        raise RuntimeError('Unexpected GitHub repository')
    if os.environ.get('GITHUB_REF') != BOOTSTRAP_REF:
        raise RuntimeError('Run promotion from the approved bootstrap branch')
    remote = urlsplit(text('remote', 'get-url', 'origin'))
    if (remote.scheme != 'https' or remote.netloc != 'github.com'
            or remote.path.rstrip('/').removesuffix('.git') != '/' + REPOSITORY):
        raise RuntimeError('Unexpected origin repository')
    tag = verify_inputs()
    guard_remote(remote_refs())
    # Avoid downloading historical source blobs. Fetch the payload parent
    # without filtering so thin-pack bases remain available even when the
    # bootstrap replaces an old workflow with its approved new contents.
    git('fetch', '--no-tags', '--filter=blob:none', 'origin',
        EXPECTED['source']['parent'])
    git('fetch', '--no-tags', '--no-filter', '--depth=1', 'origin', EXPECTED['payload']['base'])
    for kind in ('source', 'payload'):
        arguments = ['index-pack', '--stdin']
        if EXPECTED[kind]['thin']:
            arguments.append('--fix-thin')
        git(*arguments, input_bytes=(TRANSFER / f'{kind}.pack').read_bytes())
        verify_commit(kind)
    tag_object = git('hash-object', '-t', 'tag', '-w', '--stdin', input_bytes=tag).decode('ascii').strip()
    if tag_object != EXPECTED['tag']['object']:
        raise RuntimeError('Restored annotated tag object differs')
    if text('cat-file', '-t', tag_object) != 'tag' or git('cat-file', '-p', tag_object) != tag:
        raise RuntimeError('Annotated tag bytes differ after restoration')
    if text('rev-parse', tag_object + '^{}') != EXPECTED['source']['head']:
        raise RuntimeError('Annotated tag does not peel to the source commit')
    guard_remote(remote_refs())
    # No force, no rewritten commits: all three refs succeed or none do.
    git('push', '--atomic', 'origin',
        EXPECTED['source']['head'] + ':' + EXPECTED['source']['ref'],
        tag_object + ':refs/tags/' + EXPECTED['tag']['name'],
        EXPECTED['payload']['head'] + ':' + EXPECTED['payload']['ref'])
    refs = remote_refs()
    expected_refs = {EXPECTED['source']['ref']: EXPECTED['source']['head'],
                     'refs/tags/' + EXPECTED['tag']['name']: tag_object,
                     EXPECTED['payload']['ref']: EXPECTED['payload']['head']}
    if refs != expected_refs:
        raise RuntimeError('Published references do not match the approved objects')
    print(json.dumps({'status': 'promoted', 'refs': expected_refs}, sort_keys=True))


if __name__ == '__main__':
    main()
