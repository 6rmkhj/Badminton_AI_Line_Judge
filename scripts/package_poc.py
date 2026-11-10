"""Package source and completed review results; original recordings stay separate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/integrated-poc')
    parser.add_argument('--request', default='runs/integrated-request')
    parser.add_argument('--output', default='dist/badminton-integrated-poc.zip')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run, request = project / args.run, project / args.request
    index = json.loads((run / 'index.json').read_text(encoding='utf-8'))
    if index.get('status') != 'completed' or any(v.get('status') != 'completed' for v in index.get('videos', [])):
        parser.error('Package only a completed run without processing failures')
    request_result = json.loads((request / 'request-result.json').read_text(encoding='utf-8'))
    if request_result.get('status') != 'completed':
        parser.error('The request execution is incomplete')
    sources = ['badminton', 'configs', 'docs', 'examples', 'scripts', 'tests']
    files = [project / name for name in ('README.md', 'pyproject.toml', 'start.py', 'start-windows.bat')]
    files += list(project.glob('*.pdf'))
    for folder in [*(project / name for name in sources), run, request, project / 'runs/inspection']:
        files.extend(path for path in folder.rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts and path.suffix != '.pyc'
                     and not path.name.startswith('.'))
    target = (project / args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = 'badminton-integrated-poc'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path in sorted(set(files)):
            if not path.resolve().is_relative_to(project):
                raise ValueError(f'Package input escapes project: {path.name}')
            archive.write(path, (Path(prefix) / path.relative_to(project)).as_posix())
        archive.writestr(prefix + '/START_HERE.txt',
            'Python 3.11+\nWindows: start-windows.bat\nmacOS/Linux: python3 start.py\n\n'
            'Includes source and completed analysis results. Original video/ recordings are not duplicated.\n'
            'To reanalyze, place the original project videos in video/ and install FFmpeg.\n'
            'REVIEW results are abstentions, not successful automatic IN/OUT judgments.\n')
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    target.with_suffix('.sha256').write_text(digest.hexdigest() + '  ' + target.name + '\n')
    print(json.dumps({'archive': str(target), 'bytes': target.stat().st_size,
                      'sha256': digest.hexdigest(), 'original_videos_included': False}, indent=2))


if __name__ == '__main__':
    main()
