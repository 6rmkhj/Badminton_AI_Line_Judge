"""Run source-bound full analyses with the portable Windows runtime.

This is a release regression, not a measurement of independent IN/OUT accuracy.
It deliberately uses Korean/space-containing source, output and TEMP paths.
Run after START.bat --prepare-only, using runtime/python.exe itself.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from urllib.parse import quote
from urllib.request import urlopen


SOURCE_COMMIT = '6fb6a53f3e606fa4f56d8083015cdcfb5ecd000f'
SOURCE_ROOT = ('https://raw.githubusercontent.com/6rmkhj/Badminton_AI_Line_Judge/'
               + SOURCE_COMMIT + '/')
SOURCES = {
    'A': {'path': 'video/8k 30 4k 24 a시점 1.mp4', 'bytes': 73984014,
          'sha256': '0cc88d220b046572383d67714b608881da45e15e5165b421d46b5b31667727bb',
          'provenance': 'Repository source; not byte-verified against the oversized user attachment.'},
    'B': {'path': 'video/8k 30 4k 24 b시점 1.mp4', 'bytes': 13197455,
          'sha256': 'bcac92b83d0e03f45acb66fdcf3fe6535acbef0aed53449ed96cfdf5aeb090e9',
          'provenance': 'Repository source SHA-256 matches the supplied user attachment.'},
}
CASES = [('A', 'doubles', 'IN'), ('B', 'doubles', 'REVIEW'), ('A', 'singles', 'OUT')]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def source_file(view, directory):
    specification = SOURCES[view]
    destination = directory / Path(specification['path']).name
    if destination.exists():
        if (destination.stat().st_size == specification['bytes']
                and digest(destination) == specification['sha256']):
            return destination
        raise RuntimeError('Existing regression source differs from frozen source: ' + view)
    directory.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.download')
    url = SOURCE_ROOT + quote(specification['path'])
    try:
        with urlopen(url, timeout=120) as response, partial.open('wb') as stream:
            received = 0
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > specification['bytes']:
                    raise RuntimeError('Regression source exceeds frozen size: ' + view)
                stream.write(chunk)
        if partial.stat().st_size != specification['bytes'] or digest(partial) != specification['sha256']:
            raise RuntimeError('Regression source failed SHA-256/size validation: ' + view)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def check_image_paths(work, report):
    import cv2
    import numpy as np
    from badminton.image_io import read_image, write_image

    destination = work / '이미지 저장 확인 공백'
    destination.mkdir(parents=True, exist_ok=True)
    frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    legacy = destination / 'legacy.png'
    try:
        returned = bool(cv2.imwrite(str(legacy), frame))
        old = {'returned': returned, 'file_exists': legacy.is_file()}
    except (OSError, cv2.error) as error:
        old = {'returned': False, 'file_exists': legacy.is_file(),
               'error': f'{type(error).__name__}: {error}'}
    old['failure_reproduced'] = old['returned'] is False or old['file_exists'] is False
    fixed = destination / '근거 이미지.png'
    write_image(fixed, frame)
    decoded = read_image(fixed)
    if decoded is None or not np.array_equal(decoded, frame):
        raise RuntimeError('Unicode-path PNG did not round-trip through the packaged image helper')
    report['unicode_image_io'] = {'status': 'passed', 'legacy_cv2_imwrite': old,
                                'helper_roundtrip_equal': True, 'path': str(fixed),
                                'sha256': digest(fixed)}


def relative_artifacts(result):
    """Collect every declared media path, including nested native evidence."""
    paths = set()
    suffixes = {'.jpg', '.jpeg', '.png', '.mp4', '.json'}

    def collect(value):
        if isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)
        elif isinstance(value, str):
            path = Path(value)
            if (not path.is_absolute() and '..' not in path.parts and path.suffix.lower() in suffixes
                    and (value.startswith('evidence/') or value in {'overview.jpg', 'preview.mp4', 'detections.json'})):
                paths.add(value)

    for name in ('artifacts', 'video_estimate', 'landings'):
        collect(result.get(name))
    return sorted(paths)


def check_artifacts(output, result):
    import cv2
    from badminton.image_io import read_image
    from badminton.media import probe, load_frames

    paths = relative_artifacts(result)
    required = {'preview.mp4', 'detections.json', 'overview.jpg', 'evidence/calibration.jpg',
                'evidence/annotated.mp4', 'evidence/annotated-timing.json'}
    if not required.issubset(paths):
        raise RuntimeError('Required evidence was not declared: ' + ', '.join(sorted(required - set(paths))))
    if result.get('video_estimate', {}).get('eligible') is True:
        required.add('evidence/video-estimate.jpg')
    if not required.issubset(paths):
        raise RuntimeError('Resolved video estimate has no rendered evidence')
    checks = []
    for relative in paths:
        path = output / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError('Declared artifact missing or empty: ' + relative)
        row = {'path': relative, 'bytes': path.stat().st_size, 'sha256': digest(path)}
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            decoded = read_image(path, cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                raise RuntimeError('Evidence image did not decode: ' + relative)
            row['dimensions'] = [decoded.shape[1], decoded.shape[0]]
        elif path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
        elif path.suffix == '.mp4':
            metadata = probe(path)
            decoded = load_frames(path, metadata['frame_count'])
            if len(decoded) != metadata['frame_count'] or not decoded:
                raise RuntimeError('Evidence video did not decode completely: ' + relative)
            row.update(frame_count=len(decoded), dimensions=[metadata['width'], metadata['height']])
            del decoded
            if relative == 'evidence/annotated.mp4':
                timing = json.loads((output / 'evidence/annotated-timing.json').read_text(encoding='utf-8'))
                timeline = timing['frames']
                if len(timeline) != metadata['frame_count']:
                    raise RuntimeError('Replay frame count differs from its source timeline')
                maximum_error = max(abs(actual - expected['replay_time']) for actual, expected in
                                    zip(metadata['timestamps'], timeline))
                if maximum_error > .0001:
                    raise RuntimeError('Replay timestamps differ from source timeline')
                row['timestamp_alignment_max_error_seconds'] = maximum_error
        checks.append(row)
    return checks


def run_cases(app, work, report, report_path):
    from badminton.pipeline import analyze_video, implementation_fingerprint
    from badminton.resources import apply_runtime

    config = json.loads((app / 'configs/poc.json').read_text(encoding='utf-8'))
    # The app's default CPU execution path, with a bounded frame buffer and no
    # cached results, models, annotations or reference landing coordinates.
    settings = {'device': 'cpu', 'cpu_threads': 2, 'parallel_views': 1, 'frame_memory_mb': 2048}
    apply_runtime(settings)
    config['resource_settings'] = settings
    report['implementation_fingerprint'] = implementation_fingerprint()
    report['cases'] = []
    sources = {view: source_file(view, work / '원본 영상') for view in SOURCES}
    for view, game, expected in CASES:
        output = work / f'{view} 시점 {game} 결과'
        if (output / 'result.json').exists():
            raise RuntimeError('Use a fresh output directory; cached analysis is not a release gate')
        case = {'view': view, 'game': game, 'expected_display_decision': expected,
                'source_sha256': SOURCES[view]['sha256'], 'output': str(output), 'status': 'failed'}
        report['cases'].append(case)
        started = time.monotonic()
        stages = []

        def progress(fields):
            stage = fields.get('stage')
            if stage and (not stages or stages[-1] != stage):
                stages.append(stage)
                print(json.dumps({'view': view, 'game': game, 'stage': stage}), flush=True)

        try:
            current = copy.deepcopy(config)
            current['rules'] = {'game': game, 'phase': 'rally'}
            result = analyze_video(sources[view], output, current, progress_callback=progress)
            persisted = json.loads((output / 'result.json').read_text(encoding='utf-8'))
            if persisted != json.loads(json.dumps(result, allow_nan=False)):
                raise RuntimeError('Saved result differs from the returned pipeline result')
            if result['source_sha256'] != SOURCES[view]['sha256']:
                raise RuntimeError('Analysis source identity changed')
            if stages != ['prepare', 'calibration', 'detection', 'tracking', 'contact', 'evidence', 'save']:
                raise RuntimeError('Full evidence/save pipeline did not complete: ' + str(stages))
            if result.get('evidence_warnings'):
                raise RuntimeError('Evidence generation reported a failure: ' + str(result['evidence_warnings']))
            display = result['display_decision']['decision']
            estimate = result['video_estimate']
            if display != expected or result['decision']['decision'] != 'REVIEW':
                raise RuntimeError('Source-bound display/strict verdict regression: ' + str((display, result['decision'])))
            if view == 'A' and (estimate.get('eligible') is not True
                                or result['display_decision'].get('decision_kind') != 'video_estimate'):
                raise RuntimeError('A verdict lost its explicit video-estimate provenance')
            case.update(artifacts=check_artifacts(output, result), display_decision=display,
                        strict_decision=result['decision']['decision'],
                        decision_kind=result['display_decision'].get('decision_kind'),
                        estimate_eligible=estimate.get('eligible'), evidence_warnings=[],
                        result_sha256=digest(output / 'result.json'), status='passed')
        except Exception as error:
            case.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            print(case['traceback'], file=sys.stderr, flush=True)
        finally:
            case.update(stages=stages, elapsed_seconds=round(time.monotonic() - started, 3))
            save_report(report_path, report)
    if any(case['status'] != 'passed' for case in report['cases']):
        raise RuntimeError('At least one source-bound full Windows analysis failed')


def verify(package, work, report, report_path):
    if sys.platform != 'win32':
        raise RuntimeError('The full portable analysis gate must execute on Windows')
    runtime = package / 'runtime'
    if Path(sys.executable).resolve() != (runtime / 'python.exe').resolve():
        raise RuntimeError('Run this script using the package runtime/python.exe')
    app = package / 'app'
    sys.path.insert(0, str(app))
    dependencies = {}
    for name in ('numpy', 'cv2', 'scipy', 'filelock'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(runtime):
            raise RuntimeError('Dependency was not loaded from the package: ' + name)
        dependencies[name] = {'version': module.__version__, 'file': str(path)}
    os.environ['PATH'] = str(package / 'tools') + os.pathsep + os.environ.get('PATH', '')
    for tool in ('ffmpeg', 'ffprobe'):
        actual = shutil.which(tool)
        if not actual or Path(actual).resolve() != (package / 'tools' / (tool + '.exe')).resolve():
            raise RuntimeError('The bundled media executable was not selected: ' + tool)
    manifest = json.loads((package / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    report.update(app_version=manifest['app_version'], source_commit=manifest['source_commit'],
                  source_snapshot_sha256=manifest['source_snapshot_sha256'],
                  runtime_executable=str(Path(sys.executable).resolve()),
                  bundled_runtime_verified=True, runtime_dependencies=dependencies)
    work.mkdir(parents=True, exist_ok=True)
    temporary = work / '임시 파일 공백'
    temporary.mkdir(exist_ok=True)
    # tempfile caches the discovered directory, so set it as well as the
    # environment. This reproduces a Windows account with Korean TEMP paths.
    for name in ('TMP', 'TEMP', 'TMPDIR'):
        os.environ[name] = str(temporary)
    tempfile.tempdir = str(temporary)
    report['unicode_paths'] = {'work': str(work), 'temporary': str(temporary), 'enabled': True}
    check_image_paths(work, report)
    save_report(report_path, report)
    run_cases(app, work, report, report_path)
    report['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    # Console/report transport only; application file encodings remain explicit
    # and no UTF-8 process-mode bypass is introduced for the regression.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    report = {'status': 'failed', 'scope': 'Full source-bound CPU pipeline and Unicode-path evidence regression; '
              'not independently measured contact/identity/IN-OUT accuracy.',
              'source_repository_commit': SOURCE_COMMIT, 'sources': SOURCES}
    started = time.monotonic()
    try:
        verify(args.package.resolve(), args.work_dir.resolve(), report, args.report.resolve())
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(report['traceback'], file=sys.stderr)
    report['elapsed_seconds'] = round(time.monotonic() - started, 3)
    save_report(args.report.resolve(), report)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
