"""Run fresh A/B analyses for every filename pair in a video folder.

Each view goes through the production ``analyze_video`` pipeline, then the two
views are linked with the same ``combine_views`` used by the A/B workspace. The
operator pairing (a<N> with b<N>) is treated as a confirmed same-rally selection,
matching a user who uploads both files into the A/B screen.

AI visual-review labels are read only *after* analysis, to report agreement.
They are never passed into the pipeline.

Examples::

    python -X utf8 scripts/run_all_pairs.py --workers 6
    python -X utf8 scripts/run_all_pairs.py --recombine-only      # reuse saved views
    python -X utf8 scripts/run_all_pairs.py --pairs "8k 30 4k 24 {view}시점 4"
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def discover_pairs(video_dir):
    from badminton.dataset import pair_identity
    pairs = {}
    for path in sorted(Path(video_dir).iterdir()):
        if path.suffix.lower() not in {'.mp4', '.mov', '.m4v'}:
            continue
        key, view = pair_identity(path.name)
        if key is None or view not in {'a', 'b'}:
            continue
        pairs.setdefault(key, {})[view.upper()] = path
    complete = {key: views for key, views in pairs.items() if set(views) == {'A', 'B'}}
    return dict(sorted(complete.items(), key=lambda item: _natural(item[0])))


def _natural(text):
    import re
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', text)]


def _slug(key):
    import hashlib
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:10]


def _view_output(run_dir, game, key, camera):
    return Path(run_dir) / game / _slug(key) / camera


def _analyze_one(task):
    """Worker entry point: one production analysis in a fresh process."""
    video, output, game, threads, memory_mb = task
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    from badminton.pipeline import analyze_video
    from badminton.resources import apply_runtime
    config = json.loads((ROOT / 'configs/poc.json').read_text(encoding='utf-8'))
    settings = {'device': 'cpu', 'cpu_threads': threads, 'parallel_views': 1, 'frame_memory_mb': memory_mb}
    apply_runtime(settings)
    config.update(resource_settings=settings, opencv_threads=threads,
                  rules={'game': game, 'phase': 'rally'}, max_frames=3000,
                  video_id='allpairs_' + Path(output).parent.name + '_' + Path(output).name.lower())
    started = time.monotonic()
    # A failed re-analysis must not leave an older result.json to be counted.
    for stale in ('result.json', 'analysis-error.log'):
        (Path(output) / stale).unlink(missing_ok=True)
    try:
        analyze_video(video, output, config)
        return {'video': str(video), 'output': str(output), 'status': 'completed',
                'elapsed_seconds': round(time.monotonic() - started, 2)}
    except Exception as error:  # report, never hide, a failed view
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / 'analysis-error.log').write_text(traceback.format_exc(), encoding='utf-8')
        return {'video': str(video), 'output': str(output), 'status': 'failed',
                'error': f'{type(error).__name__}: {error}', 'elapsed_seconds': round(time.monotonic() - started, 2)}


def _fresh(output):
    """A saved view is reusable only when it came from the current code."""
    from badminton.pipeline import implementation_fingerprint
    path = Path(output) / 'result.json'
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return result.get('implementation_fingerprint') == implementation_fingerprint()


def build_view(camera, video, output, rules, recompute_best=False):
    """Mirror the fresh-analysis view the A/B workspace hands to combine_views.

    ``recompute_best`` re-derives only the best-estimate tier from the saved
    evidence (a development shortcut; the final check must be a fresh run).
    """
    from badminton.pair_workspace import view_from_result
    error_log = Path(output) / 'analysis-error.log'
    result_path = Path(output) / 'result.json'
    if not result_path.is_file():
        return {'camera_id': camera, 'file': Path(video).name, 'status': 'failed', 'rules': rules,
                'source_sha256': None, 'automatic_decision': 'REVIEW', 'direct_decision': 'REVIEW',
                'assisted_decision': 'REVIEW', 'qualified': False, 'video_estimate': None,
                'estimated_decision': 'REVIEW', 'reason_codes': ['VIEW_ANALYSIS_FAILED'],
                'error': error_log.read_text(encoding='utf-8')[-800:] if error_log.is_file() else 'missing result'}
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if recompute_best:
        result.pop('best_estimate', None)  # view_from_result re-derives it from saved evidence
    view = view_from_result(result, Path(output), f'/allpairs/{camera}/', rules)
    view.update(camera_id=camera, file=Path(video).name, status='completed', rules=dict(rules))
    return view


def load_reference(project):
    """AI visual-review labels keyed by source SHA-256 (evaluation only)."""
    labels = {}
    for path in sorted((project / 'annotations/ai-visual').glob('*.json')):
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if row.get('source_sha256') and row.get('label') in {'IN', 'OUT', 'REVIEW'}:
            labels[row['source_sha256']] = {'label': row['label'], 'file': row.get('file'),
                                            'contact_px': row.get('contact_px'),
                                            'contact_frame_index': row.get('contact_frame_index')}
    return labels


def summarize_view(view):
    estimate = view.get('video_estimate') or {}
    fallback = view.get('best_estimate') or {}
    return {'status': view.get('status'), 'direct': view.get('direct_decision'),
            'estimate': estimate.get('decision'), 'estimate_codes': estimate.get('reason_codes'),
            'best_estimate': fallback.get('decision'), 'best_estimate_confidence': fallback.get('confidence'),
            'best_estimate_basis': fallback.get('basis'),
            'effective': view.get('effective_decision'), 'effective_kind': view.get('effective_decision_kind'),
            'reason_codes': view.get('reason_codes'), 'error': view.get('error')}


def combine_all(pairs, run_dir, game, reference, recompute_best=False):
    from badminton.pair_workspace import combine_views
    rules = {'game': game, 'phase': 'rally'}
    rows = []
    for key, paths in pairs.items():
        views = {camera: build_view(camera, paths[camera], _view_output(run_dir, game, key, camera), rules,
                                      recompute_best)
                 for camera in ('A', 'B')}
        result = combine_views(copy.deepcopy(views), same_event_confirmed=True,
                               event_source='operator_pair_selection', rules=rules)
        refs = {camera: reference.get(views[camera].get('source_sha256') or '', {}).get('label')
                for camera in ('A', 'B')}
        resolved = [label for label in refs.values() if label in {'IN', 'OUT'}]
        expected = resolved[0] if resolved and len(set(resolved)) == 1 else None
        rows.append({'pair_key': key, 'files': {c: paths[c].name for c in ('A', 'B')},
                     'decision': result['decision'], 'decision_kind': result.get('decision_kind'),
                     'source_camera_ids': result.get('source_camera_ids'),
                     'reason_codes': result.get('reason_codes'),
                     'confidence': result.get('confidence'),
                     'ai_reference': refs, 'ai_reference_label': expected,
                     'agrees_with_ai_reference': (None if expected is None or game != 'doubles'
                                                  else result['decision'] == expected),
                     'views': {camera: summarize_view(result['views'][camera]) for camera in ('A', 'B')}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video-dir', type=Path, default=ROOT / 'video')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/all-pairs')
    parser.add_argument('--games', default='doubles', help='Comma list: doubles,singles')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) // 3))
    parser.add_argument('--threads', type=int, default=2, help='OpenCV/decoder threads per analysis')
    parser.add_argument('--frame-memory-mb', type=int, default=3072)
    parser.add_argument('--recombine-only', action='store_true', help='Never analyze; link saved views')
    parser.add_argument('--force', action='store_true', help='Re-analyze even when current results exist')
    parser.add_argument('--pairs', default='', help='Comma list of pair keys to restrict the run')
    parser.add_argument('--report', type=Path, default=None)
    parser.add_argument('--recompute-best-estimate', action='store_true',
                        help='Development only: re-derive the best-estimate tier from saved evidence')
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')

    pairs = discover_pairs(args.video_dir)
    if args.pairs:
        wanted = {item.strip() for item in args.pairs.split(',') if item.strip()}
        pairs = {key: value for key, value in pairs.items() if key in wanted}
    games = [game.strip() for game in args.games.split(',') if game.strip()]
    print(json.dumps({'event': 'pairs', 'count': len(pairs), 'games': games}, ensure_ascii=False), flush=True)

    tasks = []
    if not args.recombine_only:
        for game in games:
            for key, paths in pairs.items():
                for camera in ('A', 'B'):
                    output = _view_output(args.run_dir, game, key, camera)
                    if not args.force and _fresh(output):
                        continue
                    tasks.append((str(paths[camera]), str(output), game, args.threads, args.frame_memory_mb))
    # Analyse the slowest (largest) sources first so the pool drains evenly.
    tasks.sort(key=lambda task: -Path(task[0]).stat().st_size)
    analysis = []
    if tasks:
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_analyze_one, task) for task in tasks]
            for number, future in enumerate(as_completed(futures), 1):
                row = future.result()
                analysis.append(row)
                print(json.dumps({'event': 'view', 'done': number, 'total': len(tasks),
                                  'wall_seconds': round(time.monotonic() - started, 1), **row},
                                 ensure_ascii=False), flush=True)

    reference = load_reference(ROOT)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'video_dir': str(args.video_dir),
              'pair_count': len(pairs), 'analysis': analysis, 'games': {}}
    for game in games:
        rows = combine_all(pairs, args.run_dir, game, reference, args.recompute_best_estimate)
        counts = {label: sum(row['decision'] == label for row in rows) for label in ('IN', 'OUT', 'REVIEW')}
        compared = [row for row in rows if row['agrees_with_ai_reference'] is not None]
        report['games'][game] = {
            'counts': counts,
            'ai_reference_agreement': {'compared': len(compared),
                                       'agree': sum(row['agrees_with_ai_reference'] for row in compared)},
            'pairs': rows}
        print(json.dumps({'event': 'summary', 'game': game, 'counts': counts,
                          'ai_reference_agreement': report['games'][game]['ai_reference_agreement']},
                         ensure_ascii=False), flush=True)
        for row in rows:
            print(json.dumps({'game': game, 'pair': row['pair_key'], 'decision': row['decision'],
                              'kind': row['decision_kind'], 'ref': row['ai_reference_label'],
                              'A': row['views']['A']['effective'], 'B': row['views']['B']['effective'],
                              'codes': row['reason_codes']}, ensure_ascii=False), flush=True)
    path = args.report or (args.run_dir / 'report.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'report', 'path': str(path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
