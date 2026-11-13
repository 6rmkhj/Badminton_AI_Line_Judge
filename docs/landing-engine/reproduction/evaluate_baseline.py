#!/usr/bin/env python3
"""Frozen AI first-landing-vicinity benchmark, never physical contact accuracy.

Examples (run each variant from an immutable source checkout):
  python scripts/evaluate_landing_engine.py --mode full --repo /tmp/landing-baseline --variant baseline
  python scripts/evaluate_landing_engine.py --mode replay --repo /tmp/landing-final --variant final-replay --baseline runs/landing-engine/baseline
  python scripts/evaluate_landing_engine.py --mode score --variant baseline

Inference receives only source video and configuration, never the references.
Replay reuses tracks/calibration and measures actual source pixels. Its runtime
is explicitly partial; accepted changes still require fresh complete execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REUSED_MODULES = ('media.py', 'detection.py', 'tracking.py', 'calibration.py',
                  'observation_support.py', 'learning.py', 'gpu_candidates.py', 'inference.py')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_hashes(repo):
    return {path.relative_to(repo).as_posix(): sha(path) for path in sorted((Path(repo) / 'badminton').glob('*.py'))}


def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(valid_number(v) for v in value)


def scorable(reference):
    interval = reference.get('landing_frame_interval')
    return (reference.get('contact_visibility') == 'visible'
            and isinstance(interval, list) and len(interval) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in interval)
            and interval[0] <= interval[1] and point(reference.get('landing_point_px'))
            and point(reference.get('coordinate_size')) and min(reference['coordinate_size']) > 0
            and valid_number(reference.get('tolerance_px')) and reference['tolerance_px'] > 0)


def compare(candidate, result, reference):
    """Identical coordinate/interval convention to the frozen generalization audit."""
    if not candidate or not scorable(reference):
        return None
    value = candidate.get('candidate_px')
    width, height = (result.get('analysis') or {}).get('width'), (result.get('analysis') or {}).get('height')
    frame = candidate.get('frame_index')
    if not point(value) or not all(valid_number(v) and v > 0 for v in (width, height)) or not valid_number(frame):
        return None
    size = reference['coordinate_size']
    predicted = [value[0] * size[0] / width, value[1] * size[1] / height]
    spatial = math.dist(predicted, reference['landing_point_px'])
    start, end = reference['landing_frame_interval']
    frame_error = max(start - frame, frame - end, 0)
    extra = reference.get('frame_tolerance', 0)
    if not valid_number(extra) or extra < 0:
        raise ValueError('Invalid frozen extra frame tolerance')
    return {'distance_px': spatial, 'frame_error': frame_error,
            'reference_coordinate_point': predicted,
            'within_spatial_tolerance': spatial <= reference['tolerance_px'],
            'within_frame_tolerance': frame_error <= extra,
            'matches_first_vicinity': spatial <= reference['tolerance_px'] and frame_error <= extra}


def candidate_kind(candidate, result):
    if not candidate:
        return 'no_candidate'
    explicit = (candidate.get('image_motion_event') or {}).get('point_kind')
    if explicit in {'observed_object_center', 'inferred_object_center'}:
        return explicit
    if 'occluded' in str(candidate.get('method', '')):
        return 'inferred_object_center'
    xy = candidate.get('candidate_px')
    for track in result.get('tracks') or []:
        if str(track.get('track_id')) != str(candidate.get('track_id')):
            continue
        for observation in track.get('observations') or []:
            actual = [observation.get('x'), observation.get('y')]
            if (observation.get('frame_index') == candidate.get('frame_index') and observation.get('observed') is not False
                    and not observation.get('interpolated') and point(xy) and point(actual) and math.dist(xy, actual) < .001):
                return 'observed_object_center'
    return 'unspecified_object_center'


def evaluate_case(row, result):
    if result.get('source_sha256') != row['source_sha256']:
        raise ValueError(f"Result source SHA mismatch: {row['key']}")
    if result.get('assistance_used') or result.get('automatic_evaluation_eligible') is False:
        raise ValueError(f"Assisted result excluded: {row['key']}")
    reference = row['reference']
    selected = result.get('selected_landing')
    predicted = compare(selected, result, reference)
    candidates = result.get('benchmark_ranked_landings', result.get('landings')) or []
    comparisons = [compare(candidate, result, reference) for candidate in candidates]
    match_ranks = [i + 1 for i, comparison in enumerate(comparisons) if comparison and comparison['matches_first_vicinity']]
    eligible = scorable(reference)
    category = ('reference_unscorable' if not eligible else 'abstention_no_candidate' if not selected else
                'prediction_geometry_unavailable' if predicted is None else
                'correct_first_vicinity' if predicted['matches_first_vicinity'] else
                'wrong_target_or_spatial_localization' if not predicted['within_spatial_tolerance'] else
                'time_mismatch')
    decision = result.get('decision') or {}
    return {'key': row['key'], 'cohort': row['cohort'], 'group_id': row['group_id'],
            'source_sha256': row['source_sha256'], 'reference_sha256': row['reference_sha256'],
            'reference_visibility': reference.get('contact_visibility'), 'localization_scorable': eligible,
            'selection_category': category, 'selected_comparison': predicted,
            'top1_match': bool(predicted and predicted['matches_first_vicinity']) if eligible else None,
            'recall_at_6': any(rank <= 6 for rank in match_ranks) if eligible else None,
            'recall_at_12': any(rank <= 12 for rank in match_ranks) if eligible else None,
            'matching_candidate_ranks': match_ranks, 'candidate_count': len(candidates),
            'candidate_kind': candidate_kind(selected, result), 'contact_observed': bool(selected and selected.get('contact_observed')),
            'selected': {k: selected.get(k) for k in ('event_id', 'track_id', 'frame_index', 'time', 'candidate_px', 'method')} if selected else None,
            'automatic_decision': decision.get('decision'), 'decision_reason_codes': sorted(set(decision.get('reason_codes') or [])),
            'selected_reason_codes': sorted(set((selected or {}).get('reason_codes') or [])),
            'timing_scope': (result.get('benchmark_execution') or {}).get('mode', 'unspecified'),
            'elapsed_seconds': (result.get('timings') or {}).get('total_seconds')}


def summarize(rows):
    eligible = [row for row in rows if row['localization_scorable']]
    errors = [row['selected_comparison'] for row in eligible if row['selected_comparison']]
    return {'video_count': len(rows), 'group_count': len({row['group_id'] for row in rows}),
            'scorable_count': len(eligible), 'excluded_count': len(rows) - len(eligible),
            'top1_match_count': sum(bool(row['top1_match']) for row in eligible),
            'recall_at_6_count': sum(bool(row['recall_at_6']) for row in eligible),
            'recall_at_12_count': sum(bool(row['recall_at_12']) for row in eligible),
            'top1_fraction': sum(bool(row['top1_match']) for row in eligible) / len(eligible) if eligible else None,
            'median_center_error_px': median(row['distance_px'] for row in errors) if errors else None,
            'median_interval_frame_error': median(row['frame_error'] for row in errors) if errors else None,
            'error_count': len(errors), 'candidate_abstention_count': sum(row['selected'] is None for row in rows),
            'candidate_kind_counts': dict(Counter(row['candidate_kind'] for row in rows)),
            'matched_candidate_kind_counts': dict(Counter(row['candidate_kind'] for row in eligible if row['top1_match'])),
            'physical_contact_observed_count': sum(row['contact_observed'] for row in rows),
            'automatic_decision_counts': dict(Counter(row['automatic_decision'] for row in rows)),
            'review_reason_counts': dict(Counter(reason for row in rows if row['automatic_decision'] == 'REVIEW' for reason in row['decision_reason_codes'])),
            'selected_reason_counts': dict(Counter(reason for row in rows for reason in row['selected_reason_codes'])),
            'selection_category_counts': dict(Counter(row['selection_category'] for row in rows)),
            'independent_contact_accuracy': None, 'independent_line_judgment_accuracy': None}


def validate_inputs(protocol, assets, video_dir, keys=(), require_sources=True):
    rows = protocol['videos']
    if keys:
        unknown = set(keys) - {r['key'] for r in rows}
        if unknown:
            raise ValueError('Unknown keys: ' + ', '.join(sorted(unknown)))
        rows = [r for r in rows if r['key'] in keys]
    checked = []
    for original in rows:
        row = deepcopy(original)
        reference_path = assets / row['reference_path']
        if not reference_path.is_file() or sha(reference_path) != row['reference_sha256']:
            raise ValueError(f"Frozen reference SHA mismatch or missing: {reference_path}")
        if row.get('reference_format', 'json') == 'json':
            reference = read(reference_path)
            if reference.get('source_sha256') != row['source_sha256']:
                raise ValueError(f"Reference source SHA mismatch: {row['key']}")
            row['reference'] = reference
        source = video_dir / row['file']
        if require_sources and (not source.is_file() or sha(source) != row['source_sha256']):
            raise ValueError(f"Source missing or SHA mismatch: {source}")
        row['source_path'] = str(source.resolve())
        checked.append(row)
    return checked


def unassisted(config):
    if ((config.get('identity') or {}).get('hit_anchors') or (config.get('identity') or {}).get('event_window_seconds')
            or config.get('request_context') or config.get('court_hint')
            or (config.get('detector') or {}).get('refinement_regions')
            or any((config.get('events') or {}).get(key) is not None for key in ('target_track_id', 'event_window_seconds'))
            or (config.get('events') or {}).get('review_refinement_regions')):
        raise ValueError('Benchmark requires unassisted whole-video configuration')


def frame_digest(frames):
    value = hashlib.sha256()
    for frame in frames:
        value.update(str(frame.shape).encode())
        value.update(frame.tobytes())
    return value.hexdigest()


def decoded_frames(result, source):
    from badminton.media import load_frames, probe
    analysis = result['analysis']
    if analysis.get('preprocessing') != 'opencv_source_bgr_area_v1':
        raise ValueError('Replay requires original source pixels, not preview input')
    metadata = probe(source)
    timestamps = result['timestamps']
    if len(timestamps) != metadata['frame_count'] or any(abs(a-b) > .0001 for a,b in zip(timestamps, metadata['timestamps'])):
        raise ValueError('Original-source timestamp/frame contract changed')
    return load_frames(source, analysis['frame_count'], result['config'].get('max_frames', 3000),
                       resize_to=(analysis['width'], analysis['height']))


def replay(base, source, output, config, code):
    from badminton.events import estimate_landings, refine_source_contacts, prioritize_review_candidates
    from badminton.identity import apply_identity
    from badminton.decision import decide
    from badminton.media import SourceFrameSequence
    from badminton.pipeline import aggregate_decision, implementation_fingerprint
    previous = base.get('benchmark_execution') or {}
    old_code = previous.get('source_files') or {}
    if any(old_code.get('badminton/' + name) != code.get('badminton/' + name) for name in REUSED_MODULES):
        raise ValueError('Replay requires identical detector/tracker/calibration/preprocessing code')
    if {k:v for k,v in base['config'].items() if k != 'events'} != {k:v for k,v in config.items() if k != 'events'}:
        raise ValueError('Replay permits event configuration changes only')
    started = time.perf_counter()
    frames = decoded_frames(base, source)
    frames_sha = frame_digest(frames)
    if frames_sha != previous.get('analysis_frames_sha256'):
        raise ValueError('Replay source analysis pixels differ from baseline')
    result = deepcopy(base)
    tracks, timestamps, calibration = result['tracks'], result['timestamps'], result['calibration']
    event_config = dict(config.get('events') or {})
    resources = config.get('resource_settings') or {}
    event_config.update(frame_preprocessing=result['analysis']['preprocessing'],
                        inference_device=resources.get('device', config.get('inference_device', 'cpu')),
                        gpu_memory_percent=resources.get('gpu_memory_percent', 50),
                        gpu_parallel_views=resources.get('parallel_views', 1), review_refinement_regions=[])
    landings = estimate_landings(tracks, frames, timestamps, calibration, event_config)
    landings = apply_identity(landings, tracks, timestamps, calibration, config.get('identity') or {})
    if event_config.get('source_contact_enabled', True):
        with SourceFrameSequence(source, len(frames)) as originals:
            landings = refine_source_contacts(landings, originals, timestamps,
                analysis_size=(frames[0].shape[1], frames[0].shape[0]), config=event_config, output_dir=output)
    for i, landing in enumerate(landings):
        landing['event_id'] = f'event_{i:03d}'
        landing['decision'] = decide(landing, calibration, config.get('rules'))
    ordered = prioritize_review_candidates(landings, [])
    result.update(config=config, implementation_fingerprint=implementation_fingerprint(), landings=landings,
                  selected_landing=ordered[0] if ordered else None,
                  decision=aggregate_decision([landing['decision'] for landing in landings]), artifacts={},
                  timings={'total_seconds': time.perf_counter()-started, 'timing_scope': 'replay_events_identity_native_contact_decision_only'})
    return result, frames_sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('full', 'replay', 'score'), default='full')
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--assets-root', type=Path, default=ROOT)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/landing-engine/protocol.json')
    parser.add_argument('--video-dir', type=Path, default=ROOT/'video')
    parser.add_argument('--output-root', type=Path, default=ROOT/'runs/landing-engine')
    parser.add_argument('--variant', required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--keys', nargs='+', default=[])
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.variant):
        parser.error('variant must be a safe folder name')
    protocol = read(args.protocol)
    rows = validate_inputs(protocol, args.assets_root.resolve(), args.video_dir.resolve(), args.keys)
    if args.preflight_only:
        print(json.dumps({'status':'preflight_passed', 'videos':len(rows), 'protocol_sha256':sha(args.protocol)}))
        return 0
    run = args.output_root.resolve() / args.variant
    if args.mode == 'score':
        if not (run/'run.json').is_file():
            raise ValueError('Missing frozen run manifest')
        receipt = read(run/'run.json')
        if receipt['protocol_sha256'] != sha(args.protocol):
            raise ValueError('Scoring protocol differs from execution')
    else:
        args.repo = args.repo.resolve()
        sys.path.insert(0, str(args.repo))
        from badminton.pipeline import analyze_video, implementation_fingerprint
        import badminton
        if Path(badminton.__file__).resolve().parent != args.repo/'badminton':
            raise ValueError('Imported engine is not the requested source checkout')
        import cv2
        config = read(args.config or args.repo/'configs/poc.json')
        unassisted(config)
        cv2.setNumThreads(config.get('opencv_threads', 2))
        code = source_hashes(args.repo)
        receipt = {'schema_version':1, 'mode':args.mode, 'variant':args.variant, 'repo':str(args.repo),
                   'protocol_sha256':sha(args.protocol), 'evaluator_sha256':sha(Path(__file__)),
                   'source_files':code, 'source_tree_sha256':digest(code), 'config':config,
                   'config_sha256':digest(config), 'implementation_fingerprint':implementation_fingerprint(),
                   'keys':[row['key'] for row in protocol['videos']], 'references_used_for_inference':False}
        if args.mode == 'replay':
            if not args.baseline:
                parser.error('--baseline is required for replay')
            if args.baseline.resolve() == run:
                raise ValueError('Replay output must not overwrite baseline')
            receipt['baseline_run_sha256'] = sha(args.baseline/'run.json')
        existing = run/'run.json'
        if existing.exists() and read(existing) != receipt:
            raise ValueError('Variant already contains a different code/config/input freeze; choose a new --variant')
        write(existing, receipt)

        def process(row):
            destination = run / row['key']
            result_path = destination/'result.json'
            if result_path.exists():
                cached = read(result_path)
                if ((cached.get('benchmark_execution') or {}).get('run_receipt_sha256') == digest(receipt)
                        and cached.get('source_sha256') == row['source_sha256']):
                    return row['key'], 'cached'
                raise ValueError('Unrecognized existing result: ' + str(result_path))
            destination.mkdir(parents=True, exist_ok=True)
            if source_hashes(args.repo) != code:
                raise ValueError('Source tree changed before execution')
            if args.mode == 'full':
                result = analyze_video(row['source_path'], destination, config)
                frames = decoded_frames(result, row['source_path'])
                frames_sha = frame_digest(frames)
                del frames
            else:
                baseline_path = args.baseline / row['key'] / 'result.json'
                base = read(baseline_path)
                if base.get('source_sha256') != row['source_sha256']:
                    raise ValueError('Replay baseline source mismatch')
                result, frames_sha = replay(base, row['source_path'], destination, config, code)
            if source_hashes(args.repo) != code:
                raise ValueError('Source tree changed during execution; discard this mixed-code variant')
            from badminton.events import prioritize_review_candidates
            result['benchmark_ranked_landings'] = prioritize_review_candidates(result.get('landings') or [], [])
            result['benchmark_execution'] = {'mode':args.mode, 'run_receipt_sha256':digest(receipt),
                'source_files':code, 'analysis_frames_sha256':frames_sha, 'reference_labels_read':False,
                'baseline_result_sha256':sha(baseline_path) if args.mode == 'replay' else None}
            write(result_path, result)
            return row['key'], 'completed'

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process, row) for row in rows]
            for future in as_completed(futures):
                key, status = future.result()
                print(json.dumps({'key':key, 'status':status}, ensure_ascii=False), flush=True)
    records = []
    for row in rows:
        result_path = run/row['key']/'result.json'
        result = read(result_path)
        if (result.get('benchmark_execution') or {}).get('run_receipt_sha256') != digest(receipt):
            raise ValueError('Result does not match frozen run receipt: ' + row['key'])
        record = evaluate_case(row, result)
        record.update(result_sha256=sha(result_path), result_path=str(result_path))
        records.append(record)
    report = {'schema_version':1, 'scope':protocol['scope'], 'protocol_sha256':sha(args.protocol),
              'run':receipt, 'summary':summarize(records),
              'cohorts':{name:summarize([r for r in records if r['cohort']==name]) for name in sorted({r['cohort'] for r in records})},
              'cases':records}
    target = run/('evaluation.json' if len(records)==len(receipt['keys']) else 'evaluation-subset-'+digest([row['key'] for row in rows])[:12]+'.json')
    write(target, report)
    print(json.dumps({'report':str(target), 'summary':report['summary']}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
