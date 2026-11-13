"""Audit selections against frozen AI source-video references.

This is a failure-analysis tool, not a measurement of independent contact or
line-judgment accuracy. Development is the default. Holdout files remain sealed
unless explicit release arguments validate a saved commit/fingerprint freeze.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def config_fingerprint(config):
    if not isinstance(config, dict):
        return None
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(finite(v) for v in value)


def interval(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in value)
            and value[0] <= value[1])


def tolerance(reference):
    # A provided frame interval already expresses reference uncertainty. A
    # missing extra tolerance means zero additional frames, never an expanded
    # interval chosen after seeing predictions.
    spatial, temporal = reference.get('tolerance_px'), reference.get('frame_tolerance', 0)
    if not finite(spatial) or spatial <= 0 or not finite(temporal) or temporal < 0:
        return None
    return float(spatial), float(temporal)


def prediction_point(candidate, result, reference):
    value = candidate.get('candidate_px')
    size = reference.get('coordinate_size')
    analysis = result.get('analysis') or {}
    width, height = analysis.get('width'), analysis.get('height')
    if not point(value) or not point(size) or not all(finite(v) and v > 0 for v in (width, height, *size)):
        return None
    return [value[0] * size[0] / width, value[1] * size[1] / height]


def comparison(candidate, result, reference, expected=None):
    expected = expected or reference
    target = expected.get('landing_point_px')
    frames = expected.get('landing_frame_interval')
    allowed = tolerance(reference)
    position = prediction_point(candidate, result, reference)
    frame = candidate.get('frame_index')
    if not point(target) or not interval(frames) or not allowed or position is None or not finite(frame):
        return None
    spatial, temporal = allowed
    distance = math.hypot(position[0] - target[0], position[1] - target[1])
    frame_error = max(frames[0] - frame, frame - frames[1], 0)
    return {'frame_index': frame, 'reference_coordinate_point': position,
            'distance_px': round(distance, 4), 'frame_error': frame_error,
            'within_spatial_tolerance': distance <= spatial,
            'within_frame_tolerance': frame_error <= temporal,
            'matches_first_vicinity': distance <= spatial and frame_error <= temporal}


def visible_scorable(reference):
    return (reference.get('contact_visibility') == 'visible'
            and interval(reference.get('landing_frame_interval'))
            and point(reference.get('landing_point_px')) and tolerance(reference) is not None
            and point(reference.get('coordinate_size')) and all(v > 0 for v in reference['coordinate_size']))


def frame_point_distance(observation, expected, result, reference):
    position = prediction_point({'candidate_px': [observation.get('x'), observation.get('y')]}, result, reference)
    if position is None or not point(expected):
        return math.inf
    return math.hypot(position[0] - expected[0], position[1] - expected[1])


def continuity_audit(track, selected_frame):
    """Expose observed association risks without asserting an object's identity."""
    observations = sorted(track.get('observations', []), key=lambda o: o['frame_index'])

    def summary(items):
        radii = [o.get('radius') for o in items if finite(o.get('radius')) and o['radius'] > 0]
        polarities = Counter()
        for item in items:
            method = str(item.get('method', '')).lower()
            polarity = str(item.get('appearance_polarity', item.get('polarity', item.get('contrast_polarity', '')))).lower()
            if 'dark' in method or polarity in {'dark', 'negative'}:
                polarities['dark'] += 1
            elif 'bright' in method or polarity in {'bright', 'positive'}:
                polarities['bright'] += 1
            else:
                polarities['unknown'] += 1
        if items:
            span = math.hypot(max(o['x'] for o in items) - min(o['x'] for o in items),
                              max(o['y'] for o in items) - min(o['y'] for o in items))
        else:
            span = 0.
        return {'observation_count': len(items), 'median_radius_px': median(radii) if radii else None,
                'possible_occluder_overlap_count': sum(o.get('possible_occluder_overlap') is True for o in items),
                'proposal_polarity_counts': dict(polarities), 'motion_span_analysis_px': round(span, 4)}

    entries = []
    for index in range(3, len(observations) - 2):
        item, previous = observations[index], observations[index - 1]
        if item.get('possible_occluder_overlap') is not True or previous.get('possible_occluder_overlap') is True:
            continue
        if not finite(selected_frame) or item['frame_index'] > selected_frame:
            continue
        before, after = summary(observations[max(0, index - 6):index]), summary(observations[index:index + 6])
        if not before['median_radius_px'] or not after['median_radius_px']:
            continue
        radius_ratio = after['median_radius_px'] / before['median_radius_px']
        before_known = before['proposal_polarity_counts'].get('bright', 0) + before['proposal_polarity_counts'].get('dark', 0)
        after_known = after['proposal_polarity_counts'].get('bright', 0) + after['proposal_polarity_counts'].get('dark', 0)
        polarity_change = (before_known >= 3 and after_known >= 3
            and ((before['proposal_polarity_counts'].get('bright', 0) / before_known >= .75
                  and after['proposal_polarity_counts'].get('dark', 0) / after_known >= .5)
                 or (before['proposal_polarity_counts'].get('dark', 0) / before_known >= .75
                     and after['proposal_polarity_counts'].get('bright', 0) / after_known >= .5)))
        sustained_overlap = after['possible_occluder_overlap_count'] >= max(2, after['observation_count'] * .6)
        sustained_approach = before['motion_span_analysis_px'] >= max(12., before['median_radius_px'] * 4)
        size_change = radius_ratio >= 1.8 or radius_ratio <= .55
        entries.append({'entry_frame': item['frame_index'], 'before': before, 'after': after,
                        'radius_ratio_after_over_before': round(radius_ratio, 4),
                        'proposal_polarity_change': polarity_change, 'size_change': size_change,
                        'sustained_overlap': sustained_overlap, 'sustained_approach_motion': sustained_approach,
                        'association_risk': sustained_overlap and sustained_approach and (size_change or polarity_change)})
    return {'identity_switch_or_occluder_extension_suspected': any(e['association_risk'] for e in entries),
            'occluder_entry_observations': entries,
            'limits': 'Appearance/polarity changes and possible person overlap are diagnostic association risks, not independently confirmed identity switches.'}


def diagnose(result, detection_path, reference, chosen_comparison):
    """Separate proposal coverage from association and final candidate ranking."""
    allowed = tolerance(reference)
    if allowed is None:
        return {'failure_stage': 'reference_unscorable'}
    spatial, temporal = allowed
    frames, expected = reference['landing_frame_interval'], reference['landing_point_px']
    candidates = result.get('landings') or []
    matching = [c for c in candidates if (comparison(c, result, reference) or {}).get('matches_first_vicinity')]
    tracks = result.get('tracks') or []
    track_contact_hits = []
    for track in tracks:
        hits = [o for o in track.get('observations', [])
                if finite(o.get('frame_index')) and frames[0] - temporal <= o['frame_index'] <= frames[1] + temporal
                and (frame_point_distance(o, expected, result, reference)) <= spatial]
        if hits:
            track_contact_hits.append(track['track_id'])
    selected = result.get('selected_landing') or {}
    selected_track = next((t for t in tracks if str(t.get('track_id')) == str(selected.get('track_id'))), {})
    continuity = continuity_audit(selected_track, selected.get('frame_index'))
    detection_artifact_available = detection_path.is_file()
    detections = load(detection_path) if detection_artifact_available else []
    raw_contact_hits = []
    for record in detections:
        f = record.get('frame_index')
        if finite(f) and frames[0] - temporal <= f <= frames[1] + temporal:
            hits = [d for d in record.get('detections', [])
                    if (frame_point_distance(d, expected, result, reference)) <= spatial]
            if hits:
                raw_contact_hits.append({'frame_index': f, 'candidate_count': len(hits)})
    trajectory = [p for p in reference.get('trajectory_points', [])
                  if isinstance(p, dict) and finite(p.get('frame_index')) and finite(p.get('x')) and finite(p.get('y'))]

    def trajectory_hits(rows):
        found = []
        for expected_point in trajectory:
            if any(finite(o.get('frame_index')) and abs(o['frame_index'] - expected_point['frame_index']) <= temporal
                   and (frame_point_distance(o, [expected_point['x'], expected_point['y']], result, reference)) <= spatial
                   for o in rows):
                found.append(expected_point['frame_index'])
        return found

    raw_observations = [dict(d, frame_index=r['frame_index']) for r in detections for d in r.get('detections', [])]
    raw_trajectory_hits = trajectory_hits(raw_observations)
    track_hits = [(track.get('track_id'), trajectory_hits(track.get('observations', []))) for track in tracks]
    best_track_id, best_track_hits = max(track_hits, key=lambda pair: len(pair[1]), default=(None, []))
    selected_track_hits = trajectory_hits(selected_track.get('observations', []))
    if chosen_comparison and chosen_comparison['matches_first_vicinity']:
        failure = 'none_selected_first_vicinity'
    elif matching:
        failure = 'candidate_rank_wrong'
    elif not detection_artifact_available:
        failure = 'detection_artifact_unavailable_for_diagnosis'
    elif not raw_contact_hits:
        failure = 'detector_contact_candidate_absent'
    elif not track_contact_hits:
        failure = 'candidate_not_retained_in_track'
    elif len(raw_trajectory_hits) >= 3 and len(best_track_hits) < len(raw_trajectory_hits) * .5:
        failure = 'tracking_fragmentation_or_association'
    else:
        failure = 'event_proposal_missing_or_temporally_mislocalized'
    causal = ('identity_switch_or_occluder_extension_suspected'
              if failure != 'none_selected_first_vicinity' and continuity['identity_switch_or_occluder_extension_suspected']
              else failure)
    return {'failure_stage': failure, 'causal_failure_category': causal,
            'matching_first_vicinity_event_ids': [c.get('event_id') for c in matching],
            'matching_first_vicinity_candidate_count': len(matching),
            'detection_artifact_available': detection_artifact_available,
            'raw_contact_hit_frames': raw_contact_hits if detection_artifact_available else None,
            'contact_hit_track_ids': track_contact_hits,
            'reference_trajectory_point_count': len(trajectory),
            'raw_trajectory_hit_count': len(raw_trajectory_hits) if detection_artifact_available else None,
            'raw_trajectory_hit_frames': raw_trajectory_hits if detection_artifact_available else None,
            'best_track_trajectory_hit_count': len(best_track_hits), 'best_track_trajectory_hit_frames': best_track_hits,
            'best_track_trajectory_id': best_track_id,
            'selected_track_trajectory_hit_count': len(selected_track_hits),
            'selected_track_trajectory_hit_frames': selected_track_hits,
            'selected_track': {key: selected_track.get(key) for key in ('track_id', 'method', 'identity_ambiguous', 'observation_count')},
            'selected_track_has_flight_hypothesis': bool(selected_track.get('flight_hypothesis')),
            'selected_track_continuity': continuity,
            'diagnostic_limits': 'Spatial/temporal proximity to an AI visual reference does not prove object identity or physical contact.'}


def audit_video(row, run, references, freeze_dir, *, split='development', release=None):
    if row.get('split') != split or (split != 'development' and not release):
        raise ValueError('Holdout access requires an explicit release with a verified saved code freeze.')
    key = row['key']
    record = {'key': key, 'group': row['group'], 'view': row['view'], 'split': split,
              'source_sha256': row['sha256']}
    reference_path = references / (key + '.json')
    if not reference_path.is_file():
        return record | {'status': 'awaiting_reference', 'selection_category': 'not_audited'}
    reference_bytes = reference_path.read_bytes()
    try:
        reference = json.loads(reference_bytes)
    except json.JSONDecodeError:
        return record | {'status': 'awaiting_reference_write_completion', 'selection_category': 'not_audited'}
    reference_hash = hashlib.sha256(reference_bytes).hexdigest()
    frozen = freeze_dir / key
    frozen.mkdir(parents=True, exist_ok=True)
    original = frozen / (reference_hash + '.json')
    if not original.exists():
        original.write_bytes(reference_bytes)
    prior_versions = sorted(p.stem for p in frozen.glob('*.json'))
    record.update(reference_path=str(reference_path), reference_sha256=reference_hash,
                  reference_snapshot=str(original), contact_visibility=reference.get('contact_visibility'),
                  reference_interval=reference.get('landing_frame_interval'), reference_point=reference.get('landing_point_px'),
                  coordinate_size=reference.get('coordinate_size'), tolerance_px=reference.get('tolerance_px'),
                  frame_tolerance=reference.get('frame_tolerance'), reference_version_count=len(prior_versions),
                  effective_frame_tolerance=(tolerance(reference) or (None, None))[1],
                  localization_scorable=visible_scorable(reference))
    if reference.get('source_sha256') != row['sha256']:
        return record | {'status': 'reference_source_mismatch', 'selection_category': 'not_audited'}
    if len(prior_versions) > 1 and not reference.get('revision_visual_reason'):
        return record | {'status': 'reference_revision_requires_visual_reason', 'selection_category': 'not_audited'}
    result_path = run / key / 'result.json'
    if (run / key / 'error.txt').is_file():
        return record | {'status': 'analysis_failed', 'selection_category': 'not_audited'}
    if not result_path.is_file():
        error_path = run / key / 'error.txt'
        return record | {'status': 'analysis_failed' if error_path.is_file() else 'awaiting_analysis',
                         'selection_category': 'not_audited'}
    result = load(result_path)
    if release and result.get('implementation_fingerprint') != release['run_implementation_fingerprint']:
        return record | {'status': 'analysis_frozen_fingerprint_mismatch', 'selection_category': 'not_audited'}
    if release and config_fingerprint(result.get('config')) != release['analysis_config_sha256']:
        return record | {'status': 'analysis_frozen_configuration_mismatch', 'selection_category': 'not_audited'}
    if result.get('source_sha256') != row['sha256']:
        return record | {'status': 'analysis_source_mismatch', 'selection_category': 'not_audited'}
    validation_run = result.get('validation_run') or {}
    if split == 'holdout' and validation_run.get('detector_stage_reused'):
        return record | {'status': 'holdout_requires_fresh_detector_run', 'selection_category': 'not_audited'}
    selected = result.get('selected_landing')
    predicted = comparison(selected, result, reference) if selected else None
    category = 'reference_unscorable'
    if reference.get('contact_visibility') != 'visible':
        category = 'no_observable_ground_truth'
    elif visible_scorable(reference):
        if not selected:
            category = 'abstention_no_candidate'
        elif predicted is None:
            category = 'prediction_geometry_unavailable'
        elif predicted['matches_first_vicinity']:
            category = 'correct_first_vicinity'
        elif any((comparison(selected, result, reference, later) or {}).get('matches_first_vicinity')
                 for later in reference.get('later_bounces', [])):
            category = 'late_bounce_reference_match'
        elif not predicted['within_spatial_tolerance']:
            category = 'wrong_target_or_spatial_localization'
        elif selected['frame_index'] > reference['landing_frame_interval'][1] + tolerance(reference)[1]:
            category = 'late_time_mismatch'
        else:
            category = 'early_time_mismatch'
    decision = (result.get('decision') or {}).get('decision')
    record.update(status='audited', result_path=str(result_path), implementation_fingerprint=result.get('implementation_fingerprint'),
                  analysis_config_sha256=config_fingerprint(result.get('config')),
                  pipeline_version=result.get('pipeline_version'), selection_category=category,
                  validation_mode='detector_stage_reuse' if validation_run.get('detector_stage_reused') else 'full_pipeline',
                  validation_run=validation_run or None,
                  timing_scope=(result.get('timings') or {}).get('timing_scope', 'full_pipeline'),
                  automatic_decision=decision, decision_abstention=decision == 'REVIEW',
                  candidate_abstention=not bool(selected), selected_comparison=predicted,
                  selected={name: selected.get(name) for name in ('event_id', 'track_id', 'frame_index', 'time', 'candidate_px', 'contact_observed', 'method', 'reason_codes')}
                           if selected else None,
                  elapsed_seconds=(result.get('timings') or {}).get('total_seconds'))
    if visible_scorable(reference):
        record['diagnosis'] = diagnose(result, run / key / 'detections.json', reference, predicted)
        experiment_path = references.parent / 'audit_experiments' / run.name / (key + '-event-cap.json')
        if experiment_path.is_file():
            experiment = load(experiment_path)
            if (experiment.get('source_sha256') == row['sha256']
                    and experiment.get('reference_sha256') == reference_hash
                    and experiment.get('implementation_fingerprint') == result.get('implementation_fingerprint')
                    and experiment.get('matching_first_vicinity_candidates')
                    and not (predicted or {}).get('matches_first_vicinity')):
                record['diagnosis']['pre_experiment_failure_stage'] = record['diagnosis']['failure_stage']
                record['diagnosis']['failure_stage'] = 'event_candidate_budget_rank_loss'
                record['diagnosis']['event_cap_experiment_path'] = str(experiment_path)
                record['diagnosis']['event_cap_experiment'] = experiment
                if record['diagnosis']['causal_failure_category'] != 'identity_switch_or_occluder_extension_suspected':
                    record['diagnosis']['causal_failure_category'] = 'event_candidate_budget_rank_loss'
    return record


def metrics(records):
    audited = [r for r in records if r['status'] == 'audited']
    scorable = [r for r in audited if r.get('localization_scorable')]
    matched = [r for r in scorable if r['selection_category'] == 'correct_first_vicinity']
    return {'declared_videos': len(records),
            'declared_development_videos': sum(r['split'] == 'development' for r in records),
            'declared_holdout_videos': sum(r['split'] == 'holdout' for r in records),
            'audited_outputs': len(audited),
            'pending_or_invalid_outputs': len(records) - len(audited),
            'status_counts': dict(Counter(r['status'] for r in records)),
            'reference_visibility_counts': dict(Counter(r.get('contact_visibility', 'reference_pending') for r in records)),
            'selection_category_counts': dict(Counter(r['selection_category'] for r in records)),
            'diagnostic_failure_stage_counts': dict(Counter(r['diagnosis']['failure_stage'] for r in audited if 'diagnosis' in r)),
            'causal_failure_category_counts': dict(Counter(r['diagnosis']['causal_failure_category'] for r in audited if 'diagnosis' in r)),
            'automatic_decision_counts': dict(Counter(r.get('automatic_decision') for r in audited)),
            'first_vicinity_match_count': len(matched), 'first_vicinity_denominator': len(scorable),
            'first_vicinity_fraction': len(matched) / len(scorable) if scorable else None,
            'decision_abstention_count': sum(r.get('decision_abstention', False) for r in audited),
            'decision_abstention_denominator': len(audited),
            'candidate_abstention_count': sum(r.get('candidate_abstention', False) for r in audited),
            'candidate_abstention_denominator': len(audited),
            'independent_contact_accuracy': None, 'independent_line_judgment_accuracy': None}


def verify_holdout_release(freeze_path, run, manifest_path):
    """Validate saved freeze metadata before any holdout file is opened."""
    if not freeze_path.is_file():
        raise ValueError('A saved freeze manifest file is required for holdout release.')
    raw = freeze_path.read_bytes()
    freeze = json.loads(raw)
    if not isinstance(freeze, dict):
        raise ValueError('Freeze manifest must be a JSON object.')
    commit = freeze.get('commit', freeze.get('frozen_commit', freeze.get('source_commit')))
    fingerprint = freeze.get('implementation_fingerprint')
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Freeze manifest must include a full saved commit SHA.')
    if not isinstance(fingerprint, str) or not re.fullmatch(r'[0-9a-f]{64}', fingerprint):
        raise ValueError('Freeze manifest must include a full implementation_fingerprint SHA256.')
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if freeze.get('manifest_sha256', manifest_hash) != manifest_hash:
        raise ValueError('Freeze manifest does not match the predeclared video manifest.')
    frozen_runs = freeze.get('runs') or {}
    if not isinstance(frozen_runs, dict):
        raise ValueError('Freeze manifest runs must map run names to provenance objects.')
    run_entry = frozen_runs.get(run.name)
    if run_entry is None:
        run_entry = {'commit': commit, 'implementation_fingerprint': fingerprint}
    if (not isinstance(run_entry, dict) or not re.fullmatch(r'[0-9a-f]{40}', str(run_entry.get('commit', '')))
            or not re.fullmatch(r'[0-9a-f]{64}', str(run_entry.get('implementation_fingerprint', '')))):
        raise ValueError('Frozen run entry must name its commit and implementation fingerprint.')
    config_path = run / 'run-config.json'
    if not config_path.is_file():
        raise ValueError('The requested run must have saved run-config.json provenance.')
    run_config = load(config_path)
    if not isinstance(run_config, dict):
        raise ValueError('Saved run-config.json must contain a provenance object.')
    if run_config.get('implementation_fingerprint') != run_entry['implementation_fingerprint']:
        raise ValueError('Requested run fingerprint differs from the frozen code provenance.')
    profile_hash = config_fingerprint(run_config.get('config'))
    if profile_hash is None:
        raise ValueError('Saved run-config.json must preserve the actual analysis configuration.')
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if run_entry.get('run_config_sha256', config_hash) != config_hash:
        raise ValueError('Requested run configuration differs from the saved code/configuration freeze.')
    return {'freeze_manifest_path': str(freeze_path), 'freeze_manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'commit': commit, 'implementation_fingerprint': fingerprint,
            'run_commit': run_entry['commit'], 'run_implementation_fingerprint': run_entry['implementation_fingerprint'],
            'run_config_sha256': config_hash,
            'analysis_config_sha256': profile_hash,
            'predeclared_manifest_sha256': manifest_hash,
            'authorization': 'Explicit --split holdout --release-holdouts --freeze-manifest arguments'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/generalization'))
    parser.add_argument('--run', type=Path, default=Path('/workspace/generalization/baseline'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--key', action='append', default=[])
    parser.add_argument('--split', choices=('development', 'holdout'), default='development')
    parser.add_argument('--release-holdouts', action='store_true', help='Explicitly release the sealed holdout split only after code freeze.')
    parser.add_argument('--freeze-manifest', type=Path, help='Saved commit/fingerprint JSON; required with --release-holdouts.')
    args = parser.parse_args()
    manifest = load(args.root / 'manifest.json')
    release = None
    if args.split == 'holdout':
        if not args.release_holdouts or args.freeze_manifest is None:
            parser.error('Holdout access requires --release-holdouts and a saved --freeze-manifest commit/fingerprint file.')
        try:
            release = verify_holdout_release(args.freeze_manifest, args.run, args.root / 'manifest.json')
        except (ValueError, OSError, TypeError) as error:
            parser.error(str(error))
    elif args.release_holdouts:
        parser.error('--release-holdouts is valid only with --split holdout.')
    selected_rows = [r for r in manifest['videos'] if r['split'] == args.split]
    selected_keys = {r['key'] for r in selected_rows}
    if any(key not in selected_keys for key in args.key):
        parser.error('Only predeclared keys in the explicitly selected split may be audited.')
    if args.key:
        selected_rows = [r for r in selected_rows if r['key'] in args.key]
    for row in selected_rows:
        if Path(row['key']).name != row['key']:
            raise ValueError('Invalid development key')
    records = [audit_video(r, args.run, args.root / 'references', args.root / 'reference_snapshots',
                           split=args.split, release=release) for r in selected_rows]
    groups = defaultdict(list)
    for record in records:
        groups[record['group']].append(record)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'split': args.split,
              'run_directory': str(args.run), 'manifest_source_commit': manifest['source_commit'],
              'holdout_predictions_or_references_opened': bool(release),
              'holdout_release': release,
              'scope': 'AI visual source-review audit; vicinity agreement is not independently measured contact or line-judgment accuracy.',
              'reference_policy': manifest['reference_policy'],
              'metrics': metrics(records), 'by_capture_group': {g: metrics(rows) for g, rows in groups.items()},
              'videos': records}
    subset_suffix = '--' + '-'.join(sorted(args.key)) if args.key else ''
    output = args.output or args.root / ('audit-' + args.split + '-' + args.run.name + subset_suffix + '.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(output), 'metrics': report['metrics']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
