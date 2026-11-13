"""Isolated rejection rechecks on one exposed development case, never production.

Run once per --variant from the repository. No reserved reference is accessed.
The frozen v5 engine is imported from --repo; modifications exist only in this
process and are recorded together with the unchanged source/config hashes.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROTOTYPES = Path(__file__).parent / 'prototypes'
KEY = 'uhdhd-5-a'


def function_from_script(path, name, namespace):
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(definitions) == 1
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/tmp/landing-engine-v5'))
    parser.add_argument('--variant', choices=['control24', 'budget48', 'prefix', 'terminal', 'prefix-dark', 'terminal-dark'], required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    spec = importlib.util.spec_from_file_location('frozen_v5_evaluator', ROOT / 'scripts/evaluate_landing_engine.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    from badminton import landing_recovery as recovery, tracking, events
    cv2.setNumThreads(1)
    baseline_dir = ROOT / 'runs/landing-engine/baseline'
    baseline_path = baseline_dir / KEY / 'result.json'
    accepted_dir = ROOT / 'runs/landing-engine/minimal-v5'
    accepted_receipt = evaluator.read(accepted_dir / 'run.json')
    code = evaluator.source_hashes(args.repo)
    assert code == accepted_receipt['source_files']
    assert evaluator.sha(ROOT / 'scripts/evaluate_landing_engine.py') == accepted_receipt['evaluator_sha256']
    config = deepcopy(accepted_receipt['config'])
    assert config['events']['causal_first_event_ranking'] is True
    assert config['events']['native_rebound_guard'] is True
    assert config['events']['event_family_diversity'] is False
    base = evaluator.read(baseline_path)
    source = Path(base['source_path'])
    assert evaluator.sha(source) == base['source_sha256']
    records, detections_sha = evaluator.validated_detection_records(
        baseline_dir, KEY, base, evaluator.read(ROOT / 'docs/landing-engine/baseline-detections.json'))
    output = ROOT / 'runs/landing-engine/observation-audit/v5-rechecks' / args.variant
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise ValueError('Refusing to overwrite a completed experiment')

    original_approaches = recovery._approaches
    original_components = recovery._components
    original_extensions = tracking._rebound_extensions
    original_recover = recovery.recover_rebound_tracks
    original_estimate = events.estimate_landings
    original_competition = events.apply_competing_continuation_evidence
    terminal = args.variant.startswith('terminal')
    dark = args.variant.endswith('-dark')
    used_scripts = []
    selected_endpoints = []
    if args.variant not in {'control24', 'budget48'}:
        path = PROTOTYPES / 'prefix_probe.py'
        used_scripts.append(path)
        prefixes = function_from_script(path, 'prefixes', {
            'old_approaches': original_approaches, '_flight_fit': tracking._flight_fit, 'np': np, 'math': math})
        def traced_prefixes(tracks, budget):
            selected = prefixes(tracks, budget)
            selected_endpoints.extend({'track_id': track['track_id'], 'frame_index': observations[-1]['frame_index'],
                'point': [observations[-1]['x'], observations[-1]['y']]} for track, observations, _, _ in selected)
            return selected
        recovery._approaches = traced_prefixes
    if terminal:
        path = PROTOTYPES / 'terminal_probe.py'
        used_scripts.append(path)
        terminal_candidate = function_from_script(path, 'candidate', {
            'np': np, '_approach_occluder_support': events._approach_occluder_support})
        text = inspect.getsource(original_extensions)
        predicate = 'if len(observations) >= 3 and observations[-1]["y"] < observations[0]["y"] - .5:'
        replacement = """if (len(observations) >= 5 and observations[-1]['time'] - observations[0]['time'] >= .07
                    and np.linalg.norm(tracking._velocity(observations, tail=False)) <= .65 * np.linalg.norm(velocity)
                    and abs(observations[-1]['y'] - observations[0]['y']) <= max(8., radius * 3)):"""
        assert text.count(predicate) == 1
        text = text.replace('def _rebound_extensions(', 'def terminal_extensions(').replace(predicate, replacement)
        namespace = {**tracking.__dict__, 'tracking': tracking}
        exec(text, namespace)
        tracking._rebound_extensions = namespace['terminal_extensions']
    if dark:
        # The earlier experiment's segmentation thresholds are unchanged.
        # Preserve the common native method for the v5 appearance guard; keep
        # polarity separate instead of bypassing that guard with a new method.
        text = inspect.getsource(original_components).replace('def _components(', 'def dark_components(')
        contrast = 'cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), max(1., sx)))'
        threshold = '(gray >= 100) & (saturation <= 160)'
        assert text.count(contrast) == text.count(threshold) == 1
        text = text.replace(contrast, 'cv2.subtract(cv2.GaussianBlur(gray, (0, 0), max(1., sx)), gray)')
        text = text.replace(threshold, '(gray <= 220) & (saturation <= 190)')
        namespace = recovery.__dict__.copy()
        exec(text, namespace)
        dark_components = namespace['dark_components']
        def both_components(*positional, **keywords):
            bright = original_components(*positional, **keywords)
            dark_rows = dark_components(*positional, **keywords)
            for row in dark_rows:
                row['_experiment_polarity'] = 'dark'
            return bright + dark_rows
        recovery._components = both_components

    recovered_experimental_ids = set()
    def recover(*positional, **keywords):
        accepted, accepted_summary = [], None
        if terminal:
            patched = recovery._approaches, recovery._components, tracking._rebound_extensions
            recovery._approaches, recovery._components, tracking._rebound_extensions = (
                original_approaches, original_components, original_extensions)
            try:
                accepted, accepted_summary = original_recover(*positional, **keywords)
            finally:
                recovery._approaches, recovery._components, tracking._rebound_extensions = patched
        rows, summary = original_recover(*positional, **keywords,
            max_approaches=48 if args.variant == 'budget48' else 24)
        next_id = max([int(track['track_id']) for track in [*positional[0], *accepted]] or [0]) + 1
        for row in rows:
            if terminal:
                row['track_id'] = next_id
                row['source_track_ids'] = [next_id]
                next_id += 1
            recovered_experimental_ids.add(row['track_id'])
            for observation in row['observations']:
                if '_experiment_polarity' in observation:
                    observation['appearance_polarity'] = observation.pop('_experiment_polarity')
        if terminal:
            summary = {'accepted_native24': accepted_summary, 'experimental_terminal': summary,
                'additional_tracks': len(accepted) + len(rows), 'independent_recovery_windows': True}
        return [*accepted, *rows], summary
    recovery.recover_rebound_tracks = recover
    terminal_proposals = []
    def competition(candidates, tracks, nominal_step):
        if terminal:
            for track in tracks:
                if track['track_id'] not in recovered_experimental_ids:
                    continue
                proposal = terminal_candidate(track, base['timestamps'], base['analysis']['width'])
                if proposal is None:
                    continue
                events._apply_flight_continuity(proposal, track, nominal_step)
                terminal_proposals.append(deepcopy(proposal))
                candidates.append(proposal)
        return original_competition(candidates, tracks, nominal_step)
    events.apply_competing_continuation_evidence = competition
    all_candidates = []
    def estimate(tracks, frames, timestamps, calibration, cfg):
        assert not cfg.get('contact_model_path') and not cfg.get('source_contact_model_path')
        candidates = original_estimate(tracks, frames, timestamps, calibration, {**cfg, 'max_candidates': 100000})
        all_candidates.extend(deepcopy(candidates))
        return candidates[:int(cfg.get('max_candidates', 12))]
    events.estimate_landings = estimate
    receipt = {'case': KEY, 'variant': args.variant, 'scope': 'Exposed development case only; isolated monkeypatch experiment, not a production engine execution',
        'source_sha256': base['source_sha256'], 'baseline_result_sha256': evaluator.sha(baseline_path),
        'baseline_detections_sha256': detections_sha, 'frozen_source_files': code,
        'accepted_run_sha256': evaluator.sha(accepted_dir / 'run.json'), 'config': config,
        'config_sha256': evaluator.digest(config), 'runner_sha256': evaluator.sha(__file__),
        'evaluator_sha256': evaluator.sha(ROOT / 'scripts/evaluate_landing_engine.py'),
        'prototype_function_sources': {str(path.relative_to(ROOT)): evaluator.sha(path) for path in used_scripts},
        'dark_adaptation': 'Same pixels/thresholds; common native component method with explicit dark polarity for existing native guard eligibility' if dark else None,
        'terminal_adaptation': 'Retain accepted native24 tracks; add separate terminal-prefix tracks and proposals before existing global evidence gates; terminal kind stays distinct and is ineligible for rebound-only native guard/causal relation' if terminal else None,
        'reference_labels_read_by_inference': False, 'reserved_references_accessed': False}
    evaluator.write(output / 'run.json', receipt)
    result, frame_sha = evaluator.replay(base, source, output, config, code,
        frame_cache=Path(accepted_receipt['frame_cache']), frame_records=records)
    assert evaluator.source_hashes(args.repo) == code
    result['benchmark_ranked_landings'] = events.prioritize_review_candidates(result['landings'], [])
    result['benchmark_execution'] = {'mode': 'isolated_v5_hypothesis_replay',
        'run_receipt_sha256': evaluator.digest(receipt), 'analysis_frames_sha256': frame_sha,
        'reference_labels_read_by_inference': False, 'baseline_detections_sha256': detections_sha,
        'in_memory_experiment_patches': True, 'source_files': code}
    evaluator.write(output / 'result.json', result)
    # Reference enters only after inference output has been produced and saved.
    reference_path = ROOT / 'docs/generalization/references' / (KEY + '.json')
    reference = evaluator.read(reference_path)
    assert reference['source_sha256'] == base['source_sha256']
    row = {'key': KEY, 'cohort': 'exposed_development_recheck', 'group_id': 'uhdhd-5',
        'source_sha256': base['source_sha256'], 'reference_sha256': evaluator.sha(reference_path), 'reference': reference}
    score = evaluator.evaluate_case(row, result)
    accepted_result = evaluator.read(accepted_dir / KEY / 'result.json')
    accepted_score = evaluator.evaluate_case(row, accepted_result)
    raw_matches = [candidate for candidate in all_candidates if (evaluator.compare(candidate, result, reference) or {}).get('matches_first_vicinity')]
    compact = lambda candidate: {key: candidate.get(key) for key in ('track_id', 'frame_index', 'time', 'candidate_px', 'method', 'image_motion_event', 'flight_trajectory_support', 'reason_codes')}
    trace = {'generated_after_evidence_gates_before_cap': len(all_candidates),
        'matching_before_cap_count': len(raw_matches), 'matching_before_cap': [compact(candidate) for candidate in raw_matches],
        'terminal_proposal_count': len(terminal_proposals), 'terminal_proposals': [compact(candidate) for candidate in terminal_proposals],
        'selected_prefix_endpoints': selected_endpoints,
        'candidate_positions': [{key: candidate.get(key) for key in ('track_id', 'frame_index', 'candidate_px')} for candidate in all_candidates]}
    evaluator.write(output / 'candidate-trace.json', trace)
    score.update(accepted_v5={key: accepted_score[key] for key in ('top1_match', 'recall_at_6', 'recall_at_12')},
        fixed_metric_gain=any(score[key] and not accepted_score[key] for key in ('top1_match', 'recall_at_6', 'recall_at_12')),
        result_sha256=evaluator.sha(output / 'result.json'), trace_sha256=evaluator.sha(output / 'candidate-trace.json'),
        receipt_sha256=evaluator.sha(output / 'run.json'))
    if args.variant == 'control24':
        score['selected_coordinate_parity_with_accepted_v5'] = (
            result['selected_landing']['frame_index'] == accepted_result['selected_landing']['frame_index']
            and result['selected_landing']['candidate_px'] == accepted_result['selected_landing']['candidate_px'])
        assert score['selected_coordinate_parity_with_accepted_v5']
    evaluator.write(output / 'evaluation.json', score)
    print(json.dumps({'variant': args.variant, 'top1': score['top1_match'], 'recall6': score['recall_at_6'],
        'recall12': score['recall_at_12'], 'generated': len(all_candidates), 'matching_before_cap': len(raw_matches),
        'terminal_proposals': len(terminal_proposals), 'gain': score['fixed_metric_gain']}), flush=True)


if __name__ == '__main__':
    main()
