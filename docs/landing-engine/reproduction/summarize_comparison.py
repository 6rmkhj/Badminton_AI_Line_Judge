#!/usr/bin/env python3
"""Rebuild the landing-engine comparison from frozen, hash-bound executions.

No inference or relabeling occurs. --check verifies the saved comparison and its
captured fresh-full subset; a normal run refreshes that subset. --verify-sources
also rehashes original videos. Reserved evaluation has a separate frozen protocol
and denominator, and is never pooled with the development comparison.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ('baseline', 'causal-v4', 'family-v4', 'combined-v4', 'recovery-v4',
            'recovery-v5', 'minimal-v5')
SCORER_SHA = 'f6499a171047016e8065524537d45150e0aeb2728414afad527ef9497e228408'
EVALUATORS = {
    SCORER_SHA: 'docs/landing-engine/reproduction/evaluate_baseline.py',
    'f1e99a730365f392d54231c3097582ab334c1bd6bef1c33040f3a98fc06e6eed':
        'docs/landing-engine/reproduction/evaluate_v4.py',
    'f2135ad60d351933d3075eb19a5dbefbc4bf42f53807c09f44c139089536e9b1':
        'scripts/evaluate_landing_engine.py',
}
SELECTION_FIELDS = ('track_id', 'frame_index', 'time', 'candidate_px', 'method')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def artifact(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scorer():
    path = Path(__file__).with_name('evaluate_baseline.py')
    require(sha(path) == SCORER_SHA, 'Frozen scoring implementation changed')
    spec = importlib.util.spec_from_file_location('frozen_landing_scorer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt_info(directory, protocol_sha, expected_keys):
    receipt = read(directory / 'run.json')
    require(receipt['protocol_sha256'] == protocol_sha, f'{directory.name}: protocol changed')
    require(receipt['config_sha256'] == digest(receipt['config']), 'Configuration digest mismatch')
    require(receipt['source_tree_sha256'] == digest(receipt['source_files']), 'Source tree digest mismatch')
    require(receipt['keys'] == expected_keys, 'Declared input keys changed')
    require(receipt['references_used_for_inference'] is False, 'Assisted execution is not eligible')
    snapshot = Path(receipt['repo'])
    if snapshot.is_dir():
        for name, expected in receipt['source_files'].items():
            require(sha(snapshot / name) == expected, f'Frozen source changed: {snapshot / name}')
    return receipt


def result_case(directory, row, receipt, scoring, expected_sha=None):
    path = directory / row['key'] / 'result.json'
    data = path.read_bytes()
    result_sha = hashlib.sha256(data).hexdigest()
    require(expected_sha is None or result_sha == expected_sha, f'Result changed: {path}')
    result = json.loads(data)
    execution = result['benchmark_execution']
    require(execution['run_receipt_sha256'] == digest(receipt), f'Execution receipt mismatch: {path}')
    require(execution['source_files'] == receipt['source_files'], f'Execution code mismatch: {path}')
    require(execution['mode'] == receipt['mode'], f'Execution mode mismatch: {path}')
    require(execution.get('reference_labels_read_by_inference', execution.get('reference_labels_read')) is False,
            f'Reference assistance: {path}')
    require(result['config'] == receipt['config'], f'Execution configuration mismatch: {path}')
    case = scoring.evaluate_case(row, result)
    case['result_sha256'] = result_sha
    cache = execution.get('analysis_frame_cache') or {}
    provenance = {
        'analysis_frames_sha256': execution['analysis_frames_sha256'],
        'baseline_result_sha256': execution.get('baseline_result_sha256'),
        'baseline_detections_sha256': execution.get('baseline_detections_sha256'),
        'cache_enabled': bool(cache.get('enabled')), 'cache_reused': bool(cache.get('reused')),
        'cache_contract_sha256': cache.get('contract_sha256'),
        'reused_stages': execution.get('reused_stages', []),
    }
    if cache:
        require(cache['read_only'] is True and cache['analysis_frames_sha256'] == execution['analysis_frames_sha256'],
                f'Invalid frame-cache binding: {path}')
    return case, provenance


def selected_snapshot(row):
    return {name: row[name] for name in ('selected', 'selected_comparison', 'candidate_kind',
            'top1_match', 'recall_at_6', 'recall_at_12', 'matching_candidate_ranks',
            'contact_observed', 'automatic_decision', 'result_sha256')}


def changes(base, current):
    output = {}
    for metric in ('top1_match', 'recall_at_6', 'recall_at_12'):
        output[metric + '_gains'] = sorted(key for key in current
            if base[key][metric] is False and current[key][metric] is True)
        output[metric + '_regressions'] = sorted(key for key in current
            if base[key][metric] is True and current[key][metric] is False)
    output['not_successfully_analyzed_keys'] = sorted(set(base) - set(current))
    return output


def reserved_validation(scoring, primary_rows, primary_receipts, verify_sources):
    directory = ROOT / 'docs/landing-engine/reserved-source'
    protocol_path, release_path = directory / 'protocol.json', directory / 'release.json'
    manifest_path = directory / 'sealed-manifest.json'
    protocol, release, manifest = read(protocol_path), read(release_path), read(manifest_path)
    freeze_path = ROOT / protocol['accepted_freeze_path']
    require(sha(freeze_path) == protocol['accepted_freeze_sha256'] == release['accepted_freeze_sha256'],
            'Reserved references are not bound to the accepted freeze')
    require(sha(protocol_path) == release['protocol_sha256'], 'Reserved release protocol changed')
    require(sha(manifest_path) == release['original_sealed_manifest_sha256'] ==
            protocol['reference_policy']['original_sealed_manifest_sha256'], 'Original reserved seal changed')
    require(release['reference_copy_bytes_unchanged'] is True and
            release['no_reference_or_configuration_tuning_after_release'] is True, 'Reserved freeze policy changed')
    require(datetime.fromisoformat(manifest['frozen_at_utc']) < datetime.fromisoformat(release['released_at_utc']),
            'Reserved references were not frozen before release')
    rows = scoring.validate_inputs(protocol, ROOT, ROOT / 'video', require_sources=verify_sources)
    keys = [row['key'] for row in rows]
    require(len(keys) == len(set(keys)) == manifest['video_count'] == 4, 'Expected all four reserved inputs')
    require(not ({row['source_sha256'] for row in rows} & {row['source_sha256'] for row in primary_rows}),
            'Reserved sources overlap the development comparison')
    sealed = {row['key']: row for row in manifest['reference_files']}
    for row in rows:
        require(row['source_sha256'] == sealed[row['key']]['sha256'] and
                row['reference_sha256'] == sealed[row['key']]['reference_sha256'] and
                scoring.scorable(row['reference']) == sealed[row['key']]['scorable'], 'Sealed reserved reference changed')
    variants = {'reserved-baseline': 'baseline', 'reserved-accepted-v5': 'minimal-v5'}
    all_cases, runs = {}, {}
    for variant, corresponding in variants.items():
        run_dir = ROOT / 'runs/landing-engine' / variant
        receipt = receipt_info(run_dir, sha(protocol_path), keys)
        require(receipt['mode'] == 'full' and all(receipt[field] == primary_receipts[corresponding][field]
                for field in ('source_tree_sha256', 'config_sha256')), 'Reserved engine/configuration differs from freeze')
        evaluator = artifact(ROOT / EVALUATORS[receipt['evaluator_sha256']])
        require(evaluator['sha256'] == receipt['evaluator_sha256'], 'Reserved evaluator changed')
        report_path = run_dir / 'evaluation.json'
        report = read(report_path)
        require(report['run'] == receipt and report['protocol_sha256'] == sha(protocol_path), 'Reserved report receipt mismatch')
        require([row['key'] for row in report['cases']] == keys, 'Reserved report is incomplete')
        all_cases[variant] = {}
        for row, recorded in zip(rows, report['cases']):
            case, execution = result_case(run_dir, row, receipt, scoring, recorded['result_sha256'])
            require(all(recorded[name] == value for name, value in case.items()), 'Reserved case scores changed')
            require(not execution['reused_stages'] and not execution['cache_enabled'] and
                    execution['baseline_result_sha256'] is None, 'Reserved run was not a fresh full execution')
            all_cases[variant][row['key']] = case
        summary = scoring.summarize(list(all_cases[variant].values()))
        require(summary == report['summary'] and summary['scorable_count'] == manifest['scorable_count'] == 1,
                'Reserved summary or exclusions changed')
        runs[variant] = {
            'execution_mode': 'full', 'declared_input_count': 4, 'successfully_analyzed_count': 4, 'failed_count': 0,
            'run_receipt': artifact(run_dir / 'run.json'), 'evaluation_report': artifact(report_path),
            'source_tree_sha256': receipt['source_tree_sha256'], 'config_sha256': receipt['config_sha256'],
            'reproducible_evaluator': evaluator, 'metrics': summary,
            'reused_upstream_stages': False, 'analysis_frame_cache_used': False,
        }
    require(protocol['accepted_config_sha256'] == release['accepted_config_sha256'] ==
            primary_receipts['minimal-v5']['config_sha256'], 'Reserved acceptance configuration changed')
    comparisons = []
    for key in keys:
        base, final = all_cases['reserved-baseline'][key], all_cases['reserved-accepted-v5'][key]
        case = {name: base[name] for name in ('key', 'cohort', 'group_id', 'source_sha256', 'reference_sha256',
                'reference_visibility', 'localization_scorable')}
        case.update(baseline=selected_snapshot(base), accepted=selected_snapshot(final))
        comparisons.append(case)
    return {
        'scope': protocol['scope'], 'pooled_with_primary17': False,
        'protocol': artifact(protocol_path), 'release': artifact(release_path),
        'sealed_manifest': artifact(manifest_path), 'accepted_freeze': artifact(freeze_path),
        'reference_frozen_at_utc': manifest['frozen_at_utc'], 'released_at_utc': release['released_at_utc'],
        'tuning_after_release': False, 'runs': runs,
        'changes_from_baseline': changes(all_cases['reserved-baseline'], all_cases['reserved-accepted-v5']),
        'demonstrated_top1_gain': runs['reserved-accepted-v5']['metrics']['top1_match_count'] >
                                 runs['reserved-baseline']['metrics']['top1_match_count'],
        'interpretation': 'No demonstrated improvement on the one scorable reserved reference: both engines miss it. Three ambiguous references remain excluded. This small source-only AI check does not establish independent human/contact/IN-OUT accuracy or globally unseen generalization.',
        'cases': comparisons,
    }


def build(fresh_keys=None, verify_sources=False):
    scoring = scorer()
    protocol_path = ROOT / 'docs/landing-engine/protocol.json'
    protocol = read(protocol_path)
    rows = scoring.validate_inputs(protocol, ROOT, ROOT / 'video', require_sources=verify_sources)
    expected_keys = [row['key'] for row in rows]
    require(len(expected_keys) == len(set(expected_keys)) == 34, 'Expected frozen 34-video development set')
    by_key = {row['key']: row for row in rows}
    runs, cases, provenance, receipts = {}, {}, {}, {}
    for variant in VARIANTS:
        directory = ROOT / 'runs/landing-engine' / variant
        receipt = receipts[variant] = receipt_info(directory, sha(protocol_path), expected_keys)
        evaluator = artifact(ROOT / EVALUATORS[receipt['evaluator_sha256']])
        require(evaluator['sha256'] == receipt['evaluator_sha256'], 'Archived evaluator changed')
        completion_path = directory / 'completion.json'
        completion = read(completion_path) if completion_path.exists() else None
        failures = completion['failures'] if completion else []
        success_keys = completion['success_keys'] if completion else expected_keys
        require(len(success_keys) == len(set(success_keys)), 'Duplicate successful input')
        failed_keys = [failure['key'] for failure in failures]
        require(len(failed_keys) == len(set(failed_keys)), 'Duplicate failed input')
        require(set(success_keys).isdisjoint(failed_keys) and set(success_keys) | set(failed_keys) == set(expected_keys),
                f'{variant}: completed/failed partition is incomplete')
        if completion:
            require(completion['declared_videos'] == 34 and completion['successfully_analyzed_videos'] == len(success_keys)
                    and completion['failed_videos'] == len(failures), 'Completion counts disagree')
        report_paths = [directory / 'evaluation.json'] if not failures else sorted(directory.glob('evaluation-subset-*.json'))
        candidates = [(path, read(path)) for path in report_paths if path.exists()]
        candidates = [(path, report) for path, report in candidates if {row['key'] for row in report['cases']} == set(success_keys)]
        require(len(candidates) == 1, f'{variant}: expected one exact completed-cohort report')
        report_path, report = candidates[0]
        require(report['run'] == receipt and report['protocol_sha256'] == sha(protocol_path), 'Report receipt mismatch')
        cases[variant], provenance[variant] = {}, {}
        for recorded in report['cases']:
            key = recorded['key']
            case, execution = result_case(directory, by_key[key], receipt, scoring, recorded['result_sha256'])
            require(all(recorded[name] == value for name, value in case.items()), f'Recomputed scores differ: {variant}/{key}')
            cases[variant][key], provenance[variant][key] = case, execution
        require(len(cases[variant]) == len(report['cases']), 'Report duplicates cases')
        recomputed = scoring.summarize(list(cases[variant].values()))
        require(recomputed == report['summary'], f'{variant}: summary mismatch')
        fail_entries = []
        for failure in failures:
            key = failure['key']
            metadata = directory / key / 'failure.json'
            log = directory / key / 'failure.log'
            require(read(metadata) == failure, f'Failure metadata changed: {variant}/{key}')
            require(failure['source_sha256'] == by_key[key]['source_sha256'] and failure['run_sha256'] == sha(directory / 'run.json'),
                    'Failure receipt/source mismatch')
            require(sha(log) == failure['traceback_sha256'], 'Failure traceback changed')
            fail_entries.append({'key': key, 'status': failure['status'], 'returncode': failure['returncode'],
                'reference_scorable': scoring.scorable(by_key[key]['reference']),
                'metadata': artifact(metadata), 'traceback': artifact(log),
                'exception': log.read_text(encoding='utf-8').strip().splitlines()[-1]})
        baseline_config = receipts['baseline']['config']
        require({k: v for k, v in receipt['config'].items() if k != 'events'} ==
                {k: v for k, v in baseline_config.items() if k != 'events'}, 'Ablation changes upstream configuration')
        if variant != 'baseline':
            require(receipt['baseline_run_sha256'] == runs['baseline']['run_receipt']['sha256'], 'Replay baseline changed')
            for key, execution in provenance[variant].items():
                require(execution['baseline_result_sha256'] == cases['baseline'][key]['result_sha256'], 'Replay input changed')
                require(execution['analysis_frames_sha256'] == provenance['baseline'][key]['analysis_frames_sha256'],
                        'Replay analysis pixels changed')
        runs[variant] = {
            'execution_mode': receipt['mode'], 'declared_input_count': 34,
            'successfully_analyzed_count': len(cases[variant]), 'failed_count': len(failures),
            'run_receipt': artifact(directory / 'run.json'), 'evaluation_report': artifact(report_path),
            'completion_receipt': artifact(completion_path) if completion else None,
            'source_tree_sha256': receipt['source_tree_sha256'], 'config_sha256': receipt['config_sha256'],
            'evaluator_sha256': receipt['evaluator_sha256'], 'implementation_fingerprint': receipt['implementation_fingerprint'],
            'reproducible_evaluator': evaluator,
            'event_config_delta_from_baseline': {k: v for k, v in receipt['config']['events'].items()
                if v != baseline_config['events'].get(k)},
            'baseline_run_receipt_sha256': receipt.get('baseline_run_sha256'),
            'baseline_detection_inventory_sha256': receipt.get('baseline_detection_inventory_sha256'),
            'analysis_frame_cache': {'enabled_case_count': sum(p['cache_enabled'] for p in provenance[variant].values()),
                'reused_case_count': sum(p['cache_reused'] for p in provenance[variant].values()),
                'configured_max_bytes': receipt.get('frame_cache_max_bytes')},
            'metrics': recomputed, 'failures': fail_entries,
        }
    cohorts = {}
    for cohort in ('regression24', 'exposed8', 'controls2', 'primary32'):
        included = lambda row: row['cohort'] != 'controls2' if cohort == 'primary32' else row['cohort'] == cohort
        cohorts[cohort] = {variant: scoring.summarize([row for row in cases[variant].values() if included(row)]) for variant in VARIANTS}
    changes_by_variant = {variant: changes(cases['baseline'], cases[variant]) for variant in VARIANTS[1:]}
    accepted = cases['minimal-v5']
    comparisons, changed_successes = [], []
    for key in expected_keys:
        base, final = cases['baseline'][key], accepted[key]
        case = {name: base[name] for name in ('key', 'cohort', 'group_id', 'source_sha256', 'reference_sha256',
                'reference_visibility', 'localization_scorable')}
        case.update(baseline=selected_snapshot(base), accepted=selected_snapshot(final))
        comparisons.append(case)
        fields = [field for field in SELECTION_FIELDS if (base['selected'] or {}).get(field) != (final['selected'] or {}).get(field)]
        if base['candidate_kind'] != final['candidate_kind']:
            fields.append('candidate_kind')
        if base['top1_match'] and fields:
            changed_successes.append({'key': key, 'changed_fields': fields, 'strict_match_preserved': final['top1_match']})
    freeze_path = ROOT / 'docs/landing-engine/accepted-freeze.json'
    freeze = read(freeze_path)
    require(freeze['config_sha256'] == receipts['minimal-v5']['config_sha256'] and
            freeze['source_freeze']['source_tree_sha256'] == receipts['minimal-v5']['source_tree_sha256'], 'Acceptance freeze changed')
    require(freeze['baseline_report_sha256'] == runs['baseline']['evaluation_report']['sha256'] and
            freeze['accepted_report_sha256'] == runs['minimal-v5']['evaluation_report']['sha256'], 'Accepted reports changed')
    require(sorted(freeze['strict_gains']) == changes_by_variant['minimal-v5']['top1_match_gains'] and
            freeze['strict_regressions'] == changes_by_variant['minimal-v5']['top1_match_regressions'], 'Acceptance outcome differs')

    # Only completed, receipt-bound full runs count. A saved subset remains stable
    # under --check while the full-run controller produces additional evidence.
    full_dir = ROOT / 'runs/landing-engine/full-v5'
    full_receipt = receipt_info(full_dir, sha(protocol_path), expected_keys)
    require(full_receipt['mode'] == 'full' and all(full_receipt[k] == receipts['minimal-v5'][k]
            for k in ('source_tree_sha256', 'config_sha256', 'evaluator_sha256')), 'Fresh execution differs from accepted freeze')
    if fresh_keys is None:
        fresh_keys = [key for key in expected_keys if (full_dir / key / 'result.json').exists()
                      and read(full_dir / key / 'result.json').get('benchmark_execution')]
    fresh_cases, fresh_rows = [], []
    parity_fields = ('selected', 'selected_comparison', 'candidate_kind', 'top1_match', 'recall_at_6',
                     'recall_at_12', 'contact_observed', 'automatic_decision', 'decision_reason_codes')
    for key in fresh_keys:
        require(key in by_key, 'Fresh subset contains unknown input')
        case, execution = result_case(full_dir, by_key[key], full_receipt, scoring)
        require(not execution['reused_stages'] and not execution['cache_enabled'] and
                execution['baseline_result_sha256'] is None, 'Fresh execution reused upstream stages')
        require(execution['analysis_frames_sha256'] == provenance['minimal-v5'][key]['analysis_frames_sha256'], 'Fresh analysis pixels changed')
        differences = [name for name in parity_fields if case[name] != accepted[key][name]]
        fresh_cases.append({'key': key, 'result': artifact(full_dir / key / 'result.json'),
            'source_sha256': case['source_sha256'], 'reference_sha256': case['reference_sha256'],
            'analysis_frames_sha256': execution['analysis_frames_sha256'],
            'accepted_replay_comparison_fields': list(parity_fields), 'different_fields': differences,
            'selection': selected_snapshot(case)})
        fresh_rows.append(case)
    return {
        'schema_version': 1,
        'scope': 'Frozen development comparison of approximate first-floor-vicinity candidate timing and object-center localization; no independent physical-contact or IN/OUT accuracy claim.',
        'protocol': artifact(protocol_path), 'reproducer': artifact(Path(__file__)),
        'scoring_implementation': artifact(Path(__file__).with_name('evaluate_baseline.py')),
        'accepted_freeze': artifact(freeze_path),
        'reference_policy': {
            'previously_exposed_primary_videos': 32, 'primary_scorable_references': 17,
            'separate_control_videos': 2, 'control_scorable_references': 2,
            'labels': 'Frozen AI visual approximate first-vicinity object centers and intervals, not independent human ground truth.',
            'matching': 'Frozen per-reference coordinate scaling, interval, pixel tolerance and extra frame tolerance; both time and space must pass.',
            'independent_contact_accuracy': None, 'independent_line_judgment_accuracy': None,
            'held_out_claim': False, 'references_used_for_inference': False,
            'groups': protocol['grouping'], 'reserved_evaluation': 'Separate post-freeze protocol and denominator in reserved_validation; never pooled with primary17.'},
        'runs': runs, 'cohorts': cohorts, 'changes_from_baseline': changes_by_variant,
        'baseline_to_accepted_cases': comparisons,
        'changed_prior_success_selections': changed_successes,
        'fresh_full_validation': {
            'scope': 'Selected completed integration subset only; does not claim all 34 accepted cases were freshly rerun.',
            'run_receipt': artifact(full_dir / 'run.json'), 'captured_keys': fresh_keys,
            'source_tree_sha256': full_receipt['source_tree_sha256'], 'config_sha256': full_receipt['config_sha256'],
            'evaluator_sha256': full_receipt['evaluator_sha256'],
            'reused_upstream_stages': False, 'analysis_frame_cache_used': False,
            'metrics': scoring.summarize(fresh_rows), 'cases': fresh_cases},
        'reserved_validation': reserved_validation(scoring, rows, receipts, verify_sources),
        'execution_notes': [
            'Baseline uses the full original-video pipeline for all 34 inputs. Other complete-cohort phases replay from hash-bound baseline detector/tracker/calibration outputs.',
            'Recovery variants freshly measure native source pixels before events, identity, native contact and decision. Native recovery observations are additional tracks, not detector-stage detections.',
            'Replay may reuse exact read-only analysis-frame cache pixels bound to original source, preprocessing, dimensions and timestamps. Native recovery/contact still reads original-resolution video. Replay timing is not whole-pipeline timing.',
            'v4 raw recovery failed on uhd4k-4-b in both phases; successful scores cover 33 inputs. That reference is unscorable, but the analysis failure remains explicitly reported.',
            'All retained decisions remain REVIEW because first physical contact, target identity and calibration are not established. Candidate agreement is not a legal line call.',
            'Comparison verification re-scores hash-verified results with the frozen scorer and verifies reference bytes; --verify-sources additionally rehashes original video files.'
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--verify-sources', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'docs/landing-engine/comparison.json'
    existing = read(output) if output.exists() else None
    if args.check:
        require(existing is not None, 'Stored comparison is missing')
    fresh_keys = existing['fresh_full_validation']['captured_keys'] if args.check else None
    result = build(fresh_keys, args.verify_sources)
    if args.check:
        require(result == existing, 'Stored comparison differs from frozen execution evidence')
        print('Comparison verified: 7 development phases, explicit v4 failures, captured fresh-full subset, separate complete reserved4 full runs.')
    else:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(output)


if __name__ == '__main__':
    main()
