#!/usr/bin/env python3
"""Regenerate the compact causal/family ablation audit from frozen run reports.

Run from any directory. ``--check`` verifies the stored audit without writing.
The optional source_evidence_figure extension is preserved for the source audit.
This summarizes recorded executions; it does not rerun inference or relabel data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ('baseline', 'causal-v4', 'family-v4')
SELECTION_FIELDS = ('track_id', 'frame_index', 'time', 'candidate_px', 'method')


def read(path):
    return Path(path).read_text(encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def artifact(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def counts(rows):
    scored = [row for row in rows if row['localization_scorable']]
    return {'video_count': len(rows), 'scorable_count': len(scored),
            'excluded_count': len(rows) - len(scored),
            'top1_match_count': sum(row['top1_match'] is True for row in scored),
            'recall_at_6_count': sum(row['recall_at_6'] is True for row in scored),
            'recall_at_12_count': sum(row['recall_at_12'] is True for row in scored),
            'matched_candidate_kind_counts': dict(sorted(Counter(
                row['candidate_kind'] for row in scored if row['top1_match']).items())),
            'physical_contact_observed_count': sum(row['contact_observed'] is True for row in rows),
            'automatic_decision_counts': dict(sorted(Counter(row['automatic_decision'] for row in rows).items()))}


def build():
    protocol_path = ROOT / 'docs/landing-engine/protocol.json'
    protocol = json.loads(read(protocol_path))
    reports, receipts, by_key, runs, snapshots = {}, {}, {}, {}, {}
    expected_keys = [row['key'] for row in protocol['videos']]
    for variant in VARIANTS:
        directory = ROOT / 'runs/landing-engine' / variant
        reports[variant] = report = json.loads(read(directory / 'evaluation.json'))
        receipts[variant] = receipt = json.loads(read(directory / 'run.json'))
        assert report['run'] == receipt, f'{variant}: report/run mismatch'
        assert report['protocol_sha256'] == receipt['protocol_sha256'] == sha(protocol_path)
        assert receipt['config_sha256'] == digest(receipt['config'])
        assert receipt['source_tree_sha256'] == digest(receipt['source_files'])
        assert receipt['references_used_for_inference'] is False
        assert [row['key'] for row in report['cases']] == expected_keys
        assert len(expected_keys) == len(set(expected_keys)) == 34
        by_key[variant] = {row['key']: row for row in report['cases']}
        for key, value in counts(report['cases']).items():
            assert report['summary'][key] == value, f'{variant}: summary mismatch for {key}'
        snapshot = receipt['source_tree_sha256']
        snapshots.setdefault(snapshot, {'execution_path': receipt['repo'], 'files': receipt['source_files']})
        source_directory = Path(receipt['repo'])
        if source_directory.is_dir():
            assert all(sha(source_directory / name) == expected for name, expected in receipt['source_files'].items()), \
                f'{variant}: execution source snapshot changed'
        evaluator_path = ROOT / 'scripts' / (
            'evaluate_landing_engine.py' if variant == 'baseline' else 'evaluate_landing_engine_next.py')
        archived_evaluator = ROOT / 'docs/landing-engine/reproduction' / (
            'evaluate_baseline.py' if variant == 'baseline' else 'evaluate_v4.py')
        assert sha(archived_evaluator) == receipt['evaluator_sha256'], f'{variant}: archived evaluator changed'
        # Canonical evaluator filenames may later move; the execution receipt's
        # digest remains the authority for which evaluator produced this run.
        runs[variant] = {
            'execution_mode': receipt['mode'], 'run_receipt': artifact(directory / 'run.json'),
            'evaluation_report': artifact(directory / 'evaluation.json'),
            'evaluator': {'execution_path': evaluator_path.relative_to(ROOT).as_posix(),
                          'sha256': receipt['evaluator_sha256'],
                          'archived_source_path': archived_evaluator.relative_to(ROOT).as_posix()},
            'source_snapshot_sha256': snapshot,
            'implementation_fingerprint': receipt['implementation_fingerprint'],
            'config_sha256': receipt['config_sha256'],
            'event_config_delta_from_baseline': {
                key: value for key, value in receipt['config']['events'].items()
                if value != receipts['baseline']['config']['events'].get(key)},
            'baseline_run_receipt_sha256': receipt.get('baseline_run_sha256')}
        assert {key: value for key, value in receipt['config'].items() if key != 'events'} == {
            key: value for key, value in receipts['baseline']['config'].items() if key != 'events'}
        if variant != 'baseline':
            assert receipt['baseline_run_sha256'] == runs['baseline']['run_receipt']['sha256']

    groups = {'primary32': lambda row: row['cohort'] != 'controls2',
              'old24': lambda row: row['cohort'] == 'regression24',
              'previously_exposed_new8': lambda row: row['cohort'] == 'exposed8',
              'controls2': lambda row: row['cohort'] == 'controls2',
              'pooled34': lambda row: True}
    summaries = {group: {variant: counts([row for row in reports[variant]['cases'] if included(row)])
                        for variant in VARIANTS} for group, included in groups.items()}
    cases, changes = [], {variant: {'top1_gains': [], 'top1_regressions': [],
        'recall_at_6_gains': [], 'recall_at_6_regressions': [],
        'recall_at_12_gains': [], 'recall_at_12_regressions': [], 'changed_selections': []}
        for variant in VARIANTS[1:]}
    for key in expected_keys:
        base = by_key['baseline'][key]
        case = {name: base[name] for name in ('key', 'cohort', 'group_id', 'source_sha256',
                'reference_sha256', 'reference_visibility', 'localization_scorable')}
        case.update(baseline_selection=base['selected'], baseline_comparison=base['selected_comparison'], variants={})
        for variant in VARIANTS:
            row = by_key[variant][key]
            assert all(row[name] == base[name] for name in ('source_sha256', 'reference_sha256', 'localization_scorable'))
            entry = {name: row[name] for name in ('top1_match', 'recall_at_6', 'recall_at_12',
                'matching_candidate_ranks', 'candidate_kind', 'contact_observed', 'automatic_decision', 'result_sha256')}
            fields = [field for field in SELECTION_FIELDS if (row['selected'] or {}).get(field) != (base['selected'] or {}).get(field)]
            entry['selection_changed'] = bool(fields)
            if row['selected'] != base['selected']:
                entry['selected'] = row['selected']
            if row['selected_comparison'] != base['selected_comparison']:
                entry['selected_comparison'] = row['selected_comparison']
            if variant != 'baseline':
                if fields:
                    changes[variant]['changed_selections'].append({'key': key, 'changed_fields': fields})
                for metric, label in [('top1_match', 'top1'), ('recall_at_6', 'recall_at_6'), ('recall_at_12', 'recall_at_12')]:
                    if base[metric] is False and row[metric] is True:
                        changes[variant][label + '_gains'].append(key)
                    elif base[metric] is True and row[metric] is False:
                        changes[variant][label + '_regressions'].append(key)
            case['variants'][variant] = entry
        cases.append(case)
    assert changes['causal-v4']['top1_gains'] == ['ordinary-1-a']
    assert changes['causal-v4']['top1_regressions'] == []
    assert all(changes['causal-v4'][name] == [] for name in (
        'recall_at_6_gains', 'recall_at_6_regressions', 'recall_at_12_gains', 'recall_at_12_regressions'))
    assert all(value == [] for value in changes['family-v4'].values())
    assert [summaries['primary32'][variant]['top1_match_count'] for variant in VARIANTS] == [7, 8, 7]
    assert all(summaries['primary32'][variant]['scorable_count'] == 17 for variant in VARIANTS)
    assert all(summaries['controls2'][variant]['top1_match_count'] == 2 for variant in VARIANTS)
    assert all(summaries['pooled34'][variant]['physical_contact_observed_count'] == 0
               and summaries['pooled34'][variant]['automatic_decision_counts'] == {'REVIEW': 34}
               for variant in VARIANTS)
    return {
        'schema_version': 1,
        'scope': 'Frozen full-34 development ablations of causal event ordering and duplicate-family budgeting; not independent contact or line-judgment accuracy.',
        'reference_policy': {
            'primary': '32 previously exposed videos: old24 contains 13 scorable references; exposed8 contains 4. Primary denominator is 17, with 15 excluded for visibility/ambiguity.',
            'controls': 'Two original control videos are reported separately (2/2 scorable), not added to the primary score.',
            'pooled': '34/19 is an explicitly pooled regression summary, not the primary result.',
            'labels': 'Frozen AI visual approximate first-landing-vicinity object centers, not independently human-verified cork contact points or IN/OUT labels.',
            'match': 'Both the frozen frame interval (plus its predeclared frame tolerance) and scaled pixel-distance tolerance must pass; no post-prediction tolerance changes.',
            'inference_received_references': False,
            'independent_contact_accuracy': None, 'independent_line_judgment_accuracy': None},
        'protocol': artifact(protocol_path),
        'reproducer': artifact(Path(__file__)),
        'baseline_config': receipts['baseline']['config'],
        'source_snapshots': snapshots, 'runs': runs,
        'execution_notes': [
            'Baseline executes the full source-video pipeline. v4 variants replay events/identity/native-contact/decision using baseline tracks and calibration plus hash-verified actual source analysis pixels.',
            'The optional read-only frame cache is bound to original source, timestamps, preprocessing, dimensions and the baseline decoded-frame SHA. Replay timing is not full-pipeline timing.',
            'Frozen v4 receipts do not describe later family-member ordering fixes, native-recovery experiments or the final release source.'],
        'summaries': summaries, 'changes_from_baseline': changes,
        'conclusion': {
            'causal_ordering': 'Primary first-vicinity matches increase 7/17 to 8/17; controls remain 2/2. Only ordinary-1-a changes, from a later observed rebound to the earlier inferred first-vicinity candidate. Existing successes and recall@6/@12 are preserved.',
            'family_diversity_default': False,
            'family_diversity_reason': 'No top-1 or recall@6/@12 gain in the complete 34-video ablation; keep this optional mechanism disabled by default.',
            'physical_contact_observed_per_variant': 0,
            'formal_decisions_per_variant': {'REVIEW': 34},
            'accuracy_claim': 'Improved approximate candidate localization/order only; no measured physical-contact or legal line-call accuracy.'},
        'case_encoding': 'Every variant has exact report scores and result SHA. Missing selected/selected_comparison fields inherit baseline_selection/baseline_comparison in that case; selection_changed compares track, frame, time, point and method, excluding display event_id.',
        'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'docs/landing-engine/causal-family-experiments.json'
    result = build()
    existing = json.loads(read(output)) if output.exists() else {}
    if 'source_evidence_figure' in existing:
        result['source_evidence_figure'] = existing['source_evidence_figure']
    if args.check:
        if result != existing:
            raise SystemExit('Stored sequence ablation audit differs from its frozen execution reports')
        print('Frozen causal/family audit matches all 34 cases and receipts.')
    else:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(output)


if __name__ == '__main__':
    main()
