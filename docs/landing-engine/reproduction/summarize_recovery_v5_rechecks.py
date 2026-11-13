"""Preserve bounded, source-bound summaries of the isolated v5 rechecks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / 'runs/landing-engine/observation-audit/v5-rechecks'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    records = []
    runners = {sha(path): str(path.relative_to(ROOT)) for path in [
        Path(__file__).with_name('recheck_recovery_v5.py'),
        Path(__file__).with_name('recheck_recovery_v5_pre_harness_move.py')]}
    for variant in ['control24', 'budget48', 'prefix', 'terminal', 'prefix-dark', 'terminal-dark']:
        directory = RUNS / variant
        result, trace, receipt, score = [read(directory / name) for name in [
            'result.json', 'candidate-trace.json', 'run.json', 'evaluation.json']]
        assert sha(directory / 'result.json') == score['result_sha256']
        assert sha(directory / 'candidate-trace.json') == score['trace_sha256']
        assert sha(directory / 'run.json') == score['receipt_sha256']
        assert receipt['runner_sha256'] in runners
        assert receipt['reference_labels_read_by_inference'] is False
        records.append({'variant': variant, 'evaluation': score,
            'recovery_summary': result['native_endpoint_recovery'],
            'generated_after_evidence_gates_before_cap': trace['generated_after_evidence_gates_before_cap'],
            'matching_before_cap_count': trace['matching_before_cap_count'],
            'terminal_proposal_count': trace['terminal_proposal_count'],
            'terminal_proposals': trace['terminal_proposals'],
            'selected_prefix_endpoints': trace['selected_prefix_endpoints'],
            'ranked_candidate_positions': [{key: candidate.get(key) for key in ('track_id', 'frame_index', 'candidate_px', 'method')}
                for candidate in result['benchmark_ranked_landings']],
            'runner': runners[receipt['runner_sha256']], 'runner_sha256': receipt['runner_sha256'],
            'input_source_sha256': receipt['source_sha256'],
            'analysis_frames_sha256': result['benchmark_execution']['analysis_frames_sha256'],
            'config_sha256': receipt['config_sha256'],
            'prototype_function_sources': receipt['prototype_function_sources'],
            'adaptations': [value for value in [receipt['dark_adaptation'], receipt['terminal_adaptation']] if value],
            'artifacts': {name: {'path': str((directory / name).relative_to(ROOT)), 'sha256': sha(directory / name)}
                for name in ['run.json', 'result.json', 'candidate-trace.json', 'evaluation.json']}})
    control = read(RUNS / 'control24/run.json')
    assert all(record['config_sha256'] == control['config_sha256'] for record in records)
    assert records[0]['evaluation']['selected_coordinate_parity_with_accepted_v5']
    if any(record['evaluation']['fixed_metric_gain'] for record in records):
        raise ValueError('A measured gain needs review; refusing to publish the no-gain stopping report')
    summary = {'schema_version': 1, 'case': control['case'],
        'scope': 'Five previously rejected hypotheses re-executed against accepted v5 settings, plus an unchanged native24 parity control. One previously exposed development source only; no generalization or physical-contact accuracy claim.',
        'inference_source': {'source_sha256': control['source_sha256'],
            'baseline_result_sha256': control['baseline_result_sha256'],
            'baseline_detections_sha256': control['baseline_detections_sha256'],
            'accepted_run_sha256': control['accepted_run_sha256']},
        'frozen_source_files': control['frozen_source_files'], 'config': control['config'],
        'config_sha256': control['config_sha256'], 'evaluator_sha256': control['evaluator_sha256'],
        'current_evaluator_path': 'scripts/evaluate_landing_engine.py',
        'runner_revision_note': 'The evaluator was moved to its canonical filename during the batch without a byte change. The first four execution runner bytes are preserved under pre_harness_move; the final runner changes only those three evaluator path literals. A failed import before the dark run produced no inference output and was retried with the same evaluator SHA.',
        'source_decoding': 'verified_seek=False; native pixels read again, exact read-only analysis frame cache and per-frame detector occluder records validated against baseline SHA contracts',
        'terminal_evidence_scope': 'Terminal proposals keep terminal_velocity_impulse kind. Existing descent/rebound-only native appearance and causal relation gates are called but intentionally do not certify this distinct mode; continuity, competing continuation, fragment, scene and ranking stages remain active.',
        'candidate_trace_scope': 'Counts and source-position traces are collected after existing evidence gates and before the global candidate cap; they do not claim the count before all event-generation/evidence filtering.',
        'results': records,
        'fixed_metric_gains': [record['variant'] for record in records if record['evaluation']['fixed_metric_gain']],
        'decision': 'All five additional hypotheses remain rejected: no fixed top1, recall@6 or recall@12 gain on the exposed failure case under accepted v5 settings. No production engine/configuration change follows.',
        'stopping_scope': 'These declared applicable alternatives did not improve this current engine; this is not proof that every future landing model is exhausted.',
        'reserved_references_read_by_runner': False,
        'curator': str(Path(__file__).relative_to(ROOT)), 'curator_sha256': sha(Path(__file__))}
    destination = ROOT / 'docs/landing-engine/v5-recovery-rechecks.json'
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    report_path = ROOT / 'docs/landing-engine/recovery-experiments.json'
    report = read(report_path)
    report['accepted_v5_interaction_rechecks'] = {
        'path': str(destination.relative_to(ROOT)), 'sha256': sha(destination),
        'case': control['case'], 'rejected_hypothesis_count': 5, 'control_count': 1,
        'fixed_metric_gains': summary['fixed_metric_gains'],
        'decision': summary['decision'], 'stopping_scope': summary['stopping_scope']}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'path': str(destination), 'sha256': sha(destination),
        'variants': [(record['variant'], record['generated_after_evidence_gates_before_cap'],
            record['matching_before_cap_count'], record['evaluation']['fixed_metric_gain']) for record in records]}))


if __name__ == '__main__':
    main()
