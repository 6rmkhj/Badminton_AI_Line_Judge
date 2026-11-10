"""Exercise the real paired-job API using all 37 existing A/B candidates.

Reuses existing source-bound review evidence; does not run or claim new model
inference, fabricate labels, or modify the historical analysis/annotations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8770')
    parser.add_argument('--report', type=Path, default=Path('runs/ai-reviewed-poc/visual-review.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/pair-validation/all-pairs.json'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))

    def request(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        call = Request(args.url.rstrip('/')+path, data=body,
                       headers={'Content-Type': 'application/json'} if body else {})
        with urlopen(call, timeout=60) as response:
            return json.load(response)

    library = request('/api/pair-library')
    assert len(library['videos']) == 74, 'The full project library must be available'
    checks = []
    for pair in report['filename_pair_audit']['items']:
        views = {row['view'].upper(): {'video_id': row['video_id']} for row in pair['views']}
        created = request('/api/pair-jobs', {'views': views, 'same_event_confirmed': False, 'reuse_review': True})
        deadline = time.monotonic()+90
        while True:
            job = request('/api/pair-jobs/'+created['job_id'])
            if job['status'] not in ('queued', 'running'):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Stored review combination timed out: '+pair['pair_key'])
            time.sleep(.1)
        assert job['status'] == 'completed', job
        result = job['result']
        expected = (pair['views'][0]['assisted_decision']
                    if pair['event_correspondence_validated'] and pair['category'] == 'both_resolved_same_label'
                    else 'REVIEW')
        assert result['decision'] == expected, (pair['pair_key'], expected, result)
        assert result['ground_truth_created'] is False
        assert result['automatic_benchmark_eligible'] is False
        checks.append({'pair_key': pair['pair_key'], 'job_id': job['job_id'],
                       'decision': result['decision'], 'expected': expected,
                       'source_camera_ids': result['source_camera_ids'],
                       'event_correspondence_source': result.get('event_correspondence_source'),
                       'reason_codes': result['reason_codes']})
        print(json.dumps(checks[-1], ensure_ascii=False), flush=True)
    counts = {label: sum(row['decision'] == label for row in checks) for label in ('IN', 'OUT', 'REVIEW')}
    assert len(checks) == 37
    assert counts == {'IN': 15, 'OUT': 5, 'REVIEW': 17}, counts
    output = {'passed': True, 'created_at': datetime.now(timezone.utc).isoformat(),
              'scope': '37 actual filename candidate pairs via HTTP, reusing source-bound AI reviews; no new model inference',
              'same_event_operator_confirmation': False, 'video_count': 74, 'pair_count': 37,
              'counts': counts, 'checks': checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'pair_count': 37, 'counts': counts}, ensure_ascii=False))


if __name__ == '__main__':
    main()
