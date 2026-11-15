"""Markdown summary of a run_all_pairs report (per pair verdict, tier, AI reference).

    python -X utf8 scripts/summarize_all_pairs.py runs/all-pairs-v2/report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KIND = {'verified': '직접 판정', 'video_estimate': '영상 추정', 'best_estimate': '최선 추정', 'unresolved': '보류'}


def main(path):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    for stream in (sys.stdout,):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    for game, block in report['games'].items():
        counts = block['counts']
        agreement = block['ai_reference_agreement']
        print(f"### {game}\n")
        print(f"IN {counts['IN']} · OUT {counts['OUT']} · REVIEW {counts['REVIEW']} · "
              f"AI 시각 검수 레이블과 일치 {agreement['agree']}/{agreement['compared']}\n")
        print('| 쌍 | 최종 | 근거 단계 | 판정 시점 | 신뢰도 | A | B | AI 검수 레이블 |')
        print('| --- | --- | --- | --- | --- | --- | --- | --- |')
        for row in block['pairs']:
            views = row['views']
            label = row.get('ai_reference_label') or '—'
            mark = '' if row.get('agrees_with_ai_reference') is not False else ' ✗'
            confidence = row.get('confidence')
            print(f"| {row['pair_key'].replace('{view}', '*')} | **{row['decision']}**{mark} | "
                  f"{KIND.get(row.get('decision_kind'), row.get('decision_kind'))} | "
                  f"{'/'.join(row.get('source_camera_ids') or []) or '—'} | "
                  f"{'' if confidence is None else f'{confidence:.2f}'} | "
                  f"{views['A'].get('effective') or '—'} | {views['B'].get('effective') or '—'} | {label} |")
        print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'runs/all-pairs/report.json')
