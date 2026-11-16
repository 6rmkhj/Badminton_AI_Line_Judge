"""Compare variant analyses with the original-quality analyses of the same rallies.

For each pair and view: verdict, best-estimate basis, the shown landing in
normalised image coordinates and time, and the AI reference landing (mapped by
time and normalised position, evaluation only).
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def refs():
    table = {}
    for path in (ROOT / 'annotations/ai-visual').glob('*.json'):
        row = json.loads(path.read_text(encoding='utf-8'))
        table[row.get('file')] = row
    return table


def views(run):
    out = {}
    for path in (ROOT / run).glob('doubles/*/*/result.json'):
        r = json.loads(path.read_text(encoding='utf-8'))
        name = Path(r.get('source_path') or r['file']).name
        w, h = r['analysis']['width'], r['analysis']['height']
        best = r.get('best_estimate') or {}
        shown = r.get('display_landing')
        point = best.get('point_px')
        out[name] = {'verdict': (r.get('display_decision') or {}).get('decision'), 'kind': (r.get('display_decision') or {}).get('decision_kind'),
                     'basis': best.get('basis'), 'time': best.get('time'),
                     'norm': [point[0] / w, point[1] / h] if point else None,
                     'warnings': len(r.get('evidence_warnings') or []), 'width': w,
                     'display': None if not shown else shown.get('event_id'),
                     'display_track_matches': (shown or {}).get('track_id') == best.get('track_id') if shown else None}
    return out


def hit(row, ref):
    if not row['norm'] or row['time'] is None or not ref or not ref.get('contact_px') or ref.get('time') is None:
        return None
    d = np.hypot((row['norm'][0] - ref['contact_px'][0] / 960) * 960, (row['norm'][1] - ref['contact_px'][1] / 540) * 540)
    return bool(d <= 25 and abs(row['time'] - ref['time']) <= 0.27)


def main():
    table = refs()
    base = views('runs/all-pairs-v3')
    for variant in sys.argv[1:]:
        current = views(f'runs/variants/run-{variant}')
        flips = shown_hits = base_hits = refd = 0
        print('==', variant, len(current), 'views')
        for name, row in sorted(current.items()):
            original = base[name]
            ref = table.get(name)
            h, hb = hit(row, ref), hit(original, ref)
            if ref and ref.get('contact_px'):
                refd += 1; shown_hits += bool(h); base_hits += bool(hb)
            flag = '' if row['verdict'] == original['verdict'] else '  <-- verdict changed'
            flips += bool(flag)
            print(f"  {name:38s} {original['verdict']}/{original['basis'] or original['kind']:17s} -> {row['verdict']}/{row['basis'] or row['kind']:17s}"
                  f" hit {hb}->{h} warn {row['warnings']}{flag}")
        print(f'  views with changed verdict: {flips}; referenced {refd}: hits original {base_hits} variant {shown_hits}')


if __name__ == '__main__':
    main()
