"""Heldout AI-label candidate agreement, never independently verified accuracy.

Compare the same raw-source resized frames with the old bright-only proposals
and dark-cork proposals ranked by the explicitly AI-supervised scorer. Only
labeled frames are scored by the CNN; temporal proposal generation uses the
whole clip. Neither contact truth nor an IN/OUT decision is created here.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from badminton.detection import detect_sequence, _nms
from badminton.learning import _collect_labels, load_candidate_scorer, PREPROCESSING
from badminton.media import load_frames, probe
from badminton.dataset import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('runs/inspection/input-manifest-before.json'))
    p.add_argument('--annotations', type=Path, default=Path('annotations/ai-visual'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('runs/ai-reviewed-poc/detector-validation.json'))
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    args = p.parse_args()
    cv2.setNumThreads(1); torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    records, excluded, errors = _collect_labels(manifest, args.annotations, 'ai_visual_review')
    if errors:
        raise ValueError('; '.join(errors))
    scorer = load_candidate_scorer(args.model)
    if scorer.provenance.get('label_source') != 'ai_visual_review':
        raise ValueError('This diagnostic expects explicitly AI-supervised weights')
    videos = {v['id']: v for v in manifest['videos']}
    rows=[]; totals={name:Counter() for name in ('bright_baseline','dark_unranked','dark_ai_ranked')}
    for vid in sorted({r['video_id'] for r in records if r['split'] == args.split}):
        labels = [r for r in records if r['video_id']==vid]
        video = videos[vid]
        if file_hash(Path(video['path'])) != video['sha256']:
            raise ValueError(f'Source changed: {vid}')
        size = (labels[0]['analysis_width'], labels[0]['analysis_height'])
        metadata = probe(video['path'])
        frames = load_frames(video['path'], expected_count=len(metadata['timestamps']), resize_to=size)
        config = {'person_backend':'none', 'highres_enabled':False,'max_candidates':512}
        bright = detect_sequence(frames, metadata['timestamps'], config)
        dark = detect_sequence(frames, metadata['timestamps'], {**config,'include_dark_components':True})
        indexed={}
        for fi in sorted({r['frame_index'] for r in labels}):
            if fi >= len(frames):
                raise ValueError('Label exceeds original source timeline')
            scored = scorer.score(frames[fi], dark[fi]['detections'], preprocessing=PREPROCESSING)
            for d in scored:
                d['confidence'] = float(d['learned_confidence'])
            indexed[fi] = {'bright_baseline':_nms(bright[fi]['detections'],2.25,64),
                           'dark_unranked':_nms(dark[fi]['detections'],2.25,64),
                           'dark_ai_ranked':_nms(scored,2.25,64)}
        for label in labels:
            row={'video_id':vid,'file':video['file'],'split':args.split,
                 'frame_index':label['frame_index'],'point_px':[label['x'],label['y']],
                 'ai_label':label['label_name'],'candidates':{}}
            for name, ds in indexed[label['frame_index']].items():
                distance=min((math.hypot(d['x']-label['x'],d['y']-label['y']) for d in ds),default=None)
                hit=distance is not None and distance <= 6.0
                row['candidates'][name]={'nearest_distance_px':distance,'within_6px':hit,'candidate_count':len(ds)}
                totals[name][label['label_name']+'_points'] += 1
                totals[name][label['label_name']+'_points_with_candidate'] += int(hit)
            rows.append(row)
        print(json.dumps({'completed_video':video['file'],'frames':len(frames),'labeled_points':len(labels)},ensure_ascii=False),flush=True)
        del frames,bright,dark,indexed
    report={'scope':'Heldout AI-labeled point candidate agreement, not precision/recall on complete human annotations or IN/OUT accuracy',
            'split':args.split,'match_radius_px':6.0,'independent_human_ground_truth':False,
            'automatic_judgment_accuracy':None,'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'detector_code_sha256':hashlib.sha256(Path('badminton/detection.py').read_bytes()).hexdigest(),
            'source_preprocessing':PREPROCESSING,'high_resolution_roi_enabled':False,
            'proposal_pool_limit':512,'final_candidate_limit':64,'excluded_annotation_counts':excluded,
            'video_count':len({r['video_id'] for r in rows}),'point_count':len(rows),
            'counts':{k:dict(v) for k,v in totals.items()},'points':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report['counts'],indent=2))


if __name__=='__main__':
    main()
