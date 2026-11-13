"""Fresh original A1/B1 regression runs only, after an explicit clean source freeze."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REFERENCE = Path('/workspace/shuttle-fix/verify_results.py')
REFERENCE_SHA = '9f378d467227e36c18cfe7c746f3b912580480caff7cf76afa0284ee2d45c207'
SOURCES = {
    'A': (Path('/workspace/shuttle-fix/input/8k 30 4k 24 a시점 1.mp4'),
          '0cc88d220b046572383d67714b608881da45e15e5165b421d46b5b31667727bb'),
    'B': (Path('/workspace/attachments/3cb8c6d7-2c99-457a-9d82-2ba5906b8097/8k 30 4k 24 b시점 1.mp4'),
          'bcac92b83d0e03f45acb66fdcf3fe6535acbef0aed53449ed96cfdf5aeb090e9'),
}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def git(source, *args):
    return subprocess.check_output(['git','-C',str(source),*args], text=True).strip()


def verify_freeze(source, commit, fingerprint_function, fingerprint):
    dirty = (git(source, 'diff', '--name-only', '-z', 'HEAD').split('\0')
             + git(source, 'ls-files', '--others', '--exclude-standard', '-z').split('\0'))
    changed_code = [name for name in dirty if name and not name.startswith('docs/') and name != 'README.md']
    if git(source, 'rev-parse', 'HEAD') != commit or changed_code:
        raise ValueError('Runtime/config/test source must remain at the supplied freeze; only pending README/docs may change.')
    if fingerprint_function() != fingerprint:
        raise ValueError('Frozen implementation fingerprint changed.')
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError('Original AI control references changed; no retuning is permitted.')


def references():
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError('Original AI control reference SHA mismatch.')
    for node in ast.parse(REFERENCE.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='references' for t in node.targets):
            value = ast.literal_eval(node.value)
            if set(value) != {'A','B'}:
                raise ValueError('Only original A1/B1 control references are allowed.')
            return value
    raise ValueError('Missing immutable control references.')


def assess(result, detections, ref, source_pts, source_sha):
    selected = result.get('selected_landing') or {}
    point, frame = selected.get('candidate_px'), selected.get('frame_index')
    track = next((t for t in result.get('tracks',[]) if t['track_id']==selected.get('track_id')), {})
    points = {o['frame_index']:o for o in track.get('observations',[])}
    records = {row['frame_index']:row for row in detections}
    measured = bool(points) and all(o.get('observed') is not False and not o.get('interpolated')
        and any(p.get('observed') is not False and not p.get('interpolated')
                and p['x']==o['x'] and p['y']==o['y']
                for p in records.get(o['frame_index'],{}).get('detections',[]))
        for o in track.get('observations',[]))
    analysis = result.get('analysis') or {}
    same_space = [analysis.get('width'),analysis.get('height')]==[960,540]
    error = math.dist(point, ref['point']) if point and same_space else None
    timestamps = result.get('timestamps',[])
    aligned = (len(timestamps)==len(source_pts)==analysis.get('frame_count')
               and all(abs(a-b)<=1e-6 for a,b in zip(timestamps,source_pts)))
    frame_time = (isinstance(frame,int) and 0<=frame<len(timestamps)
                  and abs(selected.get('time',-1)-timestamps[frame])<=1e-6)
    within = bool(error is not None and error<=ref['radius']
                  and ref['frames'][0]<=frame<=ref['frames'][1])
    checks=[]
    for f, xy in ref['points'].items():
        observed=points.get(f)
        actual=[observed['x'],observed['y']] if observed else None
        checks.append({'frame_index':f,'visible_reference_px':xy,'observed_px':actual,
                       'error_px':round(math.dist(actual,xy),3) if actual else None})
    gaps_interpolated=any(g.get('observations_interpolated') for g in track.get('gaps',[]))
    valid=(within and measured and aligned and frame_time and not gaps_interpolated
           and result.get('source_sha256')==source_sha and result.get('assistance_used') is False
           and result.get('automatic_evaluation_eligible') is True
           and selected.get('contact_observed') is False and result.get('decision',{}).get('decision')=='REVIEW')
    return {'status':'passed' if valid else 'failed','source_sha256':result.get('source_sha256'),
        'reference':ref,'selected_frame':frame,'selected_time':selected.get('time'),'selected_px':point,
        'point_discrepancy_px':round(error,3) if error is not None else None,
        'within_approximate_reference':within,'selected_track_id':selected.get('track_id'),
        'track_method':track.get('method'),'contact_observed':selected.get('contact_observed'),
        'decision':result.get('decision',{}).get('decision'),'assistance_used':result.get('assistance_used'),
        'all_selected_observations_measured':measured,'selected_observation_count':len(points),
        'gaps_interpolated':gaps_interpolated,'normalized_source_pts_verified':aligned,
        'selected_frame_time_verified':frame_time,'coordinate_size_verified':same_space,
        'trajectory_checks':checks,'image_motion_event':selected.get('image_motion_event'),
        'selected_flight_support':selected.get('flight_trajectory_support'),'timings':result.get('timings',{})}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/workspace/badminton-release-0.5.4'))
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--implementation-fingerprint',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true',help='Required to start decoding/analysis after freeze approval')
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',args.source_commit) or not re.fullmatch('[0-9a-f]{64}',args.implementation_fingerprint):
        parser.error('Supply full frozen commit SHA and implementation SHA256.')
    if not args.execute:
        print(json.dumps({'status':'prepared_not_executed','source':str(args.source),'output':str(args.output),
                          'controls':['A','B'],'fresh_analysis':True,'reference_script_sha256':REFERENCE_SHA}))
        return
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    sys.path.insert(0,str(args.source.resolve()))
    import cv2
    from badminton.media import probe
    from badminton.pipeline import analyze_video,implementation_fingerprint
    verify_freeze(args.source,args.source_commit,implementation_fingerprint,args.implementation_fingerprint)
    refs=references()
    config=json.loads((args.source/'configs/poc.json').read_text())
    if (config.get('request_context') or config.get('identity',{}).get('hit_anchors')
            or config.get('detector',{}).get('refinement_regions')
            or config.get('events',{}).get('review_refinement_regions')
            or config.get('events',{}).get('target_track_id') is not None
            or config.get('detector',{}).get('guided_recovery_enabled')
            or config.get('detector',{}).get('native_temporal_residual_recovery')
            or config.get('events',{}).get('temporal_observation_support')):
        raise ValueError('Original controls require automatic CV settings without manual/experimental guidance.')
    for source,expected in SOURCES.values():
        if sha(source)!=expected:
            raise ValueError('Original control source SHA mismatch: '+str(source))
    if args.output.exists():
        raise ValueError('Fresh controls require a new, nonexistent output directory; cached results are never reused.')
    args.output.mkdir(parents=True)
    cv2.setNumThreads(1)
    report={'status':'running','source_commit':args.source_commit,
        'implementation_fingerprint':args.implementation_fingerprint,'config':config,
        'config_sha256':sha(args.source/'configs/poc.json'),'reference_script_sha256':REFERENCE_SHA,
        'reference_provenance':'Frozen AI source-frame visual references; original development controls, not independent human ground truth',
        'fresh_original_analysis':True,'cached_results_reused':False,'holdout_files_opened':False,
        'source_freeze_scope':'Exact code commit and unchanged runtime/config/tests; pending README/docs allowed',
        'runtime_thread_limits':{'opencv':1,'openblas':1,'omp':1,'mkl':1,'concurrent_videos':1},
        'runner_sha256':sha(__file__),'started_at':datetime.now(timezone.utc).isoformat(),'views':{}}
    write(args.output/'run-config.json',report)
    started=time.perf_counter()
    for camera,(source,expected) in SOURCES.items():
        verify_freeze(args.source,args.source_commit,implementation_fingerprint,args.implementation_fingerprint)
        metadata=probe(source)
        phase=[None]
        def progress(event):
            if event.get('stage')!=phase[0]:
                phase[0]=event.get('stage');print(camera,phase[0],flush=True)
        result=analyze_video(source,args.output/camera,deepcopy(config),progress_callback=progress)
        verify_freeze(args.source,args.source_commit,implementation_fingerprint,args.implementation_fingerprint)
        if result.get('implementation_fingerprint')!=args.implementation_fingerprint or result.get('config')!=config:
            raise ValueError('Result fingerprint/config does not match the frozen automatic control run.')
        records=json.loads((args.output/camera/'detections.json').read_text())
        row=assess(result,records,refs[camera],metadata['timestamps'],expected)
        row['source_metadata']={key:metadata.get(key) for key in ('width','height','fps','frame_count','duration','rotation')}
        row['normalized_source_pts_sha256']=hashlib.sha256(json.dumps(metadata['timestamps'],separators=(',',':'),allow_nan=False).encode()).hexdigest()
        row['source_pts_hash_serialization']='JSON array, compact separators, normalized presentation timestamps in seconds'
        row['result_sha256']=sha(args.output/camera/'result.json')
        row['detections_sha256']=sha(args.output/camera/'detections.json')
        row['evidence_paths']=[]
        row['evidence_sha256']={}
        for item in result.get('artifacts',{}).get('evidence',[]):
            if item.get('event_id')==(result.get('selected_landing') or {}).get('event_id'):
                for field in ('image','source_sequence','trajectory_context'):
                    if item.get(field):
                        relative=Path(item[field]);path=args.output/camera/relative
                        if relative.is_absolute() or '..' in relative.parts or not path.is_file():
                            raise ValueError('Missing or unsafe selected control evidence.')
                        row['evidence_paths'].append(item[field])
                        row['evidence_sha256'][item[field]]=sha(path)
        report['views'][camera]=row
        write(args.output/'validation.json',report)
        print(json.dumps({'camera':camera,**{k:row[k] for k in ('status','selected_frame','selected_px','point_discrepancy_px')}}),flush=True)
    report.update(status='passed' if all(v['status']=='passed' for v in report['views'].values()) else 'failed',
                  finished_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started)
    write(args.output/'validation.json',report)
    if report['status']!='passed':
        raise SystemExit('Original development control regression failed; review results without changing frozen references.')


if __name__=='__main__':
    main()
