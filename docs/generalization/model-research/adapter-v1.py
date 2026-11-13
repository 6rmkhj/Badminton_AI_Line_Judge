"""Bounded development-only TrackNet grid/CV ensemble experiment.

No reference coordinates or event windows enter inference, candidate extraction,
tracking, or ranking. Candidate labels stay explicitly uncalibrated.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import copy
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = Path('/workspace/generalization')
MODEL_SHA = 'b0ed780c3cc2c4f8eab6efa08d1489667381ecdcced7fad52abccbd176204ed6'
PREPROCESSING = 'cached_bgr960_fullclip_median_uint8_then_fixed_grid_rgb_opencv_inter_cubic_512x288_nchw_bg_plus_8_frames_div255'
REPLAY_SNAPSHOT = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


class GridModel:
    def __init__(self, frames):
        if sha(ROOT / 'tracknetv3.onnx') != MODEL_SHA:
            raise ValueError('Unexpected model bytes')
        self.frames = frames
        self.h, self.w = frames.shape[1:3]
        tw, th = round(self.w * .6), round(self.h * .6)
        self.grid = [[x, y, tw, th] for y in (0, self.h - th) for x in (0, self.w - tw)]
        t0 = time.perf_counter()
        median = np.median(frames, axis=0).astype(np.uint8)
        self.backgrounds = [self.resize(median[y:y+h, x:x+w]) for x,y,w,h in self.grid]
        self.setup_preprocess_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.net = cv2.dnn.readNetFromONNX(str(ROOT / 'tracknetv3.onnx'))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model_load_seconds = time.perf_counter() - t0

    @staticmethod
    def resize(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (512,288), interpolation=cv2.INTER_CUBIC).transpose(2,0,1)

    def batch(self, start):
        count = min(8, len(self.frames) - start)
        indices = list(range(start, start + count))
        indices += [indices[-1]] * (8 - len(indices))
        candidates = [[] for _ in range(count)]
        heatmaps, infer_seconds, preprocess_seconds = [], 0., 0.
        for tile, (tx,ty,tw,th) in enumerate(self.grid):
            t0 = time.perf_counter()
            blocks = [self.backgrounds[tile]] + [self.resize(self.frames[i,ty:ty+th,tx:tx+tw]) for i in indices]
            inp = np.ascontiguousarray(np.concatenate(blocks, axis=0)[None], dtype=np.float32) / 255.
            preprocess_seconds += time.perf_counter() - t0
            self.net.setInput(inp)
            t0 = time.perf_counter()
            raw = self.net.forward()
            infer_seconds += time.perf_counter() - t0
            if raw.shape != (1,8,288,512) or not np.isfinite(raw).all():
                raise ValueError('Unexpected TrackNet output')
            heatmaps.append(raw.copy())
            for offset, hm in enumerate(raw[0,:count]):
                n, labels, stats, _ = cv2.connectedComponentsWithStats((hm > .5).astype(np.uint8), connectivity=8)
                for label in range(1,n):
                    x,y,w,h,area = map(int,stats[label])
                    candidates[offset].append({'x':tx+(x+w/2)*tw/512, 'y':ty+(y+h/2)*th/288,
                        'area_heatmap_px':area,'peak':float(hm[labels==label].max()),'tile_index':tile})
        for items in candidates:
            items.sort(key=lambda c:c['area_heatmap_px'], reverse=True)
        return candidates, np.stack(heatmaps), infer_seconds, preprocess_seconds


def source_patch_observation(component, gray):
    """Measure appearance support near the heatmap center from actual cache pixels.

    Geometry stays the model's image-conditioned location. Radius/polarity use the
    nearest local contrast component within 3px; absent support rejects a proposal.
    This measures image content, never confirms shuttle identity or floor contact.
    """
    cx, cy = component['x'], component['y']
    ix, iy = int(round(cx)), int(round(cy))
    h,w = gray.shape
    x0,x1,y0,y1 = max(0,ix-6),min(w,ix+7),max(0,iy-6),min(h,iy+7)
    patch = gray[y0:y1,x0:x1]
    if not patch.size:
        return None
    background = float(np.median(patch))
    delta = patch.astype(np.float32)-background
    supports = []
    for sign, polarity in ((1,'bright'),(-1,'dark')):
        n,labels,stats,_ = cv2.connectedComponentsWithStats((sign*delta>=8).astype(np.uint8),connectivity=8)
        for label in range(1,n):
            ys,xs = np.where(labels==label)
            distance = float(np.sqrt(((xs+x0-cx)**2+(ys+y0-cy)**2).min()))
            if distance <= 3.:
                supports.append((distance,-float(np.mean(sign*delta[labels==label])),polarity,label,labels,stats[label]))
    if not supports:
        return None
    distance,_,polarity,label,labels,stats = min(supports,key=lambda s:(s[0],s[1]))
    x,y,bw,bh,area = map(int,stats)
    ys,xs = np.where(labels==label)
    pixels = [[int(px+x0),int(py+y0),int(patch[py,px])] for py,px in zip(ys,xs)]
    return {'x':cx,'y':cy,'radius':max(1.,math.sqrt(area/math.pi)),
        'bbox':[x0+x,y0+y,bw,bh],'appearance_polarity':polarity,
        'confidence':component['peak'],'confidence_kind':'model_score_not_calibrated', 'calibrated':False,
        'method':'tracknetv3_fixed_grid_image_heatmap_with_source_patch',
        'global_resolution_observation':True,'native_resolution_observation':False,
        'observed':True,'interpolated':False,'model_sha256':MODEL_SHA,
        'model_heatmap_component':component,'model_preprocessing':PREPROCESSING,
        'model_confirms_identity':False,'model_confirms_contact':False,
        'source_patch_coordinate_size':[w,h],'source_patch_gray_pixels':pixels,
        'source_patch_background_gray':background,'source_patch_nearest_support_distance':distance,
        'source_patch_contrast_threshold':8.,'source_patch_search_radius':6,
        'local_contrast':float(np.mean(np.abs(delta[labels==label])))}


def replay(key, base, frames, records, out, name):
    global REPLAY_SNAPSHOT
    sys.path.insert(0,'/workspace/badminton-performance')
    from badminton.tracking import track_detections
    from badminton.events import estimate_landings, prioritize_review_candidates
    from badminton.identity import apply_identity
    from badminton.pipeline import select_requested_landings, aggregate_decision, implementation_fingerprint
    from badminton.decision import decide
    module_names = ['tracking','events','identity','identity_continuity','candidate_identity','pipeline','decision']
    hashes_before = {n:sha(Path('/workspace/badminton-performance/badminton')/(n+'.py')) for n in module_names}
    if REPLAY_SNAPSHOT is None:
        REPLAY_SNAPSHOT = hashes_before
    elif hashes_before != REPLAY_SNAPSHOT:
        raise RuntimeError('Experimental replay modules changed between comparison methods')
    start = time.perf_counter()
    cfg = copy.deepcopy(base['config'])
    event_cfg = dict(cfg.get('events') or {})
    event_cfg.update(source_contact_enabled=False,frame_preprocessing=base['analysis']['preprocessing'])
    tracks = track_detections(records,cfg.get('tracker'))
    track_seconds = time.perf_counter()-start
    t0 = time.perf_counter()
    landings = estimate_landings(tracks,list(frames),base['timestamps'],base['calibration'],event_cfg)
    identity_cfg = dict(cfg.get('identity') or {})
    landings = apply_identity(landings,tracks,base['timestamps'],base['calibration'],identity_cfg)
    for i,event in enumerate(landings):
        event['event_id'] = f'event_{i:03d}'
        event['decision'] = decide(event,base['calibration'],cfg.get('rules'))
    requested = select_requested_landings(landings,identity_cfg)
    selected = prioritize_review_candidates(requested,[])
    hashes_after = {n:sha(Path('/workspace/badminton-performance/badminton')/(n+'.py')) for n in module_names}
    if hashes_before != hashes_after:
        raise RuntimeError('Experimental replay modules changed during replay; do not interpret result')
    result = {k:copy.deepcopy(base[k]) for k in ['source_sha256','source_path','metadata','timestamps','analysis','calibration']}
    result.update(key=key,mode=name,config=cfg,tracks=tracks,landings=landings,
        selected_landing=selected[0] if selected else None,decision=aggregate_decision([e['decision'] for e in requested]),
        diagnostic_only=True,source_contact_refinement_performed=False,
        scope='Cached-source observations replay; current experimental tracker/events/identity; no native source contact refinement, no physical-contact accuracy claim.',
        implementation_fingerprint=implementation_fingerprint(),replay_module_sha256=hashes_before,
        timings={'tracking_seconds':track_seconds,'events_seconds':time.perf_counter()-t0,'total_seconds':time.perf_counter()-start})
    dump(out/f'{name}-result.json',result)
    candidate=result['selected_landing'] or {}
    compact={'mode':name,'track_count':len(tracks),'event_count':len(landings),'primary':{k:candidate.get(k) for k in ['frame_index','candidate_px','track_id','score','flight_trajectory_support']},'timings':result['timings']}
    print(json.dumps(compact),flush=True)
    return compact


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--clip',choices=['8k-2-b','8k-5-b'],default='8k-2-b')
    parser.add_argument('--fixture-only',action='store_true')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    # Only these explicit development keys are permitted; no holdout reads.
    key=args.clip
    base=json.loads((DATA/'final'/key/'result.json').read_text())
    cache=DATA/'detector-diagnosis'/f'{key}-frames960.npy'
    frames=np.load(cache,mmap_mode='r')
    if len(frames)!=len(base['timestamps']):
        raise ValueError('Frame/timestamp count mismatch')
    out=ROOT/'fusion'/key
    out.mkdir(parents=True,exist_ok=True)
    start_time=time.perf_counter()
    model=GridModel(frames)
    if args.fixture_only:
        if key!='8k-2-b':raise ValueError('Only the existing B2 fixture is defined')
        rows,heatmaps,infer_s,prep_s=model.batch(42)
        prior=np.load(ROOT/'grid-8k-2-b-f42-49.npz')['heatmaps']
        old=json.loads((ROOT/'grid-8k-2-b-f42-49.json').read_text())
        comparison={'preprocessing':PREPROCESSING,'old_preprocessing':old['preprocessing'],
            'shape':list(heatmaps.shape),'mean_absolute_heatmap_difference':float(np.mean(np.abs(heatmaps-prior))),
            'max_absolute_heatmap_difference':float(np.max(np.abs(heatmaps-prior))),
            'inference_seconds':infer_s,'preprocess_seconds':prep_s,
            'frames':[{'frame_index':42+i,'new_components':row,'old_components':old['detections'][i]['components']} for i,row in enumerate(rows)]}
        dump(out/'opencv-vs-pil-fixture.json',comparison)
        np.savez_compressed(out/'opencv-fixture-heatmaps.npz',heatmaps=heatmaps)
        print(json.dumps({k:v for k,v in comparison.items() if k!='frames'}),flush=True)
        return
    if not (ROOT/'fusion/8k-2-b/opencv-vs-pil-fixture.json').exists():
        raise ValueError('OpenCV/PIL fixture comparison must precede the full run')
    cv_records=json.loads((DATA/'final'/key/'detections.json').read_text())
    if len(cv_records)!=len(frames):raise ValueError('CV frame count mismatch')
    if any(len(r['detections'])>80 for r in cv_records):
        raise ValueError('This fixed experiment expects at most80 original CV candidates')
    model_records=[]; fusion=copy.deepcopy(cv_records);raw_records=[]
    infer_seconds=prep_seconds=0.;rejected=deduplicated=0
    for start in range(0,len(frames),8):
        candidates,heatmaps,infer_s,prep_s=model.batch(start)
        infer_seconds+=infer_s;prep_seconds+=prep_s
        np.savez_compressed(out/f'heatmaps-f{start:03d}.npz',heatmaps=heatmaps)
        for offset,items in enumerate(candidates):
            idx=start+offset;gray=cv2.cvtColor(frames[idx],cv2.COLOR_BGR2GRAY);accepted=[]
            for item in items:
                if any(math.hypot(item['x']-p['x'],item['y']-p['y'])<=2 for p in accepted):
                    deduplicated+=1;continue
                observation=source_patch_observation(item,gray)
                if observation is None:rejected+=1;continue
                accepted.append(observation)
                if len(accepted)>=8:break
            row={'frame_index':idx,'time':base['timestamps'][idx],'detections':accepted,
                 'occlusion_regions':copy.deepcopy(cv_records[idx].get('occlusion_regions',[]))}
            model_records.append(row)
            raw_records.append({'frame_index':idx,'time_seconds':base['timestamps'][idx],'components':items})
            original=copy.deepcopy(fusion[idx]['detections'])
            fusion[idx]['detections'].extend(copy.deepcopy(accepted))
            assert fusion[idx]['detections'][:len(original)]==original
            assert len(fusion[idx]['detections'])<=len(original)+8
        print(json.dumps({'clip':key,'frames_complete':min(start+8,len(frames)),'frame_count':len(frames),'inference_seconds':infer_seconds,'elapsed_seconds':time.perf_counter()-start_time}),flush=True)
    dump(out/'model-detections.json',model_records)
    dump(out/'fusion-detections.json',fusion)
    dump(out/'raw-grid-components.json',raw_records)
    metadata={'clip':key,'split':'development','source_sha256':base['source_sha256'],'cache_sha256':sha(cache),'cache_path':str(cache),
        'model_sha256':MODEL_SHA,'adapter_sha256':sha(__file__),'grid':model.grid,'threshold':.5,'preprocessing':PREPROCESSING,
        'timestamp_basis':'normalized_source_pts','model_component_budget':8,'original_cv_candidates_preserved':True,
        'original_candidate_count':sum(len(r['detections']) for r in cv_records),'added_candidate_count':sum(len(r['detections']) for r in model_records),
        'patch_unsupported_rejections':rejected,'overlap_duplicate_rejections':deduplicated,
        'inference_seconds':infer_seconds,'preprocess_seconds':prep_seconds+model.setup_preprocess_seconds,'model_load_seconds':model.model_load_seconds,
        'max_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'threads':cv2.getNumThreads(),
        'scope':'Full development clip; fixed8frame batches and four unhinted grid tiles. Last batch repeats last real input only for padding if needed; only real source frame outputs kept. No references/landing windows used. Model-only and fusion observations all tied to actual cached source pixels; model is uncalibrated.'}
    dump(out/'metadata.json',metadata)
    comparisons=[replay(key,base,frames,records,out,name) for name,records in [('cv_only',cv_records),('model_only',model_records),('fusion',fusion)]]
    dump(out/'replay-comparison.json',{'metadata':metadata,'frozen_primary':{k:(base['selected_landing'] or {}).get(k) for k in ['frame_index','candidate_px','track_id']},'comparisons':comparisons})


if __name__=='__main__':main()
