"""Development-only validation using immutable prior detector observations.

Not a production cache feature or a full fresh detector benchmark. Never reads
references, runs holdouts, or changes candidate coordinates/confidence values.
"""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, math, os, platform, subprocess, sys, threading, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

OLD_COMMIT='920f7445cb6ebe7fc3055dcd33c6785dbd97316d'
OLD_FINGERPRINT='721558d430c93973a8c1383bc72f55a958a1b12bb78743694adca44bb94dd53f'
ROOT=Path('/workspace/generalization')

def sha(data): return hashlib.sha256(data).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def git_bytes(repo,revision,path): return subprocess.check_output(['git','-C',str(repo),'show',f'{revision}:{path}'])
def require(condition,message):
    if not condition: raise ValueError(message)
def function_ast(source,name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
def media_without_sequence(source):
    tree=ast.parse(source)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ClassDef) and n.name=='SourceFrameSequence')
        and not (isinstance(n,ast.Import) and len(n.names)==1 and n.names[0].name=='math')]
    return ast.dump(tree,include_attributes=False)

def validate_cache(row,result,records,config):
    require(row.get('split')=='development','Only development videos may be replayed')
    require(result.get('source_sha256')==row['sha256'],'Cached source SHA differs from manifest')
    require(result.get('implementation_fingerprint')==OLD_FINGERPRINT,'Unexpected cached implementation fingerprint')
    require(result.get('config')==config,'Cached configuration differs from unchanged final configuration')
    require(result.get('assistance_used') is False and result.get('automatic_evaluation_eligible') is True,'Assisted/ineligible cache rejected')
    a=result.get('analysis') or {};times=result.get('timestamps')
    require(a.get('preprocessing')=='opencv_source_bgr_area_v1','Unexpected cached preprocessing')
    require(all(isinstance(a.get(k),int) and not isinstance(a.get(k),bool) and a[k]>0 for k in ['width','height','frame_count']),'Invalid cached analysis dimensions')
    require(isinstance(times,list) and len(times)==a.get('frame_count') and len(records)==len(times),'Cached frame/timestamp count mismatch')
    require(all(isinstance(t,(int,float)) and not isinstance(t,bool) and math.isfinite(t) for t in times),'Nonfinite timestamp')
    require(all(b>a for a,b in zip(times,times[1:])),'Nonmonotonic timestamp')
    for index,(record,stamp) in enumerate(zip(records,times)):
        require(record.get('frame_index')==index and record.get('time')==stamp,'Cached detection frame/time mismatch')
        require(isinstance(record.get('detections'),list),'Missing raw candidate list')
        require(not (record.get('refinement') or {}).get('assistance_used'),'Assisted detector record rejected')
        require(not (record.get('refinement') or {}).get('active_regions'),'Refined detector record rejected')
    return a,times

def validate_current_inputs(frames,timestamps,detector_config,row,old,config):
    a=old['analysis'];require(list(timestamps)==old['timestamps'],'Fresh timestamps differ exactly from cache')
    require(len(frames)==a['frame_count'],'Fresh decoded frame count differs from cache')
    require(all(f.dtype.name=='uint8' and tuple(f.shape)==(a['height'],a['width'],3) for f in frames),'Fresh frame shape/dtype differs')
    cfg=dict(detector_config);source_frames=cfg.pop('source_frames',None)
    require(cfg.pop('source_path',None)==str(Path(row['path']).resolve()),'Detector source path differs')
    require(cfg.pop('frame_preprocessing',None)==a['preprocessing'],'Detector preprocessing differs')
    require(cfg==config['detector'],'Effective detector configuration differs')
    require(source_frames is not None and len(source_frames)==len(frames),'Native source sequence contract differs')
    require(source_frames.index==-1 and source_frames.capture is None,'Native sequence was read before the detector boundary')
    return {'decoded_analysis_frames_sha256':sha(b''.join(hashlib.sha256(f.tobytes()).digest() for f in frames)),
            'exact_timestamp_array_sha256':sha(canonical(timestamps)), 'frame_count':len(frames),
            'analysis_size':[a['width'],a['height']], 'preprocessing':a['preprocessing'],
            'native_decoder_invoked_for_detection':False,
            'native_decoder_sequential_compatibility':'Code-reviewed: unchanged detector indexes0..N; new SourceFrameSequence random seek only applies to skips>=32. No native pixel-equality claim is inferred from cache reuse.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path('/workspace/badminton-progress'))
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--fingerprint',required=True)
    parser.add_argument('--cache',type=Path,default=ROOT/'final')
    parser.add_argument('--manifest',type=Path,default=ROOT/'manifest.json')
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--key',action='append',default=[])
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();repo=args.repo.resolve();run=args.run.resolve()
    require(run!=args.cache.resolve(),'Must not overwrite source cache')
    require(len(args.commit)==40 and all(c in '0123456789abcdef' for c in args.commit),'Full saved revision required')
    require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==args.commit,'HEAD differs from final freeze')
    sys.path.insert(0,str(repo));import cv2,numpy
    import badminton.detection as detection
    from badminton.pipeline import analyze_video,implementation_fingerprint,write_json
    require(implementation_fingerprint()==args.fingerprint,'Final implementation fingerprint differs')
    # Every loaded Python module in the package must be from the saved revision.
    for p in (repo/'badminton').rglob('*.py'):
        require(p.read_bytes()==git_bytes(repo,args.commit,str(p.relative_to(repo))),'Uncommitted package Python change: '+str(p))
    config_path=repo/'configs/poc.json';config=load(config_path)
    require(config_path.read_bytes()==git_bytes(repo,args.commit,'configs/poc.json'),'Uncommitted profile change')
    require(config==load(args.cache/'run-config.json')['config'],'Final profile differs from detector cache profile')
    detector_bytes=(repo/'badminton/detection.py').read_bytes()
    require(detector_bytes==git_bytes(repo,OLD_COMMIT,'badminton/detection.py'),'Detector code changed; fresh detection required')
    old_pipeline=git_bytes(repo,OLD_COMMIT,'badminton/pipeline.py');new_pipeline=(repo/'badminton/pipeline.py').read_bytes()
    require(function_ast(old_pipeline,'_analyze_video')==function_ast(new_pipeline,'_analyze_video'),'Pipeline stage/input contract changed; review required')
    old_media=git_bytes(repo,OLD_COMMIT,'badminton/media.py');new_media=(repo/'badminton/media.py').read_bytes()
    require(media_without_sequence(old_media)==media_without_sequence(new_media),'Media preprocessing outside reviewed source-reader class changed')
    approved=load(Path(__file__).with_name('detector-replay-reviewed-policy.json'))
    require(sha(new_media)==approved['reviewed_media_sha256'],'Native reader differs from independently reviewed compatibility policy')
    require((repo/'badminton/calibration.py').read_bytes()==git_bytes(repo,OLD_COMMIT,'badminton/calibration.py'),'Calibration input side effects must be re-reviewed')
    require(config['detector'].get('backend')=='opencv_candidates','Only the frozen OpenCV detector is supported')
    require(not any(k.endswith('_path') and ('model' in k or 'scorer' in k) for section in ['detector','events'] for k in config.get(section,{})),'External learned artifacts are outside replay scope')
    manifest=load(args.manifest);rows=[r for r in manifest['videos'] if r['split']=='development']
    allowed={r['key'] for r in rows};require(set(args.key)<=allowed,'Holdout/unlisted key rejected')
    if args.key:rows=[r for r in rows if r['key'] in args.key]
    cache_entries={}
    for row in rows:
        require(Path(row['key']).name==row['key'],'Unsafe key')
        row['path']=str((Path(row['path']) if row.get('path') else args.manifest.parent/'input'/row['file']).resolve())
        p=args.cache/row['key'];require(not (p/'error.txt').exists(),'Cache has failed-run marker')
        old=load(p/'result.json');records=load(p/'detections.json');validate_cache(row,old,records,config)
        require(sha(Path(row['path']).read_bytes())==row['sha256'],'Current source bytes differ')
        cache_entries[row['key']]={'source_run_result':str(p/'result.json'),'source_run_result_sha256':sha((p/'result.json').read_bytes()),
            'source_detections':str(p/'detections.json'),'source_detections_sha256':sha((p/'detections.json').read_bytes()),
            'source_commit':OLD_COMMIT,'source_implementation_fingerprint':OLD_FINGERPRINT,'source_sha256':row['sha256'],
            'source_analysis_config_sha256':sha(canonical(old['config'])),
            'original_detector_seconds':old['timings']['detection_seconds'],
            'original_run_total_seconds':old['timings']['total_seconds']}
    metadata={'validation_mode':'development_detector_stage_reuse','split':'development','holdouts_opened':False,
        'commit':args.commit,'implementation_fingerprint':args.fingerprint,'config':config,'repo':str(repo),
        'manifest_sha256':sha(args.manifest.read_bytes()),'replay_runner_sha256':sha(Path(__file__).read_bytes()),
        'reviewed_policy_sha256':sha(Path(__file__).with_name('detector-replay-reviewed-policy.json').read_bytes()),
        'detector_sha256':sha(detector_bytes),'old_detector_sha256':sha(git_bytes(repo,OLD_COMMIT,'badminton/detection.py')),
        'media_sha256':sha(new_media),'old_media_sha256':sha(old_media),
        'runtime':{'python':platform.python_version(),'opencv':cv2.__version__,'numpy':numpy.__version__},
        'runtime_limit':'Original detector outputs did not separately record runtime library versions; reuse does not prove exact historical environment identity.',
        'scope':'Fresh source decode/preview/calibration, tracking, events, decision and native evidence. Detector observations reused unchanged from a previous full native run. Not16 full fresh detector passes; timing is replay-only and cannot establish full-pipeline speedup.',
        'cache_entries':cache_entries,'created_at':datetime.now(timezone.utc).isoformat()}
    run.mkdir(parents=True,exist_ok=True)
    require(not any((run/r['key']).exists() for r in rows),'Output already exists; use a fresh run directory')
    write_json(run/'run-config.json',metadata)
    write_json(run/'detector-replay-run.json',metadata)
    if args.preflight_only:print('PREFLIGHT_OK',len(rows),flush=True);return
    cv2.setNumThreads(config.get('opencv_threads',2));local=threading.local()
    def replay(frames,timestamps,detector_config,*,progress_callback=None):
        context=local.context;row=context['row'];p=args.cache/row['key'];entry=cache_entries[row['key']]
        require(sha((p/'result.json').read_bytes())==entry['source_run_result_sha256'],'Cached result changed after preflight')
        require(sha((p/'detections.json').read_bytes())==entry['source_detections_sha256'],'Cached detector observations changed after preflight')
        old=load(p/'result.json');records=load(p/'detections.json');validate_cache(row,old,records,config)
        checks=validate_current_inputs(frames,timestamps,detector_config,row,old,config)
        context['provenance']={**entry,**checks,'detector_sha256':sha(detector_bytes),'detector_stage_reused':True,
            'assistance_used':False,'reference_files_read':False,'fresh_stages':['decode_preview','calibration','tracking','events','native_contact_evidence','trajectory_context','decision','evidence'],
            'timing_scope':'Partial pipeline replay with unchanged detector artifacts; not full fresh runtime.'}
        if progress_callback:progress_callback({'completed_frames':len(frames),'total_frames':len(frames),'detail':'동일 탐지 관측 재사용 · 새 궤적/접촉/근거 검증'})
        return records
    detection.detect_sequence=replay
    def process(row):
        local.context={'row':row};output=run/row['key'];previous=[None]
        def progress(e):
            if e.get('stage')!=previous[0]:previous[0]=e.get('stage');print(row['key'],e['stage'],flush=True)
        try:
            require(implementation_fingerprint()==args.fingerprint,'Source changed during run')
            result=analyze_video(row['path'],output,copy.deepcopy(config),progress_callback=progress)
            require(implementation_fingerprint()==args.fingerprint==result['implementation_fingerprint'],'Source changed during run')
            require(result['source_sha256']==row['sha256'],'Fresh source hash differs')
            require(result['assistance_used'] is False and result['automatic_evaluation_eligible'] is True,'Unexpected assistance')
            provenance=local.context['provenance'];result['validation_run']=provenance
            result['timings']['timing_scope']='detector_stage_reuse';result['timings']['original_reused_detector_seconds']=provenance['original_detector_seconds']
            require(load(output/'detections.json')==load(args.cache/row['key']/'detections.json'),'Replayed detector observations mutated')
            provenance['saved_replayed_detections_sha256']=sha((output/'detections.json').read_bytes())
            write_json(output/'detector-reuse-provenance.json',provenance);write_json(output/'result.json',result)
            s=result.get('selected_landing') or {};summary={'key':row['key'],'status':'completed','mode':'detector_stage_reuse','frame':s.get('frame_index'),'point':s.get('candidate_px'),'replay_seconds':result['timings']['total_seconds']}
        except Exception as error:
            output.mkdir(parents=True,exist_ok=True);(output/'error.txt').write_text(traceback.format_exc());summary={'key':row['key'],'status':'failed','error':str(error)}
        print(json.dumps(summary),flush=True);return summary
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        summaries=[future.result() for future in as_completed([pool.submit(process,row) for row in rows])]
    write_json(run/'execution-summary.json',summaries)
    print('DEVELOPMENT_REPLAY_COMPLETE',len(summaries),flush=True)
    if any(s['status']!='completed' for s in summaries):raise SystemExit(1)
if __name__=='__main__':main()
