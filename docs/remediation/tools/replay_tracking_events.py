"""Exact old24 tracking/event sequence on real pixels; never opens holdout refs.

This stage diagnostic omits detector execution, native contact refinement,
preview/evidence rendering. Run the same-stage baseline before comparisons.
"""
import argparse,copy,hashlib,json,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/remediation');BASE=Path('/workspace/generalization/release');BASE_COMMIT='d7f3d3303daa40c699d579c17e40269173220937'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def snapshot(repo,out):
 files=sorted((repo/'badminton').rglob('*.py'));before={str(p.relative_to(repo)):sha(p) for p in files}
 for p in files:
  target=out/p.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 after={str(p.relative_to(repo)):sha(p) for p in files}
 if before!=after:raise ValueError('Source changed while snapshotting')
 return before

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--key',action='append',required=True);p.add_argument('--config',type=Path);p.add_argument('--preflight-only',action='store_true');p.add_argument('--fixed-observations',action='store_true',help='Component ablation: use frozen detector observations even if current detector changed; not full-pipeline validation');p.add_argument('--uncapped-events',action='store_true',help='Retain all generated event candidates (no final top12 slice)');p.add_argument('--frozen-source-run',type=Path,help='Extend a previous exact source/config snapshot to further old24 keys');p.add_argument('--temporal-support',choices=['off','on'],help='Explicit same-source support-wrapper component ablation; cached native evidence is never preserved');p.add_argument('--external-observations-manifest',type=Path,help='Separate measured-proposal ablation with source/report-bound detector artifacts');p.add_argument('--scene-context',choices=['off','on'],help='Explicit same-source scene-context ranking ablation through estimate_landings');args=p.parse_args()
 rows={r['key']:r for r in load(ROOT/'regression-manifest.json')['videos']}
 if not set(args.key)<=set(rows):raise ValueError('Only old24 development keys are allowed')
 external=load(args.external_observations_manifest) if args.external_observations_manifest else None
 if external and (not args.fixed_observations or external.get('assistance_used') is not False or external.get('reference_labels_read_by_inference') is not False or not set(args.key)<=set(external.get('videos',{}))):raise ValueError('External observations require explicit unassisted development-ablation provenance')
 frozen=load(args.frozen_source_run/'run-config.json') if args.frozen_source_run else None
 if frozen and args.config:raise ValueError('A frozen source extension must preserve its saved config')
 config=copy.deepcopy(frozen['config']) if frozen else load(args.config or args.repo/'configs/poc.json');original_config=copy.deepcopy(config)
 if args.uncapped_events:config.setdefault('events',{})['max_candidates']=2147483647
 if args.temporal_support is not None:config.setdefault('events',{})['temporal_observation_support']=args.temporal_support=='on'
 if args.scene_context is not None:config.setdefault('events',{})['scene_context_ranking']=args.scene_context=='on'
 args.run.mkdir(parents=True,exist_ok=True)
 if (args.run/'source').exists():raise ValueError('Use a fresh experiment directory')
 source=args.run/'source';tree=snapshot(args.frozen_source_run/'source' if frozen else args.repo,source);sys.path.insert(0,str(source))
 expected_config=copy.deepcopy(frozen['config']) if frozen else config
 if args.temporal_support is not None:expected_config.setdefault('events',{})['temporal_observation_support']=args.temporal_support=='on'
 if args.scene_context is not None:expected_config.setdefault('events',{})['scene_context_ranking']=args.scene_context=='on'
 if frozen and (tree!=frozen['code']['source_files'] or config!=expected_config or args.fixed_observations!=frozen.get('fixed_observations_component_ablation',False) or args.uncapped_events!=frozen.get('uncapped_events',False)):raise ValueError('Frozen source, config or stage mode changed')
 import cv2;cv2.setNumThreads(1)
 from badminton.pipeline import implementation_fingerprint,select_requested_landings,aggregate_decision,model_fingerprints
 from badminton.detection import summarize_detection_quality
 from badminton import __version__
 from badminton.tracking import track_detections
 from badminton.events import estimate_landings,prioritize_review_candidates,review_refinement_regions
 from badminton.identity import apply_identity
 from badminton.decision import decide
 fp=implementation_fingerprint();code={'source_tree_sha256':hashlib.sha256(canonical(tree)).hexdigest(),'source_files':tree,'implementation_fingerprint':fp,'repository_head':frozen['code']['repository_head'] if frozen else subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip()}
 if frozen and code!=frozen['code']:raise ValueError('Frozen implementation fingerprint changed')
 stage_code={}
 for name in ['detection.py','calibration.py']:
  old=subprocess.check_output(['git','-C',str(args.repo),'show',f'{BASE_COMMIT}:badminton/{name}'])
  stage_code[name]={'frozen_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':sha(source/'badminton'/name)}
  if (source/'badminton'/name).read_bytes()!=old and (name!='detection.py' or not args.fixed_observations):raise ValueError(f'{name} changed; baseline detector/calibration stage reuse is invalid')
 mode='tracking_events_actual_source_pixels_no_native_refinement'+('_fixed_observations_component_ablation' if args.fixed_observations else '')+('_uncapped_events' if args.uncapped_events else '')+('_support_wrapper_ablation' if args.temporal_support is not None else '')
 if args.scene_context is not None:mode+='_scene_context_ranking_ablation'
 if external:mode+='_external_measured_proposals_ablation'
 scope='Compare same-stage baseline. Full-pipeline5/13 remains a separate frozen reference result.'
 if args.fixed_observations:scope+=' Frozen-observation component ablation only: current detector changes are not executed or validated.'
 if external:scope+=' This separate run replaces frozen proposals with source-bound external measured proposals; it is excluded from the fixed-old-observation comparison.'
 provenance={'mode':mode,'code':code,'config':config,'original_config_sha256':hashlib.sha256(canonical(original_config)).hexdigest(),'effective_config_sha256':hashlib.sha256(canonical(config)).hexdigest(),'detector_and_calibration_code':stage_code,'fixed_observations_component_ablation':args.fixed_observations,'uncapped_events':args.uncapped_events,'created_at':datetime.now(timezone.utc).isoformat(),'new_holdouts_opened':False,'reference_labels_read_by_inference':False,'omitted_stages':['detector','native_contact_refinement','preview','evidence'],'scope':scope}
 provenance['runner_sha256']=sha(Path(__file__))
 provenance['support_wrapper']={'enabled':bool((config.get('events') or {}).get('temporal_observation_support',False)),'preserve_current_detector_evidence':False,'max_retry_count':1}
 if external:provenance['external_observations_manifest']={'path':str(args.external_observations_manifest),'sha256':sha(args.external_observations_manifest),'contents':external}
 shutil.copyfile(Path(__file__),args.run/'replay_tracking_events.py')
 if frozen:provenance['frozen_source_run_config_sha256']=sha(args.frozen_source_run/'run-config.json')
 (args.run/'run-config.json').write_text(json.dumps(provenance,indent=2)+'\n')
 for key in args.key:
  row=rows[key];base=load(BASE/key/'result.json');a=base['analysis'];cache=load(ROOT/'frame_cache'/key/'provenance.json');path=Path(cache['path'])
  if cache['source_sha256']!=row['sha256'] or cache['baseline_result_sha256']!=sha(BASE/key/'result.json') or sha(path)!=cache['npy_sha256']:raise ValueError('Stale/unverified source frame cache')
  if config.get('analysis_width')!=base['config'].get('analysis_width') or config.get('calibration')!=base['config'].get('calibration') or (not args.fixed_observations and config.get('detector')!=base['config'].get('detector')):raise ValueError('Detector/preprocessing/calibration configuration changed')
  if any((config.get('events') or {}).get(k) for k in ['contact_model_path','source_contact_model_path']):raise ValueError('Learned contact inference is outside this CV-only stage replay')
  if config.get('request_context') or (config.get('identity') or {}).get('hit_anchors') or (config.get('events') or {}).get('target_track_id') is not None or (config.get('detector') or {}).get('refinement_regions') or (config.get('events') or {}).get('review_refinement_regions'):raise ValueError('Assisted selection not allowed in automatic replay')
  frames=np.load(path,mmap_mode='r',allow_pickle=False)
  if frames.dtype!=np.uint8 or tuple(frames.shape)!=(a['frame_count'],a['height'],a['width'],3):raise ValueError('Frame cache dimensions differ')
  if cache['timestamps_sha256']!=hashlib.sha256(canonical(base['timestamps'])).hexdigest() or cache['preprocessing']!=a['preprocessing']:raise ValueError('Timing/preprocessing differs')
  detections_path=BASE/key/'detections.json';external_case=None
  if external:
   external_case=external['videos'][key];detections_path=Path(external_case['detections_path'])
   if external_case['source_sha256']!=row['sha256'] or sha(detections_path)!=external_case['detections_sha256'] or sha(external_case['report_path'])!=external_case['report_sha256']:raise ValueError('External measurement provenance differs')
   external_records=load(detections_path);frozen_records=load(BASE/key/'detections.json')
   if [(v['frame_index'],v['time']) for v in external_records]!=[(v['frame_index'],v['time']) for v in frozen_records]:raise ValueError('External records changed source frame indices/timestamps')
  if args.preflight_only:print(key,'PREFLIGHT_OK',flush=True);continue
  frames=[frames[i] for i in range(len(frames))]  # Same list contract as production, actual memmap pixel views.
  records=load(detections_path);original_records=records;old_hash=hashlib.sha256(canonical(records)).hexdigest();start=time.perf_counter()
  tracks=track_detections(records,config.get('tracker'));cfg=copy.deepcopy(config.get('events') or {});cfg['frame_preprocessing']=a['preprocessing'];cfg['review_refinement_regions']=[]
  identity=copy.deepcopy(config.get('identity') or {})
  if identity.get('event_window_seconds') is not None:cfg['event_window_seconds']=identity['event_window_seconds']
  landings=estimate_landings(tracks,frames,base['timestamps'],base['calibration'],cfg);initial_seconds=time.perf_counter()-start;support_refinement=None
  if cfg.get('temporal_observation_support',False):
   from badminton.review_support import annotate_event_neighborhoods
   records,support_refinement=annotate_event_neighborhoods(records,frames,tracks,landings,preserve_current_detector_evidence=False)
   print(key,'support assessed',support_refinement['assessed_observations'],'observations',flush=True)
   if support_refinement['assessed_observations']:
    retry_start=time.perf_counter();tracks=track_detections(records,config.get('tracker'));landings=estimate_landings(tracks,frames,base['timestamps'],base['calibration'],cfg);support_refinement['retracking_and_events_seconds']=time.perf_counter()-retry_start;support_refinement['retry_count']=1
   else:support_refinement['retry_count']=0
   if support_refinement.get('preserved_current_native_observations') or support_refinement.get('prior_annotations_reused'):raise ValueError('Cached native evidence was incorrectly reused')
   unannotated=copy.deepcopy(records)
   for row_record in unannotated:
    for observation in row_record.get('detections',[]):observation.pop('observation_support',None)
   raw=copy.deepcopy(original_records)
   for row_record in raw:
    for observation in row_record.get('detections',[]):observation.pop('observation_support',None)
   if canonical(raw)!=canonical(unannotated):raise ValueError('Support wrapper changed the raw proposal pool')
  landings=apply_identity(landings,tracks,base['timestamps'],base['calibration'],identity)
  for i,landing in enumerate(landings):landing['event_id']=f'event_{i:03d}';landing['decision']=decide(landing,base['calibration'],config.get('rules'))
  requested=select_requested_landings(landings,identity);ranked=prioritize_review_candidates(requested,[]);selected=ranked[0] if ranked else None
  decision=aggregate_decision([e['decision'] for e in requested])
  if hashlib.sha256(canonical(original_records)).hexdigest()!=old_hash:raise ValueError('Input candidate observations mutated')
  out=args.run/key;out.mkdir();result=copy.deepcopy(base);result.update(tracks=tracks,landings=landings,selected_landing=selected,decision=decision,config=config,implementation_fingerprint=fp,assistance_used=False,automatic_evaluation_eligible=True,artifacts={},timings={'total_seconds':time.perf_counter()-start,'timing_scope':'tracking_events_only_no_decode_detector_native_evidence'})
  result.update(pipeline_version=__version__,detection_quality=summarize_detection_quality(records,tracks),identity={'requested_event_count':len(requested),'hit_anchor_count':0,'event_window_seconds':identity.get('event_window_seconds')},review_refinement={'status':'not_requested','regions':[],'matching_event_ids':[],'selected_event_id':selected['event_id'] if selected else None,'selected_in_requested_region':False,'assistance_used':False},model_fingerprints=model_fingerprints(config),scope=provenance['scope'])
  result['motion_support_refinement']=support_refinement;result['timings']['initial_tracking_and_events_seconds']=initial_seconds
  result['analysis']['source_scaling']='Reused native detector observations; actual analysis pixels; native contact refinement omitted'
  result['experiment_provenance']={'mode':provenance['mode'],**code,'source_sha256':row['sha256'],'frame_cache_provenance_sha256':sha(ROOT/'frame_cache'/key/'provenance.json'),'source_analysis_sequence_sha256':cache['analysis_sequence_sha256'],'source_detections_sha256':sha(BASE/key/'detections.json'),'source_result_sha256':sha(BASE/key/'result.json'),'baseline_implementation_fingerprint':base['implementation_fingerprint'],'omitted_stages':provenance['omitted_stages'],'reference_labels_read':False,'fixed_observations_component_ablation':args.fixed_observations,'uncapped_events':args.uncapped_events,'frozen_detector_config':base['config'].get('detector'),'current_detector_config_not_executed':config.get('detector'),'detector_and_calibration_code':stage_code,'effective_config_sha256':provenance['effective_config_sha256']}
  result['validation_run']={'detector_stage_reused':True,'mode':provenance['mode'],'timing_scope':result['timings']['timing_scope']}
  result['experiment_provenance']['support_wrapper']=provenance['support_wrapper'];result['experiment_provenance']['output_detections_canonical_sha256']=hashlib.sha256(canonical(records)).hexdigest()
  if external_case:
   result['experiment_provenance']['frozen_release_detections_sha256']=result['experiment_provenance']['source_detections_sha256'];result['experiment_provenance']['source_detections_sha256']=sha(detections_path);result['experiment_provenance']['external_measured_proposals']=external_case
  (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');(out/'detections.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
  print(key,'completed',round(result['timings']['total_seconds'],3),'seconds',flush=True)
if __name__=='__main__':main()
