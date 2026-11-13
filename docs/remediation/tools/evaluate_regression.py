"""Strict old24 development evaluator; new heldout keys/references are inaccessible."""
import argparse,hashlib,importlib.util,json,math,re
from collections import Counter
from pathlib import Path
ROOT=Path('/workspace/remediation');BASE=Path('/workspace/generalization/release')
AUDITOR=Path('/workspace/generalization/audit.py')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def identity(result):
 s=result.get('selected_landing') or {};return {'frame_index':s.get('frame_index'),'candidate_px':s.get('candidate_px')}
def trajectory(result):
 s=result.get('selected_landing') or {};track=next((t for t in result.get('tracks',[]) if t.get('track_id')==s.get('track_id')),{});keys=['frame_index','time','x','y','radius','observed','interpolated']
 return [[o.get(k) for k in keys] for o in track.get('observations',[])]
def candidate_kind(result):
 s=result.get('selected_landing') or {};motion=s.get('image_motion_event') or {}
 if not s:return 'no_candidate'
 if motion.get('point_kind')=='inferred_object_center' or 'occluded' in str(s.get('method')):return 'inferred_object_center'
 if motion.get('point_kind')=='observed_object_center':return 'observed_object_center'
 return 'unspecified_candidate_center'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline-run',type=Path,default=BASE);p.add_argument('--output',type=Path,required=True);p.add_argument('--key',action='append',default=[]);p.add_argument('--qualitative-review',type=Path);p.add_argument('--review-selected-changes',action='store_true',help='Queue source review for any changed selected frame/point or trajectory, including numerically unscorable cases');args=p.parse_args()
 manifest=load(ROOT/'regression-manifest.json');inventory=load(ROOT/'regression-reference-inventory.json');rows=manifest['videos'];allowed={r['key'] for r in rows}
 if not set(args.key)<=allowed:raise ValueError('Only the old24 regression keys are allowed; new holdouts are sealed')
 for entry in inventory['references']:
  if sha(entry['path'])!=entry['sha256']:raise ValueError('Immutable reference changed: '+entry['key'])
 if args.key:rows=[r for r in rows if r['key'] in args.key]
 spec=importlib.util.spec_from_file_location('frozen_auditor',AUDITOR);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 reviews=load(args.qualitative_review) if args.qualitative_review else {'cases':[]};review_index={r['key']:r for r in reviews['cases']}
 records=[]
 for row in rows:
  key=row['key'];reference=ROOT/'regression_references'/f'{key}.json'
  full_before=a.audit_video(row,BASE,ROOT/'regression_references',ROOT/'reference_snapshots',split='development')
  before=full_before if args.baseline_run.resolve()==BASE.resolve() else a.audit_video(row,args.baseline_run,ROOT/'regression_references',ROOT/'reference_snapshots',split='development')
  if before['status']!='audited':raise ValueError('Missing/invalid comparison baseline: '+key)
  after=a.audit_video(row,args.run,ROOT/'regression_references',ROOT/'reference_snapshots',split='development')
  base=load(BASE/key/'result.json');comparison_base=base if args.baseline_run.resolve()==BASE.resolve() else load(args.baseline_run/key/'result.json');current=load(args.run/key/'result.json') if after['status']=='audited' else None
  if current:
   mode=(current.get('experiment_provenance') or {}).get('mode')
   if mode and mode.startswith('tracking_events_'):
    if args.baseline_run.resolve()==BASE.resolve():raise ValueError('Stage experiment requires --baseline-run pointing to the same-stage baseline')
    stage_base=load(args.baseline_run/key/'result.json');previous=stage_base.get('experiment_provenance') or {}
    if previous.get('mode')!=mode or any(previous.get(k)!=(current.get('experiment_provenance') or {}).get(k) for k in ['source_sha256','source_analysis_sequence_sha256','source_detections_sha256']):raise ValueError('Stage baseline uses different stages or source observations')
   if not re.fullmatch('[0-9a-f]{64}',str(current.get('implementation_fingerprint',''))):raise ValueError('Missing experiment fingerprint: '+key)
   if current.get('assistance_used') is not False or current.get('automatic_evaluation_eligible') is not True:raise ValueError('Assisted result cannot enter automatic regression score: '+key)
  old_match=before.get('selection_category')=='correct_first_vicinity';new_match=after.get('selection_category')=='correct_first_vicinity'
  protected=full_before.get('selection_category')=='correct_first_vicinity' or key=='ordinary-12-a';same=bool(current and identity(base)==identity(current));review=review_index.get(key);review_status=None
  if review:
   if not current or review.get('result_sha256')!=sha(args.run/key/'result.json') or review.get('reference_sha256')!=sha(reference) or identity({'selected_landing':review.get('selection')})!=identity(current):raise ValueError('Stale qualitative review: '+key)
   if review.get('baseline_result_sha256') and review['baseline_result_sha256']!=sha(args.baseline_run/key/'result.json'):raise ValueError('Stale qualitative comparison baseline: '+key)
   review_status=review.get('status')
  if protected and not new_match and not same and not review_status:review_status='needs_source_review_protected_object_changed'
  if new_match and not old_match and not review_status:review_status='needs_source_review_new_strict_gain'
  selected_changed=bool(current and identity(comparison_base)!=identity(current));trajectory_changed=bool(current and trajectory(comparison_base)!=trajectory(current))
  if args.review_selected_changes and (selected_changed or trajectory_changed) and not review_status:review_status='needs_source_review_selected_change'
  s=(current or {}).get('selected_landing') or {};track=next((t for t in (current or {}).get('tracks',[]) if t['track_id']==s.get('track_id')),{});obs=track.get('observations',[])
  records.append({'key':key,'source_sha256':row['sha256'],'reference_sha256':sha(reference),'before':before,'frozen_full_baseline':full_before,'after':after,'strict_gain':new_match and not old_match,'strict_regression':old_match and not new_match,'baseline_qualitative_object_protected':protected,'qualitative_status':review_status,'same_selection_as_full_baseline':same,'candidate_kind':candidate_kind(current or {}),'contact_observed':s.get('contact_observed',False),'all_track_entries_measured':bool(obs) and all(o.get('observed') is True for o in obs),'track_observation_count':len(obs),'candidate_kind_baseline':candidate_kind(base),'execution_scope':(current or {}).get('experiment_provenance') or (current or {}).get('validation_run') or {'mode':'full_pipeline_artifact'},'experiment_result_sha256':sha(args.run/key/'result.json') if current else None})
  records[-1].update(selected_frame_point_changed=selected_changed,selected_trajectory_changed=trajectory_changed,qualitative_change_assessment=(review or {}).get('change_assessment'),qualitative_regression=bool((review or {}).get('change_assessment')=='regression' or review_status=='regression_wrong_object' or (review_status=='wrong_object' and protected)))
 metrics=a.metrics([r['after'] for r in records]);baseline=a.metrics([r['before'] for r in records]);full_baseline=a.metrics([r['frozen_full_baseline'] for r in records])
 if len(rows)==24 and (full_baseline['first_vicinity_match_count'],full_baseline['first_vicinity_denominator'])!=(5,13):raise ValueError('Frozen5/13 baseline changed')
 out={'scope':'All old24 are development/regression for this round. Strict original13-visible criteria unchanged. Stage-only outcomes are diagnostics, not full fresh end-to-end accuracy.','new_holdout_files_opened':False,'manifest_sha256':sha(ROOT/'regression-manifest.json'),'reference_inventory_sha256':sha(ROOT/'regression-reference-inventory.json'),'auditor_sha256':sha(AUDITOR),'evaluator_sha256':sha(Path(__file__)),'run':str(args.run),'comparison_baseline_run':str(args.baseline_run),'baseline_metrics':baseline,'frozen_full_baseline_metrics':full_baseline,'experiment_metrics':metrics,'strict_gains':[r['key'] for r in records if r['strict_gain']],'strict_regressions':[r['key'] for r in records if r['strict_regression']],'qualitative_regressions':[r['key'] for r in records if r['qualitative_regression']],'source_reviews_pending':[r['key'] for r in records if (r['qualitative_status'] or '').startswith('needs_source_review')],'selected_frame_point_changes':[r['key'] for r in records if r['selected_frame_point_changed']],'selected_trajectory_changes':[r['key'] for r in records if r['selected_trajectory_changed']],'candidate_kinds':dict(Counter(r['candidate_kind'] for r in records)),'strict_match_candidate_kinds':dict(Counter(r['candidate_kind'] for r in records if r['after'].get('selection_category')=='correct_first_vicinity')),'videos':records}
 write(args.output,out);print(json.dumps({k:out[k] for k in ['strict_gains','strict_regressions','qualitative_regressions','source_reviews_pending','candidate_kinds','strict_match_candidate_kinds']}))
if __name__=='__main__':main()
