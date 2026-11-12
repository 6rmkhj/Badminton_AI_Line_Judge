"""Final frozen old24 stage + released new8 full-fresh evaluation."""
from pathlib import Path
import hashlib,importlib.util,json
R=Path('/workspace/remediation')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kind(result):
 s=result.get('selected_landing') or {};m=s.get('image_motion_event') or {}
 if not s:return 'no_candidate'
 if m.get('point_kind')=='inferred_object_center' or 'occluded' in str(s.get('method')):return 'inferred_object_center'
 return m.get('point_kind','unspecified_candidate_center')
def signature(result):
 s=result.get('selected_landing') or {};t=next((t for t in result['tracks'] if t['track_id']==s.get('track_id')),{});return {'frame_index':s.get('frame_index'),'candidate_px':s.get('candidate_px'),'observations':[[o.get(k) for k in ['frame_index','time','x','y','radius','observed','interpolated']] for o in t.get('observations',[])]}
def main():
 freeze=load(R/'final-evaluation-freeze.json');release=load(R/'new8-release-receipt.json');assert release['preinspection_freeze_sha256']==sha(R/'final-evaluation-freeze.json');assert release['all8_full_fresh_completed'] is True
 spec=importlib.util.spec_from_file_location('audit','/workspace/generalization/audit.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
 assert sha(Path('/workspace/generalization/audit.py'))==freeze['criterion_auditor_sha256']
 for v in load(R/'regression-reference-inventory.json')['references']:assert sha(Path(v['path']))==v['sha256']
 inventory=load(R/'sealed_references/freeze-manifest.json');assert sha(R/'sealed_references/freeze-manifest.json')==freeze['new8_reference_inventory_sha256'];ref_hashes={Path(k).stem:v for k,v in inventory['reference_sha256'].items()}
 for key,value in ref_hashes.items():assert sha(R/'sealed_references'/f'{key}.json')==(value if isinstance(value,str) else value['sha256'])
 binding={'run_implementation_fingerprint':freeze['accepted_core']['implementation_fingerprint'],'analysis_config_sha256':freeze['canonical_config_sha256']};cohorts={}
 for name,manifest,run,refs,split in [('old24_stage','regression-manifest.json','final-old24-stage-v1','regression_references','development'),('new8_full_fresh','holdout-manifest.json','final-fresh-0.5.4','sealed_references','holdout')]:
  rows=load(R/manifest)['videos'];records=[]
  for row in rows:
   key=row['key'];path=R/run/key/'result.json';result=load(path);assert result['implementation_fingerprint']==binding['run_implementation_fingerprint'];assert a.config_fingerprint(result['config'])==binding['analysis_config_sha256'];assert result['source_sha256']==row['sha256'];assert result.get('assistance_used') is False and result.get('automatic_evaluation_eligible') is True
   record=a.audit_video(row,R/run,R/refs,R/'final-reference-snapshots',split=split,release=binding);assert record['status']=='audited';record.update(result_sha256=sha(path),detections_sha256=sha(R/run/key/'detections.json'),assistance_used=result.get('assistance_used'),automatic_evaluation_eligible=result.get('automatic_evaluation_eligible'),candidate_kind=kind(result),execution_scope='tracking_events_actual_pixels_detector_calibration_reused_native_contact_evidence_omitted' if split=='development' else 'full_fresh_pipeline')
   if split=='development':
    record['execution_provenance']=result['experiment_provenance'];record['matches_reviewed_scene_stage_selection_and_trajectory']=signature(result)==signature(load(R/'scene-context-on-v1'/key/'result.json'))
   else:
    receipt=load(R/run/key/'execution-receipt.json');assert receipt['result_sha256']==sha(path) and receipt['detector_stage_reused'] is False;record['execution_receipt_sha256']=sha(R/run/key/'execution-receipt.json')
   records.append(record)
  cohorts[name]={'metrics':a.metrics(records),'cases':records,'scope':'All old24 are development; actual-pixel stage only' if split=='development' else 'Same-project current-round holdouts, full fresh after freeze; AI source references, not independent human contact truth'}
 out={'schema_version':1,'accepted_code_freeze':freeze['accepted_core'],'preinspection_freeze_sha256':sha(R/'final-evaluation-freeze.json'),'holdout_release_receipt_sha256':sha(R/'new8-release-receipt.json'),'protocol_sha256':freeze['protocol_sha256'],'old24_reference_inventory_sha256':freeze['old24_reference_inventory_sha256'],'new8_reference_inventory_sha256':freeze['new8_reference_inventory_sha256'],'old24_manifest_sha256':freeze['old24_manifest_sha256'],'new8_manifest_sha256':freeze['new8_manifest_sha256'],'canonical_config_sha256':freeze['canonical_config_sha256'],'criterion_auditor_sha256':freeze['criterion_auditor_sha256'],'evaluator_sha256':sha(Path(__file__)),'cohorts':cohorts,'limits':'Strict agreement with frozen approximate source-only AI frame/point references. Views are correlated within pairs. Not physical-contact or IN/OUT accuracy. Development-stage and fresh-heldout scopes stay separate.'};(R/'final-cohort-audit.json').write_text(json.dumps(out,indent=2)+'\n')
 for name,cohort in cohorts.items():
  print(name,cohort['metrics']);print('SELECTIONS',[(v['key'],v['selection_category'],v['selected']['frame_index'] if v.get('selected') else None,v['candidate_kind']) for v in cohort['cases']])
 print('OLD24_EXACT_SCENE_MATCH',all(v['matches_reviewed_scene_stage_selection_and_trajectory'] for v in cohorts['old24_stage']['cases']))
if __name__=='__main__':main()
