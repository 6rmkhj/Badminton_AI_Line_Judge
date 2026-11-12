from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
R=Path('/workspace/remediation');repo=Path('/workspace/badminton-release-0.5.4')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
f=load(R/'final-code-freeze.json');assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==f['source_commit'];assert sha(repo/'configs/poc.json')==f['config_sha256'];config=load(repo/'configs/poc.json')
holdouts=load(R/'holdout-manifest.json')['videos'];old={v['key']:v for v in load(R/'regression-manifest.json')['videos']};rows=[dict(v,evaluation_role='new_round_holdout_full_fresh') for v in sorted(holdouts,key=lambda v:-v['bytes'])]+[dict(old[k],evaluation_role='development_gain_full_fresh_check') for k in ['ordinary-4-a','ordinary-12-a']]
manifest={'videos':rows,'scope':'Eight predeclared round holdouts followed by two fresh development gain checks; controls run separately. Source references never enter inference.'};manifest_path=R/'final-fresh-manifest.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
source_files={str(p.relative_to(repo)):sha(p) for p in sorted((repo/'badminton').rglob('*.py'))};readiness=load(R/'reference-readiness.json')
freeze={'status':'sealed_before_fresh_predictions','created_at':datetime.now(timezone.utc).isoformat(),'accepted_core':f,'accepted_core_record_sha256':sha(R/'final-code-freeze.json'),'config':config,'canonical_config_sha256':hashlib.sha256(canonical(config)).hexdigest(),'source_files':source_files,'source_tree_sha256':hashlib.sha256(canonical(source_files)).hexdigest(),'protocol_sha256':sha(R/'protocol.json'),'old24_manifest_sha256':sha(R/'regression-manifest.json'),'old24_reference_inventory_sha256':sha(R/'regression-reference-inventory.json'),'new8_manifest_sha256':sha(R/'holdout-manifest.json'),'new8_reference_inventory_sha256':readiness['freeze_manifest_sha256'],'new8_reference_readiness_sha256':sha(R/'reference-readiness.json'),'fresh_manifest_sha256':sha(manifest_path),'criterion_auditor_sha256':sha(Path('/workspace/generalization/audit.py')),'planned_runs':{'old24_stage':str(R/'final-old24-stage-v1'),'new8_and_two_gain_checks_full_fresh':str(R/'final-fresh-0.5.4'),'original_controls_full_fresh':str(R/'release-controls-0.5.4')},'holdout_release_rule':'Root explicitly authorized final inspection only after all eight predictions complete under this exact source/config freeze; no algorithm/criteria tuning afterward.','new8_positions_or_predictions_opened_by_evaluator':False}
p=R/'final-evaluation-freeze.json'
if p.exists():raise ValueError('Preinspection freeze already exists; do not overwrite')
p.write_text(json.dumps(freeze,indent=2)+'\n');(R/'final-fresh-0.5.4').mkdir(exist_ok=True)
print('SEALED',sha(p),'fresh tasks',len(rows))
