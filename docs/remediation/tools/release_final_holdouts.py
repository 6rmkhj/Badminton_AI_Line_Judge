"""Release final evaluation only after all eight frozen predictions complete."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys
R=Path('/workspace/remediation')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def main():
 freeze_path=R/'final-evaluation-freeze.json';freeze=load(freeze_path);rows=load(R/'holdout-manifest.json')['videos'];state=load(R/'final-fresh-0.5.4/status.json')
 if len(rows)!=8 or any(state.get(v['key'],{}).get('status')!='completed' for v in rows):raise ValueError('All eight frozen full predictions must complete before release')
 repo=Path(freeze['accepted_core']['source_path']);assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==freeze['accepted_core']['source_commit'];assert sha(R/'final-code-freeze.json')==freeze['accepted_core_record_sha256'];assert sha(repo/'configs/poc.json')==freeze['accepted_core']['config_sha256']
 for name,digest in freeze['source_files'].items():assert sha(repo/name)==digest
 for name,field in [('protocol.json','protocol_sha256'),('regression-manifest.json','old24_manifest_sha256'),('regression-reference-inventory.json','old24_reference_inventory_sha256'),('holdout-manifest.json','new8_manifest_sha256'),('reference-readiness.json','new8_reference_readiness_sha256'),('final-fresh-manifest.json','fresh_manifest_sha256')]:assert sha(R/name)==freeze[field]
 assert sha(Path('/workspace/generalization/audit.py'))==freeze['criterion_auditor_sha256'];sys.path.insert(0,str(repo));from badminton.pipeline import implementation_fingerprint
 assert implementation_fingerprint()==freeze['accepted_core']['implementation_fingerprint']
 items=[]
 for row in rows:
  key=row['key'];folder=R/'final-fresh-0.5.4'/key;receipt=load(folder/'execution-receipt.json');assert receipt==state[key];assert receipt['mode']=='full_fresh_pipeline' and receipt['detector_stage_reused'] is False;assert receipt['source_sha256']==row['sha256'];assert receipt['implementation_fingerprint']==freeze['accepted_core']['implementation_fingerprint'];assert receipt['canonical_config_sha256']==freeze['canonical_config_sha256'];assert receipt['freeze_sha256']==sha(freeze_path);assert sha(folder/'result.json')==receipt['result_sha256'];assert not (folder/'error.txt').exists();items.append({'key':key,'result_sha256':receipt['result_sha256'],'execution_receipt_sha256':sha(folder/'execution-receipt.json'),'completed_at':receipt['completed_at']})
 inventory=R/'sealed_references/freeze-manifest.json';assert sha(inventory)==freeze['new8_reference_inventory_sha256']
 output=R/'new8-release-receipt.json'
 if output.exists():raise ValueError('Release receipt already exists; do not overwrite')
 shutil.copyfile(freeze_path,R/'final-evaluation-freeze.preinspection.json')
 receipt={'released_at':datetime.now(timezone.utc).isoformat(),'authorization':'Root explicitly authorized final new8 inspection only after all eight predictions complete under7306/ec9a727 frozen source/config; no algorithm or criterion tuning afterward.','preinspection_freeze_sha256':sha(freeze_path),'source_commit':freeze['accepted_core']['source_commit'],'implementation_fingerprint':freeze['accepted_core']['implementation_fingerprint'],'config_sha256':freeze['accepted_core']['config_sha256'],'canonical_config_sha256':freeze['canonical_config_sha256'],'new8_manifest_sha256':freeze['new8_manifest_sha256'],'new8_reference_inventory_sha256':freeze['new8_reference_inventory_sha256'],'all8_full_fresh_completed':True,'before_inspection_code_config_checks_passed':True,'predictions':items};output.write_text(json.dumps(receipt,indent=2)+'\n');print('RELEASED_ALL8',sha(output));data=load(inventory);print('REFERENCE_INVENTORY_SCHEMA',{k:type(v).__name__ for k,v in data.items()})
 for k,v in data.items():
  if isinstance(v,list) and v and isinstance(v[0],dict):print('ENTRY_FIELDS',k,list(v[0]))
if __name__=='__main__':main()
