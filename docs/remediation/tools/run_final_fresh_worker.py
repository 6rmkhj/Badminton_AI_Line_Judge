"""Frozen full-pipeline worker; logs no heldout selection or reference content."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys,time,traceback
from filelock import FileLock
ROOT=Path('/workspace/remediation')
def load(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',required=True);args=parser.parse_args();freeze_path=ROOT/'final-evaluation-freeze.json';freeze=load(freeze_path);repo=Path(freeze['accepted_core']['source_path']);out=ROOT/'final-fresh-0.5.4';manifest=ROOT/'final-fresh-manifest.json'
 if sha(manifest)!=freeze['fresh_manifest_sha256']:raise ValueError('Frozen manifest changed')
 rows=load(manifest)['videos'];sys.path.insert(0,str(repo));import cv2;cv2.setNumThreads(1)
 from badminton.pipeline import analyze_video,implementation_fingerprint
 expected=freeze['accepted_core']['implementation_fingerprint'];status_path=out/'status.json';lock=FileLock(str(out/'.queue.lock'))
 def verify():
  if sha(repo/'configs/poc.json')!=freeze['accepted_core']['config_sha256'] or implementation_fingerprint()!=expected:raise ValueError('Frozen source/config changed')
  if any(sha(repo/name)!=digest for name,digest in freeze['source_files'].items()):raise ValueError('Frozen package source changed')
 verify();write(out/f'worker-{args.worker}.json',{'worker':args.worker,'pid':os.getpid(),'freeze_sha256':sha(freeze_path),'runner_sha256':sha(Path(__file__)),'opencv_threads':1,'heldout_selection_logging':False})
 while True:
  with lock:
   state=load(status_path) if status_path.exists() else {}
   row=next((v for v in rows if v['key'] not in state),None)
   if row is None:break
   key=row['key'];state[key]={'status':'running','worker':args.worker,'started_at':datetime.now(timezone.utc).isoformat(),'evaluation_role':row['evaluation_role']};write(status_path,state)
  print(key,'START',row['evaluation_role'],flush=True);previous=[None];start=time.perf_counter()
  def progress(event):
   stage=event.get('stage')
   if stage!=previous[0]:previous[0]=stage;print(key,stage,flush=True)
  try:
   verify()
   if sha(Path(row['path']))!=row['sha256']:raise ValueError('Predeclared source bytes differ')
   folder=out/key
   if (folder/'result.json').exists():raise ValueError('Fresh result already exists')
   result=analyze_video(row['path'],folder,freeze['config'],progress_callback=progress);verify()
   if result.get('implementation_fingerprint')!=expected or result.get('source_sha256')!=row['sha256'] or result.get('config')!=freeze['config']:raise ValueError('Result binding differs from frozen source/config/input')
   receipt={'status':'completed','worker':args.worker,'evaluation_role':row['evaluation_role'],'source_sha256':row['sha256'],'result_sha256':sha(folder/'result.json'),'implementation_fingerprint':expected,'canonical_config_sha256':freeze['canonical_config_sha256'],'mode':'full_fresh_pipeline','detector_stage_reused':False,'elapsed_seconds':time.perf_counter()-start,'completed_at':datetime.now(timezone.utc).isoformat(),'freeze_sha256':sha(freeze_path)};write(folder/'execution-receipt.json',receipt)
  except Exception:
   folder=out/key;folder.mkdir(exist_ok=True);(folder/'error.txt').write_text(traceback.format_exc());receipt={'status':'failed','worker':args.worker,'evaluation_role':row['evaluation_role'],'elapsed_seconds':time.perf_counter()-start,'error_log':str(folder/'error.txt')}
  with lock:
   state=load(status_path);state[key]=receipt;write(status_path,state)
  print(key,receipt['status'],round(receipt['elapsed_seconds'],2),'seconds',flush=True)
 print('WORKER_COMPLETE',args.worker,flush=True)
if __name__=='__main__':main()
