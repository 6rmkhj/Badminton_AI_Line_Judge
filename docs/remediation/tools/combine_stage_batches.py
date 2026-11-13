"""Join completed old24 stage batches while retaining exact per-batch provenance."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path('/workspace/remediation')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',action='append',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 allowed={r['key'] for r in load(ROOT/'regression-manifest.json')['videos']};found={};common=None;batches=[]
 for batch in args.batch:
  cfg=load(batch/'run-config.json');signature={k:cfg[k] for k in ['mode','code','config']}
  if common is not None and signature!=common:raise ValueError('Batches used different source, effective config or stage scope')
  common=signature;entries=[]
  for key in sorted(allowed):
   path=batch/key/'result.json'
   if not path.exists():continue
   if key in found:raise ValueError('Duplicate key: '+key)
   result=load(path);provenance=result['experiment_provenance']
   if result['implementation_fingerprint']!=common['code']['implementation_fingerprint'] or result['config']!=common['config'] or provenance['source_tree_sha256']!=common['code']['source_tree_sha256'] or provenance['mode']!=common['mode']:raise ValueError('Per-result source/config/stage provenance differs')
   found[key]=batch/key;entries.append({'key':key,'result_sha256':sha(path),'detections_sha256':sha(batch/key/'detections.json')})
  batches.append({'path':str(batch),'run_config_sha256':sha(batch/'run-config.json'),'results':entries})
 if set(found)!=allowed:raise ValueError('All old24 must complete before combination: missing '+str(sorted(allowed-set(found))))
 if args.output.exists():raise ValueError('Use a fresh compound output directory')
 args.output.mkdir(parents=True)
 for key,folder in found.items():(args.output/key).symlink_to(folder.resolve(),target_is_directory=True)
 (args.output/'compound-run.json').write_text(json.dumps({'mode':'compound_exact_stage_batches','scope':common,'batches':batches,'new_holdout_files_opened':False},indent=2)+'\n')
 print('combined',len(found),'exact-snapshot results')
if __name__=='__main__':main()
