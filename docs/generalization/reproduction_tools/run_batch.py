import os,sys,json,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import cv2
root=Path(sys.argv[4]) if len(sys.argv)>4 else Path('/workspace/generalization');repo=Path(sys.argv[1]);run=root/sys.argv[2];split=sys.argv[3] if len(sys.argv)>3 else 'all'
sys.path.insert(0,str(repo))
from badminton.pipeline import analyze_video,implementation_fingerprint
cv2.setNumThreads(2)
manifest=json.loads((root/'manifest.json').read_text());config=json.loads((repo/'configs/poc.json').read_text()); fingerprint=implementation_fingerprint()
rows=[r for r in manifest['videos'] if split=='all' or r['split']==split];rows.sort(key=lambda row: row['split'] != 'development');run.mkdir(parents=True,exist_ok=True)
if split not in {'all','development','holdout'}:raise ValueError('Choose all, development, or holdout')
for row in rows:
 row['path']=str(Path(row['path']) if row.get('path') else root/'input'/row['file'])
 if not Path(row['path']).is_file():raise FileNotFoundError('Missing predeclared input: '+row['path'])
(run/'run-config.json').write_text(json.dumps({'repo':str(repo),'config':config,'implementation_fingerprint':fingerprint,'split':split},indent=2)+'\n')
def process(row):
 out=run/row['key'];existing=out/'result.json'
 if existing.exists():
  result=json.loads(existing.read_text())
  if (not (out/'error.txt').exists() and result.get('implementation_fingerprint')==fingerprint and result.get('config')==config
      and result.get('source_sha256')==row['sha256']):return row['key'],'cached'
 previous=[None]
 def progress(e):
  if e.get('stage')!=previous[0]:
   previous[0]=e.get('stage');print(row['key'],e['stage'],flush=True)
 try:
  result=analyze_video(row['path'],out,config,progress_callback=progress)
  if result.get('implementation_fingerprint')!=fingerprint or implementation_fingerprint()!=fingerprint:
   raise RuntimeError('Implementation changed during this analysis; discard the mixed-code run and rerun from frozen source.')
  (out/'error.txt').unlink(missing_ok=True)
  selected=result.get('selected_landing') or {}
  stats={'key':row['key'],'status':'completed','frame':selected.get('frame_index'),'point':selected.get('candidate_px'),'time':selected.get('time'),'total_seconds':result['timings']['total_seconds']}
 except Exception as e:
  stats={'key':row['key'],'status':'failed','error':str(e)};out.mkdir(exist_ok=True);(out/'error.txt').write_text(traceback.format_exc())
 if row['split']=='holdout':
  stats={key:value for key,value in stats.items() if key not in {'frame','point','time'}}
 print(json.dumps(stats),flush=True);return stats
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(process,row) for row in rows]
 for f in as_completed(futures):f.result()
print('BATCH_COMPLETE',len(rows),flush=True)
