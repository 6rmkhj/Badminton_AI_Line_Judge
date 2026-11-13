import sys,json,time,copy,math
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path.cwd()))
import badminton.landing_recovery as recovery
from badminton.tracking import _flight_fit
from badminton.media import SourceFrameSequence,load_frames
from badminton.events import estimate_landings
cv2.setNumThreads(1)
old_approaches=recovery._approaches

def prefixes(tracks,budget):
 extended=list(tracks);seen=set()
 for track in tracks:
  fh=track.get('flight_hypothesis')
  if not isinstance(fh,dict):continue
  obs=track['observations'][:fh['approach_count']]
  for i in range(6,len(obs)-2):
   history=[o for o in obs[:i] if obs[i-1]['time']-o['time']<=.35]
   if len(history)<6 or history[-1]['time']-history[0]['time']<.16:continue
   errors=[]
   for o in obs[i:i+3]:
    p,_,resid=_flight_fit(history,o['time'],window=len(history));errors.append(math.dist(p,(o['x'],o['y'])))
   radius=float(np.median([o['radius'] for o in history]))
   if resid>max(2.,radius*1.5) or max(errors)<=max(4.,radius*2):continue
   signature=tuple((o['frame_index'],o['x'],o['y']) for o in history)
   if signature in seen:continue
   seen.add(signature)
   extended.append({**track,'flight_hypothesis':{**fh,'approach_count':i,'rebound_count':0},'observations':obs[:i]})
 selected=old_approaches(extended,budget)
 print('prefixcount',len(extended)-len(tracks),'selected',[(t['track_id'],o[-1]['frame_index'],round(o[-1]['x'],1),round(o[-1]['y'],1)) for t,o,v,ra in selected],flush=True)
 return selected
recovery._approaches=prefixes
for key in sys.argv[1:]:
 r=json.load(open(f'runs/landing-engine/baseline/{key}/result.json'));source=r['source_path'];start=time.monotonic()
 with SourceFrameSequence(source,len(r['timestamps'])) as originals:
  recovered,summary=recovery.recover_rebound_tracks(r['tracks'],originals,r['timestamps'],(r['analysis']['width'],r['analysis']['height']))
 frames=load_frames(source,len(r['timestamps']),3000,resize_to=(r['analysis']['width'],r['analysis']['height']))
 candidates=estimate_landings(r['tracks']+recovered,frames,r['timestamps'],r['calibration'],r['config']['events'])
 print(key,summary,'seconds',time.monotonic()-start,flush=True)
 print([(c['track_id'],c['frame_index'],[round(v,2) for v in c['candidate_px']],(c.get('flight_trajectory_support') or {}).get('supported_for_review')) for c in candidates[:12]],flush=True)
 Path(f'runs/landing-engine/observation-audit/{key}-prefix.json').write_text(json.dumps({'summary':summary,'tracks':recovered,'landings':candidates},indent=2))
