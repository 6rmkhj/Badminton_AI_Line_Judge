import sys,json,time
from pathlib import Path
import cv2
sys.path.insert(0,str(Path.cwd()))
from badminton.landing_recovery import recover_rebound_tracks
from badminton.media import SourceFrameSequence,load_frames
from badminton.events import estimate_landings,prioritize_review_candidates
cv2.setNumThreads(1)
for key in sys.argv[1:]:
 r=json.load(open(f'runs/landing-engine/baseline/{key}/result.json')); source=r['source_path']; start=time.monotonic()
 with SourceFrameSequence(source,len(r['timestamps'])) as originals:
  recovered,summary=recover_rebound_tracks(r['tracks'],originals,r['timestamps'],(r['analysis']['width'],r['analysis']['height']),max_approaches=48)
 frames=load_frames(source,len(r['timestamps']),3000,resize_to=(r['analysis']['width'],r['analysis']['height']))
 candidates=estimate_landings(r['tracks']+recovered,frames,r['timestamps'],r['calibration'],r['config']['events'])
 print(key,summary,'seconds',time.monotonic()-start,flush=True)
 print([(c['track_id'],c['frame_index'],[round(v,2) for v in c['candidate_px']],(c.get('flight_trajectory_support') or {}).get('supported_for_review')) for c in candidates[:12]],flush=True)
 Path(f'runs/landing-engine/observation-audit/{key}-recovery48.json').write_text(json.dumps({'summary':summary,'tracks':recovered,'landings':candidates},indent=2))
