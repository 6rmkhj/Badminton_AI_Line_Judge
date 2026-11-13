"""Development checks against AI visual inspection, not independent ground truth."""
import json,math,sys
from pathlib import Path
run=Path(sys.argv[1]);out={'reference_provenance':'AI visual inspection of original frames; development checks only, not independently measured accuracy','views':{}}
references={'A':{'frames':[48,50],'point':[361,442],'radius':12,'points':{43:[359,277],44:[360,299],45:[360,322],47:[360,380],48:[360,412],49:[361,440],50:[362,430],51:[363,424],52:[363,417],53:[363,416]}},'B':{'frames':[41,47],'point':[397,202],'radius':15,'points':{26:[407.5,129],30:[402,141],34:[399,158],36:[398.5,167.5],38:[398,178],40:[397.8,188.8],41:[397.5,195]}}}
for view,ref in references.items():
 r=json.load(open(run/view/'result.json'));s=r.get('selected_landing') or {};p=s.get('candidate_px');err=math.dist(p,ref['point']) if p else None
 t=next((t for t in r['tracks'] if str(t['track_id'])==str(s.get('track_id'))),{});obs={o['frame_index']:o for o in t.get('observations',[])}
 checks=[]
 for f,xy in ref['points'].items():
  o=obs.get(f);checks.append({'frame_index':f,'visible_reference_px':xy,'observed_px':[o['x'],o['y']] if o else None,'error_px':round(math.dist([o['x'],o['y']],xy),3) if o else None})
 out['views'][view]={'reference':ref,'selected_frame':s.get('frame_index'),'selected_time':s.get('time'),'selected_px':p,'point_discrepancy_px':round(err,3) if err is not None else None,'within_approximate_reference':bool(err is not None and err<=ref['radius'] and ref['frames'][0]<=s.get('frame_index',-1)<=ref['frames'][1]),'selected_track_id':s.get('track_id'),'track_method':t.get('method'),'contact_observed':s.get('contact_observed'),'decision':r['decision']['decision'],'assistance_used':r.get('assistance_used'),'image_motion_event':s.get('image_motion_event'),'trajectory_checks':checks,'timings':r['timings']}
print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
