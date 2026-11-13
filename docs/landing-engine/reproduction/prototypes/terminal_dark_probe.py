import sys,json,time,copy,math,inspect
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path.cwd()))
import badminton.landing_recovery as recovery
import badminton.tracking as tracking
from badminton.media import SourceFrameSequence,load_frames
from badminton.events import estimate_landings,prioritize_review_candidates,_approach_occluder_support
from badminton.scene_context import annotate_scene_context
cv2.setNumThreads(1)
# Reuse the existing bounded measured-path beam; replace only its final
# outgoing-motion predicate for this isolated ablation.
text=inspect.getsource(tracking._rebound_extensions)
text=text.replace('def _rebound_extensions(', 'def terminal_extensions(')
text=text.replace('if len(observations) >= 3 and observations[-1]["y"] < observations[0]["y"] - .5:',
 '''if (len(observations) >= 5 and observations[-1]['time'] - observations[0]['time'] >= .07
                    and np.linalg.norm(tracking._velocity(observations, tail=False)) <= .65 * np.linalg.norm(velocity)
                    and abs(observations[-1]['y'] - observations[0]['y']) <= max(8., radius * 3)):''')
namespace={**tracking.__dict__,'tracking':tracking};exec(text,namespace)
tracking._rebound_extensions=namespace['terminal_extensions']
# Same coherent-prefix ablation already tested separately.
prefix_code=Path('runs/landing-engine/observation-audit/prefix_probe.py').read_text()
body=prefix_code[prefix_code.index('def prefixes('):prefix_code.index('recovery._approaches=prefixes')]
ns={'old_approaches':recovery._approaches,'_flight_fit':tracking._flight_fit,'np':np,'math':math};exec(body,ns)
recovery._approaches=ns['prefixes']

def candidate(track,timestamps,width):
 n=track['flight_hypothesis']['approach_count'];before=track['observations'][:n];after=track['observations'][n:]
 end=before[-1];before=[o for o in before if end['time']-o['time']<=.35]
 gap=after[0]['time']-end['time'];nominal=float(np.median(np.diff(timestamps)))
 if len(before)<5 or before[-1]['time']-before[0]['time']<.12 or not 0<gap<=max(.08,3*nominal):return
 def fit(rows,degree):
  ts=np.array([o['time']-end['time'] for o in rows]);xy=np.array([[o['x'],o['y']] for o in rows]);co=np.polyfit(ts,xy,degree);pred=np.column_stack([np.polyval(co[:,ax],ts) for ax in range(2)]);res=float(np.sqrt(np.mean(np.sum((pred-xy)**2,axis=1))));return co,res
 incoming,err=fit(before,2);outgoing,out_err=fit(after,1)
 vi=incoming[-2];vo=outgoing[0];radius=max(1.,float(np.median([o['radius'] for o in before])))
 if (max(err,out_err)>max(3.,radius*1.5) or np.linalg.norm(vo)>.65*np.linalg.norm(vi)
     or np.dot(vi,vo)<0 or before[-1]['y']-before[0]['y']<max(20.,width*.02)):return
 ts=np.linspace(0,gap,41);left=np.column_stack([np.polyval(incoming[:,ax],ts) for ax in range(2)]);right=np.column_stack([np.polyval(outgoing[:,ax],ts) for ax in range(2)]);error=np.linalg.norm(left-right,axis=1);i=int(np.argmin(error))
 if error[i]>max(4.,radius*2):return
 point=((left[i]+right[i])/2).tolist();time=end['time']+float(ts[i]);index=int(np.argmin(abs(np.asarray(timestamps)-time)))
 visibility=_approach_occluder_support(before,after)
 if visibility['support_wholly_overlaps_possible_occluder']:return
 return {'track_id':track['track_id'],'frame_index':index,'time':timestamps[index], 'candidate_px':point,
 'contact_observed':False,'contact_px':None,'identity_confidence':.15,'identity_source':'unverified_image_trajectory',
 'confidence':.3,'confidence_kind':'uncalibrated_candidate_score','contact_radius_px':radius,'uncertainty_px':max(6.,radius*2,np.linalg.norm(vi)*gap/2),
 'reason_codes':['TERMINAL_IMAGE_VELOCITY_IMPULSE','FIRST_CONTACT_NOT_OBSERVED','CORK_CONTACT_REGION_UNRESOLVED','TARGET_IDENTITY_UNVERIFIED'],
 'image_motion_event':{'kind':'terminal_velocity_impulse','inference_only':True,'point_kind':'inferred_object_center','first_in_track':True,
 'frame_interval':[end['frame_index'],after[0]['frame_index']],'time_interval':[end['time'],after[0]['time']],
 'trajectory':[{k:o[k] for k in ('frame_index','time','x','y')} for o in before[-5:]+after]},
 'flight_trajectory_support':{'supported_for_review':True,'method':'coherent_approach_and_terminal_velocity_impulse',
 'approach_observation_count':len(before),'approach_displacement_y_px':before[-1]['y']-before[0]['y'],
 'fit_residual_px':err,'outgoing_fit_residual_px':out_err,'speed_ratio':float(np.linalg.norm(vo)/np.linalg.norm(vi)),
 'verified_identity':False,'physical_contact_verified':False},
 'method':'experimental_terminal_velocity_impulse'}

exec(Path('runs/landing-engine/observation-audit/dark_component_patch.py').read_text())
for key in sys.argv[1:]:
 r=json.load(open(f'runs/landing-engine/baseline/{key}/result.json'));start=time.monotonic()
 with SourceFrameSequence(r['source_path'],len(r['timestamps'])) as source:
  recovered,summary=recovery.recover_rebound_tracks(r['tracks'],source,r['timestamps'],(r['analysis']['width'],r['analysis']['height']))
 frames=load_frames(r['source_path'],len(r['timestamps']),3000,resize_to=(r['analysis']['width'],r['analysis']['height']))
 proposals=[c for t in recovered if (c:=candidate(t,r['timestamps'],r['analysis']['width']))]
 proposals=annotate_scene_context(proposals,frames)
 config={**r['config']['events'],'max_candidates':200}
 original=estimate_landings(r['tracks'],frames,r['timestamps'],r['calibration'],config)
 ranked=prioritize_review_candidates(original+proposals,[])[:12]
 print(key,summary,'terminalcount',len(proposals),'seconds',time.monotonic()-start,flush=True)
 print('terminal',[(c['track_id'],c['frame_index'],[round(v,2) for v in c['candidate_px']],c['flight_trajectory_support']['speed_ratio']) for c in proposals],flush=True)
 print('ranked',[(c['track_id'],c['frame_index'],[round(v,2) for v in c['candidate_px']],c['method']) for c in ranked],flush=True)
 Path(f'runs/landing-engine/observation-audit/{key}-terminal-dark.json').write_text(json.dumps({'summary':summary,'tracks':recovered,'landings':ranked,'terminal_proposals':proposals},indent=2))
