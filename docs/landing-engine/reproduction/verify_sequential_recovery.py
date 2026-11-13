import json,hashlib,sys,time
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path.cwd()))
from badminton.landing_recovery import recover_rebound_tracks
from badminton.media import SourceFrameSequence,load_frames,file_hash
cv2.setNumThreads(1)
key='uhd4k-4-b';base_path=Path(f'runs/landing-engine/baseline/{key}/result.json');r=json.loads(base_path.read_text());records_path=base_path.with_name('detections.json');records=json.loads(records_path.read_text())
expected=r['source_sha256'];assert file_hash(r['source_path'])==expected
cache=None
for meta in Path('runs/landing-engine/frame-cache').glob('*.json'):
 data=json.loads(meta.read_text())
 if data.get('source_sha256')==expected and meta.with_suffix('.npy').exists():cache=np.load(meta.with_suffix('.npy'),mmap_mode='r',allow_pickle=False);break
if cache is None:cache=load_frames(r['source_path'],len(r['timestamps']),3000,resize_to=(r['analysis']['width'],r['analysis']['height']))
audit={}
class AuditFrames:
 def __init__(self,source):self.source=source
 def __len__(self):return len(self.source)
 def __getitem__(self,index):
  frame=self.source[index]
  small=cv2.resize(frame,(r['analysis']['width'],r['analysis']['height']),interpolation=cv2.INTER_AREA)
  if not np.array_equal(small,cache[index]):raise AssertionError(f'Actual source frame {index} differs from frozen sequential analysis pixels')
  audit[index]={'raw_source_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),'analysis_pixels_sha256':hashlib.sha256(small.tobytes()).hexdigest()}
  return frame
start=time.monotonic()
with SourceFrameSequence(r['source_path'],len(r['timestamps']),verified_seek=False) as source:
 additional,summary=recover_rebound_tracks(r['tracks'],AuditFrames(source),r['timestamps'],(r['analysis']['width'],r['analysis']['height']),frame_records=records)
 seeks=dict(source.seek_statistics)
assert 126 in audit and len(r['timestamps'])==127
report={'case':key,'source_sha256':expected,'source_frame_count':len(r['timestamps']),
 'previous_failure':'v4 sparse verified seeking exhausted before original source frame126; baseline full sequential decode succeeded',
 'recovery_decoder_options':{'verified_seek':False},'source_access':'Sequential decoder grabs; only requested source frames are retrieved and bounded crops retained',
 'seek_statistics':seeks,'retrieved_frame_count':len(audit),'all_retrieved_frames_match_frozen_sequential_analysis_pixels':True,
 'frame126':audit[126],'retrieved_frame_indices':sorted(audit),'recovery_summary':summary,'additional_track_count':len(additional),
 'elapsed_seconds':time.monotonic()-start,'baseline_result_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),
 'baseline_detections_sha256':hashlib.sha256(records_path.read_bytes()).hexdigest(),
 'recovery_module_sha256':hashlib.sha256(Path('badminton/landing_recovery.py').read_bytes()).hexdigest(),
 'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'scope':'Actual source decoding and native-recovery regression verification; no inference labels or source coordinates supplied by references'}
Path('docs/landing-engine/native-decoding-regression.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
