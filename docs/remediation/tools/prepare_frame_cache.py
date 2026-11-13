"""Verify/import old24 actual-frame caches; source decoding needs explicit opt-in."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from filelock import FileLock
ROOT=Path('/workspace/remediation');BASE=Path('/workspace/generalization/release')
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def sequence_digest(frames):return hashlib.sha256(b''.join(hashlib.sha256(f.tobytes()).digest() for f in frames)).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--key',required=True);p.add_argument('--existing',type=Path);p.add_argument('--decode-source',action='store_true');args=p.parse_args()
 rows={r['key']:r for r in load(ROOT/'regression-manifest.json')['videos']}
 if args.key not in rows:raise ValueError('New holdouts are not allowed')
 row=rows[args.key];base=load(BASE/args.key/'result.json');a=base['analysis'];expected=(base.get('validation_run') or {}).get('decoded_analysis_frames_sha256');method=None
 folder=ROOT/'frame_cache'/args.key;folder.mkdir(parents=True,exist_ok=True)
 with FileLock(str(folder/'.lock')):
  if args.existing:
   if not expected:raise ValueError('Historical cache has no frozen pixel digest; use explicit fresh source decode')
   path=args.existing.resolve();frames=np.load(path,mmap_mode='r',allow_pickle=False);method='frozen_release_analysis_pixel_digest_match'
  elif args.decode_source:
   if sha(row['path'])!=row['sha256']:raise ValueError('Source hash differs')
   sys.path.insert(0,'/workspace/badminton-progress');import cv2;cv2.setNumThreads(1)
   from badminton.media import load_frames,probe
   meta=probe(row['path'])
   if meta['timestamps']!=base['timestamps']:raise ValueError('Fresh source timestamps differ')
   frames=np.asarray(load_frames(row['path'],a['frame_count'],resize_to=(a['width'],a['height'])))
   path=folder/'frames.npy';np.save(path,frames,allow_pickle=False);method='fresh_source_decode_with_exact_frozen_timestamps'
  else:raise ValueError('Choose a verified --existing cache or explicitly authorize --decode-source')
  if frames.dtype!=np.uint8 or tuple(frames.shape)!=(a['frame_count'],a['height'],a['width'],3):raise ValueError('Actual frame cache shape/dtype differs')
  digest=sequence_digest(frames)
  if expected and digest!=expected:raise ValueError('Actual pixels differ from frozen release input')
  record={'key':args.key,'source_sha256':row['sha256'],'path':str(path),'npy_sha256':sha(path),'analysis_sequence_sha256':digest,'frame_count':a['frame_count'],'analysis_size':[a['width'],a['height']],'timestamps_sha256':hashlib.sha256(json.dumps(base['timestamps'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'preprocessing':a['preprocessing'],'verification':method,'frozen_release_pixel_digest_available':bool(expected),'baseline_result_sha256':sha(BASE/args.key/'result.json')}
  (folder/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record))
if __name__=='__main__':main()
