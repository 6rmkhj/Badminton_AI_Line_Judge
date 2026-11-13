"""Native suffix occlusion metadata must come from exactly frozen detector frames."""
import importlib.util
import json
from pathlib import Path
import types
import sys

import pytest

spec=importlib.util.spec_from_file_location('eval5',Path(__file__).parents[1]/'scripts/evaluate_landing_engine.py')
evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)


def fixture(tmp_path):
    folder=tmp_path/'clip';folder.mkdir()
    base={'source_sha256':'source','analysis':{'frame_count':2},'timestamps':[0.,.04],
          'benchmark_execution':{'analysis_frames_sha256':'pixels'}}
    records=[{'frame_index':i,'time':t,'detections':[],'occlusion_regions':[]} for i,t in enumerate(base['timestamps'])]
    evaluator.write(tmp_path/'run.json',{'source':'frozen'})
    evaluator.write(folder/'result.json',base);evaluator.write(folder/'detections.json',records)
    row={'source_sha256':'source','result_sha256':evaluator.sha(folder/'result.json'),
         'detections_sha256':evaluator.sha(folder/'detections.json'),'analysis_frames_sha256':'pixels',
         'frame_count':2,'timestamps_sha256':evaluator.digest(base['timestamps'])}
    inventory={'baseline_run_sha256':evaluator.sha(tmp_path/'run.json'),'videos':{'clip':row}}
    return base,records,inventory


def test_detector_artifact_hash_and_frame_contract(tmp_path):
    base,records,inventory=fixture(tmp_path)
    returned,sha=evaluator.validated_detection_records(tmp_path,'clip',base,inventory)
    assert returned==records and sha==inventory['videos']['clip']['detections_sha256']
    records[0]['occlusion_regions']=[{'bbox':[0,0,100,100]}]
    evaluator.write(tmp_path/'clip/detections.json',records)
    with pytest.raises(ValueError,match='detections SHA'):
        evaluator.validated_detection_records(tmp_path,'clip',base,inventory)


def test_correctly_hashed_but_misaligned_frames_rejected(tmp_path):
    base,records,inventory=fixture(tmp_path)
    records[1]['time']=.05;evaluator.write(tmp_path/'clip/detections.json',records)
    inventory['videos']['clip']['detections_sha256']=evaluator.sha(tmp_path/'clip/detections.json')
    with pytest.raises(ValueError,match='frame/timestamp'):
        evaluator.validated_detection_records(tmp_path,'clip',base,inventory)


def test_inventory_cannot_be_rebound_to_different_baseline(tmp_path):
    base,records,inventory=fixture(tmp_path)
    evaluator.write(tmp_path/'run.json',{'source':'changed'})
    with pytest.raises(ValueError,match='baseline run'):
        evaluator.validated_detection_records(tmp_path,'clip',base,inventory)


def test_native_replay_does_not_silently_omit_detector_records():
    with pytest.raises(ValueError,match='per-frame detector'):
        evaluator.recover_tracks({'camera_stability':{'status':'ok'}},'source',
            {'native_endpoint_recovery':True},[],[0],(960,540))


def test_recovery_receives_original_detector_records(monkeypatch):
    captured=[]
    class Source:
        def __init__(self,*args,**kwargs):assert kwargs == {'verified_seek':False}
        def __enter__(self):return self
        def __exit__(self,*args):pass
    records=[{'frame_index':0,'occlusion_regions':[{'bbox':[1,2,3,4]}]}]
    def recover(tracks,source,timestamps,size,*,frame_records):
        captured.append(frame_records);return [],{}
    monkeypatch.setitem(sys.modules,'badminton.media',types.SimpleNamespace(SourceFrameSequence=Source))
    monkeypatch.setitem(sys.modules,'badminton.landing_recovery',types.SimpleNamespace(recover_rebound_tracks=recover))
    evaluator.recover_tracks({'camera_stability':{'status':'ok'}},'source',
        {'native_endpoint_recovery':True},[],[0],(960,540),frame_records=records)
    assert captured==[records]
