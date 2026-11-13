"""Replay must apply native recovery only under the same camera gate as inference."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest

path = Path(__file__).parents[1] / 'scripts/evaluate_landing_engine.py'
spec = importlib.util.spec_from_file_location('recovery_evaluator', path)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


@pytest.mark.parametrize('config,base,status', [
    ({}, {'camera_stability':{'status':'ok'}}, 'disabled'),
    ({'native_endpoint_recovery':True}, {}, 'skipped'),
    ({'native_endpoint_recovery':True}, {'camera_stability':{'status':'moving'}}, 'skipped'),
])
def test_recovery_gate_does_not_create_tracks(config,base,status):
    tracks = [{'track_id':1}]
    returned, summary = evaluator.recover_tracks(base,'source.mp4',config,tracks,[0,1],(960,540))
    assert returned is tracks
    assert summary['status'] == status


def test_recovery_appends_measured_tracks_and_closes_source(monkeypatch):
    calls = []
    class Source:
        def __init__(self,source,length,*,verified_seek):
            assert verified_seek is False
            calls.append(('open',source,length))
        def __enter__(self): return self
        def __exit__(self,*args): calls.append(('close',))
    def recover(tracks, originals, timestamps, size, *, frame_records):
        assert tracks == [{'track_id':1}]
        assert isinstance(originals,Source)
        assert timestamps == [0,1]
        assert size == (960,540)
        assert frame_records == [{'frame_index':0}, {'frame_index':1}]
        return [{'track_id':'native','observations':[{'observed':True}]}], {'status':'completed'}
    monkeypatch.setitem(sys.modules,'badminton.media',types.SimpleNamespace(SourceFrameSequence=Source))
    monkeypatch.setitem(sys.modules,'badminton.landing_recovery',types.SimpleNamespace(recover_rebound_tracks=recover))
    baseline_tracks = [{'track_id':1}]
    tracks,summary = evaluator.recover_tracks({'camera_stability':{'status':'ok'}},'original.mp4',
                {'native_endpoint_recovery':True},baseline_tracks,[0,1],(960,540),
                frame_records=[{'frame_index':0}, {'frame_index':1}])
    assert len(tracks) == 2
    assert baseline_tracks == [{'track_id':1}]
    assert summary['status'] == 'completed'
    assert calls == [('open','original.mp4',2),('close',)]


def test_recovery_flag_rejects_truthy_nonboolean():
    with pytest.raises(ValueError, match='boolean'):
        evaluator.recover_tracks({},'original.mp4',{'native_endpoint_recovery':'yes'},[],[],(960,540))


def test_replay_cannot_silently_reuse_previously_recovered_tracks(monkeypatch):
    monkeypatch.setitem(sys.modules,'badminton.events',types.SimpleNamespace(
        estimate_landings=None,refine_source_contacts=None,prioritize_review_candidates=None))
    monkeypatch.setitem(sys.modules,'badminton.identity',types.SimpleNamespace(apply_identity=None))
    monkeypatch.setitem(sys.modules,'badminton.decision',types.SimpleNamespace(decide=None))
    monkeypatch.setitem(sys.modules,'badminton.media',types.SimpleNamespace(SourceFrameSequence=None))
    monkeypatch.setitem(sys.modules,'badminton.pipeline',types.SimpleNamespace(aggregate_decision=None,implementation_fingerprint=None,model_fingerprints=None))
    with pytest.raises(ValueError, match='unrecovered baseline'):
        evaluator.replay({'config':{'events':{'native_endpoint_recovery':True}}},'source',Path('.'),{}, {})


def pixel_fixture(tmp_path,monkeypatch):
    import numpy as np
    frames = [np.full((3,4,3),index,dtype=np.uint8) for index in range(2)]
    source = tmp_path/'source.mp4'; source.write_bytes(b'original-source')
    result = {'source_sha256':evaluator.sha(source),'analysis':{'frame_count':2,'width':4,'height':3,
              'preprocessing':'opencv_source_bgr_area_v1'},'timestamps':[0.0,0.04],
              'benchmark_execution':{'analysis_frames_sha256':evaluator.frame_digest(frames)}}
    calls = []
    def decode(*args): calls.append(1); return [frame.copy() for frame in frames]
    monkeypatch.setattr(evaluator,'decoded_frames',decode)
    return source,result,frames,calls


def test_pixel_cache_exact_readonly_reuse_without_decoding(tmp_path,monkeypatch):
    import numpy as np
    source,result,original,calls = pixel_fixture(tmp_path,monkeypatch)
    first,info = evaluator.cached_analysis_frames(result,source,tmp_path/'cache')
    second,reused = evaluator.cached_analysis_frames(result,source,tmp_path/'cache')
    assert calls == [1]
    assert info['reused'] is False and reused['reused'] is True
    assert evaluator.frame_digest(first) == evaluator.frame_digest(second) == evaluator.frame_digest(original)
    assert all(not frame.flags.writeable for frame in second)
    with pytest.raises(ValueError): second[0][0,0,0] = 9
    assert not list((tmp_path/'cache').glob('*.tmp.npy'))


def test_pixel_cache_rejects_corrupt_pixels(tmp_path,monkeypatch):
    import numpy as np
    source,result,_,_ = pixel_fixture(tmp_path,monkeypatch)
    _,info = evaluator.cached_analysis_frames(result,source,tmp_path/'cache')
    altered = np.load(info['path'],mmap_mode='r+',allow_pickle=False)
    altered[0,0,0,0] = 99; altered.flush(); del altered
    with pytest.raises(ValueError,match='exact-pixel SHA'):
        evaluator.cached_analysis_frames(result,source,tmp_path/'cache')


def test_pixel_cache_source_and_timestamp_contract(tmp_path,monkeypatch):
    source,result,_,calls = pixel_fixture(tmp_path,monkeypatch)
    evaluator.cached_analysis_frames(result,source,tmp_path/'cache')
    result['timestamps'] = [0.0,0.05]
    _,info = evaluator.cached_analysis_frames(result,source,tmp_path/'cache')
    assert info['reused'] is False and calls == [1,1]
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source SHA'):
        evaluator.cached_analysis_frames(result,source,tmp_path/'cache')


def test_pixel_cache_rejects_wrong_decoded_pixels_and_capacity(tmp_path,monkeypatch):
    source,result,frames,_ = pixel_fixture(tmp_path,monkeypatch)
    result['benchmark_execution']['analysis_frames_sha256'] = '0'*64
    with pytest.raises(ValueError,match='Decoded pixels differ'):
        evaluator.cached_analysis_frames(result,source,tmp_path/'cache')
    result['benchmark_execution']['analysis_frames_sha256'] = evaluator.frame_digest(frames)
    with pytest.raises(ValueError,match='size limit'):
        evaluator.cached_analysis_frames(result,source,tmp_path/'cache',max_cache_bytes=1)
    assert not list((tmp_path/'cache').glob('*.npy'))
    monkeypatch.setattr(evaluator.shutil,'disk_usage',lambda path:types.SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='free space'):
        evaluator.cached_analysis_frames(result,source,tmp_path/'cache')


def test_concurrent_pixel_cache_population_is_atomic(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    source,result,frames,calls = pixel_fixture(tmp_path,monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _:evaluator.cached_analysis_frames(result,source,tmp_path/'cache'),range(2)))
    assert calls == [1]
    assert sorted(info['reused'] for _,info in outputs) == [False,True]
    assert all(evaluator.frame_digest(data)==evaluator.frame_digest(frames) for data,_ in outputs)
