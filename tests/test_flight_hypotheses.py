"""Finite, unverified free-flight alternatives retain only real detections."""
import json

from badminton.tracking import track_detections


def scene(missing=()):
    records=[]
    for frame in range(18):
        y = 40 + 5*frame + .3*frame**2 if frame <= 10 else 120 - 5*(frame-10) + .3*(frame-10)**2
        points = [] if frame in missing else [{'x':100+.2*frame,'y':y,'radius':1.5,'confidence':.7}]
        # Other high-confidence objects provide no confirmed target identity.
        points += [{'x':200+frame,'y':80+(-1)**frame*2,'radius':2,'confidence':.95},
                   {'x':40,'y':140,'radius':1,'confidence':.95}]
        records.append({'frame_index':frame,'time':frame/30,'detections':points})
    return records


def test_free_flight_search_is_opt_in_and_returns_observed_approach_with_rebound():
    rows=scene()
    assert not any('flight_hypothesis' in track for track in track_detections(rows))
    tracks=track_detections(rows, {'flight_search_enabled':True,'flight_max_beams':128,'flight_max_hypotheses':20})
    hypotheses=[t for t in tracks if 'flight_hypothesis' in t]
    assert hypotheses
    matching=[t for t in hypotheses if t['flight_hypothesis']['rebound_count'] >= 3
              and any(o['frame_index']==10 and abs(o['x']-102)<1 and abs(o['y']-120)<1
                      for o in t['observations'])]
    assert matching
    for track in hypotheses:
        info=track['flight_hypothesis']
        assert track['identity_ambiguous'] and not track['target_court_confirmed']
        assert not info['verified_identity'] and not info['physical_contact_confirmed']
        assert not info['observations_interpolated']
        assert track['source_track_ids']==[track['track_id']]
        for observation in track['observations']:
            assert observation['observed']
            original=rows[observation['frame_index']]['detections']
            assert any(d['x']==observation['x'] and d['y']==observation['y'] for d in original)
    json.dumps(tracks,allow_nan=False)


def test_flight_paths_preserve_missing_frames_and_output_budget():
    rows=scene(missing=(7,))
    tracks=track_detections(rows, {'flight_search_enabled':True,'flight_max_beams':128,'flight_max_hypotheses':5})
    hypotheses=[t for t in tracks if 'flight_hypothesis' in t]
    assert len(hypotheses)<=15  # A base path and at most two observed rebound alternatives.
    crossing=[t for t in hypotheses if t['observations'][0]['frame_index']<7<t['observations'][-1]['frame_index']
              and all(abs(o['x']-(100+.2*o['frame_index']))<1 for o in t['observations'])]
    assert crossing
    for track in crossing:
        assert all(o['frame_index']!=7 for o in track['observations'])
        assert any(gap['start_frame']<=7<=gap['end_frame'] and not gap['resolved']
                   and not gap['observations_interpolated'] for gap in track['gaps'])
        assert len(track['observations'])<=24


def test_tracking_frame_progress_reports_actual_work_for_both_search_passes():
    rows=scene()
    events=[]
    track_detections(rows, {'flight_search_enabled':True,'flight_max_beams':64,'flight_max_hypotheses':2},
                     progress_callback=events.append)
    counted=[e for e in events if 'completed_units' in e]
    assert counted
    assert all(0<=e['completed_units']<=e['total_units']==len(rows) for e in counted)
    completed=[e['detail'] for e in counted if e['completed_units']==len(rows)]
    assert '프레임 사이의 후보 궤적 연결' in completed
    assert '일관된 비행 궤적 탐색' in completed
    assert events[-1]=={'detail':'비행·반동 궤적 연결 검토'}


def test_high_frame_rate_flight_prefers_complete_observations_over_artificial_gaps():
    rows=[{'frame_index':i,'time':i/120,'detections':[
        {'x':100.,'y':20.+4*i,'radius':1.,'confidence':.8}]} for i in range(32)]
    tracks=track_detections(rows, {'flight_search_enabled':True,'flight_max_beams':64,'flight_max_hypotheses':3})
    paths=[t for t in tracks if 'flight_hypothesis' in t]
    assert paths
    best=max(paths,key=lambda t:len(t['observations']))
    assert len(best['observations'])>=21
    assert best['gaps']==[]
    assert all(b['frame_index']==a['frame_index']+1
               for a,b in zip(best['observations'],best['observations'][1:]))
