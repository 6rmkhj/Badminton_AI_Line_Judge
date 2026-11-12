"""Observed reversals must not make a constant-velocity filter swap targets."""
import json

from badminton.tracking import track_detections


def candidate(x, y, confidence=.9, radius=2):
    return {'x':x,'y':y,'confidence':confidence,'radius':radius}


def reversal_records(distractor=False):
    # Resolved downward travel followed by an observed upward reversal. A new
    # object on the old extrapolation must not inherit that trajectory's ID.
    points = [(100,50),(100,70),(101,95),(101,122),(101,151),(101,181),
              (102,216),(102,207),(103,197),(104,190),(104,188),(106,189)]
    records=[]
    for index, (x, y) in enumerate(points):
        detections=[candidate(x,y)]
        if distractor and index >= 8:
            detections.append(candidate(102,223+(index-8)*20,confidence=.6,radius=1))
        records.append({'frame_index':index,'time':index/30,'detections':detections})
    return points, records


def test_resolved_reversal_retains_observed_path_when_old_prediction_has_a_distractor():
    points, records = reversal_records(distractor=True)
    tracks = track_detections(records)
    shuttle = next(track for track in tracks if track['observations'][0]['frame_index'] == 0)
    assert [(o['x'],o['y']) for o in shuttle['observations']] == points
    assert len(shuttle['motion_model_changes']) == 1
    change = shuttle['motion_model_changes'][0]
    assert change['frame_index'] == 7
    assert change['verified_identity'] is False and change['contact_confirmed'] is False
    assert shuttle['target_court_confirmed'] is False
    assert all(o['observed'] for o in shuttle['observations'])
    # This is a regression of the constant-velocity model, not a detector test.
    old = track_detections(records, {'maneuver_adaptation':False})
    original = next(track for track in old if track['observations'][0]['frame_index'] == 0)
    assert original['observations'][8]['y'] == 223
    json.dumps(tracks, allow_nan=False)


def test_single_clean_reversal_keeps_all_observations_without_extra_tracks():
    points, records = reversal_records()
    tracks = track_detections(records)
    assert len(tracks) == 1
    assert [(o['x'],o['y']) for o in tracks[0]['observations']] == points
    assert len(tracks[0]['motion_model_changes']) == 1
    assert not tracks[0]['identity_ambiguous']


def test_small_measurement_jitter_does_not_become_a_motion_reversal():
    records=[{'frame_index':i,'time':i/30,'detections':[candidate(100,100+(-1)**i)]}
             for i in range(15)]
    tracks = track_detections(records)
    assert len(tracks) == 1 and tracks[0]['motion_model_changes'] == []


def test_motion_adaptation_does_not_erase_existing_identity_ambiguity():
    _, records = reversal_records()
    records[3]['detections'].append(candidate(101,123))
    tracks = track_detections(records)
    original = next(track for track in tracks if track['observations'][0]['frame_index'] == 0)
    assert original['identity_ambiguous']
    assert original['ambiguity_events']
    assert all(event['verified_identity'] is False and event['contact_confirmed'] is False
               for event in original['motion_model_changes'])


def test_missing_interval_cannot_supply_the_observed_reversal_for_model_adaptation():
    rows=[]
    for index, y in enumerate([50,80,110,140,None,100,80,60]):
        rows.append({'frame_index':index,'time':index/30,
                     'detections':[] if y is None else [candidate(100,y)]})
    tracks = track_detections(rows)
    assert all(event['frame_index'] != 5 for track in tracks for event in track['motion_model_changes'])
    assert all(not edge['verified_identity'] for track in tracks for edge in track['reconnections'])
    assert all(observation['frame_index'] != 4 for track in tracks for observation in track['observations'])
