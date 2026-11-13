from copy import deepcopy

import numpy as np
import pytest

from badminton.event_continuity import resolve_fragment_turns, _rows


def competing_turns(fps=30):
    def sample(frame, y, radius=2):
        return {'frame_index': frame, 'time': frame / fps, 'x': float(frame),
                'y': float(y), 'radius': radius, 'observed': True}

    prefix = [sample(frame, frame * 2) for frame in range(7)]
    suffix = [sample(frame, 20 - frame) for frame in range(9, 13)]
    tracks = [
        {'track_id': 1, 'flight_hypothesis': {},
         'observations': prefix + [sample(8, 11.6, .5)] + suffix},
        {'track_id': 2, 'flight_hypothesis': {},
         'observations': deepcopy(prefix) + [sample(7, 14), sample(8, 13)] + deepcopy(suffix)},
    ]
    candidates = [{'track_id': identifier, 'frame_index': frame, 'time': frame / fps,
                   'candidate_px': [float(frame), float(frame * 2)], 'contact_observed': False,
                   'flight_trajectory_support': {'supported_for_review': True, 'reason_codes': []},
                   'image_motion_event': {'kind': 'descent_and_rebound', 'point_kind': 'observed_object_center'},
                   'reason_codes': []} for identifier, frame in [(1, 6), (2, 7)]]
    return candidates, tracks


@pytest.mark.parametrize('fps', [30, 60])
def test_missing_real_peak_cannot_be_replaced_by_tiny_outgoing_fragment(fps):
    candidates, tracks = competing_turns(fps)
    original = deepcopy((candidates, tracks))
    result = resolve_fragment_turns(candidates, tracks, 1 / fps)
    assert result[0]['flight_trajectory_support']['supported_for_review'] is False
    assert result[1]['flight_trajectory_support']['supported_for_review'] is True
    assert result[0]['candidate_px'] == candidates[0]['candidate_px']
    assert result[0]['contact_observed'] is False
    evidence = result[0]['competing_peak_evidence']
    assert evidence['continued_peak_frame_index'] == 7
    assert evidence['identity_confirmed'] is False
    assert evidence['physical_contact_confirmed'] is False
    assert (candidates, tracks) == original


@pytest.mark.parametrize('condition', [
    'resolved_outgoing', 'different_flight', 'no_shared_suffix', 'small_later_fragment',
    'unobserved_peak', 'interpolated_peak', 'inferred_peak', 'verified_early_contact', 'not_descending',
])
def test_other_turns_are_not_vetoed_without_combined_measured_evidence(condition):
    candidates, tracks = competing_turns()
    if condition == 'resolved_outgoing':
        tracks[0]['observations'][7]['radius'] = 2
    elif condition == 'different_flight':
        for row in tracks[1]['observations']:
            row['x'] += 30
    elif condition == 'no_shared_suffix':
        for row in tracks[1]['observations']:
            if row['frame_index'] > 7:
                row['x'] += 2
    elif condition == 'small_later_fragment':
        tracks[1]['observations'][7]['radius'] = .4
    elif condition == 'unobserved_peak':
        tracks[1]['observations'][7]['observed'] = False
    elif condition == 'interpolated_peak':
        tracks[1]['observations'][7].update(observed=True, interpolated=True)
    elif condition == 'inferred_peak':
        candidates[1]['image_motion_event']['point_kind'] = 'inferred_object_center'
    elif condition == 'verified_early_contact':
        candidates[0]['contact_observed'] = True
    elif condition == 'not_descending':
        tracks[1]['observations'][7]['y'] = 11
    assert resolve_fragment_turns(candidates, tracks, 1 / 30) == candidates


def test_unusable_cadence_does_not_supply_evidence():
    candidates, tracks = competing_turns()
    assert resolve_fragment_turns(candidates, tracks, float('nan')) == candidates


def pixel_support(row, **changes):
    return {'schema_version':1,'status':'persistent_background','evidence_valid':True,
            'reject_for_moving_association':True,'frame_index':row['frame_index'],
            'point_px':[row['x'],row['y']],'coordinate_size':[128,128],**changes}


@pytest.mark.parametrize('part',['peak','shared_prefix','shared_suffix','early_fragment'])
def test_explicit_persistent_background_cannot_supply_competing_turn_evidence(part):
    candidates,tracks=competing_turns()
    for track in tracks:
        for row in track['observations']:
            chosen=(part=='peak' and track['track_id']==2 and row['frame_index']==7
                    or part=='shared_prefix' and row['frame_index']<4
                    or part=='shared_suffix' and row['frame_index']>=9
                    or part=='early_fragment' and track['track_id']==1 and row['frame_index']==8)
            if chosen:
                row['observation_support']=pixel_support(row)
    original=deepcopy((candidates,tracks))
    assert resolve_fragment_turns(candidates,tracks,1/30)==candidates
    assert (candidates,tracks)==original


@pytest.mark.parametrize('state',['unknown','stale_frame','stale_point','transient'])
def test_unknown_or_stale_pixel_support_stays_neutral_in_competing_peak_rule(state):
    candidates,tracks=competing_turns()
    peak=tracks[1]['observations'][7]
    changes={'unknown':{'status':'unknown','evidence_valid':False,'reject_for_moving_association':False},
             'stale_frame':{'frame_index':8},'stale_point':{'point_px':[100,100]},
             'transient':{'status':'transient','reject_for_moving_association':False}}
    peak['observation_support']=pixel_support(peak,**changes[state])
    result=resolve_fragment_turns(candidates,tracks,1/30)
    assert result[0]['flight_trajectory_support']['supported_for_review'] is False


def test_quadratic_fit_requires_three_distinct_actual_timestamps(monkeypatch):
    candidates,tracks=competing_turns()
    for track in tracks:
        for row in track['observations']:
            if row['frame_index']<=6:
                row['time']=0. if row['frame_index']<3 else .2
    def forbidden(*args,**kwargs):
        pytest.fail('Two unique timestamps must never reach the quadratic solver')
    monkeypatch.setattr('badminton.event_continuity.np.polyfit',forbidden)
    assert resolve_fragment_turns(candidates,tracks,1/30)==candidates


def test_three_distinct_timestamps_can_support_fit_even_with_duplicate_pts():
    candidates,tracks=competing_turns()
    for track in tracks:
        for row in track['observations']:
            if row['frame_index']<=6:
                row['time']=[0.,0.,.1,.1,.2,.2,.2][row['frame_index']]
    result=resolve_fragment_turns(candidates,tracks,1/30)
    assert result[0]['flight_trajectory_support']['supported_for_review'] is False


@pytest.mark.parametrize('invalid',[float('nan'),float('inf'),float('-inf'),10**400,True])
def test_nonfinite_or_non_numeric_time_cannot_be_a_measured_row(invalid):
    _,tracks=competing_turns()
    row=tracks[0]['observations'][0]
    row['time']=invalid
    assert all(item['frame_index']!=0 for item in _rows(tracks[0]))


@pytest.mark.parametrize('failure',['nonfinite_coefficients','rank_deficient','solver_error'])
def test_numerically_unusable_fit_never_vetoes_a_candidate(monkeypatch,failure):
    candidates,tracks=competing_turns()
    def unusable(*args,**kwargs):
        assert kwargs.get('full') is True
        if failure=='solver_error':
            raise np.linalg.LinAlgError('singular fit')
        coefficients=np.full((3,2),np.nan) if failure=='nonfinite_coefficients' else np.zeros((3,2))
        return coefficients,np.array([]),2 if failure=='rank_deficient' else 3,np.ones(3),1e-14
    monkeypatch.setattr('badminton.event_continuity.np.polyfit',unusable)
    assert resolve_fragment_turns(candidates,tracks,1/30)==candidates


def test_large_finite_integer_coordinates_do_not_create_object_dtype_fit_errors():
    candidates,tracks=competing_turns()
    for track in tracks:
        for row in track['observations']:
            if row['frame_index']<=6:
                row['x']=10**300
    assert resolve_fragment_turns(candidates,tracks,1/30)==candidates
