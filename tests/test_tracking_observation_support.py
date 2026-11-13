from copy import deepcopy

import pytest

from badminton.tracking import track_detections, _offline_links, _flight_hypotheses, _continuation_indices
from badminton.candidate_identity import apply_competing_continuation_evidence


def observation(frame, x, y, *, background=False):
    row = {'frame_index': frame, 'time': frame / 30, 'x': float(x), 'y': float(y),
           'radius': 1., 'confidence': .9, 'observed': True, 'appearance_polarity': 'bright'}
    if background:
        row['observation_support'] = {'schema_version': 1, 'status': 'persistent_background',
            'evidence_valid': True, 'reject_for_moving_association': True,
            'frame_index': frame, 'point_px': [float(x), float(y)], 'coordinate_size': [200, 200],
            'confirms_identity': False, 'confirms_contact': False}
    return row


def records(rows):
    return [{'frame_index': i, 'time': i / 30, 'detections': detections, 'occlusion_regions': []}
            for i, detections in enumerate(rows)]


def transient(row):
    row['observation_support'] = {'schema_version': 1, 'status': 'transient',
        'evidence_valid': True, 'reject_for_moving_association': False,
        'frame_index': row['frame_index'], 'point_px': [row['x'], row['y']],
        'coordinate_size': [200, 200], 'confirms_identity': False, 'confirms_contact': False}
    return row


def test_second_slot_keeps_pixel_supported_alternative_inside_unchanged_gate():
    history = [dict(observation(i, 10 + i, 10 + i), radius=2.) for i in range(5)]
    points = [observation(5, 20, 20), observation(5, 21, 21), transient(observation(5, 22, 22))]
    assert _continuation_indices(points, [1., 2., 3.], history, 5.) == [0, 2]
    assert _continuation_indices(points, [1., 2., 6.], history, 5.) == [0, 1]
    points[2]['observation_support']['frame_index'] = 99
    assert _continuation_indices(points, [1., 2., 3.], history, 5.) == [0, 1]


def test_unknown_evidence_keeps_original_two_nearest_alternatives():
    history = [observation(i, 10 + i, 10 + i) for i in range(5)]
    points = [observation(5, 20, 20), observation(5, 21, 21), observation(5, 22, 22)]
    assert _continuation_indices(points, [3., 1., 2.], history, 5.) == [1, 2]


def test_stationary_diagnostic_observations_and_raw_records_are_preserved():
    inputs = records([[observation(i, 40, 40, background=True)] for i in range(5)])
    original = deepcopy(inputs)
    result = track_detections(inputs, {'flight_search_enabled': True})
    assert len(result) == 1
    assert result[0]['method'] == 'gated_kalman_hungarian_tracklets'
    assert len(result[0]['observations']) == 5
    assert result[0]['moving_observation_support']['persistent_background_observation_count'] == 5
    assert result[0]['moving_observation_support']['identity_verified'] is False
    assert inputs == original


def test_moving_path_does_not_attach_to_a_known_static_feature_when_real_sample_is_nearby():
    inputs = records([[observation(i, 10 + 5 * i, 10 + 5 * i)] for i in range(3)]
                     + [[observation(3, 25, 25, background=True), observation(3, 26, 26)]])
    original = deepcopy(inputs)
    result = track_detections(inputs)
    moving = next(track for track in result if track['observations'][0]['frame_index'] == 0)
    assert moving['observations'][-1]['x'] == 26.
    assert moving['observations'][-1]['y'] == 26.
    assert inputs == original


def test_stale_negative_evidence_is_neutral_in_geometric_assignment():
    inputs = records([[observation(i, 10 + 5 * i, 10 + 5 * i)] for i in range(3)]
                     + [[observation(3, 25, 25, background=True), observation(3, 26, 26)]])
    inputs[-1]['detections'][0]['observation_support']['frame_index'] = 99
    result = track_detections(inputs)
    moving = next(track for track in result if track['observations'][0]['frame_index'] == 0)
    assert moving['observations'][-1]['x'] == 25.


def test_dedicated_flight_does_not_chain_known_background_features_but_unknown_is_neutral():
    background = records([[observation(i, 20 + 5 * i, 20 + 3 * i, background=True)] for i in range(10)])
    unknown = records([[observation(i, 20 + 5 * i, 20 + 3 * i)] for i in range(10)])
    original = deepcopy(background)
    assert _flight_hypotheses(background, {}, 100) == []
    assert _flight_hypotheses(unknown, {}, 100)
    assert background == original


def test_offline_moving_gap_link_cannot_end_on_known_stationary_support():
    def track(key, rows):
        return {'track_id': key, 'observations': rows, 'identity_ambiguous': False,
                'gaps': [], 'reconnections': [], 'reconnection_hypotheses': [], 'source_track_ids': [key]}
    before = track(1, [observation(i, 10 + 5 * i, 20) for i in range(3)])
    after = track(2, [observation(i, 30, 20, background=True) for i in range(4, 7)])
    result = _offline_links([before, after], {})
    assert len(result) == 2
    assert all(not track['reconnection_hypotheses'] for track in result)


@pytest.mark.parametrize('kind', ['persistent_background', 'interpolated'])
def test_unsafe_continuation_cannot_veto_an_observed_rebound_branch(kind):
    prefix = [observation(i, 100 + 2 * i, 20 + i * i) for i in range(10)]
    switched = [observation(i, 100 + 2 * i, 101 - 5 * (i - 9)) for i in range(10, 13)]
    for point in switched:
        point.update(radius=.25, appearance_polarity='dark')
    continued = [observation(i, 100 + 2 * i, 20 + i * i,
                            background=kind == 'persistent_background') for i in range(10, 13)]
    if kind == 'interpolated':
        for point in continued:
            point['interpolated'] = True
    tracks = [{'track_id': 1, 'flight_hypothesis': {}, 'observations': prefix + switched},
              {'track_id': 2, 'flight_hypothesis': {}, 'observations': deepcopy(prefix) + continued}]
    candidate = {'track_id': 1, 'frame_index': 9, 'time': 9 / 30, 'contact_observed': False,
        'reason_codes': [], 'image_motion_event': {'kind': 'descent_and_rebound'},
        'flight_trajectory_support': {'supported_for_review': True}}
    result = apply_competing_continuation_evidence([candidate], tracks, 1 / 30)
    assert result[0]['flight_trajectory_support']['supported_for_review'] is True
    assert 'competing_continuation_evidence' not in result[0]


@pytest.mark.parametrize('flag', [{'observed': False}, {'interpolated': True}])
def test_declared_nonmeasured_proposals_never_become_observed_motion_tracks(flag):
    inputs = records([[dict(observation(i, 20 + 5 * i, 20 + 3 * i), **flag)] for i in range(10)])
    original = deepcopy(inputs)
    assert track_detections(inputs, {'flight_search_enabled': True}) == []
    assert inputs == original
