from copy import deepcopy

import pytest

from badminton.landing_families import resolve_landing_families


TIMES = [index * .04 for index in range(40)]
SIZE = [320, 360]


def court():
    return {'primary_court': {
        'status': 'selected', 'scope': 'near_half', 'region_supported': True,
        'interior_polygon_px': [[50, 300], [250, 300], [250, 80], [50, 80]],
        'physical_court_polygon_px': [[50, 300], [250, 300], [250, 80], [50, 80]],
        'judgment_scope_polygon_px': [[26, 324], [274, 324], [274, 56], [26, 56]],
        'geometry_uncertainty_px': 2.,
        'interior_limit': 'near_short_service_line_not_an_out_boundary'}}


def track(identifier=1, x=100):
    observations = [{'frame_index': index, 'time': TIMES[index], 'x': float(x),
                     'y': float(100 + 10 * index), 'radius': 1., 'observed': True}
                    for index in range(13)]
    observations += [{'frame_index': index, 'time': TIMES[index], 'x': float(x),
                      'y': float(220 - 6 * (index - 12)), 'radius': 1., 'observed': True}
                     for index in range(13, 17)]
    return {'track_id': identifier, 'observations': observations}


def event(identifier='event_000', track_id=1, x=100, y=220, frame=12):
    return {'event_id': identifier, 'track_id': track_id, 'frame_index': frame, 'time': TIMES[frame],
            'candidate_px': [x, y], 'contact_observed': False, 'contact_px': None,
            'identity_confidence': .25, 'confidence': .4, 'uncertainty_px': 2., 'contact_radius_px': 1.,
            'flight_trajectory_support': {'supported_for_review': True},
            'image_motion_event': {'kind': 'descent_and_rebound', 'point_kind': 'observed_object_center',
                                   'frame_interval': [frame - 1, frame + 1],
                                   'time_interval': [TIMES[frame - 1], TIMES[frame + 1]]}}


def resolve(events, tracks, calibration=None):
    return resolve_landing_families(events, tracks, TIMES, calibration or court(), SIZE)


def test_unique_observed_flight_selects_family_without_certifying_contact_or_identity():
    candidate = event()
    original = deepcopy(candidate)
    result = resolve([candidate], [track()])
    assert result['status'] == 'selected'
    family = result['selected_family']
    assert family['member_event_ids'] == ['event_000']
    assert family['representative']['contact_observed'] is False
    assert family['physical_contact_confirmed'] is False
    assert family['target_identity_confirmed'] is False
    assert candidate == original
    family['members'][0]['confidence'] = 1
    assert candidate == original


def test_shared_prefix_alternatives_form_one_family_and_keep_full_location_envelope():
    first, second = event(), event('event_001', 2, y=222)
    other = track(2)
    for row in other['observations'][13:]:
        row['x'] += 8
    result = resolve([first, second], [track(), other])
    assert result['status'] == 'selected'
    family = result['selected_family']
    assert family['member_event_ids'] == ['event_000', 'event_001']
    assert family['possible_candidate_polygon_px'] == [[97, 217], [103, 217], [103, 225], [97, 225]]
    evidence = family['selection_evidence']
    assert evidence['shared_measured_prefixes'][0]['shared_observation_count'] >= 5
    assert evidence['independent_evidence_count'] == 1
    assert evidence['hypotheses_are_not_independent_votes'] is True


def test_sparse_same_flight_alternative_stays_in_uncertainty_union():
    other = track(2)
    other['observations'] = [row for row in other['observations'] if row['frame_index'] <= 8 or row['frame_index'] >= 12]
    alternative = event('event_001', 2, y=224)
    alternative['uncertainty_px'] = 5
    result = resolve([event(), alternative], [track(), other])
    assert result['status'] == 'selected'
    assert len(result['selected_family']['members']) == 2
    assert result['selected_family']['possible_candidate_polygon_px'][2][1] == 230
    assert not result['excluded']


def test_independent_valid_flights_cannot_be_selected_by_confidence():
    first, second = event(), event('event_001', 2, x=180)
    first['confidence'] = .999
    second['confidence'] = .001
    result = resolve([first, second], [track(), track(2, x=180)])
    assert result['status'] == 'review'
    assert result['selected_family'] is None
    assert len(result['families']) == 2
    assert result['reason_codes'] == ['MULTIPLE_PRIMARY_COURT_FLIGHT_FAMILIES']


def test_spatial_coincidence_without_shared_measured_prefix_is_not_a_duplicate():
    other = track(2)
    for row in other['observations'][:12]:
        row['x'] += (12 - row['frame_index']) * .5
    result = resolve([event(), event('event_001', 2)], [track(), other])
    assert result['status'] == 'review'
    assert len(result['families']) == 2


def test_separated_motion_intervals_are_not_merged_despite_shared_prefix():
    later = event('event_001', 2, frame=16)
    other = track(2)
    for row in other['observations'][13:]:
        row['y'] = 220.
    result = resolve([event(), later], [track(), other])
    assert result['status'] == 'review'
    assert len(result['families']) == 2


def test_interpolated_shared_points_cannot_supply_family_or_court_association():
    other = track(2)
    for row in other['observations'][:12]:
        row['interpolated'] = True
    result = resolve([event('event_001', 2)], [other])
    assert result['status'] == 'review'
    assert not result['families']


def test_beyond_observed_court_depth_does_not_become_an_out_event():
    result = resolve([event(y=40)], [track()])
    assert result['status'] == 'review'
    assert result['excluded'][0]['reason_codes'] == ['OUTSIDE_OBSERVED_PRIMARY_COURT_REGION']


def test_uncertainty_crossing_artificial_depth_edge_remains_ambiguous():
    result = resolve([event(y=82)], [track()])
    assert result['status'] == 'review'
    assert result['ambiguous'][0]['reason_codes'] == ['UNCERTAINTY_OVERLAPS_UNOBSERVED_COURT_DEPTH']


def test_nearby_external_path_is_not_assigned_to_main_court():
    outside = track(2, x=265)
    result = resolve([event('event_001', 2, x=265)], [outside])
    assert result['status'] == 'review'
    assert result['excluded'][0]['reason_codes'] == ['PRIMARY_COURT_PATH_ASSOCIATION_UNSUPPORTED']


def test_physical_singles_alley_remains_available_for_court_association():
    calibration = court()
    calibration['primary_court']['interior_polygon_px'] = [[80, 300], [220, 300], [220, 80], [80, 80]]
    result = resolve([event(x=65)], [track(x=65)], calibration)
    assert result['status'] == 'selected'


def test_gap_turn_conflicting_with_same_measured_descending_flight_is_explained():
    original = track()
    ghost_track = {'track_id': 2, 'observations': deepcopy(original['observations'][:6]) + [
        {'frame_index': 11, 'time': TIMES[11], 'x': 170., 'y': 150., 'radius': 1., 'observed': True},
        {'frame_index': 12, 'time': TIMES[12], 'x': 170., 'y': 140., 'radius': 1., 'observed': True}]}
    ghost = event('ghost', 2, x=170, y=170, frame=8)
    ghost['image_motion_event'].update(kind='occluded_descent_and_rebound', point_kind='inferred_object_center',
                                       estimated_time=TIMES[8], frame_interval=[5, 11], time_interval=[TIMES[5], TIMES[11]])
    result = resolve([ghost, event()], [ghost_track, original])
    assert result['status'] == 'selected'
    assert result['selected_family']['member_event_ids'] == ['event_000']
    excluded = next(item for item in result['excluded'] if item['event_id'] == 'ghost')
    assert excluded['reason_codes'] == ['GAP_TURN_CONTRADICTED_BY_SHARED_MEASURED_DESCENT']
    assert excluded['shared_prefix_count'] == 6
    assert len(excluded['continuation_frame_indices']) >= 5


def test_only_inferred_peak_cannot_become_selected_observed_family():
    inferred = event()
    inferred['image_motion_event']['point_kind'] = 'inferred_object_center'
    result = resolve([inferred], [track()])
    assert result['status'] == 'review'
    assert result['reason_codes'] == ['EVENT_FAMILY_ONLY_INFERRED_LOCATIONS']


def test_horizontal_mirror_preserves_selection():
    candidate, trajectory, calibration = event(), track(), court()
    candidate['candidate_px'][0] = SIZE[0] - candidate['candidate_px'][0]
    for row in trajectory['observations']:
        row['x'] = SIZE[0] - row['x']
    for key in ('interior_polygon_px', 'physical_court_polygon_px', 'judgment_scope_polygon_px'):
        for point in calibration['primary_court'][key]:
            point[0] = SIZE[0] - point[0]
    assert resolve([candidate], [trajectory], calibration)['status'] == 'selected'


@pytest.mark.parametrize('damage', ['nan_point', 'nan_uncertainty', 'missing_track', 'reversed_times',
                                   'unselected_court', 'invalid_motion', 'invalid_flight'])
def test_invalid_inputs_cannot_supply_selected_family(damage):
    candidate, calibration, times = event(), court(), list(TIMES)
    if damage == 'nan_point':
        candidate['candidate_px'][0] = float('nan')
    elif damage == 'nan_uncertainty':
        candidate['uncertainty_px'] = float('nan')
    elif damage == 'missing_track':
        candidate.pop('track_id')
    elif damage == 'reversed_times':
        times.reverse()
    elif damage == 'invalid_motion':
        candidate['image_motion_event'] = 'invalid'
    elif damage == 'invalid_flight':
        candidate['flight_trajectory_support'] = 'invalid'
    else:
        calibration['primary_court']['status'] = 'review'
    result = resolve_landing_families([candidate], [track()], times, calibration, SIZE)
    assert result['status'] == 'review'
    assert result['selected_family'] is None
