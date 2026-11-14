"""Scoped image-boundary decisions remain independent of metric calibration."""
from copy import deepcopy

import pytest

from badminton.decision import decide


def selected_court():
    inside = [500, 350]
    return {'status': 'review', 'method': 'adaptive_colour_hough_court_template',
            'confidence': .4, 'homography': None, 'metric_coordinates_available': False,
            'image_width': 1000, 'image_height': 700,
            'camera_stability': {'status': 'ok', 'all_frames_sampled': True, 'reason_codes': []},
            'primary_court': {'status': 'selected', 'scope': 'near_half', 'confidence': .9,
                'region_supported': True,
                'polygon_px': [[200, 500], [800, 500], [650, 200], [350, 200]],
                'judgment_scope_polygon_px': [[100, 580], [900, 580], [720, 140], [280, 140]],
                'boundary_segments_px': [
                    {'role': role, 'segment_px': segment, 'observed': True,
                     'inside_point_px': inside, 'uncertainty_px': 3}
                    for role, segment in [('near', [200, 500, 800, 500]),
                                          ('left', [200, 500, 350, 200]),
                                          ('right', [800, 500, 650, 200])]]}}


def observed_contact(point=(500, 400)):
    return {'contact_observed': True, 'contact_px': list(point), 'confidence': .95,
            'identity_confidence': .95, 'contact_radius_px': 0, 'uncertainty_px': 1,
            'frame_index': 31, 'time': 1.25}


@pytest.mark.parametrize(('point', 'verdict'), [((500, 400), 'IN'), ((500, 520), 'OUT'),
                                              ((220, 400), 'OUT'), ((780, 400), 'OUT'),
                                              ((500, 500), 'REVIEW'), ((500, 497), 'REVIEW')])
def test_observed_primary_boundaries_classify_local_contact_without_metres(point, verdict):
    calibration = selected_court()
    original = deepcopy(calibration)
    result = decide(observed_contact(point), calibration)
    assert result['decision'] == verdict
    assert result['coordinate_system'] == 'analysis_pixels'
    assert result['court_position_m'] is None
    assert result['uncertainty_m'] is None
    assert result['contact_time_seconds'] == 1.25
    assert result['contact_frame'] == 31
    assert calibration == original


def test_closure_edge_cannot_produce_out():
    # This lies past the supported near-court region, toward the net/far court.
    result = decide(observed_contact((500, 160)), selected_court())
    assert result['decision'] == 'REVIEW'
    assert 'ENTIRE_POSSIBLE_CONTACT_OUTSIDE' not in result['reason_codes']


def test_expanded_neighborhood_does_not_extrapolate_an_observed_side_boundary():
    # In the padded neighborhood, but past the measured left-side endpoint.
    result = decide(observed_contact((330, 180)), selected_court())
    assert result['decision'] == 'REVIEW'
    left = next(row for row in result['boundaries'] if row['role'] == 'left')
    assert left['max_inside_margin_px'] < -left['uncertainty_px']
    assert left['within_observed_extent'] is False
    assert left['robust_outside'] is False


@pytest.mark.parametrize('point', [(920, 400), (80, 400), (500, 80), (500, 620)])
def test_distant_neighbor_or_unseen_region_is_not_an_out_verdict(point):
    result = decide(observed_contact(point), selected_court())
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['CONTACT_OUTSIDE_PRIMARY_COURT_SCOPE']


@pytest.mark.parametrize('change', [{'contact_observed': False}, {'occluded': True},
                                    {'first_contact_missing': True}, {'identity_confidence': .79},
                                    {'confidence': .79}, {'confidence': float('nan')}])
def test_contact_and_identity_gates_remain_unchanged(change):
    assert decide({**observed_contact(), **change}, selected_court())['decision'] == 'REVIEW'


@pytest.mark.parametrize('rules', [{'game': 'singles'}, {'phase': 'service', 'half': 'near', 'side': 'left'}])
def test_unobserved_singles_or_service_boundary_roles_are_not_assumed(rules):
    result = decide(observed_contact(), selected_court(), rules)
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['PRIMARY_COURT_RULES_UNSUPPORTED']


@pytest.mark.parametrize('stability', [None, {}, {'status': 'review', 'all_frames_sampled': True},
    {'status': 'ok', 'all_frames_sampled': False}, {'status': 'ok'},
    {'status': 'ok', 'all_frames_sampled': True, 'reason_codes': None},
    {'status': 'ok', 'all_frames_sampled': True, 'reason_codes': ['CAMERA_MOVED_FROM_CALIBRATION_REFERENCE']}])
def test_camera_stability_requires_complete_observed_coverage(stability):
    calibration = selected_court()
    calibration['camera_stability'] = stability
    result = decide(observed_contact(), calibration)
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['PRIMARY_COURT_CAMERA_UNSTABLE']


def test_manual_reference_is_not_automatic_boundary_acceptance():
    calibration = selected_court()
    calibration['manual_reference'] = True
    result = decide(observed_contact(), calibration)
    assert result['decision'] == 'REVIEW'
    assert 'MANUAL_CALIBRATION_REFERENCE_ONLY' in result['reason_codes']


@pytest.mark.parametrize('confidence', [.64, float('nan'), float('inf'), -1, 1.1])
def test_selected_court_support_must_be_valid_and_sufficient(confidence):
    calibration = selected_court()
    calibration['primary_court']['confidence'] = confidence
    assert decide(observed_contact(), calibration)['decision'] == 'REVIEW'


@pytest.mark.parametrize('field,value', [('segment_px', [200, 500, 200, 500]),
    ('segment_px', [200, 500, float('nan'), 500]), ('observed', False), ('role', 'net'),
    ('inside_point_px', [500, 500]), ('uncertainty_px', -1), ('uncertainty_px', float('nan'))])
def test_malformed_boundary_abstains_even_for_an_apparently_inside_point(field, value):
    calibration = selected_court()
    calibration['primary_court']['boundary_segments_px'][0][field] = value
    result = decide(observed_contact(), calibration)
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['INVALID_CONTACT_GEOMETRY_OR_CALIBRATION']


def test_contradictory_inside_reference_cannot_reverse_an_out_verdict():
    calibration = selected_court()
    calibration['primary_court']['boundary_segments_px'][0]['inside_point_px'] = [500, 550]
    result = decide(observed_contact((500, 520)), calibration)
    assert result['decision'] == 'REVIEW'


def test_three_names_for_one_observed_line_do_not_form_a_court():
    calibration = selected_court()
    for boundary in calibration['primary_court']['boundary_segments_px']:
        boundary['segment_px'] = [200, 500, 800, 500]
    result = decide(observed_contact(), calibration)
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['INVALID_CONTACT_GEOMETRY_OR_CALIBRATION']


def test_arbitrary_interior_past_finite_side_endpoints_cannot_prove_in():
    calibration = selected_court()
    polygon = [[350, 200], [650, 200], [635, 170], [365, 170]]
    calibration['primary_court'].update(polygon_px=polygon, interior_polygon_px=polygon)
    result = decide(observed_contact((500, 185)), calibration)
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['INVALID_CONTACT_GEOMETRY_OR_CALIBRATION']


def test_near_corners_must_be_observed_at_boundary_endpoints():
    calibration = selected_court()
    calibration['primary_court']['boundary_segments_px'][1]['segment_px'] = [225, 450, 350, 200]
    result = decide(observed_contact(), calibration)
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['INVALID_CONTACT_GEOMETRY_OR_CALIBRATION']


def test_boundary_endpoint_direction_does_not_change_supported_court():
    calibration = selected_court()
    for boundary in calibration['primary_court']['boundary_segments_px']:
        x1, y1, x2, y2 = boundary['segment_px']
        boundary['segment_px'] = [x2, y2, x1, y1]
    assert decide(observed_contact(), calibration)['decision'] == 'IN'


@pytest.mark.parametrize('polygon', [None, [[0, 0], [10, 10]],
                                    [[100, 580], [720, 140], [900, 580], [280, 140]]])
def test_missing_or_invalid_neighborhood_cannot_classify_adjacent_courts(polygon):
    calibration = selected_court()
    calibration['primary_court']['judgment_scope_polygon_px'] = polygon
    result = decide(observed_contact(), calibration)
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['INVALID_CONTACT_GEOMETRY_OR_CALIBRATION']


def test_conservative_interior_can_prove_in_without_complete_court_polygon():
    calibration = selected_court()
    primary = calibration['primary_court']
    primary.update(region_supported=True, polygon_px=None, scope='local',
                   interior_polygon_px=[[260, 460], [740, 460], [620, 250], [380, 250]])
    assert decide(observed_contact(), calibration)['decision'] == 'IN'
    assert decide(observed_contact((500, 480)), calibration)['decision'] == 'REVIEW'
    assert decide(observed_contact((500, 520)), calibration)['decision'] == 'OUT'


def test_unsupported_interior_cannot_prove_in():
    calibration = selected_court()
    primary = calibration['primary_court']
    primary.update(region_supported=False, polygon_px=None, scope='local',
                   interior_polygon_px=[[260, 460], [740, 460], [620, 250], [380, 250]])
    assert decide(observed_contact(), calibration)['decision'] == 'REVIEW'
    assert decide(observed_contact((500, 520)), calibration)['decision'] == 'OUT'


def test_missing_outer_role_cannot_prove_in_from_arbitrary_interior():
    calibration = selected_court()
    calibration['primary_court']['interior_polygon_px'] = calibration['primary_court']['polygon_px']
    calibration['primary_court']['boundary_segments_px'].pop()
    assert decide(observed_contact(), calibration)['decision'] == 'REVIEW'


def test_interior_outside_judgment_scope_is_invalid():
    calibration = selected_court()
    calibration['primary_court'].update(polygon_px=None,
        interior_polygon_px=[[350, 200], [650, 200], [550, 90], [450, 90]])
    result = decide(observed_contact(), calibration)
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['INVALID_CONTACT_GEOMETRY_OR_CALIBRATION']


def test_contact_footprint_touch_counts_when_overlap_survives_uncertainty():
    landing = observed_contact((500, 505))
    landing['contact_polygon_px'] = [[490, 490], [510, 490], [510, 510], [490, 510]]
    assert decide(landing, selected_court())['decision'] == 'IN'
    landing['contact_polygon_px'] = [[490, 498], [510, 498], [510, 512], [490, 512]]
    assert decide(landing, selected_court())['decision'] == 'REVIEW'


def test_scope_must_contain_all_possible_contact_positions():
    landing = observed_contact((500, 570))
    landing['uncertainty_px'] = 15
    result = decide(landing, selected_court())
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['CONTACT_OUTSIDE_PRIMARY_COURT_SCOPE']


def test_accepted_metric_calibration_keeps_existing_rule_engine():
    calibration = selected_court()
    calibration.update(status='ok', confidence=.95,
                       homography=[[.01, 0, 0], [0, .01, 0], [0, 0, 1]], line_fit_error_px=.5)
    result = decide(observed_contact(), calibration, {'game': 'singles'})
    assert result['decision'] == 'IN'
    assert result['court_position_m'] == pytest.approx([5, 4])
    assert result.get('decision_method') != 'primary_court_observed_boundaries'
