"""The always-decide best-estimate tier: geometry, tier order and provenance."""
from copy import deepcopy

import numpy as np
import pytest

from badminton.best_estimate import (BASIS_PRIOR, DECISION_KIND, SOURCE, classify_point,
                                     estimate_best_verdict, unavailable_estimate)

RULES = {'game': 'doubles', 'phase': 'rally'}


def selected_court():
    inside = [500, 350]
    return {'status': 'review', 'method': 'adaptive_colour_hough_court_template',
            'confidence': .4, 'homography': None, 'image_width': 1000, 'image_height': 700,
            'camera_stability': {'status': 'ok', 'all_frames_sampled': True, 'reason_codes': []},
            'primary_court': {'status': 'selected', 'scope': 'near_half', 'confidence': .9, 'game': 'doubles',
                'rules': dict(RULES), 'region_supported': True,
                'polygon_px': [[200, 500], [800, 500], [650, 200], [350, 200]],
                'judgment_scope_polygon_px': [[100, 580], [900, 580], [720, 140], [280, 140]],
                'boundary_segments_px': [
                    {'role': role, 'segment_px': segment, 'observed': True,
                     'inside_point_px': inside, 'uncertainty_px': 3}
                    for role, segment in [('near', [200, 500, 800, 500]),
                                          ('left', [200, 500, 350, 200]),
                                          ('right', [800, 500, 650, 200])]]}}


def rectifying_homography():
    """Image (x, y) -> court metres for a fronto-parallel 100 px/m test plane."""
    return [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]]


def landing(point, frame=40, time=1.6, **extra):
    row = {'event_id': f'event_{frame:03d}', 'track_id': 1, 'frame_index': frame, 'time': time,
           'candidate_px': list(point), 'contact_observed': False, 'contact_px': None,
           'confidence': .4, 'identity_confidence': .15, 'uncertainty_px': 3.0,
           'reason_codes': ['IMAGE_SPACE_DESCENT_AND_REBOUND', 'FIRST_CONTACT_NOT_OBSERVED'],
           'image_motion_event': {'kind': 'descent_and_rebound', 'first_in_track': True,
                                  'point_kind': 'observed_object_center'},
           'flight_trajectory_support': {'supported_for_review': True, 'approach_displacement_y_px': 90.0}}
    row.update(extra)
    return row


def view_result(calibration, landings=(), **fields):
    return {'decision': {'decision': 'REVIEW', 'reason_codes': ['FIRST_CONTACT_NOT_OBSERVED']},
            'video_estimate': {'decision': 'REVIEW', 'eligible': False,
                               'reason_codes': ['VIDEO_ESTIMATE_NO_UNIQUE_EVENT'], 'family_resolution': {}},
            'landings': list(landings), 'tracks': [], 'calibration': calibration,
            'analysis': {'width': 1000, 'height': 700, 'frame_count': 120},
            'timestamps': list(np.arange(120) / 30), **fields}


@pytest.mark.parametrize(('point', 'verdict', 'role'), [((500, 400), 'IN', None), ((500, 520), 'OUT', 'near'),
                                                      ((220, 400), 'OUT', 'left'), ((780, 400), 'OUT', 'right')])
def test_primary_boundaries_give_a_signed_margin(point, verdict, role):
    calibration = selected_court()
    original = deepcopy(calibration)
    row = classify_point(point, calibration, RULES)
    assert row['decision'] == verdict and row['method'] == 'primary_court_boundaries'
    assert (row['margin_px'] >= 0) == (verdict == 'IN')
    if role:
        assert row['boundary_role'] == role
    assert calibration == original


def test_line_touch_is_in_and_the_margin_is_measured_to_the_nearest_boundary():
    row = classify_point((500, 500), selected_court(), RULES)
    assert row['decision'] == 'IN' and row['margin_px'] == pytest.approx(0, abs=1e-6)


def test_rejected_court_model_never_overrides_observed_lines_beyond_scope():
    calibration = selected_court()
    calibration['homography'] = rectifying_homography()  # status stays 'review'
    row = classify_point((500, 120), calibration, RULES)  # beyond the near-half scope
    assert row['method'] == 'primary_court_boundaries'
    assert 'BEST_ESTIMATE_COURT_LINES_EXTENDED' in row['reason_codes']


def test_court_model_hypothesis_is_used_and_labelled_without_primary_court():
    calibration = {'status': 'review', 'homography': rectifying_homography(), 'image_width': 1000,
                   'image_height': 700, 'primary_court': {'status': 'not_found'}}
    inside = classify_point((300, 600), calibration, RULES)    # (3.0 m, 6.0 m)
    outside = classify_point((700, 600), calibration, RULES)   # 7.0 m > 6.1 m doubles width
    assert inside['decision'] == 'IN' and inside['method'] == 'homography_court_model'
    assert outside['decision'] == 'OUT' and outside['margin_m'] == pytest.approx(-0.9)
    assert 'BEST_ESTIMATE_COURT_MODEL_HYPOTHESIS' in inside['reason_codes']
    singles = classify_point((40, 600), calibration, {'game': 'singles', 'phase': 'rally'})
    assert singles['decision'] == 'OUT'  # 0.40 m is inside the doubles alley, outside singles


def test_outline_hypothesis_is_the_last_geometry_and_no_geometry_returns_none():
    calibration = {'status': 'review', 'polygon_px': [[100, 600], [900, 600], [700, 200], [300, 200]],
                   'primary_court': {'status': 'not_found'}}
    assert classify_point((500, 400), calibration, RULES)['method'] == 'court_outline_hypothesis'
    assert classify_point((50, 400), calibration, RULES)['decision'] == 'OUT'
    singles = classify_point((500, 400), calibration, {'game': 'singles', 'phase': 'rally'})
    assert 'BEST_ESTIMATE_DOUBLES_OUTLINE_FOR_SINGLES' in singles['reason_codes']
    assert classify_point((500, 400), {'status': 'review'}, RULES) is None
    assert classify_point(None, selected_court(), RULES) is None


def test_verified_and_video_estimate_tiers_are_reused_unchanged():
    verified = view_result(selected_court(), decision={'decision': 'OUT', 'contact_px': [500, 520]})
    row = estimate_best_verdict(verified, RULES)
    assert (row['decision'], row['basis'], row['confidence']) == ('OUT', 'verified_contact', BASIS_PRIOR['verified_contact'])
    estimate = view_result(selected_court())
    estimate['video_estimate'] = {'decision': 'IN', 'eligible': True, 'uncertainty_px': 3.0,
                                  'possible_contact_polygon_px': [[490, 390], [510, 390], [510, 410], [490, 410]],
                                  'geometry': {'minimum_clearance_px': 40.0}, 'member_event_ids': ['event_001']}
    row = estimate_best_verdict(estimate, RULES)
    assert row['decision'] == 'IN' and row['basis'] == 'video_estimate'
    assert 0.5 < row['confidence'] <= BASIS_PRIOR['video_estimate']


def test_straddling_envelope_takes_the_side_holding_most_of_the_area():
    result = view_result(selected_court())
    result['video_estimate'].update(possible_contact_polygon_px=[[480, 470], [520, 470], [520, 510], [480, 510]],
                                    uncertainty_px=2.0, reason_codes=['VIDEO_ESTIMATE_BOUNDARY_AMBIGUOUS'])
    row = estimate_best_verdict(result, RULES)
    assert row['basis'] == 'estimate_envelope' and row['decision'] == 'IN'
    assert row['classification']['inside_area_fraction'] > 0.5


def test_unresolved_view_still_decides_from_landing_evidence():
    result = view_result(selected_court(), [landing((500, 520), frame=60, time=2.0)])
    original = deepcopy(result)
    row = estimate_best_verdict(result, RULES)
    assert row['decision'] == 'OUT' and row['decision_kind'] == DECISION_KIND and row['source'] == SOURCE
    assert row['contact_observed'] is False and row['target_identity_verified'] is False
    assert 0.5 <= row['confidence'] < BASIS_PRIOR['verified_contact']
    assert row['rules'] == RULES and result == original


def test_review_only_without_any_geometry_or_location():
    assert estimate_best_verdict(view_result(selected_court()), RULES)['decision'] == 'REVIEW'
    no_court = view_result({'status': 'review'}, [landing((500, 400))])
    row = estimate_best_verdict(no_court, RULES)
    assert row['decision'] == 'REVIEW' and row['reason_codes'] == ['BEST_ESTIMATE_UNAVAILABLE']
    assert unavailable_estimate(RULES)['confidence'] is None


def test_unconfirmed_candidate_family_is_not_strong_evidence():
    result = view_result(selected_court(), [landing((500, 520), frame=60, time=2.0)])
    result['video_estimate']['family_resolution'] = {
        'status': 'native_confirmation_required',
        'candidate_family': {'representative': {'event_id': 'event_060'}, 'member_event_ids': ['event_060']}}
    row = estimate_best_verdict(result, RULES)
    assert row['basis'] == 'landing_candidate'
    result['video_estimate']['family_resolution'] = {
        'status': 'selected', 'selected_family': {'representative': {'event_id': 'event_060'},
                                                 'member_event_ids': ['event_060']}}
    assert estimate_best_verdict(result, RULES)['basis'] == 'flight_family'


def test_envelope_the_transport_would_reject_is_not_reused_as_a_video_estimate():
    result = view_result(selected_court(), [landing((500, 400))])
    result['video_estimate'] = {'decision': 'IN', 'eligible': True, 'uncertainty_px': 3.0,
                                'possible_contact_polygon_px': [[990, 390], [1000, 390], [1000, 410], [990, 410]],
                                'geometry': {'minimum_clearance_px': 40.0}, 'member_event_ids': ['event_001']}
    row = estimate_best_verdict(result, RULES)
    assert row['basis'] != 'video_estimate' and row['decision'] in ('IN', 'OUT')


def test_extrapolated_boundary_is_labelled_and_less_reliable():
    from badminton.best_estimate import geometry_reliability
    calibration = selected_court()
    observed = classify_point((220, 400), calibration, RULES)
    calibration['primary_court']['boundary_segments_px'][1]['observed'] = False  # left side
    extrapolated = classify_point((220, 400), calibration, RULES)
    assert observed['boundary_observed'] is True and extrapolated['boundary_observed'] is False
    assert 'BEST_ESTIMATE_BOUNDARY_EXTRAPOLATED' in extrapolated['reason_codes']
    assert geometry_reliability(extrapolated) < geometry_reliability(observed)


def test_processing_failure_is_distinct_and_refresh_keeps_the_analysis():
    from badminton.best_estimate import ESTIMATOR_VERSION, failed_estimate, refresh_best_estimate
    failed = failed_estimate(RULES, ValueError('broken input'))
    assert failed['decision'] == 'REVIEW' and failed['reason_codes'] == ['BEST_ESTIMATE_PROCESSING_FAILED']
    assert 'broken input' in failed['error']
    result = view_result(selected_court(), [landing((500, 520), frame=60, time=2.0)], source_sha256='a' * 64)
    refreshed = refresh_best_estimate(result, RULES)
    assert refreshed['best_estimate']['estimator_version'] == ESTIMATOR_VERSION
    assert refreshed['best_estimate']['source_sha256'] == 'a' * 64
    assert refresh_best_estimate(refreshed, RULES) is refreshed  # current version is not recomputed
    stale = {**refreshed, 'best_estimate': {**refreshed['best_estimate'], 'estimator_version': 'old',
                                            'decision': 'IN', 'artifact_path': 'evidence/best-estimate.jpg'}}
    again = refresh_best_estimate(stale, RULES)
    assert 'artifact_path' not in again['best_estimate']  # the image showed a different verdict


def test_weak_out_location_follows_the_in_prior_and_says_so():
    calibration = selected_court()
    result = view_result(calibration, [landing((500, 501), frame=60, time=2.0,
                                               image_motion_event={'kind': 'other'},
                                               flight_trajectory_support=None)])
    row = estimate_best_verdict(result, RULES)
    assert row['location_decision'] == 'OUT' and row['decision'] == 'IN'
    assert 'BEST_ESTIMATE_PRIOR_OVERRODE_WEAK_LOCATION' in row['reason_codes']
    assert row['evidence_strength'] < 0.3 and row['p_in'] >= 0.5


def test_clear_out_outside_the_scope_is_possible_only_without_an_in_scope_landing():
    calibration = selected_court()
    calibration['primary_court']['physical_near_corners_px'] = [[200, 500], [600, 500]]  # zone 1.5 m ~ 98 px
    outside = landing((500, 640), frame=70, time=2.3)      # 60 px past the scope edge, inside the zone
    row = estimate_best_verdict(view_result(calibration, [outside]), RULES)
    assert row['decision'] == 'OUT' and row['location_decision'] == 'OUT'
    inside = landing((500, 400), frame=40, time=1.3)
    row = estimate_best_verdict(view_result(calibration, [outside, inside]), RULES)
    assert row['decision'] == 'IN' and row['event_ids'] == ['event_040']
    far = landing((500, 690), frame=80, time=2.6)         # 110 px past the scope: beyond the zone
    assert estimate_best_verdict(view_result(calibration, [far]), RULES)['decision'] == 'IN'


def test_unstable_camera_weakens_the_boundary_comparison():
    stable = estimate_best_verdict(view_result(selected_court(), [landing((500, 520), frame=60, time=2.0)]), RULES)
    calibration = selected_court()
    calibration['camera_stability'] = {'status': 'review', 'all_frames_sampled': True,
                                       'reason_codes': ['CAMERA_MOVED_FROM_CALIBRATION_REFERENCE']}
    moved = estimate_best_verdict(view_result(calibration, [landing((500, 520), frame=60, time=2.0)]), RULES)
    assert 'BEST_ESTIMATE_CAMERA_UNSTABLE' in moved['reason_codes']
    assert moved['evidence_strength'] < stable['evidence_strength']


def test_envelope_overridden_by_the_prior_is_disclosed():
    result = view_result(selected_court())
    result['video_estimate'].update(possible_contact_polygon_px=[[480, 482], [520, 482], [520, 522], [480, 522]],
                                    uncertainty_px=2.0, reason_codes=['VIDEO_ESTIMATE_BOUNDARY_AMBIGUOUS'])
    row = estimate_best_verdict(result, RULES)
    assert row['basis'] == 'estimate_envelope'
    if row['decision'] != row['location_decision']:
        assert 'BEST_ESTIMATE_PRIOR_OVERRODE_WEAK_LOCATION' in row['reason_codes']
