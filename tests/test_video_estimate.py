from copy import deepcopy

import numpy as np
import pytest

from badminton.decision import decide
from badminton.video_estimate import evaluate_possible_region
from test_primary_court_decision import selected_court


def court():
    value = selected_court()
    value['primary_court'].update(game='doubles', rules={'game': 'doubles', 'phase': 'rally'})
    return value


@pytest.mark.parametrize('polygon,label', [
    ([[450, 380], [470, 380], [470, 400], [450, 400]], 'IN'),
    ([[215, 380], [230, 380], [230, 400], [215, 400]], 'OUT'),
    ([[490, 520], [510, 520], [510, 530], [490, 530]], 'OUT'),
    ([[240, 390], [290, 390], [290, 410], [240, 410]], 'REVIEW'),
    ([[180, 390], [820, 390], [820, 410], [180, 410]], 'REVIEW'),
    ([[480, 150], [520, 150], [520, 165], [480, 165]], 'REVIEW'),
])
def test_every_possible_position_must_agree(polygon, label):
    original = court()
    result = evaluate_possible_region(polygon, 2, original)
    assert result['decision'] == label
    assert result['universal_region_test'] is True and result['geometry_only'] is True
    assert original == court()


def test_uncertainty_envelope_is_not_a_physical_footprint_touching_in():
    polygon = [[240, 390], [290, 390], [290, 410], [240, 410]]
    physical = {'contact_observed': True, 'contact_px': [270, 400], 'contact_polygon_px': polygon,
                'confidence': .95, 'identity_confidence': .95, 'uncertainty_px': 2}
    assert decide(physical, court())['decision'] == 'IN'
    # These coordinates describe mutually possible positions, not one physical
    # cork touching the court. Some possibilities are OUT, so it is unresolved.
    assert evaluate_possible_region(polygon, 2, court())['decision'] == 'REVIEW'


def test_location_error_can_prevent_a_previously_in_estimate():
    polygon = [[280, 395], [290, 395], [290, 405], [280, 405]]
    assert evaluate_possible_region(polygon, 1, court())['decision'] == 'IN'
    assert evaluate_possible_region(polygon, 35, court())['decision'] == 'REVIEW'


def test_valid_interior_only_record_does_not_require_a_redundant_polygon():
    calibration = court()
    primary = calibration['primary_court']
    primary['interior_polygon_px'] = primary.pop('polygon_px')
    assert evaluate_possible_region([[450, 380], [470, 380], [470, 400], [450, 400]], 2, calibration)['decision'] == 'IN'


def test_full_court_uses_physical_extent_without_mutating_original_calibration():
    calibration = court()
    primary = calibration['primary_court']
    primary.update(scope='full', physical_court_polygon_px=deepcopy(primary['polygon_px']))
    primary.pop('judgment_scope_polygon_px')
    primary['boundary_segments_px'].append({'role': 'far', 'segment_px': [350, 200, 650, 200],
        'observed': True, 'inside_point_px': [500, 350], 'uncertainty_px': 3})
    before = deepcopy(calibration)
    assert evaluate_possible_region([[450, 380], [470, 380], [470, 400], [450, 400]], 2, calibration)['decision'] == 'IN'
    assert calibration == before


def test_finite_boundary_cannot_be_extrapolated_for_out():
    polygon = [[325, 177], [335, 177], [335, 183], [325, 183]]
    assert evaluate_possible_region(polygon, 1, court())['decision'] == 'REVIEW'


@pytest.mark.parametrize('field,value', [
    ('manual_reference', True), ('camera_stability', None),
    ('camera_stability', {'status': 'ok', 'all_frames_sampled': False}),
    ('camera_stability', {'status': 'review', 'all_frames_sampled': True}),
])
def test_unavailable_boundary_evidence_never_becomes_an_estimate(field, value):
    calibration = court()
    calibration[field] = value
    result = evaluate_possible_region([[450, 380], [470, 380], [470, 400], [450, 400]], 2, calibration)
    assert result['decision'] == 'REVIEW'


@pytest.mark.parametrize('change', [
    {'game': 'singles'}, {'rules': {'game': 'singles', 'phase': 'rally'}},
    {'confidence': .64}, {'confidence': float('nan')}, {'region_supported': False},
    {'status': 'review'}, {'boundary_segments_px': []},
])
def test_court_mode_and_observed_support_are_required(change):
    calibration = court()
    calibration['primary_court'].update(change)
    assert evaluate_possible_region([[450, 380], [470, 380], [470, 400], [450, 400]], 2, calibration)['decision'] == 'REVIEW'


@pytest.mark.parametrize('polygon,error', [
    ([[450, 380]], 2), ([[450, 380], [450, 380], [450, 380]], 2),
    ([[450, 380], [470, float('nan')], [470, 400], [450, 400]], 2),
    ([[450, 380], [470, 380], [470, 400], [450, 400]], float('nan')),
    ([[450, 380], [470, 380], [470, 400], [450, 400]], -1),
    ([[450, 380], [470, 380], [470, 400], [450, 400]], True),
])
def test_invalid_possible_regions_are_rejected(polygon, error):
    assert evaluate_possible_region(polygon, error, court())['decision'] == 'REVIEW'


def test_same_region_and_uncertainty_scale_consistently():
    calibration = court()
    polygon = [[215, 380], [230, 380], [230, 400], [215, 400]]
    reference = evaluate_possible_region(polygon, 2, calibration)
    for scale in (.5, 2):
        changed = deepcopy(calibration)
        changed['image_width'] *= scale
        changed['image_height'] *= scale
        primary = changed['primary_court']
        for key in ('polygon_px', 'judgment_scope_polygon_px'):
            primary[key] = (np.asarray(primary[key])*scale).tolist()
        for boundary in primary['boundary_segments_px']:
            for key in ('segment_px', 'inside_point_px'):
                boundary[key] = (np.asarray(boundary[key])*scale).tolist()
            boundary['uncertainty_px'] *= scale
        result = evaluate_possible_region((np.asarray(polygon)*scale).tolist(), 2*scale, changed)
        assert result['decision'] == reference['decision'] == 'OUT'
        assert result['minimum_clearance_px'] == pytest.approx(reference['minimum_clearance_px']*scale, abs=.0002)


def selected_family_result():
    candidate = {'event_id': 'event_000', 'track_id': 1, 'frame_index': 12, 'time': .48,
                 'candidate_px': [460, 390], 'contact_observed': False, 'identity_confidence': .25}
    family = {'family_id': 'family_000', 'member_event_ids': ['event_000'],
              'members': [candidate], 'representative': candidate, 'track_ids': [1],
              'frame_interval': [11, 13], 'time_interval': [.44, .52],
              'physical_contact_confirmed': False, 'target_identity_confirmed': False}
    return {'status': 'selected', 'selected_family': family, 'families': [family],
            'excluded': [], 'reason_codes': ['UNIQUE_PRIMARY_COURT_FLIGHT_FAMILY']}


@pytest.mark.parametrize('resolution', [
    {'status': 'review', 'selected_family': None, 'families': [], 'reason_codes': ['NO_SUPPORTED_PRIMARY_COURT_FLIGHT_FAMILY']},
    {'status': 'review', 'selected_family': None, 'families': [
        {'family_id': 'family_000', 'members': []}, {'family_id': 'family_001', 'members': []}],
     'reason_codes': ['MULTIPLE_PRIMARY_COURT_FLIGHT_FAMILIES']},
])
def test_unresolved_family_never_calls_native_or_geometry(monkeypatch, resolution):
    from badminton import landing_families, native_landing, video_estimate

    monkeypatch.setattr(landing_families, 'resolve_landing_families', lambda *args: resolution)

    def forbidden(*args, **kwargs):
        pytest.fail('A missing unique family must stop before native evidence or geometry')

    monkeypatch.setattr(native_landing, 'estimate_native_landing', forbidden)
    monkeypatch.setattr(video_estimate, 'evaluate_possible_region', forbidden)
    before = deepcopy(resolution)
    result = video_estimate.estimate_video_verdict([], [], [], [0., .04], [1000, 700], court())
    assert result['decision'] == 'REVIEW' and result['eligible'] is False
    assert result['reason_codes'] == ['VIDEO_ESTIMATE_NO_UNIQUE_EVENT']
    assert result['family_resolution']['reason_codes'] == resolution['reason_codes']
    assert 'native_evidence' not in result and 'geometry' not in result
    assert resolution == before


def test_unresolved_native_evidence_never_calls_geometry(monkeypatch):
    from badminton import landing_families, native_landing, video_estimate

    resolution = selected_family_result()
    native = {'status': 'review', 'reason_codes': ['NATIVE_TRAJECTORY_UNSUPPORTED'],
              'physical_contact_confirmed': False}
    monkeypatch.setattr(landing_families, 'resolve_landing_families', lambda *args: resolution)
    monkeypatch.setattr(native_landing, 'estimate_native_landing', lambda *args, **kwargs: native)

    def forbidden(*args, **kwargs):
        pytest.fail('Unsupported native evidence must stop before line geometry')

    monkeypatch.setattr(video_estimate, 'evaluate_possible_region', forbidden)
    result = video_estimate.estimate_video_verdict([], [], [], [0., .04], [1000, 700], court())
    assert result['decision'] == 'REVIEW' and result['eligible'] is False
    assert result['reason_codes'] == ['VIDEO_ESTIMATE_NATIVE_SUPPORT_MISSING']
    assert result['selected_family_id'] == 'family_000'
    assert result['native_evidence'] == native and 'geometry' not in result
    assert result['contact_observed'] is False and result['target_identity_verified'] is False


@pytest.mark.parametrize('polygon,label', [
    ([[450, 380], [470, 380], [470, 400], [450, 400]], 'IN'),
    ([[215, 380], [230, 380], [230, 400], [215, 400]], 'OUT'),
    ([[240, 390], [290, 390], [290, 410], [240, 410]], 'REVIEW'),
])
def test_supported_native_region_passes_through_real_geometry_and_preserves_provenance(
        monkeypatch, tmp_path, polygon, label):
    from badminton import landing_families, native_landing, video_estimate

    resolution = selected_family_result()
    family = resolution['selected_family']
    landings = family['members']
    tracks = [{'track_id': 1, 'observations': []}]
    timestamps = [index * .04 for index in range(40)]
    source_frames, size, calibration = [object()], [1000, 700], court()
    rules, options = {'game': 'doubles', 'phase': 'rally'}, {'enabled': True}
    native = {'status': 'supported', 'possible_contact_polygon_px': polygon, 'uncertainty_px': 2.,
              'frame_interval': [11, 13], 'time_interval': [.44, .52],
              'artifact_path': 'evidence/native-landing.jpg', 'physical_contact_confirmed': False}
    calls = []
    before = deepcopy(resolution)

    def resolve(received_landings, received_tracks, received_times, received_calibration, received_size):
        assert received_landings is landings and received_tracks is tracks
        assert received_times is timestamps and received_calibration is calibration and received_size is size
        calls.append('family')
        return resolution

    def estimate(received_family, received_tracks, received_frames, received_times, received_size,
                 received_calibration, *, output_dir, config):
        assert received_family is family and received_family['members'] == landings
        assert received_tracks is tracks and received_frames is source_frames and received_times is timestamps
        assert received_size is size and received_calibration is calibration
        assert output_dir == tmp_path and config is options
        calls.append('native')
        return native

    monkeypatch.setattr(landing_families, 'resolve_landing_families', resolve)
    monkeypatch.setattr(native_landing, 'estimate_native_landing', estimate)
    # Keep real court geometry: native "supported" alone cannot make this IN/OUT.
    result = video_estimate.estimate_video_verdict(landings, tracks, source_frames, timestamps, size,
                                                  calibration, rules, output_dir=tmp_path, config=options)
    assert calls == ['family', 'native']
    assert result['decision'] == result['geometry']['decision'] == label
    assert result['eligible'] is (label != 'REVIEW')
    assert result['rules'] == result['geometry']['rules'] == rules
    assert result['possible_contact_polygon_px'] == polygon and result['uncertainty_px'] == 2.
    assert result['frame_interval'] == native['frame_interval']
    assert result['time_interval'] == native['time_interval']
    assert result['artifact_path'] == native['artifact_path']
    assert result['selected_family_id'] == family['family_id']
    assert result['member_event_ids'] == family['member_event_ids']
    assert result['native_evidence'] == native
    assert result['decision_kind'] == 'video_estimate' and result['source'] == video_estimate.SOURCE
    assert result['coordinate_system'] == 'analysis_pixels'
    assert result['contact_observed'] is False and result['target_identity_verified'] is False
    assert result['evaluation_eligible'] is False and result['calibrated_accuracy'] is None
    assert resolution == before
    assert 'members' not in result['family_resolution']['selected_family']
    assert 'members' not in result['family_resolution']['families'][0]
