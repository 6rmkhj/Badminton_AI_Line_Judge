from copy import deepcopy

import cv2
import numpy as np
import pytest

from badminton.evidence import draw_court_lines, write_calibration_overlay
from badminton.pair_workspace import _calibration_preview, _calibration_summary
from badminton.pipeline import _draw_frame


def partial_calibration():
    return {'status': 'review', 'line_segments_px': [[20, 100, 200, 100]],
            'polygon_px': [[50, 60], [180, 60], [180, 140], [50, 140]],
            'metric_coordinates_available': False,
            'reason_codes': ['PARTIAL_LINES_NO_UNIQUE_COURT_PLANE']}


def test_partial_line_candidates_visible_without_drawing_an_unconfirmed_court():
    frame = np.zeros((160, 240, 3), np.uint8)
    calibration = partial_calibration()
    original = deepcopy(calibration)
    overview = _draw_frame(frame, calibration, [], 0)
    assert overview[100, 100].any()  # Observed segment is retained on REVIEW.
    assert not overview[60, 100].any()  # Unconfirmed quadrilateral stays hidden.
    assert not frame.any()
    assert calibration == original


def test_camera_motion_prevents_reusing_reference_lines_on_replay():
    calibration = partial_calibration()
    calibration['camera_stability'] = {'reason_codes': ['CAMERA_MOVED_FROM_CALIBRATION_REFERENCE']}
    canvas = np.zeros((160, 240, 3), np.uint8)
    draw_court_lines(canvas, calibration)
    assert not canvas.any()
    draw_court_lines(canvas, calibration, reference=True)
    assert canvas[100, 100].any()


def test_calibration_preview_saved_even_without_a_court_plane(tmp_path):
    frames = [np.zeros((160, 240, 3), np.uint8)] * 3
    calibration = partial_calibration()
    relative = write_calibration_overlay(frames, calibration, tmp_path)
    image = cv2.imread(str(tmp_path / relative))
    assert image.shape == frames[0].shape
    assert image[100, 100].max() > 100
    assert _calibration_preview(tmp_path, {'calibration_overlay': relative}, '/pair-media/test/') == '/pair-media/test/' + relative
    summary = _calibration_summary({'calibration': calibration})
    assert summary['line_detection_status'] == 'detected'
    assert summary['detected_line_count'] == 1
    assert summary['court_geometry_status'] == 'unconfirmed'
    assert summary['metric_accuracy_validated'] is False


def test_historical_partial_regions_remain_available_in_diagnostics():
    calibration = {'status': 'failed', 'supported_line_regions_px': [
        {'line_segment_px': [20, 100, 200, 100]}, {'line_segment_px': [1, 2, float('nan'), 4]}]}
    summary = _calibration_summary({'calibration': calibration})
    assert summary['detected_line_count'] == 1
    assert summary['court_geometry_status'] == 'unconfirmed'
    canvas = np.zeros((160, 240, 3), np.uint8)
    draw_court_lines(canvas, calibration)
    assert canvas[100, 100].any()
    assert _calibration_summary({}) is None


@pytest.mark.parametrize('relative', ['../outside.jpg', 'evidence/../../outside.jpg', 'https://example.com/image.jpg', 'evidence/preview.svg'])
def test_calibration_preview_does_not_expose_external_paths(tmp_path, relative):
    assert _calibration_preview(tmp_path, {'calibration_overlay': relative}, '/pair-media/test/') is None


def test_manual_geometry_does_not_become_confirmed_in_ui_summary():
    summary = _calibration_summary({'calibration': {'status': 'ok', 'manual_reference': True}})
    assert summary['court_geometry_status'] == 'unconfirmed'
    assert summary['line_detection_status'] == 'not_detected'


def selected_calibration():
    calibration = partial_calibration()
    calibration['camera_stability'] = {'status': 'ok', 'all_frames_sampled': True, 'reason_codes': []}
    calibration['primary_court'] = {
        'status': 'selected', 'scope': 'near_half', 'region_supported': True,
        'line_segments_px': [[50, 60, 50, 140], [50, 140, 180, 140], [180, 140, 180, 60]],
        'boundary_segments_px': [
            {'role': 'left', 'segment_px': [50, 60, 50, 140], 'observed': True},
            {'role': 'near', 'segment_px': [50, 140, 180, 140], 'observed': True},
            {'role': 'right', 'segment_px': [180, 140, 180, 60], 'observed': True}],
        'near_corners_px': [[50, 140], [180, 140]],
        'polygon_px': [[50, 60], [180, 60], [180, 140], [50, 140]],
        'interior_polygon_px': [[50, 60], [180, 60], [180, 140], [50, 140]],
    }
    return calibration


def test_selected_court_hides_neighbour_candidates_and_does_not_invent_far_edge():
    canvas = np.zeros((160, 240, 3), np.uint8)
    calibration = selected_calibration()
    original = deepcopy(calibration)
    draw_court_lines(canvas, calibration)
    assert not canvas[100, 25].any()  # The old candidate outside the selected court.
    assert canvas[140, 100].max() > 150  # Actual near boundary.
    assert canvas[100, 50].max() > 150  # Actual side boundary.
    assert canvas[60, 100].max() < 30  # Interior closure is only faint fill, not a line.
    assert 0 < canvas[80, 100].max() < 30
    assert calibration == original


def test_unresolved_primary_selection_suppresses_global_fallback_even_on_accepted_fit():
    canvas = np.zeros((160, 240, 3), np.uint8)
    calibration = selected_calibration()
    calibration['status'] = 'ok'
    calibration['primary_court']['status'] = 'review'
    draw_court_lines(canvas, calibration)
    assert not canvas.any()
    summary = _calibration_summary({'calibration': calibration})
    assert summary['detected_line_count'] == 0
    assert summary['primary_court_status'] == 'review'
    assert summary['local_region_supported'] is False


def test_selected_court_summary_separates_local_judgment_from_metric_accuracy():
    summary = _calibration_summary({'calibration': selected_calibration()})
    assert summary['primary_court_status'] == 'selected'
    assert summary['detected_line_count'] == 3
    assert summary['judgment_scope'] == 'near_half'
    assert summary['local_region_supported'] is True
    assert summary['court_geometry_status'] == 'unconfirmed'
    assert summary['metric_accuracy_validated'] is False
    assert '주 코트 하나를 선택' in summary['detail']


def test_selected_court_camera_movement_disables_replay_and_local_readiness():
    canvas = np.zeros((160, 240, 3), np.uint8)
    calibration = selected_calibration()
    calibration['camera_stability'] = {'reason_codes': ['CAMERA_MOVED_FROM_CALIBRATION_REFERENCE']}
    draw_court_lines(canvas, calibration)
    assert not canvas.any()
    summary = _calibration_summary({'calibration': calibration})
    assert summary['camera_moved'] is True
    assert summary['local_region_supported'] is False
    draw_court_lines(canvas, calibration, reference=True)
    assert canvas[140, 100].max() > 150


def test_unobserved_boundary_does_not_become_a_solid_court_line():
    canvas = np.zeros((160, 240, 3), np.uint8)
    calibration = selected_calibration()
    calibration['primary_court']['boundary_segments_px'].append(
        {'role': 'far', 'segment_px': [50, 60, 180, 60], 'observed': False})
    draw_court_lines(canvas, calibration)
    assert canvas[60, 100].max() < 30


@pytest.mark.parametrize('stability', [None, {}, {'status': 'ok'},
    {'status': 'ok', 'all_frames_sampled': False}, {'status': 'review', 'all_frames_sampled': True}])
def test_local_readiness_requires_camera_stability_for_every_frame(stability):
    calibration = selected_calibration()
    calibration['camera_stability'] = stability
    summary = _calibration_summary({'calibration': calibration})
    assert summary['primary_court_status'] == 'selected'
    assert summary['local_region_supported'] is False
    assert 'IN/OUT을 확정하지 않습니다' in summary['detail']
