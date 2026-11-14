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
