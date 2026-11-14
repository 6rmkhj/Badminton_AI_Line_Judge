import cv2
import numpy as np
import pytest

from badminton.calibration import COURT_CORNERS_M, TEMPLATE_LINES_M, calibrate


def test_full_metric_court_keeps_coordinate_system_but_changes_active_sidelines():
    frame = np.full((540, 800, 3), (110, 130, 145), np.uint8)
    corners = np.array([[120, 500], [680, 500], [570, 40], [230, 40]], np.float32)
    inverse = cv2.getPerspectiveTransform(COURT_CORNERS_M, corners)
    for endpoints in TEMPLATE_LINES_M:
        line = cv2.perspectiveTransform(np.array([endpoints], np.float32), inverse)[0]
        cv2.line(frame, tuple(np.rint(line[0]).astype(int)), tuple(np.rint(line[1]).astype(int)), (210, 70, 10), 4)
    doubles = calibrate([frame], config={'game': 'doubles'})
    singles = calibrate([frame], config={'game': 'singles'})
    assert doubles['status'] == singles['status'] == 'ok'
    np.testing.assert_allclose(singles['homography'], doubles['homography'])
    np.testing.assert_allclose(singles['polygon_px'], doubles['polygon_px'])
    active = np.array([singles['primary_court']['polygon_px']], np.float32)
    projected = cv2.perspectiveTransform(active, np.array(singles['homography']))[0]
    np.testing.assert_allclose(projected, [[.46, 0], [5.64, 0], [5.64, 13.4], [.46, 13.4]], atol=.001)
    assert singles['primary_court']['game'] == 'singles'
    assert doubles['primary_court']['game'] == 'doubles'


def test_active_game_overrides_nested_selector_mode(monkeypatch):
    import badminton.calibration as module
    captured = []
    monkeypatch.setattr(module, '_calibrate_full_court', lambda *args: {
        'status': 'review', 'line_segments_px': [[1, 2, 3, 4]], 'manual_reference': False})
    monkeypatch.setattr('badminton.primary_court.select_primary_court',
                        lambda frames, result, config: captured.append(config) or {'status': 'review'})
    calibrate([], config={'game': 'singles', 'primary_court': {'game': 'doubles'}})
    assert captured[0]['game'] == 'singles'


def test_invalid_game_is_not_silently_drawn_as_doubles():
    with pytest.raises(ValueError, match='game'):
        calibrate([], config={'game': 'mixed'})
