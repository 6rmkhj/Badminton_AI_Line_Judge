import cv2
import numpy as np

from badminton.calibration import calibrate, COURT_CORNERS_M, TEMPLATE_LINES_M


def rendered_court(corners=None, missing=()):
    image = np.full((720, 1000, 3), (110, 130, 145), np.uint8)
    corners = np.array(corners or [[180, 660], [820, 660], [700, 70], [300, 70]], np.float32)
    h = cv2.getPerspectiveTransform(COURT_CORNERS_M, corners)
    for index, (a, b) in enumerate(TEMPLATE_LINES_M):
        if index in missing:
            continue
        projected = cv2.perspectiveTransform(np.array([[a, b]], np.float32), h)[0]
        cv2.line(image, tuple(np.rint(projected[0]).astype(int)), tuple(np.rint(projected[1]).astype(int)), (210, 70, 10), 4)
    return image, corners


def test_blank_and_one_line_cannot_produce_full_court_geometry():
    blank = np.zeros((360, 640, 3), np.uint8)
    assert calibrate([blank])["status"] == "failed"
    cv2.line(blank, (30, 270), (610, 200), (255, 50, 0), 5)
    result = calibrate([blank])
    assert result["status"] != "ok"
    assert result["homography"] is None


def test_full_projected_template_recovers_metric_court():
    image, corners = rendered_court()
    result = calibrate([image])
    selected = result["primary_court"]
    assert selected["status"] == "selected" and selected["scope"] == "full"
    assert len(selected["line_segments_px"]) == 4
    assert {edge["role"] for edge in selected["boundary_segments_px"]} == {"left", "right", "near", "far"}
    assert result["status"] == "ok", result
    assert result["supported_line_count"] >= 9
    mapped = cv2.perspectiveTransform(corners[None], np.array(result["homography"]))[0]
    assert np.max(np.linalg.norm(mapped - COURT_CORNERS_M, axis=1)) < .15
    assert result["metric_accuracy_validated"] is False


def test_missing_inner_court_lines_does_not_certify_arbitrary_quad():
    image, _ = rendered_court(missing=range(4, 12))
    result = calibrate([image])
    assert result["status"] != "ok"


def test_manual_reference_is_explicit_and_never_automatic_success():
    image, corners = rendered_court()
    result = calibrate([image], {"reference_points_px": corners.tolist()})
    assert result["status"] == "review"
    assert result["method"] == "manual_reference"
    assert result["homography"] is not None


def test_malformed_roi_and_reference_fail_cleanly():
    image, _ = rendered_court()
    assert calibrate([image], {"roi": [1, 2]})["status"] == "failed"
    assert calibrate([image], {"reference_points_px": [[1, 2]]})["status"] == "failed"
