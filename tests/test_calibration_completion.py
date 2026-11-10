import cv2
import numpy as np
import pytest

from badminton.calibration import (
    COURT_CORNERS_M, TEMPLATE_LINES_M, assess_camera_stability, calibrate,
    guard_calibration_stability, lens_geometry, points_for_homography,
)


def court_image():
    image = np.full((540, 800, 3), (110, 130, 145), np.uint8)
    corners = np.array([[120, 500], [680, 500], [570, 40], [230, 40]], np.float32)
    transform = cv2.getPerspectiveTransform(COURT_CORNERS_M, corners)
    for start, end in TEMPLATE_LINES_M:
        line = cv2.perspectiveTransform(np.array([[start, end]], np.float32), transform)[0]
        cv2.line(image, tuple(np.rint(line[0]).astype(int)), tuple(np.rint(line[1]).astype(int)), (210, 70, 10), 4)
    return image, corners


def textured_image():
    rng = np.random.default_rng(51)
    image = np.full((300, 500, 3), 60, np.uint8)
    for point, shade in zip(rng.integers([12, 12], [488, 288], size=(100, 2)), rng.integers(90, 240, size=100)):
        cv2.rectangle(image, tuple(point), tuple(point + 6), (int(shade),) * 3, -1)
    return image


def test_static_background_survives_local_foreground_motion():
    first = textured_image()
    second = first.copy()
    cv2.rectangle(second, (150, 100), (205, 170), (255, 255, 255), -1)
    result = assess_camera_stability([first, second, first])
    assert result["status"] == "ok", result
    assert result["max_displacement_px"] < 1


def test_camera_motion_invalidates_previously_accepted_metric_fit():
    first = textured_image()
    moved = cv2.warpAffine(first, np.array([[1, 0, 11], [0, 1, 5]], float), (500, 300))
    stability = assess_camera_stability([first, moved])
    assert stability["status"] == "review", stability
    assert "CAMERA_MOVED_FROM_CALIBRATION_REFERENCE" in stability["reason_codes"]
    original = {"status": "ok", "homography": np.eye(3).tolist(),
                "metric_coordinates_available": True, "judgment_region_px": [[0, 0]], "reason_codes": []}
    guarded = guard_calibration_stability(original, stability)
    assert guarded["status"] == "review"
    assert guarded["homography"] == original["homography"]  # Fit remains a useful hypothesis.
    assert guarded["judgment_region_px"] is None
    assert guarded["metric_coordinates_available"] is False
    assert original["status"] == "ok"


def test_featureless_scene_does_not_claim_static_camera():
    blank = np.zeros((300, 500, 3), np.uint8)
    result = assess_camera_stability([blank, blank])
    assert result["status"] == "review"
    assert "CAMERA_STABILITY_UNOBSERVABLE" in result["reason_codes"]
    assert result["max_displacement_px"] is None


def test_default_checks_transient_motion_between_old_sample_positions():
    first = textured_image()
    moved = cv2.warpAffine(first, np.array([[1, 0, 10], [0, 1, 6]], float), (500, 300))
    frames = [first] * 25
    # The previous 24-point uniform sampling omitted frame 23 of a 25-frame clip.
    frames[23] = moved
    result = assess_camera_stability(frames)
    assert result["all_frames_sampled"] is True
    assert 23 in result["sampled_frame_indices"]
    assert result["status"] == "review"
    assert "CAMERA_MOVED_FROM_CALIBRATION_REFERENCE" in result["reason_codes"]


def test_explicit_subsampling_cannot_certify_unmeasured_contact_frames():
    first = textured_image()
    result = assess_camera_stability([first] * 25, {"max_sample_frames": 24})
    assert result["all_frames_sampled"] is False
    assert result["status"] == "review"
    assert "CAMERA_STABILITY_PARTIAL_COVERAGE" in result["reason_codes"]
    original = {"status": "ok", "reason_codes": [], "metric_coordinates_available": True}
    # The guard independently rejects incomplete coverage even if an imported
    # diagnostic incorrectly labels its sampled observations as accepted.
    guarded = guard_calibration_stability(original, {**result, "status": "ok"})
    assert guarded["status"] == "review"
    assert guarded["metric_coordinates_available"] is False


def test_partial_line_regions_never_imply_a_metric_plane():
    image = np.zeros((400, 640, 3), np.uint8)
    cv2.line(image, (30, 300), (610, 200), (220, 60, 0), 6)
    result = calibrate([image])
    assert result["status"] != "ok"
    assert result["partial_line_mode"] is True
    assert result["supported_line_regions_px"]
    assert result["homography"] is None
    assert result["judgment_region_px"] is None
    assert result["metric_coordinates_available"] is False
    assert all(region["court_line_role"] is None for region in result["supported_line_regions_px"])


def test_known_lens_intrinsics_scale_and_correct_observed_points():
    lens = lens_geometry({"camera_matrix": [[1200, 0, 800], [0, 1200, 600], [0, 0, 1]],
                          "dist_coeffs": [.12, -.025, .001, -.001, 0],
                          "image_size": [1600, 1200]}, 800, 600)
    matrix = np.asarray(lens["camera_matrix"])
    assert matrix[0, 0] == 600 and matrix[0, 2] == 400
    normalized = np.array([[-.5, -.3, 1], [.4, .2, 1]], float)
    distorted, _ = cv2.projectPoints(normalized, np.zeros(3), np.zeros(3), matrix, np.asarray(lens["dist_coeffs"]))
    corrected = points_for_homography(distorted.reshape(-1, 2), {
        "lens": lens, "homography_pixel_space": "undistorted_analysis_pixels"})
    expected = (normalized @ matrix.T)[:, :2]
    np.testing.assert_allclose(corrected, expected, atol=.002)
    np.testing.assert_array_equal(points_for_homography(expected, {}), expected)


def test_lens_calibration_declares_homography_space_and_keeps_raw_ui_polygon():
    image, corners = court_image()
    result = calibrate([image], config={"lens": {
        "camera_matrix": [[650, 0, 400], [0, 650, 270], [0, 0, 1]],
        "dist_coeffs": [0, 0, 0, 0, 0], "image_size": [800, 540]}})
    assert result["status"] == "ok", result
    assert result["lens_distortion_corrected"] is True
    assert result["homography_pixel_space"] == "undistorted_analysis_pixels"
    mapped = cv2.perspectiveTransform(points_for_homography(corners, result)[None], np.asarray(result["homography"]))[0]
    np.testing.assert_allclose(mapped, COURT_CORNERS_M, atol=.15)
    np.testing.assert_allclose(result["polygon_px"], result["polygon_homography_space_px"])


def test_invalid_lens_metadata_fails_without_ignoring_distortion():
    image, _ = court_image()
    result = calibrate([image], config={"lens": {"camera_matrix": np.eye(3).tolist(), "dist_coeffs": [1]}})
    assert result["status"] == "failed"
    assert result["reason_codes"] == ["INVALID_KNOWN_LENS_CALIBRATION"]
    with pytest.raises(ValueError):
        points_for_homography([[1, 2]], {"homography_pixel_space": "undistorted_analysis_pixels"})


def test_learned_line_mask_still_requires_court_geometry(monkeypatch):
    image = np.zeros((360, 640, 3), np.uint8)
    mask = np.zeros((1, 1, 360, 640), np.float32)
    cv2.rectangle(mask[0, 0], (100, 40), (540, 320), 1, 4)
    class FakeModel:
        def setInput(self, value):
            assert value.shape == (1, 3, 512, 512)
        def forward(self):
            return mask
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda _: FakeModel())
    result = calibrate([image], config={"line_model_path": "fixture.onnx"})
    assert result["method"] == "onnx_line_segmentation_hough_court_template"
    assert result["line_model_accuracy_validated"] is False
    assert result["status"] != "ok"


def test_invalid_learned_line_output_does_not_fallback_to_apparent_success(monkeypatch):
    image, _ = court_image()
    class FakeModel:
        def setInput(self, value):
            pass
        def forward(self):
            return np.full((1, 1, 32, 32), 17, np.float32)
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda _: FakeModel())
    result = calibrate([image], config={"line_model_path": "fixture.onnx"})
    assert result["status"] == "failed"
    assert result["reason_codes"] == ["LINE_SEGMENTATION_MODEL_FAILED"]


def test_lens_manual_reference_stays_reference_only():
    image, corners = court_image()
    result = calibrate([image], {"corners_px": corners.tolist()}, config={"lens": {
        "camera_matrix": [[650, 0, 400], [0, 650, 270], [0, 0, 1]], "dist_coeffs": [.05, 0, 0, 0]}})
    assert result["status"] == "review"
    assert result["manual_reference"] is True
    assert result["metric_coordinates_available"] is False
    np.testing.assert_allclose(result["polygon_px"], corners, atol=.01)
