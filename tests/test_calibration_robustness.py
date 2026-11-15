"""Regression scenes with known geometry, illumination and competing courts.

These are deterministic geometry tests, not an estimate of real-video accuracy.
The coloured distractors deliberately have more pixels than the painted lines.
"""

import cv2
import numpy as np
import pytest

from badminton.calibration import (
    COURT_CORNERS_M,
    TEMPLATE_LINES_M,
    calibrate,
    segment_lines,
)


def projected_court(floor=(110, 130, 145), tape=(235, 235, 235), scale=1,
                    size=(800, 540), corners=None, thickness=4):
    width, height = size
    image = np.full((height * scale, width * scale, 3), floor, np.uint8)
    corners = np.array(corners if corners is not None else
                       [[120, 500], [680, 500], [570, 40], [230, 40]], np.float32) * scale
    transform = cv2.getPerspectiveTransform(COURT_CORNERS_M, corners)
    painted = np.zeros(image.shape[:2], np.uint8)
    for start, end in TEMPLATE_LINES_M:
        line = cv2.perspectiveTransform(np.array([[start, end]], np.float32), transform)[0]
        a, b = np.rint(line).astype(int)
        cv2.line(image, tuple(a), tuple(b), tape, thickness * scale)
        cv2.line(painted, tuple(a), tuple(b), 255, thickness * scale)
    return image, corners, painted


def assert_correct_plane(result, corners, tolerance_m=.20):
    assert result["status"] == "ok", {
        key: result.get(key)
        for key in ("status", "reason_codes", "confidence", "supported_line_count", "line_colour")
    }
    actual = cv2.perspectiveTransform(corners[None], np.asarray(result["homography"], dtype=float))[0]
    assert np.linalg.norm(actual - COURT_CORNERS_M, axis=1).max() < tolerance_m
    assert result["metric_coordinates_available"] is True
    # Synthetic line agreement must never advertise surveyed metric accuracy.
    assert result["metric_accuracy_validated"] is False


@pytest.mark.parametrize(
    "floor,tape",
    [
        ((110, 130, 145), (235, 235, 235)),
        ((140, 75, 35), (235, 235, 235)),
        ((45, 55, 60), (140, 140, 140)),
        ((55, 95, 65), (25, 210, 235)),
        ((110, 130, 145), (210, 70, 10)),
        ((110, 130, 145), (210, 210, 10)),
    ],
    ids=["white-wood", "white-blue-floor", "dim-white", "yellow-green-floor", "blue-wood", "cyan-wood"],
)
def test_court_colour_and_brightness_do_not_change_recovered_geometry(floor, tape):
    image, corners, _ = projected_court(floor, tape)
    assert_correct_plane(calibrate([image]), corners)


def test_blue_decoration_does_not_hide_a_white_court():
    image, corners, _ = projected_court()
    cv2.rectangle(image, (8, 90), (95, 330), (220, 65, 10), -1)
    assert_correct_plane(calibrate([image]), corners)


def test_blue_floor_is_not_segmented_as_tape():
    image, _, painted = projected_court((140, 75, 35))
    mask, _ = segment_lines(image)
    assert mask.dtype == np.uint8 and mask.shape == painted.shape
    assert np.count_nonzero(mask) < mask.size * .20
    assert np.mean(mask[painted != 0] != 0) > .80


@pytest.mark.parametrize("colour", [(150, 70, 30), (80, 140, 80), (220, 220, 220)])
def test_uniform_coloured_floor_has_no_detected_lines(colour):
    image = np.full((360, 640, 3), colour, np.uint8)
    mask, _ = segment_lines(image)
    result = calibrate([image])
    assert not np.any(mask)
    assert not result["line_segments_px"]
    assert result["status"] != "ok"
    assert result["metric_coordinates_available"] is False


def test_uneven_lighting_preserves_the_same_court_plane():
    image, corners, _ = projected_court((70, 90, 95))
    brightness = np.linspace(.38, 1.0, image.shape[1], dtype=np.float32)[None, :, None]
    image = np.rint(image * brightness).astype(np.uint8)
    assert_correct_plane(calibrate([image]), corners)


def test_moving_occluders_are_removed_by_temporal_background():
    image, corners, _ = projected_court()
    frames = []
    for left in np.linspace(160, 580, 9).astype(int):
        frame = image.copy()
        cv2.rectangle(frame, (left, 110), (left + 45, 350), (20, 30, 30), -1)
        frames.append(frame)
    assert_correct_plane(calibrate(frames), corners)


def test_double_resolution_keeps_metric_geometry_and_pixel_coordinates():
    image, corners, _ = projected_court(scale=2)
    result = calibrate([image])
    assert_correct_plane(result, corners)
    assert (result["image_width"], result["image_height"]) == (1600, 1080)
    polygon = np.asarray(result["polygon_px"])
    assert np.linalg.norm(polygon - corners, axis=1).max() < 12


def test_two_complete_courts_require_target_selection_across_colours():
    white, _, _ = projected_court()
    blue, _, _ = projected_court(tape=(210, 70, 10))
    image = np.concatenate([white, blue], axis=1)
    result = calibrate([image])
    assert result["status"] != "ok"
    assert result["metric_coordinates_available"] is False
    assert result["judgment_region_px"] is None


def test_roi_selects_a_target_court_without_manual_metric_corners():
    white, corners, _ = projected_court()
    blue, _, _ = projected_court(tape=(210, 70, 10))
    image = np.concatenate([white, blue], axis=1)
    roi = [[0, 0], [799, 0], [799, 539], [0, 539]]
    result = calibrate([image], {"roi_polygon_px": roi})
    assert_correct_plane(result, corners)
    assert result["manual_reference"] is False
    for segment in result["line_segments_px"]:
        assert max(segment[0], segment[2]) < 800


def test_dim_partial_line_remains_visible_without_inventing_metric_court():
    image = np.full((400, 640, 3), (40, 50, 55), np.uint8)
    cv2.line(image, (30, 300), (610, 200), (130, 130, 130), 5)
    result = calibrate([image])
    assert result["line_segments_px"]
    assert result["supported_line_regions_px"]
    assert result["partial_line_mode"] is True
    assert result["status"] != "ok"
    assert result["homography"] is None
    assert result["metric_coordinates_available"] is False
    assert all(region["court_line_role"] is None for region in result["supported_line_regions_px"])


def test_clipped_baseline_cannot_be_replaced_by_an_inner_service_line():
    # The near baseline is below the image. Mapping an internal service line
    # onto that baseline produces a plausible polygon but a wrong metric plane.
    image, _, _ = projected_court(
        tape=(210, 70, 10), size=(1000, 720),
        corners=[[-70, 780], [1070, 780], [660, 100], [340, 100]],
    )
    result = calibrate([image])
    assert result["line_segments_px"]
    assert result["status"] == "review"
    assert result["metric_coordinates_available"] is False
    assert result["judgment_region_px"] is None


def test_thick_tape_is_fitted_to_its_centerline():
    # Thick paint has two long edges. Choosing either edge shifts every line
    # intersection even when the resulting template has a high support score.
    image, corners, _ = projected_court(
        tape=(210, 70, 10), size=(1000, 720), thickness=12,
        corners=[[180, 660], [820, 660], [700, 70], [300, 70]],
    )
    result = calibrate([image])
    assert_correct_plane(result, corners)
    error_px = np.linalg.norm(np.asarray(result["polygon_px"]) - corners, axis=1)
    assert error_px.max() < 3, error_px.tolist()


@pytest.mark.parametrize("shape", [(0, 640, 3), (360, 0, 3)])
def test_empty_frame_dimensions_fail_without_opencv_exception(shape):
    result = calibrate([np.empty(shape, np.uint8)])
    assert result["status"] == "failed"
    assert result["reason_codes"] == ["INVALID_VIDEO_FRAME_SHAPE"]
    assert result["homography"] is None
    assert result["line_segments_px"] == []
    assert result["metric_coordinates_available"] is False


def test_white_sign_does_not_replace_detected_partial_blue_court_lines():
    image = np.full((540, 960, 3), (110, 130, 145), np.uint8)
    line_heights = [300, 350, 400, 450]
    for y in line_heights:
        cv2.line(image, (50, y), (910, y - 25), (210, 70, 10), 5)
    cv2.rectangle(image, (300, 35), (600, 245), (235, 235, 235), 4)

    result = calibrate([image])
    assert result["status"] != "ok"
    assert result["metric_coordinates_available"] is False
    assert result["judgment_region_px"] is None
    assert result["supported_line_regions_px"]
    # Check the actual long painted segments, rather than just their count or
    # colour label: a rectangular sign also supplies four plausible lines.
    for y in line_heights:
        assert any(
            abs(x2 - x1) > 600
            and abs(y1 - (y - (x1 - 50) * 25 / 860)) < 3
            and abs(y2 - (y - (x2 - 50) * 25 / 860)) < 3
            for x1, y1, x2, y2 in result["line_segments_px"]
        ), {"missing_line_y": y, "segments": result["line_segments_px"]}


@pytest.mark.parametrize(
    "inside,outside",
    [((140, 75, 35), (45, 45, 120)), ((60, 120, 65), (90, 90, 90))],
    ids=["blue-court-red-surround", "green-court-gray-surround"],
)
def test_full_court_surface_can_differ_from_surrounding_floor(inside, outside):
    _, corners, painted = projected_court()
    image = np.full((540, 800, 3), outside, np.uint8)
    cv2.fillPoly(image, [corners.astype(np.int32)], inside)
    image[painted != 0] = (235, 235, 235)
    assert_correct_plane(calibrate([image]), corners)


def test_partial_boundary_line_survives_different_floor_colours_on_each_side():
    image = np.full((540, 960, 3), (60, 120, 65), np.uint8)
    # The painted boundary separates two floor surfaces. Its neighbours are
    # not expected to have the same colour or match a single global average.
    cv2.fillPoly(image, [np.array([[360, 0], [959, 0], [959, 539], [600, 539]])],
                 (45, 45, 120))
    a = np.array([360 + 240 * 40 / 539, 40], dtype=float)
    b = np.array([360 + 240 * 500 / 539, 500], dtype=float)
    cv2.line(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
             (235, 235, 235), 5)
    result = calibrate([image])
    assert result["status"] != "ok"
    assert result["metric_coordinates_available"] is False
    assert result["homography"] is None
    assert result["supported_line_regions_px"]
    direction = b - a
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    assert any(
        np.linalg.norm(np.asarray(segment[2:]) - segment[:2]) > 400
        and np.max(np.abs((np.asarray(segment).reshape(2, 2) - a) @ normal)) < 3
        for segment in result["line_segments_px"]
    ), result["line_segments_px"]


def test_partial_court_interior_lines_survive_a_different_surround_surface():
    corners = np.array([[180, 450], [780, 450], [680, 90], [280, 90]], np.float32)
    image = np.full((540, 960, 3), (45, 45, 120), np.uint8)
    cv2.fillPoly(image, [corners.astype(np.int32)], (60, 120, 65))
    transform = cv2.getPerspectiveTransform(COURT_CORNERS_M, corners)
    references = []
    for index, (start, end) in enumerate(TEMPLATE_LINES_M):
        if index in (3, 7, 9, 11):
            continue
        line = cv2.perspectiveTransform(np.array([[start, end]], np.float32), transform)[0]
        a, b = np.rint(line).astype(int)
        cv2.line(image, tuple(a), tuple(b), (235, 235, 235), 4)
        references.append((index, line))

    result = calibrate([image])
    assert result["status"] != "ok"
    assert result["metric_coordinates_available"] is False
    assert result["line_segments_px"]
    segments = np.asarray(result["line_segments_px"]).reshape(-1, 2, 2)
    starts, vectors = segments[:, 0], segments[:, 1] - segments[:, 0]
    lengths_squared = np.sum(vectors ** 2, axis=1)
    for index, (a, b) in references:
        # Each painted reference needs support along its extent. Retaining just
        # the three outer boundaries while removing interior lines must fail.
        samples = a + np.linspace(.05, .95, 50)[:, None] * (b - a)
        offsets = samples[:, None] - starts
        position = np.sum(offsets * vectors, axis=2) / np.maximum(lengths_squared, 1e-9)
        nearest = starts + np.clip(position, 0, 1)[:, :, None] * vectors
        errors = np.linalg.norm(samples[:, None] - nearest, axis=2).min(axis=1)
        coverage = np.mean(errors <= 3)
        assert coverage >= .65, {"template_line": index, "coverage": coverage}
