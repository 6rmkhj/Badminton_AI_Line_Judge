"""Independent role, visibility and ambiguity gates for foreground selection."""
import cv2
import numpy as np
import pytest

from badminton.primary_court import select_primary_court


MODEL_LINES = [
    [[0, 0], [0, 6.7]], [[6.1, 0], [6.1, 6.7]],
    [[0, 0], [6.1, 0]], [[0, .76], [6.1, .76]],
    [[.46, 0], [.46, 6.7]], [[5.64, 0], [5.64, 6.7]],
    [[3.05, 0], [3.05, 4.72]], [[0, 4.72], [6.1, 4.72]],
]


def scene(scale=1, colour='blue'):
    corners = np.array([[180, 490], [830, 420], [650, 210], [340, 200]], np.float32)*scale
    transform = cv2.getPerspectiveTransform(
        np.array([[0, 0], [6.1, 0], [6.1, 6.7], [0, 6.7]], np.float32), corners)
    lines = cv2.perspectiveTransform(np.asarray(MODEL_LINES, np.float32).reshape(1, -1, 2), transform)[0].reshape(-1, 4)
    image = np.full((round(540*scale), round(960*scale), 3), (115, 140, 160), np.uint8)
    paint = {'blue': (210, 80, 40), 'white': (245, 245, 245), 'yellow': (20, 210, 235)}[colour]
    for x1, y1, x2, y2 in lines:
        cv2.line(image, tuple(np.rint([x1, y1]).astype(int)), tuple(np.rint([x2, y2]).astype(int)),
                 paint, max(2, round(4*scale)), cv2.LINE_AA)
    return image, {'line_colour': colour, 'line_segments_px': lines.tolist()}, corners


@pytest.mark.parametrize('colour', ['blue', 'white', 'yellow'])
@pytest.mark.parametrize('scale', [.5, 1, 1.5])
def test_role_pattern_selects_one_foreground_court_across_paint_and_resolution(colour, scale):
    image, observations, corners = scene(scale, colour)
    result = select_primary_court([image], observations)
    assert result['status'] == 'selected', result
    assert result['region_supported'] is True
    assert np.max(np.linalg.norm(np.array(result['near_corners_px'])-corners[:2], axis=1)) < 5*scale
    assert result['scope'] == 'near_half'
    assert result['metric_coordinates_available'] is False
    assert {line['role'] for line in result['boundary_segments_px']} == {'left', 'right', 'near'}
    assert all(line['observed'] for line in result['boundary_segments_px'])
    assert result['interior_limit'] == 'near_short_service_line_not_an_out_boundary'
    assert np.asarray(result['interior_polygon_px'])[2:, 1].min() > corners[2:, 1].max()


def test_missing_centre_role_does_not_promote_service_box_to_court():
    image, observations, _ = scene()
    observations['line_segments_px'].pop(6)
    result = select_primary_court([image], observations)
    assert result['status'] != 'selected'
    assert result['region_supported'] is False


def test_line_coordinates_alone_cannot_certify_paint_that_is_not_in_the_video():
    image, observations, _ = scene()
    image[:] = (115, 140, 160)
    result = select_primary_court([image], observations)
    assert result['status'] != 'selected'
    assert not result['boundary_segments_px']


def test_unrelated_candidates_never_leak_into_selected_output():
    image, observations, corners = scene()
    unrelated = [[0, 40, 950, 70], [30, 10, 60, 300], [60, 490, 100, 250]]
    observations['line_segments_px'].extend(unrelated)
    result = select_primary_court([image], observations)
    assert result['status'] == 'selected'
    assert np.max(np.linalg.norm(np.array(result['near_corners_px'])-corners[:2], axis=1)) < 5
    for line in result['line_segments_px']:
        assert not any(np.allclose(line, bad) for bad in unrelated)
    # Baseline stops at the target corners rather than extending to neighbours.
    baseline = next(line for line in result['boundary_segments_px'] if line['role'] == 'near')
    np.testing.assert_allclose(np.array(baseline['segment_px']).reshape(2, 2), result['near_corners_px'])


def test_mirrored_camera_view_selects_the_same_court():
    image, observations, corners = scene()
    lines = np.array(observations['line_segments_px'])
    lines[:, [0, 2]] = image.shape[1]-1-lines[:, [0, 2]]
    observations['line_segments_px'] = lines.tolist()
    result = select_primary_court([image[:, ::-1].copy()], observations)
    expected = corners[:2][::-1].copy()
    expected[:, 0] = image.shape[1]-1-expected[:, 0]
    assert result['status'] == 'selected'
    assert np.max(np.linalg.norm(np.array(result['near_corners_px'])-expected, axis=1)) < 5


@pytest.mark.parametrize('observations', [{}, {'line_segments_px': [1, 2, 3]},
                                         {'line_segments_px': [[0, 0, np.nan, 5]]}])
def test_incomplete_or_invalid_observations_are_not_accepted(observations):
    image, _, _ = scene()
    result = select_primary_court([image], observations)
    assert result['status'] == 'not_found'
    assert result['judgment_scope_polygon_px'] is None
