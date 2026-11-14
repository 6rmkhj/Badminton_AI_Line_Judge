"""The same observed court must use the boundary for the requested game.

Real-scene coordinates below are assertion-only approximations from the visible
inner sidelines. Synthetic contact inputs deliberately certify the observation
gates to test geometry; they are not automatic video contact accuracy labels.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from badminton.decision import decide
from badminton.primary_court import select_primary_court


FIXTURES = Path(__file__).parent / 'fixtures' / 'primary-court'
REFERENCES = {'A': [[431, 464], [801, 300]], 'B': [[447, 471], [869, 324]]}


def actual_scene(view):
    return (cv2.imread(str(FIXTURES / f'{view}.png')),
            json.loads((FIXTURES / f'{view}-observations.json').read_text()))


def contact(point, **changes):
    return {'contact_observed': True, 'contact_px': list(point), 'confidence': .95,
            'identity_confidence': .95, 'contact_radius_px': 0, 'uncertainty_px': 1,
            'frame_index': 49, 'time': 1.633267, **changes}


def geometry_scene(scale=1, mirror=False):
    corners = np.array([[180, 490], [830, 420], [650, 210], [340, 200]], np.float32)*scale
    transform = cv2.getPerspectiveTransform(
        np.array([[0, 0], [6.1, 0], [6.1, 6.7], [0, 6.7]], np.float32), corners)
    model = [
        [[0, 0], [0, 6.7]], [[6.1, 0], [6.1, 6.7]],
        [[0, 0], [6.1, 0]], [[0, .76], [6.1, .76]],
        [[.46, 0], [.46, 6.7]], [[5.64, 0], [5.64, 6.7]],
        [[3.05, 0], [3.05, 4.72]], [[0, 4.72], [6.1, 4.72]],
    ]
    lines = cv2.perspectiveTransform(np.asarray(model, np.float32).reshape(1, -1, 2), transform)[0].reshape(-1, 4)
    expected = cv2.perspectiveTransform(np.array([[[.46, 0], [5.64, 0]]], np.float32), transform)[0]
    frame = np.full((round(540*scale), round(960*scale), 3), (115, 140, 160), np.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(frame, tuple(np.rint([x1, y1]).astype(int)), tuple(np.rint([x2, y2]).astype(int)),
                 (210, 80, 40), max(2, round(4*scale)), cv2.LINE_AA)
    if mirror:
        frame = frame[:, ::-1].copy()
        lines[:, [0, 2]] = frame.shape[1]-1-lines[:, [0, 2]]
        expected = expected[::-1].copy()
        expected[:, 0] = frame.shape[1]-1-expected[:, 0]
    return frame, {'line_colour': 'blue', 'line_segments_px': lines.tolist()}, expected


@pytest.mark.parametrize('view', ['A', 'B'])
def test_real_singles_selection_uses_both_inner_sidelines(view):
    frame, calibration = actual_scene(view)
    doubles = select_primary_court([frame], calibration, {'game': 'doubles'})
    singles = select_primary_court([frame], calibration, {'game': 'singles'})
    assert doubles['status'] == singles['status'] == 'selected'
    assert doubles['game'] == 'doubles'
    assert singles['game'] == 'singles'
    assert singles['rules'] == {'game': 'singles', 'phase': 'rally'}
    np.testing.assert_allclose(singles['near_corners_px'], REFERENCES[view], atol=12)
    assert cv2.contourArea(np.asarray(singles['polygon_px'], np.float32)) < cv2.contourArea(
        np.asarray(doubles['polygon_px'], np.float32))
    assert {boundary['role'] for boundary in singles['boundary_segments_px']} == {'left', 'right', 'near'}
    assert all(boundary['observed'] is True for boundary in singles['boundary_segments_px'])
    for segment in singles['line_segments_px']:
        # A changed label alone is insufficient: no old doubles side boundary
        # or service-line endpoint should remain outside the active court.
        for point in np.asarray(segment).reshape(2, 2):
            assert cv2.pointPolygonTest(np.asarray(singles['polygon_px'], np.float32), tuple(point), True) >= -7


@pytest.mark.parametrize('scale', [.5, 1, 1.5])
@pytest.mark.parametrize('mirror', [False, True])
def test_singles_boundary_selection_scales_and_mirrors(scale, mirror):
    frame, observations, expected = geometry_scene(scale, mirror)
    selected = select_primary_court([frame], observations, {'game': 'singles'})
    assert selected['status'] == 'selected', selected
    np.testing.assert_allclose(selected['near_corners_px'], expected, atol=5*scale)


def test_actual_a_doubles_alley_contact_is_in_for_doubles_and_out_for_singles():
    frame, observed = actual_scene('A')
    footprint = [[353, 435], [370, 435], [370, 454], [353, 454]]
    event = contact([361, 447], contact_polygon_px=footprint, uncertainty_px=6)
    for game, expected in [('doubles', 'IN'), ('singles', 'OUT')]:
        selected = select_primary_court([frame], observed, {'game': game})
        calibration = {**observed, 'primary_court': selected}
        verdict = decide(event, calibration, {'game': game, 'phase': 'rally'})
        assert verdict['decision'] == expected, verdict
        assert verdict.get('court_position_m') is None
        assert verdict['contact_time_seconds'] == event['time']
        # The supported scope must include the physical side alley even when
        # that alley is no longer part of the active singles IN region.
        scope = np.asarray(selected['judgment_scope_polygon_px'], np.float32)
        assert all(cv2.pointPolygonTest(scope, tuple(map(float, point)), True) > 12 for point in footprint)


@pytest.mark.parametrize(('selected_game', 'requested_game'), [('singles', 'doubles'), ('doubles', 'singles')])
def test_boundary_rule_mismatch_never_reuses_the_other_game(selected_game, requested_game):
    frame, observed = actual_scene('A')
    selected = select_primary_court([frame], observed, {'game': selected_game})
    result = decide(contact([700, 300]), {**observed, 'primary_court': selected},
                    {'game': requested_game, 'phase': 'rally'})
    assert result['decision'] == 'REVIEW'


def test_legacy_primary_without_rule_metadata_remains_doubles_only():
    frame, observed = actual_scene('A')
    selected = select_primary_court([frame], observed, {'game': 'doubles'})
    selected.pop('game', None)
    selected.pop('rules', None)
    calibration = {**observed, 'primary_court': selected}
    assert decide(contact([700, 300]), calibration)['decision'] == 'IN'
    assert decide(contact([700, 300]), calibration, {'game': 'singles'})['decision'] == 'REVIEW'


def test_inconsistent_rule_metadata_is_not_accepted():
    frame, observed = actual_scene('A')
    selected = select_primary_court([frame], observed, {'game': 'singles'})
    selected['rules'] = {'game': 'doubles', 'phase': 'rally'}
    result = decide(contact([700, 300]), {**observed, 'primary_court': selected}, {'game': 'singles'})
    assert result['decision'] == 'REVIEW'


def test_changing_only_game_label_cannot_promote_legacy_doubles_boundaries_to_singles():
    frame, observed = actual_scene('A')
    selected = select_primary_court([frame], observed, {'game': 'doubles'})
    selected.pop('rules')
    selected['game'] = 'singles'
    result = decide(contact([361, 447]), {**observed, 'primary_court': selected}, {'game': 'singles'})
    assert result['decision'] == 'REVIEW'


@pytest.mark.parametrize('change', [{'contact_observed': False}, {'identity_confidence': .2},
                                    {'confidence': .2}, {'occluded': True}])
def test_singles_selection_does_not_promote_missing_contact_or_identity(change):
    frame, observed = actual_scene('A')
    selected = select_primary_court([frame], observed, {'game': 'singles'})
    result = decide(contact([361, 447], **change), {**observed, 'primary_court': selected}, {'game': 'singles'})
    assert result['decision'] == 'REVIEW'


def test_singles_tape_must_be_visible_in_source_pixels():
    frame, observed, _ = geometry_scene()
    # Keep the detector's line coordinates but remove the actual singles tape.
    # A prediction from regulation spacing cannot substitute for measured paint.
    for x1, y1, x2, y2 in np.asarray(observed['line_segments_px'])[4:6]:
        cv2.line(frame, tuple(np.rint([x1, y1]).astype(int)), tuple(np.rint([x2, y2]).astype(int)),
                 (115, 140, 160), 14, cv2.LINE_AA)
    selected = select_primary_court([frame], observed, {'game': 'singles'})
    assert selected['status'] != 'selected'
    assert not selected['boundary_segments_px']


@pytest.mark.parametrize('config', ['singles', {'game': None}, {'game': 'mixed'},
                                    {'game': ['singles']}, {'game': True}])
def test_invalid_game_configuration_does_not_silently_select_doubles(config):
    frame, observed, _ = geometry_scene()
    selected = select_primary_court([frame], observed, config)
    assert selected['status'] != 'selected'
    assert selected['reason_codes'] == ['PRIMARY_COURT_RULES_UNSUPPORTED']
