"""Near-half court fallback for cameras the behind-the-baseline selector rejects.

Fixtures are 960x540 per-pixel medians of the nine calibration frames of two
B-side views (``b시점 17`` and ``b시점 10``). The probe points are AI visual-review
landing locations; they are evaluation references, never detector inputs.
"""
from pathlib import Path

import pytest

from badminton.best_estimate import classify_point
from badminton.calibration import calibrate
from badminton.image_io import read_image

FIXTURES = Path(__file__).parent / 'fixtures'
STABLE = {'status': 'ok', 'all_frames_sampled': True, 'reason_codes': []}


def court(name, game='doubles', **options):
    frame = read_image(FIXTURES / 'near-half' / name)
    assert frame is not None and frame.shape == (540, 960, 3)
    return calibrate([frame], None, {'game': game, **options})


@pytest.mark.parametrize(('name', 'point', 'verdict'), [
    ('b17-background.png', (349, 312), 'IN'),   # inside the near doubles alley
    ('b10-background.png', (748, 477), 'OUT'),  # behind the near back line
])
def test_fallback_selects_the_near_court_and_judges_reference_points(name, point, verdict):
    calibration = court(name)
    primary = calibration['primary_court']
    assert primary['status'] == 'selected' and primary['fallback'] is True
    assert primary['game'] == 'doubles' and primary['rules'] == {'game': 'doubles', 'phase': 'rally'}
    assert {row['role'] for row in primary['boundary_segments_px']} == {'left', 'right', 'near'}
    assert all(row['observed'] is True and row['uncertainty_px'] > 0 for row in primary['boundary_segments_px'])
    assert primary['interior_limit'] == 'near_short_service_line_not_an_out_boundary'
    assert calibration['primary_court_selector_attempt']['status'] != 'selected'
    row = classify_point(point, {**calibration, 'camera_stability': STABLE}, {'game': 'doubles', 'phase': 'rally'})
    assert row['method'] == 'primary_court_boundaries' and row['decision'] == verdict
    assert abs(row['margin_px']) > 5


def test_singles_fallback_uses_the_inner_sidelines():
    doubles = court('b17-background.png')
    singles = court('b17-background.png', game='singles')
    assert singles['primary_court']['status'] == 'selected' and singles['primary_court']['game'] == 'singles'
    point = (349, 312)
    wide = classify_point(point, {**doubles, 'camera_stability': STABLE}, {'game': 'doubles', 'phase': 'rally'})
    narrow = classify_point(point, {**singles, 'camera_stability': STABLE}, {'game': 'singles', 'phase': 'rally'})
    assert narrow['margin_px'] < wide['margin_px']


def test_fallback_can_be_disabled_and_never_replaces_an_accepted_court():
    disabled = court('b17-background.png', primary_court={'near_half_fallback': False})
    assert disabled['primary_court']['status'] != 'selected'
    assert 'primary_court_selector_attempt' not in disabled
    original = read_image(FIXTURES / 'primary-court' / 'A.png')
    accepted = calibrate([original], None, {'game': 'doubles'})
    assert accepted['primary_court']['status'] == 'selected'
    assert accepted['primary_court'].get('fallback') is not True
    assert 'primary_court_selector_attempt' not in accepted
