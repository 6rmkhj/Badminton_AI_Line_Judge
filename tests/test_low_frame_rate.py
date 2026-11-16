"""Low frame-rate (15 fps) clips: a landing hidden in a sampling gap must stay usable.

The samples reproduce the 8k a7 rally re-encoded at 854x480 and 15 fps: a
clean incoming flight, one missing interval at the floor, then a small rebound.
"""
import numpy as np

from badminton.events import _occluded_rebound_candidates


def _track(rate):
    step = 1.0 / rate
    path = [(30, 304, 120), (31, 299, 128), (32, 294, 137), (33, 288, 149), (34, 282, 164), (36, 271, 201),
            (39, 256, 256), (40, 255, 253), (41, 252, 253), (42, 248, 252), (43, 246, 251)]
    observations = [{'frame_index': frame, 'time': frame * step, 'x': float(x), 'y': float(y), 'radius': 1.2,
                     'confidence': .6, 'observed': True, 'possible_occluder_overlap': False}
                    for frame, x, y in path]
    return {'track_id': 414, 'flight_hypothesis': {'method': 'test'}, 'observations': observations}, step


def test_fifteen_fps_gap_landing_keeps_its_flight_support():
    track, step = _track(15)
    timestamps = list(np.arange(80) * step)
    frames = [np.zeros((480, 854, 3), np.uint8)]
    proposals = _occluded_rebound_candidates(track, track['observations'], timestamps, frames, {}, step)
    assert proposals, 'the gap intersection itself must be proposed'
    support = proposals[0]['flight_trajectory_support']
    assert support['supported_for_review'] is True, support['reason_codes']
    assert support['post_observation_count'] >= 3
    assert abs(proposals[0]['candidate_px'][0] - 258) < 12


def test_rebound_window_is_unchanged_at_normal_frame_rates():
    """At 24 fps and above the fit window already holds five samples."""
    from badminton.events import REBOUND_CHECK_SAMPLES
    import math
    for rate in (24, 30, 60):
        window = max(3, min(32, int(math.ceil(.133 / (1.0 / rate))) + 1))
        assert window >= REBOUND_CHECK_SAMPLES
