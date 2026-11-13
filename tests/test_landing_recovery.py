from copy import deepcopy

import cv2
import numpy as np
import pytest

from badminton.landing_recovery import _components, recover_rebound_tracks


def frame(point=None, scale=1):
    image = np.full((100 * scale, 100 * scale, 3), 60, dtype=np.uint8)
    if point:
        cv2.circle(image, tuple(round(v * scale) for v in point), 2 * scale, (240, 240, 240), -1)
    return image


def approach():
    return {'track_id': 2, 'identity_ambiguous': True,
        'flight_hypothesis': {'approach_count': 6, 'rebound_count': 0},
        'observations': [{'frame_index': i, 'time': i / 30, 'x': 40., 'y': 20. + i * 5,
                          'radius': 2., 'confidence': .7, 'observed': True,
                          'possible_occluder_overlap': False} for i in range(6)]}


def test_components_need_current_pixel_change_on_both_sides():
    still = frame((40, 45))
    blank = frame()
    options = dict(origin=(0, 0), scale=(1, 1), frame_index=4, timestamp=4 / 30, radius=2)
    assert _components(still, still, still, **options) == []
    assert _components(still, blank, still, **options) == []
    assert _components(blank, still, still, **options) == []
    found = _components(blank, still, blank, **options)
    assert len(found) == 1
    assert [found[0]['x'], found[0]['y']] == [40., 45.]
    assert found[0]['contact_observed'] is False
    assert found[0]['identity_verified'] is False


def test_source_coordinates_scale_without_projecting_new_observations():
    options = dict(origin=(100, 80), scale=(2, 2), frame_index=7, timestamp=.2, radius=2)
    found = _components(frame(scale=2), frame((30, 40), scale=2), frame(scale=2), **options)
    assert len(found) == 1
    assert found[0]['source_px'] == [160., 160.]
    assert [found[0]['x'], found[0]['y']] == [80., 80.]
    assert found[0]['observed'] is True


def test_recovers_measured_rebound_without_mutating_or_qualifying_parent_track():
    track = approach()
    track.update(observation_count=999, observed_frame_fraction=1., motion_span_px=999,
                 identity_confidence=.99, target_court_confirmed=True,
                 moving_observation_support={'transient_frame_indices': [999]})
    original = deepcopy(track)
    ys = [20, 25, 30, 35, 40, 45, 42, 39, 36, 34, 32, 30, 28, 26, 24, 22]
    frames = [frame((40, y)) for y in ys]
    recovered, summary = recover_rebound_tracks([track], frames, [i / 30 for i in range(len(frames))], (100, 100))
    assert recovered and summary['additional_tracks'] == len(recovered)
    assert track == original
    assert summary['crop_bytes'] <= summary['crop_budget_bytes']
    for candidate in recovered:
        assert candidate['identity_ambiguous'] is True
        assert candidate['flight_hypothesis']['physical_contact_confirmed'] is False
        assert candidate['observation_count'] == len(candidate['observations'])
        assert candidate['target_court_confirmed'] is False
        assert candidate['identity_confidence'] < .5
        assert 999 not in candidate['moving_observation_support']['transient_frame_indices']
        for row in candidate['observations'][6:]:
            assert row['observed'] is True
            assert row['source_px'][0] == 40.
            assert abs(row['source_px'][1] - ys[row['frame_index']]) <= 2
            x, y = [round(v) for v in row['source_px']]
            assert np.all(frames[row['frame_index']][y, x] == 240)
            assert row['method'] == 'native_bidirectional_temporal_component'
            assert row['source_size'] == [100, 100]
            assert row['temporal_comparison_frame_indices'][0] < row['frame_index']
            assert row['temporal_comparison_frame_indices'][1] > row['frame_index']


def test_stationary_patch_never_creates_rebound_despite_a_descending_prior():
    frames = [frame((40, 45)) for _ in range(18)]
    recovered, summary = recover_rebound_tracks([approach()], frames, [i / 30 for i in range(18)], (100, 100))
    assert recovered == [] and summary['raw_components'] == 0


@pytest.mark.parametrize('budget', [True, -1, 65, 3.5])
def test_recovery_budget_is_bounded(budget):
    with pytest.raises(ValueError, match='max_approaches'):
        recover_rebound_tracks([], [], [], (100, 100), max_approaches=budget)


def test_zero_budget_does_not_decode_source():
    recovered, summary = recover_rebound_tracks([approach()], None, [i / 30 for i in range(18)],
                                               (100, 100), max_approaches=0)
    assert recovered == [] and summary['decoded_frames'] == 0


@pytest.mark.parametrize('times,size', [([0, float('nan')], (100, 100)),
    ([0, 0], (100, 100)), ([0, -.1], (100, 100)), ([0, .1], (0, 100)),
    ([0, .1], (float('inf'), 100)), ([0, .1], (True, 100))])
def test_invalid_time_or_coordinate_domains_fail_explicitly(times, size):
    with pytest.raises(ValueError):
        recover_rebound_tracks([], None, times, size)


def test_invalid_source_pixels_fail_explicitly():
    with pytest.raises(ValueError, match='uint8 BGR'):
        recover_rebound_tracks([approach()], [None] * 18, [i / 30 for i in range(18)], (100, 100))


def test_interpolated_or_invalid_track_points_cannot_seed_recovery():
    for field, value in [('interpolated', True), ('x', float('nan')), ('time', .7)]:
        track = approach()
        track['observations'][2][field] = value
        recovered, summary = recover_rebound_tracks([track], None, [i / 30 for i in range(18)], (100, 100))
        assert recovered == [] and summary['decoded_frames'] == 0


def test_crop_memory_budget_skips_windows_before_retaining_pixels(monkeypatch):
    monkeypatch.setattr('badminton.landing_recovery.MAX_CROP_BYTES', 1024)
    class Source:
        def __init__(self):
            self.reads = []
        def __len__(self):
            return 18
        def __getitem__(self, index):
            self.reads.append(index)
            return frame((40, 45))
    source = Source()
    recovered, summary = recover_rebound_tracks([approach()], source, [i / 30 for i in range(18)], (100, 100))
    assert recovered == []
    assert summary['budget_skipped_windows'] == 1
    assert summary['crop_bytes'] == 0
    assert summary['decoded_frames'] == 1
    assert source.reads == [0]


def test_recovered_hypotheses_remain_compatible_with_manual_identity_anchors():
    from badminton.events import estimate_landings
    from badminton.identity import apply_identity
    track = approach()
    ys = [20, 25, 30, 35, 40, 45, 42, 39, 36, 34, 32, 30, 28, 26, 24, 22]
    frames = [frame((40, y)) for y in ys]
    times = [i / 30 for i in range(len(frames))]
    recovered, _ = recover_rebound_tracks([track], frames, times, (100, 100))
    assert recovered
    tracks = [track, *recovered]
    events = estimate_landings(tracks, frames, times, {})
    result = apply_identity(events, tracks, times, {'image_width': 100, 'image_height': 100},
                            {'hit_anchors': [{'time': .1, 'point_px': [40, 35], 'radius_px': 5}]})
    assert result and all(candidate['contact_observed'] is False for candidate in result)
    assert all(t['recovery_parent_track_id'] == track['track_id'] for t in recovered)


def branch_fixture():
    before = [{'frame_index': i, 'time': i / 30, 'x': 20. + i * 4, 'y': 20. + i * 5,
               'radius': 3., 'confidence': .8, 'observed': True} for i in range(7)]
    post = [{'frame_index': i, 'time': i / 30, 'x': x, 'y': y, 'radius': radius,
             'confidence': .6, 'observed': True, 'native_resolution_observation': True,
             'method': 'native_bidirectional_temporal_component'}
            for i, x, y, radius in [(7, 46., 42., 1.2), (8, 46.2, 42.1, .9), (9, 46.1, 41.8, .8)]]
    continued = [{'frame_index': i, 'time': i / 30, 'x': 20. + i * 4, 'y': 20. + i * 5,
                  'radius': 3., 'confidence': .8, 'observed': True} for i in range(7, 10)]
    wrong = {'track_id': 10, 'flight_hypothesis': {}, 'observations': before + post}
    other = {'track_id': 11, 'flight_hypothesis': {}, 'observations': deepcopy(before) + continued}
    candidate = {'track_id': 10, 'time': .2, 'frame_index': 6, 'candidate_px': [44., 50.],
                 'contact_observed': False, 'identity_confidence': .2, 'reason_codes': [],
                 'image_motion_event': {'kind': 'descent_and_rebound'},
                 'flight_trajectory_support': {'supported_for_review': True}}
    return candidate, wrong, other


def test_native_guard_preserves_candidate_while_withholding_fragment_flight_priority():
    from badminton.landing_recovery import guard_native_rebound_branches
    candidate, wrong, other = branch_fixture()
    snapshot = deepcopy((candidate, wrong, other))
    result = guard_native_rebound_branches([candidate], [wrong, other], 1 / 30)[0]
    assert result['flight_trajectory_support']['supported_for_review'] is False
    assert result['candidate_px'] == candidate['candidate_px']
    assert result['contact_observed'] is False
    assert result['identity_confidence'] == candidate['identity_confidence']
    assert result['native_rebound_guard']['native_frame_indices'] == [7, 8, 9]
    assert result['native_rebound_guard']['candidate_removed'] is False
    assert (candidate, wrong, other) == snapshot


@pytest.mark.parametrize('condition', ['constant_size', 'global_observations', 'qualified_contact',
                                      'single_small_sample', 'interpolated_sample', 'missing_interval'])
def test_native_guard_requires_sustained_actual_native_size_collapse(condition):
    from badminton.landing_recovery import guard_native_rebound_branches
    candidate, wrong, other = branch_fixture()
    tracks = [wrong, other]
    if condition == 'constant_size':
        for row in wrong['observations'][7:]:
            row['radius'] = 3
    elif condition == 'global_observations':
        for row in wrong['observations'][7:]:
            row['method'] = 'opencv_motion_bright_component'
    elif condition == 'qualified_contact':
        candidate['contact_observed'] = True
    elif condition == 'single_small_sample':
        wrong['observations'][7]['radius'] = 3
        wrong['observations'][8]['radius'] = 3
    elif condition == 'interpolated_sample':
        wrong['observations'][8]['interpolated'] = True
    elif condition == 'missing_interval':
        for row in wrong['observations'][7:]:
            row['time'] += .2
    result = guard_native_rebound_branches([candidate], tracks, 1 / 30)[0]
    assert result == candidate


def test_same_frame_occluder_boxes_apply_detector_semantics_to_native_points():
    from badminton.landing_recovery import apply_recovery_occlusion_context
    point = {'frame_index': 7, 'time': .2, 'x': 40., 'y': 50., 'confidence': .8}
    record = {'frame_index': 7, 'time': .2, 'occlusion_regions': [{'bbox': [30, 40, 10, 10]}]}
    apply_recovery_occlusion_context([point], record)
    assert point['possible_occluder_overlap'] is True
    assert point['confidence'] == .68
    assert point['occlusion_context']['person_confirmed'] is False
    empty = {'frame_index': 7, 'time': .2, 'x': 40., 'y': 50., 'confidence': .8}
    apply_recovery_occlusion_context([empty], {**record, 'occlusion_regions': []})
    assert empty['possible_occluder_overlap'] is False


@pytest.mark.parametrize('record', [None, {}, {'frame_index': 7, 'time': .2},
    {'frame_index': 8, 'time': .2, 'occlusion_regions': []},
    {'frame_index': 7, 'time': .3, 'occlusion_regions': []},
    {'frame_index': 7, 'time': .2, 'occlusion_regions': [{'bbox': [0, 0, -1, 20]}]}])
def test_missing_or_stale_occluder_context_is_unknown_not_clear(record):
    from badminton.landing_recovery import apply_recovery_occlusion_context
    point = {'frame_index': 7, 'time': .2, 'x': 40., 'y': 50., 'confidence': .8}
    apply_recovery_occlusion_context([point], record)
    assert 'possible_occluder_overlap' not in point
    assert point['confidence'] == .8


def test_moving_fragment_or_possible_occluder_does_not_establish_appearance_continuity():
    from badminton.landing_recovery import guard_native_rebound_branches
    candidate, wrong, _ = branch_fixture()
    for row in wrong['observations'][:7]:
        row['possible_occluder_overlap'] = False
    for i, row in enumerate(wrong['observations'][7:]):
        row['possible_occluder_overlap'] = True
        row['y'] -= i * 3
    result = guard_native_rebound_branches([candidate], [wrong], 1 / 30)[0]
    assert result['flight_trajectory_support']['supported_for_review'] is False
    assert result['native_rebound_guard']['method'] == 'native_component_appearance_consistency'
    assert result['contact_observed'] is False
    assert result['candidate_px'] == candidate['candidate_px']
    # A moving, size-consistent object near a person is not rejected by this cue.
    for row in wrong['observations'][7:]:
        row['radius'] = 3
    assert guard_native_rebound_branches([candidate], [wrong], 1 / 30)[0] == candidate
