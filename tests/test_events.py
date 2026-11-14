import numpy as np
import pytest

from badminton.events import estimate_landings, apply_contact_evidence


def track(points, *, ambiguous=False, times=None):
    times = times or [i / 30 for i in range(len(points))]
    return {"track_id": 7, "identity_ambiguous": ambiguous, "gaps": [],
            "observations": [{"frame_index": i, "time": t, "x": x, "y": y,
                              "radius": 2, "confidence": 0.9}
                             for i, ((x, y), t) in enumerate(zip(points, times))]}


def test_slowdown_proposes_review_not_physical_contact():
    points = [(10 + i * 4, 10 + i * 3) for i in range(7)] + [(34, 28)] * 5
    frames = [np.zeros((100, 100, 3), np.uint8) for _ in points]
    times = [i / 30 for i in range(len(points))]
    events = estimate_landings([track(points)], frames, times,
                              {"status": "ok", "polygon_px": [[0, 0], [99, 0], [99, 99], [0, 99]]})
    assert events
    candidate = events[0]
    assert "MOTION_SLOWDOWN_AND_PERSISTENCE" in candidate["reason_codes"]
    assert candidate["contact_observed"] is False
    assert candidate["contact_px"] is None
    assert "FIRST_CONTACT_NOT_OBSERVED" in candidate["reason_codes"]
    assert candidate["target_court_score"] == 1
    assert candidate["identity_confidence"] <= 0.55


def test_stationary_background_is_not_a_landing():
    points = [(20, 20)] * 10
    assert estimate_landings([track(points)], [], [i / 30 for i in range(10)], {}) == []


def test_target_support_uses_selected_primary_court_not_competing_plane():
    from badminton.events import _target_support
    calibration = {"polygon_px": [[0, 0], [99, 0], [99, 99], [0, 99]],
                   "camera_stability": {"status": "ok", "all_frames_sampled": True},
                   "primary_court": {"status": "selected",
                                     "interior_polygon_px": [[20, 20], [60, 20], [60, 60], [20, 60]]}}
    assert _target_support([{"x": 40, "y": 40}], calibration) == 1
    assert _target_support([{"x": 80, "y": 40}], calibration) == 0
    calibration["primary_court"]["status"] = "review"
    assert _target_support([{"x": 40, "y": 40}], calibration) == 0


@pytest.mark.parametrize("stability", [None, {"status": "review"},
    {"status": "ok", "all_frames_sampled": False},
    {"status": "ok", "all_frames_sampled": True, "reason_codes": ["CAMERA_MOVED_FROM_CALIBRATION_REFERENCE"]}])
def test_unstable_primary_court_cannot_prioritize_a_stale_target_region(stability):
    from badminton.events import _target_support
    calibration = {"camera_stability": stability,
                   "primary_court": {"status": "selected",
                                     "interior_polygon_px": [[20, 20], [60, 20], [60, 60], [20, 60]]}}
    assert _target_support([{"x": 40, "y": 40}], calibration) == 0


def test_missing_observations_and_crossing_id_remain_explicit():
    points = [(10 + i * 4, 10 + i * 3) for i in range(7)] + [(34, 28)] * 5
    times = [i / 30 if i < 7 else i / 30 + 0.3 for i in range(len(points))]
    tr = track(points, ambiguous=True, times=times)
    tr["gaps"] = [{"duration": 0.3}]
    events = estimate_landings([tr], [np.zeros((80, 80, 3), np.uint8) for _ in points],
                              [i / 30 for i in range(30)], {})
    assert any("TRACK_IDENTITY_AMBIGUOUS" in e["reason_codes"] for e in events)
    assert all(e["contact_observed"] is False for e in events)
    assert all(e["identity_confidence"] <= 0.25 for e in events)


def test_selected_track_cannot_override_unobserved_contact():
    points = [(10 + i * 3, 20 + i * 2) for i in range(6)] + [(25, 30)] * 5
    events = estimate_landings([track(points)], [np.zeros((80, 80, 3), np.uint8) for _ in points],
                              [i / 30 for i in range(len(points))], {}, {"target_track_id": 7})
    assert events and events[0]["identity_source"] == "operator_selected_track"
    assert events[0]["contact_px"] is None


def test_track_termination_is_low_confidence_draft():
    points = [(10 + i * 4, 10 + i * 2) for i in range(8)]
    events = estimate_landings([track(points)], [], [i / 30 for i in range(8)], {})
    assert events[0]["confidence"] <= 0.2
    assert "TRACK_TERMINATED_WITHOUT_CONTACT" in events[0]["reason_codes"]


def test_first_image_bounce_is_preferred_to_later_slide_and_stop_without_claiming_contact():
    # A free-flying object descends, rebounds while still moving horizontally,
    # then makes another smaller bounce and finally slides to a stop. These are
    # image-coordinate software fixtures, not annotated badminton evidence.
    ys = [20, 35, 55, 80, 110, 150, 140, 132, 127, 130, 139, 146, 140, 137,
          140, 143, 144, 144, 144, 144, 144, 144]
    points = [(10 + i * 5, y) for i, y in enumerate(ys[:16])] + [(85, y) for y in ys[16:]]
    frames = [np.zeros((180, 180, 3), np.uint8) for _ in points]
    times = [i / 30 for i in range(len(points))]
    events = estimate_landings([track(points)], frames, times, {})
    first = events[0]
    assert first['frame_index'] == 5
    assert first['candidate_px'] == [35., 150.]
    assert first['image_motion_event']['first_in_track'] is True
    assert first['image_motion_event']['frame_interval'] == [4, 6]
    assert first['image_motion_event']['time_interval'] == [times[4], times[6]]
    assert first['image_motion_event']['inference_only'] is True
    assert first['contact_observed'] is False and first['contact_px'] is None
    assert 'FIRST_CONTACT_NOT_OBSERVED' in first['reason_codes']
    assert any(event['frame_index'] > 5 for event in events)


def test_image_bounce_does_not_depend_on_full_vector_direction_reversal():
    # Horizontal motion dominates the velocity dot product through this bounce.
    points = [(10 + i * 15, y) for i, y in enumerate([10, 14, 18, 22, 26, 22, 18, 17, 18])]
    frames = [np.zeros((80, 180, 3), np.uint8) for _ in points]
    times = [i / 30 for i in range(len(points))]
    events = estimate_landings([track(points)], frames, times, {})
    assert events[0]['frame_index'] == 4
    assert 'IMAGE_SPACE_DESCENT_AND_REBOUND' in events[0]['reason_codes']
    assert events[0]['contact_observed'] is False


def test_weak_image_rebound_uses_first_quantized_plateau_sample():
    ys = [10, 15, 20, 25, 30, 30, 28, 27, 26, 27]
    tr = track([(50 + i, y) for i, y in enumerate(ys)])
    # One frame after the plateau is missing; no observation is manufactured.
    for observation in tr['observations'][6:]:
        observation['frame_index'] += 1
        observation['time'] += 1 / 30
    frames = [np.zeros((80, 100, 3), np.uint8) for _ in range(len(ys) + 1)]
    events = estimate_landings([tr], frames, [i / 30 for i in range(len(frames))], {})
    first = events[0]
    assert first['frame_index'] == 4 and first['candidate_px'] == [54., 30.]
    assert first['image_motion_event']['plateau_frame_interval'] == [4, 5]
    assert first['image_motion_event']['frame_interval'] == [3, 7]
    assert first['contact_observed'] is False


def test_high_frame_rate_keeps_equivalent_flight_review_support():
    original_y = [20, 35, 55, 80, 110, 150, 140, 132, 127, 130, 139]
    times = np.arange((len(original_y) - 1) * 4 + 1) / 120
    ys = np.interp(times, np.arange(len(original_y)) / 30, original_y)
    moving = track([(20 + time * 60, y) for time, y in zip(times, ys)], times=list(times), ambiguous=True)
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    frames = [np.zeros((180, 180, 3), np.uint8) for _ in times]
    events = estimate_landings([moving], frames, list(times), {})
    assert events[0]['frame_index'] == 20
    assert events[0]['flight_trajectory_support']['supported_for_review'] is True
    assert events[0]['contact_observed'] is False


def test_coordinate_noise_and_missing_intervals_do_not_create_image_bounce():
    times = [i / 30 for i in range(9)]
    points = [(10 + i * 5, 20 + .3 * (i % 2)) for i in range(9)]
    events = estimate_landings([track(points)], [], times, {})
    assert not any('image_motion_event' in event for event in events)
    points = [(10 + i * 5, y) for i, y in enumerate([10, 14, 18, 22, 26, 22, 18, 17, 18])]
    gap_times = [t if i < 4 else t + .5 for i, t in enumerate(times)]
    events = estimate_landings([track(points, times=gap_times)], [], times, {})
    assert not any('image_motion_event' in event for event in events)


def test_supported_free_flight_ranks_before_high_score_body_reversal_without_confirming_identity():
    ys = [20, 35, 55, 80, 110, 150, 140, 132, 127, 130, 139]
    moving = track([(20 + i * 2, y) for i, y in enumerate(ys)], ambiguous=True)
    moving['track_id'] = 1
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search', 'verified_identity': False}
    for observation in moving['observations']:
        observation['confidence'] = .45
    body = track([(100 + i * 2, y) for i, y in enumerate([20, 35, 55, 80, 65, 55, 45, 40, 40, 40, 40])])
    body['track_id'] = 2
    frames = [np.zeros((180, 180, 3), np.uint8) for _ in ys]
    events = estimate_landings([body, moving], frames, [i / 30 for i in range(len(ys))], {})
    selected = events[0]
    assert selected['track_id'] == 1 and selected['frame_index'] == 5
    assert selected['flight_trajectory_support']['supported_for_review'] is True
    assert selected['flight_trajectory_support']['verified_identity'] is False
    assert selected['identity_confidence'] <= .25
    assert selected['contact_observed'] is False and selected['contact_px'] is None
    assert selected['confidence'] < next(event['confidence'] for event in events if event['track_id'] == 2)


def test_review_ranking_uses_measured_flight_extent_and_fit_before_raw_detector_confidence():
    from badminton.events import prioritize_review_candidates
    candidates = []
    for identifier, descent, residual, confidence in [('body', 35., 2., .95), ('flight', 80., .5, .4)]:
        candidates.append({'track_id': identifier, 'confidence': confidence, 'identity_confidence': .15,
            'contact_observed': False, 'contact_px': None,
            'flight_trajectory_support': {'supported_for_review': True,
                'approach_displacement_y_px': descent, 'fit_residual_px': residual}})
    selected = prioritize_review_candidates(candidates, [])[0]
    assert selected['track_id'] == 'flight'
    assert selected['confidence'] == .4 and selected['identity_confidence'] == .15
    assert selected['contact_observed'] is False and selected['contact_px'] is None


@pytest.mark.parametrize('identity_first', [False, True])
@pytest.mark.parametrize('source', ['human_review', 'qualified_learned_contact_localization'])
def test_qualified_observed_contact_precedes_inferred_flight_but_explicit_region_still_wins(identity_first, source):
    from badminton.events import prioritize_review_candidates
    observed = {'track_id': 'observed', 'frame_index': 3, 'candidate_px': [80., 80.],
        'contact_observed': True, 'contact_evidence_source': source,
        'confidence': .95, 'identity_confidence': .9}
    inferred = {'track_id': 'inferred', 'frame_index': 3, 'candidate_px': [10., 10.],
        'contact_observed': False, 'confidence': .2, 'identity_confidence': .15,
        'flight_trajectory_support': {'supported_for_review': True,
            'approach_displacement_y_px': 100., 'fit_residual_px': .1}}
    assert prioritize_review_candidates([inferred, observed], [], identity_first=identity_first)[0] is observed
    region = {'frame_start': 2, 'frame_end': 4, 'bbox': [5., 5., 15., 15.]}
    assert prioritize_review_candidates([inferred, observed], [region], identity_first=identity_first)[0] is inferred
    inferred['request_match'] = 'selected'
    assert prioritize_review_candidates([inferred, observed], [], identity_first=identity_first)[0] is inferred


def test_flight_tag_does_not_promote_a_jittering_mixed_track():
    ys = [20, 60, 25, 80, 90, 150, 140, 132, 127, 130, 139]
    mixed = track([(20 + i * 2, y) for i, y in enumerate(ys)], ambiguous=True)
    mixed['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    events = estimate_landings([mixed], [np.zeros((180, 180, 3), np.uint8) for _ in ys],
                               [i / 30 for i in range(len(ys))], {})
    assert not any(event.get('flight_trajectory_support', {}).get('supported_for_review') for event in events)


@pytest.mark.parametrize('tail', [[98, 98, 98, 98, 98], [115, 108]])
def test_single_jump_to_stationary_pixels_or_short_tail_does_not_support_flight_rebound(tail):
    # A well-fit descent can attach to an unrelated pixel or stop at the end
    # of the clip. Keep its draft, but neither supplies repeated rebound data.
    ys = [10, 15, 24, 37, 54, 75, 100, 129, *tail]
    moving = track([(30 + i, y) for i, y in enumerate(ys)], ambiguous=True)
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    frames = [np.zeros((180, 180, 3), np.uint8)] * len(ys)
    events = estimate_landings([moving], frames, [i / 30 for i in range(len(ys))], {})
    rebound = next(event for event in events if event.get('image_motion_event'))
    support = rebound['flight_trajectory_support']
    assert support['supported_for_review'] is False
    assert 'MEASURED_REBOUND_SUPPORT_INSUFFICIENT' in rebound['reason_codes']
    assert rebound['contact_observed'] is False and rebound['contact_px'] is None
    assert moving['observations'][-1]['y'] == tail[-1]


def test_rebound_branch_jump_and_small_later_jump_fail_measured_post_motion_fit():
    ys = [10, 15, 24, 37, 54, 75, 100, 129, 162, 127, 127.1, 127.2, 122]
    moving = track([(30 + i, y) for i, y in enumerate(ys)],
                   times=[i / 60 for i in range(len(ys))], ambiguous=True)
    # Supply a long enough smooth observed descent for approach support at 60fps.
    for i, observation in enumerate(moving['observations']):
        observation['time'] = (i if i < 9 else i + 1) / 60
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    frames = [np.zeros((180, 180, 3), np.uint8)] * (len(ys) + 1)
    events = estimate_landings([moving], frames, [i / 60 for i in range(len(frames))], {})
    rebound = next(event for event in events if event.get('image_motion_event'))
    support = rebound['flight_trajectory_support']
    assert support['upward_interval_count'] >= 2
    assert support['post_motion_fit_residual_px'] > 3.
    assert support['supported_for_review'] is False
    assert rebound['contact_observed'] is False


def test_missing_outgoing_interval_cannot_complete_measured_rebound_support():
    ys = [10, 15, 24, 37, 54, 75, 100, 129, 162, 157.5, 157.48, 150]
    moving = track([(30 + i, y) for i, y in enumerate(ys)], times=[i / 60 for i in range(len(ys))])
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    # Two nearby observations, then an unrelated point after a long missing
    # interval. That last point cannot turn a short tail into a measured bounce.
    moving['observations'][-1]['time'] += 5 / 60
    moving['observations'][-1]['frame_index'] += 5
    times = [i / 60 for i in range(len(ys) + 5)]
    frames = [np.zeros((200, 180, 3), np.uint8)] * len(times)
    events = estimate_landings([moving], frames, times, {})
    candidate = next(event for event in events if event.get('image_motion_event'))
    assert candidate['flight_trajectory_support']['post_observation_count'] == 2
    assert candidate['flight_trajectory_support']['supported_for_review'] is False
    assert candidate['contact_observed'] is False


def test_long_missing_incoming_span_cannot_supply_normal_flight_support():
    indices = [0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22]
    points = [(30 + i, 10 + .3 * i * i) for i in indices[:8]] + [(50, 110), (51, 104), (52, 100)]
    moving = track(points, times=[i / 30 for i in indices], ambiguous=True)
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    for observation, frame in zip(moving['observations'], indices):
        observation['frame_index'] = frame
    frames = [np.zeros((180, 180, 3), np.uint8)] * 23
    events = estimate_landings([moving], frames, [i / 30 for i in range(23)], {})
    candidate = next(event for event in events if event['frame_index'] == 19)
    assert candidate['image_motion_event']['kind'] == 'descent_and_rebound'
    assert not candidate.get('flight_trajectory_support', {}).get('supported_for_review')
    assert candidate['contact_observed'] is False


def test_eligible_occluded_descent_retains_measured_history_only_for_review_ranking():
    indices = [0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 13, 15, 16, 17, 18]
    points = [(30 + i * 2, 10 + .8 * i * i) for i in indices[:11]] + [(60, y) for y in [133, 125, 120, 119]]
    moving = track(points, times=[i / 30 for i in indices], ambiguous=True)
    moving['track_id'] = 'moving'
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    for observation, frame in zip(moving['observations'], indices):
        observation.update(frame_index=frame, confidence=.4)
    ys = [10, 15, 24, 37, 54, 75, 100, 129, 118, 111, 107, 109]
    other = track([(150 - i * 2, y) for i, y in enumerate(ys)], ambiguous=True)
    other['track_id'] = 'other'
    other['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    frames = [np.zeros((200, 200, 3), np.uint8)] * 19
    events = estimate_landings([other, moving], frames, [i / 30 for i in range(19)], {})
    event = events[0]
    assert event['track_id'] == 'moving' and event['frame_index'] == 13
    support = event['flight_trajectory_support']
    assert support['approach_frame_interval'] == [8, 13]
    assert support['approach_observation_count'] == 5
    history = support['ranking_history']
    assert history['includes_gaps'] is True and history['maximum_gap_seconds'] == pytest.approx(.1)
    assert history['frame_interval'] == [3, 13]
    assert history['observation_count'] == 8 and len(history['observations']) == 8
    assert history['missing_association_resolved'] is False and history['identity_verified'] is False
    assert history['changes_local_eligibility'] is False
    assert event['contact_observed'] is False


def test_ineligible_event_history_cannot_override_local_support_requirement():
    from badminton.events import prioritize_review_candidates
    supported = {'track_id': 'local', 'flight_trajectory_support': {'supported_for_review': True,
        'approach_displacement_y_px': 20., 'fit_residual_px': 1.}}
    unsupported = {'track_id': 'gapped', 'flight_trajectory_support': {'supported_for_review': False,
        'ranking_history': {'qualified_for_ranking': True, 'displacement_y_px': 500., 'fit_residual_px': 0.}}}
    assert prioritize_review_candidates([unsupported, supported], [])[0] is supported


@pytest.mark.parametrize('change',['size_collapse','appearance_break'])
def test_post_gap_fragments_cannot_veto_supported_local_rebound(change):
    ys=[10,15,24,37,54,75,100,129,162,150,142,138,137,136,135]
    moving=track([(30+i*5,y) for i,y in enumerate(ys)],ambiguous=True)
    moving['flight_hypothesis']={'method':'coherent_free_flight_search'}
    for i,observation in enumerate(moving['observations']):
        frame=i+(15 if i>=12 else 0)
        observation.update(frame_index=frame,time=frame/30,
            radius=3. if i<5 else 1.8 if 9<=i<=11 else .6,
            appearance_polarity='bright',possible_occluder_overlap=i>=5)
        if change=='appearance_break':
            observation.update(radius=3. if i<=8 else 2.5 if i<=11 else .6,
                appearance_polarity='bright' if i<=8 else 'unknown' if i<=11 else 'dark',
                possible_occluder_overlap=i>=8)
    frames=[np.zeros((200,180,3),np.uint8)]*30
    events=estimate_landings([moving],frames,[i/30 for i in range(30)],{})
    event=next(event for event in events if event['frame_index']==8)
    assert event['flight_trajectory_support']['supported_for_review'] is True
    assert event['flight_trajectory_support']['post_observation_count']==3
    assert not event['identity_continuity']['flight_support_veto']
    assert event['identity_continuity']['evidence'][0]['post_distinct_observation_count']==3
    assert event['contact_observed'] is False


def test_combined_appearance_break_withholds_flight_priority_but_preserves_candidate():
    ys = [10, 15, 24, 37, 54, 75, 100, 129, 162, 150, 142, 138, 140]
    moving = track([(30 + i, y) for i, y in enumerate(ys)], ambiguous=True)
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    for i, observation in enumerate(moving['observations']):
        observation.update(appearance_polarity='bright' if i < 6 else 'dark',
                           possible_occluder_overlap=i >= 6,
                           radius=2. if i < 6 else 5.)
    frames = [np.zeros((200, 180, 3), np.uint8)] * len(ys)
    events = estimate_landings([moving], frames, [i / 30 for i in range(len(ys))], {})
    rebound = next(event for event in events if event.get('image_motion_event'))
    assert rebound['identity_continuity']['flight_support_veto'] is True
    assert rebound['flight_trajectory_support']['supported_for_review'] is False
    assert 'OCCLUDER_ENTRY_AND_APPEARANCE_DISCONTINUITY' in rebound['reason_codes']
    assert rebound['identity_confidence'] <= .25
    assert rebound['contact_observed'] is False and rebound['contact_px'] is None


@pytest.mark.parametrize('visibility,expected', [('overlap', False), ('crossing', True), ('emerging', True), ('unknown', True)])
def test_wholly_body_overlapping_approach_cannot_supply_free_flight_priority(visibility, expected):
    ys = [10, 15, 24, 37, 54, 75, 100, 129, 118, 111, 107, 109]
    moving = track([(30 + i, y) for i, y in enumerate(ys)], ambiguous=True)
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    for observation in moving['observations']:
        if visibility != 'unknown':
            observation['possible_occluder_overlap'] = True
    if visibility == 'crossing':
        # A real flight can pass over a player after a clearly observed approach.
        moving['observations'][0]['possible_occluder_overlap'] = False
    if visibility == 'emerging':
        for observation in moving['observations'][8:]:
            observation['possible_occluder_overlap'] = False
    frames = [np.zeros((180, 180, 3), np.uint8)] * len(ys)
    event = estimate_landings([moving], frames, [i / 30 for i in range(len(ys))], {})[0]
    assert event['flight_trajectory_support']['supported_for_review'] is expected
    assert ('NO_CLEAR_TRAJECTORY_OUTSIDE_POSSIBLE_OCCLUDER' in event['reason_codes']) is (not expected)
    assert event['contact_observed'] is False


def test_later_upward_jitter_cannot_rescue_initial_downward_shoe_motion():
    ys = [10, 15, 24, 37, 54, 75, 100, 129, 162,
          160.5, 161.456, 162.237, 158.278, 160, 161.119, 160.974, 162.971]
    times = [i / 60 for i in range(len(ys))]
    moving = track([(30 + i, y) for i, y in enumerate(ys)], times=times, ambiguous=True)
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    frames = [np.zeros((200, 180, 3), np.uint8)] * len(ys)
    events = estimate_landings([moving], frames, times, {})
    rebound = next(event for event in events if event.get('image_motion_event'))
    support = rebound['flight_trajectory_support']
    assert support['upward_interval_count'] >= 2
    assert support['rebound_displacement_y_px'] > 1.5
    assert support['initial_rebound_displacement_y_px'] < 0
    assert support['supported_for_review'] is False


@pytest.mark.parametrize('tail_kind', ['unobserved', 'interpolated', 'persistent_background', 'stale_background'])
def test_unobserved_interpolated_tail_cannot_supply_rebound_support(tail_kind):
    ys = [10, 15, 24, 37, 54, 75, 100, 129, 115, 108, 103, 100]
    moving = track([(30 + i, y) for i, y in enumerate(ys)], ambiguous=True)
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    for observation in moving['observations'][-2:]:
        if tail_kind == 'unobserved':
            observation['observed'] = False
        elif tail_kind == 'interpolated':
            observation.update(observed=True, interpolated=True)
        else:
            observation['observation_support'] = {
                'schema_version': 1, 'status': 'persistent_background', 'evidence_valid': True,
                'reject_for_moving_association': True,
                'frame_index': observation['frame_index'],
                'point_px': [observation['x'] + (1 if tail_kind == 'stale_background' else 0), observation['y']],
                'coordinate_size': [180, 180]}
    frames = [np.zeros((180, 180, 3), np.uint8)] * len(ys)
    events = estimate_landings([moving], frames, [i / 30 for i in range(len(ys))], {})
    assert events
    if tail_kind == 'stale_background':
        assert any(event.get('flight_trajectory_support', {}).get('supported_for_review') for event in events)
        return
    assert not any(event.get('flight_trajectory_support', {}).get('supported_for_review') for event in events)
    assert all(10 not in event['evidence_frame_indices'] and 11 not in event['evidence_frame_indices'] for event in events)


@pytest.mark.parametrize('fps', [24, 30])
def test_measured_weak_rebound_keeps_support_with_one_missing_post_turn_frame(fps):
    ys = [10, 15, 20, 25, 30, 30, 28, 27, 26, 27]
    moving = track([(30 + i, y) for i, y in enumerate(ys)], times=[i / fps for i in range(len(ys))])
    moving['flight_hypothesis'] = {'method': 'coherent_free_flight_search'}
    for observation in moving['observations'][6:]:
        observation['frame_index'] += 1
        observation['time'] += 1 / fps
    frames = [np.zeros((80, 100, 3), np.uint8)] * (len(ys) + 1)
    event = estimate_landings([moving], frames, [i / fps for i in range(len(frames))], {})[0]
    assert event['frame_index'] == 4
    assert event['flight_trajectory_support']['supported_for_review'] is True
    assert event['flight_trajectory_support']['upward_interval_count'] >= 2
    assert event['contact_observed'] is False


def test_occluded_flight_intersection_reports_inferred_point_without_filling_observations():
    times = [i * .04 for i in range(15)]
    frames = [np.zeros((200, 200, 3), np.uint8) for _ in times]
    visible = [0, 1, 2, 3, 4, 10, 11, 12, 13]
    observations = []
    for frame_index in visible:
        time = times[frame_index]
        y = 10 + 500 * time if frame_index < 5 else 150 - 100 * (time - .28) + 200 * (time - .28) ** 2
        observations.append({'frame_index': frame_index, 'time': time, 'x': 40 + 5 * time,
                             'y': y, 'radius': 1., 'confidence': .7})
    tr = {'track_id': 12, 'identity_ambiguous': True, 'observations': observations,
          'flight_hypothesis': {'method': 'coherent_free_flight_search'}}
    events = estimate_landings([tr], frames, times, {})
    inferred = next(event for event in events if event.get('image_motion_event', {}).get('kind') == 'occluded_descent_and_rebound')
    assert inferred['frame_index'] == 7
    assert inferred['candidate_px'] == pytest.approx([41.4, 150.])
    assert inferred['image_motion_event']['frame_interval'] == [4, 10]
    assert inferred['image_motion_event']['time_interval'] == [times[4], times[10]]
    assert inferred['image_motion_event']['point_kind'] == 'inferred_object_center'
    assert inferred['first_contact_missing'] is True
    assert inferred['contact_observed'] is False and inferred['contact_px'] is None
    assert inferred['flight_trajectory_support']['verified_identity'] is False
    assert [item['frame_index'] for item in tr['observations']] == visible

    # A missing turn plus exactly three observed post-gap samples still
    # supplies two measured upward intervals. A quadratic through only three
    # points cannot independently validate its fit, and is labeled accordingly.
    tr['observations'] = observations[:-1]
    sparse = estimate_landings([tr], frames, times, {})
    sparse_inferred = next(event for event in sparse if event.get('image_motion_event', {}).get('kind') == 'occluded_descent_and_rebound')
    assert sparse_inferred['flight_trajectory_support']['supported_for_review'] is True
    assert sparse_inferred['flight_trajectory_support']['post_motion_fit_validated'] is False
    assert sparse_inferred['flight_trajectory_support']['post_motion_fit_residual_px'] is None
    tr['observations'] = observations

    # The same y-turn does not link two spatially unrelated objects across a gap.
    for observation in observations[5:]:
        observation['x'] += 30
    events = estimate_landings([tr], frames, times, {})
    assert not any(event.get('image_motion_event', {}).get('kind') == 'occluded_descent_and_rebound' for event in events)


def test_occluded_rebound_fits_equivalent_physical_windows_at_30_and_60_fps():
    inferred_events = []
    for fps in [30, 60]:
        times = [i / fps for i in range(int(.6 * fps) + 1)]
        observations = []
        for frame_index, time in enumerate(times):
            if time > 1 / 6 + 1e-8 and time < .4 - 1e-8:
                continue
            if time > .533334:
                continue
            y = 10 + 500 * time if time < .2 else 150 - 100 * (time - .28) + 200 * (time - .28) ** 2
            observations.append({'frame_index': frame_index, 'time': time,
                                 'x': 40 + 5 * time, 'y': y, 'radius': 1., 'confidence': .7,
                                 'possible_occluder_overlap': time < .2})
        moving = {'track_id': 1, 'identity_ambiguous': True, 'observations': observations,
                  'flight_hypothesis': {'method': 'coherent_free_flight_search'}}
        frames = [np.zeros((200, 200, 3), np.uint8)] * len(times)
        events = estimate_landings([moving], frames, times, {})
        inferred_events.append(next(event for event in events if event.get('image_motion_event', {}).get('kind') == 'occluded_descent_and_rebound'))
    for event in inferred_events:
        assert event['image_motion_event']['estimated_time'] == pytest.approx(.28)
        assert event['candidate_px'] == pytest.approx([41.4, 150.])
        assert event['flight_trajectory_support']['supported_for_review'] is True
        assert event['flight_trajectory_support']['approach_wholly_overlaps_possible_occluder'] is True
        assert event['flight_trajectory_support']['support_wholly_overlaps_possible_occluder'] is False
        assert event['contact_observed'] is False
    assert inferred_events[0]['flight_trajectory_support']['approach_displacement_y_px'] == pytest.approx(
        inferred_events[1]['flight_trajectory_support']['approach_displacement_y_px'])


def contact_evidence():
    return {"video_id": "clip-a", "source": "human_review", "verified": True,
            "coordinate_system": "analysis_pixels", "time": 1.2, "contact_px": [45, 60],
            "contact_radius_px": 2, "uncertainty_px": 3,
            "first_contact_observed": True, "target_identity_confirmed": True}


def test_human_contact_is_separate_from_automatic_predictions():
    original = {"time": 1.3, "contact_observed": False, "contact_px": None,
                "reason_codes": ["FIRST_CONTACT_NOT_OBSERVED", "CALIBRATION_NOT_CONFIRMED"]}
    result = apply_contact_evidence(original, contact_evidence(), video_id="clip-a", duration_seconds=3,
                                    image_size=(100, 80))
    assert result["contact_observed"] is True
    assert result["inference_origin"] == "human_review"
    assert result["evaluation_eligible"] is False
    assert result["contact_px"] == [45, 60]
    assert "CALIBRATION_NOT_CONFIRMED" in result["reason_codes"]
    assert original["contact_observed"] is False


def test_reviewed_time_uses_trusted_frame_timestamps():
    candidate = {"time": 1.5, "frame_index": 45}
    result = apply_contact_evidence(candidate, contact_evidence(), video_id="clip-a", duration_seconds=3,
                                    image_size=(100, 80), timestamps=[i / 30 for i in range(90)])
    assert result["frame_index"] == 36
    assert result["candidate_frame_index"] == 45
    unknown = apply_contact_evidence(candidate, contact_evidence(), video_id="clip-a", duration_seconds=3,
                                     image_size=(100, 80))
    assert unknown["frame_index"] is None


def model_events(monkeypatch, response, preprocessing="opencv_source_bgr_area_v1"):
    from badminton import contact_model
    def infer(*args, **kwargs):
        assert kwargs.get("preprocessing") == preprocessing
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(contact_model, "infer_contact", infer)
    points = [(10 + i * 4, 10 + i * 3) for i in range(7)] + [(34, 28)] * 5
    frames = [np.zeros((100, 100, 3), np.uint8) for _ in points]
    return estimate_landings([track(points)], frames, [i / 30 for i in range(len(points))],
                             {"status": "ok", "polygon_px": [[0, 0], [99, 0], [99, 99], [0, 99]]},
                             {"contact_model_path": "mock-checkpoint.pt", "frame_preprocessing": preprocessing})


def qualified_contact_response():
    return {"contact_observed": True, "contact_px": [35, 30], "proposed_contact_px": [35, 30],
            "contact_radius_px": 0, "uncertainty_px": 2.5, "confidence": 0.995,
            "contact_evidence_source": "qualified_learned_contact_localization",
            "model_origin": "learned_temporal_contact", "reason_codes": [],
            "provenance": {"automatic_contact_qualified": True,
                           "preprocessing": "opencv_source_bgr_area_v1",
                           "runtime_preprocessing": "opencv_source_bgr_area_v1"}}


def test_qualified_contact_model_keeps_target_identity_separate(monkeypatch):
    events = model_events(monkeypatch, qualified_contact_response())
    result = events[0]
    assert result["contact_observed"] is True
    assert result["contact_px"] == [35, 30]
    assert result["candidate_px"] != result["contact_px"]
    assert result["identity_confidence"] <= 0.55
    assert "TARGET_IDENTITY_UNVERIFIED" in result["reason_codes"]
    assert result["inference_origin"] == "learned_contact_model"


@pytest.mark.parametrize("changes", [
    {"provenance": {"automatic_contact_qualified": False}},
    {"provenance": {"automatic_contact_qualified": True,
                    "preprocessing": "opencv_source_bgr_area_v1", "runtime_preprocessing": "h264_proxy"}},
    {"contact_px": [200, 30]}, {"uncertainty_px": -1},
    {"reason_codes": ["CONTACT_MODEL_CAPTURE_DOMAIN_MISMATCH"]},
])
def test_unqualified_or_invalid_model_cannot_assert_contact(monkeypatch, changes):
    result = model_events(monkeypatch, {**qualified_contact_response(), **changes})[0]
    assert result["contact_observed"] is False
    assert result["contact_px"] is None
    assert "FIRST_CONTACT_NOT_OBSERVED" in result["reason_codes"]


def test_model_error_is_visible_and_fails_closed(monkeypatch):
    result = model_events(monkeypatch, ValueError("invalid checkpoint"))[0]
    assert result["contact_model"]["status"] == "failed"
    assert "CONTACT_MODEL_ERROR" in result["reason_codes"]
    assert result["contact_observed"] is False


@pytest.mark.parametrize("preprocessing", [None, "h264_proxy_bgr_resize", "unknown"])
def test_contact_model_requires_known_matching_frame_preprocessing(monkeypatch, preprocessing):
    result = model_events(monkeypatch, qualified_contact_response(), preprocessing=preprocessing)[0]
    assert result["contact_observed"] is False
    assert result["contact_px"] is None
    assert result["model_proposed_contact_px"] == [35, 30]
    assert "CONTACT_MODEL_PREPROCESSING_MISMATCH" in result["reason_codes"]


@pytest.mark.parametrize("changes", [
    {"verified": False}, {"first_contact_observed": False}, {"target_identity_confirmed": False},
    {"video_id": "wrong-video"}, {"coordinate_system": "source_pixels"}, {"time": 5},
    {"contact_px": [-2, 10]}, {"contact_px": [100, 40]}, {"uncertainty_px": -1},
    {"contact_px": [float("nan"), 5]},
])
def test_unverified_or_out_of_bounds_contact_is_rejected(changes):
    evidence = {**contact_evidence(), **changes}
    with pytest.raises(ValueError):
        apply_contact_evidence({}, evidence, video_id="clip-a", duration_seconds=3, image_size=(100, 80))
