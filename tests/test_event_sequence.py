from copy import deepcopy

import pytest
import numpy as np

from badminton.event_sequence import (diversify_event_families,
    annotate_causal_rebounds, order_causal_rebounds)


def duplicate_events():
    observations = [{'frame_index': frame, 'time': frame / 30, 'x': float(frame),
                     'y': float(20 - abs(frame - 8)), 'radius': 2., 'observed': True}
                    for frame in range(3, 14)]
    tracks = [{'track_id': identifier, 'observations': deepcopy(observations)} for identifier in [1, 2, 3]]
    candidates = [{'track_id': identifier, 'frame_index': 8, 'time': 8 / 30,
                   'candidate_px': [8., 20.], 'contact_radius_px': 2.,
                   'contact_observed': False, 'contact_px': None, 'confidence': .5,
                   'identity_confidence': .25,
                   'image_motion_event': {'kind': 'descent_and_rebound',
                                          'point_kind': 'observed_object_center'}}
                  for identifier in [1, 2, 3]]
    return candidates, tracks


def test_duplicate_event_uses_one_budget_slot_and_preserves_all_evidence():
    candidates, tracks = duplicate_events()
    original = deepcopy((candidates, tracks))
    result = diversify_event_families(candidates, tracks, 1 / 30)
    assert len(result) == 1
    family = result[0].pop('event_family')
    assert result[0] == candidates[0]
    assert family['member_count'] == 3
    assert [alternative['candidate'] for alternative in family['alternatives']] == candidates[1:]
    assert family['physical_contact_verified'] is False
    assert family['identity_verified'] is False
    assert (candidates, tracks) == original


@pytest.mark.parametrize('condition', ['separate_path', 'no_incoming', 'no_outgoing', 'later_bounce',
    'different_location', 'different_evidence_type', 'observed_contact', 'selected_identity',
    'selected_request', 'ambiguous_request', 'assisted'])
def test_coincidence_alone_cannot_merge_distinct_or_protected_events(condition):
    candidates, tracks = duplicate_events()
    candidates, tracks = candidates[:2], tracks[:2]
    if condition == 'separate_path':
        for row in tracks[1]['observations']:
            row['x'] += .1
    elif condition in {'no_incoming', 'no_outgoing'}:
        for row in tracks[1]['observations']:
            if (row['frame_index'] < 8) == (condition == 'no_incoming'):
                row['observed'] = False
    elif condition == 'later_bounce':
        candidates[1].update(frame_index=12, time=.4)
    elif condition == 'different_location':
        candidates[1]['candidate_px'][0] += 3
    elif condition == 'different_evidence_type':
        candidates[1]['image_motion_event'].update(kind='occluded_descent_and_rebound', point_kind='inferred_object_center')
    elif condition == 'observed_contact':
        candidates[1]['contact_observed'] = True
    elif condition == 'selected_identity':
        candidates[1]['identity_source'] = 'operator_selected_track'
    elif condition == 'selected_request':
        candidates[1]['request_match'] = 'selected'
    elif condition == 'ambiguous_request':
        candidates[1]['request_match'] = 'ambiguous'
    elif condition == 'assisted':
        candidates[1]['assistance_used'] = True
    assert diversify_event_families(candidates, tracks, 1 / 30) == candidates


def test_spatial_neighbor_chain_cannot_merge_distant_endpoints():
    candidates, tracks = duplicate_events()
    candidates[1]['candidate_px'][0] += 1.5
    candidates[2]['candidate_px'][0] += 3.
    result = diversify_event_families(candidates, tracks, 1 / 30)
    assert len(result) == 2
    assert result[0]['event_family']['member_count'] == 2
    assert result[1] == candidates[2]


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), 0, -1, True, 10**400])
def test_invalid_cadence_keeps_candidates(invalid):
    candidates, tracks = duplicate_events()
    assert diversify_event_families(candidates, tracks, invalid) == candidates


@pytest.mark.parametrize('condition', ['interpolated', 'static_background'])
def test_non_motion_rows_cannot_supply_shared_support(condition):
    candidates, tracks = duplicate_events()
    for track in tracks:
        for row in track['observations']:
            if condition == 'interpolated':
                row['interpolated'] = True
            else:
                row['observation_support'] = {
                    'schema_version': 1, 'status': 'persistent_background',
                    'evidence_valid': True, 'reject_for_moving_association': True,
                    'frame_index': row['frame_index'], 'point_px': [row['x'], row['y']],
                    'coordinate_size': [128, 128]}
    assert diversify_event_families(candidates, tracks, 1 / 30) == candidates


def rebound_sequence(fps=30):
    def sample(frame, y):
        return {'frame_index': frame, 'time': frame / fps, 'x': frame * .5,
                'y': float(y), 'radius': 2., 'observed': True}
    arc = [sample(frame, 100 - (frame - 10) * (30 - frame) * .3) for frame in range(10, 31)]
    incoming = [sample(frame, 20 + frame * 8) for frame in range(1, 10)]
    outgoing = [sample(frame, 100 - (frame - 30) * 2) for frame in range(31, 36)]
    tracks = [{'track_id': 1, 'observations': incoming + deepcopy(arc)},
              {'track_id': 2, 'observations': deepcopy(arc) + outgoing}]
    candidates = [{'track_id': identifier, 'frame_index': frame, 'time': frame / fps,
                   'candidate_px': [frame * .5, 100.], 'contact_observed': False,
                   'contact_px': None, 'contact_radius_px': 2., 'identity_confidence': .2,
                   'flight_trajectory_support': {'supported_for_review': True,
                       'approach_displacement_y_px': 80.},
                   'image_motion_event': {'kind': 'descent_and_rebound',
                       'point_kind': 'observed_object_center'}}
                  for identifier, frame in [(1, 10), (2, 30)]]
    return candidates, tracks


def test_shared_measured_diminishing_arc_orders_earlier_turn_without_changing_geometry():
    candidates, tracks = rebound_sequence()
    original = deepcopy((candidates, tracks))
    annotated = annotate_causal_rebounds(candidates, tracks, 1 / 30)
    sequence = annotated[1]['causal_event_sequence']
    assert sequence['shared_observation_count'] == 19
    assert sequence['physical_contact_verified'] is False
    assert sequence['low_racket_hit_excluded'] is False
    unrelated = deepcopy(candidates[0])
    unrelated.update(track_id=3, time=.1)
    result = order_causal_rebounds([annotated[1], unrelated, annotated[0]])
    assert [candidate['track_id'] for candidate in result] == [1, 2, 3]
    for actual, expected in zip(annotated, candidates):
        assert actual['candidate_px'] == expected['candidate_px']
        assert actual['contact_observed'] is False
        assert actual['identity_confidence'] == expected['identity_confidence']
    assert (candidates, tracks) == original


@pytest.mark.parametrize('fps', [24, 60, 120])
def test_sequence_velocity_comparison_uses_actual_cadence(fps):
    candidates, tracks = rebound_sequence(fps)
    result = annotate_causal_rebounds(candidates, tracks, 1 / fps)
    assert result[1]['causal_event_sequence']['outgoing_image_speed_ratio'] == pytest.approx(1 / 3, abs=.0001)
    assert order_causal_rebounds(result[::-1])[0]['track_id'] == 1


@pytest.mark.parametrize('condition', ['different_arc', 'conflicting_arc', 'missing_arc',
    'energetic_late_hit', 'energetic_early_hit', 'different_height', 'ineligible_early',
    'observed_contact', 'interpolated_arc', 'tiny_arc', 'truncated_energetic_late_hit'])
def test_earlier_time_alone_cannot_override_ranking(condition):
    candidates, tracks = rebound_sequence()
    if condition == 'different_arc':
        for row in tracks[1]['observations']:
            row['x'] += 1
    elif condition == 'conflicting_arc':
        tracks[1]['observations'][8]['x'] += 1
    elif condition == 'missing_arc':
        for track in tracks:
            track['observations'] = [row for row in track['observations'] if not 18 <= row['frame_index'] <= 23]
    elif condition == 'energetic_late_hit':
        for row in tracks[1]['observations']:
            if row['frame_index'] > 30:
                row['y'] = 100 - (row['frame_index'] - 30) * 10
    elif condition == 'truncated_energetic_late_hit':
        tracks[1]['observations'] = [row for row in tracks[1]['observations'] if row['frame_index'] <= 33]
        for row in tracks[1]['observations']:
            if row['frame_index'] > 30:
                row['y'] = 100 - (row['frame_index'] - 30) * 7
    elif condition == 'energetic_early_hit':
        candidates[0]['flight_trajectory_support']['approach_displacement_y_px'] = 10
    elif condition == 'different_height':
        candidates[0]['candidate_px'][1] -= 15
    elif condition == 'ineligible_early':
        candidates[0]['flight_trajectory_support']['supported_for_review'] = False
    elif condition == 'observed_contact':
        candidates[0]['contact_observed'] = True
    elif condition == 'interpolated_arc':
        for track in tracks:
            for row in track['observations']:
                if 10 < row['frame_index'] < 30:
                    row['interpolated'] = True
    elif condition == 'tiny_arc':
        for track in tracks:
            for row in track['observations']:
                if 10 < row['frame_index'] < 30:
                    row['y'] = 100 - (100 - row['y']) * .1
    assert annotate_causal_rebounds(candidates, tracks, 1 / 30) == candidates


def test_predecessor_is_not_synthesized_when_missing_from_budget():
    candidates, tracks = rebound_sequence()
    annotated = annotate_causal_rebounds(candidates, tracks, 1 / 30)
    assert order_causal_rebounds(annotated[1:]) == annotated[1:]


def test_later_verified_contact_remains_above_heuristic_predecessor():
    candidates, tracks = rebound_sequence()
    annotated = annotate_causal_rebounds(candidates, tracks, 1 / 30)
    annotated[1]['contact_observed'] = True
    assert order_causal_rebounds(annotated[::-1]) == annotated[::-1]


def test_short_flight_fragments_can_join_through_repeated_measured_bridge():
    candidates, tracks = rebound_sequence()
    bridge = {'track_id': 3, 'observations': [deepcopy(row) for row in tracks[0]['observations']
                                            if 11 <= row['frame_index'] <= 28]}
    tracks[0]['observations'] = [row for row in tracks[0]['observations'] if row['frame_index'] <= 15]
    tracks[1]['observations'] = [row for row in tracks[1]['observations'] if row['frame_index'] >= 24]
    assert annotate_causal_rebounds(candidates, tracks, 1 / 30) == candidates
    result = annotate_causal_rebounds(candidates, [*tracks, bridge], 1 / 30)
    assert result[1]['causal_event_sequence']['bridge_track_id'] == '3'
    assert order_causal_rebounds(result[::-1])[0]['track_id'] == 1
    # A predicted gap-filling bridge cannot replace actual image observations.
    for row in bridge['observations']:
        row['interpolated'] = True
    assert annotate_causal_rebounds(candidates, [*tracks, bridge], 1 / 30) == candidates


def test_causal_reference_resolves_merged_predecessor_on_repeated_ranking():
    candidates, tracks = rebound_sequence()
    annotated = annotate_causal_rebounds(candidates, tracks, 1 / 30)
    representative = deepcopy(candidates[0])
    representative['track_id'] = 4
    tracks.append({'track_id': 4, 'observations': deepcopy(tracks[0]['observations'])})
    grouped = diversify_event_families([representative, *annotated], tracks, 1 / 30)
    assert len(grouped) == 2
    assert grouped[0]['event_family']['alternatives'][0]['candidate']['track_id'] == 1
    reordered = order_causal_rebounds(grouped[::-1])
    assert [row['track_id'] for row in reordered] == [4, 2]
    assert order_causal_rebounds(reordered) == reordered


@pytest.mark.parametrize('model_key', [None, 'contact_model_path', 'source_contact_model_path'])
def test_unqualified_model_proposals_are_not_merged_before_their_patch_is_evaluated(monkeypatch, model_key):
    from badminton.events import estimate_landings
    calls = []
    def grouping(candidates, tracks, step):
        calls.append(len(candidates))
        return candidates
    monkeypatch.setattr('badminton.event_sequence.diversify_event_families', grouping)
    monkeypatch.setattr('badminton.events._apply_contact_model', lambda *args, **kwargs: None)
    ys = [10, 20, 30, 40, 50, 60, 70, 80, 70, 60, 50, 40]
    observations = [{'frame_index': i, 'time': i / 30, 'x': 50., 'y': float(y), 'radius': 2.,
                     'confidence': .5, 'observed': True} for i, y in enumerate(ys)]
    cfg = {'event_family_diversity': True}
    if model_key:
        cfg[model_key] = 'model.pt'
    frames = [np.zeros((100, 100, 3), np.uint8) for _ in ys]
    result = estimate_landings([{'track_id': 1, 'observations': observations}], frames,
                              [i / 30 for i in range(len(ys))], {}, cfg)
    assert result
    assert bool(calls) is (model_key is None)


def branched_later_family():
    candidates, tracks = rebound_sequence()
    earlier, good = candidates
    branch = deepcopy(good)
    branch['track_id'] = 3
    track = deepcopy(tracks[1])
    track['track_id'] = 3
    for row in track['observations']:
        if row['frame_index'] in {33, 34, 35}:
            row['y'] = {33: 93., 34: 88., 35: 80.}[row['frame_index']]
    tracks.append(track)
    annotated = annotate_causal_rebounds([branch, good, earlier], tracks, 1 / 30)
    assert 'causal_event_sequence' not in annotated[0]
    assert 'causal_event_sequence' in annotated[1]
    return annotated, tracks


def test_relation_from_merged_later_member_survives_repeated_ordering_and_budget_cap():
    candidates, tracks = branched_later_family()
    unrelated = [dict(deepcopy(candidates[0]), track_id=100 + i, frame_index=60 + i,
                      time=2 + i / 30, candidate_px=[80., 60.]) for i in range(13)]
    initially_ranked = order_causal_rebounds([candidates[0], *unrelated, *candidates[1:]])
    grouped = diversify_event_families(initially_ranked, tracks, 1 / 30)
    original = deepcopy(grouped)
    corrected = order_causal_rebounds(grouped)
    assert [row['track_id'] for row in corrected[:2]] == [1, 3]
    assert corrected[1]['causal_event_family_order']['relation_source_members'] == [{'track_id': 2, 'frame_index': 30}]
    assert corrected[1]['contact_observed'] is False
    assert order_causal_rebounds(corrected[:12]) == corrected[:12]
    assert grouped == original


def test_distinct_preceding_families_are_ambiguous_not_an_earliest_event_vote():
    candidates, tracks = branched_later_family()
    grouped = diversify_event_families(candidates, tracks, 1 / 30)
    later, earlier = grouped
    other = deepcopy(earlier)
    other.update(track_id=4, candidate_px=[30., 100.])
    later['causal_event_sequence'] = deepcopy(candidates[1]['causal_event_sequence'])
    later['causal_event_sequence']['preceding_candidate'] = ['4', other['frame_index'], other['candidate_px']]
    result = order_causal_rebounds([later, earlier, other])
    assert [row['track_id'] for row in result] == [3, 1, 4]
    assert result[0]['causal_event_family_order']['status'] == 'ambiguous_or_missing_preceding_family'
    assert result[0]['causal_event_family_order']['resolved_predecessor_count'] == 2


def test_multiple_relation_aliases_to_one_preceding_family_are_not_a_conflict():
    candidates, tracks = branched_later_family()
    grouped = diversify_event_families(candidates, tracks, 1 / 30)
    later, earlier = grouped
    earlier_alias = deepcopy(earlier)
    earlier_alias['track_id'] = 4
    earlier['event_family'] = {'alternatives': [{'candidate': earlier_alias}]}
    later['causal_event_sequence'] = deepcopy(candidates[1]['causal_event_sequence'])
    later['causal_event_sequence']['preceding_candidate'][0] = '4'
    result = order_causal_rebounds([later, earlier])
    assert [row['track_id'] for row in result] == [1, 3]
    assert result[1]['causal_event_family_order']['status'] == 'unique_preceding_family'


@pytest.mark.parametrize('protected', ['representative_contact', 'predecessor_contact',
    'representative_request', 'predecessor_request', 'member_contact'])
def test_family_member_relation_cannot_override_explicit_evidence(protected):
    candidates, tracks = branched_later_family()
    later, earlier = diversify_event_families(candidates, tracks, 1 / 30)
    target = (earlier if protected.startswith('predecessor') else
              later['event_family']['alternatives'][0]['candidate'] if protected.startswith('member') else later)
    if protected.endswith('contact'):
        target['contact_observed'] = True
    else:
        target['request_match'] = 'selected'
    assert [row['track_id'] for row in order_causal_rebounds([later, earlier])] == [3, 1]


def test_missing_alternative_predecessor_cannot_be_replaced_by_an_available_other_parent():
    candidates, tracks = branched_later_family()
    later, earlier = diversify_event_families(candidates, tracks, 1 / 30)
    later['causal_event_sequence'] = deepcopy(candidates[1]['causal_event_sequence'])
    later['causal_event_sequence']['preceding_candidate'][0] = 'missing'
    result = order_causal_rebounds([later, earlier])
    assert [row['track_id'] for row in result] == [3, 1]
    assert result[0]['causal_event_family_order']['status'] == 'ambiguous_or_missing_preceding_family'


def test_estimate_landings_resolves_member_edge_before_maximum_candidate_cap():
    from badminton.events import estimate_landings
    _, tracks = rebound_sequence()
    # A less well-fit first approach ranks below two branches of the same later
    # event. Only the second-ranked later branch retains a diminishing arc.
    for row in tracks[0]['observations']:
        if row['frame_index'] < 10:
            row['y'] = 100. - 5 * (10 - row['frame_index'])
            row['x'] += 1.5 * (-1) ** row['frame_index']
    for track in tracks:
        track['flight_hypothesis'] = {}
        for row in track['observations']:
            row['confidence'] = .5
    branch = deepcopy(tracks[1])
    branch['track_id'] = 3
    for row in branch['observations']:
        row['confidence'] = .9
        if row['frame_index'] in {33, 34, 35}:
            row['y'] = {33: 93., 34: 88., 35: 80.}[row['frame_index']]
    tracks.append(branch)
    frames = [np.zeros((120, 120, 3), np.uint8) for _ in range(36)]
    timestamps = [i / 30 for i in range(36)]
    original = estimate_landings(tracks, frames, timestamps, {}, {'max_candidates': 12})
    assert [candidate['track_id'] for candidate in original[:3]] == [3, 2, 1]
    result = estimate_landings(tracks, frames, timestamps, {}, {
        'max_candidates': 1, 'event_family_diversity': True, 'causal_first_event_ranking': True})
    assert [(candidate['track_id'], candidate['frame_index']) for candidate in result] == [(1, 10)]
    assert result[0]['contact_observed'] is False
