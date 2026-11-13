from copy import deepcopy
import json

import pytest

from badminton.candidate_identity import apply_competing_continuation_evidence


def fixture():
    prefix = [{'frame_index': i, 'time': i / 30, 'x': 100 + 2 * i, 'y': 20 + i * i,
               'radius': 2., 'appearance_polarity': 'bright', 'observed': True} for i in range(10)]
    switched = [{'frame_index': i, 'time': i / 30, 'x': 100 + 2 * i, 'y': 101 - 5 * (i - 9),
                 'radius': .5, 'appearance_polarity': 'dark', 'observed': True} for i in range(10, 13)]
    continued = [{'frame_index': i, 'time': i / 30, 'x': 100 + 2 * i, 'y': 20 + i * i,
                  'radius': 2., 'appearance_polarity': 'bright', 'observed': True} for i in range(10, 13)]
    tracks = [{'track_id': 1, 'flight_hypothesis': {}, 'observations': prefix + switched},
              {'track_id': 2, 'flight_hypothesis': {}, 'observations': deepcopy(prefix) + continued}]
    candidates = [{'track_id': key, 'frame_index': frame, 'time': frame / 30,
                   'candidate_px': [100 + 2 * frame, 20 + frame * frame],
                   'confidence': .7, 'identity_confidence': .15, 'contact_observed': False, 'contact_px': None,
                   'reason_codes': ['FIRST_CONTACT_NOT_OBSERVED'],
                   'image_motion_event': {'kind': 'descent_and_rebound'},
                   'flight_trajectory_support': {'supported_for_review': True}}
                  for key, frame in [(1, 9), (2, 12)]]
    return candidates, tracks


@pytest.mark.parametrize('noisy_prefix', [False, True])
def test_shared_actual_continuation_and_extreme_appearance_break_only_withhold_earlier_branch_priority(noisy_prefix):
    candidates, tracks = fixture()
    if noisy_prefix:
        for track in tracks:
            for i in [2, 5]:
                track['observations'][i].update(radius=.5, appearance_polarity='dark')
    original_candidates, original_tracks = deepcopy(candidates), deepcopy(tracks)
    result = apply_competing_continuation_evidence(candidates, tracks, 1 / 30)
    assert result[0]['flight_trajectory_support']['supported_for_review'] is False
    evidence = result[0]['competing_continuation_evidence']
    assert evidence['other_track_id'] == '2'
    assert evidence['shared_observation_count'] >= 5
    assert evidence['continuation_frame_indices'] == [10, 11, 12]
    assert evidence['identity_confirmed'] is False and evidence['physical_contact_confirmed'] is False
    assert result[0]['confidence'] == .7 and result[0]['identity_confidence'] == .15
    assert result[0]['contact_observed'] is False and result[0]['contact_px'] is None
    assert result[1]['flight_trajectory_support']['supported_for_review'] is True
    assert candidates == original_candidates and tracks == original_tracks
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('change', ['no_shared_prefix', 'one_coincident_point', 'interpolated_continuation',
                                  'size_preserved', 'polarity_preserved', 'unknown_appearance'])
def test_an_alternative_or_appearance_change_alone_cannot_veto_a_real_rebound(change):
    candidates, tracks = fixture()
    if change == 'no_shared_prefix':
        for row in tracks[1]['observations']:
            row['x'] += 30
    elif change == 'one_coincident_point':
        for row in tracks[1]['observations'][11:]:
            row['x'] += 30
    elif change == 'interpolated_continuation':
        for row in tracks[1]['observations'][10:]:
            row['observed'] = False
    elif change == 'size_preserved':
        for row in tracks[0]['observations'][10:]:
            row['radius'] = 2.
    elif change == 'polarity_preserved':
        for row in tracks[0]['observations'][10:]:
            row['appearance_polarity'] = 'bright'
    else:
        for row in tracks[0]['observations']:
            row.pop('appearance_polarity', None)
    result = apply_competing_continuation_evidence(candidates, tracks, 1 / 30)
    assert result[0]['flight_trajectory_support']['supported_for_review'] is True
    assert 'competing_continuation_evidence' not in result[0]


def test_hypothetical_later_event_without_an_observed_continuation_cannot_veto():
    candidates, tracks = fixture()
    tracks[1]['observations'] = tracks[1]['observations'][:10]
    result = apply_competing_continuation_evidence(candidates, tracks, 1 / 30)
    assert result[0]['flight_trajectory_support']['supported_for_review'] is True
