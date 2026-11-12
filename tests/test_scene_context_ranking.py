from copy import deepcopy

import pytest

from badminton.events import prioritize_review_candidates, validate_contact_modes


def candidate(name, rank, *, flight=True, factor=1., **fields):
    return {'name': name, 'frame_index': 5, 'time': .2, 'candidate_px': [25., 20.],
            'confidence': .8, 'identity_confidence': .5,
            'flight_trajectory_support': {'supported_for_review': flight,
                'approach_displacement_y_px': rank, 'fit_residual_px': 0.},
            'scene_context': {'test_factor': factor}, **fields}


@pytest.fixture(autouse=True)
def fixed_factors(monkeypatch):
    # Pixel provenance/geometry are tested independently by the helper suite;
    # these cases exercise the product's ranking contract between evidence tiers.
    monkeypatch.setattr('badminton.scene_context.scene_rank_factor',
                        lambda item: item['scene_context']['test_factor'])


def test_soft_context_reorders_flights_without_changing_raw_evidence():
    candidates = [candidate('wall_appearance', 100, factor=.5), candidate('other_flight', 70)]
    original = deepcopy(candidates)
    result = prioritize_review_candidates(candidates, [])
    assert result[0]['name'] == 'other_flight'
    assert candidates == original
    assert all(item['flight_trajectory_support']['supported_for_review'] for item in result)


@pytest.mark.parametrize('first', ['flight', 'requested', 'contact', 'guided'])
def test_soft_context_cannot_cross_stronger_evidence_tiers(first):
    strong = candidate('strong', 1, factor=.5, confidence=.01)
    weaker = candidate('weaker', 100, confidence=1.)
    regions = []
    if first == 'flight':
        weaker['flight_trajectory_support']['supported_for_review'] = False
    elif first == 'requested':
        strong['identity_source'] = 'operator_selected_track'
    elif first == 'contact':
        strong.update(contact_observed=True, contact_evidence_source='human_review')
        strong['flight_trajectory_support']['supported_for_review'] = False
    else:
        weaker['candidate_px'] = [80., 80.]
        regions = [{'frame_start': 0, 'frame_end': 10, 'bbox': [0., 0., 40., 40.]}]
    assert prioritize_review_candidates([weaker, strong], regions)[0]['name'] == 'strong'


def test_native_contact_budget_retains_identity_first_order():
    strong_identity = candidate('identity', 0, flight=False, factor=.5, identity_confidence=.9)
    weaker_identity = candidate('other', 0, flight=False, identity_confidence=.8)
    assert prioritize_review_candidates([weaker_identity, strong_identity], [],
                                        identity_first=True)[0]['name'] == 'identity'


def test_absent_context_keeps_existing_ranking():
    candidates = [candidate('first', 100), candidate('second', 70)]
    for item in candidates:
        item.pop('scene_context')
    assert prioritize_review_candidates(candidates, [])[0]['name'] == 'first'


@pytest.mark.parametrize('invalid', [1, 'true', None, []])
def test_scene_option_requires_boolean(invalid):
    with pytest.raises(ValueError, match='scene_context_ranking'):
        validate_contact_modes({'scene_context_ranking': invalid})
