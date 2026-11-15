from copy import deepcopy

from badminton.pipeline import select_display_landing, select_evidence_landings


def estimate():
    return {'selected_family_id': 'family_001', 'member_event_ids': ['event_002', 'event_003'],
            'native_evidence': {'status': 'supported'},
            'family_resolution': {'status': 'selected', 'selected_family': {
                'family_id': 'family_001', 'member_event_ids': ['event_002', 'event_003'],
                'representative': {'event_id': 'event_003'}}}}


def test_native_evidence_selects_its_own_trajectory_not_an_unrelated_ranked_candidate():
    candidates = [{'event_id': event, 'contact_observed': False, 'identity_confidence': .4}
                  for event in ['event_000', 'event_002', 'event_003']]
    before = deepcopy(candidates)
    for verdict in ['IN', 'OUT', 'REVIEW']:
        inference = {**estimate(), 'decision': verdict}
        selected = select_display_landing(candidates, inference, {'decision': 'REVIEW'})
        assert selected is candidates[2]
        assert candidates == before  # Presentation must not promote evidence flags.


def test_unmeasured_or_inconsistent_family_cannot_redirect_evidence():
    candidates = [{'event_id': 'event_000'}, {'event_id': 'event_003'}]
    variants = [estimate() for _ in range(4)]
    variants[0]['native_evidence']['status'] = 'review'
    variants[1]['family_resolution']['status'] = 'review'
    variants[2]['member_event_ids'] = ['unknown']
    variants[3]['selected_family_id'] = 'different'
    for inference in variants:
        assert select_display_landing(candidates, inference, {'decision': 'REVIEW'}) is candidates[0]
    assert select_display_landing(candidates, estimate(), {'decision': 'IN'}) is candidates[0]
    assert select_display_landing([], estimate(), {'decision': 'REVIEW'}) is None


def test_selected_event_beyond_gallery_limit_keeps_its_trajectory_and_source_context():
    ranked = [{'event_id': f'event_{i:03}'} for i in range(10)]
    selected = ranked[8]
    evidence = select_evidence_landings(ranked, selected)
    assert evidence[0] is selected  # Identity attaches trajectory_context in analyze_video.
    assert len(evidence) == 6
    assert evidence[1:] == ranked[:5]
    assert len(ranked) == 10 and ranked[0]['event_id'] == 'event_000'
    assert select_evidence_landings(ranked, None) == ranked[:6]
