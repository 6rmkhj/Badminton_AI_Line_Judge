"""Review-directed search changes candidates, never contact or identity truth."""
import math

import cv2
import numpy as np
import pytest

from badminton.detection import _candidate_budget, _nms, _refined_budget, detect_sequence


def proposal(x, y, score=.8, radius=1):
    return {'x': x, 'y': y, 'confidence': score, 'radius': radius}


def linear_nms(items, distance, limit):
    result = []
    for item in sorted(items, key=lambda d: (-d['confidence'], d['x'], d['y'])):
        if not any(math.hypot(item['x']-old['x'], item['y']-old['y'])
                   <= max(distance, min(item['radius'], old['radius'])) for old in result):
            result.append(item)
        if len(result) >= limit:
            break
    return result


def test_bucketed_nms_preserves_ranking_and_variable_radius_suppression():
    random = np.random.default_rng(31)
    items = [proposal(float(x), float(y), float(score), float(radius))
             for x, y, score, radius in zip(random.uniform(-20, 200, 600),
                                           random.uniform(-20, 100, 600),
                                           random.uniform(0, 1, 600), random.uniform(.2, 18, 600))]
    items.extend([proposal(0, 0, 1, 3), proposal(3, 0, 1, 3), proposal(-3, 0, 1, 3)])
    for distance in (.5, 3, 8):
        for limit in (1, 64, 512):
            assert _nms(items, distance, limit) == linear_nms(items, distance, limit)


def test_spatial_budget_recovers_remote_weak_candidate_without_changing_scores():
    clutter = [proposal(10+i*4, 10, .99-i*.001) for i in range(20)]
    remote = proposal(280, 160, .35)
    items = clutter + [remote]
    assert remote not in _candidate_budget(items, (180, 320), 8)
    selected = _candidate_budget(items, (180, 320), 8, mode='spatial')
    assert remote in selected and len(selected) == 8
    assert all(item in items for item in selected)
    assert remote['confidence'] == .35


def test_refinement_budget_keeps_global_search_and_avoids_duplicate_observations():
    items = [proposal(10+i*5, 10, .99-i*.001) for i in range(20)]
    target = proposal(270, 150, .2)
    regions = [{'bbox': [250, 130, 50, 40]}]
    selected, count = _refined_budget(items+[target], (180, 320), 8, 'ranked', 3, regions)
    assert target in selected and count == 1
    assert len(selected) == 8 and items[0] in selected
    assert sum(item is target for item in selected) == 1
    assert _refined_budget(items+[target], (180, 320), 1, 'ranked', 3, regions) == ([target], 1)


def test_source_refinement_is_bounded_to_requested_frames_and_keeps_provenance():
    raw = []
    for index in range(6):
        frame = np.full((512, 768, 3), 175, np.uint8)
        frame[450, 700+index*4] = 20
        raw.append(frame)
    low = [cv2.resize(frame, (192, 128), interpolation=cv2.INTER_AREA) for frame in raw]
    base = {'person_backend': 'none', 'source_frames': raw, 'include_dark_components': True,
            'highres_roi_size': 64, 'highres_max_rois': 2, 'highres_exploration_rois': 1}
    stamps = [i/30 for i in range(6)]
    baseline = detect_sequence(low, stamps, base)
    refined = detect_sequence(low, stamps, {**base, 'refinement_regions': [
        {'frame_start': 1, 'frame_end': 4, 'bbox': [165, 100, 25, 25]}]})
    assert not any(row['detections'] for row in baseline)
    for index in range(1, 5):
        record = refined[index]
        target = next(item for item in record['detections']
                      if abs(item['x']-(700+index*4)/4) < 1 and abs(item['y']-112.5) < 1)
        assert target['native_resolution_observation']
        assert target['roi_origin'] == 'review_refinement_region'
        assert target['confidence_kind'] == 'heuristic_rank_not_probability'
        assert not record['refinement']['provides_contact_evidence']
        assert not record['refinement']['provides_target_identity']
        assert record['high_resolution']['roi_count'] <= 2
        assert len(record['detections']) <= 64
    for index in (0, 5):
        assert not refined[index]['refinement']['active_regions']
        assert not refined[index]['refinement']['assistance_used']


@pytest.mark.parametrize('region', [
    {'frame_start': True, 'frame_end': 2, 'bbox': [1, 1, 4, 4]},
    {'frame_start': 0, 'frame_end': 3, 'bbox': [1, 1, 4, 4]},
    {'frame_start': 0, 'frame_end': 2, 'bbox': [-1, 1, 4, 4]},
    {'frame_start': 0, 'frame_end': 2, 'bbox': [8, 8, 4, 4]},
    {'frame_start': 0, 'frame_end': 2, 'bbox': [1, 1, float('nan'), 4]},
])
def test_refinement_rejects_invalid_source_geometry_and_timeline(region):
    frames = [np.zeros((10, 10, 3), np.uint8) for _ in range(3)]
    with pytest.raises(ValueError, match='refinement'):
        detect_sequence(frames, [0, 1, 2], {'refinement_regions': [region]})
