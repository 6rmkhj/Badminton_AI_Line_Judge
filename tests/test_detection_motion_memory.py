"""Temporal search reserves protect measured continuations under finite caps."""
import copy

import numpy as np

from badminton.detection import (_NativeRoiDetector, _ProposalMotionMemory,
                                 _candidate_budget, _merge_native)


def point(x, y, confidence=.4):
    return {'x': float(x), 'y': float(y), 'radius': 1., 'confidence': confidence,
            'global_resolution_observation': True}


def test_motion_reserve_keeps_dim_continuation_without_promoting_its_score():
    memory = _ProposalMotionMemory()
    distractors = [point(12 + 4 * i, 12, .95) for i in range(14)]
    for frame in range(8):
        target = point(20 + frame * 5, 30)
        candidates = copy.deepcopy(distractors) + [target]
        memory.update(candidates, frame)
        selected = _candidate_budget(candidates, (540, 960), 8, mode='spatial', distance=1)
        if frame >= 2:
            assert target in selected
            assert target['temporal_search_observations'] >= 3
            assert target['temporal_search_confirms_identity'] is False
            assert target['confidence'] == .4
        assert len(selected) <= 8
        assert not any(d.get('temporal_search_support') for d in candidates[:-1])
    assert len(memory.paths) <= 640


def test_native_search_keeps_short_gap_bounded_without_emitting_a_detection():
    memory = _ProposalMotionMemory()
    for index in range(4):
        memory.update([point(50 + index * 8, 55)], index)
    memory.update([], 4)
    guides = memory.search_guides(4)
    assert guides and all(set(g) == {'x', 'y', 'support'} for g in guides)
    detector = _NativeRoiDetector({'highres_max_rois': 4, 'highres_exploration_rois': 1,
                                  'highres_roi_size': 64}, (100, 200),
                                 np.zeros((100, 200), np.uint8), np.full((100, 200), 255, np.uint8), 40)
    detector.motion_guides = guides
    boxes = detector._boxes([], (400, 800, 3), 4)
    assert len(boxes) <= 4
    assert boxes[0]['origin'] == 'observed_motion_search'
    assert all(len(box['bbox']) == 4 for box in boxes)
    for index in range(5, 9):
        memory.update([], index)
    assert memory.search_guides(8) == []


def test_native_refinement_preserves_search_support_without_identity_evidence():
    item = {**point(20, 30), 'temporal_search_support': 4., 'temporal_search_observations': 5,
            'temporal_search_confirms_identity': False}
    native = {**point(20.2, 30.2, .5), 'native_resolution_observation': True}
    merged = _merge_native([item], [native], 2)
    assert len(merged) == 1
    assert merged[0]['temporal_search_support'] == 4.
    assert merged[0]['temporal_search_confirms_identity'] is False
    assert merged[0]['confidence'] == .5


def test_native_measurements_extend_memory_after_global_observations_disappear():
    memory = _ProposalMotionMemory()
    for index in range(3):
        memory.update([point(40 + index * 6, 30)], index)
    for index in range(3, 9):
        memory.update([], index)
        measured = point(40 + index * 6, 30)
        memory.update([measured], index, extend=True)
        assert measured['temporal_search_observations'] >= 3
        assert measured['temporal_search_confirms_identity'] is False
        guides = memory.search_guides(index)
        assert any(abs(g['x'] - measured['x']) < .01 for g in guides)
        assert all(np.isfinite(g['support']) for g in guides)
        assert len(memory.paths) <= 640


def test_native_motion_recovery_follows_faint_pixels_after_brightness_transition():
    import cv2
    from badminton.detection import detect_sequence
    originals = []
    frames = []
    for index in range(9):
        source = np.full((400, 800, 3), 130, np.uint8)
        brightness = 240 if index < 3 else 145
        cv2.circle(source, (4 * (40 + index * 6), 180), 6, (brightness,) * 3, -1)
        originals.append(source)
        frames.append(cv2.resize(source, (200, 100), interpolation=cv2.INTER_AREA))
    config = {'source_frames': originals, 'person_backend': 'none', 'candidate_selection': 'spatial',
              'max_candidates': 16, 'global_candidate_reserve': 8, 'temporal_candidate_memory': True,
              'highres_roi_size': 128, 'highres_max_rois': 4, 'highres_exploration_rois': 1,
              'native_local_components': True}
    ordinary = detect_sequence(frames, [i / 30 for i in range(9)], config)
    recovered = detect_sequence(frames, [i / 30 for i in range(9)], {**config, 'native_motion_recovery': True})
    assert not any(abs(item['x'] - 70) < 2 and abs(item['y'] - 45) < 2
                   for item in ordinary[5]['detections'])
    for index in range(3, 9):
        near = [item for item in recovered[index]['detections']
                if abs(item['x'] - (40 + index * 6)) < 2 and abs(item['y'] - 45) < 2]
        assert near
        assert any(item.get('native_resolution_observation') for item in near)
        assert recovered[index]['high_resolution']['roi_count'] <= 4
        assert len(recovered[index]['detections']) <= 16
