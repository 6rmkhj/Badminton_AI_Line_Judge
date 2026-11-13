import json

import cv2
import numpy as np
import pytest

from badminton.detection import detect_sequence
from badminton.tracking import track_detections


def test_two_shuttle_candidates_and_static_line_rejection():
    frames = []
    for i in range(12):
        frame = np.full((180, 320, 3), (35, 85, 35), dtype=np.uint8)
        cv2.line(frame, (0, 150), (319, 150), (245, 245, 245), 2)
        cv2.circle(frame, (35 + i * 10, 50), 3, (250, 250, 250), -1)
        cv2.circle(frame, (270 - i * 9, 110), 3, (255, 255, 255), -1)
        frames.append(frame)
    records = detect_sequence(frames, [i / 30 for i in range(12)])
    assert len(records) == 12
    for i, record in enumerate(records):
        assert any(abs(d["x"] - (35 + i * 10)) < 2 and abs(d["y"] - 50) < 2 for d in record["detections"])
        assert any(abs(d["x"] - (270 - i * 9)) < 2 and abs(d["y"] - 110) < 2 for d in record["detections"])
        assert not any(abs(d["y"] - 150) < 2 for d in record["detections"])
        assert record["detector"]["trained_model"] is False
    json.dumps(records, allow_nan=False)


def test_roi_and_input_validation():
    frames = []
    for i in range(5):
        frame = np.zeros((100, 200, 3), np.uint8)
        cv2.circle(frame, (20 + i * 5, 50), 3, (255, 255, 255), -1)
        cv2.circle(frame, (160 - i * 5, 50), 3, (255, 255, 255), -1)
        frames.append(frame)
    records = detect_sequence(frames, list(range(5)), {"roi_polygon": [[0, 0], [99, 0], [99, 99], [0, 99]]})
    assert all(d["x"] < 100 for r in records for d in r["detections"])
    with pytest.raises(ValueError, match="strictly increasing"):
        detect_sequence(frames, [0, 1, 1, 2, 3])
    with pytest.raises(ValueError, match="equal length"):
        detect_sequence(frames, [0])
    with pytest.raises(ValueError, match="existing model_path"):
        detect_sequence(frames, list(range(5)), {"backend": "onnx", "model_path": "/missing/model.onnx"})


def test_large_moving_white_objects_are_not_fragmented_into_shuttles():
    frames = []
    for i in range(8):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100 + i * 5, 40), (145 + i * 5, 140), (245, 245, 245), -1)
        cv2.circle(frame, (30 + i * 5, 30), 3, (250, 250, 250), -1)
        frames.append(frame)
    records = detect_sequence(frames, [i / 30 for i in range(8)])
    for i, record in enumerate(records):
        assert any(abs(d['x'] - (30 + i * 5)) < 2 for d in record['detections'])
        assert all(not (98 + i * 5 <= d['x'] <= 147 + i * 5 and 38 <= d['y'] <= 142) for d in record['detections'])


def detection(x, y=50):
    return {"x": float(x), "y": float(y), "radius": 2.0, "confidence": 0.9}


def record(i, candidates, occluders=None):
    return {"frame_index": i, "time": i / 30, "detections": candidates, "occlusion_regions": occluders or []}


def test_constant_velocity_crossing_keeps_both_paths():
    records = []
    for i in range(20):
        candidates = [detection(20 + i * 8), detection(190 - i * 8)]
        if i % 2:
            candidates.reverse()
        records.append(record(i, candidates))
    tracks = track_detections(records)
    assert len(tracks) == 2
    tracks.sort(key=lambda t: t["observations"][0]["x"])
    assert [o["x"] for o in tracks[0]["observations"]] == [20 + i * 8 for i in range(20)]
    assert [o["x"] for o in tracks[1]["observations"]] == [190 - i * 8 for i in range(20)]
    assert all(t["target_court_confirmed"] is False for t in tracks)
    json.dumps(tracks, allow_nan=False)


def test_occluded_gap_is_reconnected_without_fabricated_observations():
    records = [record(i, [] if i in (3, 4) else [detection(20 + i * 6)],
                      [{"bbox": [30, 20, 60, 70]}] if i in (3, 4) else []) for i in range(10)]
    tracks = track_detections(records)
    assert len(tracks) == 1
    track = tracks[0]
    assert [o["frame_index"] for o in track["observations"]] == [0, 1, 2, 5, 6, 7, 8, 9]
    assert track["gaps"][0]["possible_occlusion"]
    assert track["gaps"][0]["missing_frame_count"] == 2
    assert track["reconnections"]
    assert not track["gaps"][0]["observations_interpolated"]


def test_ambiguous_reappearance_is_not_forced_into_old_identity():
    records = [record(i, [detection(20 + i * 5)]) for i in range(3)]
    records += [record(3, []), record(4, [])]
    for i in range(5, 10):
        records.append(record(i, [detection(20 + i * 5 - 2), detection(20 + i * 5 + 2)]))
    tracks = track_detections(records)
    first = next(t for t in tracks if t["observations"][0]["frame_index"] == 0)
    assert len(first["observations"]) == 3
    assert first["identity_ambiguous"]
    assert len(tracks) >= 2


def test_single_frame_noise_does_not_form_track_and_invalid_time_fails():
    records = [record(i, [detection(20 + i * 5), detection(200, i * 100)]) for i in range(8)]
    tracks = track_detections(records)
    assert len(tracks) == 1
    with pytest.raises(ValueError, match="strictly increasing"):
        track_detections([record(0, []), record(0, [])])


def test_nonfinite_candidates_do_not_poison_assignment():
    rows = [record(i, [detection(20 + i * 4), {**detection(100), "radius": float("nan")}]) for i in range(6)]
    tracks = track_detections(rows)
    assert len(tracks) == 1
    assert len(tracks[0]["observations"]) == 6
    with pytest.raises(ValueError, match="Invalid tracker"):
        track_detections(rows, {"max_gate_px": float("nan")})


def test_offline_links_require_two_sided_velocity_consistency():
    rows = [record(i, [detection(20 + i * 5)] if i < 4 or i >= 11 else []) for i in range(16)]
    linked = track_detections(rows, {"max_gap_seconds": 0.1})
    assert len(linked) == 1
    assert linked[0]["source_track_ids"] == [1, 2]
    assert linked[0]["reconnections"][0]["method"] == "forward_backward_velocity_consistency"
    # A new track moving back towards the old endpoint is not the same ballistic
    # continuation even when one endpoint lies in the forward search gate.
    reversed_rows = [record(i, [detection(20 + i * 5)] if i < 4 else [] if i < 11 else [detection(75 - (i - 11) * 5)]) for i in range(16)]
    separated = track_detections(reversed_rows, {"max_gap_seconds": 0.1})
    assert len(separated) == 2


def test_scorer_domain_errors_are_not_hidden_or_silently_ranked(monkeypatch):
    from badminton import learning
    class Scorer:
        provenance = {"preprocessing": "opencv_source_bgr_area_v1"}
        def score(self, frame, detections, preprocessing=None):
            if preprocessing != self.provenance["preprocessing"]:
                raise ValueError("Candidate scorer preprocessing mismatch")
            return [{**item, "learned_confidence": 0.1} for item in detections]
    monkeypatch.setattr(learning, "load_candidate_scorer", lambda path: Scorer())
    frames = []
    for i in range(5):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + i * 5, 50), 3, (255, 255, 255), -1)
        frames.append(frame)
    options = {"scorer_path": "test-fixture", "use_learned_confidence": True}
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        detect_sequence(frames, list(range(5)), options)
    with pytest.raises(ValueError, match="preprocessing mismatch"):
        detect_sequence(frames, list(range(5)), {**options, "frame_preprocessing": "h264_preview"})
    result = detect_sequence(frames, list(range(5)), {**options, "frame_preprocessing": "opencv_source_bgr_area_v1"})
    assert any(record["detections"] for record in result)
    assert all(item["confidence"] == 0.1 and item["confidence_kind"] == "learned_score_not_calibrated"
               for record in result for item in record["detections"])


def test_dark_cork_proposals_are_opt_in_and_ignore_static_dark_lines():
    frames = []
    for i in range(10):
        frame = np.full((180, 320, 3), 175, dtype=np.uint8)
        cv2.line(frame, (0, 150), (319, 150), (60, 60, 60), 2)
        cv2.circle(frame, (30 + 20*i, 60), 3, (45, 45, 45), -1)
        frames.append(frame)
    stamps = [i/30 for i in range(10)]
    baseline = detect_sequence(frames, stamps, {'person_backend': 'none'})
    improved = detect_sequence(frames, stamps, {'person_backend': 'none', 'include_dark_components': True})
    assert not any(r['detections'] for r in baseline)
    for i, record in enumerate(improved):
        assert any(abs(d['x']-(30+20*i)) < 3 and abs(d['y']-60) < 3 for d in record['detections'])
        assert not any(abs(d['y']-150) < 3 for d in record['detections'])
        assert record['detector']['trained_model'] is False
        assert record['detector']['include_dark_components'] is True


def test_low_contrast_motion_blur_survives_without_turning_static_lines_into_shuttles():
    frames = []
    for index in range(8):
        frame = np.full((540, 960, 3), 175, np.uint8)
        cv2.line(frame, (0, 160), (959, 160), (60, 60, 60), 2)
        # A long exposure can make the shuttle a dim, thin streak rather than
        # a round white spot. Its local contrast is below the old dark cutoff.
        cv2.rectangle(frame, (30 + index * 25, 55), (31 + index * 25, 67), (162, 162, 162), -1)
        frames.append(frame)
    config = {'person_backend': 'none', 'include_dark_components': True,
              'dark_contrast_threshold': 8, 'dark_motion_threshold': 8, 'max_component_aspect': 12,
              'candidate_selection': 'spatial'}
    records = detect_sequence(frames, [index / 30 for index in range(8)], config)
    for index, record in enumerate(records):
        assert any(abs(item['x'] - (30.5 + index * 25)) < 1 and abs(item['y'] - 61) < 2
                   for item in record['detections'])
        assert all(abs(item['y'] - 160) > 3 for item in record['detections'])


@pytest.mark.parametrize('key,value', [('dark_contrast_threshold', 0),
                                       ('dark_max_saturation', 256),
                                       ('max_component_aspect', float('nan'))])
def test_dark_component_configuration_is_bounded(key, value):
    with pytest.raises(ValueError, match=key):
        detect_sequence([np.zeros((40, 60, 3), np.uint8)], [0], {key: value})


def test_dark_search_keeps_a_bright_shuttle_touching_a_dark_foreground_edge():
    frames = []
    for index in range(8):
        frame = np.full((180, 320, 3), 175, np.uint8)
        cv2.rectangle(frame, (60 + 12 * index, 40), (105 + 12 * index, 140), (65, 65, 65), -1)
        cv2.circle(frame, (105 + 12 * index, 90), 3, (250, 250, 250), -1)
        frames.append(frame)
    stamps = [index / 30 for index in range(8)]
    for dark_search in (False, True):
        records = detect_sequence(frames, stamps, {'person_backend': 'none',
            'include_dark_components': dark_search, 'dark_contrast_threshold': 8,
            'dark_motion_threshold': 8, 'candidate_selection': 'spatial'})
        for index, record in enumerate(records):
            item = next(item for item in record['detections']
                        if abs(item['x'] - (105 + 12 * index)) < 2 and abs(item['y'] - 90) < 2)
            assert item['appearance_polarity'] == 'bright'
