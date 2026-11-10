"""Native pixels and model occlusion hypotheses must survive integration honestly."""
import json

import cv2
import numpy as np
import pytest

from badminton.detection import _HogPeople, detect_sequence, summarize_detection_quality


class LazyFrames:
    def __init__(self, frames):
        self.frames = frames
        self.reads = []

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        self.reads.append(index)
        return self.frames[index]


def video(radius=1, count=6):
    original = []
    for index in range(count):
        frame = np.full((256, 384, 3), 25, np.uint8)
        cv2.circle(frame, (40 + index * 12, 40), radius, (255, 255, 255), -1)
        original.append(frame)
    analysis = [cv2.resize(frame, (96, 64), interpolation=cv2.INTER_AREA) for frame in original]
    return original, analysis, [index / 30 for index in range(count)]


def test_native_pixels_recover_tiny_proposals_invisible_after_downsampling():
    original, analysis, timestamps = video()
    baseline = detect_sequence(analysis, timestamps, {"person_backend": "none"})
    assert not any(record["detections"] for record in baseline)
    lazy = LazyFrames(original)
    records = detect_sequence(analysis, timestamps, {"source_frames": lazy, "person_backend": "none",
                                                     "highres_roi_size": 128})
    assert lazy.reads == list(range(len(original)))
    for index, record in enumerate(records):
        target = [(40 + index * 12) / 4, 10]
        item = next(item for item in record["detections"]
                    if np.linalg.norm(np.array([item["x"], item["y"]]) - target) < 1)
        assert item["native_resolution_observation"] is True
        assert np.allclose(item["source_px"], np.array(target) * 4, atol=1)
        assert item["source_size"] == [384, 256]
        assert item["confidence_kind"] == "heuristic_rank_not_probability"
        assert record["high_resolution"]["full_source_scan"] is False
        assert record["high_resolution"]["roi_count"] <= 8
    quality = summarize_detection_quality(records, [])
    assert quality["native_roi_frames"] == len(original)
    assert quality["native_retained_detection_count"] >= len(original)
    assert not quality["recall_measured"]
    json.dumps(records, allow_nan=False)


def test_global_and_native_observations_merge_without_duplicate_tracks():
    original, analysis, timestamps = video(radius=8)
    records = detect_sequence(analysis, timestamps, {"source_frames": original, "person_backend": "none"})
    for index, record in enumerate(records):
        target = [(40 + index * 12) / 4, 10]
        close = [item for item in record["detections"]
                 if np.linalg.norm(np.array([item["x"], item["y"]]) - target) < 3]
        assert len(close) == 1
        assert close[0]["native_resolution_observation"]
        assert close[0]["source_bbox"][2] > close[0]["bbox"][2]


def test_native_candidates_respect_analysis_roi_and_budget():
    original, analysis, timestamps = video()
    # All native shuttle pixels are in the excluded top half. Source exploration
    # must not silently bypass the caller's analysis-space ROI.
    roi = [[0, 32], [95, 32], [95, 63], [0, 63]]
    records = detect_sequence(analysis, timestamps, {"source_frames": original, "person_backend": "none",
                                                     "roi_polygon": roi, "max_candidates": 2})
    assert all(len(record["detections"]) <= 2 for record in records)
    assert all(item["y"] >= 32 for record in records for item in record["detections"])


def test_native_discoveries_cannot_bypass_the_global_candidate_cap():
    original = []
    for frame_index in range(3):
        frame = np.full((256, 384, 3), 25, np.uint8)
        for index in range(12):
            frame[30 + frame_index * 4, 20 + index * 12] = 255
        original.append(frame)
    analysis = [cv2.resize(frame, (96, 64), interpolation=cv2.INTER_AREA) for frame in original]
    records = detect_sequence(analysis, [0, 1 / 30, 2 / 30],
                              {"source_frames": original, "person_backend": "none", "max_candidates": 2})
    assert records[0]["high_resolution"]["proposal_count"] > 2
    assert records[0]["candidate_limit_reached"]
    assert all(len(row["detections"]) <= 2 for row in records)


def test_same_resolution_and_explicit_disable_are_truthfully_reported():
    _, analysis, timestamps = video(radius=8)
    ordinary = detect_sequence(analysis, timestamps, {"source_frames": analysis, "person_backend": "none"})
    assert all(row["high_resolution"]["status"] == "not_higher_resolution" for row in ordinary)
    lazy = LazyFrames(analysis)
    disabled = detect_sequence(analysis, timestamps, {"source_frames": lazy, "highres_enabled": False,
                                                     "person_backend": "none"})
    assert lazy.reads == []
    assert all(row["high_resolution"]["status"] == "disabled" for row in disabled)
    assert [row["detections"] for row in ordinary] == [row["detections"] for row in disabled]


def test_static_source_line_is_not_a_moving_shuttle():
    original, analysis, timestamps = video()
    for frame in original:
        cv2.line(frame, (0, 120), (383, 120), (250, 250, 250), 2)
    analysis = [cv2.resize(frame, (96, 64), interpolation=cv2.INTER_AREA) for frame in original]
    records = detect_sequence(analysis, timestamps, {"source_frames": original, "person_backend": "none"})
    assert all(abs(item["y"] - 30) > 2 for row in records for item in row["detections"])


def test_player_model_cache_records_age_and_never_certifies_target_player():
    class Hog:
        calls = 0

        def detectMultiScale(self, image, **kwargs):
            self.calls += 1
            return (np.array([[20, 30, 50, 120]]), np.array([.8])) if self.calls == 1 else ([], [])

    detector = _HogPeople({"person_interval": 4})
    detector.hog = Hog()
    frame = np.zeros((240, 320, 3), np.uint8)
    fresh, metadata = detector.detect(frame, 0)
    assert metadata["sampled"] and fresh[0]["age_frames"] == 0
    cached, metadata = detector.detect(frame, 3)
    assert not metadata["sampled"]
    assert cached[0]["observation_frame_index"] == 0 and cached[0]["age_frames"] == 3
    assert cached[0]["person_model_detected"]
    assert not cached[0]["person_confirmed"]
    assert not cached[0]["target_court_player_confirmed"]
    assert not cached[0]["interpolated_observation"]
    empty, metadata = detector.detect(frame, 4)
    assert metadata["sampled"] and empty == []


def test_native_source_geometry_and_configuration_are_validated():
    original, analysis, timestamps = video()
    malformed = [frame[:, :200] for frame in original]
    with pytest.raises(ValueError, match="aspect ratio"):
        detect_sequence(analysis, timestamps, {"source_frames": malformed})
    with pytest.raises(ValueError, match="ROI size"):
        detect_sequence(analysis, timestamps, {"source_frames": original, "highres_max_rois": 0})
    with pytest.raises(ValueError, match="person_backend"):
        detect_sequence(analysis, timestamps, {"person_backend": "unknown"})
