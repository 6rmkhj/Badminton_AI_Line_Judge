"""Checks that diagnostic line coverage cannot count infinite extrapolations."""
import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location('calibration_evaluation', Path(__file__).resolve().parents[1] / 'scripts/evaluate_calibration.py')
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


def test_reference_coverage_uses_observed_segment_extent():
    result = EVALUATION.visible_line_coverage([[0, 0, 10, 0]], [[[100, 0], [200, 0]]])
    assert result['mean_sample_coverage'] == 0
    assert result['segments_at_least_80_percent_covered'] == 0


def test_visible_line_metric_recognizes_close_detection_and_missing_detection():
    references = [[[10, 20], [110, 20]]]
    assert EVALUATION.visible_line_coverage([[10, 23, 110, 23]], references)['mean_sample_coverage'] == 1
    missing = EVALUATION.visible_line_coverage([], references)
    assert missing['mean_sample_coverage'] == 0
    assert missing['segments'][0]['median_distance_px'] is None
    assert EVALUATION.visible_line_coverage([], []) is None


def test_known_negative_region_reports_only_supported_length_inside_it():
    result = EVALUATION.non_court_clutter([[0, 0, 10, 0], [0, 20, 10, 20]],
                                         [[[0, -2], [10, -2], [10, 2], [0, 2]]])
    assert result['wholly_in_non_court_region_segment_count'] == 1
    assert result['non_court_length_fraction'] == pytest.approx(.5)
    assert result['non_court_detected_length_px'] == pytest.approx(10)


def test_replay_refuses_changed_cached_pixels(tmp_path):
    folder = tmp_path / 'fixture'
    folder.mkdir()
    frames = np.zeros((3, 8, 10, 3), np.uint8)
    case = {'key': 'fixture', 'frame_indices': [0, 10, 20], 'analysis_size': [10, 8],
            'frames_sha256': hashlib.sha256(frames.tobytes()).hexdigest()}
    np.savez_compressed(folder / 'frames.npz', frames=frames)
    assert len(EVALUATION.load_frames(case, tmp_path, tmp_path)) == 3
    frames[1, 4, 5] = 255
    np.savez_compressed(folder / 'frames.npz', frames=frames)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        EVALUATION.load_frames(case, tmp_path, tmp_path)
