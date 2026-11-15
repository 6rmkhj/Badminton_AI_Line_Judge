"""Prevent wood-grain candidates from replacing visible painted court lines.

The inputs are lossless medians of nine repository-video frames. These compact
single-image regressions differ from the nine-frame quantitative benchmark:
motion occlusion and temporal mask recovery are not evaluated here. Approximate
AI references are used only for assertions, never passed to calibration.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from badminton.calibration import calibrate


FIXTURES = Path(__file__).parent / "fixtures" / "court-calibration"
CASES = json.loads((FIXTURES / "references.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["key"])
def test_painted_lines_survive_competing_wood_grain_candidates(case):
    path = FIXTURES / case["image"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["image_sha256"]
    frame = cv2.imread(str(path))
    assert frame is not None
    assert list(frame.shape[1::-1]) == case["analysis_size"]

    # The engine receives only pixels. In particular, no colour, line labels,
    # source identity, court corners or reference segments are supplied.
    result = calibrate([frame])
    assert result["status"] != "ok"
    assert result["metric_coordinates_available"] is False
    assert result["judgment_region_px"] is None
    assert result["line_segments_px"]

    detected = np.asarray(result["line_segments_px"], dtype=float).reshape(-1, 2, 2)
    starts = detected[:, 0]
    vectors = detected[:, 1] - starts
    lengths_squared = np.maximum(np.sum(vectors ** 2, axis=1), 1e-9)
    coverage = []
    for reference in case["visible_line_references_px"]:
        samples = np.linspace(reference[0], reference[1], 64)
        fraction = np.sum((samples[:, None] - starts) * vectors, axis=2) / lengths_squared
        nearest = starts + np.clip(fraction, 0, 1)[:, :, None] * vectors
        distance = np.linalg.norm(samples[:, None] - nearest, axis=2).min(axis=1)
        coverage.append(float(np.mean(distance <= 5)))

    # Five independently marked segments protect against counting unrelated
    # floorboards as detections. A median can retain local player occlusion;
    # require broad recovery without pretending every pixel remains visible.
    assert np.mean(coverage) >= .80, coverage
    assert sum(value >= .80 for value in coverage) >= 4, coverage
    assert min(coverage) >= .50, coverage
