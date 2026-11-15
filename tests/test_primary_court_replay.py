"""Keep the foreground outer court distinct from its centre service line.

The fixtures retain original video pixels and the unlabelled detector output.
Approximate AI visual corner references are assertions only. They are neither
detector inputs nor independently measured metric ground truth.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from badminton.primary_court import select_primary_court
from badminton.decision import decide


FIXTURES = Path(__file__).parent / "fixtures" / "primary-court"
CASES = json.loads((FIXTURES / "references.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["view"])
def test_primary_court_ends_at_two_outer_corners_not_service_centre(case):
    for key in ("frame", "observation"):
        path = FIXTURES / case[key + "_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case[key + "_sha256"]
    frame = cv2.imread(str(FIXTURES / case["frame_file"]))
    observed = json.loads((FIXTURES / case["observation_file"]).read_text(encoding="utf-8"))
    assert frame.shape == (540, 960, 3)

    # No corner, source identity, centre-line role or expected decision reaches
    # the selector. The exact same signature is used by live calibration.
    selected = select_primary_court([frame], observed)
    assert selected["status"] == "selected", selected
    corners = np.asarray(selected["near_corners_px"], float)
    reference = np.asarray(case["reference_near_corners_px"], float)
    assert corners.shape == (2, 2)
    np.testing.assert_allclose(corners, reference, atol=15)

    # The old tempting shortcut selected half of the court by treating the
    # prominent centre service line as its right sideline.
    centre = np.asarray(case["reference_centre_service_baseline_intersection_px"])
    assert np.linalg.norm(corners[1] - centre) > 90
    boundaries = selected["boundary_segments_px"]
    assert {line["role"] for line in boundaries} == {"left", "right", "near"}
    assert all(line["observed"] is True for line in boundaries)
    near = np.asarray(next(line["segment_px"] for line in boundaries if line["role"] == "near"))
    near = near.reshape(2, 2)
    assert min(np.linalg.norm(near - corners, axis=1).max(),
               np.linalg.norm(near[::-1] - corners, axis=1).max()) < 15
    assert selected["scope"] == "near_half"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["view"])
def test_selected_real_court_supports_local_pixel_verdicts(case):
    frame = cv2.imread(str(FIXTURES / case["frame_file"]))
    observed = json.loads((FIXTURES / case["observation_file"]).read_text(encoding="utf-8"))
    calibration = {**observed, "primary_court": select_primary_court([frame], observed)}
    assert calibration["status"] == "review"
    assert calibration["metric_coordinates_available"] is False

    # Purpose-built test contacts, not actual shuttle landings in these clips.
    # The inside point is beyond the centre service line, so falsely selecting
    # that line as the right boundary would reverse the expected verdict.
    for key, expected in (("inside", "IN"), ("outside", "OUT"), ("outside_scope", "REVIEW")):
        contact = {"contact_px": case[f"geometry_probe_{key}_px"],
                   "contact_observed": True, "confidence": 1.0,
                   "identity_confidence": 1.0, "uncertainty_px": 1.0,
                   "contact_radius_px": 0.0}
        verdict = decide(contact, calibration, {"game": "doubles", "phase": "rally"})
        assert verdict["decision"] == expected, verdict
        assert verdict.get("court_position_m") is None
        assert decide({**contact, "contact_observed": False}, calibration)["decision"] == "REVIEW"
