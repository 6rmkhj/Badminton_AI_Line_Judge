import math

import pytest

from badminton.decision import decide, evaluate_contact, rule_region


def test_doubles_boundary_is_in_but_uncertain_boundary_is_review():
    assert evaluate_contact([[0.0, 5.0]])["decision"] == "IN"
    assert evaluate_contact([[0.0, 5.0]], uncertainty_m=.01)["decision"] == "REVIEW"
    assert evaluate_contact([[-.03, 5]], uncertainty_m=.01)["decision"] == "OUT"


def test_footprint_touch_counts_even_with_outside_centre():
    assert evaluate_contact([[-.1, 4], [0, 4], [0, 4.1], [-.1, 4.1]])["decision"] == "IN"
    assert evaluate_contact([[-.1, 4], [.03, 4], [.03, 4.1], [-.1, 4.1]], uncertainty_m=.02)["decision"] == "IN"


def test_corner_uncertainty_uses_euclidean_separation():
    assert evaluate_contact([[-.03, -.04]], uncertainty_m=.051)["decision"] == "REVIEW"
    assert evaluate_contact([[-.03, -.04]], uncertainty_m=.049)["decision"] == "OUT"


def test_singles_alleys_are_out():
    assert evaluate_contact([[.2, 7]], {"game": "singles"})["decision"] == "OUT"
    assert evaluate_contact([[.2, 7]], {"game": "doubles"})["decision"] == "IN"


def test_service_rules_restrict_near_far_and_doubles_back_line():
    near = {"game": "doubles", "phase": "service", "half": "near", "side": "left"}
    far = {**near, "half": "far", "side": "right"}
    assert evaluate_contact([[1, .5]], near)["decision"] == "OUT"
    assert evaluate_contact([[1, .76]], near)["decision"] == "IN"
    assert evaluate_contact([[4, 10]], near)["decision"] == "OUT"
    assert evaluate_contact([[4, 10]], far)["decision"] == "IN"
    assert evaluate_contact([[4, 13]], far)["decision"] == "OUT"
    assert evaluate_contact([[4, 13]], {**far, "game": "singles"})["decision"] == "IN"


def test_service_box_cannot_be_guessed():
    with pytest.raises(ValueError):
        rule_region({"phase": "service"})


def accepted_calibration():
    return {"status": "ok", "confidence": .95, "method": "colour_hough_court_template",
            "homography": [[.01, 0, 0], [0, .01, 0], [0, 0, 1]], "line_fit_error_px": 0.5}


def observed_contact():
    return {"contact_observed": True, "contact_px": [300, 600], "confidence": .95,
            "identity_confidence": .95, "contact_radius_px": 1, "uncertainty_px": 1}


def test_airborne_or_heuristic_candidate_never_projects_to_ground():
    result = decide({**observed_contact(), "contact_observed": False}, accepted_calibration())
    assert result["decision"] == "REVIEW"
    assert result["court_position_m"] is None
    assert "FIRST_CONTACT_NOT_OBSERVED" in result["reason_codes"]


@pytest.mark.parametrize("changes", [{"status": "review"}, {"method": "manual_reference"},
                                      {"confidence": float("nan")}, {"homography": [[0]*3]*3}])
def test_unaccepted_or_invalid_calibration_abstains(changes):
    assert decide(observed_contact(), {**accepted_calibration(), **changes})["decision"] == "REVIEW"


@pytest.mark.parametrize("changes", [{"occluded": True}, {"identity_confidence": .2},
                                      {"confidence": float("nan")}, {"contact_px": [math.inf, 1]}])
def test_bad_contact_evidence_abstains(changes):
    assert decide({**observed_contact(), **changes}, accepted_calibration())["decision"] == "REVIEW"


def test_observed_certain_contact_allows_rule_engine_output():
    result = decide(observed_contact(), accepted_calibration())
    assert result["decision"] == "IN"
    assert result["court_position_m"] == pytest.approx([3, 6], abs=.01)
    assert result["uncertainty_m"] >= .02


def test_event_time_key_is_preserved_in_contact_decision():
    result = decide({**observed_contact(), "time": 1.625, "frame_index": 49}, accepted_calibration())
    assert result["decision"] == "IN"
    assert result["contact_time_seconds"] == 1.625
    assert result["contact_frame"] == 49


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_invalid_geometry_uncertainty_rejected(bad):
    with pytest.raises(ValueError):
        evaluate_contact([[1, 1]], uncertainty_m=bad)
