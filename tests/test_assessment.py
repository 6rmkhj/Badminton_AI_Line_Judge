from copy import deepcopy
import json

import pytest

from badminton.assessment import assess_result
from badminton.decision import decide
from badminton.pipeline import aggregate_decision


def calibration(**changes):
    return {"status": "ok", "confidence": .95, "method": "colour_hough_court_template",
            "homography": [[.01, 0, 0], [0, .01, 0], [0, 0, 1]], "line_fit_error_px": .5, **changes}


def landing(**changes):
    return {"event_id": "event_000", "candidate_px": [300, 600], "contact_observed": True,
            "contact_px": [300, 600], "confidence": .95, "identity_confidence": .95,
            "contact_radius_px": 1, "uncertainty_px": 1, **changes}


def assessed(candidate=None, plane=None, rules=None):
    candidate = landing() if candidate is None else candidate
    plane = calibration() if plane is None else plane
    candidate = {**candidate, "decision": decide(candidate, plane, rules)}
    result = {"landings": [candidate], "selected_landing": candidate, "calibration": plane,
              "decision": candidate["decision"]}
    return assess_result(result), result


def stages(assessment):
    return {stage["id"]: stage for stage in assessment["stages"]}


def test_found_object_center_is_distinct_from_unobserved_contact_and_identity():
    assessment, result = assessed(landing(contact_observed=False, contact_px=None, identity_confidence=.25,
        image_motion_event={"point_kind": "observed_object_center"}), calibration(status="review"))
    rows = stages(assessment)
    assert assessment["line_verdict"] == "REVIEW"
    assert assessment["localization_kind"] == "observed_object_center"
    assert rows["candidate"]["status"] == "found"
    assert set(assessment["blocking_stages"]) == {"contact", "identity", "calibration"}
    assert rows["geometry"]["status"] == "not_evaluated"
    assert rows["event"]["status"] == "passed"
    assert rows["contact"]["reason_codes"] == ["FIRST_CONTACT_NOT_OBSERVED"]
    assert assessment["reason_codes"] == result["decision"]["reason_codes"]


def test_gap_intersection_remains_an_inferred_location_without_contact():
    assessment, _ = assessed(landing(contact_observed=False, contact_px=None, occluded=True,
        image_motion_event={"point_kind": "inferred_object_center"}))
    assert assessment["localization_kind"] == "inferred_object_center"
    assert stages(assessment)["candidate"]["status"] == "found"
    assert set(stages(assessment)["contact"]["reason_codes"]) == {"FIRST_CONTACT_NOT_OBSERVED", "CONTACT_OCCLUDED_OR_MISSING"}


@pytest.mark.parametrize("candidate, expected", [(landing(), "IN"), (landing(contact_px=[-30, 600]), "OUT")])
def test_valid_contact_uses_existing_geometry_verdict(candidate, expected):
    assessment, _ = assessed(candidate)
    assert assessment["line_verdict"] == expected
    assert assessment["localization_kind"] == "contact_observation"
    assert assessment["blocking_stages"] == []
    assert all(row["status"] == "passed" for key, row in stages(assessment).items() if key != "candidate")


def test_boundary_uncertainty_is_distinct_from_missing_contact():
    assessment, _ = assessed(landing(contact_px=[0, 600]))
    assert assessment["blocking_stages"] == ["geometry"]
    assert stages(assessment)["contact"]["status"] == "passed"
    assert stages(assessment)["geometry"]["reason_codes"] == ["CONTACT_UNCERTAINTY_OVERLAPS_BOUNDARY"]


def test_invalid_confidence_does_not_guess_which_independent_gate_passed():
    assessment, _ = assessed(landing(confidence=float("nan")))
    for identifier in ("contact", "identity", "calibration"):
        assert stages(assessment)[identifier]["status"] == "unknown"
        assert stages(assessment)[identifier]["reason_codes"] == ["INVALID_CONFIDENCE_VALUE"]
    assert stages(assessment)["geometry"]["status"] == "not_evaluated"
    json.dumps(assessment, allow_nan=False)


def test_incomplete_rules_do_not_claim_unexecuted_evidence_gates_passed():
    assessment, _ = assessed(rules={"phase": "service"})
    assert assessment["blocking_stages"] == ["geometry"]
    for identifier in ("contact", "identity", "calibration"):
        assert stages(assessment)[identifier]["status"] == "not_evaluated"


def test_no_candidate_and_legacy_missing_decisions_remain_explicit():
    empty = assess_result({"landings": [], "decision": aggregate_decision([])})
    assert stages(empty)["candidate"]["status"] == "absent"
    assert stages(empty)["event"]["reason_codes"] == ["NO_LANDING_CANDIDATE"]
    legacy = assess_result({"landings": [landing()], "selected_landing": landing(), "decision": {"decision": "REVIEW"}})
    assert stages(legacy)["candidate"]["status"] == "found"
    assert stages(legacy)["contact"]["status"] == "not_evaluated"


def test_aggregate_ambiguity_does_not_overwrite_selected_contact_evidence():
    _assessment, result = assessed()
    other = landing(event_id="event_001", contact_observed=False)
    other["decision"] = decide(other, calibration())
    result["landings"].append(other)
    result["decision"] = aggregate_decision([candidate["decision"] for candidate in result["landings"]])
    before = deepcopy(result)
    assessment = assess_result(result)
    assert assessment["line_verdict"] == "REVIEW"
    assert assessment["blocking_stages"] == ["event"]
    assert stages(assessment)["contact"]["status"] == "passed"
    assert "FIRST_CONTACT_NOT_OBSERVED" in assessment["reason_codes"]
    assert "FIRST_CONTACT_NOT_OBSERVED" not in assessment["selected_reason_codes"]
    assert result == before


def test_requested_set_count_preserves_identity_filtering():
    _assessment, result = assessed()
    result["landings"].append(landing(event_id="excluded"))
    assessment = assess_result(result, requested_landings=[result["selected_landing"]])
    assert assessment["candidate_count"] == 2
    assert assessment["requested_candidate_count"] == 1
    assert stages(assessment)["event"]["status"] == "passed"
    assert assessment["line_verdict"] == "IN"


def test_candidate_reason_codes_cannot_override_actual_decision():
    assessment, _ = assessed(landing(reason_codes=["FIRST_CONTACT_NOT_OBSERVED", "CALIBRATION_NOT_CONFIRMED"]))
    assert assessment["blocking_stages"] == []
    assert assessment["line_verdict"] == "IN"
