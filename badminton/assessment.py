"""Explain existing verdict gates without changing their evidence or outcome.

The stage statuses describe the selected review candidate. Event resolution and
the line verdict describe the entire requested set, which may contain other
unresolved candidates. Scores are never reinterpreted as probabilities here.
"""
from __future__ import annotations

import math


_CONTACT = {"FIRST_CONTACT_NOT_OBSERVED", "CONTACT_OCCLUDED_OR_MISSING", "CONTACT_EVIDENCE_UNCERTAIN"}
_IDENTITY = {"TARGET_SHUTTLE_IDENTITY_UNCERTAIN"}
_CALIBRATION = {"CALIBRATION_NOT_ACCEPTED", "MANUAL_CALIBRATION_REFERENCE_ONLY", "CALIBRATION_SUPPORT_INSUFFICIENT",
                "PRIMARY_COURT_CAMERA_UNSTABLE"}
_GEOMETRY = {"INVALID_OR_INCOMPLETE_RULE_SELECTION", "INVALID_CONTACT_GEOMETRY_OR_CALIBRATION",
             "CONTACT_POSITION_UNCERTAINTY_TOO_LARGE", "CONTACT_UNCERTAINTY_OVERLAPS_BOUNDARY",
             "PRIMARY_COURT_RULES_UNSUPPORTED", "CONTACT_OUTSIDE_PRIMARY_COURT_SCOPE"}
_TERMINAL = {"CONTACT_INSIDE_OR_TOUCHING_VALID_BOUNDARY", "ENTIRE_POSSIBLE_CONTACT_OUTSIDE"}
_INVALID_CONFIDENCE = "INVALID_CONFIDENCE_VALUE"
_PRECONDITIONS = _CONTACT | _IDENTITY | _CALIBRATION | {_INVALID_CONFIDENCE}
_STATUS_LABELS = {"found": "후보 발견", "absent": "후보 없음", "passed": "조건 충족",
                  "blocked": "확인 필요", "not_evaluated": "아직 평가하지 않음", "unknown": "근거 부족"}


def _codes(value):
    codes = value.get("reason_codes", []) if isinstance(value, dict) else []
    return list(dict.fromkeys(code for code in codes if isinstance(code, str))) if isinstance(codes, list) else []


def _point(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value))


def _stage(identifier, label, status, detail, codes=()):
    return {"id": identifier, "label": label, "status": status,
            "status_label": _STATUS_LABELS[status], "detail": detail, "reason_codes": list(codes)}


def assess_result(result, *, requested_landings=None):
    """Additive, JSON-safe diagnostics from the *actual* decision reason codes.

    No confidence thresholds, contact promotion, candidate selection, or court
    projection are duplicated here. Missing historical decisions remain unknown.
    ``requested_landings`` preserves the pipeline's identity/event filtering.
    """
    landings = result.get("landings") or []
    requested = landings if requested_landings is None else requested_landings
    selected = result.get("selected_landing") or {}
    overall = result.get("decision") or {}
    selected_decision = selected.get("decision") or (overall if len(requested) == 1 and selected else {})
    codes = _codes(selected_decision)
    overall_codes = _codes(overall)
    code_set = set(codes)
    verdict = overall.get("decision", "REVIEW")
    gate_evaluated = (bool(code_set & (_PRECONDITIONS | _GEOMETRY | _TERMINAL))
                      and "INVALID_OR_INCOMPLETE_RULE_SELECTION" not in code_set)
    stages = []

    motion = selected.get("image_motion_event") or {}
    point_kind = motion.get("point_kind", "object_center")
    if selected.get("contact_observed") is True and _point(selected.get("contact_px")):
        point_kind = "contact_observation"
        location_detail = "접촉 근거의 위치가 기록되어 있습니다. 접촉과 라인 판정 조건은 아래에서 별도로 확인합니다."
    elif point_kind == "inferred_object_center":
        location_detail = "관측이 없는 구간의 전후 경로로 물체 중심을 추정했습니다. 직접 관측한 접촉점이 아닙니다."
    elif point_kind == "observed_object_center":
        location_detail = "실제 추적 관측에서 물체 중심을 찾았습니다. 코르크의 바닥 접촉점과 다를 수 있습니다."
    else:
        location_detail = "움직임에서 검토할 후보 위치를 찾았습니다. 실제 착지 여부는 아직 별도 확인이 필요합니다."
    has_location = bool(selected) and (_point(selected.get("candidate_px")) or _point(selected.get("contact_px")))
    stages.append(_stage("candidate", "착지 후보 위치", "found" if has_location else "absent" if not landings else "unknown",
                         location_detail if has_location else "검토할 착지 후보 위치를 찾지 못했습니다."))

    definitions = [
        ("contact", "최초 바닥 접촉", _CONTACT,
         "최초 바닥 접촉 근거의 판정 조건을 충족했습니다.",
         "움직임이나 반동만으로 최초 바닥 접촉을 확정할 수 없습니다. 접촉 구간의 원본 근거가 필요합니다."),
        ("identity", "대상 셔틀 식별", _IDENTITY,
         "대상 셔틀 식별 근거의 판정 조건을 충족했습니다.",
         "후보가 해당 랠리의 셔틀인지 확정하지 못했습니다. 다른 물체나 다른 코트의 움직임일 수 있습니다."),
        ("calibration", "코트 보정 근거", _CALIBRATION,
         "코트 보정의 사전 근거 조건을 충족했습니다.",
         "라인 위치를 판정할 코트 보정 근거가 부족합니다. 코트 선, 촬영 범위와 카메라 움직임을 확인해야 합니다."),
    ]
    for identifier, label, relevant, passed, blocked in definitions:
        reasons = [code for code in codes if code in relevant]
        status = "blocked" if reasons else "unknown" if _INVALID_CONFIDENCE in code_set else "passed" if gate_evaluated else "not_evaluated"
        if _INVALID_CONFIDENCE in code_set:
            reasons.append(_INVALID_CONFIDENCE)
        detail = blocked if status == "blocked" else passed if status == "passed" else (
            "유효하지 않은 신뢰도 값이 있어 이 조건의 충족 여부를 확인할 수 없습니다." if status == "unknown"
            else "이 후보에 대한 판정 근거 평가 기록이 없습니다.")
        stages.append(_stage(identifier, label, status, detail, reasons))

    ambiguity = [code for code in overall_codes if code in {"MULTIPLE_LANDING_CANDIDATES", "NO_LANDING_CANDIDATE"}]
    event_status = "blocked" if ambiguity else "passed" if len(requested) == 1 else "unknown"
    event_detail = ("여러 후보가 남아 하나의 최초 착지 사건으로 확정하지 못했습니다. 화면의 대표 후보만으로 전체 판정을 확정하지 않습니다."
                    if "MULTIPLE_LANDING_CANDIDATES" in ambiguity else
                    "판정할 착지 후보가 없습니다." if "NO_LANDING_CANDIDATE" in ambiguity else
                    "판정 대상 후보가 하나로 좁혀졌습니다. 접촉과 신원 근거는 별도 조건입니다." if event_status == "passed" else
                    "판정 대상 사건을 하나로 좁혔다는 근거가 없습니다.")
    stages.append(_stage("event", "판정 대상 사건", event_status, event_detail, ambiguity))
    geometry_reasons = [code for code in codes if code in _GEOMETRY]
    geometry_status = "blocked" if geometry_reasons else "passed" if code_set & _TERMINAL else "not_evaluated"
    if "CONTACT_UNCERTAINTY_OVERLAPS_BOUNDARY" in geometry_reasons:
        geometry_detail = "접촉 위치의 불확실성 범위가 라인 경계와 겹쳐 IN/OUT을 구분할 수 없습니다."
    elif "CONTACT_OUTSIDE_PRIMARY_COURT_SCOPE" in geometry_reasons:
        geometry_detail = "접촉점이 선택한 주 코트의 확인 가능한 범위 밖에 있습니다. 옆 코트나 보이지 않는 먼 영역을 OUT으로 판정하지 않습니다."
    elif "PRIMARY_COURT_RULES_UNSUPPORTED" in geometry_reasons:
        geometry_detail = "선택한 코트 선과 판정 규칙이 다르거나 서비스 경계가 확인되지 않았습니다. 선택한 단식·복식 규칙으로 다시 분석해야 합니다."
    elif geometry_status == "blocked":
        geometry_detail = "규칙 선택, 접촉 좌표, 코트 변환 또는 위치 오차 조건을 충족하지 못했습니다."
    elif geometry_status == "passed":
        geometry_detail = "접촉 위치와 오차 범위를 라인 경계와 비교했습니다."
    else:
        geometry_detail = "접촉·대상 식별·코트 보정 조건이 확인되면 라인과 위치 오차를 비교합니다."
    stages.append(_stage("geometry", "라인 경계와 위치 오차", geometry_status, geometry_detail, geometry_reasons))

    candidate_summary = "착지 후보 발견" if has_location else "착지 후보 없음" if not landings else "후보 위치 확인 필요"
    return {"schema_version": 1, "scope": "selected_candidate_evidence_and_requested_set_verdict",
            "summary": f"{candidate_summary} · " + ("라인 판정 보류" if verdict == "REVIEW" else f"라인 판정 {verdict}"),
            "line_verdict": verdict, "candidate_count": len(landings), "requested_candidate_count": len(requested),
            "selected_event_id": selected.get("event_id"), "localization_kind": point_kind if has_location else None,
            "stages": stages, "blocking_stages": [stage["id"] for stage in stages if stage["status"] == "blocked"],
            "reason_codes": overall_codes, "selected_reason_codes": codes,
            "changes_decision": False}
