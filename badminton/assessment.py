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


_ANALYSIS_FAILURES = {
    "INVALID_VIDEO_ESTIMATE": ("input", "분석 근거 일치 확인",
        "영상·경기 규칙·프레임 범위와 추정 근거가 일치하지 않아 이 추정을 사용할 수 없습니다.",
        "현재 원본과 선택한 단식·복식 규칙으로 새 분석을 실행하세요."),
    "VIDEO_ESTIMATE_PROCESSING_FAILED": ("processing", "원본 근거 처리",
        "원본 착지 범위를 계산하는 중 오류가 발생했습니다.",
        "실행 오류와 분석 로그를 확인한 뒤 같은 원본으로 다시 실행하세요."),
    "VIDEO_ESTIMATE_DISABLED": ("processing", "영상 추정 설정",
        "영상 추정 계산이 꺼져 있어 직접 접촉 근거만 평가했습니다.",
        "영상 추정을 켜고 원본 영상으로 새 분석을 실행하세요."),
    "VIDEO_ESTIMATE_RULES_UNSUPPORTED": ("calibration", "선택한 경기 경계",
        "관측한 코트 경계와 선택한 경기 규칙이 일치하지 않습니다.",
        "단식·복식 선택과 촬영한 코트를 확인하고 같은 규칙으로 다시 분석하세요."),
    "VIDEO_ESTIMATE_CALIBRATION_UNSUPPORTED": ("calibration", "관측 코트 경계",
        "접촉 가능 영역과 비교할 주 코트 경계가 충분히 확인되지 않았습니다.",
        "판정할 코트의 양쪽 앞 모서리와 해당 라인이 보이는 원본 구간을 사용하세요."),
    "VIDEO_ESTIMATE_CAMERA_UNSTABLE": ("calibration", "카메라 고정",
        "착지 구간과 코트 보정 구간 사이에 카메라 고정을 확인하지 못했습니다.",
        "카메라가 움직이지 않은 구간으로 다시 분석하세요."),
    "VIDEO_ESTIMATE_NO_UNIQUE_EVENT": ("event", "판정할 착지 경로",
        "주 코트에서 하나의 착지 경로로 좁힐 관측이 부족합니다.",
        "마지막 타격부터 착지 뒤까지 이어지는 원본 구간을 사용하고, 여러 랠리가 있으면 판정할 구간으로 나누세요."),
    "VIDEO_ESTIMATE_NATIVE_SUPPORT_MISSING": ("contact", "첫 착지 가능 구간",
        "원본 프레임에서 착지 전후의 연속된 물체 영역을 확보하지 못했습니다.",
        "착지 전후 원본을 확인하고, 대상이 가려지거나 작게 보이면 같은 사건을 더 선명하게 담은 시점을 사용하세요."),
    "VIDEO_ESTIMATE_INVALID_ENVELOPE": ("geometry", "접촉 가능 영역",
        "접촉 가능한 위치와 오차 범위를 유효하게 계산하지 못했습니다.",
        "착지 전후 프레임 누락과 영상 해상도를 확인하고 원본으로 다시 분석하세요."),
    "VIDEO_ESTIMATE_BOUNDARY_AMBIGUOUS": ("geometry", "라인 경계와 위치 오차",
        "가능한 접촉 영역이 라인 양쪽에 걸쳐 IN·OUT을 구분할 수 없습니다.",
        "이 접촉 구간을 더 선명하게 담은 시점으로 위치 오차를 좁혀야 합니다."),
    "VIDEO_ESTIMATE_OUTSIDE_SUPPORTED_SCOPE": ("geometry", "관측 경계의 판정 범위",
        "접촉 가능 영역이 이 시점에서 확인한 주 코트 범위를 벗어납니다.",
        "해당 착지와 가까운 경기 경계가 함께 보이는 시점을 사용하세요."),
}

_NATIVE_FAILURE_DETAILS = {
    "NATIVE_ESTIMATE_OCCLUDER_OVERLAP": ("착지 후보와 선수 등 가리는 물체가 겹칩니다.", "가림이 적은 반대 시점의 같은 구간을 확인하세요."),
    "NATIVE_ESTIMATE_TEMPORAL_WINDOW_INCOMPLETE": ("클립에 착지 전후 구간이 충분히 포함되지 않았습니다.", "착지 전후가 모두 포함되도록 원본 구간을 늘려 다시 분석하세요."),
    "NATIVE_ESTIMATE_INCOMING_GAP": ("착지 직전 경로에 관측 공백이 있습니다.", "누락 프레임이 없는 원본이나 공백 구간이 보이는 반대 시점을 사용하세요."),
    "NATIVE_ESTIMATE_SOURCE_TIMESTAMP_GAP": ("착지 부근 원본 프레임의 시간 간격이 끊겨 있습니다.", "누락 프레임이 없는 원본이나 같은 구간을 연속 촬영한 반대 시점을 사용하세요."),
    "NATIVE_ESTIMATE_NO_VISIBLE_PEAK": ("원본에서 착지 부근 물체를 연속으로 관측하지 못했습니다.", "착지 부근의 원본 화질·가림을 확인하고 더 선명한 시점을 사용하세요."),
    "NATIVE_ESTIMATE_SECOND_OBJECT_SUPPORT": ("접촉 구간에 별도의 움직이는 물체 경로도 남아 있습니다.", "같은 랠리의 마지막 타격과 이어지는 경로를 반대 시점에서 대조하세요."),
    "NATIVE_ESTIMATE_MULTIPLE_OR_UNRESOLVED_SOURCE_COMPONENTS": ("원본 접촉 구간의 여러 물체 영역을 한 셔틀 경로로 연결하지 못했습니다.", "대상 셔틀이 분리되어 보이는 시점이나 원본 구간을 사용하세요."),
    "NATIVE_ESTIMATE_NATIVE_REVERSAL_UNSUPPORTED": ("원본 관측에서 착지 전후의 방향 전환을 뒷받침하지 못했습니다.", "하강과 착지 뒤 움직임이 함께 포함된 원본을 확인하세요."),
}


def assess_analysis(view, *, direct_qualified=False, estimate_valid=False):
    """Describe the actual analysis path selected by pair validation.

    The caller owns source/rules/geometry validation and supplies its results.
    No raw confidence or ``eligible`` flag can promote a stage here. Strict
    physical-contact diagnostics remain separately available as ``assessment``.
    Unresolved estimates show their actual stopping stage, never four unrelated
    direct-contact blockers inherited from a different representative candidate.
    """
    estimate = view.get("video_estimate")
    estimate = estimate if isinstance(estimate, dict) else {}
    direct = view.get("direct_decision")
    label = direct if direct_qualified and direct in ("IN", "OUT") else (
        estimate.get("decision") if estimate_valid and estimate.get("decision") in ("IN", "OUT") else "REVIEW")
    kind = "verified" if direct_qualified and label != "REVIEW" else "video_estimate" if label != "REVIEW" else "unresolved"
    base = {"schema_version": 1, "title": "분석 판정 근거", "line_verdict": label,
            "decision_kind": kind, "changes_decision": False, "stages": [], "reason_codes": [],
            "blocking_stages": [], "next_action": None,
            "scope_label": "실제로 사용한 분석 근거를 표시합니다. 직접 접촉 관측 조건은 상세에서 따로 확인할 수 있습니다."}
    if kind == "verified":
        base.update(summary=f"직접 근거 판정 {label}",
                    explanation="이 시점의 유효한 접촉·라인 근거를 사용했습니다.",
                    stages=[_stage("verified", "접촉·라인 판정 근거", "passed",
                                   "직접 판정 경로에서 접촉·대상 식별·경기 경계 조건을 충족했습니다.")])
        return base
    if kind == "video_estimate":
        times = estimate.get("time_interval")
        interval = f"{times[0]:.3f}–{times[1]:.3f}초" if _point(times) and times[1] >= times[0] >= 0 else "착지 전후 구간"
        members = estimate.get("member_event_ids")
        count = len(members) if isinstance(members, list) else 0
        geometry = estimate.get("geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        margin = geometry.get("minimum_clearance_px")
        margin_text = (f" 경계·위치 오차를 반영한 최소 여유는 {margin:.2f}픽셀입니다."
                       if isinstance(margin, (int, float)) and not isinstance(margin, bool) and math.isfinite(margin) and margin >= 0 else "")
        stages = [
            _stage("contact", "첫 착지 가능 구간", "passed",
                   f"{interval}의 원본 물체 영역과 움직임으로 가능한 접촉 범위를 계산했습니다. 직접 접촉 순간을 관측했다는 뜻은 아닙니다."),
            _stage("identity", "주 코트 셔틀 경로 연결", "passed",
                   "주 코트에서 관측한 접근 경로와 착지 후보를 연결했습니다. 대상 식별은 이 영상 경로에 근거한 추정입니다."),
            _stage("event", "판정 대상 사건 통합", "passed",
                   f"같은 관측 경로의 대안 후보 {count}개를 하나의 착지 사건으로 묶어 비교했습니다."),
            _stage("geometry", "라인 경계와 위치 오차", "passed",
                   ("오차를 포함한 접촉 가능 영역 전체가 확인된 코트 안쪽입니다." if label == "IN"
                    else "오차를 포함한 접촉 가능 영역 전체가 관측된 경기 경계 밖입니다.") + margin_text),
        ]
        for stage, status in zip(stages, ("범위 계산", "경로 연결", "한 사건으로 통합", "비교 완료")):
            stage["status_label"] = status
        base.update(summary=f"영상 추정 {label} · 원본 근거와 경계 비교 완료", stages=stages,
                    explanation="원본의 연속 관측과 접촉 가능 영역에 근거한 영상 추정입니다.",
                    reason_codes=_codes(estimate))
        return base
    codes = _codes(estimate)
    if view.get("status") == "failed":
        identifier, title, detail, action = ("processing", "영상 분석 실행",
            "이 시점의 분석이 완료되지 않았습니다. 아래 실행 오류와 로그를 확인하세요.",
            "실행 오류를 해결한 뒤 원본 영상을 다시 분석하세요.")
        codes = ["VIEW_ANALYSIS_FAILED"]
    else:
        if estimate.get("eligible") is True or estimate.get("decision") in ("IN", "OUT"):
            codes = list(dict.fromkeys(["INVALID_VIDEO_ESTIMATE", *codes]))
        entry = next((_ANALYSIS_FAILURES[code] for code in codes if code in _ANALYSIS_FAILURES), None)
        if entry is not None:
            identifier, title, detail, action = entry
            native = estimate.get("native_evidence")
            if identifier == "contact" and isinstance(native, dict):
                correction = next((_NATIVE_FAILURE_DETAILS[code] for code in _codes(native)
                                   if code in _NATIVE_FAILURE_DETAILS), None)
                if correction:
                    detail, action = correction
        else:
            identifier, title, detail, action = ("analysis", "현재 분석 근거",
                "이 결과에는 사용할 수 있는 직접 판정 또는 원본 영상 추정이 없습니다.",
                "최신 분석기로 기존 검수 재사용을 끄고 원본 영상을 새로 분석하세요. 상세 근거와 실제 판정 보류 사유도 확인할 수 있습니다.")
    stage = _stage(identifier, title, "blocked", detail, codes)
    base.update(summary="영상 판정 보류 · " + title, explanation=detail,
                stages=[stage], blocking_stages=[identifier], reason_codes=codes, next_action=action)
    return base
