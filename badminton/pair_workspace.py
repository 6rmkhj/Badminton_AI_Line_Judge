"""Persisted A/B jobs; link source-bound verdicts without fusing camera pixels."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import threading
import time
import traceback
from urllib.parse import quote
import uuid

MAX_UPLOAD_BYTES = 1024 ** 3
MAX_FRAMES = 3000
MAX_ACTIVE_JOBS = 4
LABELS = {"IN", "OUT", "REVIEW"}
RULES = {"game": "doubles", "phase": "rally"}
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,140}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
LOGGER = logging.getLogger(__name__)
TERMINAL = {"completed", "completed_with_errors", "failed", "interrupted"}
ANALYSIS_STAGES = {
    "prepare": (1, "영상 준비"), "calibration": (2, "코트 확인"),
    "detection": (3, "셔틀콕 후보 탐지"), "tracking": (4, "궤적 연결"),
    "contact": (5, "착지·라인 확인"), "evidence": (6, "근거 영상 생성"),
    "save": (7, "결과 저장"),
}
REASONS = {
    "SAME_EVENT_UNCONFIRMED": "두 영상이 같은 랠리인지 확인되지 않았습니다.",
    "KNOWN_DIFFERENT_RALLY": "기존 동작 검수에서 서로 다른 랠리로 확인된 영상입니다.",
    "DUPLICATE_SOURCE": "A와 B에 같은 영상 내용이 입력되었습니다. 서로 다른 시점이 필요합니다.",
    "CONFLICTING_VIEW_DECISIONS": "A와 B의 유효한 IN/OUT 판정이 충돌합니다.",
    "VIEW_ANALYSIS_FAILED": "한쪽 영상 처리가 실패했습니다. 완료된 시점의 유효한 판정은 유지합니다.",
    "NO_QUALIFIED_CONTACT": "양쪽 모두 최초 접촉과 라인 관계를 확정할 근거가 부족합니다.",
    "QUALIFIED_VIEWS_AGREE": "같은 랠리의 두 시점에서 판정 근거가 일치합니다.",
    "VERDICT_FROM_QUALIFIED_VIEW": "접촉 근거가 충분한 시점의 유효한 판정을 최종 결과로 사용했습니다.",
    "RULES_MISMATCH": "시점별 판정에 적용된 경기 규칙이 다릅니다.",
    "SOURCE_CHANGED": "검수 당시 원본과 현재 영상의 내용이 다릅니다.",
    "STALE_REVIEW": "저장된 검수 기록의 출처 또는 근거를 검증하지 못했습니다.",
    "VIDEO_ESTIMATE_FROM_VIEW": "원본 영상의 착지 가능 영역 전체가 같은 결과인 시점의 영상 추정을 사용했습니다. 직접 관측한 접촉 판정과 구분합니다.",
    "VIDEO_ESTIMATES_AGREE": "같은 랠리의 두 시점에서 착지 가능 영역을 이용한 영상 추정이 일치합니다.",
    "CONFLICTING_VIDEO_ESTIMATES": "A와 B의 유효한 영상 추정 IN/OUT이 충돌합니다.",
    "VERIFIED_ESTIMATE_CONFLICT": "같은 사건의 확인된 판정과 영상 추정이 서로 달라 검토가 필요합니다.",
    "UNLINKED_ESTIMATE_CONFLICT": "연결이 미확인인 다른 시점의 영상 추정은 확인된 판정과 다릅니다. 확인된 시점의 결과를 유지합니다.",
    "INVALID_VIDEO_ESTIMATE": "영상 추정의 출처·규칙·프레임 또는 위치 범위를 검증하지 못했습니다.",
}


class PairWorkspaceError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def parse_pair_rules(value):
    """Normalize the supported game selection; service boxes remain explicit elsewhere."""
    if (not isinstance(value, dict) or set(value) - {"game", "phase"}
            or not isinstance(value.get("game", "doubles"), str)
            or value.get("game", "doubles") not in {"singles", "doubles"}
            or value.get("phase", "rally") != "rally"):
        raise PairWorkspaceError(400, "경기 종류는 단식 또는 복식, 판정 구간은 랠리를 선택하세요.")
    return {"game": value.get("game", "doubles"), "phase": "rally"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _failure_details(error, stage):
    """Keep the actual OS/encoder error visible instead of just its class name."""
    detail = " ".join(str(error).split())[:600]
    error_type = type(error).__name__
    label = ANALYSIS_STAGES.get(stage, (None, "영상 분석"))[1]
    message = f"{label} 실패: {error_type}" + (f" · {detail}" if detail else "")
    diagnostics = {"stage": stage, "error_type": error_type, "message": detail}
    for name in ("errno", "winerror"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            diagnostics[name] = value
    return message, diagnostics


def _seconds_between(start, end):
    try:
        return max(0., (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except (TypeError, ValueError):
        return 0.


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return deepcopy(default)


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _safe(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise PairWorkspaceError(403, "허용되지 않는 파일 경로입니다.")
    root = Path(root).resolve()
    candidate = root
    for part in relative.split("/"):
        candidate = candidate / part
        if candidate.is_symlink():
            raise PairWorkspaceError(403, "심볼릭 링크는 사용할 수 없습니다.")
    if not candidate.resolve().is_relative_to(root):
        raise PairWorkspaceError(403, "허용되지 않는 파일 경로입니다.")
    return candidate


def _identifier(value):
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise PairWorkspaceError(400, "잘못된 영상 또는 작업 ID입니다.")
    return value


def _finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _review_region(value):
    """Validate an optional displayed-video rectangle before any job is created."""
    if value is None:
        return None
    fields = ("x", "y", "width", "height")
    if (not isinstance(value, dict) or set(value) != set(fields)
            or not all(_finite(value[field]) for field in fields)):
        raise PairWorkspaceError(400, "분석 영역은 x, y, width, height 네 개의 유한한 숫자여야 합니다.")
    x, y, width, height = (float(value[field]) for field in fields)
    if (not 0 <= x < 1 or not 0 <= y < 1 or not 0 < width <= 1 or not 0 < height <= 1
            or x + width > 1 or y + height > 1):
        raise PairWorkspaceError(400, "분석 영역은 영상 안의 0~1 좌표이며 너비와 높이는 0보다 커야 합니다.")
    return dict(zip(fields, (x, y, width, height)))


def _manual_review_regions(region, metadata, analysis_width=1280):
    """Map display-oriented coordinates to the same even dimensions as prepare_preview.

    Round outward to retain small edge selections within at least one analysis
    pixel. This is a search prior across the full source timeline, not a court
    calibration, object identity, or contact annotation.
    """
    region = _review_region(region)
    if region is None:
        return []
    width = min(int(analysis_width), metadata["width"])
    if width < 160:
        raise PairWorkspaceError(400, "분석 영상 너비는 160픽셀 이상이어야 합니다.")
    width -= width % 2
    height = max(2, round(metadata["height"] * width / metadata["width"] / 2) * 2)
    def edge(value, rounding):
        # Normalizing an integer pixel and multiplying back can add a few ULPs;
        # those are not a requested extra row/column of search area.
        nearest = round(value)
        return nearest if abs(value - nearest) <= 1e-9 else rounding(value)
    left = min(width - 1, edge(region["x"] * width, math.floor))
    top = min(height - 1, edge(region["y"] * height, math.floor))
    right = min(width, max(left + 1, edge((region["x"] + region["width"]) * width, math.ceil)))
    bottom = min(height, max(top + 1, edge((region["y"] + region["height"]) * height, math.ceil)))
    return [{"frame_start": 0, "frame_end": metadata["frame_count"] - 1,
             "bbox": [left, top, right - left, bottom - top]}]


def _landing_candidate(result):
    """Expose a useful image-space hypothesis without promoting it to contact."""
    landing, analysis = result.get("selected_landing") or {}, result.get("analysis") or {}
    if not isinstance(landing, dict) or not isinstance(analysis, dict) or landing.get("contact_observed") is True:
        return None
    point, stamp, frame = landing.get("candidate_px"), landing.get("time"), landing.get("frame_index")
    width, height, count = analysis.get("width"), analysis.get("height"), analysis.get("frame_count")
    if (not isinstance(point, (list, tuple)) or len(point) != 2 or not all(_finite(v) for v in point)
            or not _finite(width) or not _finite(height) or width <= 0 or height <= 0
            or not 0 <= point[0] < width or not 0 <= point[1] < height
            or not _finite(stamp) or stamp < 0 or not isinstance(frame, int) or isinstance(frame, bool)
            or frame < 0 or not isinstance(count, int) or isinstance(count, bool) or not frame < count):
        return None
    motion = landing.get("image_motion_event") or {}
    if not isinstance(motion, dict):
        motion = {}
    inferred = (motion.get("kind") == "occluded_descent_and_rebound"
                or motion.get("point_kind") == "inferred_object_center"
                or landing.get("point_kind") == "inferred_object_center")
    kind = motion.get("kind") if motion.get("kind") in {"descent_and_rebound", "occluded_descent_and_rebound"} else "motion_candidate"
    candidate = {"status": "approximate", "kind": kind,
                 "time_seconds": float(stamp), "frame_index": frame, "point_px": list(point),
                 "analysis_size": {"width": width, "height": height},
                 "point_kind": "inferred_object_center" if inferred else "object_center",
                 "contact_observed": False, "inference_only": True}
    indices, times = motion.get("frame_interval"), motion.get("time_interval")
    if (isinstance(indices, (list, tuple)) and len(indices) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in indices)
            and 0 <= indices[0] <= frame <= indices[1] < count):
        candidate["frame_interval"] = list(indices)
    if (isinstance(times, (list, tuple)) and len(times) == 2 and all(_finite(v) for v in times)
            and 0 <= times[0] <= stamp <= times[1]):
        candidate["time_interval"] = list(times)
    return candidate


def _calibration_summary(result):
    """Describe selected local boundaries separately from metric court fitting."""
    calibration = result.get("calibration")
    if not isinstance(calibration, dict):
        return None
    primary = calibration.get("primary_court")
    selected = isinstance(primary, dict) and primary.get("status") == "selected"
    game = (primary.get("game", calibration.get("game", "doubles")) if isinstance(primary, dict)
            else calibration.get("game", "doubles"))
    game_label = "단식" if game == "singles" else "복식"
    segments = (primary.get("line_segments_px", []) if selected else []
                if isinstance(primary, dict) else calibration.get("line_segments_px"))
    if not isinstance(segments, list):
        segments = ([] if isinstance(primary, dict) else
                    [region.get("line_segment_px") for region in calibration.get("supported_line_regions_px", [])
                     if isinstance(region, dict)])
    count = sum(isinstance(segment, (list, tuple)) and len(segment) == 4
                and all(_finite(value) for value in segment) for segment in segments)
    accepted = calibration.get("status") == "ok" and calibration.get("manual_reference") is not True
    stability = calibration.get("camera_stability") or {}
    moved = "CAMERA_MOVED_FROM_CALIBRATION_REFERENCE" in stability.get("reason_codes", [])
    stable = stability.get("status") == "ok" and stability.get("all_frames_sampled") is True and not moved
    detail = (f"직선 후보 {count}개를 찾았습니다. " if count else "직선 후보를 찾지 못했습니다. ")
    detail += ("전체 코트 규격과 선 배치가 맞아 좌표 보정 조건을 충족했습니다." if accepted else
               "전체 코트 좌표 보정은 확인되지 않아 라인의 역할과 IN/OUT 경계를 확정할 수 없습니다.")
    legend = "하늘색: 탐지한 직선 후보 · 노란색: 전체 코트 보정이 확인된 외곽선"
    if isinstance(primary, dict):
        if selected:
            detail = "앞쪽 두 모서리가 보이는 주 코트 하나를 선택했습니다. 선택한 코트의 관측 경계만 표시합니다. "
            detail += ("단식 안쪽 사이드라인을 판정 경계로 사용합니다. " if game == "singles" else
                       "복식 바깥쪽 사이드라인을 판정 경계로 사용합니다. ")
            detail += ("옅은 초록 영역은 관측 경계로 확인한 안쪽 판정 범위입니다. "
                       if primary.get("region_supported") is True else "안쪽 판정 범위는 추가 확인이 필요합니다. ")
            detail += ("접촉 위치와 셔틀 식별이 확인되면 지원되는 범위에서 IN/OUT을 판정합니다." if stable else
                       "영상 전체의 카메라 고정 여부가 확인되지 않아 이 경계로 IN/OUT을 확정하지 않습니다.")
            legend = f"초록 선: {game_label} 판정 경계 · 초록 원: 앞쪽 두 모서리 · 하늘색: 선택한 코트의 관측선"
            if primary.get("region_supported") is True:
                legend += " · 옅은 초록: 확인된 안쪽 범위 (색 면의 닫힘은 외곽선이 아님)"
        else:
            detail = "앞쪽 두 모서리에 대응하는 주 코트를 확정하지 못했습니다. 다른 코트의 선을 판정 경계로 표시하지 않습니다."
            legend = "주 코트 선택이 확인되면 해당 코트의 관측 경계만 표시합니다."
    if moved:
        detail += " 카메라 이동이 감지되어 고정 라인 표시는 진단 이미지에서만 제공합니다."
    return {"line_detection_status": "detected" if count else "not_detected", "detected_line_count": count,
            "game": game,
            "court_geometry_status": "supported" if accepted else "unconfirmed", "detail": detail,
            "primary_court_status": primary.get("status") if isinstance(primary, dict) else None,
            "judgment_scope": primary.get("scope") if selected else None,
            "local_region_supported": selected and primary.get("region_supported") is True and stable,
            "overlay_legend": legend,
            "camera_moved": moved, "metric_accuracy_validated": calibration.get("metric_accuracy_validated") is True,
            "reason_codes": [code for code in calibration.get("reason_codes", []) if isinstance(code, str)]}


def _calibration_preview(output, artifacts, prefix):
    relative = artifacts.get("calibration_overlay")
    if (not isinstance(relative, str) or not relative.startswith("evidence/")
            or Path(relative).suffix.lower() not in {".jpg", ".png"}):
        return None
    try:
        path = _safe(output, relative)
        if not path.is_file() or path.stat().st_size == 0:
            return None
    except (PairWorkspaceError, OSError):
        return None
    return prefix + quote(relative, safe="/")


def _trajectory_replay(output, artifacts, prefix):
    """Expose only an existing, local MP4 and bounded source-timeline metadata."""
    clip = artifacts.get("annotated_clip")
    if not isinstance(clip, dict):
        return None, None
    relative = clip.get("path")
    if not isinstance(relative, str) or not relative.startswith("evidence/") or Path(relative).suffix.lower() != ".mp4":
        return None, None
    try:
        path = _safe(output, relative)
        if not path.is_file() or path.stat().st_size == 0:
            return None, None
    except (PairWorkspaceError, OSError):
        return None, None
    info = {}
    origin = clip.get("source_time_origin")
    if _finite(origin) and origin >= 0:
        info["source_time_origin"] = origin
    for field in ("source_start_frame", "source_end_frame", "frame_count"):
        value = clip.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            info[field] = value
    return prefix + quote(relative, safe="/"), info


def _valid_video_estimate(estimate, rules, source_sha256):
    """Validate a source-bound *estimate*, never a replacement contact observation.

    The native estimator owns evidence and geometry evaluation. This transport
    boundary checks that a complete result belongs to the current source/rules
    and fits trusted frame/image bounds before it can enter pair aggregation.
    """
    if (not isinstance(source_sha256, str) or not HASH.fullmatch(source_sha256)
            or not isinstance(estimate, dict) or type(estimate.get("schema_version")) is not int
            or estimate.get("schema_version") != 1
            or estimate.get("decision") not in ("IN", "OUT") or estimate.get("eligible") is not True
            or estimate.get("decision_kind") != "video_estimate"
            or estimate.get("source") != "native_temporal_landing_envelope"
            or estimate.get("rules") != rules or estimate.get("source_sha256") != source_sha256
            or estimate.get("contact_observed") is not False
            or estimate.get("target_identity_verified") is not False
            or not isinstance(estimate.get("native_evidence"), dict)
            or estimate["native_evidence"].get("status") != "supported"
            or not isinstance(estimate.get("family_resolution"), dict)
            or estimate["family_resolution"].get("status") != "selected"
            or not isinstance(estimate.get("geometry"), dict)
            or estimate["geometry"].get("status") != "resolved"):
        return False
    family, members = estimate.get("selected_family_id"), estimate.get("member_event_ids")
    codes = estimate.get("reason_codes")
    if (not isinstance(family, str) or not family
            or not isinstance(codes, list) or any(not isinstance(code, str) for code in codes)
            or not isinstance(members, list) or not members
            or any(not isinstance(value, str) or not value for value in members)
            or len(set(members)) != len(members)):
        return False
    size = estimate.get("analysis_size")
    if not isinstance(size, dict):
        return False
    width, height, count = (size.get(key) for key in ("width", "height", "frame_count"))
    if (not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in (width, height, count))):
        return False
    frames, times = estimate.get("frame_interval"), estimate.get("time_interval")
    bounds, frame_times = estimate.get("time_bounds"), estimate.get("frame_time_interval")
    if (not isinstance(frames, (list, tuple)) or len(frames) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in frames)
            or not 0 <= frames[0] <= frames[1] < count):
        return False
    if any(not isinstance(values, (list, tuple)) or len(values) != 2
           or not all(_finite(value) for value in values) for values in (times, bounds, frame_times)):
        return False
    if (bounds[0] < 0 or bounds[1] < bounds[0] or frame_times[1] < frame_times[0] or times[1] < times[0]
            or frame_times[0] < bounds[0] - 1e-6 or frame_times[1] > bounds[1] + 1e-6
            or times[0] < frame_times[0] - 1e-6 or times[1] > frame_times[1] + 1e-6):
        return False
    error, polygon = estimate.get("uncertainty_px"), estimate.get("possible_contact_polygon_px")
    if (not _finite(error) or not 0 <= error <= math.hypot(width, height)
            or not isinstance(polygon, (list, tuple)) or not 3 <= len(polygon) <= 64):
        return False
    if any(not isinstance(point, (list, tuple)) or len(point) != 2
           or not all(_finite(value) for value in point)
           or not (0 <= point[0] < width and 0 <= point[1] < height) for point in polygon):
        return False
    # Convex possible-position regions must have area and cannot self-intersect.
    crosses = [(polygon[(i+1) % len(polygon)][0] - polygon[i][0]) * (point[1] - polygon[i][1])
               - (polygon[(i+1) % len(polygon)][1] - polygon[i][1]) * (point[0] - polygon[i][0])
               for i in range(len(polygon)) for point in polygon]
    if not (any(value > 1e-8 for value in crosses) and all(value >= -1e-8 for value in crosses)
            or any(value < -1e-8 for value in crosses) and all(value <= 1e-8 for value in crosses)):
        return False
    try:
        json.dumps(estimate, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _video_estimate_summary(result, output, prefix):
    """Expose bounded native evidence with a safe local image URL."""
    raw = result.get("video_estimate")
    if not isinstance(raw, dict):
        return None
    fields = ("schema_version", "decision", "eligible", "decision_kind", "source", "rules",
              "contact_observed", "target_identity_verified", "reason_codes", "reasons",
              "possible_contact_polygon_px", "uncertainty_px", "frame_interval", "time_interval",
              "selected_family_id", "member_event_ids", "native_evidence", "family_resolution",
              "geometry", "native_summary", "family_summary", "assumptions")
    summary = {key: deepcopy(raw[key]) for key in fields if key in raw}
    summary["source_sha256"] = result.get("source_sha256")
    analysis = result.get("analysis") or {}
    summary["analysis_size"] = {key: analysis.get(key) for key in ("width", "height", "frame_count")}
    times = result.get("timestamps")
    count, frames = analysis.get("frame_count"), raw.get("frame_interval")
    if (isinstance(times, list) and isinstance(count, int) and not isinstance(count, bool)
            and len(times) == count and times and all(_finite(value) for value in times)
            and all(right > left for left, right in zip(times, times[1:]))):
        summary["time_bounds"] = [times[0], times[-1]]
        summary["duration_seconds"] = times[-1]
        if (isinstance(frames, (list, tuple)) and len(frames) == 2
                and all(isinstance(value, int) and not isinstance(value, bool) for value in frames)
                and 0 <= frames[0] <= frames[1] < count):
            summary["frame_time_interval"] = [times[frames[0]], times[frames[1]]]
    summary["artifact_url"] = _calibration_preview(output, {"calibration_overlay": raw.get("artifact_path")}, prefix)
    rules = (result.get("config") or {}).get("rules", RULES)
    valid = (raw.get("source_sha256", result.get("source_sha256")) == result.get("source_sha256")
             and _valid_video_estimate(summary, rules, result.get("source_sha256")))
    if not valid:
        summary.update(decision="REVIEW", eligible=False)
        if raw.get("eligible") is True or raw.get("decision") in ("IN", "OUT"):
            codes = summary.get("reason_codes")
            codes = [code for code in codes if isinstance(code, str)] if isinstance(codes, list) else []
            summary["reason_codes"] = list(dict.fromkeys([*codes, "INVALID_VIDEO_ESTIMATE"]))
        try:
            json.dumps(summary, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            summary = {"decision": "REVIEW", "eligible": False, "decision_kind": "video_estimate",
                       "reason_codes": ["INVALID_VIDEO_ESTIMATE"], "artifact_url": None}
    return summary


def combine_views(views, *, same_event_confirmed=False, event_source=None, different_event=False, rules=None):
    """Keep a valid single-view verdict without claiming an unconfirmed transfer.

    Unresolved or failed peers cannot veto a completed source-bound verdict.
    Event correspondence is still required to transfer it to the other view;
    known different events and conflicting valid decisions remain unresolved.
    """
    selected_rules = parse_pair_rules(RULES if rules is None else rules)
    reasons, warnings = [], []
    hashes = [views[c].get("source_sha256") for c in ("A", "B")]
    valid_hashes = {camera: isinstance(views[camera].get("source_sha256"), str)
                    and bool(HASH.fullmatch(views[camera]["source_sha256"])) for camera in ("A", "B")}
    duplicate = all(valid_hashes.values()) and hashes[0] == hashes[1]
    if not all(valid_hashes.values()):
        warnings.append("SOURCE_CHANGED")
    if duplicate:
        warnings.append("DUPLICATE_SOURCE")
    if different_event:
        reasons.append("KNOWN_DIFFERENT_RALLY")
        same_event_confirmed = False
    elif not same_event_confirmed:
        warnings.append("SAME_EVENT_UNCONFIRMED")
    if any(v.get("status") == "failed" for v in views.values()):
        warnings.append("VIEW_ANALYSIS_FAILED")
    rules_match = {camera: view.get("rules", RULES) == selected_rules for camera, view in views.items()}
    if not all(rules_match.values()):
        warnings.append("RULES_MISMATCH")
    qualified = {camera: views[camera]["direct_decision"] for camera in ("A", "B")
                 if views[camera].get("direct_decision") in {"IN", "OUT"}
                 and views[camera].get("qualified") is True and views[camera].get("status") == "completed"
                 and valid_hashes[camera] and rules_match[camera]}
    estimates = {camera: views[camera]["video_estimate"]["decision"] for camera in ("A", "B")
                 if views[camera].get("status") == "completed" and valid_hashes[camera] and rules_match[camera]
                 and _valid_video_estimate(views[camera].get("video_estimate"), selected_rules,
                                          views[camera].get("source_sha256"))}
    if any(isinstance(view.get("video_estimate"), dict)
           and (isinstance(view["video_estimate"].get("reason_codes"), list)
                and "INVALID_VIDEO_ESTIMATE" in view["video_estimate"]["reason_codes"]
                or view["video_estimate"].get("eligible") is True and camera not in estimates)
           for camera, view in views.items()):
        warnings.append("INVALID_VIDEO_ESTIMATE")
    labels = set(qualified.values())
    if len(labels) > 1:
        reasons.append("CONFLICTING_VIEW_DECISIONS")
    verified_estimate_conflict = [(direct_camera, estimate_camera)
                                 for direct_camera, direct in qualified.items()
                                 for estimate_camera, estimate in estimates.items() if direct != estimate]
    if any(a == b or same_event_confirmed for a, b in verified_estimate_conflict):
        reasons.append("VERIFIED_ESTIMATE_CONFLICT")
    elif verified_estimate_conflict:
        warnings.append("UNLINKED_ESTIMATE_CONFLICT")
    selected = qualified or estimates
    kind = "verified" if qualified else "video_estimate"
    if not qualified and len(set(estimates.values())) > 1:
        reasons.append("CONFLICTING_VIDEO_ESTIMATES")
    if not selected:
        reasons.append("NO_QUALIFIED_CONTACT")
    selected_labels = set(selected.values())
    label = next(iter(selected_labels)) if len(selected_labels) == 1 and not reasons else "REVIEW"
    linked = (same_event_confirmed and not duplicate and all(valid_hashes.values())
              and all(rules_match.values()) and not different_event)
    donors = list(selected) if linked else list(selected)[:1]
    if label != "REVIEW":
        reasons = (["QUALIFIED_VIEWS_AGREE" if linked and len(qualified) == 2 else "VERDICT_FROM_QUALIFIED_VIEW"]
                   if kind == "verified" else
                   ["VIDEO_ESTIMATES_AGREE" if linked and len(estimates) == 2 else "VIDEO_ESTIMATE_FROM_VIEW"])
    else:
        reasons = list(dict.fromkeys(reasons + warnings))
    public_views = deepcopy(views)
    for camera, view in public_views.items():
        direct = view.get("direct_decision", "REVIEW") if camera in qualified else "REVIEW"
        transfer = label != "REVIEW" and kind == "verified" and linked and view.get("status") == "completed"
        view["assisted_decision"] = label if transfer else direct
        view["verdict_source_camera_ids"] = donors if transfer else [camera] if direct in {"IN", "OUT"} else []
        view.pop("assistance_mode", None)
        if transfer and camera not in qualified:
            view["assistance_mode"] = "other_view_verdict"
        view["estimated_decision"] = estimates.get(camera, "REVIEW")
        for key in list(view):
            if key.startswith("_"):
                view.pop(key)
    result = {
        "decision": label, "reason_codes": reasons, "reasons": [REASONS[code] for code in reasons],
        "decision_kind": kind if label != "REVIEW" else "unresolved",
        "warning_codes": warnings, "warnings": [REASONS[code] for code in warnings],
        "same_event_confirmed": bool(same_event_confirmed),
        "event_correspondence_source": event_source if same_event_confirmed else None,
        "source_camera_ids": donors if label != "REVIEW" else [],
        "decision_scope": ("linked_views" if linked else "single_view") if label != "REVIEW" else "unresolved",
        "views": public_views, "rules": selected_rules, "court_position_m": None,
        "assistance_used": True, "automatic_benchmark_eligible": False, "evaluation_eligible": False,
        "ground_truth_created": False, "human_verified": False,
        "video_estimate_used": label != "REVIEW" and kind == "video_estimate",
        "scope": "A/B verdict linkage; no 3D reconstruction or common pixel coordinates.",
    }
    from .review_diagnostics import pair_diagnostics
    result["review_diagnostics"] = pair_diagnostics(result)
    if result["video_estimate_used"]:
        result["review_diagnostics"].update(
            summary="착지 가능 영역으로 영상 추정을 계산했습니다. 직접 관측한 최초 접촉은 아직 확정되지 않았습니다.",
            note="영상 추정은 관측된 경로와 원본 영역의 조건부 계산이며, 직접 접촉 확인이나 검증된 정확도를 뜻하지 않습니다.",
            decision_kind="video_estimate")
    result["refinement_available"] = any(view.get("refinement_available") is True for view in views.values())
    return result


class PairWorkspace:
    def __init__(self, project_root, workspace_root, library_root=None, resource_settings_provider=None):
        self.project = Path(project_root).resolve()
        self.root = Path(workspace_root).resolve()
        self.library_root = Path(library_root).resolve() if library_root is not None else self.project / "runs/integrated-poc"
        self.review_root = self.project / "runs/ai-reviewed-poc"
        self.report_path = self.review_root / "visual-review.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paired-analysis")
        self.jobs = {}
        self._job_clocks = {}
        self._view_clocks = {}
        self._stage_clocks = {}
        self._progress_writes = {}
        self._closed = False
        self.resource_settings_provider = resource_settings_provider
        for path in sorted((self.root / "jobs").glob("*/job.json")):
            try:
                job = _read(path)
                if job.get("job_id") != path.parent.name or not SAFE_ID.fullmatch(job["job_id"]):
                    continue
                if job.get("status") not in TERMINAL:
                    # The last persisted measurement is the last known active
                    # time; server downtime must never become analysis time.
                    stopped_at = job.get("updated_at", job.get("created_at", _now()))
                    job.update(status="interrupted", phase="interrupted",
                               last_phase=job.get("phase") or ("queued" if job.get("status") == "queued" else "analyzing"),
                               error="서버가 종료되어 작업이 중단되었습니다. 다시 제출하세요.",
                               finished_at=stopped_at, interrupted_at=_now())
                    if job.get("started_at"):
                        job["elapsed_seconds"] = _seconds_between(job["started_at"], stopped_at)
                    for view in job.get("views", {}).values():
                        if view.get("status") not in TERMINAL:
                            view.update(status="interrupted", finished_at=stopped_at)
                            if view.get("started_at"):
                                view["elapsed_seconds"] = _seconds_between(view["started_at"], stopped_at)
                            progress = view.get("progress")
                            if progress:
                                progress.update(stage_status="interrupted", updated_at=stopped_at)
                                progress["stage_elapsed_seconds"] = _seconds_between(progress.get("stage_started_at"), stopped_at)
                            history = view.get("stage_history", [])
                            if history and history[-1].get("status") == "running":
                                history[-1].update(status="interrupted", finished_at=stopped_at,
                                    elapsed_seconds=_seconds_between(history[-1].get("started_at"), stopped_at))
                    _write(path, job)
                self.jobs[job["job_id"]] = job
            except (ValueError, OSError, AttributeError, TypeError):
                continue

    def close(self):
        with self.lock:
            self._closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)

    def asset_path(self, relative):
        path = _safe(self.root, relative)
        if not relative.startswith("jobs/") or "/views/" not in relative:
            raise PairWorkspaceError(403, "공개된 분석 근거 경로가 아닙니다.")
        if path.suffix.lower() not in {".mp4", ".jpg", ".jpeg", ".png"}:
            raise PairWorkspaceError(403, "지원하지 않는 근거 파일입니다.")
        return path

    def review_asset_path(self, relative):
        if not isinstance(relative, str) or not relative.startswith("assets/"):
            raise PairWorkspaceError(403, "검수 근거 경로가 아닙니다.")
        path = _safe(self.review_root, relative)
        if path.suffix.lower() not in {".mp4", ".jpg", ".jpeg", ".png"}:
            raise PairWorkspaceError(403, "지원하지 않는 근거 파일입니다.")
        return path

    def library_asset_path(self, relative):
        path = _safe(self.library_root, relative)
        if not relative.startswith("videos/") or path.suffix.lower() not in {".mp4", ".jpg", ".jpeg", ".png"}:
            raise PairWorkspaceError(403, "공개된 영상 목록 근거 경로가 아닙니다.")
        return path

    def _report(self):
        report = _read(self.report_path, {})
        if (not isinstance(report, dict) or report.get("status") != "completed"
                or report.get("review_source") != "ai_visual_review" or report.get("ground_truth_eligible") is not False):
            return {"videos": []}
        return report

    def _catalog(self):
        result = {}
        for row in self._report().get("videos", []):
            identifier = row.get("video_id")
            if isinstance(identifier, str) and SAFE_ID.fullmatch(identifier):
                result[identifier] = {"kind": "review", "row": row}
        index = _read(self.library_root / "index.json", {})
        for row in index.get("videos", []):
            identifier = row.get("id", row.get("video_id"))
            if isinstance(identifier, str) and SAFE_ID.fullmatch(identifier) and identifier not in result:
                result[identifier] = {"kind": "analysis", "row": row}
        return result

    def _review_valid(self, row):
        annotation = row.get("annotation") or {}
        digest = row.get("source_sha256")
        integrity, review = row.get("source_integrity") or {}, row.get("review") or {}
        if (not isinstance(digest, str) or not HASH.fullmatch(digest)
                or integrity.get("verified") is not True or integrity.get("sha256") != digest
                or annotation.get("source_sha256") != digest or annotation.get("source") != "ai_visual_review"
                or annotation.get("status") != "reviewed_by_ai" or annotation.get("human_verified") is not False
                or annotation.get("video_id") != row.get("video_id")
                or review.get("valid") is not True or review.get("errors")):
            return False
        if _read(_safe(self.project, f"annotations/ai-visual/{row['video_id']}.json")) != annotation:
            return False
        assets = row.get("assets") or {}
        preview, evidence = assets.get("preview"), assets.get("evidence") or []
        if not preview or not evidence:
            return False
        try:
            return self.review_asset_path(preview).is_file() and all(
                self.review_asset_path(item.get("path")).is_file() for item in evidence)
        except (PairWorkspaceError, TypeError):
            return False

    def _review_preview(self, row):
        value = (row.get("assets") or {}).get("preview")
        if value:
            try:
                if self.review_asset_path(value).is_file():
                    return "/pair-review-assets/" + quote(value, safe="/")
            except PairWorkspaceError:
                pass
        return None

    def library(self):
        videos = []
        for identifier, entry in self._catalog().items():
            row = entry["row"]
            if entry["kind"] == "review":
                valid = self._review_valid(row)
                size = (row.get("annotation") or {}).get("source_size") or {}
                videos.append({"id": identifier, "file": row.get("file", identifier),
                    "sha256": row.get("source_sha256"), "preview_url": self._review_preview(row),
                    "duration_seconds": None, "width": size.get("width"), "height": size.get("height"),
                    "direct_decision": row.get("direct_assisted_decision", "REVIEW") if valid else "REVIEW",
                    "assisted_decision": row.get("assisted_decision", "REVIEW") if valid else "REVIEW",
                    "source": "ai_visual_review", "has_review": valid})
            else:
                preview = row.get("preview_path") or f"videos/{identifier}/preview.mp4"
                try:
                    has_preview = _safe(self.library_root, preview).is_file()
                except PairWorkspaceError:
                    has_preview = False
                videos.append({"id": identifier, "file": row.get("file", identifier), "sha256": row.get("sha256"),
                    "preview_url": "/pair-library-assets/" + quote(preview, safe="/") if has_preview else None,
                    "duration_seconds": None, "width": None, "height": None,
                    "direct_decision": row.get("decision", "REVIEW"), "assisted_decision": row.get("decision", "REVIEW"),
                    "source": "automatic", "has_review": False})
        for video in videos:
            # Use original-source metadata only; a preview may have been resized.
            stored = _read(self.library_root / "videos" / video["id"] / "result.json", {})
            metadata = deepcopy(stored.get("metadata") or {})
            metadata.pop("timestamps", None)
            metadata.setdefault("extension", Path(video["file"]).suffix.lower())
            for dimension in ("width", "height"):
                if video.get(dimension) is not None:
                    metadata.setdefault(dimension, video[dimension])
            video["metadata"] = metadata
            video["duration_seconds"] = metadata.get("duration", metadata.get("duration_seconds"))
            video["width"] = metadata.get("width", video["width"])
            video["height"] = metadata.get("height", video["height"])
        videos.sort(key=lambda row: row["file"])
        return {"videos": videos, "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_frames": MAX_FRAMES},
                "rules": deepcopy(RULES), "supported_games": ["doubles", "singles"]}

    def stage_upload(self, stream, length, filename):
        from .media import probe
        if not isinstance(length, int) or isinstance(length, bool) or not 0 < length <= MAX_UPLOAD_BYTES:
            raise PairWorkspaceError(413, "영상 한 개는 1GiB 이하여야 합니다.")
        if (not isinstance(filename, str) or not filename or len(filename) > 240
                or Path(filename).name != filename or PureWindowsPath(filename).name != filename
                or any(ord(c) < 32 for c in filename) or filename in {".", ".."}):
            raise PairWorkspaceError(400, "올바른 영상 파일 이름이 필요합니다.")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}:
            raise PairWorkspaceError(415, "MP4, MOV, MKV, AVI, WEBM 영상을 선택하세요.")
        identifier = uuid.uuid4().hex
        folder = self.root / "uploads" / identifier
        folder.mkdir(parents=True)
        destination = folder / ("source" + suffix)
        digest = hashlib.sha256()
        try:
            with destination.open("xb") as handle:
                remaining = length
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk or len(chunk) > remaining:
                        raise PairWorkspaceError(400, "영상 업로드 길이가 일치하지 않습니다.")
                    digest.update(chunk)
                    handle.write(chunk)
                    remaining -= len(chunk)
            metadata = probe(destination)
            if not 1 <= metadata["frame_count"] <= MAX_FRAMES:
                raise PairWorkspaceError(413, f"영상 한 개는 {MAX_FRAMES}프레임 이하여야 합니다. 랠리 구간을 잘라 입력하세요.")
            if min(metadata.get("width", 0), metadata.get("height", 0)) < 160:
                raise PairWorkspaceError(400, "영상의 가로와 세로는 160픽셀 이상이어야 합니다.")
            metadata.pop("timestamps", None)
            record = {"upload_id": identifier, "file": filename, "sha256": digest.hexdigest(),
                      "metadata": metadata, "bytes": length, "source": destination.relative_to(self.root).as_posix(),
                      "created_at": _now()}
            _write(folder / "upload.json", record)
        except PairWorkspaceError:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        except Exception as error:
            shutil.rmtree(folder, ignore_errors=True)
            raise PairWorkspaceError(400, "업로드한 파일에서 유효한 영상 프레임을 읽지 못했습니다.") from error
        return {key: deepcopy(record[key]) for key in ("upload_id", "file", "sha256", "metadata", "bytes")}

    def _input(self, value, catalog):
        if (not isinstance(value, dict)
                or set(value) - {"review_region"} not in ({"video_id"}, {"upload_id"})):
            raise PairWorkspaceError(400, "A와 B에 각각 영상 목록 ID 또는 업로드 ID 하나를 지정하세요.")
        region = _review_region(value.get("review_region"))
        guidance = {"review_region": region} if region is not None else {}
        if "video_id" in value:
            identifier = _identifier(value["video_id"])
            if identifier not in catalog:
                raise PairWorkspaceError(404, "영상 목록에 없는 영상입니다.")
            return {"video_id": identifier, "file": catalog[identifier]["row"].get("file", identifier), **guidance}
        identifier = _identifier(value["upload_id"])
        upload = _read(_safe(self.root, f"uploads/{identifier}/upload.json"))
        if not upload or upload.get("upload_id") != identifier:
            raise PairWorkspaceError(404, "업로드한 영상을 찾을 수 없습니다.")
        metadata = deepcopy(upload.get("metadata") or {})
        metadata.pop("timestamps", None)
        return {"upload_id": identifier, "file": upload["file"], "metadata": metadata, **guidance}

    def submit(self, body):
        resources = self.resource_settings_provider() if self.resource_settings_provider else None
        if not isinstance(body, dict) or not isinstance(body.get("views"), dict) or set(body["views"]) != {"A", "B"}:
            raise PairWorkspaceError(400, "A와 B 두 시점을 함께 입력하세요.")
        same_event, reuse = body.get("same_event_confirmed", False), body.get("reuse_review", False)
        refine = body.get("refine_review_regions", False)
        if not isinstance(same_event, bool) or not isinstance(reuse, bool) or not isinstance(refine, bool):
            raise PairWorkspaceError(400, "같은 랠리 확인 및 검수 재사용 값은 참/거짓이어야 합니다.")
        if refine and reuse:
            raise PairWorkspaceError(400, "집중 재분석은 기존 판정 재사용을 끈 새 분석으로 실행하세요.")
        rules = parse_pair_rules(body.get("rules", RULES))
        catalog = self._catalog()
        selected = {camera: self._input(body["views"][camera], catalog) for camera in ("A", "B")}
        if reuse and any(value.get("review_region") is not None for value in selected.values()):
            raise PairWorkspaceError(400, "지정한 영역은 기존 판정 재사용을 끈 새 분석으로 실행하세요.")
        with self.lock:
            if self._closed:
                raise PairWorkspaceError(503, "분석 서버가 종료 중입니다.")
            if sum(job["status"] not in TERMINAL for job in self.jobs.values()) >= MAX_ACTIVE_JOBS:
                raise PairWorkspaceError(429, "대기 중인 A/B 작업이 많습니다. 기존 작업이 끝난 뒤 다시 시도하세요.")
            identifier, now = uuid.uuid4().hex, _now()
            job = {"job_id": identifier, "status": "queued", "phase": "queued", "created_at": now, "updated_at": now,
                   "started_at": None, "finished_at": None, "elapsed_seconds": 0., "queue_seconds": 0.,
                   "views": {camera: {"status": "queued", "file": value["file"], "elapsed_seconds": 0.,
                                      "started_at": None, "finished_at": None, "stage_history": [],
                                      "progress": {"stage": "queued", "stage_label": "분석 대기", "stage_index": 0,
                                                   "stage_count": 7, "stage_status": "queued", "updated_at": now}}
                             for camera, value in selected.items()},
                   "result": None, "request": {"views": selected, "same_event_confirmed": same_event,
                                              "reuse_review": reuse, "refine_review_regions": refine, "rules": rules}}
            self.jobs[identifier] = job
            if resources is not None:
                job["request"]["resource_settings"] = deepcopy(resources)
            self._persist(job)
            self.executor.submit(self._run, identifier)
        return {"job_id": identifier, "status": "queued"}

    def _persist(self, job):
        self._refresh_timing(job)
        _write(self.root / "jobs" / job["job_id"] / "job.json", job)

    def _refresh_timing(self, job):
        """Refresh live measurements under self.lock, without writing a file."""
        clock = time.perf_counter()
        identifier = job["job_id"]
        if job.get("status") not in TERMINAL and identifier in self._job_clocks:
            job["elapsed_seconds"] = round(max(0., clock - self._job_clocks[identifier]), 6)
        for camera, view in job.get("views", {}).items():
            key = (identifier, camera)
            if view.get("status") == "running" and key in self._view_clocks:
                view["elapsed_seconds"] = round(max(0., clock - self._view_clocks[key]), 6)
                if key in self._stage_clocks:
                    elapsed = round(max(0., clock - self._stage_clocks[key]), 6)
                    view["progress"]["stage_elapsed_seconds"] = elapsed
                    if view.get("stage_history"):
                        view["stage_history"][-1]["elapsed_seconds"] = elapsed

    def _snapshot(self, job):
        self._refresh_timing(job)
        snapshot = deepcopy(job)
        snapshot["server_time"] = _now()
        if job.get("status") == "queued":
            snapshot["queue_seconds"] = _seconds_between(job.get("created_at"), snapshot["server_time"])
        return snapshot

    def get_job(self, identifier):
        _identifier(identifier)
        with self.lock:
            if identifier not in self.jobs:
                raise PairWorkspaceError(404, "A/B 작업을 찾을 수 없습니다.")
            return self._snapshot(self.jobs[identifier])

    def list_jobs(self):
        with self.lock:
            rows = sorted(self.jobs.values(), key=lambda item: item["created_at"], reverse=True)
            snapshots = [self._snapshot(job) for job in rows[:100]]
            keys = ("job_id", "status", "phase", "last_phase", "created_at", "updated_at", "views", "started_at",
                    "finished_at", "elapsed_seconds", "queue_seconds", "server_time")
            return {"jobs": [{key: job[key] for key in keys if key in job}
                             | {"decision": (job.get("result") or {}).get("decision"),
                                "decision_kind": (job.get("result") or {}).get("decision_kind")} for job in snapshots]}

    def _analysis_progress(self, identifier, camera, event):
        stage = event.get("stage")
        if stage not in ANALYSIS_STAGES and stage not in {"resolve", "stored_review"}:
            return
        index, label = ANALYSIS_STAGES.get(stage, (0, "원본 확인" if stage == "resolve" else "기존 검수 근거 확인"))
        with self.lock:
            job = self.jobs[identifier]
            view = job["views"][camera]
            if view.get("status") != "running":
                return
            now, clock = _now(), time.perf_counter()
            key = (identifier, camera)
            previous = view.get("progress", {})
            changed = previous.get("stage") != stage
            if changed:
                self._refresh_timing(job)
                history = view.setdefault("stage_history", [])
                if history and history[-1].get("status") == "running":
                    history[-1].update(status="completed", finished_at=now)
                history.append({"stage": stage, "stage_label": label, "stage_index": index,
                                "started_at": now, "elapsed_seconds": 0., "status": "running"})
                self._stage_clocks[key] = clock
                progress = {"stage": stage, "stage_label": label, "stage_index": index, "stage_count": 7,
                            "stage_started_at": now, "stage_elapsed_seconds": 0., "stage_status": "running"}
            else:
                progress = deepcopy(previous)
            # Accept only serializable, bounded display fields. Counters refer
            # to actual work completed, never a time-based simulated percent.
            total, completed = event.get("total_units"), event.get("completed_units")
            valid_counter = (isinstance(total, int) and not isinstance(total, bool) and total > 0
                    and isinstance(completed, int) and not isinstance(completed, bool) and completed >= 0)
            detail_changed = isinstance(event.get("detail"), str) and event["detail"] != previous.get("detail")
            if detail_changed and not valid_counter:
                for field in ("completed_units", "total_units", "unit"):
                    progress.pop(field, None)
            if valid_counter:
                progress.update(total_units=total, completed_units=min(completed, total))
                if event.get("unit") == "frames":
                    progress["unit"] = "frames"
            if isinstance(event.get("detail"), str):
                progress["detail"] = event["detail"][:240]
            progress["updated_at"] = now
            view["progress"] = progress
            job["updated_at"] = now
            self._refresh_timing(job)
            final_units = valid_counter and completed >= total and previous.get("completed_units") != total
            if changed or detail_changed or final_units or clock - self._progress_writes.get(key, 0.) >= .5:
                self._persist(job)
                self._progress_writes[key] = clock

    def _progress(self, identifier, camera, status):
        with self.lock:
            job = self.jobs[identifier]
            view = job["views"][camera]
            now, clock = _now(), time.perf_counter()
            key = (identifier, camera)
            if status == "running":
                view.update(status="running", started_at=now, elapsed_seconds=0.)
                self._view_clocks[key] = clock
                self._analysis_progress(identifier, camera, {"stage": "resolve"})
            else:
                self._refresh_timing(job)
                progress = view.get("progress", {})
                if view.get("stage_history"):
                    view["stage_history"][-1].update(status="completed" if status == "completed" else status,
                                                    finished_at=now)
                if status == "completed":
                    progress = {"stage": "completed", "stage_label": "시점 분석 완료", "stage_index": 7,
                                "stage_count": 7, "stage_status": "completed", "updated_at": now,
                                "last_stage": progress.get("stage")}
                else:
                    progress.update(stage_status=status, updated_at=now)
                view.update(status=status, finished_at=now, progress=progress)
                self._view_clocks.pop(key, None)
                self._stage_clocks.pop(key, None)
                self._progress_writes.pop(key, None)
            job["updated_at"] = now
            self._persist(job)

    def _finish_view(self, identifier, camera, base):
        self._progress(identifier, camera, base["status"])
        with self.lock:
            measured = self.jobs[identifier]["views"][camera]
            for key in ("started_at", "finished_at", "elapsed_seconds", "progress", "stage_history"):
                if key in measured:
                    base[key] = deepcopy(measured[key])
            self.jobs[identifier]["views"][camera] = deepcopy(base)
            self._persist(self.jobs[identifier])
        return base

    def _original(self, entry):
        row = entry["row"]
        if entry["kind"] == "review":
            relative = (row.get("source_integrity") or {}).get("path")
            path = _safe(self.project, relative) if relative else None
            return path if path and path.is_file() else None
        relative = row.get("result_path") or f"videos/{row['id']}/result.json"
        result = _read(_safe(self.library_root, relative), {})
        recorded = result.get("source_path")
        if recorded:
            candidate = Path(recorded)
            if candidate.is_file() and candidate.resolve().is_relative_to(self.project):
                return candidate
        name = row.get("file")
        if isinstance(name, str) and Path(name).name == name:
            candidate = _safe(self.project, "video/" + name)
            if candidate.is_file():
                return candidate
        return None

    def _direct_review(self, row, rules=None):
        from .visual_review import judge_visual_geometry
        annotation, review = row["annotation"], row["review"]
        label = row.get("direct_assisted_decision", "REVIEW")
        if (label not in {"IN", "OUT"} or annotation.get("label") != label
                or annotation.get("target_contact_identity_confirmed") is not True
                or not (annotation.get("contact_observed") is True or annotation.get("contact_interval_observed") is True)
                or annotation.get("rules") != (RULES if rules is None else rules)):
            return "REVIEW"
        geometry = judge_visual_geometry(annotation, annotation.get("analysis_size") or {})
        return label if geometry.get("decision") == label and review.get("qualified") is True else "REVIEW"

    def _resolve(self, selected, catalog):
        from .media import file_hash
        if "video_id" in selected:
            entry = catalog.get(selected["video_id"])
            if entry is None:
                raise PairWorkspaceError(409, "작업 대기 중 영상 목록이 변경되었습니다.")
            row = entry["row"]
            digest = row.get("source_sha256") if entry["kind"] == "review" else row.get("sha256")
            original = self._original(entry)
            if original is not None and file_hash(original) != digest:
                raise PairWorkspaceError(409, REASONS["SOURCE_CHANGED"])
            return original, digest, [selected["video_id"]], entry
        upload = _read(_safe(self.root, f"uploads/{selected['upload_id']}/upload.json"))
        if not upload:
            raise PairWorkspaceError(404, "업로드 영상을 찾을 수 없습니다.")
        original = _safe(self.root, upload["source"])
        digest = file_hash(original)
        if digest != upload["sha256"]:
            raise PairWorkspaceError(409, REASONS["SOURCE_CHANGED"])
        matches = [identifier for identifier, entry in catalog.items()
                   if (entry["row"].get("source_sha256") if entry["kind"] == "review" else entry["row"].get("sha256")) == digest]
        # Duplicate contents (e.g. B17/B18) cannot choose an arbitrary event ID.
        entry = catalog[matches[0]] if len(matches) == 1 else None
        return original, digest, matches, entry

    def _view(self, identifier, camera, selected, reuse, catalog, refine=False):
        started = time.perf_counter()
        self._progress(identifier, camera, "running")
        rules = parse_pair_rules(self.jobs[identifier]["request"].get("rules", RULES))
        base = {"camera_id": camera, "file": selected["file"], "status": "completed", "rules": rules,
                "source_sha256": None, "automatic_decision": "REVIEW", "direct_decision": "REVIEW",
                "assisted_decision": "REVIEW", "qualified": False, "preview_url": None, "evidence_urls": [],
                "estimated_decision": "REVIEW", "video_estimate": None, "evidence_warnings": [],
                "trajectory_replay_url": None, "trajectory_replay_info": None,
                "calibration_summary": None, "calibration_preview_url": None,
                "contact_time_seconds": None, "reason_codes": [], "rationale": "", "_candidate_ids": []}
        try:
            original, digest, candidates, entry = self._resolve(selected, catalog)
            base.update(source_sha256=digest, _candidate_ids=candidates)
            from .review_refinement import review_regions
            hint = entry["row"]["annotation"] if entry and entry["kind"] == "review" and self._review_valid(entry["row"]) else {}
            base["refinement_available"] = bool(review_regions(hint)) and original is not None
            stored_rule_match = bool(entry and entry["kind"] == "review"
                                     and entry["row"].get("annotation", {}).get("rules") == rules)
            if reuse and entry is not None and entry["kind"] == "review" and not stored_rule_match:
                base["review_reuse_skipped"] = "RULES_CHANGED"
                if original is None:
                    raise PairWorkspaceError(409, "저장된 검수의 경기 규칙이 선택한 규칙과 다릅니다. 원본을 업로드해 선택한 규칙으로 새 분석하세요.")
            if reuse and entry is not None and entry["kind"] == "review" and stored_rule_match:
                self._analysis_progress(identifier, camera, {"stage": "stored_review"})
                row = entry["row"]
                if not self._review_valid(row):
                    raise PairWorkspaceError(409, REASONS["STALE_REVIEW"])
                direct = self._direct_review(row, rules)
                annotation = row["annotation"]
                base.update(direct_decision=direct, assisted_decision=direct, qualified=direct in {"IN", "OUT"},
                    automatic_decision=row.get("automatic_decision", "REVIEW"), review_source="ai_visual_review",
                    processing_mode="stored_ai_review", stored_video_id=row["video_id"],
                    preview_url=self._review_preview(row),
                    evidence_urls=["/pair-review-assets/" + quote(item["path"], safe="/") for item in row["assets"]["evidence"]],
                    contact_time_seconds=annotation.get("time") if direct in {"IN", "OUT"} else None,
                    reason_codes=deepcopy(row["review"].get("direct_reason_codes", row["review"].get("reason_codes", []))),
                    rationale=annotation.get("rationale", ""),
                    review_details={"observation_status": annotation.get("observation_status"),
                        "follow_up_actions": deepcopy(annotation.get("follow_up_actions") or []),
                        "focus_time_seconds": annotation.get("time")},
                    source_verification="current_original_sha256" if original is not None else "packaged_source_bound_review")
            else:
                if original is None:
                    raise PairWorkspaceError(409, "새 분석에 필요한 원본이 없습니다. 원본 A/B를 업로드하거나 저장된 검수를 사용하세요.")
                from .pipeline import analyze_video
                from .media import probe
                import cv2
                metadata = probe(original)
                base["metadata"] = {key: value for key, value in metadata.items() if key != "timestamps"}
                if not 1 <= metadata["frame_count"] <= MAX_FRAMES:
                    raise PairWorkspaceError(413, f"영상 한 개는 {MAX_FRAMES}프레임 이하여야 합니다.")
                config = _read(self.project / "configs/poc.json", {"analysis_width": 960, "max_frames": MAX_FRAMES})
                config.update(video_id=f"pair_{identifier}_{camera.lower()}", rules=deepcopy(rules), max_frames=MAX_FRAMES)
                resources = self.jobs[identifier]["request"].get("resource_settings")
                if resources is not None:
                    config["resource_settings"] = deepcopy(resources)
                    config["opencv_threads"] = resources["cpu_threads"]
                manual_region = selected.get("review_region")
                if manual_region is not None:
                    regions = _manual_review_regions(manual_region, metadata, config.get("analysis_width", 1280))
                elif refine:
                    regions = review_regions(hint, min(metadata["width"], config.get("analysis_width", 960)),
                                             metadata["frame_count"])
                else:
                    regions = []
                if regions:
                    config.setdefault("detector", {}).update(refinement_regions=regions, include_dark_components=True)
                    config.setdefault("events", {})["review_refinement_regions"] = deepcopy(regions)
                    base["refinement"] = {"source": "operator_review_region" if manual_region is not None else "ai_visual_review",
                        "source_sha256": digest,
                        "regions": regions, "assistance_used": True, "automatic_benchmark_eligible": False,
                        "evaluation_eligible": False,
                        "scope": "영역은 후보 검색에만 사용하며 코트 보정·셔틀콕 식별·접촉 확정 근거가 아닙니다."}
                    base.update(assistance_used=True, evaluation_eligible=False)
                    if manual_region is not None:
                        base["refinement"].update(normalized_region=deepcopy(manual_region),
                            coordinate_system="displayed_video_normalized", mapping="outward_integer_analysis_pixels",
                            ai_visual_refinement_requested=refine, ai_visual_refinement_applied=False,
                            ai_visual_refinement_available=base["refinement_available"],
                            priority="operator_region_over_ai_visual_review")
                        config.setdefault("request_context", {})["review_region_guidance"] = deepcopy(base["refinement"])
                if resources is None:
                    cv2.setNumThreads(max(1, min(2, int(config.get("opencv_threads", 2)))))
                output = self.root / "jobs" / identifier / "views" / camera
                result = analyze_video(original, output, config=config,
                    progress_callback=lambda event: self._analysis_progress(identifier, camera, event))
                if result.get("source_sha256") != digest:
                    raise PairWorkspaceError(409, REASONS["SOURCE_CHANGED"])
                decision = result.get("decision", {}).get("decision", "REVIEW")
                if decision not in LABELS:
                    decision = "REVIEW"
                prefix = f"/pair-media/jobs/{identifier}/views/{camera}/"
                artifacts, evidence, trajectory_context = result.get("artifacts") or {}, [], None
                for item in artifacts.get("evidence", []):
                    for field in ("image", "source_sequence", "source_crop"):
                        relative = item.get(field)
                        if isinstance(relative, str) and _safe(output, relative).is_file():
                            evidence.append(prefix + quote(relative, safe="/"))
                    relative = item.get("trajectory_context")
                    if isinstance(relative, str) and _safe(output, relative).is_file():
                        trajectory_context = prefix + quote(relative, safe="/")
                contact = (result.get("decision") or {}).get("contact_time_seconds")
                if contact is None and decision != "REVIEW":
                    contact = (result.get("selected_landing") or {}).get("time")
                if (result.get("selected_landing") or {}).get("contact_observed") is False:
                    contact = None
                replay_url, replay_info = _trajectory_replay(output, artifacts, prefix)
                video_estimate = _video_estimate_summary(result, output, prefix)
                preview = artifacts.get("preview")
                preview_url = (prefix + quote(preview, safe="/")
                               if isinstance(preview, str) and _safe(output, preview).is_file() else None)
                base.update(automatic_decision=decision, direct_decision=decision, assisted_decision=decision,
                    qualified=decision in {"IN", "OUT"},
                    video_estimate=video_estimate,
                    estimated_decision=video_estimate["decision"] if video_estimate else "REVIEW",
                    review_source=("operator_region_guided_search" if manual_region is not None else
                                   "ai_review_guided_search" if regions else "automatic"),
                    processing_mode=("user_region_reanalysis" if manual_region is not None else
                                     "review_guided_reanalysis" if regions else "fresh_analysis"),
                    preview_url=preview_url,
                    evidence_warnings=deepcopy(result.get("evidence_warnings") or []),
                    evidence_urls=evidence, contact_time_seconds=contact if _finite(contact) else None,
                    landing_candidate=_landing_candidate(result), trajectory_context_url=trajectory_context,
                    trajectory_replay_url=replay_url, trajectory_replay_info=replay_info,
                    calibration_summary=_calibration_summary(result),
                    calibration_preview_url=_calibration_preview(output, artifacts, prefix),
                    reason_codes=(result.get("decision") or {}).get("reason_codes", []),
                    rationale="새 원본 분석의 접촉·경계 검증 결과입니다.", analysis_timings=result.get("timings", {}),
                    execution=deepcopy(result.get("execution")), assessment=deepcopy(result.get("assessment")),
                    source_verification="current_original_sha256")
            return self._finish_view(identifier, camera, base)
        except Exception as error:
            with self.lock:
                stage = self.jobs[identifier]["views"][camera].get("progress", {}).get("stage")
            message, diagnostics = _failure_details(error, stage)
            LOGGER.exception("Pair analysis failed: job=%s camera=%s stage=%s", identifier, camera, stage)
            relative_log = f"jobs/{identifier}/views/{camera}/analysis-error.log"
            try:
                log_path = self.root / relative_log
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(f"job={identifier} camera={camera} stage={stage}\n" + traceback.format_exc(),
                                    encoding="utf-8")
                diagnostics["log_path"] = relative_log
            except OSError:
                LOGGER.exception("Could not save the analysis error log for job=%s camera=%s", identifier, camera)
            base.update(status="failed", reason_codes=["VIEW_ANALYSIS_FAILED"], review_source="none",
                        processing_mode="failed", error=message, error_details=diagnostics,
                        elapsed_seconds=round(time.perf_counter() - started, 6))
            return self._finish_view(identifier, camera, base)

    def _correspondence(self, views, catalog):
        accepted, rejected, actions = False, False, []
        for camera, partner in (("A", "B"), ("B", "A")):
            own_ids, other_ids = views[camera]["_candidate_ids"], set(views[partner]["_candidate_ids"])
            for identifier in own_ids:
                entry = catalog.get(identifier)
                if not entry or entry["kind"] != "review" or not self._review_valid(entry["row"]):
                    continue
                row, annotation = entry["row"], entry["row"]["annotation"]
                for candidate in annotation.get("cross_view_candidates", []):
                    if (candidate.get("other_video_id") in other_ids and candidate.get("status") == "different_rally"
                            and candidate.get("same_event_confirmed_by_ai") is False):
                        rejected = True
                match, validation = annotation.get("cross_view_match") or {}, row.get("cross_view_review") or {}
                other_id = match.get("other_video_id")
                other_entry = catalog.get(other_id)
                valid_pair = (len(own_ids) == len(other_ids) == 1 and other_id in other_ids
                    and other_entry is not None and other_entry["kind"] == "review"
                    and self._review_valid(other_entry["row"])
                    and row.get("source_sha256") == views[camera]["source_sha256"]
                    and other_entry["row"].get("source_sha256") == views[partner]["source_sha256"])
                if (valid_pair and match.get("same_event_confirmed_by_ai") is True
                        and (validation.get("event_correspondence_validated") is True or validation.get("qualified") is True)
                        and validation.get("originating_video_id") == other_id
                        and len(match.get("matched_actions", [])) >= 2):
                    accepted = True
                    actions = [{"camera_id": camera, "other_camera_id": partner, **deepcopy(action)}
                               for action in match["matched_actions"]]
        return accepted, rejected, actions

    def _run(self, identifier):
        started = time.perf_counter()
        with self.lock:
            job = self.jobs[identifier]
            now = _now()
            job.update(status="running", phase="analyzing", started_at=now, updated_at=now,
                       queue_seconds=_seconds_between(job.get("created_at"), now))
            self._job_clocks[identifier] = started
            self._persist(job)
            request = deepcopy(job["request"])
        try:
            catalog = self._catalog()
            parallel_views = (request.get("resource_settings") or {}).get("parallel_views", 2)
            with ThreadPoolExecutor(max_workers=parallel_views, thread_name_prefix="pair-view") as pool:
                futures = {pool.submit(self._view, identifier, camera, selected, request["reuse_review"], catalog,
                                       request.get("refine_review_regions", False)): camera
                           for camera, selected in request["views"].items()}
                views = {futures[future]: future.result() for future in as_completed(futures)}
            views = {camera: views[camera] for camera in ("A", "B")}
            with self.lock:
                job.update(phase="linking", views=deepcopy(views), updated_at=_now())
                self._persist(job)
            accepted, rejected, actions = self._correspondence(views, catalog)
            if not request["reuse_review"] and not request.get("refine_review_regions", False):
                accepted, actions = False, []
            confirmed = accepted or request["same_event_confirmed"]
            source = "verified_ai_action_correspondence" if accepted else "operator_pair_selection" if confirmed else None
            result = combine_views(views, same_event_confirmed=confirmed, event_source=source,
                                   different_event=rejected, rules=request.get("rules", RULES))
            result.update(matched_actions=actions, clock_synchronization_claimed=False,
                          refinement_requested=request.get("refine_review_regions", False),
                          user_region_camera_ids=[camera for camera, selected in request["views"].items()
                                                  if selected.get("review_region") is not None],
                          refinement_applied_camera_ids=[camera for camera, view in views.items() if view.get("refinement")],
                          elapsed_seconds=round(time.perf_counter() - started, 6), completed_at=_now())
            with self.lock:
                failed = any(v["status"] == "failed" for v in views.values())
                if failed:
                    # Linking can produce a REVIEW result after a view failed;
                    # the failed work still belongs to the analysis phase.
                    job["last_phase"] = "analyzing"
                job.update(status="completed_with_errors" if failed else "completed",
                           phase="failed" if failed else "completed", result=result,
                           updated_at=_now(), finished_at=result["completed_at"],
                           elapsed_seconds=result["elapsed_seconds"])
                self._job_clocks.pop(identifier, None)
                self._persist(job)
        except Exception as error:
            LOGGER.exception("Pair job failed: job=%s", identifier)
            with self.lock:
                now = _now()
                self._refresh_timing(job)
                for camera, view in job.get("views", {}).items():
                    if view.get("status") not in TERMINAL:
                        self._progress(identifier, camera, "failed")
                job.update(status="failed", phase="failed", last_phase=job.get("phase", "analyzing"),
                           error=f"A/B 작업 실패: {type(error).__name__} · {' '.join(str(error).split())[:600]}",
                           updated_at=now, finished_at=now,
                           elapsed_seconds=round(time.perf_counter() - started, 6))
                self._job_clocks.pop(identifier, None)
                self._persist(job)
