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
    "BEST_ESTIMATE_FROM_VIEW": "직접 판정과 영상 추정이 없어, 근거가 더 강한 시점의 최선 추정 위치로 IN/OUT을 판정했습니다.",
    "BEST_ESTIMATES_AGREE": "두 시점의 최선 추정 IN/OUT이 일치합니다.",
    "VIEW_CONFLICT_RESOLVED_BY_CONFIDENCE": "두 시점의 판정이 달라 근거 강도·신뢰도·경계 여유 순으로 더 강한 시점의 결과를 사용했습니다.",
    "VIEW_CONFLICT_TIE_BROKEN_BY_CAMERA_ORDER": "두 시점의 판정이 다르고 근거 강도·신뢰도·경계 여유가 같아 A 시점의 결과를 사용했습니다. 이 결과는 임의 순서에 따른 것입니다.",
    "CONFLICTING_BEST_ESTIMATES": "A와 B의 최선 추정 IN/OUT이 서로 다릅니다.",
    "INVALID_BEST_ESTIMATE": "최선 추정의 출처·규칙 또는 위치 값을 검증하지 못했습니다.",
}

# Strongest evidence first. A lower tier never overrides a higher one.
DECISION_TIERS = ("verified", "video_estimate", "best_estimate")
_TIER_DEFAULT_CONFIDENCE = {"verified": 0.97, "video_estimate": 0.9, "best_estimate": 0.5}


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
            detail = ("모서리 옆 카메라 배치용 근거리 반코트 맞춤으로 카메라 앞 주 코트를 선택했습니다. "
                      "선택한 코트의 관측 경계만 표시합니다. " if primary.get("fallback") is True else
                      "앞쪽 두 모서리가 보이는 주 코트 하나를 선택했습니다. 선택한 코트의 관측 경계만 표시합니다. ")
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
            "primary_court_fallback": bool(selected and primary.get("fallback") is True),
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


def view_from_result(result, output, prefix, rules=None):
    """Public view fields for one fresh, source-bound analysis result.

    Shared by the A/B workspace and the batch pair harness so both hand the
    same per-view evidence to :func:`combine_views`.
    """
    output = Path(output)
    from .best_estimate import refresh_best_estimate
    result = refresh_best_estimate(result, rules if rules is not None else (result.get("config") or {}).get("rules", RULES))
    decision = (result.get("decision") or {}).get("decision", "REVIEW")
    if decision not in LABELS:
        decision = "REVIEW"
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
    return dict(automatic_decision=decision, direct_decision=decision, assisted_decision=decision,
        qualified=decision in {"IN", "OUT"},
        source_sha256=result.get("source_sha256"),
        video_estimate=video_estimate,
        estimated_decision=video_estimate["decision"] if video_estimate else "REVIEW",
        best_estimate=_best_estimate_summary(result, rules, output, prefix),
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


_BEST_FIELDS = ("schema_version", "estimator_version", "decision", "decision_kind", "source", "basis",
                "confidence", "evidence_strength", "geometry_reliability", "location_decision", "p_in",
                "prior_in", "rules",
                "classification", "reason_codes", "reasons", "contact_observed", "target_identity_verified",
                "point_px", "event_ids", "frame_index", "time", "frame_interval", "time_interval",
                "candidate_score", "track_id", "alternatives", "scope", "artifact_path", "error",
                "assistance_used", "evaluation_eligible")


def _best_estimate_summary(result, rules=None, output=None, prefix=None):
    """Bounded best-estimate record bound to the analysed source and rules."""
    raw = result.get("best_estimate")
    if not isinstance(raw, dict):
        return None
    summary = {key: deepcopy(raw[key]) for key in _BEST_FIELDS if key in raw}
    summary["source_sha256"] = result.get("source_sha256")
    if output is not None and prefix is not None:
        summary["artifact_url"] = _calibration_preview(output, {"calibration_overlay": raw.get("artifact_path")}, prefix)
    expected = rules if rules is not None else (result.get("config") or {}).get("rules", RULES)
    if (raw.get("source_sha256", result.get("source_sha256")) != result.get("source_sha256")
            or not _valid_best_estimate(summary, expected, result.get("source_sha256"))):
        codes = summary.get("reason_codes")
        codes = [code for code in codes if isinstance(code, str)] if isinstance(codes, list) else []
        if raw.get("decision") in ("IN", "OUT"):
            codes.append("INVALID_BEST_ESTIMATE")
        summary.update(decision="REVIEW", confidence=None, reason_codes=list(dict.fromkeys(codes)))
    try:
        json.dumps(summary, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        summary = {"decision": "REVIEW", "decision_kind": "best_estimate", "confidence": None,
                   "reason_codes": ["INVALID_BEST_ESTIMATE"]}
    return summary


def _valid_best_estimate(estimate, rules, source_sha256):
    """A best estimate is a labelled heuristic; validate its provenance only."""
    from .best_estimate import BASIS_PRIOR, DECISION_KIND, SOURCE
    if (not isinstance(source_sha256, str) or not HASH.fullmatch(source_sha256)
            or not isinstance(estimate, dict) or estimate.get("schema_version") != 1
            or type(estimate.get("schema_version")) is not int
            or estimate.get("decision") not in ("IN", "OUT")
            or estimate.get("decision_kind") != DECISION_KIND or estimate.get("source") != SOURCE
            or estimate.get("basis") not in BASIS_PRIOR
            or estimate.get("rules") != rules or estimate.get("source_sha256") != source_sha256
            or estimate.get("contact_observed") is not False
            or estimate.get("target_identity_verified") is not False):
        return False
    confidence = estimate.get("confidence")
    if not _finite(confidence) or not 0.5 <= confidence <= 1:
        return False
    point = estimate.get("point_px")
    if point is not None and (not isinstance(point, (list, tuple)) or len(point) != 2
                              or not all(_finite(value) for value in point)):
        return False
    codes = estimate.get("reason_codes")
    return isinstance(codes, list) and all(isinstance(code, str) for code in codes)


def _add_analysis_assessments(result):
    """Refresh presentation from validated evidence without changing verdicts.

    This also upgrades the display of stored jobs. It never reruns inference,
    rewrites their direct diagnostics, or transfers one camera's observations
    to another camera.
    """
    from .assessment import assess_analysis
    views = result.get("views")
    if not isinstance(views, dict):
        return
    try:
        rules = parse_pair_rules(result.get("rules", RULES))
    except PairWorkspaceError:
        rules = None
    valid_direct, valid_estimates, valid_best = set(), set(), set()
    for camera, view in views.items():
        if camera not in ("A", "B") or not isinstance(view, dict):
            continue
        digest = view.get("source_sha256")
        usable = (rules is not None and view.get("status") == "completed"
                  and isinstance(digest, str) and bool(HASH.fullmatch(digest))
                  and view.get("rules", RULES) == rules)
        direct = usable and view.get("qualified") is True and view.get("direct_decision") in ("IN", "OUT")
        estimated = usable and _valid_video_estimate(view.get("video_estimate"), rules, digest)
        best = usable and _valid_best_estimate(view.get("best_estimate"), rules, digest)
        if direct:
            valid_direct.add(camera)
        if estimated:
            valid_estimates.add(camera)
        if best:
            valid_best.add(camera)
        assessment = assess_analysis(view, direct_qualified=bool(direct), estimate_valid=bool(estimated),
                                     best_valid=bool(best))
        view.update(analysis_assessment=assessment, effective_decision=assessment["line_verdict"],
                    effective_decision_kind=assessment["decision_kind"],
                    estimated_decision=view["video_estimate"]["decision"] if estimated else "REVIEW",
                    best_estimated_decision=view["best_estimate"]["decision"] if best else "REVIEW")
    source_ids = result.get("source_camera_ids")
    source_ids = source_ids if isinstance(source_ids, list) else []
    kind = result.get("decision_kind", "verified")
    eligible_sources = {"verified": valid_direct, "video_estimate": valid_estimates,
                        "best_estimate": valid_best}.get(kind, set())
    # A donor must supply the final label from the final tier (its own effective
    # label may come from a stronger tier only when that tier is the final one).
    tier_label = {"verified": lambda view: view.get("direct_decision"),
                  "video_estimate": lambda view: view["video_estimate"].get("decision"),
                  "best_estimate": lambda view: view["best_estimate"].get("decision")}.get(kind)
    donors = [camera for camera in ("A", "B") if camera in source_ids and camera in eligible_sources
              and tier_label is not None and tier_label(views[camera]) == result.get("decision")]
    final_resolved = result.get("decision") in ("IN", "OUT") and bool(donors)
    for camera, view in views.items():
        if not isinstance(view, dict) or "analysis_assessment" not in view:
            continue
        assessment = view["analysis_assessment"]
        supplementary = final_resolved and camera not in donors and view.get("status") == "completed"
        assessment.update(supplementary=supplementary, contributes_to_final=bool(final_resolved and camera in donors),
                          final_source_camera_ids=donors if final_resolved else [], final_context=None)
        if supplementary:
            own = view.get("effective_decision")
            key = result.get("conflict_resolution_key")
            why = {"evidence_strength": "근거 강도(위치 근거의 신뢰성)가 더 낮아",
                   "confidence": "신뢰도가 더 낮아", "boundary_clearance": "경계 여유가 더 작아",
                   "camera_order": "근거가 같아 A 시점을 우선하는 순서에 따라"}.get(key, "근거 단계가 더 낮아")
            if own in ("IN", "OUT") and own != result["decision"]:
                note = (f"이 시점의 자체 결과는 {own}이지만 {why} 최종 판정에 쓰지 않았습니다. "
                        "상세에서 이 시점의 근거를 비교할 수 있습니다.")
            elif own in ("IN", "OUT"):
                note = "이 시점의 자체 결과도 같으며, 최종 판정은 근거가 더 강한 시점을 기준으로 표시합니다."
            else:
                note = ("이 시점에는 독립적인 IN·OUT 근거가 없어 보조 영상으로 제공합니다. "
                        "이 시점의 확인되지 않은 근거는 상세에서 볼 수 있습니다.")
            assessment["final_context"] = f"최종 {result['decision']}은 {'/'.join(donors)} 시점의 근거를 사용합니다. " + note


def _tier_strength(camera, kind, bests):
    """Ordering key inside one evidence tier.

    Best estimates are ordered by evidence strength first: a view whose label
    rests on located evidence outranks one that mostly follows the IN prior,
    whatever their confidences. Then confidence; camera A breaks exact ties.
    """
    best = bests.get(camera) or {}
    confidence = best.get("confidence") if _finite(best.get("confidence")) else None
    if kind != "best_estimate" and (confidence is None or best.get("basis") not in (
            "verified_contact" if kind == "verified" else "video_estimate",)):
        confidence = _TIER_DEFAULT_CONFIDENCE[kind]
    strength = best.get("evidence_strength") if kind == "best_estimate" and _finite(best.get("evidence_strength")) else 1.0
    classification = best.get("classification") if isinstance(best.get("classification"), dict) else {}
    # Unit-free boundary clearance (pixels and metres are not comparable).
    clearance = (abs(classification["margin_px"]) / 10.0 if _finite(classification.get("margin_px"))
                 else abs(classification["margin_m"]) / 0.12 if _finite(classification.get("margin_m")) else 0.0)
    return (strength, confidence if confidence is not None else _TIER_DEFAULT_CONFIDENCE[kind], clearance, camera == "A")


def combine_views(views, *, same_event_confirmed=False, event_source=None, different_event=False, rules=None):
    """Return the strongest available A/B verdict, always IN/OUT when any view has evidence.

    Tiers, strongest first: verified direct contact, native video estimate,
    then the per-view best estimate. A lower tier never overrides a higher one.
    Inside a tier, disagreeing views are resolved by confidence and boundary
    clearance; the disagreement stays visible as a warning. Unresolved or
    failed peers cannot veto a completed source-bound verdict, and a known
    different rally only prevents transferring one view's verdict to the other.
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
        warnings.append("KNOWN_DIFFERENT_RALLY")
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
    bests = {camera: views[camera]["best_estimate"] for camera in ("A", "B")
             if views[camera].get("status") == "completed" and valid_hashes[camera] and rules_match[camera]
             and _valid_best_estimate(views[camera].get("best_estimate"), selected_rules,
                                      views[camera].get("source_sha256"))}
    if any(isinstance(view.get("best_estimate"), dict)
           and "INVALID_BEST_ESTIMATE" in (view["best_estimate"].get("reason_codes") or [])
           for view in views.values()):
        warnings.append("INVALID_BEST_ESTIMATE")
    if len(set(qualified.values())) > 1:
        warnings.append("CONFLICTING_VIEW_DECISIONS")
    verified_estimate_conflict = [(direct_camera, estimate_camera)
                                 for direct_camera, direct in qualified.items()
                                 for estimate_camera, estimate in estimates.items() if direct != estimate]
    if any(a == b or same_event_confirmed for a, b in verified_estimate_conflict):
        warnings.append("VERIFIED_ESTIMATE_CONFLICT")
    elif verified_estimate_conflict:
        warnings.append("UNLINKED_ESTIMATE_CONFLICT")
    if len(set(estimates.values())) > 1:
        warnings.append("CONFLICTING_VIDEO_ESTIMATES")
    best_labels = {camera: best["decision"] for camera, best in bests.items()}
    if len(set(best_labels.values())) > 1:
        warnings.append("CONFLICTING_BEST_ESTIMATES")
    kind, selected = next(((name, labels) for name, labels in
                           zip(DECISION_TIERS, (qualified, estimates, best_labels)) if labels),
                          ("unresolved", {}))
    if not selected:
        reasons.append("NO_QUALIFIED_CONTACT")
    linked = (same_event_confirmed and not duplicate and all(valid_hashes.values())
              and all(rules_match.values()) and not different_event)
    if kind == "best_estimate":
        # A best estimate that reuses a higher tier the transport rejected for
        # this view must not borrow that tier's full strength.
        for camera, best in list(bests.items()):
            if ((best.get("basis") == "video_estimate" and camera not in estimates)
                    or (best.get("basis") == "verified_contact" and camera not in qualified)):
                bests[camera] = {**best, "evidence_strength": min(_finite(best.get("evidence_strength"))
                                                                  and best["evidence_strength"] or 0.6, 0.6)}
    ordered = sorted(selected, key=lambda camera: _tier_strength(camera, kind, bests), reverse=True)
    label, donors, conflict_resolved, confidence, deciding_key = "REVIEW", [], False, None, None
    if ordered:
        label = selected[ordered[0]]
        agreeing = [camera for camera in ordered if selected[camera] == label]
        conflict_resolved = len(agreeing) != len(ordered)
        donors = agreeing if linked and not conflict_resolved else agreeing[:1]
        winner = float(_tier_strength(ordered[0], kind, bests)[1])
        if conflict_resolved:
            # A disagreeing peer of the same tier lowers the final confidence.
            runner_up = next(camera for camera in ordered if selected[camera] != label)
            loser = float(_tier_strength(runner_up, kind, bests)[1])
            winner = max(0.5, winner - (loser - 0.5))
            first, second = _tier_strength(ordered[0], kind, bests), _tier_strength(runner_up, kind, bests)
            deciding_key = next((name for name, a, b in zip(
                ("evidence_strength", "confidence", "boundary_clearance", "camera_order"), first, second)
                if a != b), "camera_order")
        confidence = round(winner, 4)
    if label != "REVIEW":
        both = linked and len(selected) == 2 and not conflict_resolved
        reasons = [{"verified": "QUALIFIED_VIEWS_AGREE" if both else "VERDICT_FROM_QUALIFIED_VIEW",
                    "video_estimate": "VIDEO_ESTIMATES_AGREE" if both else "VIDEO_ESTIMATE_FROM_VIEW",
                    "best_estimate": "BEST_ESTIMATES_AGREE" if both else "BEST_ESTIMATE_FROM_VIEW"}[kind]]
        if conflict_resolved:
            reasons.append("VIEW_CONFLICT_TIE_BROKEN_BY_CAMERA_ORDER" if deciding_key == "camera_order"
                           else "VIEW_CONFLICT_RESOLVED_BY_CONFIDENCE")
    else:
        reasons = list(dict.fromkeys(reasons + warnings))
    warnings = list(dict.fromkeys(warnings))
    public_views = deepcopy(views)
    for camera, view in public_views.items():
        direct = view.get("direct_decision", "REVIEW") if camera in qualified else "REVIEW"
        # A dissenting view keeps its own verdict; only agreement is transferred.
        transfer = (label != "REVIEW" and kind == "verified" and linked and not conflict_resolved
                    and view.get("status") == "completed")
        view["assisted_decision"] = label if transfer else direct
        view["verdict_source_camera_ids"] = donors if transfer else [camera] if direct in {"IN", "OUT"} else []
        view.pop("assistance_mode", None)
        if transfer and camera not in qualified:
            view["assistance_mode"] = "other_view_verdict"
        view["estimated_decision"] = estimates.get(camera, "REVIEW")
        view["best_estimated_decision"] = best_labels.get(camera, "REVIEW")
        for key in list(view):
            if key.startswith("_"):
                view.pop(key)
    resolved = label != "REVIEW"
    result = {
        "decision": label, "reason_codes": reasons, "reasons": [REASONS[code] for code in reasons],
        "decision_kind": kind if resolved else "unresolved",
        "confidence": confidence if resolved else None,
        "warning_codes": warnings, "warnings": [REASONS[code] for code in warnings],
        "same_event_confirmed": bool(same_event_confirmed),
        "event_correspondence_source": event_source if same_event_confirmed else None,
        "source_camera_ids": donors if resolved else [],
        "decision_scope": ("linked_views" if linked else "single_view") if resolved else "unresolved",
        "views": public_views, "rules": selected_rules, "court_position_m": None,
        "assistance_used": True, "automatic_benchmark_eligible": False, "evaluation_eligible": False,
        "ground_truth_created": False, "human_verified": False,
        "video_estimate_used": resolved and kind == "video_estimate",
        "best_estimate_used": resolved and kind == "best_estimate",
        "conflict_resolved": bool(resolved and conflict_resolved),
        "conflict_resolution_key": deciding_key if resolved and conflict_resolved else None,
        "scope": "A/B verdict linkage; no 3D reconstruction or common pixel coordinates.",
    }
    _add_analysis_assessments(result)
    from .review_diagnostics import pair_diagnostics
    result["review_diagnostics"] = pair_diagnostics(result)
    if result["video_estimate_used"]:
        result["review_diagnostics"].update(
            summary="착지 가능 영역으로 영상 추정을 계산했습니다. 직접 관측한 최초 접촉은 아직 확정되지 않았습니다.",
            note="영상 추정은 관측된 경로와 원본 영역의 조건부 계산이며, 직접 접촉 확인이나 검증된 정확도를 뜻하지 않습니다.",
            decision_kind="video_estimate")
    result["refinement_available"] = any(view.get("refinement_available") is True for view in views.values())
    return result


def _revalidate_snapshot_result(result):
    """Recheck stored support before presenting a historical final verdict.

    No inference runs and no stored file is rewritten. A lost original donor,
    changed verdict kind, or new conflict can only downgrade a recorded verdict;
    opening history cannot promote REVIEW or silently switch its supporting view.
    """
    original = {key: deepcopy(result.get(key)) for key in
                ("decision", "decision_kind", "source_camera_ids", "decision_scope", "reason_codes")}
    recorded_label = original["decision"]
    recorded_kind = original["decision_kind"] or "verified"
    recorded_sources = original["source_camera_ids"]
    recorded_codes = original["reason_codes"]
    recorded_codes = recorded_codes if isinstance(recorded_codes, list) else []
    try:
        recomputed = combine_views(result["views"],
            same_event_confirmed=result.get("same_event_confirmed") is True,
            event_source=result.get("event_correspondence_source"),
            different_event=("KNOWN_DIFFERENT_RALLY" in recorded_codes
                             or "KNOWN_DIFFERENT_RALLY" in (result.get("warning_codes") or [])),
            rules=result.get("rules", RULES))
    except (KeyError, TypeError, ValueError, AttributeError, PairWorkspaceError):
        recomputed = None
    if recorded_label == "REVIEW":
        # Preserve a recorded abstention even if later code would qualify the
        # same stored evidence. New analysis is the explicit way to recompute it.
        if recomputed and recomputed["decision"] == "REVIEW":
            result.update(recomputed)
        _add_analysis_assessments(result)
        return
    donors_valid = (isinstance(recorded_sources, list) and bool(recorded_sources)
                    and all(camera in ("A", "B") for camera in recorded_sources)
                    and len(set(recorded_sources)) == len(recorded_sources) and recomputed is not None
                    and all(recomputed["views"][camera]["effective_decision"] == recorded_label
                            and recomputed["views"][camera]["effective_decision_kind"] == recorded_kind
                            for camera in recorded_sources))
    agrees = (recomputed is not None and recomputed["decision"] == recorded_label
              and recomputed["decision_kind"] == recorded_kind)
    if recorded_label in ("IN", "OUT") and donors_valid and agrees:
        result.update(recomputed)
        # A newly qualifying peer must not silently broaden historical provenance.
        result["source_camera_ids"] = recorded_sources
        result["decision_scope"] = original["decision_scope"]
        _add_analysis_assessments(result)
        return
    if recomputed is not None:
        result.update(recomputed)
    else:
        saved_views = result.get("views")
        saved_views = saved_views if isinstance(saved_views, dict) else {}
        result["views"] = {camera: saved_views[camera] if isinstance(saved_views.get(camera), dict)
                           else {"status": "failed", "error": "저장된 시점 근거를 읽을 수 없습니다."}
                           for camera in ("A", "B")}
    current_codes = result.get("reason_codes")
    codes = [code for code in current_codes if isinstance(code, str) and code in REASONS] if (
        result.get("decision") == "REVIEW" and isinstance(current_codes, list)) else []
    codes.append("STALE_REVIEW")
    if recorded_kind == "video_estimate" and not donors_valid:
        codes.append("INVALID_VIDEO_ESTIMATE")
    if recorded_kind == "best_estimate" and not donors_valid:
        codes.append("INVALID_BEST_ESTIMATE")
    codes = list(dict.fromkeys(codes))
    result.update(decision="REVIEW", decision_kind="unresolved", decision_scope="unresolved",
                  source_camera_ids=[], video_estimate_used=False, best_estimate_used=False,
                  confidence=None, conflict_resolved=False, conflict_resolution_key=None, reason_codes=codes,
                  reasons=[REASONS[code] for code in codes], historical_decision=original,
                  snapshot_revalidation={"status": "historical_verdict_rejected",
                                         "inference_rerun": False, "stored_result_modified": False})
    _add_analysis_assessments(result)
    current_views = result.get("views")
    for camera, view in (current_views if isinstance(current_views, dict) else {}).items():
        if not isinstance(view, dict):
            continue
        direct_valid = (view.get("effective_decision_kind") == "verified"
                        and view.get("effective_decision") in ("IN", "OUT"))
        view["assisted_decision"] = view["effective_decision"] if direct_valid else "REVIEW"
        view["verdict_source_camera_ids"] = [camera] if direct_valid else []
        view.pop("assistance_mode", None)
    from .review_diagnostics import pair_diagnostics
    result["review_diagnostics"] = pair_diagnostics(result)


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
        if isinstance(snapshot.get("result"), dict):
            _revalidate_snapshot_result(snapshot["result"])
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
            use_stored = reuse and entry is not None and entry["kind"] == "review" and stored_rule_match
            fallback_from_stored = False
            if use_stored and original is not None:
                if not self._review_valid(entry["row"]):
                    raise PairWorkspaceError(409, REASONS["STALE_REVIEW"])
                # An unresolved stored review gives no verdict. With the original
                # at hand, analyse it fresh so the view can still reach a
                # verified, video or best estimate; keep the review otherwise.
                fallback_from_stored = self._direct_review(entry["row"], rules) not in {"IN", "OUT"}
            if use_stored and not fallback_from_stored:
                self._stored_view(identifier, camera, base, entry["row"], rules, original)
            elif fallback_from_stored:
                from .inference import InferenceDeviceError
                from .resources import ResourceSettingsError
                attempt = dict(base, review_reuse_skipped="STORED_REVIEW_UNRESOLVED")
                try:
                    self._fresh_view(identifier, camera, attempt, original, digest, selected, refine, hint, rules)
                    base = attempt
                except (InferenceDeviceError, ResourceSettingsError):
                    raise  # settings the operator must fix; never hidden behind the stored review
                except PairWorkspaceError as error:
                    if error.message == REASONS["SOURCE_CHANGED"]:
                        raise
                    self._keep_stored_after_failed_fallback(identifier, camera, base, entry["row"], rules,
                                                            original, error)
                except Exception as error:  # the stored review remains the result, with the failure shown
                    self._keep_stored_after_failed_fallback(identifier, camera, base, entry["row"], rules,
                                                            original, error)
            else:
                if original is None:
                    raise PairWorkspaceError(409, "새 분석에 필요한 원본이 없습니다. 원본 A/B를 업로드하거나 저장된 검수를 사용하세요.")
                self._fresh_view(identifier, camera, base, original, digest, selected, refine, hint, rules)
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

    def _keep_stored_after_failed_fallback(self, identifier, camera, base, row, rules, original, error):
        LOGGER.exception("Fresh analysis after an unresolved stored review failed: job=%s camera=%s",
                         identifier, camera)
        message = f"{type(error).__name__}: {' '.join(str(error).split())[:300]}"
        base["fresh_analysis_fallback_error"] = message
        base.setdefault("evidence_warnings", []).append({
            "artifact": "fresh_analysis_fallback", "error_type": type(error).__name__,
            "message": "미해결 저장 검수 대신 시도한 새 분석이 실패해 저장 검수 결과를 유지했습니다. " + message})
        self._stored_view(identifier, camera, base, row, rules, original)

    def _stored_view(self, identifier, camera, base, row, rules, original):
        self._analysis_progress(identifier, camera, {"stage": "stored_review"})
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

    def _fresh_view(self, identifier, camera, base, original, digest, selected, refine, hint, rules):
        from .pipeline import analyze_video
        from .media import probe
        from .review_refinement import review_regions
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
        prefix = f"/pair-media/jobs/{identifier}/views/{camera}/"
        base.update(view_from_result(result, output, prefix, rules),
            review_source=("operator_region_guided_search" if manual_region is not None else
                           "ai_review_guided_search" if regions else "automatic"),
            processing_mode=("user_region_reanalysis" if manual_region is not None else
                             "review_guided_reanalysis" if regions else "fresh_analysis"))

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
