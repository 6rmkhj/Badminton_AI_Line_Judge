"""Evidence-backed AI visual review, separate from automatic model predictions.

AI review is an assisted interpretation of source frames, not independently
verified human ground truth. This module never modifies an analysis run and
never promotes its own visual annotations into automatic benchmark labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
from urllib.parse import quote

import cv2
import numpy as np

from .decision import rule_region
from .dataset import pair_identity


LABELS = {"IN", "OUT", "UNKNOWN"}
OUTER_ROLES = {"left", "right", "near", "far"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
OBSERVATION_STATUSES = {"contact_observed", "contact_interval", "no_contact_in_clip", "contact_occluded",
                        "contact_out_of_view", "target_ambiguous", "line_ambiguous"}
OBSERVATION_LABELS = {"contact_observed": "접촉 관측", "contact_interval": "접촉 구간 관측",
                      "no_contact_in_clip": "영상 내 접촉 없음", "contact_occluded": "접촉 가림",
                      "contact_out_of_view": "접촉이 화면 밖", "target_ambiguous": "대상 식별 불명확",
                      "line_ambiguous": "경계선 불명확", "missing": "검수 기록 없음"}
PAIR_CATEGORIES = ("both_resolved_same_label", "both_review", "one_review_with_decided_partner",
                   "conflicting_resolved_labels")
PAIR_CATEGORY_LABELS = {"both_resolved_same_label": "양쪽 판정 일치", "both_review": "양쪽 REVIEW",
                        "one_review_with_decided_partner": "상대는 판정됨·한쪽 REVIEW",
                        "conflicting_resolved_labels": "양쪽 판정 상충"}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(x) for x in value)


def _observation_status(annotation):
    value = annotation.get("observation_status") if isinstance(annotation, dict) else None
    return value if isinstance(value, str) and value in OBSERVATION_STATUSES else "missing"


def _points(value):
    return isinstance(value, list) and bool(value) and all(_point(point) for point in value)


def _in_frame(point, size):
    return _point(point) and 0 <= point[0] < size["width"] and 0 <= point[1] < size["height"]


def _safe_file(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or ":" in relative or Path(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts):
        raise ValueError("Evidence must use a relative path inside the project")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Evidence is missing or outside the project")
    return path


def _inside_sign(a, b, inside):
    delta = np.asarray(b, float) - a
    length = float(np.linalg.norm(delta))
    if length <= 1e-6:
        raise ValueError("Degenerate boundary line")
    value = float(delta[0] * (inside[1] - a[1]) - delta[1] * (inside[0] - a[0])) / length
    if abs(value) <= 1e-6:
        raise ValueError("inside_point_px must be off the boundary")
    return delta, length, 1 if value > 0 else -1


def _line_margins(a, b, inside, contacts):
    delta, length, sign = _inside_sign(a, b, inside)
    return [float(sign * (delta[0] * (p[1]-a[1]) - delta[1] * (p[0]-a[0])) / length)
            for p in contacts]


def judge_visual_geometry(annotation, analysis_size):
    """Compute image-space line margins without fabricating court metres.

    A contact support polygon denotes all plausible *ground-contact* positions,
    not a projected airborne shuttle trajectory. Robust OUT needs one confirmed
    outer boundary. Robust IN needs a full/appropriate-half court polygon or
    all four relevant outer boundary roles.
    """
    evidence = annotation.get("boundary_evidence")
    reasons = []
    report = {"decision": "REVIEW", "coordinate_system": "analysis_pixels",
              "reason_codes": reasons, "boundaries": [], "uncertainty_px": None}
    if not isinstance(evidence, dict):
        reasons.append("BOUNDARY_GEOMETRY_MISSING")
        return report
    contacts = annotation.get("contact_support_px")
    if contacts is None:
        contacts = [annotation.get("contact_px")]
    if not _points(contacts) or not all(_in_frame(p, analysis_size) for p in contacts):
        reasons.append("CONTACT_SUPPORT_INVALID")
        return report
    if len(contacts) > 256:
        reasons.append("CONTACT_SUPPORT_TOO_LARGE")
        return report
    contact_error, line_error = evidence.get("contact_uncertainty_px"), evidence.get("boundary_uncertainty_px")
    if not _number(contact_error) or contact_error <= 0 or not _number(line_error) or line_error < 0:
        reasons.append("GEOMETRY_UNCERTAINTY_MISSING_OR_INVALID")
        return report
    uncertainty = float(contact_error + line_error)
    report["uncertainty_px"] = uncertainty
    report["contact_uncertainty_px"] = float(contact_error)
    report["default_boundary_uncertainty_px"] = float(line_error)
    report["contact_support_px"] = deepcopy(contacts)
    if not isinstance(evidence.get("explanation"), str) or not evidence["explanation"].strip():
        reasons.append("BOUNDARY_REVIEW_EXPLANATION_MISSING")
        return report
    kind = evidence.get("kind")
    boundaries, complete_region = [], False
    try:
        if kind in {"court_polygon", "court_interior"}:
            polygon = evidence.get("polygon_px")
            if not _points(polygon) or not 3 <= len(polygon) <= 16 or not all(_in_frame(p, analysis_size) for p in polygon):
                raise ValueError("Court polygon must have 3–16 in-frame points")
            contour = np.asarray(polygon, np.float32)
            if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 4:
                raise ValueError("Court polygon must be convex and nondegenerate")
            interior = kind == "court_interior"
            if interior and evidence.get("interior_confirmed") is not True:
                raise ValueError("An interior polygon must be explicitly reviewed as wholly inside the valid court")
            region = "court_interior" if interior else evidence.get("polygon_region")
            if region not in {"full_court", "near_half", "far_half", "court_interior"}:
                raise ValueError("Court polygon must declare its region")
            if region in {"near_half", "far_half"} and evidence.get("target_contact_half") != region.removesuffix("_half"):
                reasons.append("CONTACT_OUTSIDE_DECLARED_HALF_COURT_SCOPE")
                return report
            roles = evidence.get("edge_roles")
            if interior and roles is not None:
                raise ValueError("An interior polygon cannot claim outer-boundary edge roles")
            if roles is not None and (not isinstance(roles, list) or len(roles) != len(polygon)
                                       or any(role not in OUTER_ROLES | {"net"} for role in roles)):
                raise ValueError("edge_roles must match the court polygon edges")
            if region == "full_court" and roles is not None and "net" in roles:
                raise ValueError("Full court cannot use the net as an outer edge")
            if region in {"near_half", "far_half"} and roles is not None and roles.count("net") != 1:
                raise ValueError("A half court must identify exactly one net edge")
            edge_errors = evidence.get("boundary_uncertainty_px_by_edge")
            if edge_errors is not None and (not isinstance(edge_errors, list) or len(edge_errors) != len(polygon)
                                            or any(not _number(error) or error < 0 for error in edge_errors)):
                raise ValueError("Per-edge uncertainty must match polygon edges and be finite and nonnegative")
            center = np.mean(contour, axis=0)
            for index, a in enumerate(polygon):
                role = ("interior" if interior else
                        roles[index] if roles else ("outer" if region == "full_court" else "unassigned_half_edge"))
                edge_error = edge_errors[index] if edge_errors is not None else line_error
                boundaries.append((role, a, polygon[(index+1) % len(polygon)], center, edge_error))
            complete_region = True
            report["polygon_region"] = region
        elif kind == "local_boundary":
            lines = evidence.get("boundaries")
            if not isinstance(lines, list) or not 1 <= len(lines) <= 4:
                raise ValueError("Local boundary evidence must name 1–4 lines")
            roles = set()
            for line in lines:
                if not isinstance(line, dict) or line.get("role") not in OUTER_ROLES or line.get("role_confirmed") is not True:
                    raise ValueError("Every line needs a confirmed outer-boundary role")
                role = line["role"]
                if role in roles:
                    raise ValueError("Boundary roles must be unique")
                roles.add(role)
                points, inside = line.get("line_px"), line.get("inside_point_px")
                if not _points(points) or len(points) != 2 or not all(_in_frame(p, analysis_size) for p in points) or not _in_frame(inside, analysis_size):
                    raise ValueError("Boundary and inside-reference points must be in-frame")
                edge_error = line.get("boundary_uncertainty_px", line_error)
                if not _number(edge_error) or edge_error < 0:
                    raise ValueError("Per-line uncertainty must be finite and nonnegative")
                boundaries.append((role, points[0], points[1], inside, edge_error))
            complete_region = roles == OUTER_ROLES
        else:
            raise ValueError("Unknown boundary geometry kind")
        for role, a, b, inside, edge_error in boundaries:
            margins = _line_margins(a, b, inside, contacts)
            uncertainty = float(contact_error + edge_error)
            report["boundaries"].append({"role": role, "min_inside_margin_px": min(margins),
                                          "max_inside_margin_px": max(margins),
                                          "boundary_uncertainty_px": float(edge_error),
                                          "uncertainty_px": uncertainty,
                                          "robust_inside": min(margins) > uncertainty,
                                          "robust_outside": max(margins) < -uncertainty,
                                          "is_outer_boundary": role in OUTER_ROLES | {"outer"}})
        report["uncertainty_px"] = max(row["uncertainty_px"] for row in report["boundaries"])
    except (ValueError, TypeError, KeyError) as error:
        reasons.append("BOUNDARY_GEOMETRY_INVALID")
        report["error"] = str(error)
        return report
    if any(row["robust_outside"] and row["is_outer_boundary"] for row in report["boundaries"]):
        report["decision"] = "OUT"
        reasons.append("CONTACT_SUPPORT_ROBUSTLY_OUTSIDE_OUTER_BOUNDARY")
    elif complete_region and all(row["robust_inside"] for row in report["boundaries"]):
        report["decision"] = "IN"
        reasons.append("CONTACT_SUPPORT_ROBUSTLY_INSIDE_VALID_COURT")
    else:
        reasons.append("INSUFFICIENT_BOUNDARIES_FOR_IN" if not complete_region else "CONTACT_GEOMETRY_AMBIGUOUS")
    return report


def validate_visual_annotation(annotation, result, project_root):
    """Validate provenance, observed temporal evidence, and computed geometry.

    ``valid`` refers to a structurally coherent review record. ``qualified``
    additionally requires enough observed evidence to issue an assisted IN/OUT.
    All decisions retain evaluation_eligible=False and human_verified=False.
    """
    root = Path(project_root).resolve()
    errors, reasons, evidence_files = [], [], []
    report = {"valid": False, "qualified": False, "assisted_decision": "REVIEW",
              "suggested_label": None, "errors": errors, "reason_codes": reasons,
              "evidence_files": evidence_files, "assistance_used": True,
              "evaluation_eligible": False, "human_verified": False,
              "ground_truth_eligible": False, "review_source": "ai_visual_review"}
    if not isinstance(annotation, dict):
        errors.append("ANNOTATION_NOT_AN_OBJECT")
        return report
    label = annotation.get("label")
    report["suggested_label"] = label if isinstance(label, str) and label in {"IN", "OUT"} else annotation.get("suggested_label")
    if not isinstance(report["suggested_label"], str) or report["suggested_label"] not in {"IN", "OUT"}:
        report["suggested_label"] = None
    if annotation.get("source") != "ai_visual_review" or annotation.get("status") != "reviewed_by_ai":
        errors.append("AI_REVIEW_PROVENANCE_REQUIRED")
    if annotation.get("human_verified") is not False:
        errors.append("AI_REVIEW_MUST_NOT_CLAIM_HUMAN_VERIFICATION")
    if annotation.get("video_id") != result.get("video_id") or annotation.get("file") != result.get("file"):
        errors.append("ANNOTATION_VIDEO_IDENTITY_MISMATCH")
    expected_hash = result.get("source_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or annotation.get("source_sha256") != expected_hash:
        errors.append("ANNOTATION_SOURCE_HASH_MISMATCH")
    if not isinstance(label, str) or label not in LABELS:
        errors.append("INVALID_VISUAL_LABEL")
    observation_status = _observation_status(annotation)
    if observation_status == "missing":
        errors.append("OBSERVATION_STATUS_MISSING_OR_INVALID")
    if annotation.get("target_contact_identity_confirmed") is not True:
        reasons.append("TARGET_CONTACT_IDENTITY_NOT_CONFIRMED")
    for key in ("reviewer", "rationale"):
        if not isinstance(annotation.get(key), str) or not annotation[key].strip():
            errors.append(key.upper() + "_MISSING")
    size, metadata = result.get("analysis", {}), result.get("metadata", {})
    recorded_size = annotation.get("analysis_size")
    if (not isinstance(recorded_size, dict) or any(not _integer(size.get(d)) or size[d] <= 0 or recorded_size.get(d) != size[d] for d in ("width", "height"))):
        errors.append("ANNOTATION_ANALYSIS_SIZE_MISMATCH")
    recorded_source = annotation.get("source_size")
    if (not isinstance(recorded_source, dict) or any(not _integer(metadata.get(d)) or recorded_source.get(d) != metadata[d] for d in ("width", "height"))):
        errors.append("ANNOTATION_SOURCE_SIZE_MISMATCH")
    rules = annotation.get("rules")
    try:
        if not isinstance(rules, dict):
            raise ValueError("Explicit rules required")
        rule_region(rules)
        if rules.get("phase", "rally") != "rally":
            reasons.append("SERVICE_GEOMETRY_NOT_SUPPORTED_BY_VISUAL_REVIEW")
    except (ValueError, TypeError):
        errors.append("INVALID_REVIEW_RULES")
    timestamps = result.get("timestamps", [])
    if not isinstance(timestamps, list) or not timestamps or not all(_number(t) for t in timestamps):
        errors.append("SOURCE_TIMELINE_UNAVAILABLE")
        timestamps = []
    inspected = annotation.get("inspected_frame_indices")
    if (not isinstance(inspected, list) or not inspected or
            any(not _integer(i) or not 0 <= i < len(timestamps) for i in inspected)):
        errors.append("INSPECTED_FRAMES_MISSING_OR_INVALID")
        inspected = []
    files = annotation.get("evidence_paths")
    if not isinstance(files, list) or not files:
        errors.append("VISUAL_EVIDENCE_MISSING")
    else:
        for relative in files:
            try:
                path = _safe_file(root, relative)
                if path.suffix.lower() not in IMAGE_SUFFIXES or cv2.imread(str(path)) is None:
                    raise ValueError("Evidence must be a decodable image")
                evidence_files.append(path.relative_to(root).as_posix())
            except (OSError, ValueError):
                errors.append("VISUAL_EVIDENCE_PATH_OR_IMAGE_INVALID")
    has_observation = annotation.get("contact_observed") is True
    interval_observation = annotation.get("contact_interval_observed") is True
    if observation_status == "contact_observed" and not has_observation:
        errors.append("OBSERVATION_STATUS_DISAGREES_WITH_CONTACT_EVIDENCE")
    if observation_status == "contact_interval" and not interval_observation:
        errors.append("OBSERVATION_STATUS_DISAGREES_WITH_CONTACT_EVIDENCE")
    if observation_status in {"no_contact_in_clip", "contact_occluded", "contact_out_of_view", "target_ambiguous"} and (has_observation or interval_observation):
        reasons.append("OBSERVATION_STATUS_DOES_NOT_SUPPORT_TARGET_CONTACT")
    frame, timestamp = annotation.get("contact_frame_index"), annotation.get("time")
    interval = annotation.get("contact_interval_frames")
    if has_observation or interval_observation:
        if not _integer(frame) or not 0 <= frame < len(timestamps):
            errors.append("CONTACT_FRAME_INVALID")
        elif not _number(timestamp) or abs(timestamp-timestamps[frame]) > 1e-4:
            errors.append("CONTACT_TIME_DOES_NOT_MATCH_SOURCE_FRAME")
        if not isinstance(recorded_size, dict) or not all(_integer(recorded_size.get(d)) for d in ("width", "height")):
            pass
        elif annotation.get("contact_px") is not None and not _in_frame(annotation["contact_px"], recorded_size):
            errors.append("CONTACT_POINT_OUTSIDE_ANALYSIS_FRAME")
        if interval_observation:
            if (not isinstance(interval, list) or len(interval) != 2 or
                    any(not _integer(i) for i in interval) or not 0 <= interval[0] <= interval[1] < len(timestamps)):
                errors.append("CONTACT_INTERVAL_INVALID")
            elif not _integer(frame) or not interval[0] <= frame <= interval[1]:
                errors.append("CONTACT_FRAME_OUTSIDE_INTERVAL")
            elif interval[0] == 0 or interval[1] == len(timestamps)-1:
                reasons.append("CONTACT_INTERVAL_NOT_TEMPORALLY_BRACKETED")
            elif not all(i in inspected for i in range(max(0, interval[0]-1), min(len(timestamps), interval[1]+2))):
                reasons.append("CONTACT_INTERVAL_ADJACENT_FRAMES_NOT_INSPECTED")
            if not _points(annotation.get("contact_support_px")):
                reasons.append("CONTACT_INTERVAL_SUPPORT_MISSING")
        elif _integer(frame) and 0 <= frame < len(timestamps):
            if frame == 0 or frame == len(timestamps)-1 or not {frame-1, frame, frame+1}.issubset(set(inspected)):
                reasons.append("CONTACT_ADJACENT_FRAMES_NOT_INSPECTED")
    else:
        reasons.append("GROUND_CONTACT_NOT_OBSERVED")
    if annotation.get("label") == "UNKNOWN":
        reasons.append("VISUAL_REVIEW_UNRESOLVED")
    report["valid"] = not errors
    if not errors and (has_observation or interval_observation):
        geometry = judge_visual_geometry(annotation, recorded_size)
        report["geometry"] = geometry
        if geometry["decision"] not in {"IN", "OUT"}:
            reasons.extend(geometry["reason_codes"])
        elif annotation.get("label") != geometry["decision"]:
            reasons.append("VISUAL_LABEL_DISAGREES_WITH_COMPUTED_GEOMETRY")
        elif not reasons:
            report.update(qualified=True, assisted_decision=geometry["decision"])
            reasons.extend(geometry["reason_codes"])
    report["reason_codes"] = list(dict.fromkeys(reasons))
    report["errors"] = list(dict.fromkeys(errors))
    return report


def _read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _automatic_label(result):
    decision = result.get("decision")
    return decision if isinstance(decision, str) else (decision or {}).get("decision", "ERROR")


def _copy_asset(path, output, relative):
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return relative.as_posix()


def _verify_source_file(result, root):
    """Rehash source bytes; an annotation matching stale metadata is insufficient."""
    source = Path(result.get("source_path") or root / "video" / result.get("file", ""))
    if not source.is_file():
        source = root / "video" / result.get("file", "")
    source = source.resolve()
    if not source.is_relative_to(root) or not source.is_file():
        return {"verified": False, "reason": "ORIGINAL_SOURCE_MISSING_OR_OUTSIDE_PROJECT"}
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    matches = actual == result.get("source_sha256")
    return {"verified": matches, "sha256": actual, "path": source.relative_to(root).as_posix(),
            "reason": None if matches else "ORIGINAL_SOURCE_CHANGED_SINCE_ANALYSIS"}


def _apply_cross_view_review(records, results, root, output):
    """Transfer only directly qualified source evidence across reviewed events.

    Frame/action anchors demonstrate an explicitly reviewed correspondence;
    this does not estimate synchronized clocks or local-view contact pixels.
    The donor must have its own contact evidence, never another transferred call.
    """
    by_id = {row["video_id"]: row for row in records}
    proposals, conflicts = [], set()
    for row in records:
        annotation = row.get("annotation") or {}
        match = annotation.get("cross_view_match")
        if match is None:
            continue
        reasons, images, timing_check = [], [], None
        donor_id = match.get("other_video_id") if isinstance(match, dict) else None
        donor = by_id.get(donor_id) if isinstance(donor_id, str) else None
        if (not isinstance(match, dict) or match.get("same_event_confirmed_by_ai") is not True
                or not isinstance(match.get("rationale"), str) or not match["rationale"].strip()):
            reasons.append("CROSS_VIEW_MATCH_NOT_EXPLICITLY_REVIEWED")
        if donor is None or donor is row:
            reasons.append("CROSS_VIEW_DONOR_INVALID")
        elif donor["source_sha256"] == row["source_sha256"]:
            reasons.append("CROSS_VIEW_DUPLICATE_CONTENT_IS_NOT_ANOTHER_VIEW")
        if not row["review"]["valid"] or not row["source_integrity"]["verified"]:
            reasons.append("CROSS_VIEW_RECEIVER_RECORD_INVALID")
        if donor:
            if (not donor["review"]["valid"] or not donor["source_integrity"]["verified"]
                    or donor["direct_assisted_decision"] not in {"IN", "OUT"}
                    or (donor.get("annotation") or {}).get("target_contact_identity_confirmed") is not True):
                reasons.append("CROSS_VIEW_DONOR_HAS_NO_DIRECT_QUALIFIED_CONTACT")
            if annotation.get("rules") != (donor.get("annotation") or {}).get("rules"):
                reasons.append("CROSS_VIEW_RULES_MISMATCH")
        anchors = match.get("matched_actions") if isinstance(match, dict) else None
        if not isinstance(anchors, list) or not 2 <= len(anchors) <= 100:
            reasons.append("CROSS_VIEW_NEEDS_TWO_DISTINCT_REVIEWED_ACTIONS")
        elif donor:
            own_inspected = set(annotation.get("inspected_frame_indices", []))
            donor_inspected = set((donor.get("annotation") or {}).get("inspected_frame_indices", []))
            own_times = results[row["video_id"]].get("timestamps", [])
            donor_times = results[donor["video_id"]].get("timestamps", [])
            pairs, descriptions = [], set()
            for anchor in anchors:
                if not isinstance(anchor, dict):
                    reasons.append("CROSS_VIEW_ACTION_ANCHOR_INVALID")
                    continue
                own_frame, other_frame = anchor.get("frame_index"), anchor.get("other_frame_index")
                description = anchor.get("description")
                if (not _integer(own_frame) or not _integer(other_frame)
                        or not 0 <= own_frame < len(own_times) or not 0 <= other_frame < len(donor_times)
                        or own_frame not in own_inspected or other_frame not in donor_inspected
                        or not isinstance(description, str) or not description.strip()):
                    reasons.append("CROSS_VIEW_ACTION_NOT_IN_INSPECTED_SOURCE_FRAMES")
                    continue
                pairs.append((own_frame, other_frame))
                descriptions.add(description.strip())
            ordered = sorted(pairs)
            if (len(set(pairs)) < 2 or len(descriptions) < 2
                    or len({p[0] for p in pairs}) != len(pairs)
                    or len({p[1] for p in pairs}) != len(pairs)
                    or any(b[1] <= a[1] for a, b in zip(ordered, ordered[1:]))):
                reasons.append("CROSS_VIEW_ACTIONS_NOT_DISTINCT_AND_TEMPORALLY_ORDERED")
            if len(pairs) >= 2:
                def period(times):
                    positive = [b-a for a, b in zip(times, times[1:]) if b > a]
                    return float(statistics.median(positive)) if positive else 0.0
                own_period, donor_period = period(own_times), period(donor_times)
                allowance = max(0.15, 4 * max(own_period, donor_period))
                offsets = [float(donor_times[other]-own_times[own]) for own, other in pairs]
                spread = max(offsets)-min(offsets)
                timing_check = {"action_offsets_seconds": offsets, "offset_spread_seconds": spread,
                                "allowed_spread_seconds": allowance,
                                "median_frame_period_seconds": {"receiver": own_period, "donor": donor_period},
                                "clock_synchronization_claimed": False,
                                "scope": "Reject inconsistent action correspondence; a constant unknown clock/clip offset is allowed."}
                if spread > allowance + 1e-6:
                    reasons.append("CROSS_VIEW_ACTION_TIMING_INCONSISTENT")
        evidence = match.get("evidence_paths") if isinstance(match, dict) else None
        if not isinstance(evidence, list) or not evidence:
            reasons.append("CROSS_VIEW_CORRESPONDENCE_IMAGES_MISSING")
        else:
            for relative in evidence:
                try:
                    path = _safe_file(root, relative)
                    if path.suffix.lower() not in IMAGE_SUFFIXES or cv2.imread(str(path)) is None:
                        raise ValueError("Invalid correspondence image")
                    images.append(path)
                except (OSError, ValueError):
                    reasons.append("CROSS_VIEW_CORRESPONDENCE_IMAGE_INVALID")
        if donor and not reasons and row["direct_assisted_decision"] in {"IN", "OUT"} and row["direct_assisted_decision"] != donor["direct_assisted_decision"]:
            reasons.append("CROSS_VIEW_QUALIFIED_CONTACTS_CONFLICT")
            conflicts.update((row["video_id"], donor["video_id"]))
        proposals.append((row, donor, match, reasons, images, timing_check))
    for row, donor, match, reasons, images, timing_check in proposals:
        if row["video_id"] in conflicts or donor and donor["video_id"] in conflicts:
            reasons.append("CROSS_VIEW_QUALIFIED_CONTACTS_CONFLICT")
        reasons = list(dict.fromkeys(reasons))
        row["cross_view_review"] = {"qualified": not reasons, "reason_codes": reasons,
                                    "originating_video_id": donor["video_id"] if donor else None,
                                    "clock_synchronization_claimed": False, "action_timing_check": timing_check}
        for index, image in enumerate(images):
            asset = _copy_asset(image, output, Path("assets") / row["video_id"] / f"cross-view-{index:03d}{image.suffix.lower()}")
            row["assets"]["evidence"].append({"path": asset, "source": image.relative_to(root).as_posix(), "kind": "cross_view_correspondence"})
        if reasons:
            if row["direct_assisted_decision"] in {"IN", "OUT"} and row["video_id"] not in conflicts:
                # A second view may not observe the contact at all. Its failed
                # optional corroboration cannot erase a valid direct observation.
                row["cross_view_review"]["optional_correspondence_ignored"] = True
                continue
            row["assisted_decision"] = "REVIEW"
            row["review"].update(qualified=False, assisted_decision="REVIEW")
            row["review"]["reason_codes"].extend(reasons)
            continue
        if row["direct_assisted_decision"] in {"IN", "OUT"}:
            row["cross_view_review"]["note"] = "Matching direct judgments; no contact evidence transferred."
            continue
        origin = donor["annotation"]
        row["assisted_decision"] = donor["direct_assisted_decision"]
        row["assistance_mode"] = "cross_view_ai_review"
        row["originating_video_id"] = donor["video_id"]
        row["donor_contact"] = {key: deepcopy(origin.get(key)) for key in (
            "contact_observed", "contact_interval_observed", "contact_frame_index", "contact_interval_frames",
            "time", "contact_px", "contact_support_px", "analysis_size", "source_size", "source_sha256")}
        row["donor_contact"]["coordinate_view"] = donor["video_id"]
        row["review"].update(qualified=True, assisted_decision=row["assisted_decision"])
        row["review"]["direct_reason_codes"] = row["review"]["reason_codes"]
        row["review"]["reason_codes"] = ["SAME_EVENT_CONFIRMED_BY_REVIEWED_ACTIONS", "CONTACT_VERDICT_FROM_DIRECTLY_OBSERVED_OTHER_VIEW"]
        row["cross_view_review"]["matched_actions"] = deepcopy(match["matched_actions"])
        row["cross_view_review"]["donor_geometry"] = deepcopy(donor["review"].get("geometry"))
    for identifier in conflicts:
        row = by_id[identifier]
        row["assisted_decision"] = "REVIEW"
        row["review"].update(qualified=False, assisted_decision="REVIEW")
        if "CROSS_VIEW_QUALIFIED_CONTACTS_CONFLICT" not in row["review"]["reason_codes"]:
            row["review"]["reason_codes"].append("CROSS_VIEW_QUALIFIED_CONTACTS_CONFLICT")


def audit_filename_pairs(records):
    """Expose missed opposite-view follow-ups without propagating filenames.

    A filename pair is a search candidate only. Accepted event correspondence
    must still come from the separately validated visual action observations.
    This diagnostic does not mutate or combine any video decision.
    """
    groups, hashes, pair_by_id = defaultdict(list), defaultdict(list), {}
    unpaired = []
    for row in records:
        filename = row.get("file")
        key, view = pair_identity(Path(filename).name) if isinstance(filename, str) else (None, None)
        if key is not None:
            key = (Path(filename).parent / key).as_posix()
            groups[key].append((view, row))
            pair_by_id[row["video_id"]] = key
        else:
            unpaired.append(row["video_id"])
        digest = row.get("source_sha256")
        if isinstance(digest, str) and digest:
            hashes[digest].append(row)

    def correspondence(receiver, partner):
        annotation = receiver.get("annotation") or {}
        match = annotation.get("cross_view_match")
        review = receiver.get("cross_view_review") or {}
        target_id = partner["video_id"]
        result = {"receiver_video_id": receiver["video_id"], "candidate_partner_video_id": target_id,
                  "status": "missing", "validated": False, "reason_codes": [], "rationale": ""}
        if isinstance(match, dict):
            if match.get("other_video_id") != target_id:
                result.update(status="points_to_another_video", linked_video_id=match.get("other_video_id"))
            else:
                result["rationale"] = str(match.get("rationale") or "")
                result["reason_codes"] = list(review.get("reason_codes") or [])
                accepted = (match.get("same_event_confirmed_by_ai") is True
                            and review.get("qualified") is True
                            and review.get("originating_video_id") == target_id)
                result.update(status="accepted" if accepted else "rejected" if review else "not_validated",
                              validated=accepted)
        candidates = annotation.get("cross_view_candidates")
        if isinstance(candidates, list):
            examined = next((candidate for candidate in candidates if isinstance(candidate, dict)
                             and candidate.get("other_video_id") == target_id), None)
            if examined is not None:
                result["examined_candidate_status"] = examined.get("status")
                result["examined_candidate_rationale"] = str(examined.get("rationale") or "")
                if result["status"] == "missing":
                    result.update(status="rejected_different_event" if examined.get("status") == "different_rally"
                                  and examined.get("same_event_confirmed_by_ai") is False else "examined_not_confirmed",
                                  rationale=str(examined.get("rationale") or ""))
        return result

    items, ambiguous_groups = [], []
    for key, members in sorted(groups.items()):
        if len(members) != 2 or {view for view, _ in members} != {"a", "b"}:
            ambiguous_groups.append({"pair_key": key, "video_ids": [row["video_id"] for _, row in members],
                                     "reason": "Expected exactly one A and one B filename"})
            unpaired.extend(row["video_id"] for _, row in members)
            continue
        members.sort(key=lambda value: value[0])
        left, right = members[0][1], members[1][1]
        labels = [row.get("assisted_decision", "REVIEW") for row in (left, right)]
        resolved = [value in {"IN", "OUT"} for value in labels]
        if all(resolved):
            category = "both_resolved_same_label" if labels[0] == labels[1] else "conflicting_resolved_labels"
        elif not any(resolved):
            category = "both_review"
        else:
            category = "one_review_with_decided_partner"
        views = [{"video_id": row["video_id"], "file": row["file"], "view": view,
                  "assisted_decision": row.get("assisted_decision", "REVIEW"),
                  "direct_assisted_decision": row.get("direct_assisted_decision", "REVIEW"),
                  "source_sha256": row.get("source_sha256")} for view, row in members]
        correspondences = [correspondence(left, right), correspondence(right, left)]
        duplicates = []
        for _, row in members:
            for duplicate in hashes.get(row.get("source_sha256"), []):
                other_pair = pair_by_id.get(duplicate["video_id"])
                if other_pair is not None and other_pair != key:
                    duplicates.append({"video_id": row["video_id"], "duplicate_video_id": duplicate["video_id"],
                                       "duplicate_file": duplicate["file"], "other_pair_key": other_pair})
        item = {"pair_key": key, "category": category, "views": views,
                "event_correspondence_validated": any(value["validated"] for value in correspondences),
                "correspondence": correspondences,
                "duplicate_content_across_pairs": bool(duplicates), "cross_pair_duplicates": duplicates,
                "duplicate_content_within_pair": bool(left.get("source_sha256"))
                                                   and left.get("source_sha256") == right.get("source_sha256"),
                "donor_candidate": None, "review_video_id": None}
        if category == "one_review_with_decided_partner":
            receiver, donor = (right, left) if resolved[0] else (left, right)
            item["review_video_id"] = receiver["video_id"]
            item["donor_candidate"] = {"video_id": donor["video_id"], "file": donor["file"],
                                       "assisted_decision": donor["assisted_decision"],
                                       "direct_assisted_decision": donor.get("direct_assisted_decision", "REVIEW")}
            item["receiver_correspondence"] = next(value for value in correspondences
                                                   if value["receiver_video_id"] == receiver["video_id"])
        items.append(item)
    counts = Counter(item["category"] for item in items)
    return {"candidate_pair_count": len(items), "counts": {category: counts[category] for category in PAIR_CATEGORIES},
            "both_review_video_count": 2 * counts["both_review"],
            "mixed_review_video_count": counts["one_review_with_decided_partner"],
            "event_correspondence_validated_pair_count": sum(item["event_correspondence_validated"] for item in items),
            "duplicate_content_across_pair_count": sum(item["duplicate_content_across_pairs"] for item in items),
            "unpaired_video_ids": sorted(unpaired), "ambiguous_filename_groups": ambiguous_groups,
            "filename_only_propagation": False,
            "scope": "Filename pairs are candidates for review, not proof of the same event. This audit never changes decisions.",
            "items": items}


def build_visual_review_report(run_dir, annotations_dir, output_dir, *, project_root):
    """Write a portable assisted-review overlay, leaving analysis immutable."""
    root, run, annotations, output = (Path(p).resolve() for p in (project_root, run_dir, annotations_dir, output_dir))
    if run == output or run.is_relative_to(output) or output.is_relative_to(run):
        raise ValueError("Visual review output must be separate from the analysis run")
    for path in (run, annotations, output):
        if not path.is_relative_to(root):
            raise ValueError("Review paths must stay inside the project")
    index = _read_json(run / "index.json")
    output.mkdir(parents=True, exist_ok=True)
    records, seen, results = [], set(), {}
    for entry in index.get("videos", []):
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in seen:
            raise ValueError("Run video identifiers must be safe and unique")
        seen.add(identifier)
        relative_result = entry.get("result_path") or f"videos/{identifier}/result.json"
        result_path = _safe_file(run, relative_result)
        result = _read_json(result_path)
        results[identifier] = result
        annotation_path = annotations / (identifier + ".json")
        annotation, review = None, None
        if annotation_path.exists():
            try:
                if not annotation_path.resolve().is_relative_to(annotations):
                    raise ValueError("Annotation escapes its directory")
                annotation = _read_json(annotation_path)
                review = validate_visual_annotation(annotation, result, root)
                if not isinstance(annotation, dict):
                    annotation = None
            except (ValueError, OSError) as error:
                review = {"valid": False, "qualified": False, "assisted_decision": "REVIEW",
                          "errors": ["ANNOTATION_UNREADABLE"], "error": str(error), "reason_codes": [],
                          "evidence_files": [], "suggested_label": None}
        else:
            review = {"valid": False, "qualified": False, "assisted_decision": "REVIEW",
                      "errors": ["ANNOTATION_MISSING"], "reason_codes": [], "evidence_files": [], "suggested_label": None}
        source_integrity = _verify_source_file(result, root)
        if not source_integrity["verified"]:
            review.update(valid=False, qualified=False, assisted_decision="REVIEW")
            review["errors"].append(source_integrity["reason"])
        record = {"video_id": identifier, "file": result.get("file"), "source_sha256": result.get("source_sha256"),
                  "automatic_decision": _automatic_label(result), "assisted_decision": review["assisted_decision"],
                  "direct_assisted_decision": review["assisted_decision"], "assistance_mode": "direct_ai_visual_review",
                  "review": review, "annotation": annotation, "assistance_used": True,
                  "evaluation_eligible": False, "ground_truth_eligible": False,
                  "human_verified": False, "assets": {"evidence": []},
                  "source_integrity": source_integrity,
                  "original_result": result_path.relative_to(root).as_posix()}
        preview = result.get("artifacts", {}).get("preview")
        if preview:
            path = _safe_file(result_path.parent, preview)
            record["assets"]["preview"] = _copy_asset(path, output, Path("assets") / identifier / "preview.mp4")
        for number, relative in enumerate(review["evidence_files"]):
            path = _safe_file(root, relative)
            asset = _copy_asset(path, output, Path("assets") / identifier / f"evidence-{number:03d}{path.suffix.lower()}")
            record["assets"]["evidence"].append({"path": asset, "source": relative})
        records.append(record)
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["source_sha256"]].append(record)
    contradictions = []
    for digest, group in by_hash.items():
        labels = {r["assisted_decision"] for r in group} & {"IN", "OUT"}
        if len(labels) > 1:
            contradictions.append({"source_sha256": digest, "video_ids": [r["video_id"] for r in group]})
            for record in group:
                record["assisted_decision"] = "REVIEW"
                record["direct_assisted_decision"] = "REVIEW"
                record["review"].update(qualified=False, assisted_decision="REVIEW")
                record["review"]["reason_codes"].append("DUPLICATE_CONTENT_LABEL_CONFLICT")
    _apply_cross_view_review(records, results, root, output)
    counts = Counter(record["assisted_decision"] for record in records)
    report = {"schema_version": 1, "review_source": "ai_visual_review", "status": "completed",
              "video_count": len(records), "reviewed_count": sum(r["annotation"] is not None for r in records),
              "valid_review_count": sum(r["review"]["valid"] for r in records),
              "source_hash_verified_count": sum(r["source_integrity"]["verified"] for r in records),
              "assisted_counts": {label: counts[label] for label in ("IN", "OUT", "REVIEW")},
              "automatic_counts": dict(Counter(r["automatic_decision"] for r in records)),
              "direct_assisted_counts": dict(Counter(r["direct_assisted_decision"] for r in records)),
              "cross_view_transferred_count": sum(r.get("assistance_mode") == "cross_view_ai_review" for r in records),
              "observation_status_counts": dict(Counter(_observation_status(r.get("annotation")) for r in records)),
              "duplicate_conflicts": contradictions, "ground_truth_eligible": False,
              "filename_pair_audit": audit_filename_pairs(records),
              "automatic_accuracy": None, "independent_human_ground_truth_count": 0,
              "scope": "AI visual assistance over source evidence; not independent ground truth or automatic accuracy.",
              "videos": records}
    (output / "visual-review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output / "index.html").write_text(_render_html(report), encoding="utf-8")
    return report


def _render_html(report):
    def esc(value):
        return html.escape(str(value if value is not None else ""), quote=True)
    def seek_button(identifier, timestamp, label):
        if not _number(timestamp):
            return ""
        return (f'<button class="seek-contact" data-seek-video="{esc(identifier)}" '
                f'data-seek-time="{timestamp:.6f}">{esc(label)} {timestamp:.3f}초로 이동</button>')
    cards = []
    for row in report["videos"]:
        annotation = row.get("annotation") or {}
        review = row["review"]
        preview = row["assets"].get("preview")
        video = f'<video controls preload="metadata" src="{quote(preview)}"></video>' if preview else ""
        evidence = "".join(f'<a href="{quote(asset["path"])}"><img loading="lazy" src="{quote(asset["path"])}" alt="검수 근거"></a>' for asset in row["assets"]["evidence"])
        geometry = review.get("geometry", {})
        transfer = ""
        if row.get("assistance_mode") == "cross_view_ai_review":
            origin = row["originating_video_id"]
            geometry = row.get("cross_view_review", {}).get("donor_geometry") or {}
            transfer = (f'<p>동일 사건을 확인한 다른 시점의 접촉 근거로 판정했습니다. '
                        f'<a class="donor-link" href="#{esc(origin)}">접촉을 관측한 영상</a> · 그 영상의 접촉 시각 '
                        f'{esc(row["donor_contact"].get("time"))}초. 아래 접촉 좌표·여유 값은 해당 시점 기준이며 '
                        '이 영상에서 접촉을 관측했다고 표시하지 않습니다.</p>'
                        + seek_button(origin, row["donor_contact"].get("time"), "다른 시점의 접촉"))
        margins = "; ".join(f'{b["role"]}: {b["min_inside_margin_px"]:.2f}…{b["max_inside_margin_px"]:.2f}px / 불확실성 {b["uncertainty_px"]:.2f}px' for b in geometry.get("boundaries", []))
        issues = ", ".join(review.get("errors", []) + review.get("reason_codes", []))
        limitations = annotation.get("limitations", [])
        limitation_text = "; ".join(map(str, limitations)) if isinstance(limitations, list) else str(limitations)
        contact_control = ""
        if annotation.get("contact_observed") is True or annotation.get("contact_interval_observed") is True:
            contact_control = seek_button(row["video_id"], annotation.get("time"), "관측한 접촉")
        observation = OBSERVATION_LABELS[_observation_status(annotation)]
        cards.append(f'''<article id="{esc(row['video_id'])}" data-decision="{esc(row['assisted_decision'])}"><h2>{esc(row['file'])}</h2>
<div class="verdict">AI 보조 판정 <strong>{esc(row['assisted_decision'])}</strong> · 기존 자동 판정 {esc(row['automatic_decision'])}</div>
{transfer}<p>{esc(annotation.get('rationale', '검수 기록 없음'))}</p>
<p class="muted">이 영상의 관측 상태: {esc(observation)} · 참고 제안: {esc(review.get('suggested_label') or '없음')} · 접촉 프레임: {esc(annotation.get('contact_frame_index'))}</p>
<p class="muted">{esc(limitation_text)}</p>{contact_control}{video}<div class="evidence">{evidence}</div>
<details><summary>판정 근거·출처</summary><p>{esc(issues)}</p><p>안쪽 여유: {esc(margins)} · 경계별 합산 불확실성의 최댓값 {esc(geometry.get('uncertainty_px'))}px</p>
<p>검수자 {esc(annotation.get('reviewer'))} · SHA256 {esc(row['source_sha256'])}</p></details></article>''')
    counts = report["assisted_counts"]
    observations = " · ".join(f'{OBSERVATION_LABELS.get(key, key)} {value}개' for key, value in report.get("observation_status_counts", {}).items())
    pair_audit = report.get("filename_pair_audit") or {"candidate_pair_count": 0, "counts": {}, "items": []}
    pair_summary = " · ".join(f'{PAIR_CATEGORY_LABELS[key]} {pair_audit["counts"].get(key, 0)}쌍' for key in PAIR_CATEGORIES)
    correspondence_labels = {"missing": "사건 대응 검수 누락", "points_to_another_video": "다른 영상과 대응됨",
                             "rejected": "사건 대응 검증 실패", "not_validated": "사건 대응 검증 대기",
                             "accepted": "사건 대응 검증 통과", "rejected_different_event": "다른 랠리로 확인",
                             "examined_not_confirmed": "검수했으나 대응 미확정"}
    def pair_link(view):
        return (f'<a class="pair-link" href="#{esc(view["video_id"])}">{esc(view["file"])}</a>'
                f' ({esc(view["assisted_decision"])})')
    mixed_rows, all_pair_rows = [], []
    for pair in pair_audit["items"]:
        views = " ↔ ".join(pair_link(view) for view in pair["views"])
        duplicate_note = " · 다른 번호쌍의 영상과 원본 내용 중복" if pair["duplicate_content_across_pairs"] else ""
        confirmed_note = "사건 대응 검증 통과" if pair["event_correspondence_validated"] else "사건 대응 미확정"
        all_pair_rows.append(f'<li data-pair-category="{esc(pair["category"])}">{views}<br>'
                             f'{esc(PAIR_CATEGORY_LABELS[pair["category"]])} · {esc(confirmed_note + duplicate_note)}</li>')
        if pair["category"] == "one_review_with_decided_partner":
            correspondence = pair["receiver_correspondence"]
            status = correspondence_labels.get(correspondence["status"], correspondence["status"])
            reasons = ", ".join(correspondence.get("reason_codes", []))
            detail = correspondence.get("rationale") or reasons or "상대 영상의 판정을 사용하기 전에 같은 사건인지 직접 확인해야 합니다."
            mixed_rows.append(f'<li data-mixed-review="{esc(pair["review_video_id"])}">{views}<br>'
                              f'<strong>{esc(status + duplicate_note)}</strong><p class="muted">{esc(detail)}</p></li>')
    mixed = "<ul>" + "".join(mixed_rows) + "</ul>" if mixed_rows else "<p>상대 영상은 판정됐지만 한쪽만 REVIEW인 후보 쌍은 없습니다.</p>"
    pair_section = f'''<section id="filename-pair-audit"><h2>반대 시점 대조 현황</h2>
<p>파일명으로 묶은 후보 {pair_audit['candidate_pair_count']}쌍입니다. 같은 번호는 같은 사건의 증거가 아닙니다. 동작 대응 검증을 통과한 경우에만 다른 시점의 판정을 연결합니다.</p>
<p id="pair-counts">{esc(pair_summary)}</p><h3>상대는 판정됐지만 한쪽만 REVIEW: {pair_audit['counts'].get('one_review_with_decided_partner', 0)}쌍</h3>{mixed}
<details><summary>후보 쌍 전체와 대응 검증 상태</summary><ul>{''.join(all_pair_rows)}</ul></details></section>'''
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>영상 AI 검수 결과</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#10151c;color:#e8edf4;line-height:1.55}}main{{max-width:1120px;margin:auto;padding:24px}}header,article{{background:#1b2431;border:1px solid #344256;padding:20px;border-radius:12px;margin:16px 0}}h1,h2{{margin-top:0}}h2,p,li{{overflow-wrap:anywhere}}a{{color:#8fc7ff}}video{{display:block;width:100%;max-height:570px;background:black}}.evidence{{display:flex;gap:10px;overflow:auto;margin-top:14px}}.evidence img{{height:150px;max-width:none}}.muted{{color:#b4c1d3;font-size:.92rem}}.verdict{{font-size:1.15rem}}strong{{color:#9fdfbd}}details{{overflow-wrap:anywhere}}#filename-pair-audit{{border:1px solid #61728a;border-radius:8px;padding:16px;margin:16px 0}}#filename-pair-audit h2{{font-size:1.15rem}}#filename-pair-audit h3{{font-size:1rem}}#filename-pair-audit li{{margin-bottom:12px}}select,button{{font:inherit;background:#253346;color:white;padding:7px;border:1px solid #66809e;border-radius:5px}}button{{cursor:pointer;margin:0 0 10px}}@media(max-width:600px){{main{{padding:10px}}header,article{{padding:12px}}#filename-pair-audit{{padding:10px}}}}
</style><main><header><h1>영상 AI 검수 결과</h1><p>총 {report['video_count']}개 · IN {counts['IN']} · OUT {counts['OUT']} · REVIEW {counts['REVIEW']}</p>
<p>원본 프레임을 보고 작성한 AI 보조 검수입니다. 사람의 독립 정답 검수나 무보조 자동 모델 정확도로 계산하지 않습니다. 근거가 충분하지 않은 영상은 REVIEW를 유지합니다. 다른 시점의 관측으로 판정한 영상: {report['cross_view_transferred_count']}개.</p>
<p id="observation-counts" class="muted">{esc(observations)}</p>
{pair_section}
<label>표시 <select id="filter"><option value="all">전체</option><option>IN</option><option>OUT</option><option>REVIEW</option></select></label> · <a href="visual-review.json">전체 검수 JSON</a></header>{''.join(cards)}</main>
<script>
const filter=document.querySelector('#filter');
filter.addEventListener('change',e=>{{for(const el of document.querySelectorAll('article'))el.hidden=e.target.value!=='all'&&el.dataset.decision!==e.target.value;}});
function reveal(id){{const article=document.getElementById(id);if(!article)return null;if(article.hidden){{filter.value='all';filter.dispatchEvent(new Event('change'));}}article.scrollIntoView({{block:'start'}});return article;}}
document.addEventListener('click',e=>{{const button=e.target.closest('button[data-seek-video]');if(button){{const article=reveal(button.dataset.seekVideo);const video=article?.querySelector('video');if(!video)return;const time=Number(button.dataset.seekTime);const seek=()=>{{video.pause();video.currentTime=Math.min(time,Math.max(0,video.duration-0.001));}};if(video.readyState>=1)seek();else{{video.addEventListener('loadedmetadata',seek,{{once:true}});video.load();}}}}const link=e.target.closest('a.donor-link, a.pair-link');if(link)reveal(link.hash.slice(1));}});
</script></html>'''
