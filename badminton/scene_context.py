"""Optional soft review ranking from actual-source background appearance.

Bottom connectivity and similar color do not identify a physical floor. This
helper only annotates unconfirmed image context; it never edits observations,
candidate geometry, flight eligibility, identity, guidance, or contact claims.
The caller controls enablement and must preserve the existing review tiers.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import cv2
import numpy as np

SCHEMA_VERSION = 1
METHOD = "bottom_connected_source_context_v1"
PARAMETERS = {
    "sample_count": 7, "segmentation_width": 480, "bottom_seed_fraction": .02,
    "lab_luminance_weight": .35, "palette_radius_seed_quantile": .95,
    "palette_radius_multiplier": 2., "palette_radius_min": 12., "palette_radius_max": 28.,
    "morphology_close_open_kernel": 5, "minimum_bottom_coverage": .6,
    "accepted_region_area_range": [.15, .8],
    "maximum_registration_shift_diagonal_fraction": .015, "minimum_phase_response": .35,
    "distance_margin_height_fraction": .05, "maximum_relative_penalty": .5,
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


PARAMETERS_SHA256 = _digest(PARAMETERS)


def _frame_digest(frame):
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _finite(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def _integer(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _explicit_evidence(candidate):
    requested = candidate.get("identity_source") == "operator_selected_track" or candidate.get("request_match") == "selected"
    observed = (candidate.get("contact_observed") is True and
                candidate.get("contact_evidence_source") in ("human_review", "qualified_learned_contact_localization"))
    return requested or observed


def _geometry(candidate, count, shape):
    index = candidate.get("frame_index")
    point = candidate.get("candidate_px")
    if not _integer(index) or not 0 <= index < count:
        return None
    if (not isinstance(point, (list, tuple, np.ndarray)) or np.ndim(point) != 1
            or len(point) != 2 or not all(_finite(v) for v in point)):
        return None
    height, width = shape[:2]
    x, y = map(float, point)
    if not 0 <= x < width or not 0 <= y < height:
        return None
    result = {"frame_index": int(index), "point_px": [x, y], "time_present": "time" in candidate}
    if "time" in candidate:
        if not _finite(candidate["time"]) or candidate["time"] < 0:
            return None
        result["time"] = float(candidate["time"])
    return result


def _region(small, scale):
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= PARAMETERS["lab_luminance_weight"]
    seed_rows = max(1, round(small.shape[0] * PARAMETERS["bottom_seed_fraction"]))
    seed = lab[-seed_rows:].reshape(-1, 3)
    palette = np.median(seed, axis=0)
    radius = float(np.clip(np.quantile(np.linalg.norm(seed - palette, axis=1), PARAMETERS["palette_radius_seed_quantile"])
                           * PARAMETERS["palette_radius_multiplier"], PARAMETERS["palette_radius_min"], PARAMETERS["palette_radius_max"]))
    mask = (np.linalg.norm(lab - palette, axis=2) <= radius).astype(np.uint8)
    size = PARAMETERS["morphology_close_open_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    _count, labels, stats, _centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    choices = [int(i) for i in np.unique(labels[-seed_rows:]) if i]
    chosen = max(choices, key=lambda i: stats[i, cv2.CC_STAT_AREA]) if choices else 0
    region = (labels == chosen).astype(np.uint8) if chosen else np.zeros_like(mask)
    details = {"region_area_fraction": float(region.mean()), "bottom_coverage": float(region[-seed_rows:].mean()),
               "palette_radius": radius, "palette_lab_weighted": palette.tolist()}
    valid = _valid_region(details)
    # Avoid meaningless very large distances when no credible region exists.
    distance = cv2.distanceTransform(1 - region, cv2.DIST_L2, 5) * scale if valid else None
    return distance, details


def _valid_region(details):
    if not isinstance(details, dict):
        return False
    area, coverage = details.get("region_area_fraction"), details.get("bottom_coverage")
    return (_finite(area) and _finite(coverage) and PARAMETERS["accepted_region_area_range"][0] <= area <= PARAMETERS["accepted_region_area_range"][1]
            and PARAMETERS["minimum_bottom_coverage"] <= coverage <= 1.)


def _scene_gate(scene):
    if not isinstance(scene, dict):
        return False
    shape, segmentation = scene.get("frame_shape"), scene.get("segmentation_size")
    count, indices, hashes = scene.get("frame_count"), scene.get("sample_indices"), scene.get("sample_frame_sha256")
    n = PARAMETERS["sample_count"]
    if (not isinstance(shape, list) or len(shape) != 3 or not all(_integer(v) and v > 0 for v in shape) or shape[2] != 3
            or not _integer(count) or count < n or not isinstance(segmentation, list) or len(segmentation) != 2
            or not all(_integer(v) and v > 0 for v in segmentation)):
        return False
    expected_size = [PARAMETERS["segmentation_width"], round(shape[0] * PARAMETERS["segmentation_width"] / shape[1])]
    if segmentation != expected_size:
        return False
    expected_indices = np.linspace(0, count - 1, n).round().astype(int).tolist()
    if indices != expected_indices or not isinstance(hashes, list) or len(hashes) != n or not all(_sha(h) for h in hashes):
        return False
    regions, registration = scene.get("source_view_regions"), scene.get("registration")
    if (not _valid_region(scene.get("median_region")) or not isinstance(regions, list) or len(regions) != n
            or not all(_valid_region(r) for r in regions) or not isinstance(registration, list) or len(registration) != n):
        return False
    limit = math.hypot(*segmentation) * PARAMETERS["maximum_registration_shift_diagonal_fraction"]
    for item in registration:
        if not isinstance(item, dict):
            return False
        shift, response = item.get("translation_px_small"), item.get("phase_response")
        if (not isinstance(shift, list) or len(shift) != 2 or not all(_finite(v) for v in shift)
                or not _finite(response) or response < PARAMETERS["minimum_phase_response"] or math.hypot(*shift) > limit):
            return False
    return True


def _build_scene(frames, frame_hash):
    count = len(frames)
    first = np.asarray(frames[0])
    if count < PARAMETERS["sample_count"] or first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("Need seven actual uint8 BGR source frames")
    height, width = first.shape[:2]
    sw = PARAMETERS["segmentation_width"]
    sh = round(height * sw / width)
    if min(height, width, sh) < 2:
        raise ValueError("Invalid analysis image dimensions")
    indices = np.linspace(0, count - 1, PARAMETERS["sample_count"]).round().astype(int).tolist()
    small = []
    for index in indices:
        frame = np.asarray(frames[index])
        if frame.shape != first.shape or frame.dtype != np.uint8:
            raise ValueError("Source frame geometry changed")
        small.append(cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA))
    center = cv2.cvtColor(small[len(small) // 2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    registration = []
    for frame in small:
        shift, response = cv2.phaseCorrelate(center, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
        registration.append({"translation_px_small": [float(v) for v in shift], "phase_response": float(response)})
    background = np.median(np.stack(small), axis=0).astype(np.uint8)
    _median_map, median_region = _region(background, width / sw)
    views = [_region(frame, width / sw) for frame in small]
    scene = {"frame_shape": list(first.shape), "frame_count": count, "segmentation_size": [sw, sh],
             "sample_indices": indices, "sample_frame_sha256": [frame_hash(i) for i in indices],
             "median_region": median_region, "source_view_regions": [view[1] for view in views],
             "registration": registration}
    scene["signature_sha256"] = _digest(scene)
    return scene, [view[0] for view in views]


def _base_evidence(reason):
    return {"schema_version": SCHEMA_VERSION, "method": METHOD, "parameters_sha256": PARAMETERS_SHA256,
            "status": "unknown", "evidence_valid": False, "reason_codes": [reason], "rank_factor": 1.,
            "floor_verified": False, "contact_verified": False, "identity_verified": False,
            "coordinates_modified": False, "flight_eligibility_modified": False}


def _factor(distance, height):
    margin = height * PARAMETERS["distance_margin_height_fraction"]
    return 1. - PARAMETERS["maximum_relative_penalty"] * min(1., max(0., (distance - margin) / margin))


def annotate_scene_context(candidates, frames):
    """Return copied candidates with shared, actual-source scene diagnostics.

    The scene is computed once. Frame hashes are memoized by source index,
    including samples already hashed while building the scene. No points or
    eligibility/identity/contact fields are edited. Invalid context is neutral.
    """
    output = [dict(candidate) for candidate in candidates]
    if not output:
        return output
    hashes = {}

    def frame_hash(index):
        if index not in hashes:
            hashes[index] = _frame_digest(frames[index])
        return hashes[index]

    try:
        if not len(frames):
            raise ValueError("No actual source frames")
        scene, maps = _build_scene(frames, frame_hash)
        usable = _scene_gate(scene)
    except (ValueError, TypeError, IndexError, cv2.error, FloatingPointError):
        for candidate in output:
            candidate["scene_context"] = _base_evidence("SOURCE_SCENE_UNAVAILABLE")
        return output
    for candidate in output:
        evidence = _base_evidence("SCENE_GATE_UNKNOWN")
        evidence["source_scene"] = scene
        candidate["scene_context"] = evidence
        if not usable:
            continue
        geometry = _geometry(candidate, scene["frame_count"], scene["frame_shape"])
        if geometry is None:
            evidence["reason_codes"] = ["CANDIDATE_GEOMETRY_INVALID"]
            continue
        index = geometry["frame_index"]
        frame = np.asarray(frames[index])
        if list(frame.shape) != scene["frame_shape"] or frame.dtype != np.uint8:
            evidence["reason_codes"] = ["CANDIDATE_SOURCE_FRAME_INVALID"]
            continue
        height, width = scene["frame_shape"][:2]
        sw, sh = scene["segmentation_size"]
        x = min(sw - 1, round(geometry["point_px"][0] * sw / width))
        y = min(sh - 1, round(geometry["point_px"][1] * sh / height))
        distances = [float(view[y, x]) for view in maps]
        if not all(math.isfinite(value) and value >= 0 for value in distances):
            evidence["reason_codes"] = ["SCENE_DISTANCE_INVALID"]
            continue
        distance = float(np.median(distances))
        binding = {"geometry": geometry, "frame_sha256": frame_hash(index), "scene_signature_sha256": scene["signature_sha256"]}
        factor = 1. if _explicit_evidence(candidate) else _factor(distance, height)
        evidence.update(status="ambiguous_image_context", evidence_valid=True,
            reason_codes=["BOTTOM_CONNECTED_APPEARANCE_ONLY", "NOT_FLOOR_OR_CONTACT_PROOF"],
            source_frame_binding=binding, source_frame_binding_sha256=_digest(binding),
            distance_px_each_source_view=distances, median_distance_px=distance,
            margin_px=height * PARAMETERS["distance_margin_height_fraction"], rank_factor=factor)
    return output


def scene_rank_factor(candidate):
    """Read a bounded factor, rejecting stale geometry or invalid diagnostics.

    This verifies the attached source binding and current candidate fields; it
    cannot re-read pixels without frames. Re-annotate after changing the source.
    The enclosing pipeline's source/implementation cache contract still applies.
    """
    if _explicit_evidence(candidate):
        return 1.
    evidence = candidate.get("scene_context")
    try:
        if (not isinstance(evidence, dict) or evidence.get("schema_version") != SCHEMA_VERSION
                or isinstance(evidence.get("schema_version"), bool) or evidence.get("method") != METHOD
                or evidence.get("parameters_sha256") != PARAMETERS_SHA256
                or evidence.get("status") != "ambiguous_image_context" or evidence.get("evidence_valid") is not True
                or any(evidence.get(key) is not False for key in ["floor_verified", "contact_verified", "identity_verified", "coordinates_modified", "flight_eligibility_modified"])):
            return 1.
        scene = evidence.get("source_scene")
        if not _scene_gate(scene) or not _sha(scene.get("signature_sha256")):
            return 1.
        if _digest({key: value for key, value in scene.items() if key != "signature_sha256"}) != scene["signature_sha256"]:
            return 1.
        binding = evidence.get("source_frame_binding")
        if (not isinstance(binding, dict) or not _sha(binding.get("frame_sha256"))
                or binding.get("scene_signature_sha256") != scene["signature_sha256"]
                or evidence.get("source_frame_binding_sha256") != _digest(binding)):
            return 1.
        geometry = _geometry(candidate, scene["frame_count"], scene["frame_shape"])
        if geometry is None or geometry != binding.get("geometry"):
            return 1.
        if geometry["frame_index"] in scene["sample_indices"]:
            sample = scene["sample_indices"].index(geometry["frame_index"])
            if binding["frame_sha256"] != scene["sample_frame_sha256"][sample]:
                return 1.
        distances = evidence.get("distance_px_each_source_view")
        if not isinstance(distances, list) or len(distances) != PARAMETERS["sample_count"] or not all(_finite(v) and v >= 0 for v in distances):
            return 1.
        distance, margin, factor = evidence.get("median_distance_px"), evidence.get("margin_px"), evidence.get("rank_factor")
        if not all(_finite(v) for v in [distance, margin, factor]) or not .5 <= factor <= 1.:
            return 1.
        if abs(distance - float(np.median(distances))) > 1e-9 or abs(margin - scene["frame_shape"][0] * PARAMETERS["distance_margin_height_fraction"]) > 1e-9:
            return 1.
        expected = _factor(distance, scene["frame_shape"][0])
        return expected if abs(expected - factor) <= 1e-12 else 1.
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, cv2.error):
        return 1.
