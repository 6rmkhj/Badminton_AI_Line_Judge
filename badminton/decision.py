"""Conservative badminton court rules and contact-region decisions.

Coordinates are metres: x spans the 6.10 m doubles court, y spans its 13.40 m
length. The near half has y < 6.70. Left/right refer to increasing image-mapped
court x, not a player's viewing direction. Dimensions include boundary lines;
calibration uncertainty must account for estimating a painted line's centre.
Only a caller explicitly supplying observed first-ground-contact evidence may
project an image contact point onto the court plane.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
_EPS = 1e-9


def rule_region(rules: dict | None = None) -> dict:
    """Return an explicit valid rectangle, or raise for unsupported rules.

    Service regions identify the *receiving* box. ``side`` is the box's left or
    right half in court coordinates. ``half`` is near/far. They are required for
    service; inferring them from camera orientation would silently choose a box.
    """
    rules = rules or {}
    game = rules.get("game", rules.get("mode", "doubles"))
    phase = rules.get("phase", "rally")
    if game not in {"singles", "doubles"}:
        raise ValueError("game must be singles or doubles")
    if phase not in {"rally", "service"}:
        raise ValueError("phase must be rally or service")
    xmin, xmax = (0.46, 5.64) if game == "singles" else (0.0, 6.1)
    ymin, ymax = 0.0, 13.4
    if phase == "service":
        half = rules.get("half", rules.get("court_half"))
        side = rules.get("side", rules.get("serving_side"))
        if half not in {"near", "far"} or side not in {"left", "right"}:
            raise ValueError("service requires half=near|far and side=left|right in court coordinates")
        if side == "left":
            xmax = 3.05
        else:
            xmin = 3.05
        if half == "near":
            ymin, ymax = (0.76 if game == "doubles" else 0.0), 4.72
        else:
            ymin, ymax = 8.68, (12.64 if game == "doubles" else 13.4)
    return {"game": game, "phase": phase, "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "polygon_m": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
            "boundary_convention": "boundary included; coordinates represent valid painted boundary extent"}


def _points(value: Any, minimum: int = 1) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum or not np.isfinite(points).all():
        raise ValueError("contact points must be finite Nx2 coordinates")
    if len(points) > 2:
        points = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(float)
    return points


def _inside(point, rect):
    x, y = point
    return rect[0] - _EPS <= x <= rect[1] + _EPS and rect[2] - _EPS <= y <= rect[3] + _EPS


def _segment_distance(p, a, b):
    delta = b - a
    length = float(delta @ delta)
    projection = a if length <= _EPS else a + np.clip(float((p-a) @ delta) / length, 0, 1) * delta
    return float(np.linalg.norm(p - projection))


def _cross(a, b, c):
    u, v = b-a, c-a
    return float(u[0]*v[1] - u[1]*v[0])


def _segments_intersect(a, b, c, d):
    values = (_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b))
    if values[0] * values[1] < -_EPS and values[2] * values[3] < -_EPS:
        return True
    return any(abs(value) <= _EPS and _segment_distance(point, start, end) <= _EPS
               for value, point, start, end in ((values[0], c, a, b), (values[1], d, a, b),
                                               (values[2], a, c, d), (values[3], b, c, d)))


def _edges(points):
    if len(points) == 1:
        return [(points[0], points[0])]
    if len(points) == 2:
        return [(points[0], points[1])]
    return list(zip(points, np.roll(points, -1, axis=0)))


def _intersection_and_distance(points, rect):
    xmin, xmax, ymin, ymax = rect
    if xmin > xmax or ymin > ymax:
        return False, float("inf")
    corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], float)
    if any(_inside(p, rect) for p in points):
        return True, 0.0
    if len(points) >= 3 and any(cv2.pointPolygonTest(points.astype(np.float32), tuple(p), False) >= 0 for p in corners):
        return True, 0.0
    distance = float("inf")
    for a, b in _edges(points):
        for c, d in _edges(corners):
            if _segments_intersect(a, b, c, d):
                return True, 0.0
            distance = min(distance, _segment_distance(a, c, d), _segment_distance(b, c, d),
                           _segment_distance(c, a, b), _segment_distance(d, a, b))
    return False, distance


def evaluate_contact(contact_polygon_m, rules: dict | None = None, uncertainty_m: float = 0.0) -> dict:
    """Classify already-confirmed contact geometry, including uncertainty.

    The input footprint is a point, segment, or convex region. IN requires its
    intersection with the valid court to persist under the position uncertainty
    radius. OUT requires the complete possible footprint to remain outside.
    Near-boundary ambiguity is REVIEW. This helper does not certify observations.
    """
    points = _points(contact_polygon_m)
    uncertainty = float(uncertainty_m)
    if not math.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("uncertainty_m must be finite and nonnegative")
    region = rule_region(rules)
    rect = (region["xmin"], region["xmax"], region["ymin"], region["ymax"])
    overlaps, distance = _intersection_and_distance(points, rect)
    robust_rect = (rect[0] + uncertainty, rect[1] - uncertainty,
                   rect[2] + uncertainty, rect[3] - uncertainty)
    robust_overlap, _ = _intersection_and_distance(points, robust_rect)
    if robust_overlap:
        decision, reasons = "IN", ["CONTACT_INSIDE_OR_TOUCHING_VALID_BOUNDARY"]
    elif not overlaps and distance > uncertainty + _EPS:
        decision, reasons = "OUT", ["ENTIRE_POSSIBLE_CONTACT_OUTSIDE"]
    else:
        decision, reasons = "REVIEW", ["CONTACT_UNCERTAINTY_OVERLAPS_BOUNDARY"]
    centre = points.mean(axis=0)
    return {"decision": decision, "reason_codes": reasons, "court_position_m": centre.tolist(),
            "contact_polygon_m": points.tolist(), "uncertainty_m": uncertainty,
            "distance_to_valid_region_m": distance, "valid_polygon_m": region["polygon_m"],
            "rules": region, "geometry_only": True}


def _review(reasons, rules=None):
    return {"decision": "REVIEW", "reason_codes": list(dict.fromkeys(reasons)),
            "court_position_m": None, "contact_polygon_m": None, "uncertainty_m": None,
            "rules": rules or {}, "geometry_only": False}


def decide(landing: dict, calibration: dict, rules: dict | None = None) -> dict:
    """Gate physical evidence before applying the court-plane geometry.

    ``contact_observed`` must describe observed *first ground contact*, never a
    heuristic stop or airborne shuttle centre. The upstream estimator is expected
    to leave it false when that evidence is unavailable.
    """
    reasons = []
    try:
        rule_region(rules)
    except (ValueError, TypeError):
        return _review(["INVALID_OR_INCOMPLETE_RULE_SELECTION"], rules)
    if not landing.get("contact_observed", False):
        reasons.append("FIRST_CONTACT_NOT_OBSERVED")
    if landing.get("occluded") or landing.get("first_contact_missing"):
        reasons.append("CONTACT_OCCLUDED_OR_MISSING")
    if calibration.get("status") != "ok":
        reasons.append("CALIBRATION_NOT_ACCEPTED")
    if calibration.get("method") == "manual_reference" or calibration.get("manual_reference"):
        reasons.append("MANUAL_CALIBRATION_REFERENCE_ONLY")
    try:
        identity = float(landing.get("identity_confidence", 0))
        contact_confidence = float(landing.get("confidence", 0))
        calibration_confidence = float(calibration.get("confidence", 0))
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (identity, contact_confidence, calibration_confidence)):
            raise ValueError("invalid confidence")
        if identity < 0.8:
            reasons.append("TARGET_SHUTTLE_IDENTITY_UNCERTAIN")
        if contact_confidence < 0.8:
            reasons.append("CONTACT_EVIDENCE_UNCERTAIN")
        if calibration_confidence < 0.65:
            reasons.append("CALIBRATION_SUPPORT_INSUFFICIENT")
    except (ValueError, TypeError):
        reasons.append("INVALID_CONFIDENCE_VALUE")
    if reasons:
        return _review(reasons, rules)
    try:
        transform = np.asarray(calibration.get("homography"), dtype=float)
        if transform.shape != (3, 3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12:
            raise ValueError("invalid homography")
        centre = _points([landing.get("contact_px")])[0]
        radius = float(landing.get("contact_radius_px", 0.0))
        error = float(landing.get("uncertainty_px", 2.5))
        calibration_error = float(calibration.get("line_fit_error_px", 2.5))
        if not all(math.isfinite(x) and x >= 0 for x in (radius, error, calibration_error)):
            raise ValueError("invalid uncertainty")
        if landing.get("contact_polygon_px") is not None:
            footprint = _points(landing["contact_polygon_px"])
        elif radius > 0:
            angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            footprint = centre + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            footprint = centre[None]
        from .calibration import points_for_homography
        plane_footprint = points_for_homography(footprint, calibration)
        homogeneous = np.column_stack((plane_footprint, np.ones(len(plane_footprint)))) @ transform.T
        if (np.abs(homogeneous[:, 2]) < 1e-7).any():
            raise ValueError("projection near horizon")
        projected = homogeneous[:, :2] / homogeneous[:, 2, None]
        # Finite-difference local metric scale, conservative maximum directional
        # displacement, plus a 20 mm line-placement floor for this PoC.
        angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
        rectified = calibration.get('line_fit_error_pixel_space') == 'undistorted_analysis_pixels'
        offsets = (error if rectified else error + calibration_error) * np.column_stack((np.cos(angles), np.sin(angles)))
        probes = np.vstack((centre, centre + offsets))
        probes = points_for_homography(probes, calibration)
        hp = np.column_stack((probes, np.ones(len(probes)))) @ transform.T
        if (np.abs(hp[:, 2]) < 1e-7).any():
            raise ValueError("uncertainty crosses horizon")
        metric = hp[:, :2] / hp[:, 2, None]
        uncertainty = float(np.linalg.norm(metric[1:] - metric[0], axis=1).max())
        if rectified:
            centre_rectified = points_for_homography(centre[None], calibration)[0]
            line_probes = centre_rectified + calibration_error * np.column_stack((np.cos(angles), np.sin(angles)))
            line_hp = np.column_stack((line_probes, np.ones(len(line_probes)))) @ transform.T
            if (np.abs(line_hp[:, 2]) < 1e-7).any():
                raise ValueError('line uncertainty crosses horizon')
            line_metric = line_hp[:, :2] / line_hp[:, 2, None]
            uncertainty += float(np.linalg.norm(line_metric - metric[0], axis=1).max())
        if not math.isfinite(uncertainty):
            raise ValueError('non-finite contact uncertainty')
        uncertainty = max(0.02, uncertainty)
        if not np.isfinite(projected).all() or uncertainty > 1.0:
            return _review(["CONTACT_POSITION_UNCERTAINTY_TOO_LARGE"], rules)
        result = evaluate_contact(projected, rules, uncertainty)
        result["geometry_only"] = False
        result["contact_frame"] = landing.get("frame_index", landing.get("frame"))
        result["contact_time_seconds"] = landing.get("time_seconds", landing.get("time", landing.get("timestamp")))
        result["uncertainty_method"] = "image uncertainty plus line-fit residual, local homography scale, 20 mm minimum; not a validated statistical bound"
        result["reason_codes"] = list(dict.fromkeys(result["reason_codes"]))
        return result
    except (ValueError, TypeError, cv2.error, IndexError, KeyError, OverflowError):
        return _review(["INVALID_CONTACT_GEOMETRY_OR_CALIBRATION"], rules)
