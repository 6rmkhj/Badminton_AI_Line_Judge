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


def _primary_polygon(value, width, height):
    """Validate an image-space support region without silently fixing its order."""
    polygon = np.asarray(value, dtype=np.float64)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or not 3 <= len(polygon) <= 16
            or not np.isfinite(polygon).all()
            or (polygon < 0).any() or (polygon >= [width, height]).any()
            or not cv2.isContourConvex(polygon.astype(np.float32))
            or abs(cv2.contourArea(polygon.astype(np.float32))) < 4):
        raise ValueError("invalid primary-court polygon")
    return polygon


def _polygon_margins(polygon, points):
    """Positive signed distances to each edge of a convex polygon."""
    center = polygon.mean(axis=0)
    margins = []
    for a, b in _edges(polygon):
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length <= _EPS:
            raise ValueError("degenerate polygon edge")
        normal = np.array([-delta[1], delta[0]]) / length
        if float((center - a) @ normal) < 0:
            normal = -normal
        margins.append((points - a) @ normal)
    return np.asarray(margins)


def _erode_primary_polygon(polygon, distance):
    """Intersect shifted half planes, retaining no region if erosion empties it."""
    region = polygon.copy()
    center = polygon.mean(axis=0)
    for a, b in _edges(polygon):
        delta = b - a
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        if float((center - a) @ normal) < 0:
            normal = -normal
        clipped = []
        for p, q in _edges(region):
            mp, mq = float((p-a) @ normal) - distance, float((q-a) @ normal) - distance
            if mp >= 0:
                clipped.append(p)
            if (mp >= 0) != (mq >= 0):
                clipped.append(p + (q-p) * mp / (mp-mq))
        if len(clipped) < 3:
            return None
        region = np.asarray(clipped)
    if abs(cv2.contourArea(region.astype(np.float32))) <= _EPS:
        return None
    return region


def _footprint_intersects_polygon(footprint, polygon):
    if (_polygon_margins(polygon, footprint) >= -_EPS).all(axis=0).any():
        return True
    if len(footprint) >= 3 and any(cv2.pointPolygonTest(
            footprint.astype(np.float32), tuple(point), False) >= 0 for point in polygon):
        return True
    return any(_segments_intersect(a, b, c, d)
               for a, b in _edges(footprint) for c, d in _edges(polygon))


def _observed_primary_region(boundaries, width, height):
    """Bound IN support by distinct measured sides and their observed corners."""
    lines = {row['role']: np.asarray(row['segment_px']).reshape(2, 2) for row in boundaries}
    errors = {row['role']: row['boundary_uncertainty_px'] for row in boundaries}
    values = list(lines.values())
    for index, line in enumerate(values):
        delta = line[1] - line[0]
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        for other in values[index+1:]:
            if np.abs((other-line[0]) @ normal).max() <= 1e-4:
                raise ValueError('different outer roles name the same boundary line')
    if not {'left', 'right', 'near'} <= lines.keys():
        return None

    def corner(first, second):
        a, b = lines[first]
        c, d = lines[second]
        u, v = b-a, d-c
        determinant = float(u[0]*v[1] - u[1]*v[0])
        if abs(determinant) / (np.linalg.norm(u)*np.linalg.norm(v)) < 1e-5:
            raise ValueError('adjacent court boundaries do not form an observed corner')
        difference = c-a
        point = a + u * float(difference[0]*v[1] - difference[1]*v[0]) / determinant
        tolerance = errors[first] + errors[second] + 1.0
        if (np.linalg.norm(lines[first]-point, axis=1).min() > tolerance
                or np.linalg.norm(lines[second]-point, axis=1).min() > tolerance):
            raise ValueError('court corner lies beyond observed boundary endpoints')
        return point

    near_left, near_right = corner('near', 'left'), corner('near', 'right')
    if 'far' in lines:
        far_left, far_right = corner('far', 'left'), corner('far', 'right')
    else:
        far_left = lines['left'][np.linalg.norm(lines['left']-near_left, axis=1).argmax()]
        far_right = lines['right'][np.linalg.norm(lines['right']-near_right, axis=1).argmax()]
    return _primary_polygon([near_left, near_right, far_right, far_left], width, height)


def _decide_primary_court(landing, calibration, rules):
    """Use observed primary-court boundaries; never fabricate court metres.

    The scoped support region limits which ground contacts this camera can
    judge. Its crop/net edges can restrict IN support but never produce OUT.
    Only observed, role-assigned outer boundary lines may produce OUT.
    """
    region = rule_region(rules)
    primary = calibration['primary_court']
    # Historical partial-court records describe doubles. Never reuse those
    # boundaries for singles merely because the caller changed the rule menu.
    game = primary.get('game', 'doubles')
    selected_rules = primary.get('rules', {'game': 'doubles', 'phase': 'rally'})
    if (region['phase'] != 'rally' or game not in ('singles', 'doubles')
            or region['game'] != game
            or selected_rules != {'game': game, 'phase': 'rally'}):
        return _review(['PRIMARY_COURT_RULES_UNSUPPORTED'], rules)
    stability = calibration.get('camera_stability')
    if (not isinstance(stability, dict) or stability.get('status') != 'ok'
            or stability.get('all_frames_sampled') is not True
            or not isinstance(stability.get('reason_codes', []), list)
            or 'CAMERA_MOVED_FROM_CALIBRATION_REFERENCE' in stability.get('reason_codes', [])):
        return _review(['PRIMARY_COURT_CAMERA_UNSTABLE'], rules)
    try:
        primary = calibration['primary_court']
        width, height = (float(calibration[key]) for key in ('image_width', 'image_height'))
        if not all(math.isfinite(v) and v > 1 for v in (width, height)):
            raise ValueError('invalid image size')
        if primary.get('scope') not in {'near_half', 'full', 'local'}:
            raise ValueError('unknown primary-court scope')
        scope = _primary_polygon(primary.get('judgment_scope_polygon_px'), width, height)
        center = _points([landing.get('contact_px')])[0]
        radius = float(landing.get('contact_radius_px', 0))
        error = float(landing.get('uncertainty_px', 2.5))
        if not all(math.isfinite(v) and 0 <= v <= math.hypot(width, height) for v in (radius, error)):
            raise ValueError('invalid contact uncertainty')
        if landing.get('contact_polygon_px') is not None:
            footprint = _points(landing['contact_polygon_px'])
            disk_radius = None
        elif radius > 0:
            angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
            footprint = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
            disk_radius = radius
        else:
            footprint = center[None]
            disk_radius = None
        if ((center < 0).any() or (center >= [width, height]).any()
                or (footprint < 0).any() or (footprint >= [width, height]).any()):
            raise ValueError('contact outside image')
        polygons = {}
        for name in ('polygon_px', 'interior_polygon_px'):
            if primary.get(name) is not None:
                polygons[name] = _primary_polygon(primary[name], width, height)
                if not (_polygon_margins(scope, polygons[name]) >= -_EPS).all():
                    raise ValueError('court region extends beyond supported judgment scope')
        boundaries = primary.get('boundary_segments_px')
        if not isinstance(boundaries, list) or not 1 <= len(boundaries) <= 4:
            raise ValueError('missing observed boundaries')
        checked, roles = [], set()
        for boundary in boundaries:
            if (not isinstance(boundary, dict) or boundary.get('observed') is not True
                    or boundary.get('role') not in {'left', 'right', 'near', 'far'}
                    or boundary['role'] in roles):
                raise ValueError('invalid observed boundary role')
            roles.add(boundary['role'])
            segment = np.asarray(boundary.get('segment_px'), dtype=float)
            if segment.shape != (4,) or not np.isfinite(segment).all():
                raise ValueError('invalid boundary segment')
            endpoints = segment.reshape(2, 2)
            inside = _points([boundary.get('inside_point_px')])[0]
            line_error = float(boundary['uncertainty_px'])
            if (not math.isfinite(line_error) or not 0 <= line_error <= math.hypot(width, height)
                    or (endpoints < 0).any() or (endpoints >= [width, height]).any()
                    or not (_polygon_margins(scope, inside[None]) >= -_EPS).all()):
                raise ValueError('invalid boundary uncertainty or extent')
            a, b = endpoints
            delta = b-a
            length = float(np.linalg.norm(delta))
            if length <= _EPS:
                raise ValueError('degenerate boundary')
            normal = np.array([-delta[1], delta[0]]) / length
            side = float((inside-a) @ normal)
            if abs(side) <= _EPS:
                raise ValueError('inside reference lies on boundary')
            normal *= 1 if side > 0 else -1
            for polygon in polygons.values():
                if ((polygon-a) @ normal < -line_error-_EPS).any():
                    raise ValueError('court support contradicts outer boundary')
            margins = (footprint-a) @ normal
            stations = (footprint-a) @ (delta / length)
            if disk_radius is not None:
                # The sampled footprint is inscribed (safe for proving IN).
                # Use the exact disk extrema for OUT so an unsampled arc can
                # never turn a touching footprint into an outside verdict.
                center_margin = float((center-a) @ normal)
                margins = np.array([center_margin-disk_radius, center_margin+disk_radius])
                center_station = float((center-a) @ (delta / length))
                stations = np.array([center_station-disk_radius, center_station+disk_radius])
            observed_extent = bool(stations.max() >= -_EPS and stations.min() <= length + _EPS)
            checked.append({'role': boundary['role'], 'segment_px': segment.tolist(),
                            'min_inside_margin_px': float(margins.min()),
                            'max_inside_margin_px': float(margins.max()),
                            'boundary_uncertainty_px': line_error,
                            'uncertainty_px': error + line_error,
                            'within_observed_extent': observed_extent,
                            'robust_outside': bool(observed_extent and margins.max() < -(error + line_error) - _EPS)})
        line_uncertainty = max(row['boundary_uncertainty_px'] for row in checked)
        observed_region = _observed_primary_region(checked, width, height)
        if observed_region is not None:
            for polygon in polygons.values():
                if not (_polygon_margins(observed_region, polygon) >= -line_uncertainty-_EPS).all():
                    raise ValueError('court interior extends past the finite observed side boundaries')
        uncertainty = error + line_uncertainty
        # Keep distant adjacent-court and unseen far-court contacts out of this
        # local verdict even if they lie outside an infinite boundary line.
        scope_margins = (_polygon_margins(scope, center[None]) - disk_radius if disk_radius is not None
                         else _polygon_margins(scope, footprint))
        if not (scope_margins >= uncertainty).all():
            return _review(['CONTACT_OUTSIDE_PRIMARY_COURT_SCOPE'], rules)
        safe_polygon = None
        required = {'left', 'right', 'near', 'far'} if primary['scope'] == 'full' else {'left', 'right', 'near'}
        if primary.get('region_supported') is True and required <= roles and observed_region is not None:
            safe_polygon = polygons.get('interior_polygon_px')
            if safe_polygon is None and primary['scope'] in {'full', 'near_half'}:
                safe_polygon = polygons.get('polygon_px')
        eroded = None if safe_polygon is None else _erode_primary_polygon(safe_polygon, uncertainty)
        if any(row['robust_outside'] for row in checked):
            decision, reasons = 'OUT', ['ENTIRE_POSSIBLE_CONTACT_OUTSIDE']
        elif eroded is not None and _footprint_intersects_polygon(footprint, eroded):
            decision, reasons = 'IN', ['CONTACT_INSIDE_OR_TOUCHING_VALID_BOUNDARY']
        else:
            decision, reasons = 'REVIEW', ['CONTACT_UNCERTAINTY_OVERLAPS_BOUNDARY']
        return {'decision': decision, 'reason_codes': reasons, 'geometry_only': False,
                'coordinate_system': 'analysis_pixels', 'decision_method': 'primary_court_observed_boundaries',
                'court_position_m': None, 'contact_polygon_m': None, 'uncertainty_m': None,
                'contact_px': center.tolist(), 'contact_polygon_px': footprint.tolist(),
                'uncertainty_px': uncertainty, 'boundaries': checked, 'rules': region,
                'court_scope': primary['scope'], 'judgment_scope_polygon_px': scope.tolist(),
                'contact_frame': landing.get('frame_index', landing.get('frame')),
                'contact_time_seconds': landing.get('time_seconds', landing.get('time', landing.get('timestamp'))),
                'uncertainty_method': 'image contact uncertainty plus observed boundary residual and tape extent; no metric projection'}
    except (ValueError, TypeError, cv2.error, IndexError, KeyError, OverflowError):
        return _review(['INVALID_CONTACT_GEOMETRY_OR_CALIBRATION'], rules)


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
    primary = calibration.get('primary_court')
    primary_selected = (calibration.get('status') != 'ok' and isinstance(primary, dict)
                        and primary.get('status') == 'selected')
    if calibration.get("status") != "ok" and not primary_selected:
        reasons.append("CALIBRATION_NOT_ACCEPTED")
    if calibration.get("method") == "manual_reference" or calibration.get("manual_reference"):
        reasons.append("MANUAL_CALIBRATION_REFERENCE_ONLY")
    try:
        identity = float(landing.get("identity_confidence", 0))
        contact_confidence = float(landing.get("confidence", 0))
        calibration_confidence = float((primary if primary_selected else calibration).get("confidence", 0))
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
    if primary_selected:
        return _decide_primary_court(landing, calibration, rules)
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
