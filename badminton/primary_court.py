"""Select the foreground court using its two visible near corners.

A full-court homography is not needed for local image-plane boundary judging.
The narrow doubles/singles alley and baseline/service spacing identify a court
projectively; independent centre and opposite singles lines must corroborate it.
The resulting model is used to assign observed tape roles, never as a calibrated
metric transform or as evidence for an unseen far baseline.
"""
from __future__ import annotations

import itertools
import math

import cv2
import numpy as np


# The two narrow, regulation spacings disambiguate an outer court from a
# service box or an adjacent court. Both camera sides are considered.
_WIDTH = 6.1
_ALLEY = .46
_LONG_SERVICE = .76
_SHORT_SERVICE = 4.72
_NEAR_HALF = 6.7
_MODEL = np.asarray([
    [[0, 0], [0, _NEAR_HALF]], [[_WIDTH, 0], [_WIDTH, _NEAR_HALF]],
    [[0, 0], [_WIDTH, 0]], [[0, _LONG_SERVICE], [_WIDTH, _LONG_SERVICE]],
    [[_ALLEY, 0], [_ALLEY, _NEAR_HALF]],
    [[_WIDTH-_ALLEY, 0], [_WIDTH-_ALLEY, _NEAR_HALF]],
    [[_WIDTH/2, 0], [_WIDTH/2, _SHORT_SERVICE]],
    [[0, _SHORT_SERVICE], [_WIDTH, _SHORT_SERVICE]],
], np.float32)
_SAMPLES = np.asarray([np.linspace(a, b, 48) for a, b in _MODEL], np.float32)


def _line(segment):
    points = np.asarray(segment, float).reshape(2, 2)
    vector = points[1] - points[0]
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return None
    axis = vector / length
    normal = np.array([-axis[1], axis[0]])
    return {'segment': points, 'length': length, 'axis': axis,
            'equation': np.r_[normal, -normal @ points[0]],
            'angle': math.atan2(axis[1], axis[0]) % math.pi}


def _intersection(first, second):
    crossing = np.cross(first['equation'], second['equation'])
    if abs(crossing[2]) < .02:
        return None
    return crossing[:2] / crossing[2]


def _angle_distance(first, second):
    difference = abs(first['angle']-second['angle']) % math.pi
    return min(difference, math.pi-difference)


def _project(points, inverse):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(1, -1, 2), inverse)[0]


def _inside_frame(points, width, height, margin=1):
    return (np.isfinite(points).all() and (points[:, 0] >= margin).all()
            and (points[:, 0] <= width-1-margin).all()
            and (points[:, 1] >= margin).all() and (points[:, 1] <= height-1-margin).all())


def _paint_signal(background, colour):
    """Recover thin tape lost by morphological opening, only near a verified grid."""
    b, g, r = cv2.split(background.astype(np.float32))
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    if colour == 'blue':
        cyan = np.where((hsv[:, :, 0] >= 89) & (hsv[:, :, 0] <= 95),
                        np.maximum(0, np.minimum(b, g)-r), 0)
        signal = np.maximum(cyan, np.maximum(0, b-np.maximum(g, r)))
        eligible = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 140) & (hsv[:, :, 1] >= 30)
    elif colour == 'yellow':
        signal = np.maximum(0, np.minimum(r, g)-b)
        eligible = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] >= 35)
    elif colour == 'white':
        signal = np.minimum(np.minimum(b, g), r)
        eligible = hsv[:, :, 1] <= 105
    else:
        return None
    radius = max(5, round(min(background.shape[:2])*.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    ridge = cv2.morphologyEx(signal, cv2.MORPH_TOPHAT, kernel)
    threshold = np.maximum(3.0, signal*.025)
    return np.where(eligible & (ridge >= threshold), ridge, 0)


def _line_support(projected, lines, tolerance):
    starts = np.asarray([line['segment'][0] for line in lines])
    axes = np.asarray([line['axis'] for line in lines])
    lengths = np.asarray([line['length'] for line in lines])
    support = []
    for samples in projected:
        direction = samples[-1]-samples[0]
        direction /= max(np.linalg.norm(direction), 1e-9)
        aligned = np.abs(axes @ direction) >= math.cos(math.radians(5))
        if not aligned.any():
            support.append(0.0)
            continue
        delta = samples[:, None]-starts[aligned]
        along = np.sum(delta*axes[aligned], axis=2)
        closest = starts[aligned] + np.clip(along, 0, lengths[aligned])[:, :, None]*axes[aligned]
        errors = np.linalg.norm(samples[:, None]-closest, axis=2).min(axis=1)
        support.append(float(np.mean(errors <= tolerance)))
    return support


def _refine_tape(ends, signal, unit):
    """Observe a narrow ridge near the predicted line, with no global snapping."""
    row = _line(ends)
    if row is None:
        return None
    normal = np.array([-row['axis'][1], row['axis'][0]])
    radius = max(6, unit*6)
    offsets = np.arange(-radius, radius+.25, max(.5, unit*.5))
    height, width = signal.shape
    points, widths = [], []
    for station in np.linspace(*ends, 120):
        coordinates = np.rint(station + offsets[:, None]*normal).astype(int)
        valid = ((coordinates[:, 0] >= 0) & (coordinates[:, 0] < width)
                 & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < height))
        values = signal[coordinates[:, 1].clip(0, height-1), coordinates[:, 0].clip(0, width-1)]
        hits = np.flatnonzero((values > 0) & valid)
        if not len(hits):
            continue
        middle = int(hits[np.argmin(np.abs(offsets[hits]))])
        left = right = middle
        while left > 0 and values[left-1] > 0 and valid[left-1]:
            left -= 1
        while right < len(offsets)-1 and values[right+1] > 0 and valid[right+1]:
            right += 1
        if left == 0 or right == len(offsets)-1 or offsets[right]-offsets[left] > max(10, unit*10):
            continue
        centre = float(np.average(offsets[left:right+1], weights=values[left:right+1]))
        points.append(station + centre*normal)
        widths.append(offsets[right]-offsets[left]+max(.5, unit*.5))
    if len(points) < 54:
        return None
    points = np.asarray(points, np.float32)
    vx, vy, x, y = cv2.fitLine(points, cv2.DIST_HUBER, 0, .01, .01).reshape(-1)
    axis, origin = np.array([vx, vy]), np.array([x, y])
    normal = np.array([-vy, vx])
    residuals = np.abs((points-origin) @ normal)
    residual = float(np.quantile(residuals, .9))
    if residual > max(2.5, unit*2.5):
        return None
    positions = (np.asarray(ends)-origin) @ axis
    observed = _line(np.array([origin+positions[0]*axis, origin+positions[1]*axis]))
    observed.update(observed_fraction=len(points)/120, residual_px=residual,
                    width_px=float(np.median(widths)),
                    uncertainty_px=max(2.5*unit, residual + float(np.median(widths))/2 + unit))
    return observed


def _scope_polygon(polygon, width, height, padding):
    offsets = np.array([[-padding, -padding], [padding, -padding],
                        [padding, padding], [-padding, padding]])
    expanded = (polygon[:, None]+offsets).reshape(-1, 2)
    expanded[:, 0] = expanded[:, 0].clip(0, width-1)
    expanded[:, 1] = expanded[:, 1].clip(0, height-1)
    return cv2.convexHull(expanded.astype(np.float32)).reshape(-1, 2)


def select_primary_court(frames, calibration, config=None):
    """Return independently supported foreground-court image-plane geometry.

    Inputs are actual analysis frames and the unmodified line detector result.
    ``confidence`` is a structural support score, not a probability. The scope
    polygon includes a small outside strip; its closing edge is never an OUT
    boundary. Global metric calibration remains governed by its own evidence.
    """
    base = {'status': 'not_found', 'scope': 'near_half', 'region_supported': False,
            'confidence': 0.0, 'confidence_kind': 'observed_court_pattern_support',
            'line_segments_px': [], 'boundary_segments_px': [], 'near_corners_px': [],
            'polygon_px': None, 'interior_polygon_px': None, 'judgment_scope_polygon_px': None,
            'method': 'near_baseline_service_alley_projective_pattern',
            'metric_accuracy_validated': False, 'metric_coordinates_available': False,
            'reason_codes': ['PRIMARY_COURT_PATTERN_NOT_FOUND']}
    if not frames or not isinstance(calibration, dict):
        return base
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        return base
    height, width = first.shape[:2]
    if min(height, width) < 80:
        return base
    if (calibration.get('image_width', width) not in (None, width)
            or calibration.get('image_height', height) not in (None, height)):
        return {**base, 'reason_codes': ['PRIMARY_COURT_IMAGE_SPACE_MISMATCH']}
    selected = [np.asarray(frames[i]) for i in np.unique(np.linspace(0, len(frames)-1, min(9, len(frames))).astype(int))]
    if any(frame.shape != first.shape or frame.dtype != np.uint8 for frame in selected):
        return base
    try:
        raw = np.asarray(calibration.get('line_segments_px', []), float).reshape(-1, 4)
    except (ValueError, TypeError):
        return base
    if len(raw) < 6 or not np.isfinite(raw).all():
        return base
    lines = [row for segment in raw[:64] if (row := _line(segment)) is not None]
    lines.sort(key=lambda row: row['length'], reverse=True)
    if len(lines) < 6:
        return base
    unit = width/960
    background = np.median(np.stack(selected), axis=0).astype(np.uint8)
    signal = _paint_signal(background, calibration.get('line_colour'))
    if signal is None:
        return {**base, 'reason_codes': ['PRIMARY_COURT_PAINT_EVIDENCE_UNAVAILABLE']}
    paint_distance = cv2.distanceTransform((signal <= 0).astype(np.uint8)*255, cv2.DIST_L2, 3)
    crosses = [line for line in lines if line['length'] >= .30*width
               and line['segment'][:, 1].min() >= .32*height][:8]
    # Bound the combinatorial proposal search in cluttered halls. All retained
    # lines still participate in independent role-support validation.
    sides = [line for line in lines if line['length'] >= .15*width][:16]
    fits = []
    for near, service in itertools.permutations(crosses, 2):
        if _angle_distance(near, service) > math.radians(22):
            continue
        possible_sides = [line for line in sides if line is not near and line is not service
                          and _angle_distance(line, near) > math.radians(18)]
        for first_side, second_side in itertools.combinations(possible_sides, 2):
            points = [_intersection(a, b) for a, b in ((first_side, near), (second_side, near),
                                                     (second_side, service), (first_side, service))]
            if any(point is None for point in points):
                continue
            strip = np.asarray(points, np.float32)
            if strip[0, 0] > strip[1, 0]:
                strip = strip[[1, 0, 3, 2]]
            if (not cv2.isContourConvex(strip) or not _inside_frame(strip, width, height)
                    or strip[:, 1].min() < height*.32 or strip[:2, 1].max() < height*.70
                    or not width*.02 < np.linalg.norm(strip[1]-strip[0]) < width*.22
                    or not (strip[[3, 2], 1] < strip[:2, 1]-5*unit).all()):
                continue
            for x0, x1 in ((0, _ALLEY), (_WIDTH-_ALLEY, _WIDTH)):
                model_strip = np.array([[x0, 0], [x1, 0], [x1, _LONG_SERVICE], [x0, _LONG_SERVICE]], np.float32)
                inverse = cv2.getPerspectiveTransform(model_strip, strip)
                polygon = _project([[0, 0], [_WIDTH, 0], [_WIDTH, _NEAR_HALF], [0, _NEAR_HALF]], inverse)
                if (not _inside_frame(polygon, width, height) or not cv2.isContourConvex(polygon)
                        or polygon[:, 1].min() < height*.20
                        or polygon[2:, 1].max() > polygon[:2, 1].min()-15*unit
                        or np.linalg.norm(polygon[1]-polygon[0]) < width*.35
                        or abs(cv2.contourArea(polygon)) < width*height*.08):
                    continue
                projected = _project(_SAMPLES, inverse).reshape(len(_MODEL), 48, 2)
                directed = _line_support(projected, lines, max(3, 5*unit))
                coords = np.rint(projected).astype(int)
                errors = paint_distance[coords[:, :, 1].clip(0, height-1), coords[:, :, 0].clip(0, width-1)]
                paint = np.mean(errors <= max(2, 2.5*unit), axis=1)
                # Six independent roles are required. A centre line cannot be
                # substituted for the opposite outer sideline or vice versa.
                if (min(directed[2:4]) < .75 or max(directed[4:6]) < .65
                        or min(directed[4:6]) < .15 or directed[6] < .65
                        or min(paint[:2]) < .45 or min(paint[4:7]) < .60):
                    continue
                score = float(.20*np.mean(directed[2:4]) + .20*np.mean(directed[4:6])
                              + .30*directed[6] + .30*np.mean(paint[:2]))
                if score >= .72:
                    fits.append({'score': score, 'inverse': inverse, 'polygon': polygon,
                                 'directional_support': directed, 'paint_support': paint.tolist()})
    if not fits:
        return base
    fits.sort(key=lambda fit: fit['score'], reverse=True)
    best = fits[0]
    competing = next((fit for fit in fits[1:] if np.max(np.linalg.norm(fit['polygon'][:2]-best['polygon'][:2], axis=1)) > 12*unit), None)
    if competing and best['score']-competing['score'] < .08:
        return {**base, 'status': 'review', 'confidence': round(best['score'], 6),
                'reason_codes': ['PRIMARY_COURT_SELECTION_AMBIGUOUS']}
    inverse = best['inverse']
    boundary_models = [('left', [[0, 0], [0, _NEAR_HALF]]),
                       ('right', [[_WIDTH, 0], [_WIDTH, _NEAR_HALF]]),
                       ('near', [[0, 0], [_WIDTH, 0]])]
    refined = {}
    for role, endpoints in boundary_models:
        line = _refine_tape(_project(endpoints, inverse), signal, unit)
        if line is None:
            return {**base, 'status': 'review', 'confidence': round(best['score'], 6),
                    'reason_codes': ['PRIMARY_COURT_OUTER_TAPE_SUPPORT_INSUFFICIENT']}
        refined[role] = line
    left_corner = _intersection(refined['left'], refined['near'])
    right_corner = _intersection(refined['right'], refined['near'])
    # Close the safe IN region at the near short-service line, not at an
    # unobserved far baseline. It is an interior limit, never an OUT boundary.
    closing = _line(_project([[0, _SHORT_SERVICE], [_WIDTH, _SHORT_SERVICE]], inverse))
    left_top = _intersection(refined['left'], closing)
    right_top = _intersection(refined['right'], closing)
    if any(point is None for point in (left_corner, right_corner, left_top, right_top)):
        return base
    polygon = np.asarray([left_corner, right_corner, right_top, left_top], np.float32)
    if not _inside_frame(polygon, width, height) or not cv2.isContourConvex(polygon):
        return base
    inside_point = polygon.mean(axis=0).tolist()
    boundaries = []
    for role, ends in [('left', [left_corner, left_top]), ('right', [right_corner, right_top]),
                       ('near', [left_corner, right_corner])]:
        row = refined[role]
        boundaries.append({'role': role, 'segment_px': np.asarray(ends).reshape(-1).tolist(),
                           'observed': True, 'inside_point_px': inside_point,
                           'uncertainty_px': round(row['uncertainty_px'], 4),
                           'observed_fraction': round(row['observed_fraction'], 4),
                           'tape_width_px': round(row['width_px'], 4),
                           'fit_residual_px': round(row['residual_px'], 4)})
    # Only target-court lines are emitted. Internal model lines are clipped to
    # the supported near region and included only when measured tape supports
    # them; unrelated raw segments never appear in this list.
    segments = [boundary['segment_px'] for boundary in boundaries]
    for x in (_ALLEY, _WIDTH-_ALLEY, _WIDTH/2):
        ends = _project([[x, 0], [x, _SHORT_SERVICE]], inverse)
        if (line := _refine_tape(ends, signal, unit)) is not None:
            segments.append(line['segment'].reshape(-1).tolist())
    service = _refine_tape(_project([[0, _LONG_SERVICE], [_WIDTH, _LONG_SERVICE]], inverse), signal, unit)
    if service is not None:
        segments.append(service['segment'].reshape(-1).tolist())
    scope = _scope_polygon(polygon, width, height, max(12, 24*unit))
    return {**base, 'status': 'selected', 'region_supported': True,
            'confidence': round(best['score'], 6), 'line_segments_px': segments,
            'boundary_segments_px': boundaries, 'near_corners_px': polygon[:2].tolist(),
            'polygon_px': polygon.tolist(), 'interior_polygon_px': polygon.tolist(),
            'judgment_scope_polygon_px': scope.tolist(),
            'interior_limit': 'near_short_service_line_not_an_out_boundary',
            'geometry_uncertainty_px': max(boundary['uncertainty_px'] for boundary in boundaries),
            'directional_support': best['directional_support'], 'paint_support': best['paint_support'],
            'candidate_count': len(fits),
            'reason_codes': ['PRIMARY_NEAR_COURT_SELECTED', 'LOCAL_OUTER_BOUNDARIES_OBSERVED',
                             'IN_REGION_LIMITED_TO_OBSERVED_NEAR_COURT']}
