"""Tape evidence, line segments and floor vanishing points for the near-half fit.

Part of the independent near-half court fallback (see near_half_court.py).
Everything here is measured from the median calibration background and the
calibration ``line_segments_px``; nothing uses reference annotations.
"""
from __future__ import annotations

import itertools
import math

import cv2
import numpy as np


# ------------------------------------------------------------------ evidence --

def paint_evidence(background, colour):
    """Tape evidence: the production ridge signal ('strong'), a thin-tape
    colour-excess ridge ('weak', catches far 1-2 px lines), their union
    ('combined') with its distance transform and ridge orientation."""
    from .primary_court import _paint_signal
    strong = _paint_signal(background, colour)
    if strong is None:
        return None
    strong = strong.astype(np.float32)
    weak = weak_tape_signal(background, colour)
    combined = strong if weak is None else np.maximum(strong, weak)
    mask = (combined > 0).astype(np.uint8)
    distance = cv2.distanceTransform((1 - mask) * 255, cv2.DIST_L2, 3)
    smooth = cv2.GaussianBlur(combined, (0, 0), 1.2)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
    gradient = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    orientation = np.mod(gradient + np.pi / 2, np.pi).astype(np.float32)
    return {'strong': strong, 'weak': weak, 'combined': combined, 'mask': mask,
            'strong_mask': (strong > 0).astype(np.uint8), 'distance': distance,
            'orientation': orientation}


def weak_tape_signal(background, colour):
    """Thin, faint tape: small-kernel colour-excess ridge. Confirmation only."""
    b, g, r = cv2.split(background.astype(np.float32))
    if colour == 'blue':
        excess = b - np.maximum(g, r)
    elif colour == 'yellow':
        excess = np.minimum(r, g) - b
    elif colour == 'white':
        excess = np.minimum(np.minimum(b, g), r) - 128
    else:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ridge = cv2.morphologyEx(np.maximum(excess, 0), cv2.MORPH_TOPHAT, kernel)
    return np.where(ridge >= 3.0, ridge, 0).astype(np.float32)


# --------------------------------------------------------------------- lines --

def _homogeneous_line(p, q):
    line = np.cross([float(p[0]), float(p[1]), 1.0], [float(q[0]), float(q[1]), 1.0])
    norm = math.hypot(line[0], line[1])
    return line / norm if norm > 1e-9 else None


def _segment_rows(segments):
    rows = []
    for segment in segments:
        p, q = np.asarray(segment[:2], float), np.asarray(segment[2:], float)
        length = float(np.linalg.norm(q - p))
        line = _homogeneous_line(p, q) if length >= 8 else None
        if line is None:
            continue
        rows.append({'p': p, 'q': q, 'length': length, 'line': line, 'mid': (p + q) / 2,
                     'direction': (q - p) / length})
    return rows


def detect_segments(evidence, calibration):
    """Calibration tape segments plus a Hough pass on the strong paint mask."""
    from .calibration import _merged_lines
    raw = []
    try:
        raw.extend(np.asarray(calibration.get('line_segments_px') or [], float).reshape(-1, 4).tolist())
    except (TypeError, ValueError):
        pass
    raw.extend(row['segment'] for row in _merged_lines(evidence['strong_mask'] * 255))
    kept = []
    for row in sorted(_segment_rows(raw), key=lambda r: -r['length']):
        duplicate = False
        for other in kept:
            if (abs(float(other['direction'] @ row['direction'])) > math.cos(math.radians(1.0))
                    and abs(float(other['line'] @ [*row['mid'], 1.0])) < 2.0
                    and abs(float(row['line'] @ [*other['mid'], 1.0])) < 2.0):
                along = sorted(float((point - other['p']) @ other['direction']) for point in (row['p'], row['q']))
                if along[1] >= -2 and along[0] <= other['length'] + 2:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(row)
    return kept


def _vp_errors(vp, mids, directions):
    """Angle (deg) between each segment and the direction from its midpoint to vp."""
    scale = max(1.0, abs(vp[0]), abs(vp[1]))
    if abs(vp[2]) > 1e-7 * scale:
        target = vp[:2] / vp[2] - mids
    else:
        target = np.broadcast_to(vp[:2], mids.shape)
    norm = np.linalg.norm(target, axis=1)
    cosine = np.abs(np.sum(directions * target, axis=1)) / np.maximum(norm, 1e-9)
    errors = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
    return np.where(norm < 1e-9, 0.0, errors)


def _refine_vp(rows, members):
    matrix = np.zeros((3, 3))
    for row, member in zip(rows, members):
        if member:
            matrix += row['length'] * np.outer(row['line'], row['line'])
    _, vectors = np.linalg.eigh(matrix)
    return vectors[:, 0]


def _is_finite_point(vp):
    return abs(vp[2]) > 1e-7 * max(1.0, abs(vp[0]), abs(vp[1]))


def vanishing_points(rows, tolerance_deg=1.5, count=8):
    """Distinct dominant vanishing points of tape segments (RANSAC + suppression)."""
    if len(rows) < 4:
        return []
    mids = np.asarray([row['mid'] for row in rows])
    directions = np.asarray([row['direction'] for row in rows])
    lengths = np.asarray([row['length'] for row in rows])
    order = np.argsort(-lengths)[:40]
    hypotheses = []
    for i, j in itertools.combinations(order, 2):
        if abs(float(directions[i] @ directions[j])) > math.cos(math.radians(1.0)):
            continue
        vp = np.cross(rows[i]['line'], rows[j]['line'])
        if not np.isfinite(vp).all() or np.linalg.norm(vp) < 1e-12:
            continue
        members = _vp_errors(vp, mids, directions) <= tolerance_deg
        hypotheses.append((float(lengths[members].sum()), vp, members))
    for i in order[:20]:  # parallel image lines meet at infinity
        vp = np.array([directions[i][0], directions[i][1], 0.0])
        members = _vp_errors(vp, mids, directions) <= tolerance_deg
        hypotheses.append((float(lengths[members].sum()), vp, members))
    hypotheses.sort(key=lambda row: -row[0])
    found = []
    for support, vp, members in hypotheses:
        if any((members & other['members']).sum() > 0.5 * min(members.sum(), other['members'].sum())
               for other in found):
            continue
        refined = _refine_vp(rows, members)
        refined_members = _vp_errors(refined, mids, directions) <= tolerance_deg
        if lengths[refined_members].sum() >= support:
            vp, members = refined, refined_members
        found.append({'vp': vp / np.linalg.norm(vp), 'support': float(lengths[members].sum()),
                      'members': members})
        if len(found) >= count:
            break
    return found


def implied_focal(vp1, vp2, width, height):
    """Focal length (px) making the two vp directions perpendicular (square
    pixels, principal point at the image centre); None if none is real."""
    shift = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1.0]])
    a, b = shift @ vp1, shift @ vp2
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    denominator = a[2] * b[2]
    if abs(denominator) < 1e-12:
        return None
    value = -(a[0] * b[0] + a[1] * b[1]) / denominator
    return math.sqrt(value) if value > 0 else None


def floor_vp_pairs(vps, rows, width, height):
    """Plausible perpendicular floor-direction pairs, best supported first."""
    pairs = []
    mids = np.asarray([row['mid'] for row in rows])
    lengths = np.asarray([row['length'] for row in rows])
    for first, second in itertools.combinations(range(len(vps)), 2):
        a, b = vps[first]['vp'], vps[second]['vp']
        focal = implied_focal(a, b, width, height)
        if focal is None:
            if _is_finite_point(a) and _is_finite_point(b):
                continue
        elif not 0.25 * width <= focal <= 6.0 * width:
            continue
        horizon = np.cross(a, b)
        norm = math.hypot(horizon[0], horizon[1])
        if norm < 1e-12:
            continue
        horizon = horizon / norm
        tilt = (math.degrees(math.atan2(-horizon[0], horizon[1])) + 90) % 180 - 90
        if abs(tilt) > 25:
            continue
        if horizon @ [width / 2, height - 1, 1.0] < 0:
            horizon = -horizon
        # Floor tape lies below the horizon. Net bands, benches and wall trim
        # parallel to the court can converge to the same points from above it.
        members = vps[first]['members'] | vps[second]['members']
        below = (mids[members] @ horizon[:2] + horizon[2]) > 3
        if lengths[members][below].sum() < 0.6 * lengths[members].sum():
            continue
        pairs.append({'vps': (a, b), 'focal_px': focal, 'tilt_deg': tilt, 'horizon': horizon,
                      'support': vps[first]['support'] + vps[second]['support']})
    pairs.sort(key=lambda row: -row['support'])
    return pairs


def merge_family(rows, vp, tolerance_deg=3.0, limit=9):
    """Infinite lines through ``vp``; collinear fragments are merged."""
    mids = np.asarray([row['mid'] for row in rows])
    directions = np.asarray([row['direction'] for row in rows])
    members = _vp_errors(vp, mids, directions) <= tolerance_deg
    merged = []
    for row in sorted((r for r, m in zip(rows, members) if m), key=lambda r: -r['length']):
        for group in merged:
            if (abs(float(group['line'] @ [*row['p'], 1.0])) < 2.5
                    and abs(float(group['line'] @ [*row['q'], 1.0])) < 2.5):
                group['points'].extend([row['p'], row['q']])
                group['length'] += row['length']
                break
        else:
            merged.append({'line': row['line'], 'points': [row['p'], row['q']], 'length': row['length']})
    for group in merged:
        points = np.asarray(group['points'], np.float32)
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).reshape(-1)
        group['line'] = _homogeneous_line((x0, y0), (x0 + vx, y0 + vy))
    merged.sort(key=lambda g: -g['length'])
    return merged[:limit]
