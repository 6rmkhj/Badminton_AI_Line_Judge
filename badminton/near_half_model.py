"""Court model, homography hypotheses, scoring and refinement for the near-half fit.

Court model in metres: x across the doubles court (0 = left doubles sideline as
seen from the camera), y from the near back boundary line (0) toward the net
(6.7). Homographies map model metres to analysis pixels.
"""
from __future__ import annotations

import itertools
import math

import cv2
import numpy as np
from scipy.optimize import least_squares

WIDTH = 6.1
ALLEY = 0.46
LONG_SERVICE = 0.76
SHORT_SERVICE = 4.72
NET = 6.7
X_ROLES = (0.0, ALLEY, WIDTH / 2, WIDTH - ALLEY, WIDTH)
Y_ROLES = (0.0, LONG_SERVICE, SHORT_SERVICE)

NEAR_LINES = (
    ('left', (0.0, 0.0), (0.0, NET)),
    ('left_singles', (ALLEY, 0.0), (ALLEY, NET)),
    ('centre', (WIDTH / 2, 0.0), (WIDTH / 2, SHORT_SERVICE)),
    ('right_singles', (WIDTH - ALLEY, 0.0), (WIDTH - ALLEY, NET)),
    ('right', (WIDTH, 0.0), (WIDTH, NET)),
    ('near', (0.0, 0.0), (WIDTH, 0.0)),
    ('long_service', (0.0, LONG_SERVICE), (WIDTH, LONG_SERVICE)),
    ('short_service', (0.0, SHORT_SERVICE), (WIDTH, SHORT_SERVICE)),
)
LINE_NAMES = tuple(row[0] for row in NEAR_LINES)
LINE_INDEX = {name: index for index, name in enumerate(LINE_NAMES)}
NEAR_HALF = np.array([[0, 0], [WIDTH, 0], [WIDTH, NET], [0, NET]], np.float64)


# ------------------------------------------------------------ homographies --

def _batched_homography(model, image):
    """DLT with h33 = 1: model, image (N, 4, 2) -> H (N, 3, 3), NaN if singular."""
    count = model.shape[0]
    a = np.zeros((count, 8, 8))
    b = np.zeros((count, 8))
    for k in range(4):
        x, y = model[:, k, 0], model[:, k, 1]
        u, v = image[:, k, 0], image[:, k, 1]
        a[:, 2 * k, 0], a[:, 2 * k, 1], a[:, 2 * k, 2] = x, y, 1
        a[:, 2 * k, 6], a[:, 2 * k, 7] = -u * x, -u * y
        a[:, 2 * k + 1, 3], a[:, 2 * k + 1, 4], a[:, 2 * k + 1, 5] = x, y, 1
        a[:, 2 * k + 1, 6], a[:, 2 * k + 1, 7] = -v * x, -v * y
        b[:, 2 * k], b[:, 2 * k + 1] = u, v
    h = np.full((count, 9), np.nan)
    good = np.abs(np.linalg.det(a)) > 1e-9
    if good.any():
        h[good, :8] = np.linalg.solve(a[good], b[good][..., None])[..., 0]
        h[good, 8] = 1.0
    return h.reshape(count, 3, 3)


def project(h, points):
    """Model metres (M, 2) -> image px for H (3, 3) or (N, 3, 3); returns (xy, w)."""
    points = np.asarray(points, np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    if h.ndim == 2:
        out = homogeneous @ h.T
    else:
        out = np.einsum('nij,mj->nmi', h, homogeneous)
    with np.errstate(divide='ignore', invalid='ignore'):
        return out[..., :2] / out[..., 2:3], out[..., 2]


def _normalise_sign(h):
    """Scale each H so that w > 0 at the near-half centre."""
    centre = np.array([WIDTH / 2, NET / 2, 1.0])
    w = h[..., 2, :] @ centre
    sign = np.where(w < 0, -1.0, 1.0)
    return h * sign[..., None, None]


def physical_validity(h, width, height, max_aspect=None):
    """Court in front of the camera, not mirrored, near half nearer than the net,
    and a substantial part of the near half inside the frame."""
    h = _normalise_sign(h)
    corners = np.array([[0, 0], [WIDTH, 0], [WIDTH, NET], [0, NET], [WIDTH / 2, NET / 2]])
    xy, w = project(h, corners)
    ok = np.isfinite(xy).all(axis=(-1, -2)) & (w > 1e-9).all(axis=-1)
    # Orientation: d(image)/dx x d(image)/dy must be negative (camera above floor).
    dx, _ = project(h, np.array([[WIDTH / 2 + 0.01, NET / 2]]))
    dy, _ = project(h, np.array([[WIDTH / 2, NET / 2 + 0.01]]))
    jx, jy = dx[..., 0, :] - xy[..., 4, :], dy[..., 0, :] - xy[..., 4, :]
    det = jx[..., 0] * jy[..., 1] - jx[..., 1] * jy[..., 0]
    ok &= np.nan_to_num(det, nan=1.0) < 0
    near_mid, _ = project(h, np.array([[WIDTH / 2, 0.0], [WIDTH / 2, NET]]))
    ok &= np.nan_to_num(near_mid[..., 0, 1] - near_mid[..., 1, 1], nan=-1.0) > 0
    grid = np.array([[x, y] for x in np.linspace(0, WIDTH, 7) for y in np.linspace(0, NET, 7)])
    gxy, gw = project(h, grid)
    inside = ((gw > 0) & (gxy[..., 0] >= 0) & (gxy[..., 0] < width)
              & (gxy[..., 1] >= 0) & (gxy[..., 1] < height))
    ok &= inside.mean(axis=-1) >= 0.30
    quad = np.nan_to_num(xy[..., :4, :])
    area = 0.5 * np.abs(np.sum(quad[..., 0] * np.roll(quad[..., 1], -1, axis=-1)
                               - np.roll(quad[..., 0], -1, axis=-1) * quad[..., 1], axis=-1))
    ok &= area >= 0.04 * width * height
    focal, aspect = camera_consistency(h, width, height)
    undetermined = np.isnan(focal)
    with np.errstate(divide='ignore', invalid='ignore'):
        plausible = ((focal >= 0.25 * width) & (focal <= 6.0 * width)
                     & (np.abs(np.log(aspect)) <= math.log(max_aspect or MAX_METRIC_ASPECT)))
    ok &= undetermined | np.nan_to_num(plausible, nan=False).astype(bool)
    return ok, h


# Real cameras have square pixels; principal-point offset and mild lens
# distortion keep correct court readings within a few percent of 1.
MAX_METRIC_ASPECT = 1.10


def camera_consistency(h, width, height):
    """Pinhole check of a court-plane homography (square pixels, principal point
    at the image centre): K^-1 H = [r1 r2 t] needs r1 perpendicular to r2 (gives
    the focal length) and |r1| = |r2| (the metric aspect, 1 for a real camera).
    A court read with swapped or stretched roles needs an aspect far from 1.
    NaN where a vanishing point at infinity leaves the focal undetermined."""
    h = np.asarray(h, float)
    single = h.ndim == 2
    h = h[None] if single else h
    shift = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1.0]])
    g = np.einsum('ij,njk->nik', shift, h)
    g = g / np.linalg.norm(g[:, :, :2], axis=(1, 2))[:, None, None]
    h1, h2 = g[:, :, 0], g[:, :, 1]
    denominator = h1[:, 2] * h2[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        f2 = -(h1[:, 0] * h2[:, 0] + h1[:, 1] * h2[:, 1]) / denominator
    determined = (np.abs(denominator) > 1e-10) & np.isfinite(f2)
    focal = np.where(determined & (f2 > 0), np.sqrt(np.where(f2 > 0, f2, 1.0)), np.nan)
    # Negative f^2: no real camera makes these directions perpendicular.
    focal = np.where(determined & (f2 <= 0), -1.0, focal)
    safe = np.where(np.isfinite(focal) & (focal > 0), focal, 1.0)
    n1 = np.sqrt((h1[:, 0] ** 2 + h1[:, 1] ** 2) / safe ** 2 + h1[:, 2] ** 2)
    n2 = np.sqrt((h2[:, 0] ** 2 + h2[:, 1] ** 2) / safe ** 2 + h2[:, 2] ** 2)
    aspect = np.where(np.isfinite(focal) & (focal > 0), n1 / np.maximum(n2, 1e-12), np.nan)
    aspect = np.where(focal == -1.0, np.inf, aspect)
    if single:
        return float(focal[0]), float(aspect[0])
    return focal, aspect


# ------------------------------------------------------------------ scoring --

def model_samples(step):
    """Points along the near-half model lines, their line index and spacing (m)."""
    rows, names, spacing = [], [], []
    for index, (_, start, end) in enumerate(NEAR_LINES):
        start, end = np.asarray(start), np.asarray(end)
        length = float(np.linalg.norm(end - start))
        count = max(2, int(round(length / step)) + 1)
        for t in np.linspace(0, 1, count):
            rows.append(start + t * (end - start))
            names.append(index)
            spacing.append(length / (count - 1))
    return np.asarray(rows), np.asarray(names), np.asarray(spacing)


def _unit_directions(names):
    out = np.zeros((len(names), 2))
    for index, (_, start, end) in enumerate(NEAR_LINES):
        direction = np.asarray(end, float) - np.asarray(start, float)
        out[names == index] = direction / np.linalg.norm(direction)
    return out


def score_batch(h, evidence, samples, names, spacing, tolerance_px, penalty, angle_deg):
    """Supported minus penalised unsupported projected tape length for H batch.

    Each model sample stands for ``spacing`` metres; its image length is that
    times the local projected scale (capped so the few samples next to the
    camera cannot dominate). Support needs paint within tolerance whose ridge
    orientation matches the projected line.
    """
    height, width = evidence['mask'].shape
    xy, w = project(h, samples)
    xy2, _ = project(h, samples + 0.02 * _unit_directions(names))
    directions = np.nan_to_num(xy2 - xy)
    scale = np.linalg.norm(directions, axis=-1) / 0.02
    finite = np.isfinite(xy).all(axis=-1) & (w > 0)
    xy = np.nan_to_num(xy, nan=-1e6, posinf=-1e6, neginf=-1e6)
    inside = finite & (xy[..., 0] >= 1) & (xy[..., 0] <= width - 2) & (xy[..., 1] >= 1) & (xy[..., 1] <= height - 2)
    xs = np.clip(np.rint(xy[..., 0]).astype(int), 0, width - 1)
    ys = np.clip(np.rint(xy[..., 1]).astype(int), 0, height - 1)
    near = evidence['distance'][ys, xs] <= tolerance_px
    theta = evidence['orientation'][ys, xs]
    phi = np.mod(np.arctan2(directions[..., 1], directions[..., 0]), np.pi)
    diff = np.abs(theta - phi)
    aligned = np.minimum(diff, np.pi - diff) <= math.radians(angle_deg)
    supported = inside & near & aligned
    weight = np.clip(np.nan_to_num(scale), 0, 400) * spacing
    supported_px = (weight * supported).sum(axis=-1)
    visible_px = (weight * inside).sum(axis=-1)
    return {'score': supported_px - penalty * (visible_px - supported_px),
            'supported_px': supported_px, 'visible_px': visible_px,
            'supported': supported, 'inside': inside, 'weight': weight}


def per_line_support(result, names, index=None):
    rows = {}
    for line, name in enumerate(LINE_NAMES):
        chosen = names == line
        weight = result['weight'] if index is None else result['weight'][index]
        inside = result['inside'] if index is None else result['inside'][index]
        supported = result['supported'] if index is None else result['supported'][index]
        visible = float((weight[chosen] * inside[chosen]).sum())
        good = float((weight[chosen] * supported[chosen]).sum())
        rows[name] = {'visible_px': round(visible, 2),
                      'support': round(good / visible, 4) if visible > 0 else None}
    return rows


# ------------------------------------------------------------- enumeration --

def _ordered_role_pairs(roles):
    return [(a, b) for a in roles for b in roles if a != b]


def enumerate_hypotheses(family_x, family_y, width, height):
    """Homographies from two lines of each family and every model-role assignment."""
    lines_x = np.asarray([row['line'] for row in family_x])
    lines_y = np.asarray([row['line'] for row in family_y])
    if len(lines_x) < 2 or len(lines_y) < 2:
        return np.zeros((0, 3, 3)), np.zeros((0, 4, 2))
    crossings = np.cross(lines_x[:, None, :], lines_y[None, :, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        points = crossings[..., :2] / crossings[..., 2:3]
    bad = (np.abs(crossings[..., 2]) < 1e-9) | ~np.isfinite(points).all(axis=-1)
    points[bad] = np.nan
    xpairs = list(itertools.combinations(range(len(lines_x)), 2))
    ypairs = list(itertools.combinations(range(len(lines_y)), 2))
    xroles = _ordered_role_pairs(X_ROLES)
    yroles = _ordered_role_pairs(Y_ROLES)
    combos = np.array([(i, j, a, b, k, l, c, d)
                       for (i, j) in xpairs for (a, b) in xroles
                       for (k, l) in ypairs for (c, d) in yroles], dtype=np.float64)
    i, j, k, l = (combos[:, n].astype(int) for n in (0, 1, 4, 5))
    xa, xb, yc, yd = combos[:, 2], combos[:, 3], combos[:, 6], combos[:, 7]
    image = np.stack([points[i, k], points[j, k], points[j, l], points[i, l]], axis=1)
    model = np.stack([np.stack([xa, yc], 1), np.stack([xb, yc], 1),
                      np.stack([xb, yd], 1), np.stack([xa, yd], 1)], axis=1)
    finite = np.isfinite(image).all(axis=(1, 2))
    box = ((image[..., 0] > -3 * width) & (image[..., 0] < 4 * width)
           & (image[..., 1] > -3 * height) & (image[..., 1] < 4 * height)).all(axis=1)
    # A homography of a planar scene seen from above flips orientation.
    def signed_area(quad):
        return 0.5 * np.sum(quad[..., 0] * np.roll(quad[..., 1], -1, axis=1)
                            - np.roll(quad[..., 0], -1, axis=1) * quad[..., 1], axis=1)
    with np.errstate(invalid='ignore'):
        flipped = np.sign(signed_area(np.nan_to_num(image))) == -np.sign(signed_area(model))
    keep = finite & box & flipped
    return _batched_homography(model[keep], image[keep]), image[keep]


# --------------------------------------------------------------- refinement --

def _sample_along(signal, points):
    """Bilinear samples of ``signal`` at float (N, K, 2) image points."""
    maps = points.astype(np.float32)
    return cv2.remap(signal, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def tape_observations(h, signal, step=0.05, radius=9.0, max_width=16.0):
    """Measured tape centres near each projected model line (point-to-line data)."""
    height, width = signal.shape
    observations = []
    offsets = np.arange(-radius, radius + 0.01, 0.5)
    for index, (_, start, end) in enumerate(NEAR_LINES):
        start, end = np.asarray(start, float), np.asarray(end, float)
        length = float(np.linalg.norm(end - start))
        count = max(2, int(round(length / step)) + 1)
        model = start + np.linspace(0, 1, count)[:, None] * (end - start)
        xy, w = project(h, model)
        xy2, _ = project(h, model + 0.02 * (end - start) / length)
        tangent = xy2 - xy
        norm = np.linalg.norm(tangent, axis=1)
        valid = (np.isfinite(xy).all(axis=1) & (w > 0) & (norm > 1e-6)
                 & (xy[:, 0] >= radius) & (xy[:, 0] <= width - 1 - radius)
                 & (xy[:, 1] >= radius) & (xy[:, 1] <= height - 1 - radius))
        if not valid.any():
            continue
        model, xy, tangent, norm = model[valid], xy[valid], tangent[valid], norm[valid]
        tangent = tangent / norm[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        probes = xy[:, None, :] + offsets[None, :, None] * normal[:, None, :]
        values = _sample_along(signal, probes)
        centre_index = len(offsets) // 2
        for row in range(len(model)):
            positive = values[row] > 0
            if not positive.any():
                continue
            hits = np.flatnonzero(positive)
            middle = hits[np.argmin(np.abs(hits - centre_index))]
            left = right = middle
            while left > 0 and positive[left - 1]:
                left -= 1
            while right < len(offsets) - 1 and positive[right + 1]:
                right += 1
            if left == 0 or right == len(offsets) - 1:
                continue
            run_width = offsets[right] - offsets[left] + 0.5
            if run_width > max_width:
                continue
            weights = values[row, left:right + 1]
            centre = float(np.average(offsets[left:right + 1], weights=weights))
            observations.append((index, model[row, 0], model[row, 1],
                                 xy[row, 0] + centre * normal[row, 0], xy[row, 1] + centre * normal[row, 1],
                                 normal[row, 0], normal[row, 1], run_width, centre))
    return np.asarray(observations, float).reshape(-1, 9)


def refine_homography(h, signal, iterations=4):
    """Robust point-to-line least squares of H on measured tape centres."""
    h = h / h[2, 2]
    for _ in range(iterations):
        data = tape_observations(h, signal)
        if len(data) < 30 or len(set(data[:, 0].astype(int))) < 3:
            return h, data
        model, observed, normal = data[:, 1:3], data[:, 3:5], data[:, 5:7]

        def residuals(params):
            candidate = np.append(params, 1.0).reshape(3, 3)
            xy, w = project(candidate, model)
            out = np.sum((xy - observed) * normal, axis=1)
            return np.where(np.isfinite(out) & (w > 0), out, 50.0)

        solution = least_squares(residuals, h.reshape(-1)[:8], loss='huber', f_scale=1.0,
                                 x_scale='jac', max_nfev=200)
        h = np.append(solution.x, 1.0).reshape(3, 3)
    return h, tape_observations(h, signal)


# ------------------------------------------------------------------ geometry --

def clip_polygon(polygon, width, height, margin=0.5):
    """Sutherland-Hodgman clip of a convex polygon to the image rectangle."""
    xmin, ymin, xmax, ymax = margin, margin, width - 1 - margin, height - 1 - margin
    points = [tuple(p) for p in np.asarray(polygon, float)]
    for axis, bound, keep_less in ((0, xmin, False), (0, xmax, True), (1, ymin, False), (1, ymax, True)):
        output = []
        for index, current in enumerate(points):
            previous = points[index - 1]
            inside_c = current[axis] <= bound if keep_less else current[axis] >= bound
            inside_p = previous[axis] <= bound if keep_less else previous[axis] >= bound
            if inside_c != inside_p:
                t = (bound - previous[axis]) / (current[axis] - previous[axis])
                output.append(tuple(previous[n] + t * (current[n] - previous[n]) for n in range(2)))
            if inside_c:
                output.append(current)
        points = output
        if not points:
            return None
    return np.asarray(points, float)


def clip_segment(a, b, width, height, margin=0.5):
    """Liang-Barsky clip of segment ab to the image rectangle."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], a[0] - margin), (d[0], width - 1 - margin - a[0]),
                 (-d[1], a[1] - margin), (d[1], height - 1 - margin - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return np.array([a + t0 * d, a + t1 * d])


def model_displacement(reference, other):
    """Max displacement (m, in the reference model frame) of the other fit's
    near-half corners: separates a different court reading (role shift,
    rotation, adjacent court) from refinement noise of the same reading."""
    points, w = project(other, NEAR_HALF)
    if not (np.isfinite(points).all() and (w > 0).all()):
        return float('inf')
    inverse = np.linalg.inv(reference)
    back, w = project(inverse, points)
    if not (np.isfinite(back).all() and (w > 0).all()):
        return float('inf')
    return float(np.max(np.linalg.norm(back - NEAR_HALF, axis=1)))
