"""Independent near-half court fit: fallback when select_primary_court fails.

Runs only when the existing selector does not return status 'selected'. It fits
the standard badminton court model (doubles 6.1 m wide, singles inset 0.46 m,
doubles long service 0.76 m inside the back line, short service 4.72 m from the
back line, centre line, net at 6.7 m) to the painted tape of the court nearest
the camera with a homography, and emits the same ``primary_court`` record.

Why a separate fit: the existing selector assumes an A-camera layout (a near
baseline running roughly horizontally across the bottom of the frame). The B
cameras film the near half diagonally from outside a corner, so the back line
and the near sideline both run obliquely and the selector finds no pattern.

Method (image evidence only; modules near_half_evidence / near_half_model):
1. Paint evidence from the per-pixel median of the calibration frames: the
   production tape ridge signal (``primary_court._paint_signal``) plus a thin
   colour-excess ridge for faint far tape, with ridge orientation.
2. Straight tape lines (calibration ``line_segments_px`` plus a Hough pass on
   the paint mask), grouped by the two floor vanishing points. The pair must be
   physically plausible: perpendicular floor directions imply a real focal
   length, a roughly level horizon, and the tape lies below that horizon.
3. Hypotheses: every pair of lines from each family with every assignment of
   model roles (x in {0, .46, 3.05, 5.64, 6.1}, y in {0, .76, 4.72}); four line
   intersections give a homography. Impossible ones are dropped: court behind
   the camera, mirrored court, far half below the near half, or a homography no
   pinhole camera with square pixels can produce (K^-1 H columns must have
   equal norm; swapped or stretched role readings fail this by ~19%).
4. Score = model-line length supported by aligned paint minus half the
   predicted-but-unpainted length. A back line placed on the long-service line
   loses: its long-service and short-service lines fall on bare floor. The best
   distinct hypotheses are refined by robust point-to-line least squares on
   measured tape centres and re-scored.
5. Verification: inner roles (long service, singles sideline, centre, short
   service) must be measured; the fit must beat the best distinct alternative
   and its own role-shifted readings (back line <-> long service line, doubles
   <-> singles sideline). Close alternatives widen per-boundary uncertainty.

``confidence`` is a structural support score in [0, 1], not a probability.
"""
from __future__ import annotations

import cv2
import numpy as np

from .near_half_evidence import (_homogeneous_line, detect_segments, floor_vp_pairs, merge_family,
                                paint_evidence, vanishing_points)
from .near_half_model import (ALLEY, LINE_INDEX, LINE_NAMES, LONG_SERVICE, NEAR_HALF, NEAR_LINES, NET,
                             SHORT_SERVICE, WIDTH, camera_consistency, clip_polygon, clip_segment,
                             enumerate_hypotheses, model_displacement, model_samples, per_line_support,
                             physical_validity, project, refine_homography, score_batch, tape_observations)

METHOD = 'independent_near_half_court_model_fit'
DEFAULTS = {
    'lines_per_family': 9,          # strongest merged lines per vanishing point
    'vp_pairs': 2,                  # plausible floor vp pairs explored
    'screen_tolerance_px': 4.0,     # initial (4-line) hypotheses
    'final_tolerance_px': 2.5,      # refined hypotheses
    'unpainted_penalty': 0.5,
    'angle_tolerance_deg': 12.0,
    'refine_count': 12,
    'min_relative_margin': 0.05,    # best vs best distinct alternative reading
    'min_shift_margin': 0.08,       # best vs its own role-shifted readings
    'distinct_model_m': 0.30,       # smaller differences are the same court reading
    'ambiguity_window': 0.15,       # alternatives this close feed boundary uncertainty
    'max_metric_aspect': 1.10,      # pinhole check: |r1|/|r2| of K^-1 H
    'min_line_support': 0.40,       # an inner role counts as measured
    'min_outer_support': 0.50,      # an outer boundary counts as observed
    'min_visible_px': 30.0,
}
ROLE_SHIFTS = (('back_line_is_long_service', 0.0, LONG_SERVICE),
               ('back_line_further_out', 0.0, -LONG_SERVICE),
               ('left_sideline_is_singles', ALLEY, 0.0),
               ('left_sideline_further_out', -ALLEY, 0.0))
SUPPORT_WEIGHTS = {'left': 1.5, 'right': 1.0, 'near': 1.5, 'left_singles': 1.0, 'right_singles': 0.5,
                   'centre': 1.0, 'long_service': 1.0, 'short_service': 1.0}


def _base(game, reason):
    return {'status': 'not_found', 'scope': 'near_half', 'region_supported': False,
            'confidence': 0.0, 'confidence_kind': 'near_half_court_model_support',
            'line_segments_px': [], 'boundary_segments_px': [], 'near_corners_px': [],
            'polygon_px': None, 'interior_polygon_px': None, 'judgment_scope_polygon_px': None,
            'method': METHOD, 'fallback': True, 'game': game,
            'rules': {'game': game, 'phase': 'rally'},
            'metric_accuracy_validated': False, 'metric_coordinates_available': False,
            'reason_codes': [reason]}


# ---------------------------------------------------------------------- fit --

def _hypotheses_for(rows, width, height, settings):
    """All physically valid hypotheses across the plausible vanishing-point pairs."""
    pairs = floor_vp_pairs(vanishing_points(rows), rows, width, height)
    batches, meta = [], []
    for pair_index, pair in enumerate(pairs[:settings['vp_pairs']]):
        horizon = pair['horizon']
        floor = [row for row in rows if min(row['p'] @ horizon[:2], row['q'] @ horizon[:2]) + horizon[2] > 3]
        families = [merge_family(floor, vp, limit=settings['lines_per_family']) for vp in pair['vps']]
        for assignment, (fx, fy) in enumerate(((families[0], families[1]), (families[1], families[0]))):
            hs, _ = enumerate_hypotheses(fx, fy, width, height)
            if len(hs):
                ok, hs = physical_validity(hs, width, height, settings['max_metric_aspect'])
                hs = hs[ok]
            if len(hs):
                batches.append(hs)
                meta.extend([(pair_index, assignment)] * len(hs))
    return (np.concatenate(batches) if batches else np.zeros((0, 3, 3))), meta, pairs


def _distinct(hs, scores, limit, threshold_px):
    order = np.argsort(-scores)
    corners = project(hs, NEAR_HALF)[0]
    chosen = []
    for index in order:
        if not np.isfinite(scores[index]):
            continue
        if all(np.nanmax(np.linalg.norm(corners[index] - corners[other], axis=1)) > threshold_px
               for other in chosen):
            chosen.append(index)
        if len(chosen) >= limit:
            break
    return chosen


def _score(h, evidence, samples, settings, tolerance_key):
    return score_batch(h, evidence, *samples, settings[tolerance_key], settings['unpainted_penalty'],
                       settings['angle_tolerance_deg'])


def _refine_seeds(hs, scores, meta, evidence, settings, width, height):
    """Refine the best distinct screened hypotheses and re-score them."""
    fine = model_samples(0.05)
    refined = []
    for index in _distinct(hs, scores, settings['refine_count'], 6.0):
        h, observations = refine_homography(hs[index], evidence['combined'])
        ok, checked = physical_validity(h[None], width, height, settings['max_metric_aspect'])
        if not ok[0]:
            continue
        h = checked[0] / checked[0][2, 2] if abs(checked[0][2, 2]) > 1e-12 else checked[0]
        result = _score(h[None], evidence, fine, settings, 'final_tolerance_px')
        refined.append({'h': h, 'score': float(result['score'][0]), 'screen_score': float(scores[index]),
                        'result': result, 'observations': observations, 'seed': int(index),
                        'vp_pair': meta[index][0], 'family_assignment': meta[index][1]})
    refined.sort(key=lambda row: -row['score'])
    return refined, fine


def _verify(best, refined, evidence, fine, settings):
    """Competitor margin, role-shift scores and close alternatives of the best fit."""
    distinct = [row for row in refined[1:] if model_displacement(best['h'], row['h']) > settings['distinct_model_m']]
    competitor = distinct[0] if distinct else None
    best['support_by_line'] = per_line_support(best['result'], fine[1], 0)
    best['competitor'] = None if competitor is None else {
        'score': competitor['score'], 'support_by_line': per_line_support(competitor['result'], fine[1], 0),
        'corners_px': project(competitor['h'], NEAR_HALF)[0].tolist()}
    best['relative_margin'] = (1.0 if competitor is None or best['score'] <= 0
                               else (best['score'] - competitor['score']) / best['score'])
    # Identical image lines read with shifted model roles (0.76 m / 0.46 m).
    best['shift_scores'] = {label: round(float(_score((best['h'] @ np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]]))[None],
                                                       evidence, fine, settings, 'final_tolerance_px')['score'][0]), 3)
                            for label, dx, dy in ROLE_SHIFTS}
    # Alternatives this close feed each boundary's model-ambiguity term.
    best['close_alternatives'] = [row['h'] for row in distinct
                                  if row['score'] >= (1 - settings['ambiguity_window']) * best['score']]
    best['evidence'] = evidence
    return best


def fit_near_half(background, calibration, config=None):
    """Return the best refined near-half hypothesis and its diagnostics (or None)."""
    settings = {**DEFAULTS, **(config or {})}
    height, width = background.shape[:2]
    evidence = paint_evidence(background, calibration.get('line_colour'))
    if evidence is None:
        return None, {'reason': 'NEAR_HALF_PAINT_EVIDENCE_UNAVAILABLE'}
    rows = [row for row in detect_segments(evidence, calibration) if max(row['p'][1], row['q'][1]) > 0.18 * height]
    hs, meta, pairs = _hypotheses_for(rows, width, height, settings)
    diagnostics = {'segment_count': len(rows), 'vp_pairs': [
        {'vps': [p['vps'][0].tolist(), p['vps'][1].tolist()], 'focal_px': p['focal_px'],
         'tilt_deg': round(p['tilt_deg'], 3), 'support': round(p['support'], 2)} for p in pairs[:4]],
        'hypothesis_count': int(len(hs))}
    if not len(hs):
        return None, {**diagnostics, 'reason': 'NEAR_HALF_NO_PLAUSIBLE_HYPOTHESIS'}
    coarse = model_samples(0.15)
    scores = np.concatenate([_score(hs[s:s + 20000], evidence, coarse, settings, 'screen_tolerance_px')['score']
                             for s in range(0, len(hs), 20000)])
    refined, fine = _refine_seeds(hs, scores, meta, evidence, settings, width, height)
    if not refined:
        return None, {**diagnostics, 'reason': 'NEAR_HALF_REFINEMENT_FAILED'}
    best = _verify(refined[0], refined, evidence, fine, settings)
    diagnostics['refined'] = [{'score': round(r['score'], 2), 'screen_score': round(r['screen_score'], 2),
                               'corners_px': np.round(project(r['h'], NEAR_HALF)[0], 1).tolist()}
                              for r in refined]
    diagnostics['pair'] = pairs[best['vp_pair']] if pairs else None
    return best, diagnostics


def select_near_half_court(frames, calibration, config=None):
    """Fallback primary-court record from an independent near-half model fit.

    Same input contract and output schema as ``select_primary_court``; call it
    only when that selector did not return status 'selected'.
    """
    config = dict(config or {})
    game = config.get('game', 'doubles')
    if game not in ('singles', 'doubles'):
        return _base(game, 'PRIMARY_COURT_RULES_UNSUPPORTED')
    base = _base(game, 'NEAR_HALF_COURT_NOT_FOUND')
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
    selected = [np.asarray(frames[i]) for i in
                np.unique(np.linspace(0, len(frames) - 1, min(9, len(frames))).astype(int))]
    if any(frame.shape != first.shape or frame.dtype != np.uint8 for frame in selected):
        return base
    background = np.median(np.stack(selected), axis=0).astype(np.uint8)
    try:
        best, diagnostics = fit_near_half(background, calibration, config.get('near_half'))
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        return {**base, 'reason_codes': ['NEAR_HALF_FIT_FAILED'], 'error': str(error)[:300]}
    if best is None:
        return {**base, 'reason_codes': [diagnostics.get('reason', 'NEAR_HALF_COURT_NOT_FOUND')],
                'diagnostics': {k: v for k, v in diagnostics.items() if k != 'pair'}}
    return build_record(best, diagnostics, width, height, game, config.get('near_half'))


# ------------------------------------------------------------------- record --

def _line_uncertainty(observations, index, unit):
    rows = observations[observations[:, 0] == index] if len(observations) else observations
    if not len(rows):
        return None, None, None
    residual = float(np.quantile(np.abs(rows[:, 8]), 0.9))
    width = float(np.median(rows[:, 7]))
    return max(2.5 * unit, residual + width / 2 + unit), residual, width


def _visible_fraction(h, index, observations, shape, step=0.05, tolerance=3.0, search=9.0):
    """Fraction of the in-frame part of a model line with a measured tape
    centre within ``tolerance`` px, and that part's image length."""
    height, width = shape
    _, start, end = NEAR_LINES[index]
    start, end = np.asarray(start, float), np.asarray(end, float)
    count = max(2, int(round(np.linalg.norm(end - start) / step)) + 1)
    model = start + np.linspace(0, 1, count)[:, None] * (end - start)
    xy, w = project(h, model)
    inside = (np.isfinite(xy).all(axis=1) & (w > 0) & (xy[:, 0] >= search) & (xy[:, 0] <= width - 1 - search)
              & (xy[:, 1] >= search) & (xy[:, 1] <= height - 1 - search))
    if not inside.any():
        return None, 0.0
    visible_px = float(np.linalg.norm(np.diff(xy[inside], axis=0), axis=1).sum()) if inside.sum() > 1 else 0.0
    if observations is None:
        return None, visible_px
    rows = observations[(observations[:, 0] == index) & (np.abs(observations[:, 8]) <= tolerance)] \
        if len(observations) else observations
    return min(1.0, len(rows) / max(1, int(inside.sum()))), visible_px


def _measure_lines(best, h, game, unit, settings):
    """Per model line: in-frame length, measured tape fractions, residual, observed flag."""
    evidence = best['evidence']
    shape = evidence['strong'].shape
    strong_obs = tape_observations(h, evidence['strong'])
    weak_obs = tape_observations(h, evidence['weak'], max_width=8.0) if evidence['weak'] is not None else None
    lines = {}
    for name in LINE_NAMES:
        index = LINE_INDEX[name]
        strong, visible_px = _visible_fraction(h, index, strong_obs, shape)
        weak, _ = _visible_fraction(h, index, weak_obs, shape)
        uncertainty, residual, tape_width = _line_uncertainty(best['observations'], index, unit)
        outer = name in ('left', 'right', 'near') or (game == 'singles' and name in ('left_singles', 'right_singles'))
        threshold = settings['min_outer_support'] if outer else settings['min_line_support']
        lines[name] = {'visible_px': round(visible_px, 2),
                       'strong_fraction': None if strong is None else round(strong, 4),
                       'weak_fraction': None if weak is None else round(weak, 4),
                       'score_support': best['support_by_line'][name]['support'],
                       'residual_px': None if residual is None else round(residual, 4),
                       'tape_width_px': None if tape_width is None else round(tape_width, 4),
                       'uncertainty_px': None if uncertainty is None else round(uncertainty, 4),
                       'observed': bool(visible_px >= settings['min_visible_px'] * unit
                                        and max(strong or 0.0, weak or 0.0) >= threshold)}
    return lines


def _intersect(first, second):
    point = np.cross(first, second)
    if abs(point[2]) < 1e-12:
        return None
    return point[:2] / point[2]


def _model_line(h, x0=None, y0=None):
    """Image line of model x = x0 or y = y0 (homogeneous, normalised)."""
    if x0 is not None:
        ends = project(h, np.array([[x0, 0.0], [x0, NET]]))[0]
    else:
        ends = project(h, np.array([[0.0, y0], [WIDTH, y0]]))[0]
    return _homogeneous_line(ends[0], ends[1])


def _in_frame(points, width, height):
    points = np.asarray(points, float)
    return bool(np.isfinite(points).all() and (points[:, 0] >= 0).all() and (points[:, 0] < width).all()
                and (points[:, 1] >= 0).all() and (points[:, 1] < height).all())


def _court_polygons(h, game):
    """Corners and polygons in the existing selector's convention: active
    sidelines (singles: inner tape), near back line, closed at the near
    short-service line; the physical polygon always uses the doubles lines."""
    side_x = {'left': 0.0, 'right': WIDTH} if game == 'doubles' else {'left': ALLEY, 'right': WIDTH - ALLEY}
    near_line, closing = _model_line(h, y0=0.0), _model_line(h, y0=SHORT_SERVICE)
    corners, tops = {}, {}
    for role in ('left', 'right'):
        side = _model_line(h, x0=side_x[role])
        corners[role], tops[role] = _intersect(side, near_line), _intersect(side, closing)
    points = [corners['left'], corners['right'], tops['right'], tops['left']]
    physical = [_intersect(_model_line(h, x0=x), line)
                for x, line in ((0.0, near_line), (WIDTH, near_line), (WIDTH, closing), (0.0, closing))]
    if any(point is None for point in points + physical):
        return None
    polygon, physical = np.asarray(points, float), np.asarray(physical, float)
    if (not np.isfinite(polygon).all() or not np.isfinite(physical).all()
            or not cv2.isContourConvex(polygon.astype(np.float32))
            or not cv2.isContourConvex(physical.astype(np.float32))):
        return None
    return {'side_x': side_x, 'corners': corners, 'tops': tops, 'polygon': polygon, 'physical': physical}


def _ambiguity(segment, alternatives, x0=None, y0=None):
    """Largest mean distance (px) between this boundary segment and the same
    role's line in any close alternative reading."""
    worst = 0.0
    a, b = np.asarray(segment[:2], float), np.asarray(segment[2:], float)
    samples = a + np.linspace(0, 1, 25)[:, None] * (b - a)
    for h in alternatives:
        line = _model_line(h, x0=x0, y0=y0)
        if line is not None:
            worst = max(worst, float(np.mean(np.abs(samples @ line[:2] + line[2]))))
    return worst


def _boundary_rows(court, lines, alternatives, inside_point, game, width, height, unit):
    """Observed (or explicitly extrapolated) outer boundaries, clipped to the frame."""
    side_name = {'left': 'left', 'right': 'right'} if game == 'doubles' else \
        {'left': 'left_singles', 'right': 'right_singles'}
    corners, tops = court['corners'], court['tops']
    boundaries, reasons = [], []
    for role, ends in (('left', (corners['left'], tops['left'])), ('right', (corners['right'], tops['right'])),
                       ('near', (corners['left'], corners['right']))):
        clipped = clip_segment(ends[0], ends[1], width, height, margin=1.0)
        if clipped is None or np.linalg.norm(clipped[1] - clipped[0]) < 10 * unit:
            reasons.append(f'NEAR_HALF_{role.upper()}_BOUNDARY_OUT_OF_FRAME')
            continue
        row = lines['near' if role == 'near' else side_name[role]]
        segment = clipped.reshape(-1).tolist()
        ambiguity = _ambiguity(segment, alternatives,
                               **({'y0': 0.0} if role == 'near' else {'x0': court['side_x'][role]}))
        measured = row['uncertainty_px'] if row['uncertainty_px'] is not None else 3.0 * unit
        # Unmeasured tape: model extrapolation, at least 3x a measured line.
        uncertainty = max(measured if row['observed'] else max(6.0 * unit, 3.0 * measured), ambiguity)
        boundaries.append({'role': role, 'segment_px': segment, 'observed': bool(row['observed']),
                           'source': ('measured_tape_on_fitted_court_model' if row['observed']
                                      else 'court_model_extrapolation_tape_not_measured'),
                           'inside_point_px': inside_point.tolist(),
                           'uncertainty_px': round(float(uncertainty), 4),
                           'model_ambiguity_px': round(float(ambiguity), 4),
                           'observed_fraction': row['strong_fraction'],
                           'thin_tape_observed_fraction': row['weak_fraction'],
                           'tape_width_px': row['tape_width_px'], 'fit_residual_px': row['residual_px']})
    return boundaries, reasons


def _blocking_reasons(lines, boundaries, best, shift_margin, settings):
    roles = {row['role'] for row in boundaries}
    observed = {row['role'] for row in boundaries if row['observed']}
    inner = [name for name in ('long_service', 'left_singles', 'right_singles', 'centre', 'short_service')
             if lines[name]['observed']]
    checks = (('NEAR_HALF_BACK_LINE_NOT_OBSERVED', 'near' in observed),
              ('NEAR_HALF_LONG_SERVICE_LINE_NOT_OBSERVED', lines['long_service']['observed']),
              ('NEAR_HALF_SINGLES_SIDELINE_NOT_OBSERVED',
               lines['left_singles']['observed'] or lines['right_singles']['observed']),
              ('NEAR_HALF_SERVICE_LINES_NOT_OBSERVED', lines['centre']['observed'] or lines['short_service']['observed']),
              ('NEAR_HALF_INNER_PATTERN_INSUFFICIENT', len(inner) >= 3),
              ('NEAR_HALF_SELECTION_AMBIGUOUS', best['relative_margin'] >= settings['min_relative_margin']),
              ('NEAR_HALF_ROLE_SHIFT_AMBIGUOUS', shift_margin >= settings['min_shift_margin']),
              ('NEAR_HALF_OUTER_BOUNDARY_OUT_OF_FRAME', {'left', 'right', 'near'} <= roles))
    return [code for code, passed in checks if not passed]


def _structural_confidence(lines, best, shift_margin, settings, unit):
    """Weighted measured-tape support, discounted when a distinct or a
    role-shifted reading comes close (full credit at a 25% score margin)."""
    visible = {n: r for n, r in lines.items() if r['visible_px'] >= settings['min_visible_px'] * unit}
    support = (sum(SUPPORT_WEIGHTS[n] * max(r['strong_fraction'] or 0, r['weak_fraction'] or 0)
                   for n, r in visible.items()) / max(1e-9, sum(SUPPORT_WEIGHTS[n] for n in visible)))
    confidence = (support * min(1.0, max(0.0, best['relative_margin']) / 0.25) ** 0.5
                  * min(1.0, max(0.0, shift_margin) / 0.25) ** 0.5)
    return confidence, support


def _inner_segments(h, lines, width, height):
    segments = []
    for name in ('left_singles', 'centre', 'right_singles', 'long_service', 'short_service'):
        if lines[name]['observed']:
            _, start, end = NEAR_LINES[LINE_INDEX[name]]
            ends = project(h, np.asarray([start, end], float))[0]
            clipped = clip_segment(ends[0], ends[1], width, height, margin=1.0)
            if clipped is not None:
                segments.append(clipped.reshape(-1).tolist())
    return segments


def build_record(best, diagnostics, width, height, game, config=None):
    """Primary-court record with the conventions of select_primary_court:
    boundaries run from the near corners to the near short-service line, the
    IN polygon closes at that line (never an OUT boundary) and the judgment
    scope is the physical near court plus a small strip."""
    from .primary_court import _scope_polygon
    settings = {**DEFAULTS, **(config or {})}
    unit = width / 960
    h = best['h'] / best['h'][2, 2]
    lines = _measure_lines(best, h, game, unit, settings)
    court = _court_polygons(h, game)
    if court is None:
        return {**_base(game, 'NEAR_HALF_POLYGON_DEGENERATE'), 'diagnostics': _diag(diagnostics, best, lines)}
    polygon, physical = court['polygon'], court['physical']
    clipped_physical = clip_polygon(physical, width, height)
    clipped_interior = clip_polygon(polygon, width, height)
    if clipped_physical is None or len(clipped_physical) < 3 or clipped_interior is None or len(clipped_interior) < 3:
        return {**_base(game, 'NEAR_HALF_POLYGON_OUTSIDE_FRAME'), 'diagnostics': _diag(diagnostics, best, lines)}
    in_frame = _in_frame(polygon, width, height) and _in_frame(physical, width, height)
    # Out-of-frame corners: point-in-polygon consumers keep working; the strict
    # decision path rejects such (unobserved) corners by design.
    reasons = [] if in_frame else ['NEAR_HALF_CORNER_OUTSIDE_FRAME']
    scope = _scope_polygon(clipped_physical.astype(np.float32), width, height, max(12, 24 * unit))
    boundaries, boundary_reasons = _boundary_rows(court, lines, best.get('close_alternatives') or [],
                                                  clipped_interior.mean(axis=0), game, width, height, unit)
    reasons += boundary_reasons
    unobserved = sorted(row['role'] for row in boundaries if not row['observed'])
    if unobserved:
        reasons.append('NEAR_HALF_BOUNDARY_EXTRAPOLATED_' + '_'.join(role.upper() for role in unobserved))
    shift_margin = min((best['score'] - value) / max(best['score'], 1e-9) for value in best['shift_scores'].values())
    blocking = _blocking_reasons(lines, boundaries, best, shift_margin, settings)
    confidence, support = _structural_confidence(lines, best, shift_margin, settings, unit)
    status = 'selected' if not blocking else 'review'
    focal, aspect = camera_consistency(h, width, height)
    return {**_base(game, 'NEAR_HALF_COURT_SELECTED'), 'status': status,
            'region_supported': status == 'selected',
            'confidence': round(float(confidence), 6), 'structural_support': round(float(support), 6),
            'line_segments_px': [row['segment_px'] for row in boundaries] + _inner_segments(h, lines, width, height),
            'boundary_segments_px': boundaries, 'near_corners_px': polygon[:2].tolist(),
            'polygon_px': polygon.tolist(), 'interior_polygon_px': polygon.tolist(),
            'physical_court_polygon_px': physical.tolist(), 'physical_near_corners_px': physical[:2].tolist(),
            'judgment_scope_polygon_px': scope.tolist(),
            'interior_limit': 'near_short_service_line_not_an_out_boundary',
            'polygon_in_frame': in_frame,
            'near_half_to_net_polygon_px': project(h, NEAR_HALF)[0].tolist(),
            'net_line_px': project(h, np.array([[0.0, NET], [WIDTH, NET]]))[0].reshape(-1).tolist(),
            'geometry_uncertainty_px': max((row['uncertainty_px'] for row in boundaries), default=None),
            'court_model_homography_m_to_px': h.tolist(),
            'court_model_convention': 'x 0..6.1 m left-to-right doubles sideline as seen from the camera, '
                                      'y 0 = near back boundary line, 6.7 = net',
            'camera_check': {'focal_px': None if not np.isfinite(focal) else round(float(focal), 2),
                             'metric_aspect': None if not np.isfinite(aspect) else round(float(aspect), 4)},
            'observed_boundary_roles': sorted(row['role'] for row in boundaries if row['observed']),
            'candidate_count': diagnostics.get('hypothesis_count'),
            'reason_codes': ((['NEAR_HALF_COURT_SELECTED', 'INDEPENDENT_NEAR_HALF_MODEL_FIT'] + reasons)
                             if status == 'selected' else blocking + reasons),
            'diagnostics': _diag(diagnostics, best, lines, shift_margin)}


def _diag(diagnostics, best, lines, shift_margin=None):
    out = {k: v for k, v in diagnostics.items() if k != 'pair'}
    pair = diagnostics.get('pair')
    if pair:
        out['vanishing_points'] = [np.asarray(v).tolist() for v in pair['vps']]
        out['implied_focal_px'] = pair['focal_px']
    out.update(score=round(best['score'], 3), relative_margin=round(float(best['relative_margin']), 4),
               shift_scores=best['shift_scores'],
               shift_margin=None if shift_margin is None else round(float(shift_margin), 4),
               lines=lines, competitor=best.get('competitor'),
               close_alternative_count=len(best.get('close_alternatives') or []))
    return out
