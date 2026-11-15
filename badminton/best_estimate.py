"""Always-decide best-estimate IN/OUT tier for one analysed view.

The strict physical-contact ``decision`` and the native ``video_estimate`` keep
their own evidence contracts. When neither resolves a view, this tier still
classifies the most plausible first-landing location against the selected
game's court boundaries and records *how* it got there: the evidence basis, the
classified point, its signed boundary margin and a heuristic confidence.

The confidence is an ordering aid built from evidence strength and boundary
clearance; it is not a calibrated probability or a measured accuracy. Nothing
here sets ``contact_observed``, verifies target identity, or changes the strict
decision or the video estimate.

Everything is computed from data stored in ``result.json`` so a saved analysis
can be re-evaluated without decoding the video again.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .decision import rule_region


logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
# The tier is recomputable from result.json, so it is versioned on its own
# instead of invalidating the (expensive) analysis fingerprint.
ESTIMATOR_VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
SOURCE = 'best_available_landing_evidence'
DECISION_KIND = 'best_estimate'

# Evidence tiers, strongest first. The prior bounds the confidence a tier can
# reach when the classified point sits well clear of every boundary.
BASIS_PRIOR = {
    'verified_contact': 0.97,
    'video_estimate': 0.9,
    'estimate_envelope': 0.78,
    'flight_family': 0.72,
    'landing_candidate': 0.64,
    'track_endpoint': 0.56,
}

REASONS = {
    'BEST_ESTIMATE_FROM_VERIFIED_CONTACT': '직접 확인한 접촉·라인 판정을 그대로 사용했습니다.',
    'BEST_ESTIMATE_FROM_VIDEO_ESTIMATE': '원본 착지 가능 영역 전체가 한쪽에 있는 영상 추정을 사용했습니다.',
    'BEST_ESTIMATE_FROM_ESTIMATE_ENVELOPE': '원본 착지 가능 영역이 경계에 걸쳐 있어 더 넓게 차지한 쪽으로 판정했습니다.',
    'BEST_ESTIMATE_FROM_FLIGHT_FAMILY': '주 코트에 연결된 착지 경로의 대표 위치로 판정했습니다.',
    'BEST_ESTIMATE_FROM_LANDING_CANDIDATE': '주 코트 범위 안의 첫 하강·반동(접근 비행 확인 우선) 착지 후보 위치로 판정했습니다.',
    'BEST_ESTIMATE_FROM_TRACK_ENDPOINT': '착지 후보가 없어 하강하던 셔틀 궤적의 마지막 관측 위치로 판정했습니다.',
    'BEST_ESTIMATE_NEAR_BOUNDARY': '판정 위치가 경계에 가까워 신뢰도가 낮습니다.',
    'BEST_ESTIMATE_PRIOR_OVERRODE_WEAK_LOCATION': '위치 근거가 약해 OUT으로 판정하지 않고, 착지가 코트 안일 가능성이 높다는 기본값을 따랐습니다.',
    'BEST_ESTIMATE_COURT_LINES_EXTENDED': '확인된 코트 범위 밖의 위치라 관측된 경계선을 연장해 판정했습니다.',
    'BEST_ESTIMATE_COURT_MODEL_HYPOTHESIS': '확정되지 않은 코트 보정 가설로 위치를 비교했습니다.',
    'BEST_ESTIMATE_DOUBLES_OUTLINE_FOR_SINGLES': '단식 경계를 찾지 못해 복식 외곽 가설로 근사했습니다.',
    'BEST_ESTIMATE_UNAVAILABLE': '코트 경계나 셔틀 위치 근거가 전혀 없어 추정할 수 없습니다.',
    'BEST_ESTIMATE_CAMERA_UNSTABLE': '영상 전체에서 카메라 고정이 확인되지 않아 경계 비교의 신뢰도를 낮췄습니다.',
    'BEST_ESTIMATE_PROCESSING_FAILED': '최선 추정을 계산하는 중 오류가 발생했습니다. 분석 기록의 오류를 확인하세요.',
    'BEST_ESTIMATE_BOUNDARY_EXTRAPOLATED': '판정에 쓴 경계선이 영상에서 직접 측정되지 않고 코트 모형으로 외삽한 선입니다.',
}

_ROLES = ('left', 'right', 'near', 'far')


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _point(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2 and all(_finite(item) for item in value))


def _polygon(value):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3 or not np.isfinite(array).all():
        return None
    return array


def _analysis_width(result):
    analysis = result.get('analysis') or {}
    width = analysis.get('width')
    calibration = result.get('calibration') or {}
    width = width if _finite(width) and width > 0 else calibration.get('image_width')
    return float(width) if _finite(width) and width > 0 else 960.0


# ---------------------------------------------------------------- geometry --

def _primary_margins(point, primary, game):
    """Signed distances (positive inside) to every observed primary boundary."""
    if not isinstance(primary, dict) or primary.get('status') != 'selected':
        return None
    if primary.get('game', 'doubles') != game:
        return None
    rows = []
    for boundary in primary.get('boundary_segments_px') or []:
        if not isinstance(boundary, dict) or boundary.get('role') not in _ROLES:
            continue
        segment, inside = boundary.get('segment_px'), boundary.get('inside_point_px')
        if not (isinstance(segment, (list, tuple)) and len(segment) == 4 and all(_finite(v) for v in segment)
                and _point(inside)):
            continue
        a = np.asarray(segment[:2], dtype=np.float64)
        b = np.asarray(segment[2:], dtype=np.float64)
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length <= 1e-9:
            continue
        normal = np.array([-delta[1], delta[0]]) / length
        side = float((np.asarray(inside, dtype=np.float64) - a) @ normal)
        if abs(side) <= 1e-9:
            continue
        normal *= 1.0 if side > 0 else -1.0
        uncertainty = boundary.get('uncertainty_px')
        rows.append({'role': boundary['role'],
                     'margin_px': float((np.asarray(point, dtype=np.float64) - a) @ normal),
                     'line_uncertainty_px': float(uncertainty) if _finite(uncertainty) and uncertainty >= 0 else 2.5,
                     'observed': boundary.get('observed') is True})
    return rows or None


def _homography_position(point, calibration):
    """Court metres for an analysis-pixel point from the stored court model."""
    try:
        transform = np.asarray(calibration.get('homography'), dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12:
            return None
        from .calibration import points_for_homography
        plane = points_for_homography(np.asarray([point], dtype=np.float64), calibration)
        homogeneous = np.column_stack((plane, np.ones(1))) @ transform.T
        if abs(homogeneous[0, 2]) < 1e-9:
            return None
        court = homogeneous[0, :2] / homogeneous[0, 2]
    except (TypeError, ValueError, KeyError, cv2.error, np.linalg.LinAlgError):
        return None
    if not np.isfinite(court).all() or not (-6 <= court[0] <= 12.1 and -6 <= court[1] <= 19.4):
        return None
    return court


def _rect_margin(court, region):
    x, y = float(court[0]), float(court[1])
    inside = min(x - region['xmin'], region['xmax'] - x, y - region['ymin'], region['ymax'] - y)
    if inside >= 0:
        return inside
    dx = max(region['xmin'] - x, 0.0, x - region['xmax'])
    dy = max(region['ymin'] - y, 0.0, y - region['ymax'])
    return -math.hypot(dx, dy)


def classify_point(point, calibration, rules=None):
    """IN/OUT for one analysis-pixel location, or ``None`` without any court.

    Observed primary-court boundaries come first; beyond their judgment scope
    they are extended, unless an *accepted* full-court model exists. A rejected
    court-model hypothesis is used only without a primary court, and the stored
    outline polygon only without either. The returned margin is positive
    inside the valid area and negative outside it.
    """
    if not _point(point) or not isinstance(calibration, dict):
        return None
    region = rule_region(rules)
    primary = calibration.get('primary_court')
    margins = _primary_margins(point, primary, region['game'])
    within_scope = None
    if margins is not None:
        scope = _polygon((primary or {}).get('judgment_scope_polygon_px'))
        if scope is not None:
            within_scope = cv2.pointPolygonTest(scope.astype(np.float32), (float(point[0]), float(point[1])), False) >= 0
    court = _homography_position(point, calibration)
    # A rejected full-court template is only a hypothesis; the observed primary
    # boundaries (extended if needed) are more reliable than it. An accepted
    # template may still judge points beyond the primary judgment scope.
    trusted_model = court is not None and calibration.get('status') == 'ok'
    codes = []
    if margins is not None and (within_scope is not False or not trusted_model):
        worst = min(margins, key=lambda row: row['margin_px'])
        if within_scope is False:
            codes.append('BEST_ESTIMATE_COURT_LINES_EXTENDED')
        if not worst['observed']:
            codes.append('BEST_ESTIMATE_BOUNDARY_EXTRAPOLATED')
        return {'decision': 'IN' if worst['margin_px'] >= 0 else 'OUT',
                'margin_px': round(worst['margin_px'], 3), 'boundary_role': worst['role'],
                'boundary_observed': worst['observed'],
                'line_uncertainty_px': round(worst['line_uncertainty_px'], 3),
                'method': 'primary_court_boundaries', 'within_judgment_scope': within_scope,
                'boundary_margins_px': {row['role']: round(row['margin_px'], 3) for row in margins},
                'reason_codes': codes}
    if court is not None:
        margin_m = _rect_margin(court, region)
        if calibration.get('status') != 'ok':
            codes.append('BEST_ESTIMATE_COURT_MODEL_HYPOTHESIS')
        return {'decision': 'IN' if margin_m >= 0 else 'OUT', 'margin_m': round(margin_m, 4),
                'court_position_m': [round(float(court[0]), 4), round(float(court[1]), 4)],
                'method': 'homography_court_model', 'calibration_status': calibration.get('status'),
                'within_judgment_scope': within_scope, 'reason_codes': codes}
    # Last resort: the stored court-outline hypothesis in image space (the
    # fitted template's doubles outline). Singles can only approximate it.
    polygon = _polygon(calibration.get('polygon_px'))
    if polygon is not None and len(polygon) >= 3:
        contour = polygon.astype(np.float32)
        if abs(cv2.contourArea(contour)) > 16:
            margin = float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))
            codes = ['BEST_ESTIMATE_COURT_MODEL_HYPOTHESIS']
            if region['game'] != 'doubles':
                codes.append('BEST_ESTIMATE_DOUBLES_OUTLINE_FOR_SINGLES')
            return {'decision': 'IN' if margin >= 0 else 'OUT', 'margin_px': round(margin, 3),
                    'boundary_role': None, 'line_uncertainty_px': 6.0,
                    'method': 'court_outline_hypothesis', 'within_judgment_scope': None,
                    'reason_codes': codes}
    return None


def _clearance(classification, uncertainty_px, width):
    """Boundary clearance in units of the combined position uncertainty."""
    if 'margin_px' in classification:
        scale = max(3.0 * width / 960.0,
                    float(uncertainty_px or 0) + classification.get('line_uncertainty_px', 2.5))
        return abs(classification['margin_px']) / scale
    # About 0.12 m covers the line width plus a typical court-model residual.
    return abs(classification.get('margin_m', 0.0)) / 0.12


def _confidence(basis, clearance, support=1.0):
    prior = BASIS_PRIOR[basis]
    factor = 1.0 - math.exp(-max(0.0, clearance))
    support = min(1.0, max(0.0, support))
    return round(0.5 + (prior - 0.5) * factor * (0.6 + 0.4 * support), 4)


# ------------------------------------------------------------ landing pool --

POOL_LIMIT = 100000
# Only track-geometry options shape the pool; image, model and operator inputs
# are excluded so the pool is reproducible from result.json alone.
_POOL_OPTIONS = ('min_observations', 'min_motion_speed_px_s', 'stationary_speed_px_s',
                 'causal_first_event_ranking', 'native_rebound_guard')


def landing_pool(result, event_config=None):
    """Every landing hypothesis the production generator finds, without a budget.

    The pipeline keeps only the top few candidates for its strict evidence
    stages; the true rally-ending landing is often ranked below that budget.
    The pool is rebuilt from tracks and timestamps alone with an image-free
    stand-in frame that carries only the analysis size.
    """
    from .events import estimate_landings
    tracks = result.get('tracks') or []
    timestamps = result.get('timestamps') or []
    if not tracks or len(timestamps) < 2:
        return []
    analysis = result.get('analysis') or {}
    width = int(analysis.get('width') or (result.get('calibration') or {}).get('image_width') or 960)
    height = int(analysis.get('height') or (result.get('calibration') or {}).get('image_height') or 540)
    source = event_config if event_config is not None else (result.get('config') or {}).get('events') or {}
    options = {key: source[key] for key in _POOL_OPTIONS if key in source}
    options.update(max_candidates=POOL_LIMIT, scene_context_ranking=False, event_family_diversity=False,
                   source_contact_enabled=False)
    calibration = dict(result.get('calibration') or {})
    if isinstance(result.get('camera_stability'), dict):
        calibration.setdefault('camera_stability', result['camera_stability'])
    stand_in = [np.zeros((max(1, height), max(1, width), 0), np.uint8)]
    return estimate_landings(tracks, stand_in, list(timestamps), calibration, options)


# -------------------------------------------------------- candidate ranking --

def _court_proximity(point, calibration, width):
    """1 inside the selected court region, decaying with distance outside it."""
    primary = calibration.get('primary_court') if isinstance(calibration, dict) else None
    polygon = None
    if isinstance(primary, dict) and primary.get('status') == 'selected':
        polygon = _polygon(primary.get('judgment_scope_polygon_px') or primary.get('physical_court_polygon_px'))
    if polygon is None and isinstance(calibration, dict):
        polygon = _polygon(calibration.get('polygon_px'))
    if polygon is None:
        return 0.5
    distance = cv2.pointPolygonTest(polygon.astype(np.float32), (float(point[0]), float(point[1])), True)
    if distance >= 0:
        return 1.0
    return float(math.exp(distance / (40.0 * width / 960.0)))


def score_landing(landing, calibration, width):
    """Heuristic first-landing evidence score in [0, 1] (not a probability)."""
    point = landing.get('contact_px') if _point(landing.get('contact_px')) else landing.get('candidate_px')
    if not _point(point):
        return None
    motion = landing.get('image_motion_event') if isinstance(landing.get('image_motion_event'), dict) else {}
    codes = set(landing.get('reason_codes') or [])
    if motion.get('kind') == 'descent_and_rebound':
        motion_score = 1.0 if motion.get('first_in_track') else 0.45
        if motion.get('point_kind') == 'inferred_object_center':
            motion_score *= 0.8
    elif 'TRACK_TERMINATED_WITHOUT_CONTACT' in codes:
        motion_score = 0.3
    else:
        motion_score = 0.55
    if 'LATER_REBOUND_AFTER_FIRST_CANDIDATE' in codes:
        motion_score = min(motion_score, 0.45)
    flight = landing.get('flight_trajectory_support') if isinstance(landing.get('flight_trajectory_support'), dict) else {}
    flight_score = 1.0 if flight.get('supported_for_review') is True else 0.4 if flight else 0.0
    drop = flight.get('approach_displacement_y_px')
    drop_score = min(1.0, max(0.0, float(drop) / (60.0 * width / 960.0))) if _finite(drop) else 0.0
    continuity = landing.get('identity_continuity') if isinstance(landing.get('identity_continuity'), dict) else {}
    veto = 0.6 if continuity.get('flight_support_veto') is True else 1.0
    confidence = landing.get('confidence')
    raw = min(1.0, max(0.0, float(confidence))) if _finite(confidence) else 0.0
    proximity = _court_proximity(point, calibration, width)
    score = (0.30 * motion_score + 0.20 * flight_score + 0.10 * drop_score
             + 0.15 * raw + 0.25 * proximity) * veto
    if landing.get('contact_observed') is True:
        score = max(score, 0.95)
    return {'score': round(float(score), 4), 'point_px': [float(point[0]), float(point[1])],
            'components': {'motion': round(motion_score, 3), 'flight': round(flight_score, 3),
                           'approach_drop': round(drop_score, 3), 'candidate_confidence': round(raw, 3),
                           'court_proximity': round(proximity, 3), 'continuity_factor': veto}}


def _track_endpoints(tracks, calibration, width, limit=24):
    """Descending/terminating track ends as last-resort landing locations."""
    rows = []
    for track in tracks or []:
        observations = [item for item in (track.get('observations') or [])
                        if isinstance(item, dict) and item.get('observed', True) is not False
                        and _finite(item.get('x')) and _finite(item.get('y')) and _finite(item.get('time'))]
        if len(observations) < 4:
            continue
        observations.sort(key=lambda item: item['time'])
        ys = np.asarray([item['y'] for item in observations], dtype=np.float64)
        lowest = int(np.argmax(ys))  # image y grows toward the floor/near side
        descent = float(ys[lowest] - ys[:lowest + 1].min()) if lowest > 0 else 0.0
        if descent < 4.0 * width / 960.0:
            continue
        end = observations[lowest]
        point = [float(end['x']), float(end['y'])]
        proximity = _court_proximity(point, calibration, width)
        score = (0.5 * min(1.0, descent / (80.0 * width / 960.0)) + 0.5 * proximity) * min(1.0, len(observations) / 12)
        rows.append({'track_id': track.get('track_id'), 'frame_index': end.get('frame_index'),
                     'time': end.get('time'), 'point_px': point, 'score': round(score, 4),
                     'descent_px': round(descent, 3)})
    rows.sort(key=lambda row: -row['score'])
    return rows[:limit]


# --------------------------------------------------------- landing selection --

# IN rate assumed when location evidence is weak. A rally-ending shuttle is
# called OUT only on reasonably strong evidence; with none, IN is likelier.
DEFAULT_PRIOR_IN = 0.7
# How much a classification can be trusted, by the boundary source used.
GEOMETRY_RELIABILITY = {'primary_court_boundaries': 1.0, 'primary_court_extended': 0.75,
                        'homography_court_model': 0.35, 'court_outline_hypothesis': 0.15}


def geometry_reliability(classification):
    if not isinstance(classification, dict):
        return 0.0
    method = classification.get('method')
    if method == 'primary_court_boundaries' and (classification.get('within_judgment_scope') is False
                                                 or classification.get('boundary_observed') is False):
        method = 'primary_court_extended'
    return GEOMETRY_RELIABILITY.get(method, 0.0)


def _scope_distance(point, calibration, extent):
    """Signed distance into the primary judgment scope (None without a court)."""
    primary = calibration.get('primary_court') if isinstance(calibration, dict) else None
    if not isinstance(primary, dict) or primary.get('status') != 'selected':
        return None
    scope = _polygon(primary.get('judgment_scope_polygon_px') or primary.get('physical_court_polygon_px'))
    if scope is None:
        return None
    return float(cv2.pointPolygonTest(scope.astype(np.float32), (float(point[0]), float(point[1])), True)) + extent


def _first_in_track(landing):
    motion = landing.get('image_motion_event') if isinstance(landing.get('image_motion_event'), dict) else {}
    return (not landing.get('causal_event_sequence') and motion.get('first_in_track') is not False
            and 'LATER_REBOUND_AFTER_FIRST_CANDIDATE' not in (landing.get('reason_codes') or []))


def _flight_supported(landing):
    flight = landing.get('flight_trajectory_support')
    return isinstance(flight, dict) and flight.get('supported_for_review') is True


# A rally-ending shuttle that is OUT usually lands within about a metre or two
# of a boundary; candidates that far outside the judged court stay eligible.
LANDING_ZONE_M = 1.5
_DOUBLES_WIDTH_M = 6.1


def _landing_zone_px(calibration):
    """Pixels per LANDING_ZONE_M at the near baseline (0 without a court)."""
    primary = calibration.get('primary_court') if isinstance(calibration, dict) else None
    corners = (primary or {}).get('physical_near_corners_px') or (primary or {}).get('near_corners_px')
    points = _polygon(corners) if isinstance(corners, (list, tuple)) and len(corners) >= 2 else None
    if points is None:
        try:
            points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        except (TypeError, ValueError):
            return 0.0
    if len(points) < 2 or not np.isfinite(points).all():
        return 0.0
    return float(np.linalg.norm(points[1] - points[0])) / _DOUBLES_WIDTH_M * LANDING_ZONE_M


# Selection rank -> how sure the pick is the rally-ending first landing.
RANK_STRENGTH = {4: 0.75, 3: 0.45, 2: 0.6, 1: 0.25}


def _pick(landings, calibration):
    """First landing (in the producer's order) meeting the strongest rule.

    Ranks: a first rebound of its track with a coherent incoming flight inside
    the judgment scope (4), any landing inside the scope (3), a first rebound
    with a coherent flight outside the scope but within the landing zone (2;
    only when nothing lies in the scope, and strong enough to call a clear
    OUT), anything else (1). Returns (landing, rank) or None.
    """
    zone = _landing_zone_px(calibration)
    rows = []
    for index, landing in enumerate(landings):
        point = landing.get('contact_px') if _point(landing.get('contact_px')) else landing.get('candidate_px')
        if not _point(point):
            continue
        extent = max(1.0, float(landing.get('uncertainty_px') or 1.0)) + max(0.0, float(landing.get('contact_radius_px') or 0.0))
        distance = _scope_distance(point, calibration, extent)
        in_scope = distance is not None and distance >= 0
        in_zone = distance is not None and distance >= -zone
        strong = _first_in_track(landing) and _flight_supported(landing)
        rank = 4 if in_scope and strong else 3 if in_scope else 2 if in_zone and strong else 1
        rows.append((rank, -index, landing))
    if not rows:
        return None
    rank, _, landing = max(rows, key=lambda row: (row[0], row[1]))
    return landing, rank


def select_landing(result, calibration, width, rules):
    """Choose the most plausible rally-ending first landing.

    The producer's ranked landings come first; when none lies in the primary
    judgment scope, the unbudgeted landing pool is searched with the same
    rule. Returns (landing, basis, selection_strength, alternatives, pool size).
    """
    landings = [item for item in (result.get('landings') or []) if isinstance(item, dict)]
    estimate = result.get('video_estimate') or {}
    resolution = estimate.get('family_resolution') or {}
    # Only a *selected* family (observed peak, unique on the primary court) is
    # strong location evidence. An inferred-only candidate family is merely a
    # search candidate until native evidence confirms it, so it is not used here.
    family = resolution.get('selected_family') if resolution.get('status') == 'selected' else None
    if isinstance(family, dict):
        representative = (family.get('representative') or {}).get('event_id')
        landing = _landing_by_id(landings, representative) or next(
            (item for item in landings if item.get('event_id') in (family.get('member_event_ids') or [])), None)
        if landing is not None:
            return landing, 'flight_family', 0.8, [], None
    pool_size = None
    picked = _pick(landings, calibration)
    if picked is None or picked[1] < 3:
        pool = landing_pool(result)
        pool_size = len(pool)
        from_pool = _pick(pool, calibration)
        if from_pool is not None and (picked is None or from_pool[1] > picked[1]):
            landing, rank = from_pool
            landing = {**landing, 'event_id': landing.get('event_id') or 'pool_candidate'}
            picked = (landing, rank)
    if picked is None:
        return None
    landing, rank = picked
    strength = RANK_STRENGTH[rank]
    alternatives = []
    for item in landings[:8]:
        point = item.get('candidate_px')
        if _point(point):
            alternatives.append({'event_id': item.get('event_id'), 'frame_index': item.get('frame_index'),
                                 'time': item.get('time'), 'point_px': [round(float(v), 2) for v in point]})
    return landing, 'landing_candidate', strength, alternatives, pool_size


def _decide_with_prior(rules, classification, basis, strength, prior_in, uncertainty_px, width,
                       inside_fraction=None, camera_unstable=False, **fields):
    """IN/OUT from the chosen location, weighted by how much it can be trusted.

    q = selection strength x geometry reliability. P(IN) mixes the location's
    own verdict (or envelope share) with the IN prior for the untrusted part.
    """
    reliability = geometry_reliability(classification)
    if camera_unstable:
        reliability *= 0.5
    clearance = _clearance(classification, uncertainty_px, width)
    sharpness = 1.0 - math.exp(-max(0.0, clearance))  # near-line locations carry less
    q = max(0.0, min(1.0, strength * reliability * (0.5 + 0.5 * sharpness)))
    own_in = inside_fraction if inside_fraction is not None else (1.0 if classification['decision'] == 'IN' else 0.0)
    p_in = q * own_in + (1.0 - q) * prior_in
    decision = 'IN' if p_in >= 0.5 else 'OUT'
    confidence = round(max(p_in, 1.0 - p_in), 4)
    codes = [{'estimate_envelope': 'BEST_ESTIMATE_FROM_ESTIMATE_ENVELOPE',
              'flight_family': 'BEST_ESTIMATE_FROM_FLIGHT_FAMILY',
              'landing_candidate': 'BEST_ESTIMATE_FROM_LANDING_CANDIDATE',
              'track_endpoint': 'BEST_ESTIMATE_FROM_TRACK_ENDPOINT'}[basis]]
    if clearance < 1:
        codes.append('BEST_ESTIMATE_NEAR_BOUNDARY')
    if camera_unstable:
        codes.append('BEST_ESTIMATE_CAMERA_UNSTABLE')
    if decision != classification['decision']:
        codes.append('BEST_ESTIMATE_PRIOR_OVERRODE_WEAK_LOCATION')
    return _result(rules, decision, basis, classification, confidence, codes,
                   evidence_strength=round(q, 4), geometry_reliability=reliability,
                   p_in=round(p_in, 4), prior_in=prior_in, location_decision=classification['decision'],
                   **fields)


# ------------------------------------------------------------- main entry --

def _result(rules, decision, basis, classification, confidence, codes, **fields):
    codes = list(dict.fromkeys(codes + (classification or {}).get('reason_codes', [])))
    return {'schema_version': SCHEMA_VERSION, 'estimator_version': ESTIMATOR_VERSION,
            'decision': decision, 'decision_kind': DECISION_KIND,
            'source': SOURCE, 'basis': basis, 'confidence': confidence, 'rules': deepcopy(rules),
            'classification': classification, 'reason_codes': codes,
            'reasons': [REASONS.get(code, code) for code in codes],
            'contact_observed': False, 'target_identity_verified': False,
            'evaluation_eligible': False, 'calibrated_accuracy': None,
            'coordinate_system': 'analysis_pixels',
            'scope': 'Best available landing evidence; a heuristic estimate, not an observed first contact.',
            **fields}


def unavailable_estimate(rules=None, **fields):
    """The only non-deciding outcome: no court geometry or no shuttle location."""
    try:
        region = rule_region(rules)
        rules = {'game': region['game'], 'phase': region['phase']}
    except (TypeError, ValueError):
        rules = deepcopy(rules)
    return _result(rules, 'REVIEW', None, None, None, ['BEST_ESTIMATE_UNAVAILABLE'], **fields)


_unavailable = unavailable_estimate


def failed_estimate(rules, error):
    """A processing error, kept distinct from 'no geometry or location'."""
    try:
        region = rule_region(rules)
        rules = {'game': region['game'], 'phase': region['phase']}
    except (TypeError, ValueError):
        rules = deepcopy(rules)
    return _result(rules, 'REVIEW', None, None, None, ['BEST_ESTIMATE_PROCESSING_FAILED'],
                   error=f'{type(error).__name__}: {str(error)[:300]}')


def write_best_estimate_overlay(frames, calibration, estimate, output_dir):
    """Scene frame with court lines, the classified point and its basis."""
    from pathlib import Path
    from .evidence import draw_court_lines
    from .image_io import write_image
    if (not frames or not isinstance(estimate, dict) or estimate.get('decision') not in ('IN', 'OUT')
            or estimate.get('basis') in (None, 'verified_contact', 'video_estimate')
            or not _point(estimate.get('point_px'))):
        return None
    index = estimate.get('frame_index')
    index = int(index) if _finite(index) else len(frames) // 2
    index = max(0, min(len(frames) - 1, index))
    canvas = frames[index].copy()
    draw_court_lines(canvas, calibration)
    x, y = (int(round(value)) for value in estimate['point_px'])
    color = (70, 230, 110) if estimate['decision'] == 'IN' else (70, 110, 255)
    cv2.circle(canvas, (x, y), 14, color, 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
    for alternative in (estimate.get('alternatives') or [])[1:6]:
        if _point(alternative.get('point_px')):
            ax, ay = (int(round(value)) for value in alternative['point_px'])
            cv2.circle(canvas, (ax, ay), 7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{estimate['decision']} (best estimate)", (max(0, x - 70), max(60, y - 22)),
                cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (20, 28, 34), -1)
    game = (estimate.get('rules') or {}).get('game', 'doubles').upper()
    confidence = estimate.get('confidence')
    confidence_text = f' | heuristic confidence {confidence:.2f}' if _finite(confidence) else ''
    cv2.putText(canvas, f"BEST ESTIMATE: {estimate['decision']} | {game} | frame {index}{confidence_text}",
                (12, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"basis: {estimate.get('basis')} - not an observed first contact",
                (12, 40), cv2.FONT_HERSHEY_SIMPLEX, .42, (220, 225, 230), 1, cv2.LINE_AA)
    path = Path(output_dir).resolve() / 'evidence/best-estimate.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, canvas)
    return path.relative_to(Path(output_dir).resolve()).as_posix()


def _envelope_majority(polygon, calibration, rules):
    """Classify a possible-contact envelope by the share of its area on each side."""
    points = _polygon(polygon)
    if points is None:
        return None
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    contour = points.astype(np.float32)
    samples = [(x, y) for x in np.linspace(xmin, xmax, 9) for y in np.linspace(ymin, ymax, 9)
               if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0]
    samples = samples or [tuple(points.mean(axis=0))]
    labels = [classify_point([float(x), float(y)], calibration, rules) for x, y in samples]
    labels = [row for row in labels if row is not None]
    if not labels:
        return None
    inside = sum(row['decision'] == 'IN' for row in labels) / len(labels)
    centre = classify_point([float(v) for v in points.mean(axis=0)], calibration, rules)
    if centre is None:
        return None
    decision = 'IN' if inside > 0.5 or (inside == 0.5 and centre['decision'] == 'IN') else 'OUT'
    return {**centre, 'decision': decision, 'inside_area_fraction': round(inside, 4),
            'sample_count': len(labels), 'point_px': [round(float(v), 3) for v in points.mean(axis=0)]}


def _envelope_in_frame(polygon, result):
    """The A/B transport only accepts an envelope strictly inside the frame.

    Reusing an envelope it would reject would label this tier 'video_estimate'
    while the pair shows no valid video estimate, so such an envelope is
    treated as a straddling one instead (the next tier).
    """
    points = _polygon(polygon)
    analysis = result.get('analysis') or {}
    width, height = analysis.get('width'), analysis.get('height')
    if points is None or not (3 <= len(points) <= 64) or not (_finite(width) and _finite(height)):
        return False
    return bool((points >= 0).all() and (points[:, 0] < width).all() and (points[:, 1] < height).all())


def _landing_by_id(landings, identifier):
    return next((item for item in landings if isinstance(item, dict) and item.get('event_id') == identifier), None)


def estimate_best_verdict(result, rules=None, *, prior_in=None):
    """Return the best available IN/OUT estimate for one analysed view.

    ``result`` is a pipeline result (or a dict with the same keys). REVIEW is
    returned only when no court geometry or no shuttle location exists at all.
    ``prior_in`` is the IN rate assumed when the location evidence is weak
    (default :data:`DEFAULT_PRIOR_IN`, or ``config.best_estimate.prior_in``).
    """
    if prior_in is None:
        configured = ((result.get('config') or {}).get('best_estimate') or {}).get('prior_in')
        prior_in = float(configured) if _finite(configured) and 0 < configured < 1 else DEFAULT_PRIOR_IN
    rules = rules if rules is not None else (result.get('config') or {}).get('rules')
    region = rule_region(rules)
    selected_rules = {'game': region['game'], 'phase': region['phase']}
    calibration = dict(result.get('calibration') or {})
    if isinstance(result.get('camera_stability'), dict) and 'camera_stability' not in calibration:
        calibration['camera_stability'] = result['camera_stability']
    width = _analysis_width(result)
    landings = [item for item in (result.get('landings') or []) if isinstance(item, dict)]
    stability = calibration.get('camera_stability') if isinstance(calibration.get('camera_stability'), dict) else {}
    unstable = (stability.get('status') != 'ok' or stability.get('all_frames_sampled') is not True
                or 'CAMERA_MOVED_FROM_CALIBRATION_REFERENCE' in (stability.get('reason_codes') or []))

    decision = result.get('decision') or {}
    if decision.get('decision') in ('IN', 'OUT'):
        point = decision.get('contact_px') or (result.get('selected_landing') or {}).get('contact_px')
        classification = classify_point(point, calibration, rules) if _point(point) else None
        return _result(selected_rules, decision['decision'], 'verified_contact', classification,
                       BASIS_PRIOR['verified_contact'], ['BEST_ESTIMATE_FROM_VERIFIED_CONTACT'],
                       point_px=point if _point(point) else None)

    estimate = result.get('video_estimate') or {}
    if (estimate.get('eligible') is True and estimate.get('decision') in ('IN', 'OUT')
            and _envelope_in_frame(estimate.get('possible_contact_polygon_px'), result)):
        polygon = _polygon(estimate.get('possible_contact_polygon_px'))
        point = [round(float(v), 3) for v in polygon.mean(axis=0)] if polygon is not None else None
        classification = classify_point(point, calibration, rules) if point else None
        geometry = estimate.get('geometry') or {}
        clearance = geometry.get('minimum_clearance_px')
        factor = (float(clearance) / max(3.0 * width / 960.0, float(estimate.get('uncertainty_px') or 0))
                  if _finite(clearance) else 1.0)
        return _result(selected_rules, estimate['decision'], 'video_estimate', classification,
                       _confidence('video_estimate', 1.0 + factor), ['BEST_ESTIMATE_FROM_VIDEO_ESTIMATE'],
                       point_px=point, event_ids=list(estimate.get('member_event_ids') or []),
                       frame_interval=estimate.get('frame_interval'), time_interval=estimate.get('time_interval'))

    if estimate.get('possible_contact_polygon_px') is not None:
        majority = _envelope_majority(estimate.get('possible_contact_polygon_px'), calibration, rules)
        if majority is not None:
            # Native evidence of a unique flight family: strong location
            # evidence, so the side holding most of the envelope decides.
            return _decide_with_prior(
                selected_rules, majority, 'estimate_envelope', 0.9, prior_in, estimate.get('uncertainty_px'),
                width, inside_fraction=majority['inside_area_fraction'], camera_unstable=unstable,
                point_px=majority['point_px'], event_ids=list(estimate.get('member_event_ids') or []),
                frame_interval=estimate.get('frame_interval'), time_interval=estimate.get('time_interval'))

    selection = select_landing(result, calibration, width, rules)
    if selection is not None:
        landing, basis, strength, alternatives, pool_size = selection
        point = landing.get('contact_px') if _point(landing.get('contact_px')) else landing.get('candidate_px')
        classification = classify_point(point, calibration, rules)
        if classification is not None:
            return _decide_with_prior(
                selected_rules, classification, basis, strength, prior_in,
                landing.get('uncertainty_px'), width, camera_unstable=unstable,
                point_px=[round(float(v), 3) for v in point], event_ids=[landing.get('event_id')],
                frame_index=landing.get('frame_index'), time=landing.get('time'),
                track_id=landing.get('track_id'), alternatives=alternatives[:8], pool_size=pool_size)

    endpoints = _track_endpoints(result.get('tracks'), calibration, width)
    for endpoint in endpoints:
        classification = classify_point(endpoint['point_px'], calibration, rules)
        if classification is None:
            continue
        return _decide_with_prior(
            selected_rules, classification, 'track_endpoint', 0.15 * endpoint['score'], prior_in, 0.0, width,
            camera_unstable=unstable,
            point_px=endpoint['point_px'], frame_index=endpoint['frame_index'],
            time=endpoint['time'], track_id=endpoint['track_id'])
    return _unavailable(selected_rules)


def refresh_best_estimate(result, rules=None):
    """Recompute a missing or outdated best estimate from the saved analysis.

    The tier reads only result.json evidence, so a newer estimator can be
    applied without decoding the video again. The overlay image is not
    redrawn here; a stale artifact path is dropped instead of misattributed.
    """
    current = result.get("best_estimate")
    if isinstance(current, dict) and current.get("estimator_version") == ESTIMATOR_VERSION:
        return result
    expected = rules if rules is not None else (result.get("config") or {}).get("rules")
    try:
        estimate = estimate_best_verdict(result, expected)
    except Exception as error:  # an optional tier must not discard the completed analysis
        logger.exception("Could not refresh the best estimate")
        estimate = failed_estimate(expected, error)
    estimate["source_sha256"] = result.get("source_sha256")
    # The overlay burns in the verdict, so it is reused only for an identical one.
    if (isinstance(current, dict) and current.get("artifact_path")
            and all(current.get(key) == estimate.get(key)
                    for key in ("point_px", "frame_index", "decision", "confidence", "basis", "rules"))):
        estimate["artifact_path"] = current["artifact_path"]
    return {**result, "best_estimate": estimate}
