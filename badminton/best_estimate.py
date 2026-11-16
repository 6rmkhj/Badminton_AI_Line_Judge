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
from .shuttle_appearance import MAX_FOREGROUND_FRACTION, METHOD as APPEARANCE_METHOD, ForegroundProbe, is_isolated


logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1
# The tier is recomputable from result.json, so it is versioned on its own
# instead of invalidating the (expensive) analysis fingerprint.
ESTIMATOR_VERSION = hashlib.sha256(
    Path(__file__).read_bytes() + (Path(__file__).parent / 'shuttle_appearance.py').read_bytes()).hexdigest()[:16]
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
    # Kept so stored pre-0.5.16 estimates still validate until refreshed.
    'track_endpoint': 0.56,
    'no_landing_found': 0.5,
}

REASONS = {
    'BEST_ESTIMATE_FROM_VERIFIED_CONTACT': '직접 확인한 접촉·라인 판정을 그대로 사용했습니다.',
    'BEST_ESTIMATE_FROM_VIDEO_ESTIMATE': '원본 착지 가능 영역 전체가 한쪽에 있는 영상 추정을 사용했습니다.',
    'BEST_ESTIMATE_FROM_ESTIMATE_ENVELOPE': '원본 착지 가능 영역이 경계에 걸쳐 있어 더 넓게 차지한 쪽으로 판정했습니다.',
    'BEST_ESTIMATE_FROM_FLIGHT_FAMILY': '주 코트에 연결된 착지 경로의 대표 위치로 판정했습니다.',
    'BEST_ESTIMATE_FROM_LANDING_CANDIDATE': '접근 비행 뒤 트랙의 첫 반동이며 작은 단독 물체인 착지 후보 위치로 판정했습니다.',
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
    'BEST_ESTIMATE_SKIPPED_NON_SHUTTLE_OBJECTS': '선수 몸·다리·라켓처럼 셔틀보다 큰 움직이는 물체 위의 후보는 셔틀 착지로 보지 않고 건너뛰었습니다.',
    'BEST_ESTIMATE_NO_SHUTTLE_LANDING': '셔틀로 볼 수 있는 착지(하강 비행 뒤 첫 반동, 작은 단독 물체)를 찾지 못해 위치를 표시하지 않고, 착지가 코트 안일 가능성이 높다는 기본값을 따랐습니다.',
    'BEST_ESTIMATE_APPEARANCE_UNCHECKED': '후보가 셔틀처럼 작은 물체인지 영상으로 확인하지 못했습니다.',
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


# How sure a shuttle-like pick is the rally-ending first landing: inside the
# judgment scope, or outside it but within the landing zone (a clear OUT).
SCOPE_STRENGTH = {'in_scope': 0.75, 'landing_zone': 0.6}


def _candidate_point(landing):
    return landing.get('contact_px') if _point(landing.get('contact_px')) else landing.get('candidate_px')


def _appearance(fraction, checked, skipped):
    return {'checked': checked, 'method': APPEARANCE_METHOD,
            'foreground_fraction': None if fraction is None else round(fraction, 4),
            'max_foreground_fraction': MAX_FOREGROUND_FRACTION, 'isolated_object': is_isolated(fraction),
            'skipped_larger_object_candidates': skipped}


def estimate_landing(result, estimate):
    """The landing record behind a located best estimate, for display.

    Returns the producer's candidate, or the matching unbudgeted-pool landing,
    whose track and frame the estimate used; None when the estimate shows no
    location (for example ``no_landing_found``) or the record cannot be found.
    """
    if not isinstance(estimate, dict) or estimate.get('decision') not in ('IN', 'OUT') \
            or estimate.get('basis') not in ('landing_candidate', 'flight_family') or not _point(estimate.get('point_px')):
        return None
    frame, track = estimate.get('frame_index'), estimate.get('track_id')
    ids = set(estimate.get('event_ids') or [])
    for item in result.get('landings') or []:
        if isinstance(item, dict) and item.get('event_id') in ids and item.get('frame_index') == frame:
            return item
    for item in landing_pool(result):
        point = _candidate_point(item)
        if (item.get('track_id') == track and item.get('frame_index') == frame and _point(point)
                and np.allclose(point, estimate['point_px'], atol=0.01)):
            return {**item, 'event_id': 'pool_' + str(item.get('event_id') or 'candidate')}
    return None


def _beyond_far_edge(point, calibration):
    """Whether a point lies past the far edge of the judged court area.

    Past that edge (the far baseline, or the short service line of a
    near-half court) the image also shows the net, the far court and the wall
    behind it, so a "rebound" there is far more often a shuttle struck in the
    air than a floor contact, and this camera does not judge that area.
    """
    primary = calibration.get('primary_court') if isinstance(calibration, dict) else None
    polygon = _polygon((primary or {}).get('physical_court_polygon_px')) if isinstance(primary, dict) else None
    if polygon is None or len(polygon) != 4:
        return False
    # Corners are ordered near-left, near-right, far-right, far-left.
    start, end = polygon[2], polygon[3]
    near = (polygon[0] + polygon[1]) / 2.0

    def side(value):
        return float((end[0] - start[0]) * (value[1] - start[1]) - (end[1] - start[1]) * (value[0] - start[0]))

    return side(np.asarray(point, dtype=np.float64)) * side(near) < 0


def _pick(landings, calibration, probe=None):
    """The first shuttle-like landing, in the producer's order.

    A landing is shuttle-like only when it is the first rebound of its track
    after a coherent incoming flight (a descent, not a knee or foot motion
    without one) and, when frames are available, a small isolated object
    rather than part of a larger moving body. Inside the judgment scope beats
    the landing zone just outside its sidelines or near baseline; anything
    farther, or past the far edge of the judged area, is not used. Returns
    (landing, zone, appearance, skipped_count) with landing None when nothing
    qualifies.
    """
    zone_px = _landing_zone_px(calibration)
    rows = []
    for index, landing in enumerate(landings):
        point = _candidate_point(landing)
        if not _point(point) or not (_first_in_track(landing) and _flight_supported(landing)):
            continue
        extent = max(1.0, float(landing.get('uncertainty_px') or 1.0)) + max(0.0, float(landing.get('contact_radius_px') or 0.0))
        distance = _scope_distance(point, calibration, extent)
        if distance is None or distance < -zone_px or (distance < 0 and _beyond_far_edge(point, calibration)):
            continue
        rows.append((1 if distance >= 0 else 0, -index, landing, point))
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    fractions = (probe.fractions([(row[2].get('frame_index'), row[3]) for row in rows])
                 if probe is not None and rows else [None] * len(rows))
    skipped = 0
    for row, fraction in zip(rows, fractions):
        if is_isolated(fraction) is False:
            skipped += 1
            continue
        zone = 'in_scope' if row[0] else 'landing_zone'
        return row[2], zone, _appearance(fraction, fraction is not None, skipped), skipped
    return None, None, None, skipped


def select_landing(result, calibration, width, rules, probe=None):
    """Choose the rally-ending first landing among shuttle-like candidates.

    The producer's ranked landings come first; when none of them is a
    shuttle-like landing inside the judgment scope, the unbudgeted landing
    pool is searched with the same rule. Returns (landing, basis, strength,
    alternatives, pool size, appearance, skipped) or
    (None, ..., skipped) when nothing qualifies.
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
            return landing, 'flight_family', 0.8, [], None, None, 0
    pool_size = None
    landing, zone, appearance, skipped = _pick(landings, calibration, probe)
    if zone != 'in_scope':
        pool = landing_pool(result)
        pool_size = len(pool)
        found, found_zone, found_appearance, found_skipped = _pick(pool, calibration, probe)
        skipped = max(skipped, found_skipped)
        if found is not None and (landing is None or found_zone == 'in_scope'):
            # Pool events are numbered independently of the producer's list, so
            # their ids are prefixed to never alias a different stored candidate.
            landing, zone, appearance = ({**found, 'event_id': 'pool_' + str(found.get('event_id') or 'candidate')},
                                         found_zone, found_appearance)
    if landing is None:
        return None, None, None, [], pool_size, None, skipped
    alternatives = []
    for item in landings[:8]:
        point = item.get('candidate_px')
        if _point(point):
            alternatives.append({'event_id': item.get('event_id'), 'frame_index': item.get('frame_index'),
                                 'time': item.get('time'), 'point_px': [round(float(v), 2) for v in point]})
    return landing, 'landing_candidate', SCOPE_STRENGTH[zone], alternatives, pool_size, appearance, skipped


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
              'landing_candidate': 'BEST_ESTIMATE_FROM_LANDING_CANDIDATE'}[basis]]
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


def estimate_best_verdict(result, rules=None, *, prior_in=None, probe=None):
    """Return the best available IN/OUT estimate for one analysed view.

    ``result`` is a pipeline result (or a dict with the same keys). REVIEW is
    returned only when no court geometry or no shuttle location exists at all.
    ``prior_in`` is the IN rate assumed when the location evidence is weak
    (default :data:`DEFAULT_PRIOR_IN`, or ``config.best_estimate.prior_in``).
    ``probe`` (a :class:`ForegroundProbe` over the analysis frames) lets the
    landing choice skip candidates on players' bodies; without it the choice
    uses track evidence alone and says so.
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

    # A moving camera breaks the clip background the appearance check needs.
    # Why the appearance check could not run, if it could not: refresh redoes
    # an estimate only for missing frames, never for a camera that moved.
    unchecked_reason = 'camera_unstable' if unstable and probe is not None else None if probe is not None else 'no_frames'
    probe = None if unstable else probe
    landing, basis, strength, alternatives, pool_size, appearance, skipped = select_landing(
        result, calibration, width, rules, probe)
    if landing is not None:
        point = _candidate_point(landing)
        classification = classify_point(point, calibration, rules)
        if classification is not None:
            estimate = _decide_with_prior(
                selected_rules, classification, basis, strength, prior_in,
                landing.get('uncertainty_px'), width, camera_unstable=unstable,
                point_px=[round(float(v), 3) for v in point], event_ids=[landing.get('event_id')],
                frame_index=landing.get('frame_index'), time=landing.get('time'),
                track_id=landing.get('track_id'), alternatives=alternatives[:8], pool_size=pool_size)
            return _with_appearance(estimate, appearance, unchecked_reason)
    if not (result.get('tracks') or landings) or not _has_geometry(calibration, rules, width, result):
        return _unavailable(selected_rules)
    return _prior_only(selected_rules, prior_in, skipped, probe is not None, pool_size, unchecked_reason)


def _has_geometry(calibration, rules, width, result):
    """Whether any court boundary can be compared (probed at the frame centre)."""
    analysis = result.get('analysis') or {}
    height = analysis.get('height') or calibration.get('image_height') or width * 9 / 16
    return classify_point([width / 2.0, float(height) / 2.0], calibration, rules) is not None


def _prior_only(rules, prior_in, skipped, checked, pool_size, unchecked_reason=None):
    """No shuttle-like landing: decide by the IN prior and show no location.

    A knee, foot or racket motion is never presented as the shuttle. The view
    keeps an IN/OUT label from the prior with zero evidence strength, so in a
    pair any view with a located landing outranks it.
    """
    decision = 'IN' if prior_in >= 0.5 else 'OUT'
    codes = ['BEST_ESTIMATE_NO_SHUTTLE_LANDING']
    if skipped:
        codes.append('BEST_ESTIMATE_SKIPPED_NON_SHUTTLE_OBJECTS')
    if not checked:
        codes.append('BEST_ESTIMATE_APPEARANCE_UNCHECKED')
    return _result(rules, decision, 'no_landing_found', None, round(max(prior_in, 1.0 - prior_in), 4), codes,
                   evidence_strength=0.0, geometry_reliability=None, p_in=round(prior_in, 4),
                   prior_in=prior_in, location_decision=None, point_px=None, pool_size=pool_size,
                   appearance={**_appearance(None, checked, skipped), 'unchecked_reason': unchecked_reason})


def _with_appearance(estimate, appearance, unchecked_reason=None):
    """Attach the shuttle-appearance check and its reason codes."""
    if appearance is None:
        return estimate
    if not appearance.get('checked'):
        # A probe that ran but could not measure (no background, frame out of
        # range) is final too; only absent frames are worth a later redo.
        appearance = {**appearance, 'unchecked_reason': unchecked_reason or 'not_measurable'}
    codes = list(estimate.get('reason_codes') or [])
    if not appearance.get('checked'):
        codes.append('BEST_ESTIMATE_APPEARANCE_UNCHECKED')
    elif appearance.get('skipped_larger_object_candidates'):
        codes.append('BEST_ESTIMATE_SKIPPED_NON_SHUTTLE_OBJECTS')
    return {**estimate, 'appearance': appearance, 'reason_codes': codes,
            'reasons': [REASONS.get(code, code) for code in codes]}


def preview_probe(result, output_dir):
    """Appearance probe over a saved analysis's ``preview.mp4`` (or None)."""
    if output_dir is None:
        return None
    analysis = result.get("analysis") or {}
    count = analysis.get("frame_count") or len(result.get("timestamps") or [])
    return ForegroundProbe.from_preview(Path(output_dir) / "preview.mp4", count)


def _needs_refresh(current, probe):
    if not isinstance(current, dict) or current.get("estimator_version") != ESTIMATOR_VERSION:
        return True
    # An estimate made without frames is redone once the preview is available.
    appearance = current.get("appearance")
    return (probe is not None and isinstance(appearance, dict) and appearance.get("checked") is False
            and appearance.get("unchecked_reason") == "no_frames")


def refresh_best_estimate(result, rules=None, output_dir=None):
    """Recompute a missing or outdated best estimate from the saved analysis.

    The tier reads result.json evidence plus, when ``output_dir`` holds the
    saved ``preview.mp4``, the analysis frames for the shuttle-appearance
    check and the evidence image. Without the preview the overlay is not
    redrawn and a stale artifact path is dropped instead of misattributed.
    """
    current = result.get("best_estimate")
    probe = preview_probe(result, output_dir)
    if not _needs_refresh(current, probe):
        return result
    expected = rules if rules is not None else (result.get("config") or {}).get("rules")
    try:
        estimate = estimate_best_verdict(result, expected, probe=probe)
    except Exception as error:  # an optional tier must not discard the completed analysis
        logger.exception("Could not refresh the best estimate")
        estimate = failed_estimate(expected, error)
    estimate["source_sha256"] = result.get("source_sha256")
    for key in ("assistance_used", "evaluation_eligible"):
        if isinstance(current, dict) and key in current and current.get("assistance_used"):
            estimate[key] = current[key]
    same = (isinstance(current, dict) and current.get("artifact_path")
            and all(current.get(key) == estimate.get(key)
                    for key in ("point_px", "frame_index", "decision", "confidence", "basis", "rules")))
    if same:
        # The overlay burns in the verdict, so it is reused only for an identical one.
        estimate["artifact_path"] = current["artifact_path"]
    elif probe is not None:
        estimate = _redraw_overlay(result, estimate, probe, output_dir)
    return {**result, "best_estimate": estimate}


def _redraw_overlay(result, estimate, probe, output_dir):
    index = estimate.get("frame_index")
    if estimate.get("decision") not in ("IN", "OUT") or not _finite(index):
        return estimate
    try:
        frame = probe.frame(int(index))
        if frame is None:
            return estimate
        # The writer indexes frames by frame_index; give it a sparse view.
        frames = _SingleFrame(frame, int(index), len(result.get("timestamps") or []) or int(index) + 1)
        path = write_best_estimate_overlay(frames, result.get("calibration") or {}, estimate, output_dir)
    except Exception:  # evidence is optional; the verdict stands without it
        logger.exception("Could not redraw the best-estimate evidence image")
        return estimate
    return {**estimate, "artifact_path": path} if path else estimate


class _SingleFrame:
    """Sequence stand-in exposing one decoded frame at its clip index."""

    def __init__(self, frame, index, count):
        self._frame, self._index, self._count = frame, index, count

    def __len__(self):
        return self._count

    def __bool__(self):
        return True

    def __getitem__(self, index):
        if index != self._index:
            raise IndexError(index)
        return self._frame
