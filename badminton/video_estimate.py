"""Bounded video estimates, separate from verified physical-contact decisions.

Every possible position must agree. An uncertainty envelope is not a physical
shuttle footprint: intersecting the court at one possible position cannot prove
IN. Nothing here modifies contact_observed, identity confidence, or strict labels.
"""
from __future__ import annotations

from copy import deepcopy
import math

import cv2
import numpy as np

from .decision import _decide_primary_court, _primary_polygon, rule_region


SOURCE = 'native_temporal_landing_envelope'
REASONS = {
    'VIDEO_ESTIMATE_DISABLED': '영상 추정 판정이 꺼져 있습니다.',
    'VIDEO_ESTIMATE_RULES_UNSUPPORTED': '선택한 코트 선과 단식·복식 랠리 규칙이 일치하지 않습니다.',
    'VIDEO_ESTIMATE_CALIBRATION_UNSUPPORTED': '영상 추정에 필요한 주 코트의 관측 경계가 부족합니다.',
    'VIDEO_ESTIMATE_CAMERA_UNSTABLE': '전체 구간에서 카메라 고정이 확인되지 않았습니다.',
    'VIDEO_ESTIMATE_NO_UNIQUE_EVENT': '주 코트에서 하나의 착지 경로로 좁히지 못했습니다.',
    'VIDEO_ESTIMATE_NATIVE_SUPPORT_MISSING': '원본 프레임에서 연속된 착지 전후 물체 영역을 확보하지 못했습니다.',
    'VIDEO_ESTIMATE_INVALID_ENVELOPE': '가능한 접촉 영역이나 위치 오차를 유효하게 계산하지 못했습니다.',
    'VIDEO_ESTIMATE_ALL_POSITIONS_INSIDE': '위치 오차를 포함한 가능한 접촉 영역 전체가 확인된 코트 내부에 있습니다.',
    'VIDEO_ESTIMATE_ALL_POSITIONS_OUTSIDE': '위치 오차를 포함한 가능한 접촉 영역 전체가 관측된 경기 경계 밖에 있습니다.',
    'VIDEO_ESTIMATE_BOUNDARY_AMBIGUOUS': '가능한 접촉 위치에 따라 IN·OUT이 달라져 추정 판정을 보류합니다.',
    'VIDEO_ESTIMATE_OUTSIDE_SUPPORTED_SCOPE': '접촉 가능한 영역이 이 시점의 확인 가능한 코트 범위를 벗어납니다.',
    'VIDEO_ESTIMATE_PROCESSING_FAILED': '원본 착지 범위를 계산하는 중 오류가 발생했습니다.',
}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _result(rules, codes, **fields):
    return {'schema_version': 1, 'decision': 'REVIEW', 'eligible': False,
            'decision_kind': 'video_estimate', 'source': SOURCE,
            'contact_observed': False, 'target_identity_verified': False,
            'evaluation_eligible': False, 'calibrated_accuracy': None,
            'coordinate_system': 'analysis_pixels', 'rules': deepcopy(rules),
            'reason_codes': codes, 'reasons': [REASONS.get(code, code) for code in codes],
            'scope': 'Video estimate from native observations; first physical contact is not directly verified.',
            **fields}


def evaluate_possible_region(polygon, uncertainty, calibration, rules=None):
    """Universal convex-region geometry, with all existing boundary safeguards.

The internal boundary helper evaluates vertices as *geometry probes* without
passing them through or fabricating physical-contact qualification. Convexity
makes all-vertices-IN sufficient for the whole uncertainty-expanded region.
OUT additionally requires one common finite observed edge for the whole region.
"""
    base = {'status': 'review', 'decision': 'REVIEW', 'universal_region_test': True,
            'geometry_only': True, 'reason_codes': [], 'boundary_margins': []}
    if not isinstance(calibration, dict):
        return {**base, 'reason_codes': ['VIDEO_ESTIMATE_CALIBRATION_UNSUPPORTED']}
    try:
        region = rule_region(rules)
        selected = {'game': region['game'], 'phase': region['phase']}
        primary = calibration.get('primary_court') or {}
        if not isinstance(primary, dict):
            raise ValueError('invalid primary court record')
        if (region['phase'] != 'rally' or primary.get('game') != region['game']
                or primary.get('rules') != selected):
            return {**base, 'reason_codes': ['VIDEO_ESTIMATE_RULES_UNSUPPORTED']}
        score = primary.get('confidence')
        if (primary.get('status') != 'selected' or primary.get('region_supported') is not True
                or calibration.get('manual_reference') or calibration.get('method') == 'manual_reference'
                or not _number(score) or not .65 <= score <= 1):
            return {**base, 'reason_codes': ['VIDEO_ESTIMATE_CALIBRATION_UNSUPPORTED']}
        stability = calibration.get('camera_stability') or {}
        if not isinstance(stability, dict):
            raise ValueError('invalid camera stability record')
        if (stability.get('status') != 'ok' or stability.get('all_frames_sampled') is not True
                or not isinstance(stability.get('reason_codes', []), list)
                or 'CAMERA_MOVED_FROM_CALIBRATION_REFERENCE' in stability.get('reason_codes', [])):
            return {**base, 'reason_codes': ['VIDEO_ESTIMATE_CAMERA_UNSTABLE']}
        width, height = calibration['image_width'], calibration['image_height']
        if not (_number(width) and _number(height) and min(width, height) > 1
                and _number(uncertainty) and 0 <= uncertainty <= math.hypot(width, height)):
            raise ValueError('invalid image dimensions or uncertainty')
        if (not isinstance(polygon, (list, tuple, np.ndarray)) or
                any(not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 2
                    or not all(_number(value) for value in point) for point in polygon)):
            raise ValueError('invalid possible-region coordinates')
        points = _primary_polygon(polygon, width, height)
        local = deepcopy(calibration)
        if primary.get('scope') == 'full' and not primary.get('judgment_scope_polygon_px'):
            from .primary_court import _scope_polygon
            physical = _primary_polygon(primary['physical_court_polygon_px'], width, height)
            local['primary_court']['judgment_scope_polygon_px'] = _scope_polygon(
                physical, int(width), int(height), max(12., 24.*width/960.)).tolist()
        probes = [_decide_primary_court({'contact_px': point.tolist(), 'contact_radius_px': 0,
                                         'uncertainty_px': uncertainty}, local, selected)
                  for point in points]
        if any('INVALID_CONTACT_GEOMETRY_OR_CALIBRATION' in row.get('reason_codes', []) for row in probes):
            raise ValueError('invalid finite boundary geometry')
        if any('CONTACT_OUTSIDE_PRIMARY_COURT_SCOPE' in row.get('reason_codes', []) for row in probes):
            return {**base, 'reason_codes': ['VIDEO_ESTIMATE_OUTSIDE_SUPPORTED_SCOPE']}
        if any(not row.get('boundaries') for row in probes):
            return {**base, 'reason_codes': ['VIDEO_ESTIMATE_CALIBRATION_UNSUPPORTED']}
        boundaries = []
        for first in probes[0]['boundaries']:
            role = first['role']
            rows = [next(row for row in probe['boundaries'] if row['role'] == role) for probe in probes]
            ends = np.asarray(first['segment_px'], dtype=float).reshape(2, 2)
            axis = ends[1] - ends[0]
            length = float(np.linalg.norm(axis))
            stations = (points - ends[0]) @ (axis / length)
            total_error = float(first['uncertainty_px'])
            finite_extent = bool(stations.min() >= total_error and stations.max() <= length-total_error)
            all_out = finite_extent and all(row['robust_outside'] for row in rows)
            boundaries.append({'role': role, 'segment_px': first['segment_px'],
                               'minimum_inside_margin_px': min(row['min_inside_margin_px'] for row in rows),
                               'maximum_inside_margin_px': max(row['max_inside_margin_px'] for row in rows),
                               'combined_uncertainty_px': total_error,
                               'whole_region_within_observed_extent': finite_extent,
                               'whole_region_outside': all_out})
        common_outside = [row for row in boundaries if row['whole_region_outside']]
        if all(row['decision'] == 'IN' for row in probes):
            label, code = 'IN', 'VIDEO_ESTIMATE_ALL_POSITIONS_INSIDE'
            interior = primary.get('interior_polygon_px')
            if interior is None:
                interior = primary.get('polygon_px')
            interior = np.asarray(interior, dtype=np.float32)
            error = max(row['combined_uncertainty_px'] for row in boundaries)
            clearance = min(cv2.pointPolygonTest(interior, tuple(map(float, point)), True) for point in points)-error
        elif common_outside:
            label, code = 'OUT', 'VIDEO_ESTIMATE_ALL_POSITIONS_OUTSIDE'
            clearance = max(-row['maximum_inside_margin_px']-row['combined_uncertainty_px'] for row in common_outside)
        else:
            label, code, clearance = 'REVIEW', 'VIDEO_ESTIMATE_BOUNDARY_AMBIGUOUS', None
        return {**base, 'status': 'resolved' if label != 'REVIEW' else 'review', 'decision': label,
                'reason_codes': [code], 'boundary_margins': boundaries,
                'minimum_clearance_px': None if clearance is None else round(float(clearance), 4),
                'outside_boundary_roles': [row['role'] for row in common_outside],
                'rules': selected}
    except (ValueError, TypeError, KeyError, IndexError, StopIteration, OverflowError, cv2.error):
        return {**base, 'reason_codes': ['VIDEO_ESTIMATE_INVALID_ENVELOPE']}


def estimate_video_verdict(landings, tracks, source_frames, timestamps, analysis_size,
                           calibration, rules=None, output_dir=None, config=None):
    """Resolve a unique measured flight, native envelope, then bounded geometry."""
    from .landing_families import resolve_landing_families
    from .native_landing import estimate_native_landing
    try:
        region = rule_region(rules)
        selected = {'game': region['game'], 'phase': region['phase']}
    except (TypeError, ValueError):
        return _result(rules, ['VIDEO_ESTIMATE_RULES_UNSUPPORTED'])
    if (config or {}).get('enabled', True) is False:
        return _result(selected, ['VIDEO_ESTIMATE_DISABLED'])
    resolution = resolve_landing_families(landings, tracks, timestamps, calibration, analysis_size)
    family = resolution.get('selected_family')
    # Preserve the reasoning, without duplicating every candidate/track in JSON.
    compact = deepcopy(resolution)
    if compact.get('selected_family'):
        compact['selected_family'].pop('members', None)
    for row in compact.get('families', []):
        row.pop('members', None)
    base = _result(selected, ['VIDEO_ESTIMATE_NO_UNIQUE_EVENT'], family_resolution=compact)
    if resolution.get('status') != 'selected' or not isinstance(family, dict):
        return base
    native = estimate_native_landing(family, tracks, source_frames, timestamps, analysis_size,
                                    calibration, output_dir=output_dir, config=config)
    base.update(selected_family_id=family['family_id'], member_event_ids=family['member_event_ids'],
                native_evidence=native)
    if native.get('status') != 'supported':
        return {**base, 'reason_codes': ['VIDEO_ESTIMATE_NATIVE_SUPPORT_MISSING'],
                'reasons': [REASONS['VIDEO_ESTIMATE_NATIVE_SUPPORT_MISSING']]}
    polygon, uncertainty = native.get('possible_contact_polygon_px'), native.get('uncertainty_px')
    geometry = evaluate_possible_region(polygon, uncertainty, calibration, selected)
    codes = geometry['reason_codes']
    base.update(geometry=geometry, possible_contact_polygon_px=polygon, uncertainty_px=uncertainty,
                frame_interval=native.get('frame_interval'), time_interval=native.get('time_interval'),
                artifact_path=native.get('artifact_path', native.get('artifact')),
                decision=geometry['decision'], eligible=geometry['status'] == 'resolved',
                reason_codes=codes, reasons=[REASONS.get(code, code) for code in codes])
    return base


def write_estimate_overlay(frames, calibration, estimate, output_dir):
    """Pair the actual scene/envelope with its native temporal pixel evidence."""
    from pathlib import Path
    from .evidence import draw_court_lines
    if estimate.get('eligible') is not True or not frames:
        return None
    output = Path(output_dir).resolve()
    interval = estimate['frame_interval']
    index = max(0, min(len(frames)-1, int(round(sum(interval)/2))))
    canvas = frames[index].copy()
    draw_court_lines(canvas, calibration)
    polygon = np.rint(estimate['possible_contact_polygon_px']).astype(np.int32)
    color = (70, 230, 110) if estimate['decision'] == 'IN' else (70, 110, 255)
    cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
    x, y = polygon.mean(axis=0).astype(int)
    cv2.putText(canvas, estimate['decision']+' (estimate)', (max(0, x-60), max(48, y-20)),
                cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (20, 28, 34), -1)
    game = (estimate.get('rules') or {}).get('game', 'doubles').upper()
    cv2.putText(canvas, f"VIDEO ESTIMATE: {estimate['decision']} | {game} | frames {interval[0]}-{interval[1]}",
                (12, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Native possible-position envelope; first physical contact is not directly verified',
                (12, 40), cv2.FONT_HERSHEY_SIMPLEX, .39, (220, 225, 230), 1, cv2.LINE_AA)
    native = (estimate.get('native_evidence') or {}).get('artifact_path')
    native = native or (estimate.get('native_evidence') or {}).get('artifact')
    if isinstance(native, str):
        path = (output / native).resolve()
        if path.is_relative_to(output) and path.is_file():
            panels = cv2.imread(str(path))
            if panels is not None:
                width = max(canvas.shape[1], min(1600, panels.shape[1]))
                scene = cv2.resize(canvas, (width, round(canvas.shape[0]*width/canvas.shape[1])))
                panels = cv2.resize(panels, (width, round(panels.shape[0]*width/panels.shape[1])))
                canvas = np.vstack([scene, panels])
    path = output / 'evidence/video-estimate.jpg'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError('Could not write video estimate evidence')
    return path.relative_to(output).as_posix()
