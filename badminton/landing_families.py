"""Resolve supported image-flight event families on the selected court.

This is an association step for video estimates. It does not establish shuttle
identity or physical floor contact, and never alters the strict landing records.
Competing paths and the union of their possible locations remain explicit.
"""
from __future__ import annotations

from copy import deepcopy
import math

import cv2
import numpy as np


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(map(_finite, value))


def _polygon(value):
    if not isinstance(value, (list, tuple)) or len(value) < 3 or not all(map(_point, value)):
        return None
    polygon = np.asarray(value, dtype=np.float32)
    return polygon if abs(cv2.contourArea(polygon)) > 1 and cv2.isContourConvex(polygon) else None


def _rows(track):
    rows = track.get('observations')
    if not isinstance(rows, (list, tuple)):
        return []
    valid = [row for row in rows if isinstance(row, dict)
             and row.get('observed') is not False and row.get('interpolated') is not True
             and all(_finite(row.get(key)) for key in ('frame_index', 'time', 'x', 'y'))]
    ordered = sorted(valid, key=lambda row: (row['time'], row['frame_index']))
    return list({row['time']: row for row in ordered}.values())


def _signature(row):
    return (row['frame_index'], round(row['x'], 3), round(row['y'], 3))


def _interval(candidate, name='time'):
    key = 'time_interval' if name == 'time' else 'frame_interval'
    value = (candidate.get('image_motion_event') or {}).get(key)
    if (_point(value) and value[0] <= value[1]
            and value[0] <= candidate['time' if name == 'time' else 'frame_index'] <= value[1]):
        return list(value)
    value = candidate['time' if name == 'time' else 'frame_index']
    return [value, value]


def _extent(candidate):
    return max(1., candidate.get('uncertainty_px', 1.)) + max(0., candidate.get('contact_radius_px', 0.))


def _common_prefix(first, second, by_id):
    cutoff = min(first['time'], second['time'])
    signatures = {_signature(row) for row in by_id.get(str(first['track_id']), []) if row['time'] < cutoff}
    return [row for row in by_id.get(str(second['track_id']), [])
            if cutoff - .85 <= row['time'] < cutoff and _signature(row) in signatures]


def _supported_prefix(common):
    return len(common) >= 5 and common[-1]['time'] - common[0]['time'] >= .12


def _family_match(first, second, by_id, step):
    a, b = _interval(first), _interval(second)
    if max(a[0], b[0]) > min(a[1], b[1]) + step * .5:
        return None
    if math.dist(first['candidate_px'], second['candidate_px']) > _extent(first) + _extent(second):
        return None
    common = _common_prefix(first, second, by_id)
    if not _supported_prefix(common):
        return None
    return {'event_ids': [first['event_id'], second['event_id']],
            'shared_observation_count': len(common),
            'shared_frame_interval': [common[0]['frame_index'], common[-1]['frame_index']],
            'shared_span_seconds': round(common[-1]['time'] - common[0]['time'], 6)}


def _contradicted_gap_turn(candidate, candidates, by_id, step):
    motion = candidate.get('image_motion_event') or {}
    if motion.get('point_kind') != 'inferred_object_center':
        return None
    start, end = _interval(candidate)
    time = motion.get('estimated_time', candidate['time'])
    if not _finite(time) or not start <= time <= end:
        return None
    for other in candidates:
        if (other is candidate
                or (other.get('image_motion_event') or {}).get('point_kind') != 'observed_object_center'
                or (other.get('flight_trajectory_support') or {}).get('supported_for_review') is not True):
            continue
        common = _common_prefix(candidate, other, by_id)
        if not _supported_prefix(common):
            continue
        continuation = [row for row in by_id.get(str(other['track_id']), []) if start <= row['time'] <= end]
        if (sum(row['time'] < time for row in continuation) < 2
                or sum(row['time'] > time for row in continuation) < 2
                or any(b['time'] - a['time'] > step * 2.5 for a, b in zip(continuation, continuation[1:]))
                or not all(b['y'] > a['y'] for a, b in zip(continuation, continuation[1:]))):
            continue
        times = np.asarray([row['time'] - time for row in continuation])
        points = np.asarray([[row['x'], row['y']] for row in continuation])
        coefficients, _, rank, _, _ = np.polyfit(times, points, 2, full=True)
        if rank < 3 or not np.isfinite(coefficients).all():
            continue
        fitted = np.column_stack([np.polyval(coefficients[:, axis], times) for axis in (0, 1)])
        residual = float(np.sqrt(np.mean(np.sum((points - fitted) ** 2, axis=1))))
        radii = [row['radius'] for row in continuation if _finite(row.get('radius')) and row['radius'] > 0]
        tolerance = max(2.5, 2 * float(np.median(radii))) if radii else 2.5
        separation = math.dist(coefficients[-1], candidate['candidate_px'])
        if residual > tolerance or separation <= _extent(candidate) + tolerance:
            continue
        return {'reason_codes': ['GAP_TURN_CONTRADICTED_BY_SHARED_MEASURED_DESCENT'],
                'continuation_event_id': other['event_id'], 'shared_prefix_count': len(common),
                'continuation_frame_indices': [row['frame_index'] for row in continuation],
                'fit_residual_px': round(residual, 4), 'position_separation_px': round(separation, 4)}
    return None


def _court_connection(candidate, observations, physical_polygon, geometry_error, step):
    # A point near a court is insufficient. Require several actual observations
    # of this incoming path inside the observed physical court (including the
    # singles side alleys). This is image association, not certified identity.
    connected = [row for row in observations if candidate['time'] - .45 <= row['time'] <= candidate['time']
                 and cv2.pointPolygonTest(physical_polygon, (float(row['x']), float(row['y'])), True) > geometry_error]
    if len(connected) < 3 or connected[-1]['time'] - connected[0]['time'] < max(.06, step * 1.9):
        return None
    if any(b['time'] - a['time'] > step * 2.5 for a, b in zip(connected, connected[1:])):
        return None
    return {'method': 'repeated_actual_path_inside_observed_primary_court',
            'frame_indices': [row['frame_index'] for row in connected],
            'span_seconds': round(connected[-1]['time'] - connected[0]['time'], 6),
            'physical_contact_confirmed': False, 'target_identity_confirmed': False}


def resolve_landing_families(landings, tracks, timestamps, calibration, analysis_size):
    """Select one supported flight family, retaining all source hypotheses.

    Different observed flights remain ambiguous; a best score is never used to
    choose between them. Observed and inferred alternatives may share a family
    only with a repeated measured prefix and overlapping time/location bounds.
    """
    result = {'status': 'review', 'selected_family': None, 'families': [], 'excluded': [],
              'ambiguous': [], 'reason_codes': [], 'method': 'observed_primary_court_flight_families',
              'physical_contact_confirmed': False, 'target_identity_confirmed': False}
    primary = calibration.get('primary_court') if isinstance(calibration, dict) else None
    if not isinstance(primary, dict):
        primary = {}
    physical = _polygon(primary.get('physical_court_polygon_px') or primary.get('interior_polygon_px'))
    interior = _polygon(primary.get('interior_polygon_px'))
    scope = _polygon(primary.get('judgment_scope_polygon_px'))
    valid_times = (isinstance(timestamps, (list, tuple, np.ndarray)) and len(timestamps) > 1
                   and all(_finite(t) for t in timestamps) and all(b > a for a, b in zip(timestamps, timestamps[1:])))
    if (primary.get('status') != 'selected' or primary.get('region_supported') is not True
            or physical is None or interior is None or not valid_times
            or not _point(analysis_size) or min(analysis_size) <= 0):
        result['reason_codes'] = ['PRIMARY_COURT_EVENT_SCOPE_UNAVAILABLE']
        return result
    step = float(np.median(np.diff(timestamps)))
    error = primary.get('geometry_uncertainty_px', 2.5)
    if not _finite(error) or error < 0:
        result['reason_codes'] = ['PRIMARY_COURT_EVENT_SCOPE_UNAVAILABLE']
        return result
    margin = max(12., 24. * analysis_size[0] / 960.) if scope is None else 0.
    scope = physical if scope is None else scope
    normal, origin = None, None
    if primary.get('scope') == 'near_half':
        if len(interior) != 4 or primary.get('interior_limit') != 'near_short_service_line_not_an_out_boundary':
            result['reason_codes'] = ['PRIMARY_COURT_EVENT_SCOPE_UNAVAILABLE']
            return result
        origin = interior[2].astype(float)
        edge = interior[3] - interior[2]
        if np.linalg.norm(edge) <= 1:
            result['reason_codes'] = ['PRIMARY_COURT_EVENT_SCOPE_UNAVAILABLE']
            return result
        normal = np.array([-edge[1], edge[0]], float) / np.linalg.norm(edge)
        normal *= 1 if np.dot(interior.mean(axis=0) - origin, normal) > 0 else -1
    if not isinstance(landings, (list, tuple)) or not isinstance(tracks, (list, tuple)):
        result['reason_codes'] = ['INVALID_EVENT_FAMILY_INPUT']
        return result
    by_id = {str(track.get('track_id')): _rows(track) for track in tracks if isinstance(track, dict)}
    candidates = []
    for candidate in landings:
        if (not isinstance(candidate, dict) or not _point(candidate.get('candidate_px'))
                or not isinstance(candidate.get('track_id'), (str, int))
                or any(not isinstance(candidate.get(key, {}), dict)
                       for key in ('image_motion_event', 'flight_trajectory_support'))
                or not all(_finite(candidate.get(key)) for key in ('frame_index', 'time'))
                or any(not _finite(candidate.get(key, default)) or candidate.get(key, default) < 0
                       for key, default in (('uncertainty_px', 1.), ('contact_radius_px', 0.)))
                or not timestamps[0] <= candidate.get('time', -1) <= timestamps[-1]):
            result['ambiguous'].append({'event_id': candidate.get('event_id') if isinstance(candidate, dict) else None,
                                        'reason_codes': ['INVALID_EVENT_FAMILY_INPUT']})
            continue
        item = deepcopy(candidate)
        item.setdefault('event_id', f'input_{len(candidates):03}')
        candidates.append(item)
    candidates.sort(key=lambda item: (item['time'], str(item['event_id']), str(item.get('track_id'))))
    retained, unassigned = [], []
    for candidate in candidates:
        identifier = candidate['event_id']
        point = np.asarray(candidate['candidate_px'], float)
        spread = _extent(candidate)
        distance = cv2.pointPolygonTest(scope, tuple(point), True)
        depth = float(np.dot(point - origin, normal)) if normal is not None else None
        facts = {'event_id': identifier, 'scope_distance_px': round(distance, 4)}
        if depth is not None:
            facts['observed_depth_distance_px'] = round(depth, 4)
        if distance + margin < -spread or (depth is not None and depth < -spread - error):
            result['excluded'].append({**facts, 'reason_codes': ['OUTSIDE_OBSERVED_PRIMARY_COURT_REGION']})
            continue
        contradiction = _contradicted_gap_turn(candidate, candidates, by_id, step)
        if contradiction:
            result['excluded'].append({**facts, **contradiction})
            continue
        observations = by_id.get(str(candidate.get('track_id')), [])
        connection = _court_connection(candidate, observations, physical, error, step)
        if connection is None:
            unassigned.append((candidate, facts))
            continue
        if depth is not None and depth < spread + error:
            result['ambiguous'].append({**facts, 'reason_codes': ['UNCERTAINTY_OVERLAPS_UNOBSERVED_COURT_DEPTH']})
            continue
        before = [row for row in observations if candidate['time'] - .45 <= row['time'] <= candidate['time']]
        if ((candidate.get('flight_trajectory_support') or {}).get('supported_for_review') is not True
                or not _supported_prefix(before)):
            result['ambiguous'].append({**facts, 'reason_codes': ['COHERENT_MEASURED_FLIGHT_UNSUPPORTED']})
            continue
        retained.append((candidate, connection))
    groups = []
    for candidate, connection in retained:
        matches = []
        for group in groups:
            agreements = [_family_match(candidate, member, by_id, step) for member in group['members']]
            if all(agreements):
                matches.append((group, agreements))
        if len(matches) == 1:
            group, agreements = matches[0]
            group['members'].append(candidate)
            group['connections'].append(connection)
            group['agreements'].extend(agreements)
        else:
            groups.append({'members': [candidate], 'connections': [connection], 'agreements': []})
    # A sparse alternative of the same measured incoming flight must remain in
    # its family's uncertainty envelope, even when its own local path has a gap.
    # This transfers association only, never identity/contact confirmation.
    for candidate, facts in unassigned:
        matches = []
        if (candidate.get('flight_trajectory_support') or {}).get('supported_for_review') is True:
            for group in groups:
                agreements = [_family_match(candidate, member, by_id, step) for member in group['members']]
                if all(agreements):
                    matches.append((group, agreements))
        if len(matches) == 1:
            group, agreements = matches[0]
            group['members'].append(candidate)
            group['agreements'].extend(agreements)
            group['connections'].append({'method': 'shared_measured_prefix_to_court_connected_family',
                                          'event_id': candidate['event_id'],
                                          'physical_contact_confirmed': False, 'target_identity_confirmed': False})
        elif len(matches) > 1:
            result['ambiguous'].append({**facts, 'reason_codes': ['EVENT_ASSIGNED_TO_MULTIPLE_FLIGHT_FAMILIES']})
        else:
            result['excluded'].append({**facts, 'reason_codes': ['PRIMARY_COURT_PATH_ASSOCIATION_UNSUPPORTED']})
    for index, group in enumerate(groups):
        members = group['members']
        observed = [item for item in members if (item.get('image_motion_event') or {}).get('point_kind') == 'observed_object_center']
        points = [item['candidate_px'] for item in members]
        xmin = min(point[0] - _extent(item) for point, item in zip(points, members))
        xmax = max(point[0] + _extent(item) for point, item in zip(points, members))
        ymin = min(point[1] - _extent(item) for point, item in zip(points, members))
        ymax = max(point[1] + _extent(item) for point, item in zip(points, members))
        family = {'family_id': f'family_{index:03}', 'members': members,
                  'representative': (observed or members)[0],
                  'member_event_ids': [item['event_id'] for item in members],
                  'track_ids': list(dict.fromkeys(item['track_id'] for item in members)),
                  'frame_interval': [min(_interval(item, 'frame')[0] for item in members),
                                     max(_interval(item, 'frame')[1] for item in members)],
                  'time_interval': [min(_interval(item)[0] for item in members), max(_interval(item)[1] for item in members)],
                  'possible_candidate_polygon_px': [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                  'selection_evidence': {'shared_measured_prefixes': group['agreements'],
                                         'primary_court_connections': group['connections'],
                                         'observed_peak_member_ids': [item['event_id'] for item in observed],
                                         'independent_evidence_count': 1,
                                         'hypotheses_are_not_independent_votes': True},
                  'physical_contact_confirmed': False, 'target_identity_confirmed': False}
        result['families'].append(family)
    if len(result['families']) == 1 and not result['ambiguous']:
        family = result['families'][0]
        if family['selection_evidence']['observed_peak_member_ids']:
            result.update(status='selected', selected_family=family,
                          reason_codes=['UNIQUE_PRIMARY_COURT_FLIGHT_FAMILY'])
        else:
            result['reason_codes'] = ['EVENT_FAMILY_ONLY_INFERRED_LOCATIONS']
    else:
        result['reason_codes'] = ['MULTIPLE_PRIMARY_COURT_FLIGHT_FAMILIES' if len(result['families']) > 1
                                  else 'PRIMARY_COURT_EVENT_UNRESOLVED' if result['ambiguous']
                                  else 'NO_SUPPORTED_PRIMARY_COURT_FLIGHT_FAMILY']
    return result
