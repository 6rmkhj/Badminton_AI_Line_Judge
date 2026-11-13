"""Measured-path event families and causal ordering for review candidates.

These operations allocate a finite review budget. They do not establish which
object is the shuttle, that a turn touched the floor, or a legal line verdict.
"""
from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
import math

import numpy as np

from .event_continuity import _rows, _signature


def _finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _eligible(candidate):
    point = candidate.get('candidate_px')
    return (candidate.get('contact_observed') is False
            and candidate.get('identity_source') != 'operator_selected_track'
            and candidate.get('request_match') not in {'selected', 'ambiguous'}
            and not candidate.get('assistance_used')
            and isinstance(point, (list, tuple)) and len(point) == 2
            and all(_finite(value) for value in [*point, candidate.get('time'), candidate.get('frame_index')])
            and (candidate.get('image_motion_event') or {}).get('kind')
                in {'descent_and_rebound', 'occluded_descent_and_rebound'})


def _key(candidate):
    return (str(candidate.get('track_id')), candidate.get('frame_index'),
            tuple(candidate.get('candidate_px') or []))


def _shared_path(left, right, by_id, before, after):
    first = by_id.get(str(left.get('track_id')), [])
    second = by_id.get(str(right.get('track_id')), [])
    signatures = {_signature(row) for row in first}
    shared = [row for row in second if _signature(row) in signatures]
    return ([row for row in shared if before[0] <= row['time'] <= before[1]],
            [row for row in shared if after[0] <= row['time'] <= after[1]])


def diversify_event_families(candidates, tracks, nominal_step):
    """Keep one representative per duplicate event, preserving its alternatives.

    The input is already ranked. Every member must match the representative
    directly; transitive position chains cannot merge distant events. A spatial
    coincidence is insufficient: at least three identical measured incoming
    points and two outgoing points must be shared by both proposed paths.
    """
    if not _finite(nominal_step) or nominal_step <= 0:
        return list(candidates)
    by_id = {str(track.get('track_id')): _rows(track) for track in tracks}
    representatives = []
    by_frame = defaultdict(list)
    for candidate in candidates:
        if not _eligible(candidate):
            representatives.append(candidate)
            continue
        matched = False
        frame_bucket = math.floor(candidate['frame_index'])
        # Only neighboring frames can denote the same event. Preserve global
        # rank within this small bucket instead of scanning every earlier event.
        near_indices = sorted(index for frame in range(frame_bucket - 1, frame_bucket + 2)
                              for index in by_frame[frame])
        for index in near_indices:
            representative = representatives[index]
            # An observed peak and a gap-intersection estimate are different
            # evidence types even when their projected coordinates coincide.
            if ((candidate.get('image_motion_event') or {}).get('point_kind')
                    != (representative.get('image_motion_event') or {}).get('point_kind')):
                continue
            if (abs(candidate['time'] - representative['time']) > min(.05, nominal_step * 1.01)
                    or abs(candidate['frame_index'] - representative['frame_index']) > 1):
                continue
            radii = [value for value in [candidate.get('contact_radius_px'),
                                        representative.get('contact_radius_px')]
                     if _finite(value) and value > 0]
            tolerance = max(1.5, min(4., min(radii) if radii else 1.5))
            if math.dist(candidate['candidate_px'], representative['candidate_px']) > tolerance:
                continue
            start, end = sorted([candidate['time'], representative['time']])
            incoming, outgoing = _shared_path(candidate, representative, by_id,
                (start - .25, start - nominal_step * .25),
                (end + nominal_step * .25, end + .2))
            if len(incoming) < 3 or len(outgoing) < 2:
                continue
            # No representative contact, score or coordinate changes. Store
            # alternatives in full so inspection can recover the original set.
            updated = deepcopy(representative)
            family = updated.setdefault('event_family', {
                'method': 'coincident_event_with_shared_measured_prefix_and_suffix',
                'representative_track_id': representative.get('track_id'),
                'representative_frame_index': representative['frame_index'],
                'alternatives': [], 'identity_verified': False,
                'physical_contact_verified': False, 'ranking_only': True})
            alternative = deepcopy(candidate)
            alternative.pop('event_family', None)
            family['alternatives'].append({'candidate': alternative,
                'shared_incoming_count': len(incoming), 'shared_outgoing_count': len(outgoing)})
            family['member_count'] = 1 + len(family['alternatives'])
            representatives[index] = updated
            matched = True
            break
        if not matched:
            by_frame[frame_bucket].append(len(representatives))
            representatives.append(candidate)
    return representatives


def _flight(candidate):
    return (_eligible(candidate)
            and (candidate.get('flight_trajectory_support') or {}).get('supported_for_review') is True)


def annotate_causal_rebounds(candidates, tracks, nominal_step):
    """Link two locally supported turns only through a measured diminishing arc.

    Independent tracks often contain the same outgoing and incoming bounce arc.
    Counting each track's first turn independently loses that chronology. This
    relation is deliberately narrower than an earliest-event preference: shared
    actual samples must form a complete, coherent arc returning to a similar
    image height, with less subsequent rise. The image evidence remains unable
    to distinguish every low racket hit from a floor rebound.
    """
    if not _finite(nominal_step) or nominal_step <= 0:
        return list(candidates)
    by_id = {str(track.get('track_id')): _rows(track) for track in tracks}
    signatures_by_id = {identifier: {_signature(row) for row in rows}
                        for identifier, rows in by_id.items()}
    output = list(candidates)
    eligible = [(i, c) for i, c in enumerate(candidates) if _flight(c)]
    for late_index, later in eligible:
        for _, earlier in sorted(eligible, key=lambda item: item[1]['time']):
            duration = later['time'] - earlier['time']
            if not max(.12, nominal_step * 4) <= duration <= 1.2:
                continue
            early_rows = by_id.get(str(earlier.get('track_id')), [])
            late_rows = by_id.get(str(later.get('track_id')), [])
            if not early_rows or not late_rows:
                continue
            early_signatures = {_signature(row) for row in early_rows}
            shared = [row for row in late_rows if earlier['time'] < row['time'] < later['time']
                      and _signature(row) in early_signatures]
            bridge_rows, bridge_id = [], None
            if len(shared) < 4 or shared[-1]['time'] - shared[0]['time'] < .08:
                # A normal tracker can bridge two short flight hypotheses.
                # Both joins need repeated identical measured observations;
                # merely extrapolating between the endpoints is insufficient.
                early_arc = {_signature(row) for row in early_rows if earlier['time'] < row['time'] < later['time']}
                late_arc = {_signature(row) for row in late_rows if earlier['time'] < row['time'] < later['time']}
                bridges = []
                for identifier, signatures in signatures_by_id.items():
                    left_count, right_count = len(signatures & early_arc), len(signatures & late_arc)
                    if left_count >= 3 and right_count >= 3:
                        bridges.append((min(left_count, right_count), left_count + right_count, identifier))
                if not bridges:
                    continue
                _, _, bridge_id = max(bridges)
                bridge_rows = by_id[bridge_id]
                shared = [row for row in bridge_rows if _signature(row) in (early_arc | late_arc)]
            rows_by_frame = {}
            conflict = False
            for row in early_rows + bridge_rows + late_rows:
                if not earlier['time'] < row['time'] < later['time']:
                    continue
                previous = rows_by_frame.get(row['frame_index'])
                if previous is not None and _signature(previous) != _signature(row):
                    conflict = True
                    break
                rows_by_frame[row['frame_index']] = row
            arc = sorted(rows_by_frame.values(), key=lambda row: row['time'])
            if conflict or len(arc) < 7:
                continue
            if (arc[0]['time'] - earlier['time'] > max(.12, duration * .3)
                    or later['time'] - arc[-1]['time'] > nominal_step * 2.5
                    or any(b['time'] - a['time'] > nominal_step * 2.5 for a, b in zip(arc, arc[1:]))):
                continue
            radius = float(np.median([row['radius'] for row in arc]))
            if abs(earlier['candidate_px'][1] - later['candidate_px'][1]) > max(12., radius * 4):
                continue
            times = np.asarray([row['time'] - earlier['time'] for row in arc], dtype=float)
            points = np.asarray([[row['x'], row['y']] for row in arc], dtype=float)
            if np.unique(times).size < 5 or not np.isfinite(points).all():
                continue
            try:
                coefficients, _, rank, _, _ = np.polyfit(times, points, 2, full=True)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if rank < 3 or not np.isfinite(coefficients).all() or coefficients[0, 1] <= 0:
                continue
            fitted = np.column_stack([np.polyval(coefficients[:, axis], times) for axis in range(2)])
            residual = float(np.sqrt(np.mean(np.sum((points - fitted) ** 2, axis=1))))
            apex_time = -coefficients[1, 1] / (2 * coefficients[0, 1])
            if not times[1] < apex_time < times[-2] or residual > max(3., radius * 1.5):
                continue
            endpoints = [np.asarray([np.polyval(coefficients[:, axis], t) for axis in range(2)])
                         for t in [0., duration]]
            errors = [math.dist(endpoint, candidate['candidate_px'])
                      for endpoint, candidate in zip(endpoints, [earlier, later])]
            if max(errors) > max(6., radius * 3):
                continue
            apex_y = float(np.polyval(coefficients[:, 1], apex_time))
            height = min(earlier['candidate_px'][1], later['candidate_px'][1]) - apex_y
            if height < max(6., radius * 3):
                continue
            outgoing = [row for row in late_rows if later['time'] < row['time'] <= later['time'] + .2]
            if (len(outgoing) < 3 or outgoing[0]['time'] - later['time'] > nominal_step * 2.5
                    or any(b['time'] - a['time'] > nominal_step * 2.5
                                       for a, b in zip(outgoing, outgoing[1:]))):
                continue
            subsequent_rise = later['candidate_px'][1] - min(row['y'] for row in outgoing)
            if not max(1.5, radius * .5) <= subsequent_rise <= height * .8:
                continue
            # A short outgoing fragment can have a small observed rise even
            # after an energetic hit. Compare fitted initial image velocities,
            # using timestamps and measured samples, before calling it decay.
            post_context = [row for row in late_rows if later['time'] <= row['time'] <= later['time'] + .2]
            post_times = np.asarray([row['time'] - later['time'] for row in post_context])
            post_points = np.asarray([[row['x'], row['y']] for row in post_context])
            if np.unique(post_times).size < 3:
                continue
            try:
                post_coefficients, _, post_rank, _, _ = np.polyfit(post_times, post_points, 2, full=True)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if post_rank < 3 or not np.isfinite(post_coefficients).all():
                continue
            post_fit = np.column_stack([np.polyval(post_coefficients[:, axis], post_times) for axis in range(2)])
            post_error = float(np.sqrt(np.mean(np.sum((post_points - post_fit) ** 2, axis=1))))
            outgoing_ratio = post_coefficients[1, 1] / coefficients[1, 1]
            if (post_error > max(3., radius * 1.5) or post_coefficients[0, 1] < -1e-6
                    or not 0 < outgoing_ratio <= .85):
                continue
            # The incoming first flight must be appreciably larger than this
            # bounce arc, reducing promotion of a fresh energetic racket hit.
            incoming = earlier.get('flight_trajectory_support') or {}
            if incoming.get('approach_displacement_y_px', 0) < height * 1.25:
                continue
            updated = deepcopy(later)
            updated['causal_event_sequence'] = {
                'method': 'shared_measured_diminishing_rebound_arc',
                'preceding_candidate': list(_key(earlier)[:2]) + [list(_key(earlier)[2])],
                'shared_observation_count': len(shared), 'arc_observation_count': len(arc),
                'bridge_track_id': bridge_id,
                'arc_frame_interval': [arc[0]['frame_index'], arc[-1]['frame_index']],
                'arc_fit_residual_px': round(residual, 4), 'arc_height_px': round(height, 4),
                'subsequent_observed_rise_px': round(subsequent_rise, 4),
                'outgoing_image_speed_ratio': round(float(outgoing_ratio), 4),
                'endpoint_fit_errors_px': [round(error, 4) for error in errors],
                'identity_verified': False, 'physical_contact_verified': False,
                'low_racket_hit_excluded': False, 'ranking_only': True}
            output[late_index] = updated
            break
    return output


def order_causal_rebounds(candidates):
    """Place a supported predecessor directly before its related later turn.

    The existing global ranking still chooses which trajectory family receives
    priority. Penalizing a later bounce alone would instead promote unrelated
    distant motion. Explicit human/model evidence and unresolved requests stay
    protected, and a missing predecessor has no effect.
    """
    def members(candidate):
        yield candidate
        for alternative in (candidate.get('event_family') or {}).get('alternatives', []):
            member = alternative.get('candidate')
            if isinstance(member, dict):
                yield member

    by_key = {}
    for candidate in candidates:
        if not _flight(candidate):
            continue
        # A repeated prioritization after the evidence cap must still resolve
        # a preceding event which was folded into another family representative.
        for member in members(candidate):
            if _flight(member):
                by_key[_key(member)] = candidate
    result, visited, visiting = [], set(), set()

    def emit(candidate):
        identifier = id(candidate)
        if identifier in visited or identifier in visiting:
            return
        visiting.add(identifier)
        predecessors, sources, unresolved = {}, [], []
        if _flight(candidate):
            for member in members(candidate):
                sequence = member.get('causal_event_sequence') or {}
                predecessor = sequence.get('preceding_candidate')
                if (not _flight(member)
                        or sequence.get('method') != 'shared_measured_diminishing_rebound_arc'
                        or not isinstance(predecessor, list) or len(predecessor) != 3
                        or not isinstance(predecessor[2], list)):
                    continue
                key = (str(predecessor[0]), predecessor[1], tuple(predecessor[2]))
                before = by_key.get(key)
                if before is None:
                    unresolved.append(predecessor)
                elif before['time'] < candidate['time']:
                    predecessors[id(before)] = before
                    sources.append({'track_id': member.get('track_id'), 'frame_index': member['frame_index']})
        output_candidate = candidate
        if len(predecessors) == 1 and not unresolved:
            before = next(iter(predecessors.values()))
            emit(before)
            if candidate.get('event_family'):
                output_candidate = deepcopy(candidate)
                output_candidate['causal_event_family_order'] = {
                    'status': 'unique_preceding_family', 'relation_source_members': sources,
                    'preceding_candidate': [str(before.get('track_id')), before['frame_index'], before['candidate_px']],
                    'ranking_only': True, 'identity_verified': False, 'physical_contact_verified': False}
        elif candidate.get('event_family') and (len(predecessors) > 1 or unresolved):
            output_candidate = deepcopy(candidate)
            output_candidate['causal_event_family_order'] = {
                'status': 'ambiguous_or_missing_preceding_family',
                'resolved_predecessor_count': len(predecessors), 'unresolved_candidates': unresolved,
                'ranking_only': True, 'identity_verified': False, 'physical_contact_verified': False}
        visiting.remove(identifier)
        visited.add(identifier)
        result.append(output_candidate)

    for candidate in candidates:
        emit(candidate)
    return result
