"""Competing measured trajectory evidence for unverified event-review priority.

An alternative curve is never proof of shuttle identity or floor contact. The
combined evidence below only withholds a flight preference from an association
branch that changes appearance while another measured branch continues its
shared incoming path.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
from statistics import median

import numpy as np


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _observations(track):
    rows = [row for row in track.get('observations', [])
            if row.get('observed') is not False
            and all(_number(row.get(key)) for key in ('frame_index', 'time', 'x', 'y'))]
    rows.sort(key=lambda row: (row['time'], row['frame_index']))
    return list({row['time']: row for row in rows}.values())


def _signature(row):
    return (row['frame_index'], round(row['x'], 3), round(row['y'], 3))


def _appearance(rows):
    radii = [row['radius'] for row in rows if _number(row.get('radius')) and row['radius'] > 0]
    polarities = Counter(row.get('appearance_polarity') for row in rows
                         if row.get('appearance_polarity') in {'bright', 'dark'})
    known = sum(polarities.values())
    dominant = max(polarities, key=polarities.get) if known else None
    return {'radius_count': len(radii), 'median_radius_px': median(radii) if radii else None,
            'known_polarity_count': known, 'dominant_polarity': dominant,
            'dominant_polarity_fraction': polarities[dominant] / known if known else None}


def _suffix(rows, time, nominal_step, count=None):
    result, previous = [], time
    for row in rows:
        if row['time'] <= time:
            continue
        if row['time'] - time > .16 or row['time'] - previous > nominal_step * 2.5:
            break
        result.append(row)
        previous = row['time']
        if count is not None and len(result) == count:
            break
    return result


def apply_competing_continuation_evidence(candidates, tracks, nominal_step):
    """Return candidates with a bounded, appearance-aware branch priority veto.

    All cues are required: at least five shared actual prefix observations over
    .12s; a sustained extreme radius collapse and stable opposite polarity on
    the proposed rebound; and three actual, appearance-consistent descending
    observations on another branch that fit the incoming quadratic. Unknown
    appearance and interpolated observations cannot supply these cues. Neither
    branch is promoted to verified identity or contact.
    """
    if not _number(nominal_step) or nominal_step <= 0:
        return list(candidates)
    by_id = {str(track['track_id']): _observations(track) for track in tracks
             if 'track_id' in track and isinstance(track.get('flight_hypothesis'), dict)}
    owners = defaultdict(set)
    for identifier, rows in by_id.items():
        for row in rows:
            owners[_signature(row)].add(identifier)
    result = list(candidates)
    for candidate_index, candidate in enumerate(candidates):
        support = candidate.get('flight_trajectory_support') or {}
        motion = candidate.get('image_motion_event') or {}
        if (support.get('supported_for_review') is not True
                or motion.get('kind') != 'descent_and_rebound'
                or candidate.get('contact_observed') is True or not _number(candidate.get('time'))):
            continue
        identifier, time = str(candidate.get('track_id')), candidate['time']
        rows = by_id.get(identifier, [])
        before = [row for row in rows if time - .35 <= row['time'] < time]
        post = _suffix(rows, time, nominal_step, count=3)
        if len(before) < 5 or len(post) < 3 or before[-1]['time'] - before[0]['time'] < .12:
            continue
        before_appearance, outgoing = _appearance(before), _appearance(post)
        baseline = before_appearance['median_radius_px']
        if baseline is None:
            continue
        # A robust size-consistent reference prevents tiny, differently
        # polarized fragments already mixed into the approach from defining
        # its appearance. The curve fit below still uses every actual point.
        resolved_before = [row for row in before if _number(row.get('radius'))
                           and .6 <= row['radius'] / baseline <= 1.8]
        incoming = _appearance(resolved_before)
        if (incoming['radius_count'] < 5 or outgoing['radius_count'] < 3
                or incoming['known_polarity_count'] < 5 or outgoing['known_polarity_count'] < 3
                or incoming['dominant_polarity_fraction'] < .67 or outgoing['dominant_polarity_fraction'] < .8
                or incoming['dominant_polarity'] == outgoing['dominant_polarity']
                or outgoing['median_radius_px'] > baseline * .35):
            continue
        times = np.asarray([row['time'] - time for row in before])
        points = np.asarray([[row['x'], row['y']] for row in before])
        coefficients = np.polyfit(times, points, 2)

        def residual(items):
            sample_times = np.asarray([row['time'] - time for row in items])
            sample_points = np.asarray([[row['x'], row['y']] for row in items])
            fitted = np.column_stack([np.polyval(coefficients[:, axis], sample_times) for axis in range(2)])
            return float(np.sqrt(np.mean(np.sum((sample_points - fitted) ** 2, axis=1))))

        tolerance = max(3., baseline * 1.5)
        incoming_error = residual(before)
        if incoming_error > tolerance:
            continue
        alternatives = Counter(other for row in before for other in owners[_signature(row)] if other != identifier)
        ordered = sorted(alternatives.items(), key=lambda item: (-item[1], item[0]))[:32]
        for other, shared_count in ordered:
            if shared_count < 5:
                break
            other_rows = by_id[other]
            other_signatures = {_signature(row) for row in other_rows}
            shared = [row for row in before if _signature(row) in other_signatures]
            if (shared[-1]['time'] - shared[0]['time'] < .12
                    or time - shared[-1]['time'] > nominal_step * 2.5):
                continue
            continuation = _suffix(other_rows, time, nominal_step, count=3)
            if len(continuation) < 3:
                continue
            appearance = _appearance(continuation)
            if (appearance['radius_count'] < 3 or appearance['known_polarity_count'] < 3
                    or appearance['dominant_polarity'] != incoming['dominant_polarity']
                    or appearance['dominant_polarity_fraction'] < .67
                    or not .6 <= appearance['median_radius_px'] / baseline <= 1.8):
                continue
            continued_error = residual(continuation)
            if (continued_error > tolerance
                    or any(right['y'] <= left['y'] for left, right in zip(continuation, continuation[1:]))):
                continue
            updated = deepcopy(candidate)
            reason = 'APPEARANCE_BREAK_WITH_COMPETING_FLIGHT_CONTINUATION'
            updated['flight_trajectory_support']['supported_for_review'] = False
            updated['flight_trajectory_support']['reason_codes'] = list(dict.fromkeys([
                *updated['flight_trajectory_support'].get('reason_codes', []), reason]))
            updated['reason_codes'] = list(dict.fromkeys([*updated.get('reason_codes', []), reason]))
            updated['competing_continuation_evidence'] = {
                'method': 'shared_observed_prefix_and_appearance_consistent_continuation',
                'status': 'association_branch_ambiguous', 'flight_support_veto': True,
                'other_track_id': other, 'shared_observation_count': len(shared),
                'shared_frame_interval': [shared[0]['frame_index'], shared[-1]['frame_index']],
                'incoming_fit_residual_px': round(incoming_error, 4),
                'continuation_fit_residual_px': round(continued_error, 4),
                'continuation_frame_indices': [row['frame_index'] for row in continuation],
                'incoming_appearance': incoming, 'rebound_appearance': outgoing,
                'incoming_appearance_sample_rule': 'median_size_consistent_observations',
                'continuation_appearance': appearance,
                'identity_confirmed': False, 'physical_contact_confirmed': False}
            result[candidate_index] = updated
            break
    return result
