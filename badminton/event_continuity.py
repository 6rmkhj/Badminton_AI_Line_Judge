"""Resolve adjacent review turns using competing actual observations.

This only withholds preference from a premature, fragment-based association.
It never moves a candidate or establishes shuttle identity or floor contact.
"""
from __future__ import annotations

from copy import deepcopy
import math
from statistics import median

import numpy as np

from .observation_support import rejects_moving_association


def _finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _rows(track):
    rows = [row for row in track.get('observations', []) if row.get('observed') is not False
            and row.get('interpolated') is not True
            and not rejects_moving_association(row)
            and all(_finite(row.get(key)) for key in ('frame_index', 'time', 'x', 'y', 'radius'))
            and row['radius'] > 0]
    return sorted({row['frame_index']: row for row in rows}.values(), key=lambda row: row['time'])


def _signature(row):
    return (row['frame_index'], round(row['x'], 3), round(row['y'], 3))


def _eligible(candidate):
    return ((candidate.get('flight_trajectory_support') or {}).get('supported_for_review') is True
            and (candidate.get('image_motion_event') or {}).get('kind') == 'descent_and_rebound'
            and (candidate.get('image_motion_event') or {}).get('point_kind') == 'observed_object_center'
            and candidate.get('contact_observed') is not True
            and _finite(candidate.get('time')))


def resolve_fragment_turns(candidates, tracks, nominal_step):
    """Keep an observed descent peak when an early branch skips it for a speck.

    Both branches must already qualify locally and share measured incoming and
    outgoing observations. A later, size-consistent peak must continue the
    incoming fit, while the earlier turn relies on a tiny conflicting outgoing
    fragment. Distinct flights and later bounces cannot supply these cues.
    """
    if not _finite(nominal_step) or nominal_step <= 0:
        return list(candidates)
    by_id = {str(track.get('track_id')): _rows(track) for track in tracks
             if isinstance(track.get('flight_hypothesis'), dict)}
    eligible = [(index, candidate) for index, candidate in enumerate(candidates) if _eligible(candidate)]
    result = list(candidates)
    for index, early in eligible:
        rows = by_id.get(str(early.get('track_id')), [])
        before = [row for row in rows if early['time'] - .25 <= row['time'] <= early['time']]
        if len(before) < 5 or before[-1]['time'] - before[0]['time'] < .08:
            continue
        baseline = median(row['radius'] for row in before)
        post = [row for row in rows if early['time'] < row['time'] <= early['time'] + .12]
        if not post or post[0]['time'] - early['time'] > nominal_step * 2.5:
            continue
        fragment = post[0]
        if fragment['radius'] > baseline * .5:
            continue
        early_signatures = {_signature(row) for row in rows}
        early_frames = {row['frame_index'] for row in rows}
        for _, later in eligible:
            delta = later['time'] - early['time']
            if not 0 < delta <= min(.08, nominal_step * 3.01):
                continue
            other_rows = by_id.get(str(later.get('track_id')), [])
            peaks = [row for row in other_rows if row['frame_index'] == later.get('frame_index')]
            if not peaks or peaks[0]['frame_index'] in early_frames:
                continue
            peak = peaks[0]
            if not .6 <= peak['radius'] / baseline <= 1.8:
                continue
            shared_before = [row for row in other_rows if early['time'] - .25 <= row['time'] <= early['time']
                             and _signature(row) in early_signatures]
            shared_after = [row for row in other_rows if later['time'] < row['time'] <= later['time'] + .16
                            and _signature(row) in early_signatures]
            if (len(shared_before) < 5 or len(shared_after) < 2
                    or shared_before[-1]['time'] - shared_before[0]['time'] < .08
                    or early['time'] - shared_before[-1]['time'] > nominal_step * 1.5):
                continue
            descent = [row for row in other_rows if early['time'] <= row['time'] <= later['time']]
            if (len(descent) < 2 or peak['y'] - descent[0]['y'] < max(.5, baseline * .25)
                    or any(b['y'] <= a['y'] or b['time'] - a['time'] > nominal_step * 2.5
                           for a, b in zip(descent, descent[1:]))):
                continue
            times = np.asarray([row['time'] - peak['time'] for row in shared_before], dtype=float)
            points = np.asarray([[row['x'], row['y']] for row in shared_before], dtype=float)
            if (not np.isfinite(times).all() or np.unique(times).size < 3
                    or not np.isfinite(points).all()):
                continue
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    coefficients, _, rank, _, _ = np.polyfit(times, points, 2, full=True)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if rank < 3 or not np.isfinite(coefficients).all():
                continue
            prediction = coefficients[-1]
            error = math.dist(prediction, (peak['x'], peak['y']))
            if not math.isfinite(error) or error > max(3., baseline * 1.5):
                continue
            updated = deepcopy(early)
            reason = 'EARLY_TURN_USES_FRAGMENT_INSTEAD_OF_SHARED_DESCENT_PEAK'
            support = updated['flight_trajectory_support']
            support['supported_for_review'] = False
            support['reason_codes'] = list(dict.fromkeys([*support.get('reason_codes', []), reason]))
            updated['reason_codes'] = list(dict.fromkeys([*updated.get('reason_codes', []), reason]))
            updated['competing_peak_evidence'] = {
                'method': 'shared_actual_prefix_suffix_and_resolved_descent_peak',
                'flight_support_veto': True, 'other_track_id': later.get('track_id'),
                'fragment_frame_index': fragment['frame_index'],
                'fragment_radius_ratio': round(fragment['radius'] / baseline, 4),
                'continued_peak_frame_index': peak['frame_index'],
                'continued_peak_px': [peak['x'], peak['y']],
                'continued_peak_radius_ratio': round(peak['radius'] / baseline, 4),
                'incoming_prediction_error_px': round(error, 4),
                'shared_before_count': len(shared_before), 'shared_after_count': len(shared_after),
                'identity_confirmed': False, 'physical_contact_confirmed': False,
                'candidate_coordinates_changed': False}
            result[index] = updated
            break
    return result
