"""Bounded native-pixel rebound recovery after measured descending paths.

The search windows come only from existing measured trajectories. A component
must differ from actual earlier and later source frames as well as its local
background. Recovered tracks remain ambiguous hypotheses, never contact proof.
"""
from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
import math
from numbers import Real

import cv2
import numpy as np

from .observation_support import rejects_moving_association, observation_support_status

MAX_CROP_BYTES = 128 * 1024 * 1024


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _source_frame(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
            or frame.shape[2] != 3 or not frame.size):
        raise ValueError('Recovery requires actual uint8 BGR source frames')
    return frame


def _approaches(tracks, budget):
    from .tracking import _flight_fit
    candidates = []
    for track in tracks:
        hypothesis = track.get('flight_hypothesis')
        if not isinstance(hypothesis, dict):
            continue
        count = hypothesis.get('approach_count', 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 6:
            continue
        observations = track.get('observations', [])[:count]
        if len(observations) < 6 or any(o.get('observed') is False or o.get('interpolated') is True
                or rejects_moving_association(o)
                or not all(_finite(o.get(key)) for key in ('time', 'x', 'y', 'radius', 'confidence'))
                or o['radius'] <= 0 or not 0 <= o['confidence'] <= 1 or not isinstance(o.get('frame_index'), int)
                or isinstance(o['frame_index'], bool) or o['frame_index'] < 0 for o in observations):
            continue
        if any(b['time'] <= a['time'] or b['frame_index'] <= a['frame_index']
               for a, b in zip(observations, observations[1:])):
            continue
        if observations[-1]['time'] - observations[0]['time'] < .16:
            continue
        end = observations[-1]
        before = [o for o in observations if end['time'] - o['time'] <= .35]
        if len(before) < 5:
            continue
        _, velocity, residual = _flight_fit(before, end['time'], window=len(before))
        radius = max(.5, float(np.median([o.get('radius', 2.) for o in before])))
        drop = before[-1]['y'] - before[0]['y']
        if (drop < 20 or velocity[1] < 25 or residual > max(2., radius * 1.5)
                or np.mean(np.diff([o['y'] for o in before]) > 0) < .8):
            continue
        candidates.append((drop / (1 + residual), track, observations, velocity, radius))
    selected = []
    for _, track, observations, velocity, radius in sorted(candidates, key=lambda row: -row[0]):
        end = observations[-1]
        if any(abs(end['time'] - old[1][-1]['time']) < .12
               and math.dist((end['x'], end['y']), (old[1][-1]['x'], old[1][-1]['y'])) < 8
               for old in selected):
            continue
        selected.append((track, observations, velocity, radius))
        if len(selected) >= budget:
            break
    return selected


def _components(previous, current, following, *, origin, scale, frame_index, timestamp, radius):
    """Return actual connected-component centroids, not projected positions."""
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    following = cv2.cvtColor(following, cv2.COLOR_BGR2GRAY)
    sx, sy = scale
    contrast = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), max(1., sx)))
    change = np.minimum(cv2.absdiff(gray, previous), cv2.absdiff(gray, following))
    saturation = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = ((contrast >= 8) & (change >= 8) & (gray >= 100) & (saturation <= 160)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    rows = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        if (area < max(3, sx * sy * .4) or area > max(16, math.pi * (radius * sx * 2.5) ** 2)
                or max(width, height) / max(1, min(width, height)) > 6):
            continue
        pixels = labels[y:y + height, x:x + width] == label
        local_contrast = float(np.mean(contrast[y:y + height, x:x + width][pixels]))
        temporal_change = float(np.mean(change[y:y + height, x:x + width][pixels]))
        if local_contrast < 12 or temporal_change < 12:
            continue
        source_x, source_y = centers[label] + origin
        rows.append({'x': float(source_x / sx), 'y': float(source_y / sy),
            'radius': max(.5, math.sqrt(area / math.pi / (sx * sy))),
            'frame_index': frame_index, 'time': timestamp, 'observed': True,
            'confidence': min(.8, .3 + min(local_contrast, temporal_change) / 100),
            'confidence_kind': 'heuristic_rank_not_probability',
            'method': 'native_bidirectional_temporal_component',
            'native_resolution_observation': True,
            'source_px': [float(source_x), float(source_y)],
            'source_bbox': [origin[0] + x, origin[1] + y, width, height],
            'local_contrast': round(local_contrast, 4),
            'bidirectional_temporal_change': round(temporal_change, 4),
            'contact_observed': False, 'identity_verified': False})
    return rows


def apply_recovery_occlusion_context(points, record):
    """Bind the detector's same-frame possible-occluder mask to new pixels.

    Missing/malformed context remains unknown. These boxes are hypotheses, not
    person or court confirmation, and use the detector's identical point test.
    """
    if not isinstance(record, dict) or not isinstance(record.get('occlusion_regions'), list):
        return
    boxes = [region.get('bbox') if isinstance(region, dict) else None for region in record['occlusion_regions']]
    if any(not isinstance(box, (tuple, list)) or len(box) != 4
           or not all(_finite(value) for value in box) or box[2] <= 0 or box[3] <= 0 for box in boxes):
        return
    for point in points:
        if (point.get('frame_index') != record.get('frame_index') or not _finite(record.get('time'))
                or abs(point['time'] - record['time']) > 1e-5):
            continue
        overlap = any(x <= point['x'] <= x + w and y <= point['y'] <= y + h for x, y, w, h in boxes)
        point['possible_occluder_overlap'] = overlap
        point['occlusion_context'] = {'method': 'same_frame_detector_regions',
            'frame_index': record['frame_index'], 'region_count': len(boxes), 'person_confirmed': False}
        if overlap:
            point['confidence'] = round(point['confidence'] * .85, 4)


def recover_rebound_tracks(tracks, source_frames, timestamps, analysis_size, *, max_approaches=24, frame_records=None):
    """Return additional measured paths and audit counts within fixed budgets.

    This optional experiment performs no whole-frame candidate reranking. It
    decodes each required source frame once and retains only bounded crops.
    Search points and source timestamps are never taken from evaluation labels.
    """
    from .tracking import _rebound_extensions
    if isinstance(max_approaches, bool) or not isinstance(max_approaches, int) or not 0 <= max_approaches <= 64:
        raise ValueError('max_approaches must be an integer between 0 and 64')
    if (not isinstance(analysis_size, (tuple, list)) or len(analysis_size) != 2
            or not all(_finite(v) and v > 0 for v in analysis_size)):
        raise ValueError('Recovery analysis_size must contain two positive finite dimensions')
    if (not all(_finite(t) and t >= 0 for t in timestamps)
            or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
        raise ValueError('Recovery timestamps must be finite, nonnegative, and strictly increasing')
    summary = {'method': 'native_bidirectional_endpoint_recovery', 'approaches': 0,
               'decoded_frames': 0, 'raw_components': 0, 'additional_tracks': 0,
               'crop_budget_bytes': MAX_CROP_BYTES, 'crop_bytes': 0, 'budget_skipped_windows': 0,
               'identity_confirmed': False, 'physical_contact_confirmed': False}
    context = {record.get('frame_index'): record for record in frame_records or []
               if isinstance(record, dict) and isinstance(record.get('frame_index'), int)
               and not isinstance(record['frame_index'], bool)}
    summary['occlusion_context_frame_count'] = len(context)
    if not max_approaches or len(timestamps) < 5:
        return [], summary
    selected = _approaches(tracks, max_approaches)
    selected = [item for item in selected if all(o['frame_index'] < len(timestamps)
                and abs(o['time'] - timestamps[o['frame_index']]) <= 1e-5 for o in item[1])]
    if not selected:
        return [], summary
    width, height = analysis_size
    if len(source_frames) != len(timestamps):
        raise ValueError('Recovery source frame count and timestamps must match')
    first = _source_frame(source_frames[0])
    sh, sw = first.shape[:2]
    sx, sy = sw / width, sh / height
    if abs(sx / sy - 1) > .03:
        raise ValueError('Recovery source and analysis must have matching aspect ratios')
    steps = np.diff(np.asarray(timestamps, dtype=float))
    nominal = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 1 / 30
    offset = max(1, min(6, int(round(.067 / nominal))))
    requests = defaultdict(list)
    windows = []
    for track, observations, velocity, radius in selected:
        end = observations[-1]
        indices = [i for i, timestamp in enumerate(timestamps)
                   if 0 < timestamp - end['time'] <= .35 and offset <= i < len(timestamps) - offset]
        if len(indices) > 32:
            indices = [indices[i] for i in np.linspace(0, len(indices) - 1, 32, dtype=int)]
        if not indices:
            continue
        padding = 20.
        left = max(0, int((end['x'] + min(0., velocity[0] * .35) - padding) * sx))
        right = min(sw, int((end['x'] + max(0., velocity[0] * .35) + padding) * sx) + 1)
        top = max(0, int((end['y'] - padding) * sy))
        bottom = min(sh, int((end['y'] + min(60., max(padding, velocity[1] * .25))) * sy) + 1)
        if right - left > 768 or bottom - top > 768 or right <= left or bottom <= top:
            continue
        needed = sorted({j for i in indices for j in (i - offset, i, i + offset)})
        crop_bytes = len(needed) * (right - left) * (bottom - top) * 3
        if summary['crop_bytes'] + crop_bytes > summary['crop_budget_bytes']:
            summary['budget_skipped_windows'] += 1
            continue
        summary['crop_bytes'] += crop_bytes
        window = {'track': track, 'observations': observations, 'radius': radius,
                  'box': (left, top, right, bottom), 'indices': indices, 'crops': {}}
        windows.append(window)
        for index in needed:
            requests[index].append(window)
    for index in sorted(requests):
        frame = first if index == 0 else _source_frame(source_frames[index])
        if frame.shape[:2] != (sh, sw):
            raise ValueError('Recovery source dimensions changed')
        for window in requests[index]:
            x0, y0, x1, y1 = window['box']
            window['crops'][index] = frame[y0:y1, x0:x1].copy()
    output = []
    next_id = max([int(t['track_id']) for t in tracks if str(t.get('track_id', '')).isdigit()] or [0]) + 1
    for window in windows:
        rows = []
        crops = window['crops']
        for index in window['indices']:
            points = _components(crops[index - offset], crops[index], crops[index + offset],
                origin=window['box'][:2], scale=(sx, sy), frame_index=index,
                timestamp=timestamps[index], radius=window['radius'])
            for point in points:
                point['source_size'] = [sw, sh]
                point['temporal_comparison_frame_indices'] = [index - offset, index + offset]
                point['appearance_polarity'] = 'bright'
            apply_recovery_occlusion_context(points, context.get(index))
            summary['raw_components'] += len(points)
            points.sort(key=lambda p: -min(p['local_contrast'], p['bidirectional_temporal_change']))
            rows.append((timestamps[index], points[:48]))
        for rebound, residual in _rebound_extensions(window['observations'], rows):
            track = deepcopy(window['track'])
            track['track_id'] = next_id
            next_id += 1
            track['observations'] = [*deepcopy(window['observations']), *rebound]
            track['identity_ambiguous'] = True
            # source_track_ids denotes exclusive merged-track ownership in the
            # identity graph. This is a competing hypothesis, not a merge.
            track['source_track_ids'] = [track['track_id']]
            track['recovery_parent_track_id'] = window['track']['track_id']
            track['reconnections'] = []
            track['reconnection_hypotheses'] = []
            track['ambiguity_events'] = [*track.get('ambiguity_events', []),
                {'reason': 'unverified_native_rebound_recovery'}]
            first_observation, last_observation = track['observations'][0], track['observations'][-1]
            for key, value in {'start_frame': first_observation['frame_index'],
                               'end_frame': last_observation['frame_index'],
                               'start_time': first_observation['time'],
                               'end_time': last_observation['time'],
                               'duration_s': last_observation['time'] - first_observation['time']}.items():
                if key in track:
                    track[key] = value
            track['gaps'] = [{'start_frame': a['frame_index'] + 1, 'end_frame': b['frame_index'] - 1,
                'missing_frame_count': b['frame_index'] - a['frame_index'] - 1,
                'duration_s': b['time'] - a['time'], 'resolved': False,
                'possible_occlusion': False, 'cause': 'unresolved_candidate_association',
                'observations_interpolated': False}
                for a, b in zip(track['observations'], track['observations'][1:])
                if b['frame_index'] > a['frame_index'] + 1]
            track['method'] = 'native_rebound_recovery_hypothesis'
            track['flight_hypothesis'].update(rebound_count=len(rebound), verified_identity=False,
                physical_contact_confirmed=False, native_recovery=True,
                recovery_fit_residual_px=residual)
            observations = track['observations']
            count = len(observations)
            span = observations[-1]['frame_index'] - observations[0]['frame_index'] + 1
            coverage = count / span
            average_confidence = float(np.mean([row['confidence'] for row in observations]))
            identity_score = min(.95, average_confidence * coverage * min(1., count / 6)) * .45
            xy = np.asarray([[row['x'], row['y']] for row in observations], dtype=float)
            track.update(identity_confidence=round(identity_score, 4),
                confidence_kind='heuristic_rank_not_probability', observation_count=count,
                observed_frame_fraction=round(coverage, 4),
                motion_span_px=round(float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))), 3),
                target_court_confirmed=False)
            background = [row['frame_index'] for row in observations if rejects_moving_association(row)]
            transient = [row['frame_index'] for row in observations if observation_support_status(row) == 'transient']
            track['moving_observation_support'] = {
                'method': 'coordinate_matched_source_pixel_evidence',
                'persistent_background_observation_count': len(background),
                'persistent_background_frame_indices': background,
                'transient_observation_count': len(transient), 'transient_frame_indices': transient,
                'unknown_observation_count': count - len(background) - len(transient),
                'raw_input_records_unmodified': True, 'identity_verified': False, 'physical_contact_verified': False}
            output.append(track)
        crops.clear()
    summary.update(approaches=len(windows), decoded_frames=len(set(requests) | {0}), additional_tracks=len(output))
    return output, summary


def guard_native_rebound_branches(candidates, tracks, nominal_step):
    """Keep tiny native fragments from inheriting a whole object's flight tier.

    A native connected component is a current-pixel observation, but a much
    smaller component after a jump can be a feather fragment, clothing edge or
    background texture. Three consecutive, consistently much smaller source
    components do not provide enough appearance continuity to promote that
    rebound for review. The candidate and all coordinates remain available;
    neither rejection nor passing this check establishes physical identity.
    """
    if not _finite(nominal_step) or nominal_step <= 0:
        return list(candidates)
    by_id = {}
    for track in tracks:
        if not isinstance(track.get('flight_hypothesis'), dict):
            continue
        rows = [row for row in track.get('observations', [])
                if row.get('observed') is not False and row.get('interpolated') is not True
                and not rejects_moving_association(row)
                and all(_finite(row.get(key)) for key in ('frame_index', 'time', 'x', 'y', 'radius'))
                and row['radius'] > 0]
        by_id[str(track['track_id'])] = sorted(
            {row['frame_index']: row for row in rows}.values(), key=lambda row: row['time'])
    output = list(candidates)
    for index, candidate in enumerate(candidates):
        support = candidate.get('flight_trajectory_support') or {}
        motion = candidate.get('image_motion_event') or {}
        if (support.get('supported_for_review') is not True or candidate.get('contact_observed') is True
                or motion.get('kind') != 'descent_and_rebound' or not _finite(candidate.get('time'))):
            continue
        time = candidate['time']
        rows = by_id.get(str(candidate.get('track_id')), [])
        before = [row for row in rows if 0 <= time - row['time'] <= .35]
        if len(before) < 5 or before[-1]['time'] - before[0]['time'] < .12:
            continue
        post, previous = [], time
        for row in rows:
            if row['time'] <= time:
                continue
            if row['time'] - time > max(.16, nominal_step * 4.01) or row['time'] - previous > nominal_step * 2.5:
                break
            post.append(row)
            previous = row['time']
            if len(post) == 3:
                break
        if len(post) != 3 or any(row.get('method') != 'native_bidirectional_temporal_component'
                or row.get('native_resolution_observation') is not True for row in post):
            continue
        baseline = float(np.median([row['radius'] for row in before]))
        ratios = [row['radius'] / baseline for row in post]
        median_ratio = float(np.median(ratios))
        # Requiring a sustained change across every measured suffix sample
        # avoids vetoing a single quantization/blur fluctuation. This is only a
        # review-support guard, not an empirically calibrated identity model.
        if median_ratio > .45 or max(ratios) > .65:
            continue
        updated = deepcopy(candidate)
        reason = 'NATIVE_REBOUND_APPEARANCE_SUPPORT_INSUFFICIENT'
        updated['flight_trajectory_support']['supported_for_review'] = False
        updated['flight_trajectory_support']['reason_codes'] = list(dict.fromkeys([
            *support.get('reason_codes', []), reason]))
        updated['reason_codes'] = list(dict.fromkeys([*candidate.get('reason_codes', []), reason]))
        updated['native_rebound_guard'] = {'method': 'native_component_appearance_consistency',
            'flight_support_veto': True, 'incoming_radius_px': baseline,
            'native_radius_ratio': median_ratio, 'native_radius_ratios': ratios,
            'incoming_frame_indices': [row['frame_index'] for row in before],
            'native_frame_indices': [row['frame_index'] for row in post],
            'median_radius_ratio_limit': .45, 'maximum_radius_ratio_limit': .65,
            'identity_confirmed': False, 'physical_contact_confirmed': False,
            'coordinates_changed': False, 'candidate_removed': False}
        output[index] = updated
    return output
