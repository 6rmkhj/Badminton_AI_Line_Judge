"""Bounded temporal pixel evidence, independent of object identity or contact.

A persistent feature can be a floor mark, a box edge, or a genuinely stationary
shuttle. It is unsuitable as new *moving-object* evidence; this module never
declares that it is not a shuttle. Moving hands/rackets can be transient too.
Unknown evidence is neutral. No proposal coordinate or input image is changed.
"""
from __future__ import annotations

import math
from copy import deepcopy
from statistics import median
import time

import cv2
import numpy as np


THRESHOLDS = {
    'minimum_contrast': 8.0,
    'maximum_background_error': 5.0,
    'maximum_translation': 4.0,
    'minimum_side_comparisons': 2,
    'minimum_persistence_fraction': .8,
    'stable_residual_contrast_ratio': .3,
    'minimum_retained_contrast_ratio': .65,
    'minimum_transient_difference': 12.0,
    'minimum_transient_contrast_ratio': .45,
    'maximum_center_transient_fraction': .2,
    'maximum_component_outside_footprint_fraction': .25,
}


def _finite(value):
    try:
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _gray(frame):
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim not in (2, 3):
        raise ValueError('Observation support requires actual uint8 gray/BGR frames.')
    if frame.ndim == 2:
        return frame.astype(np.float32)
    if frame.shape[2] not in (3, 4):
        raise ValueError('Expected gray, BGR or BGRA source pixels.')
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if frame.shape[2] == 3 else cv2.COLOR_BGRA2GRAY).astype(np.float32)


def _background_error(current, other, ring):
    difference = current - other
    offset = float(np.median(difference[ring]))
    return float(np.median(np.abs(difference[ring] - offset))), offset


def _align(current, other, ring, *, prealigned=False):
    """Accept registration only when it improves the robust background ring.

    ECC correlation alone can follow a nearby moving player. The target and
    immediate neighborhood are masked out and zero motion remains a candidate.
    Unreliable/camera-ambiguous comparisons cannot establish persistence.
    """
    base_error, offset = _background_error(current, other, ring)
    if prealigned or base_error <= 2.0:
        return other, offset, {'accepted': base_error <= THRESHOLDS['maximum_background_error'],
            'translation_px': [0., 0.], 'background_error': round(base_error, 4),
            'unregistered_background_error': round(base_error, 4), 'ecc_correlation': None,
            'method': 'caller_registered_residual_check' if prealigned else 'stable_zero_translation_ring'}
    aligned, shift, error = other, [0., 0.], base_error
    correlation = None
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(current / 255., other / 255., warp,
            cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .001),
            ring.astype(np.uint8) * 255, 3)
        correlation = float(cc) if math.isfinite(cc) else None
        if correlation is not None and correlation >= .75 and np.linalg.norm(warp[:, 2]) <= THRESHOLDS['maximum_translation']:
            proposed = cv2.warpAffine(other, warp, (current.shape[1], current.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
            candidate_error, candidate_offset = _background_error(current, proposed, ring)
            if candidate_error < base_error * .8:
                aligned, error, offset = proposed, candidate_error, candidate_offset
                shift = [float(v) for v in warp[:, 2]]
    except cv2.error:
        pass
    return aligned, offset, {'accepted': error <= THRESHOLDS['maximum_background_error'],
        'translation_px': [round(v, 4) for v in shift], 'background_error': round(error, 4),
        'unregistered_background_error': round(base_error, 4), 'ecc_correlation': correlation,
        'method': 'robust_ring_translation'}


def observation_support_status(observation):
    """Read only current, coordinate-matched evidence; stale/unknown is neutral."""
    if not isinstance(observation, dict):
        return 'unknown'
    evidence = observation.get('observation_support')
    if (not isinstance(evidence, dict) or type(evidence.get('schema_version')) is not int
            or evidence['schema_version'] != 1 or evidence.get('status') not in ('persistent_background', 'transient')
            or evidence.get('evidence_valid') is not True
            or observation.get('observed') is False or observation.get('interpolated') is True):
        return 'unknown'
    point, size = evidence.get('point_px'), evidence.get('coordinate_size')
    if (not isinstance(point, (list, tuple)) or len(point) != 2 or not all(_finite(v) for v in point)
            or not isinstance(size, (list, tuple)) or len(size) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size)
            or not all(_finite(observation.get(k)) for k in ('x', 'y'))
            or any(abs(observation[k] - value) > 1e-6 for k, value in zip(('x', 'y'), point))):
        return 'unknown'
    if (not isinstance(evidence.get('frame_index'), int) or isinstance(evidence['frame_index'], bool)
            or evidence['frame_index'] < 0):
        return 'unknown'
    if 'frame_index' in observation and (not isinstance(observation['frame_index'], int)
                                       or isinstance(observation['frame_index'], bool)
                                       or observation['frame_index'] != evidence['frame_index']):
        return 'unknown'
    coordinate_size = observation.get('coordinate_size', observation.get('source_patch_coordinate_size'))
    if coordinate_size is not None and (not isinstance(coordinate_size, (list, tuple)) or list(coordinate_size) != list(size)):
        return 'unknown'
    expected_reject = evidence['status'] == 'persistent_background'
    return evidence['status'] if evidence.get('reject_for_moving_association') is expected_reject else 'unknown'


def rejects_moving_association(observation):
    """Recognize only complete, current persistent-background moving evidence."""
    return observation_support_status(observation) == 'persistent_background'


def assess_observation_support(frames, frame_index, observation, *, config=None, frame_indices=None):
    """Return versioned evidence for a proposal in the supplied frame coordinates.

    At most seven small patches are read. ``frame_indices`` maps a supplied
    sparse sequence to original frame numbers; absent frames are never filled.
    Native-pixel callers must transform both the candidate and every patch to
    the same coordinate system before calling. The default temporal offsets
    are [-4,-2,-1,0,1,2,4]; only actual available source images are compared.
    Configuration controls computation bounds, not classification thresholds.
    ``prealigned=True`` skips ECC for caller-registered actual source patches,
    but each background-ring residual must still pass the same quality check.
    Persistence describes image appearance over the stated frames; a truly
    stationary or unresolved slow-moving shuttle cannot be identified from
    these pixels alone. Raw observations must remain available for review.
    """
    cfg = dict(config or {})
    if set(cfg) - {'patch_radius', 'prealigned'}:
        raise ValueError('Only patch_radius/prealigned are configurable; evidence thresholds are fixed.')
    prealigned = cfg.get('prealigned', False)
    if not isinstance(prealigned, bool):
        raise ValueError('prealigned must be a boolean.')
    radius = cfg.get('patch_radius', 24)
    if not isinstance(radius, int) or isinstance(radius, bool) or not 16 <= radius <= 48:
        raise ValueError('patch_radius must be an integer in16..48.')
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError('frame_index must be a nonnegative source frame number.')
    if not isinstance(observation, dict) or not all(_finite(observation.get(k)) for k in ('x', 'y')):
        raise ValueError('A measured candidate requires finite x/y coordinates.')
    indices = list(range(len(frames))) if frame_indices is None else list(frame_indices)
    if (len(indices) != len(frames) or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices)
            or indices != sorted(set(indices)) or frame_index not in indices):
        raise ValueError('frame_indices must be unique increasing actual indices containing the candidate frame.')
    lookup = dict(zip(indices, range(len(indices))))
    used = [frame_index]
    for direction in (-1, 1):
        side = [frame_index + direction * d for d in (4, 2, 1) if frame_index + direction * d in lookup]
        if len(side) < 3 and frame_index + direction * 3 in lookup:
            side.append(frame_index + direction * 3)
        used.extend(side)
    used.sort()
    result = {'schema_version': 1, 'status': 'unknown', 'evidence_valid': False,
        'reject_for_moving_association': False, 'reason_codes': [], 'frame_index': frame_index,
        'window_frame_indices': used, 'persistence_fraction': None, 'transient_support_fraction': None,
        'alignment': [], 'thresholds': dict(THRESHOLDS), 'needs_native_resolution': False,
        'confirms_identity': False, 'confirms_contact': False,
        'score_kind': 'heuristic_pixel_support_not_probability',
        'point_px': [float(observation['x']), float(observation['y'])],
        'coordinate_size': list(np.asarray(frames[lookup[frame_index]]).shape[:2][::-1])}
    if observation.get('observed') is False or observation.get('interpolated') is True:
        result['reason_codes'] = ['NONMEASURED_PROPOSAL']
        return result
    source = np.asarray(frames[lookup[frame_index]])
    height, width = source.shape[:2]
    x, y = float(observation['x']), float(observation['y'])
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError('Candidate coordinates must lie inside the actual frame.')
    box = observation.get('bbox', [x - 1, y - 1, 2, 2])
    if (not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_finite(v) for v in box)
            or box[2] <= 0 or box[3] <= 0):
        raise ValueError('Candidate bbox must contain finite x/y and positive width/height.')
    if min(box[2:]) < 1:
        result.update(needs_native_resolution=True, reason_codes=['SUBPIXEL_APPEARANCE_UNRESOLVED'])
        return result
    left, top = max(0, int(x) - radius), max(0, int(y) - radius)
    right, bottom = min(width, int(x) + radius + 1), min(height, int(y) + radius + 1)
    current = _gray(source[top:bottom, left:right])
    result['patch_bbox'] = [left, top, right - left, bottom - top]
    core = np.zeros(current.shape, bool)
    bx, by, bw, bh = box
    x1, x2 = max(0, math.floor(bx - left)), min(current.shape[1], math.ceil(bx + bw - left))
    y1, y2 = max(0, math.floor(by - top)), min(current.shape[0], math.ceil(by + bh - top))
    core[y1:y2, x1:x2] = True
    yy, xx = np.indices(current.shape)
    near = (xx - (x - left)) ** 2 + (yy - (y - top)) ** 2 <= 2.5 ** 2
    exclusion = cv2.dilate(core.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    ring = ~exclusion
    ring[:4] = ring[-4:] = False
    ring[:, :4] = ring[:, -4:] = False
    if ring.sum() < 128 or not core.any() or not near.any():
        result['reason_codes'] = ['INSUFFICIENT_BACKGROUND_CONTEXT']
        return result
    local = current - cv2.GaussianBlur(current, (0, 0), 3)
    polarity = observation.get('appearance_polarity')
    sign = -1 if polarity == 'dark' else 1 if polarity == 'bright' else (1 if np.median(local[core]) >= 0 else -1)
    contrast = sign * local
    peak = float(np.max(contrast[core]))
    support = core & (contrast >= max(3., peak * .3))
    if not support.any() or float(np.mean(contrast[support])) < THRESHOLDS['minimum_contrast']:
        result.update(needs_native_resolution=True, reason_codes=['LOCAL_APPEARANCE_UNRESOLVED'])
        return result
    strength = float(np.mean(contrast[support]))
    result.update(local_contrast=round(strength, 4), support_pixel_count=int(support.sum()))
    # A model's appearance box may describe a broad static edge behind a useful
    # localization. Such a footprint cannot disprove an independently moving
    # object at the point. Only isolated, locally bounded appearance can veto.
    component_mask = (contrast >= max(3., peak * .3)).astype(np.uint8)
    _, labels = cv2.connectedComponents(component_mask)
    identities = set(labels[support].tolist()) - {0}
    connected = np.isin(labels, list(identities))
    dilated = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    outside_fraction = float(np.mean(~dilated[connected]))
    touches_boundary = bool(connected[0].any() or connected[-1].any()
                            or connected[:, 0].any() or connected[:, -1].any())
    bounded_footprint = (not touches_boundary
                        and outside_fraction <= THRESHOLDS['maximum_component_outside_footprint_fraction'])
    result.update(component_outside_footprint_fraction=round(outside_fraction, 4),
                  component_touches_patch_boundary=touches_boundary)
    aligned_patches, stable, valid_indices = [], [], []
    for index in used:
        if index == frame_index:
            continue
        raw = np.asarray(frames[lookup[index]])
        if raw.shape != source.shape:
            raise ValueError('Compared frames must share the same actual pixel geometry.')
        other = _gray(raw[top:bottom, left:right])
        aligned, offset, quality = _align(current, other, ring, prealigned=prealigned)
        quality['frame_index'] = index
        result['alignment'].append(quality)
        if not quality['accepted']:
            continue
        residual = float(np.mean(np.abs(current - aligned - offset)[support]))
        retained = float(np.mean((sign * (aligned - cv2.GaussianBlur(aligned, (0, 0), 3)))[support]) / strength)
        quality.update(support_residual=round(residual, 4), retained_contrast_ratio=round(retained, 4))
        stable.append(residual <= max(THRESHOLDS['minimum_contrast'], strength * THRESHOLDS['stable_residual_contrast_ratio'])
                      and retained >= THRESHOLDS['minimum_retained_contrast_ratio'])
        aligned_patches.append(aligned + offset)
        valid_indices.append(index)
    sides = [sum(index < frame_index for index in valid_indices), sum(index > frame_index for index in valid_indices)]
    if min(sides) < THRESHOLDS['minimum_side_comparisons']:
        result['reason_codes'] = ['INSUFFICIENT_RELIABLE_TEMPORAL_CONTEXT']
        return result
    background = np.median(np.stack(aligned_patches), axis=0)
    difference = np.abs(current - background)
    transient_threshold = max(THRESHOLDS['minimum_transient_difference'], strength * THRESHOLDS['minimum_transient_contrast_ratio'])
    center_transient = float(np.mean(difference[near] >= transient_threshold))
    support_transient = float(np.mean(difference[support] >= transient_threshold))
    persistence = sum(stable) / len(stable)
    result.update(evidence_valid=True, persistence_fraction=round(persistence, 4),
        transient_support_fraction=round(support_transient, 4), center_transient_fraction=round(center_transient, 4),
        temporal_difference_threshold=round(transient_threshold, 4))
    if (persistence >= THRESHOLDS['minimum_persistence_fraction']
            and center_transient <= THRESHOLDS['maximum_center_transient_fraction']):
        if bounded_footprint:
            result.update(status='persistent_background', reject_for_moving_association=True,
                          reason_codes=['PERSISTENT_APPEARANCE_AT_BACKGROUND_LOCATION'])
        else:
            result['reason_codes'] = ['PERSISTENT_BROAD_STRUCTURE_AMBIGUOUS_LOCALIZATION']
    elif support_transient >= .5 or center_transient > THRESHOLDS['maximum_center_transient_fraction']:
        result.update(status='transient', reason_codes=['LOCAL_TRANSIENT_PIXEL_SUPPORT', 'TRANSIENCE_DOES_NOT_CONFIRM_IDENTITY'])
    else:
        result['reason_codes'] = ['MIXED_TEMPORAL_APPEARANCE']
    return result


def remap_native_support(observation, native_evidence, *, source_roi, scale_xy, coordinate_size):
    """Bind native-patch evidence to its unchanged analysis observation.

    The complete native evidence and its pixel units are retained separately;
    mapping does not create an observation or transfer evidence to another
    point. Only <=.001 analysis-pixel rounding disagreement is tolerated.
    """
    if (not isinstance(native_evidence, dict) or not isinstance(source_roi, (list, tuple)) or len(source_roi) != 4
            or not all(_finite(v) for v in source_roi) or min(source_roi[:2]) < 0 or min(source_roi[2:]) <= 0
            or not isinstance(scale_xy, (list, tuple)) or len(scale_xy) != 2
            or not all(_finite(v) and v > 0 for v in scale_xy)
            or not isinstance(coordinate_size, (list, tuple)) or len(coordinate_size) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in coordinate_size)
            or not isinstance(native_evidence.get('coordinate_size'), (list, tuple))
            or list(native_evidence['coordinate_size']) != list(source_roi[2:])):
        raise ValueError('Native support requires consistent bounded source-patch geometry.')
    if any(source_roi[i] + source_roi[i + 2] > coordinate_size[i] * scale_xy[i] for i in range(2)):
        raise ValueError('Native patch lies outside the stated source/analysis transform.')
    point = native_evidence.get('point_px')
    if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(_finite(v) for v in point):
        raise ValueError('Native evidence must identify its evaluated actual point.')
    mapped = [(point[i] + source_roi[i]) / scale_xy[i] for i in range(2)]
    if (not all(_finite(observation.get(key)) for key in ('x', 'y'))
            or any(abs(mapped[i] - observation[key]) > .001 for i, key in enumerate(('x', 'y')))
            or ('frame_index' in observation and observation['frame_index'] != native_evidence.get('frame_index'))):
        raise ValueError('Native evidence cannot be attached to a different observation or frame.')
    result = deepcopy(native_evidence)
    result.update(point_px=[float(observation['x']), float(observation['y'])], coordinate_size=list(coordinate_size),
        native_pixel_evidence=deepcopy(native_evidence),
        native_mapping={'source_roi': list(source_roi), 'source_pixels_per_analysis_pixel': list(scale_xy),
                        'evaluated_analysis_point_before_rounding': mapped,
                        'native_evidence_units': 'actual_source_patch_pixels'})
    return result


def annotate_observations(records, frames, *, frame_indices=None, config=None, progress_callback=None):
    """Copy records and annotate a bounded share; unassessed observations stay neutral.

    Returns ``(copied_records, report)``. The default eight candidates per frame
    prioritize measured native-motion-search candidates, then existing scores.
    A caller may explicitly request1..80; this never enlarges the raw pool or
    removes any candidate. Prior annotations are cleared in the output before
    budgeting so reanalysis cannot reuse stale pixel evidence. Native-only subpixel cases are counted for a caller
    supplied source-patch check via ``remap_native_support``.
    """
    started = time.perf_counter()
    cfg = dict(config or {})
    if set(cfg) - {'max_observations_per_frame', 'patch_radius', 'prealigned'}:
        raise ValueError('Unknown observation-support batch setting.')
    limit = cfg.pop('max_observations_per_frame', 8)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 80:
        raise ValueError('max_observations_per_frame must be in1..80.')
    patch_radius = cfg.get('patch_radius', 24)
    if (not isinstance(patch_radius, int) or isinstance(patch_radius, bool) or not 16 <= patch_radius <= 48
            or not isinstance(cfg.get('prealigned', False), bool)):
        raise ValueError('Batch patch_radius/prealigned settings must satisfy the single-observation contract.')
    output = deepcopy(list(records))
    indices = list(range(len(frames))) if frame_indices is None else list(frame_indices)
    if (len(indices) != len(frames) or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError('Batch frame_indices must map unique increasing actual frames.')
    lookup = dict(zip(indices, range(len(indices))))
    report = {'schema_version': 1, 'method': 'bounded_local_temporal_pixel_support_v1',
        'record_count': len(output), 'assessed_observations': 0, 'unique_assessments': 0,
        'persistent_background_observations': 0, 'transient_observations': 0, 'unknown_observations': 0,
        'native_needed_observations': 0, 'unassessed_due_to_budget': 0, 'invalid_geometry_observations': 0,
        'max_observations_per_frame': limit, 'thresholds': dict(THRESHOLDS),
        'original_records_preserved': True, 'coordinates_changed': False,
        'confirms_identity': False, 'confirms_contact': False}
    for position, record in enumerate(output):
        index = record.get('frame_index')
        if not isinstance(index, int) or isinstance(index, bool) or index not in lookup:
            raise ValueError('Each detection record must identify an available actual source frame.')
        window_indices = [i for i in range(max(0, index - 4), index + 5) if i in lookup]
        window_frames = [frames[lookup[i]] for i in window_indices]
        candidates = record.get('detections', [])
        if not isinstance(candidates, list):
            raise ValueError('Detection records require a list of actual candidates.')
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError('Each measured candidate must be an object.')
            candidate.pop('observation_support', None)
        def priority(item):
            order, value = item
            score = value.get('confidence', 0)
            return (not bool(value.get('native_motion_search_supported')),
                    -(float(score) if _finite(score) else 0), order)
        chosen = sorted(enumerate(candidates), key=priority)[:limit]
        report['unassessed_due_to_budget'] += max(0, len(candidates) - limit)
        cache = {}
        for _, candidate in chosen:
            signature = repr((candidate.get('x'), candidate.get('y'), candidate.get('bbox'),
                              candidate.get('appearance_polarity'), candidate.get('observed'), candidate.get('interpolated')))
            try:
                if signature not in cache:
                    cache[signature] = assess_observation_support(window_frames, index, candidate,
                        config=cfg, frame_indices=window_indices)
                    report['unique_assessments'] += 1
                evidence = deepcopy(cache[signature])
            except ValueError:
                # Invalid geometry cannot acquire or retain a negative label.
                candidate.pop('observation_support', None)
                report['invalid_geometry_observations'] += 1
                continue
            candidate['observation_support'] = evidence
            report['assessed_observations'] += 1
            report[evidence['status'] + '_observations'] += 1
            report['native_needed_observations'] += int(evidence['needs_native_resolution'])
        if progress_callback is not None:
            progress_callback({'processed_frames': position + 1, 'total_frames': len(output),
                               'assessed_observations': report['assessed_observations']})
    report.update(elapsed_seconds=round(time.perf_counter() - started, 6),
                  rejected_moving_observations=report['persistent_background_observations'])
    return output, report
