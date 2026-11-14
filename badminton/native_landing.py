"""Model-conditioned landing envelopes from actual source-frame foreground.

This path estimates a visible first image rebound. It never turns motion,
appearance, or a fitted trajectory into observed physical ground contact.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import re

import cv2
import numpy as np


_MAX_CROP_BYTES = 96 * 1024 * 1024


def _finite(value):
    return (isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value))


def _base(family):
    return {'schema_version': 1, 'status': 'review', 'decision_basis': 'video_estimate',
            'method': 'native_temporal_foreground_envelope',
            'family_id': family.get('family_id') if isinstance(family, dict) else None,
            'contact_observed': False, 'identity_confirmed': False,
            'physical_ground_contact_verified': False, 'evaluation_eligible': False,
            'confidence': None, 'confidence_kind': 'unvalidated_evidence_conditions_not_probability',
            'possible_contact_polygon_px': None, 'uncertainty_px': None,
            'frame_interval': None, 'time_interval': None,
            'support_frames': [], 'observed_frame_count': 0, 'decoded_frame_count': 0,
            'assumptions': [
                'The selected compact foreground belongs to one target flight.',
                'Its first supported image-space reversal represents a floor rebound.',
                'The contact location follows the incoming/outgoing local motion models within their measured residuals.',
                'The visible foreground extent includes motion blur and object extent; it is not a measured cork footprint.',
            ], 'reason_codes': []}


def _reject(result, reason):
    result['reason_codes'] = list(dict.fromkeys([*result['reason_codes'], reason]))
    result['observed_frame_count'] = len(result['support_frames'])
    return result


def _observations(family, tracks, timestamps):
    ids = {str(value) for value in family.get('track_ids', [])}
    ids.update(str(member.get('track_id')) for member in family.get('members', []) if isinstance(member, dict))
    grouped = defaultdict(list)
    for track in tracks:
        if str(track.get('track_id')) not in ids:
            continue
        for row in track.get('observations', []):
            index = row.get('frame_index')
            if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(timestamps)
                    or row.get('observed') is False or row.get('interpolated') is True
                    or not all(_finite(row.get(k)) for k in ('time', 'x', 'y'))
                    or abs(row['time']-timestamps[index]) > 1e-4):
                continue
            grouped[index].append(row)
    pooled = []
    for index, rows in sorted(grouped.items()):
        coordinates = np.asarray([[row['x'], row['y']] for row in rows])
        pooled.append({'frame_index': index, 'time': float(timestamps[index]),
                       'point': np.median(coordinates, axis=0),
                       'points': np.unique(coordinates, axis=0),
                       'spread': float(np.max(np.linalg.norm(coordinates-np.median(coordinates, axis=0), axis=1))),
                       'occluder': all(row.get('possible_occluder_overlap') is True for row in rows)})
    return pooled


def _component(gray, background, source_seeds, scale, unit, scale_xy=None):
    axes = np.asarray([scale, scale] if scale_xy is None else scale_xy, dtype=float)
    area_scale = float(np.prod(axes))
    delta = gray.astype(np.int16)-background.astype(np.int16)
    mask = (np.abs(delta) >= 10).astype(np.uint8)
    kernel_size = max(1, int(round(scale * .75)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    choices = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if (area < max(12, area_scale*.3*unit*unit)
                or area > area_scale*600*unit*unit
                or max(width, height) > scale*55*unit
                or max(width, height)/max(1, min(width, height)) > 8):
            continue
        yy, xx = np.where(labels[y:y+height, x:x+width] == label)
        distances = [float(np.min(np.hypot((xx+x-seed[0])/axes[0], (yy+y-seed[1])/axes[1])))
                     for seed in source_seeds]
        distance = min(distances)
        if distance > 4*unit:
            continue
        if x == 0 or y == 0 or x+width >= gray.shape[1] or y+height >= gray.shape[0]:
            continue
        support = labels[y:y+height, x:x+width] == label
        values = delta[y:y+height, x:x+width]
        positive, negative = support & (values >= 10), support & (values <= -10)
        positive_count, negative_count = int(positive.sum()), int(negative.sum())
        separation = 0.
        if positive_count and negative_count:
            p = np.column_stack(np.where(positive))[:, ::-1].mean(axis=0)
            n = np.column_stack(np.where(negative))[:, ::-1].mean(axis=0)
            separation = float(np.linalg.norm((p-n)/axes))
        shape = (positive_count >= area_scale*2*unit*unit
                 and negative_count >= area_scale*2*unit*unit
                 and .6*unit <= separation <= 16*unit)
        choices.append({'area': area, 'bbox': [x, y, width, height],
                        'positive': positive_count, 'negative': negative_count,
                        'separation': separation, 'shape_support': shape,
                        'matching_search_seeds': sum(value <= 4*unit for value in distances),
                        'search_seed_count': len(source_seeds),
                        'distance': distance, 'bottom': float(y+height-1)})
    choices.sort(key=lambda item: -item['area'])
    return choices


def _fit_contact(before, after, peak, times, unit):
    """Intersect measured local motion models; expose the interpolation bounds."""
    origin = times[peak['frame_index']]
    models = []
    for rows in (before[-4:], after[:4]):
        ts = np.asarray([times[row['frame_index']]-origin for row in rows])
        points = np.asarray([[row['bbox_px'][0]+row['bbox_px'][2]/2,
                              row['bbox_px'][1]+row['bbox_px'][3]] for row in rows])
        coefficients = np.polyfit(ts, points, 2 if len(rows) >= 4 else 1)
        if len(coefficients) == 2:
            coefficients = np.vstack([np.zeros(2), coefficients])
        fitted = np.column_stack([np.polyval(coefficients[:, axis], ts) for axis in range(2)])
        residual = float(np.max(np.linalg.norm(points-fitted, axis=1)))
        models.append((coefficients, residual))
    incoming, outgoing = models[0][0], models[1][0]
    lower, upper = times[before[-1]['frame_index']]-origin, times[after[0]['frame_index']]-origin
    roots = np.roots(incoming[:, 1]-outgoing[:, 1])
    valid = [float(root.real) for root in roots if abs(root.imag) < 1e-7 and lower <= root.real <= upper]
    if len(valid) != 1:
        return None
    contact_t = valid[0]
    coordinates = np.asarray([[np.polyval(model[:, axis], contact_t) for axis in range(2)]
                              for model in (incoming, outgoing)])
    if np.linalg.norm(coordinates[0]-coordinates[1]) > 12*unit:
        return None
    residual = max(model[1] for model in models)
    if residual > 5*unit:
        return None
    # The residual band determines possible crossing times, rather than a
    # fixed pixel bonus. Include the measured peak extent as an independent
    # exposure/blur observation, even if the fitted crossing is lower.
    step = float(np.median(np.diff(times)))
    band = max(1.*unit, sum(model[1] for model in models))
    grid = np.linspace(lower, upper, 129)
    agreement = np.abs(np.polyval(incoming[:, 1]-outgoing[:, 1], grid)) <= band
    crossing = grid[agreement]
    crossing = np.r_[crossing, contact_t]
    supported = np.asarray([[np.polyval(model[:, axis], t) for axis in range(2)]
                            for t in crossing for model in (incoming, outgoing)])
    x, y, width, height = peak['bbox_px']
    peak_box = np.array([[x, y], [x+width, y], [x+width, y+height], [x, y+height]])
    fit_outside = np.maximum(np.maximum(peak_box.min(axis=0)-supported, supported-peak_box.max(axis=0)), 0)
    overshoot = float(np.max(np.linalg.norm(fit_outside, axis=1)))
    return {'points': supported, 'residual_px': residual, 'between_frame_overshoot_px': overshoot,
            'crossing_time': float(contact_t+origin),
            'crossing_interval': [float(crossing.min()+origin), float(crossing.max()+origin)],
            'bracketing_frame_interval': [before[-1]['frame_index'], after[0]['frame_index']],
            'bracketing_time_interval': [float(lower+origin), float(upper+origin)],
            'incoming_residual_px': models[0][1], 'outgoing_residual_px': models[1][1],
            'nominal_frame_period_seconds': step,
            '_models': [incoming, outgoing], '_origin': origin,
            'model': 'local quadratic bottom-of-foreground intersection with observed residual band'}


def _artifact(crops, rows, roi, scale, polygon, family_id, output_dir, scale_xy=None):
    axes = np.asarray([scale, scale] if scale_xy is None else scale_xy, dtype=float)
    name = 'native-landing-' + re.sub(r'[^a-zA-Z0-9_-]', '_', str(family_id or 'candidate'))[:80] + '.jpg'
    target = Path(output_dir) / 'evidence' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    chosen = rows[:10]
    sheet = np.full((300*math.ceil(len(chosen)/5), 280*5, 3), (28, 24, 18), np.uint8)
    for index, row in enumerate(chosen):
        frame = crops[row['frame_index']]
        x, y, width, height = row['bbox_px']
        center = (np.array([x+width/2, y+height/2])*axes-np.array(roi[:2]))
        radius = int(max(80, scale*22))
        x0, y0 = np.maximum(0, np.floor(center-radius)).astype(int)
        x1, y1 = np.minimum(np.array(frame.shape[:2][::-1]), np.ceil(center+radius)).astype(int)
        tile = frame[y0:y1, x0:x1].copy()
        box = (np.array([[x, y], [x+width, y+height]])*axes-np.array(roi[:2])-[x0, y0]).astype(int)
        cv2.rectangle(tile, tuple(box[0]), tuple(box[1]), (60, 230, 80), 2)
        if polygon is not None and row.get('peak_frame'):
            points = (np.asarray(polygon)*axes-np.array(roi[:2])-[x0, y0]).astype(np.int32)
            cv2.polylines(tile, [points], True, (0, 165, 255), 2)
        tile = cv2.resize(tile, (280, 250), interpolation=cv2.INTER_AREA)
        left, top = index%5*280, index//5*300
        sheet[top+50:top+300, left:left+280] = tile
        cv2.putText(sheet, f'frame {row["frame_index"]} | {row["time"]:.3f}s', (left+5, top+18),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(sheet, 'VIDEO ESTIMATE / contact unverified', (left+5, top+38),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (100, 200, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError('Could not write native landing evidence')
    return 'evidence/' + name


def estimate_native_landing(family, tracks, source_frames, timestamps, analysis_size,
                            calibration, output_dir=None, config=None):
    """Return source evidence for a conditional video estimate, never IN/OUT.

    Coordinates are analysis pixels. All source frames are read once in
    increasing order; search interpolation never counts as an observation.
    The caller must independently resolve families and compare the complete
    envelope plus uncertainty to the active observed court boundaries.
    """
    result = _base(family)
    if config is not None and (not isinstance(config, dict) or set(config)-{'enabled'}):
        return _reject(result, 'NATIVE_ESTIMATE_INVALID_CONFIG')
    if (config or {}).get('enabled', True) is not True:
        return _reject(result, 'NATIVE_ESTIMATE_DISABLED')
    try:
        times = np.asarray(timestamps, dtype=float)
        width, height = map(int, analysis_size)
        if (times.ndim != 1 or len(times) != len(source_frames) or len(times) < 9
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                or min(width, height) <= 0 or not isinstance(family, dict)):
            return _reject(result, 'NATIVE_ESTIMATE_INVALID_INPUT')
        if not isinstance(calibration, dict):
            return _reject(result, 'NATIVE_ESTIMATE_INVALID_INPUT')
        stability = calibration.get('camera_stability') or {}
        if stability.get('status') != 'ok' or stability.get('all_frames_sampled') is not True:
            return _reject(result, 'NATIVE_ESTIMATE_CAMERA_UNVERIFIED')
        camera_error = stability.get('max_displacement_px', 0)
        if not _finite(camera_error) or camera_error < 0:
            return _reject(result, 'NATIVE_ESTIMATE_CAMERA_UNVERIFIED')
        camera_error = float(camera_error)
        members = family.get('members') or [family.get('representative')]
        members = [item for item in members if isinstance(item, dict) and isinstance(item.get('frame_index'), int)]
        if not members:
            return _reject(result, 'NATIVE_ESTIMATE_NO_FAMILY_MEMBERS')
        center = int(np.median([item['frame_index'] for item in members]))
        if not 4 <= center < len(times)-5:
            return _reject(result, 'NATIVE_ESTIMATE_TEMPORAL_WINDOW_INCOMPLETE')
        unit = width/960.
        observed = _observations({**family, 'members': members}, tracks, times)
        before = [row for row in observed if 0 < times[center]-row['time'] <= .4]
        after = [row for row in observed if 0 < row['time']-times[center] <= .2]
        if len(before) < 6 or len(after) < 3:
            return _reject(result, 'NATIVE_ESTIMATE_FLIGHT_OBSERVATIONS_INSUFFICIENT')
        if any(row['occluder'] for row in before[-3:]+after[:3]):
            return _reject(result, 'NATIVE_ESTIMATE_OCCLUDER_OVERLAP')
        if (before[-1]['point'][1]-before[0]['point'][1] < 20*unit
                or np.mean(np.diff([row['point'][1] for row in before]) > 0) < .8
                or after[0]['point'][1]-after[2]['point'][1] < 3*unit):
            return _reject(result, 'NATIVE_ESTIMATE_DESCENT_REBOUND_UNSUPPORTED')
        if max(row['spread'] for row in before[-3:]+after[:3]) > 55*unit:
            return _reject(result, 'NATIVE_ESTIMATE_TRACKS_DISAGREE')
        if any(right['frame_index']-left['frame_index'] > 2 for left, right in zip(before[-6:], before[-5:])):
            return _reject(result, 'NATIVE_ESTIMATE_INCOMING_GAP')
        support_indices = list(range(center-4, center+6))
        all_indices = list(range(max(0, center-9), min(len(times), center+10)))
        by_frame = {row['frame_index']: row for row in observed}
        observation_indices = np.array([row['frame_index'] for row in observed])
        observation_points = np.asarray([row['point'] for row in observed])
        seeds = np.column_stack([np.interp(support_indices, observation_indices, observation_points[:, axis]) for axis in range(2)])
        first = source_frames[all_indices[0]]
        if not isinstance(first, np.ndarray) or first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
            return _reject(result, 'NATIVE_ESTIMATE_SOURCE_INVALID')
        sh, sw = first.shape[:2]
        sx, sy = sw/width, sh/height
        if min(sx, sy) < .99 or abs(sx/sy-1) > .02:
            return _reject(result, 'NATIVE_ESTIMATE_SOURCE_SCALE_INVALID')
        # Source and analysis sizes can differ by rounding along either axis.
        # Average scale is only a morphology/size heuristic, never a coordinate transform.
        scale = (sx+sy)/2
        scale_xy = np.array([sx, sy], dtype=float)
        area_scale = sx*sy
        # Search every actual branch, including alternatives far from the
        # median. The old candidate union is a search prior, never contact proof.
        search_extent = np.vstack([seeds]+[by_frame[index]['points'] for index in support_indices if index in by_frame])
        lower = np.floor((search_extent.min(axis=0)-22*unit)*scale_xy).astype(int)
        upper = np.ceil((search_extent.max(axis=0)+22*unit)*scale_xy).astype(int)
        if np.any(lower < 0) or np.any(upper > [sw, sh]):
            return _reject(result, 'NATIVE_ESTIMATE_SOURCE_ROI_CLIPPED')
        roi = [int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1])]
        area = int(np.prod(upper-lower))
        if area*len(all_indices)*4 > _MAX_CROP_BYTES:
            return _reject(result, 'NATIVE_ESTIMATE_CROP_BUDGET_EXCEEDED')
        crops = {}
        for index in all_indices:
            frame = first if index == all_indices[0] else source_frames[index]
            if (not isinstance(frame, np.ndarray) or frame.shape != first.shape or frame.dtype != np.uint8):
                return _reject(result, 'NATIVE_ESTIMATE_SOURCE_SHAPE_CHANGED')
            crops[index] = frame[roi[1]:roi[3], roi[0]:roi[2]].copy()
        result.update(decoded_frame_count=len(crops), source_roi=roi, source_scale=scale,
                      source_scale_xy=scale_xy.tolist(),
                      source_size={'width': sw, 'height': sh}, analysis_size={'width': width, 'height': height})
        gray = {index: cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) for index, crop in crops.items()}
        background = np.median(np.stack(list(gray.values())), axis=0).astype(np.uint8)
        rows, alternatives = [], {}
        for index, seed in zip(support_indices, seeds):
            search_points = by_frame[index]['points'] if index in by_frame else np.asarray([seed])
            choices = _component(gray[index], background, search_points*scale_xy-lower, scale, unit, scale_xy)
            if not choices:
                continue
            converted = []
            for component in choices:
                x, y, box_width, box_height = component['bbox']
                converted.append({'frame_index': index, 'time': float(times[index]),
                         'observed_search_seed': index in by_frame,
                         'bbox_px': [(x+roi[0])/sx, (y+roi[1])/sy, box_width/sx, box_height/sy],
                         'foreground_area_px': component['area']/area_scale,
                         'positive_area_px': component['positive']/area_scale,
                         'negative_area_px': component['negative']/area_scale,
                         'contrast_centroid_separation_px': component['separation'],
                         'shape_support': component['shape_support'],
                         'matching_search_seed_count': component['matching_search_seeds'],
                         'search_seed_count': component['search_seed_count'],
                         'track_seed_spread_px': by_frame[index]['spread'] if index in by_frame else None,
                         'source_crop_sha256': hashlib.sha256(crops[index].tobytes()).hexdigest()})
            # Ambiguous frames must not fit their own disambiguation model.
            # Preserve them until independent source frames establish motion.
            if len(converted) > 1 and (converted[1]['foreground_area_px'] > converted[0]['foreground_area_px']*.6
                                      or any(row['shape_support'] for row in converted[1:])):
                alternatives[index] = converted
            else:
                rows.append(converted[0])
        result.update(support_frames=rows, observed_frame_count=len(rows))
        possible_peaks = [row for row in rows if abs(row['frame_index']-center) <= 1]
        if not possible_peaks:
            return _reject(result, 'NATIVE_ESTIMATE_NO_VISIBLE_PEAK')
        peak = max(possible_peaks, key=lambda row: row['bbox_px'][1]+row['bbox_px'][3])
        peak['peak_frame'] = True
        before = [row for row in rows if row['frame_index'] < peak['frame_index']]
        after = [row for row in rows if row['frame_index'] > peak['frame_index']]
        if len(before) < 3 or len(after) < 3 or sum(row['shape_support'] for row in after[:4]) < 2:
            return _reject(result, 'NATIVE_ESTIMATE_COMPACT_OBJECT_SUPPORT_INSUFFICIENT')
        if (peak['bbox_px'][1]+peak['bbox_px'][3] <= before[-1]['bbox_px'][1]+before[-1]['bbox_px'][3]+2*unit
                or peak['bbox_px'][1]+peak['bbox_px'][3] <= after[1]['bbox_px'][1]+after[1]['bbox_px'][3]+2*unit):
            return _reject(result, 'NATIVE_ESTIMATE_NATIVE_REVERSAL_UNSUPPORTED')
        fit = _fit_contact(before, after, peak, times, unit)
        if fit is None:
            return _reject(result, 'NATIVE_ESTIMATE_MOTION_INTERSECTION_UNSUPPORTED')
        alternative_audit = []
        rejected_body_components = []
        for index, choices in sorted(alternatives.items()):
            model = fit['_models'][0 if index < peak['frame_index'] else 1]
            expected = np.array([np.polyval(model[:, axis], times[index]-fit['_origin']) for axis in range(2)])
            tolerance = max(3*unit, 3*fit['residual_px'])
            matched = []
            for row in choices:
                x, y, bw, bh = row['bbox_px']
                distance = float(np.linalg.norm(np.array([x+bw/2, y+bh])-expected))
                row['independent_motion_residual_px'] = distance
                if distance <= tolerance:
                    matched.append(row)
                elif row['shape_support']:
                    rejected_body_components.append({'frame_index': index, 'bbox_px': row['bbox_px'],
                                                     'model_point': expected.tolist()})
            alternative_audit.append({'frame_index': index, 'model_tolerance_px': tolerance,
                                      'matching_component_count': len(matched), 'components': choices})
            if len(matched) != 1:
                result['alternative_component_audit'] = alternative_audit
                return _reject(result, 'NATIVE_ESTIMATE_MULTIPLE_OR_UNRESOLVED_SOURCE_COMPONENTS')
            rows.append(matched[0])
        # Isolated segmentation fragments and floor illumination can both
        # have bright/dark pixels. They are not observations of one second
        # object until temporal and spatial association supports a path.
        # As with incoming flight support, allow at most one missing frame.
        secondary_links = []
        for left_index, left in enumerate(rejected_body_components):
            for right in rejected_body_components[left_index+1:]:
                frame_gap = right['frame_index']-left['frame_index']
                if not 1 <= frame_gap <= 2:
                    continue
                a, b = np.asarray(left['bbox_px']), np.asarray(right['bbox_px'])
                a_center, b_center = a[:2]+a[2:]/2, b[:2]+b[2:]/2
                displacement = float(np.linalg.norm(b_center-a_center))
                local_motion = float(np.linalg.norm(np.asarray(right['model_point'])-left['model_point']))
                # Object extent and independently measured local flight motion
                # bound association; total raw component count cannot do so.
                association_limit = (local_motion + .5*(np.linalg.norm(a[2:])+np.linalg.norm(b[2:]))
                                     + 2*max(unit, fit['residual_px']))
                if displacement <= association_limit:
                    secondary_links.append({'frame_interval': [left['frame_index'], right['frame_index']],
                                            'component_center_displacement_px': displacement,
                                            'association_limit_px': float(association_limit)})
        result['secondary_component_links'] = secondary_links
        if secondary_links:
            result['alternative_component_audit'] = alternative_audit
            return _reject(result, 'NATIVE_ESTIMATE_SECOND_OBJECT_SUPPORT')
        rows.sort(key=lambda row: row['frame_index'])
        result.update(support_frames=rows, observed_frame_count=len(rows),
                      alternative_component_audit=alternative_audit,
                      hypothesis_provenance='All observed branch seeds searched; independent native foreground replaces unsupported candidate centers.')
        x, y, box_width, box_height = peak['bbox_px']
        measured = np.array([[x, y], [x+box_width, y+box_height]])
        lower_point = np.minimum(measured.min(axis=0), fit['points'].min(axis=0))
        upper_point = np.maximum(measured.max(axis=0), fit['points'].max(axis=0))
        polygon = [[lower_point[0], lower_point[1]], [upper_point[0], lower_point[1]],
                   [upper_point[0], upper_point[1]], [lower_point[0], upper_point[1]]]
        uncertainty = fit['residual_px'] + camera_error + max(1/sx, 1/sy, .75*unit)
        result.update(status='supported', reason_codes=['NATIVE_FOREGROUND_AND_LOCAL_REBOUND_SUPPORTED'],
                      possible_contact_polygon_px=polygon, uncertainty_px=float(uncertainty),
                      frame_interval=fit['bracketing_frame_interval'], time_interval=fit['bracketing_time_interval'],
                      estimated_time_seconds=fit['crossing_time'],
                      temporal_model={key: value for key, value in fit.items() if key != 'points' and not key.startswith('_')},
                      uncertainty_components={'local_motion_residual_px': fit['residual_px'],
                                              'background_motion_px': camera_error,
                                              'foreground_quantization_px': max(1/sx, 1/sy, .75*unit)},
                      envelope_method='visible_peak_component_union_kinematic_crossing_band')
        if output_dir is not None:
            result['artifact'] = _artifact(crops, rows, roi, scale, polygon, result['family_id'], output_dir, scale_xy)
        return result
    except (ValueError, TypeError, KeyError, IndexError, cv2.error, np.linalg.LinAlgError) as error:
        result['error'] = type(error).__name__ + ': ' + str(error)[:240]
        return _reject(result, 'NATIVE_ESTIMATE_INVALID_OR_UNAVAILABLE_EVIDENCE')
