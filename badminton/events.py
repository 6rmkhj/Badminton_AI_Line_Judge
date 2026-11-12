"""Offline landing-candidate search with explicit limits on physical evidence.

The image centroid of a small moving object is not a measured cork/floor contact.
This module therefore proposes intervals for review and never upgrades a motion
heuristic into an observed first contact. Coordinates stay in the analysis image.
"""
from __future__ import annotations

import math
import copy
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def _value(observation, name, default=0.0):
    try:
        value = float(observation.get(name, default))
        return value if math.isfinite(value) else float(default)
    except (TypeError, ValueError):
        return float(default)


def validate_contact_modes(config=None):
    """Prevent one observation's geometry from inheriting another model's proof."""
    config = config or {}
    if config.get("contact_model_path") and config.get("source_contact_model_path"):
        raise ValueError("Choose either contact_model_path or source_contact_model_path, not both preprocessing modes")


def review_refinement_regions(config, analysis_size, frame_count):
    """Use the detector's same strict pixel/frame contract for review guidance."""
    from .detection import _refinement_regions
    width, height = analysis_size
    return _refinement_regions({"refinement_regions": (config or {}).get("review_refinement_regions", [])},
                               (height, width), frame_count)


def review_region_indices(candidate, regions):
    """Return matching hints, without treating a hint as contact or identity."""
    frame, point = candidate.get("frame_index"), candidate.get("candidate_px")
    if (not isinstance(frame, int) or isinstance(frame, bool)
            or not isinstance(point, (list, tuple)) or len(point) != 2
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) for value in point)):
        return []
    return [i for i, region in enumerate(regions)
            if region["frame_start"] <= frame <= region["frame_end"]
            and region["bbox"][0] <= point[0] < region["bbox"][0] + region["bbox"][2]
            and region["bbox"][1] <= point[1] < region["bbox"][1] + region["bbox"][3]]


def prioritize_review_candidates(candidates, regions, *, identity_first=False):
    """Order the finite evidence budget; retain every supplied candidate/score."""
    def key(candidate):
        scores = (-_value(candidate, "identity_confidence"), -_value(candidate, "confidence"))
        if not identity_first:
            scores = (scores[1],)
        requested = candidate.get('identity_source') == 'operator_selected_track' or candidate.get('request_match') == 'selected'
        support = candidate.get('flight_trajectory_support') or {}
        if not isinstance(support, dict):
            support = {}
        flight = support.get('supported_for_review') is True
        observed = (candidate.get('contact_observed') is True
                    and candidate.get('contact_evidence_source') in {'human_review', 'qualified_learned_contact_localization'})
        # Raw detector brightness/contrast scores cannot decide between a long
        # coherent flight and a high-contrast shirt edge. Rank the measured
        # approach extent, discounted by its path-fit residual, for review only.
        flight_rank = max(0., _value(support, 'approach_displacement_y_px')) / (1 + max(0., _value(support, 'fit_residual_px')))
        motion_event = candidate.get('image_motion_event') or {}
        if isinstance(motion_event, dict) and motion_event.get('point_kind') == 'inferred_object_center':
            flight_rank *= .7
        return (not bool(review_region_indices(candidate, regions)), not requested, not observed, not flight,
                -flight_rank if flight else 0., *scores)
    return sorted(candidates, key=key)


def _flight_review_support(track, observations, index, image_width, nominal_step):
    """Measure a dedicated trajectory's approach; never qualify its identity.

    Dense body/clothing proposals can reverse direction too. A separate flight
    hypothesis must also retain a sustained, well-fit incoming image path before
    its rebound can consume the limited first-contact review evidence budget.
    """
    if not isinstance(track.get('flight_hypothesis'), dict):
        return None
    window = max(9, min(64, int(math.ceil(.35 / nominal_step))))
    before = observations[max(0, index - window + 1):index + 1]
    if len(before) < 5 or before[-1]['time'] - before[0]['time'] < .12:
        return None
    times = np.asarray([item['time'] - before[-1]['time'] for item in before])
    points = np.asarray([[item['x'], item['y']] for item in before])
    coefficients = np.polyfit(times, points, 2)
    fitted = np.column_stack([np.polyval(coefficients[:, axis], times) for axis in range(2)])
    residual = float(np.sqrt(np.mean(np.sum((points - fitted) ** 2, axis=1))))
    drop = float(points[-1, 1] - points[0, 1])
    increments = np.diff(points[:, 1])
    radius = max(1., _value(observations[index], 'radius', 2.))
    if (residual > max(3., radius * 1.5) or drop < max(8., image_width * .02)
            or float(np.mean(increments > 0)) < .8):
        return None
    return {'supported_for_review': True, 'method': 'coherent_approach_and_image_rebound',
            'approach_observation_count': len(before), 'fit_residual_px': round(residual, 4),
            'approach_displacement_y_px': round(drop, 4),
            'approach_frame_interval': [before[0]['frame_index'], before[-1]['frame_index']],
            'verified_identity': False, 'physical_contact_verified': False}


def _occluded_rebound_candidates(track, observations, timestamps, frames, calibration, nominal_step):
    """Infer a bounded image-trajectory intersection across a missing interval.

    Only dedicated coherent-flight hypotheses enter this path. Fits never fill
    in track observations and cannot assert measured cork/floor contact.
    """
    if not isinstance(track.get('flight_hypothesis'), dict) or len(observations) < 8 or not len(timestamps):
        return []
    all_times = np.asarray([item['time'] for item in observations])
    gaps = np.flatnonzero(np.diff(all_times) > nominal_step * 2.5)
    proposals = []
    for gap_index in gaps:
        left, right = observations[gap_index], observations[gap_index + 1]
        duration = right['time'] - left['time']
        if duration > .35:
            continue
        start = max([int(i) + 1 for i in gaps if i < gap_index] or [0])
        stop = min([int(i) + 1 for i in gaps if i > gap_index] or [len(observations)])
        before = observations[max(start, gap_index - 9):gap_index + 1]
        after = observations[gap_index + 1:min(stop, gap_index + 7)]
        if len(before) < 5 or len(after) < 3 or before[-1]['time'] - before[0]['time'] < .12:
            continue
        origin = left['time']

        def fit(items):
            times = np.asarray([item['time'] - origin for item in items])
            points = np.asarray([[item['x'], item['y']] for item in items])
            coefficients = np.polyfit(times, points, 2)
            fitted = np.column_stack([np.polyval(coefficients[:, axis], times) for axis in range(2)])
            residual = float(np.sqrt(np.mean(np.sum((points - fitted) ** 2, axis=1))))
            return coefficients, residual

        incoming, before_error = fit(before)
        outgoing, after_error = fit(after)
        radius = max(1., _value(left, 'radius', 2.), _value(right, 'radius', 2.))
        tolerance = max(3., radius * 1.5)
        drop = before[-1]['y'] - before[0]['y']
        width = frames[0].shape[1] if frames else 0
        if (max(before_error, after_error) > tolerance or drop < max(8., width * .015)
                or float(np.mean(np.diff([item['y'] for item in before]) > 0)) < .8):
            continue
        for root in sorted(np.roots(incoming[:, 1] - outgoing[:, 1]), key=lambda value: value.real):
            if abs(root.imag) > 1e-6 or not 0 < root.real < duration:
                continue
            offset = float(root.real)
            incoming_speed = float(np.polyval(np.polyder(incoming[:, 1]), offset))
            outgoing_speed = float(np.polyval(np.polyder(outgoing[:, 1]), offset))
            if incoming_speed <= 12 or outgoing_speed >= -12:
                continue
            from_before = np.asarray([np.polyval(incoming[:, axis], offset) for axis in range(2)])
            from_after = np.asarray([np.polyval(outgoing[:, axis], offset) for axis in range(2)])
            if abs(from_before[0] - from_after[0]) > max(4., radius * 2):
                continue
            point = (from_before + from_after) / 2
            if frames and not (0 <= point[0] < frames[0].shape[1] and 0 <= point[1] < frames[0].shape[0]):
                continue
            estimated_time = origin + offset
            index = int(np.argmin(np.abs(np.asarray(timestamps) - estimated_time)))
            actual_context = before[-5:] + after[:4]
            uncertainty = max(radius, tolerance * 2, abs(incoming_speed) * nominal_step / 2,
                              abs(outgoing_speed) * nominal_step / 2)
            reasons = ['OCCLUDED_IMAGE_TRAJECTORY_REBOUND', 'FIRST_CONTACT_NOT_OBSERVED',
                       'CORK_CONTACT_REGION_UNRESOLVED', 'CONTACT_OCCLUDED_OR_MISSING',
                       'TARGET_IDENTITY_UNVERIFIED', 'TEMPORAL_GAP_NEAR_CANDIDATE']
            if track.get('identity_ambiguous'):
                reasons.append('TRACK_IDENTITY_AMBIGUOUS')
            if (calibration or {}).get('status') != 'ok':
                reasons.append('CALIBRATION_NOT_CONFIRMED')
            first = not proposals
            if not first:
                reasons.append('LATER_REBOUND_AFTER_FIRST_CANDIDATE')
            proposals.append({
                'track_id': track.get('track_id'), 'frame_index': index, 'time': float(timestamps[index]),
                'candidate_px': point.tolist(), 'contact_observed': False, 'contact_px': None,
                'contact_radius_px': radius, 'uncertainty_px': round(uncertainty, 3),
                'identity_confidence': .15, 'identity_source': 'unverified_image_trajectory',
                'confidence': round(float(np.mean([_value(item, 'confidence', .5) for item in actual_context])) * .4, 4),
                'confidence_kind': 'uncalibrated_candidate_score', 'target_court_score': 0.,
                'reason_codes': reasons, 'first_contact_missing': True, 'occluded': True,
                'temporal_gap_seconds': duration,
                'evidence_frame_indices': [item['frame_index'] for item in actual_context],
                'backtrack_frame_indices': [item['frame_index'] for item in before],
                'observed_before_seconds': estimated_time - before[0]['time'],
                'observed_after_seconds': after[-1]['time'] - estimated_time,
                'contiguous_observed_before_seconds': before[-1]['time'] - before[0]['time'],
                'contiguous_observed_after_seconds': after[-1]['time'] - after[0]['time'],
                'appearance_evidence': _patch_evidence(frames, index, *point, radius),
                'method': 'offline_occluded_image_trajectory_intersection',
                'image_motion_event': {'kind': 'occluded_descent_and_rebound', 'first_in_track': first,
                    'point_kind': 'inferred_object_center', 'estimated_time': estimated_time,
                    'frame_interval': [left['frame_index'], right['frame_index']],
                    'time_interval': [left['time'], right['time']],
                    'inference_only': True, 'coordinate_system': 'analysis_pixels',
                    'trajectory': [{key: item[key] for key in ('frame_index', 'time', 'x', 'y')}
                                   for item in actual_context]},
                'flight_trajectory_support': {'supported_for_review': first,
                    'method': 'coherent_trajectory_gap_intersection', 'approach_observation_count': len(before),
                    'approach_displacement_y_px': round(float(drop), 4),
                    'fit_residual_px': round(before_error, 4), 'rebound_fit_residual_px': round(after_error, 4),
                    'verified_identity': False, 'physical_contact_verified': False},
            })
            break
    return proposals


def _mark_review_guidance(candidate, regions):
    if regions:
        candidate["review_refinement"] = {"matched_region_indices": review_region_indices(candidate, regions),
                                           "prioritization_only": True}
        candidate["assistance_used"] = True
        candidate["evaluation_eligible"] = False


def _target_support(observations, calibration):
    polygon = calibration.get("polygon_px") if isinstance(calibration, dict) else None
    if polygon is None or len(polygon) < 3:
        return 0.0
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(contour).all():
        return 0.0
    # This is only a 2D image-space consistency cue, never a floor projection.
    inside = [cv2.pointPolygonTest(contour, (o["x"], o["y"]), False) >= 0 for o in observations]
    return float(np.mean(inside)) if inside else 0.0


def _patch_evidence(frames, frame_index, x, y, radius):
    if not frames or frame_index < 0 or frame_index >= len(frames):
        return {"available": False}
    frame = frames[frame_index]
    if frame is None or not getattr(frame, "size", 0):
        return {"available": False}
    height, width = frame.shape[:2]
    half = max(5, int(math.ceil(radius * 3)))
    left, right = max(0, int(x) - half), min(width, int(x) + half + 1)
    top, bottom = max(0, int(y) - half), min(height, int(y) + half + 1)
    if left >= right or top >= bottom:
        return {"available": False}
    patch = frame[top:bottom, left:right]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return {"available": True, "roi_px": [left, top, right - left, bottom - top],
            "mean_luma": round(float(gray.mean()), 3),
            "bright_fraction": round(float((gray >= 190).mean()), 4),
            "contrast_std": round(float(gray.std()), 3),
            "physical_contact_verified": False}


def estimate_landings(tracks, frames, timestamps, calibration, config=None):
    """Return bounded landing *candidates*, preserving uncertainty and identity.

    A candidate is a slowdown followed by persistence, a direction reversal, or
    a moving track disappearing. These cues also occur on hits and occlusion.
    Motion alone leaves ``contact_observed`` false. An optional supervised
    contact model may supply a localized observation only when its held-out
    qualification and capture-domain checks pass. Identity/calibration remain
    separate requirements; human evidence is handled by apply_contact_evidence.
    ``target_track_id`` can identify the requested track but does not prove contact.
    """
    cfg = dict(config or {})
    validate_contact_modes(cfg)
    regions = []
    if cfg.get("review_refinement_regions") != [] and "review_refinement_regions" in cfg:
        if not frames or frames[0] is None:
            raise ValueError("Review refinement requires analysis image bounds")
        regions = review_refinement_regions(cfg, (frames[0].shape[1], frames[0].shape[0]), len(timestamps))
    min_observations = max(3, int(cfg.get("min_observations", 5)))
    min_speed = max(0.0, float(cfg.get("min_motion_speed_px_s", 25.0)))
    stationary_speed = max(0.0, float(cfg.get("stationary_speed_px_s", 18.0)))
    max_candidates = max(0, int(cfg.get("max_candidates", 12)))
    selected_track = cfg.get("target_track_id")
    event_window = cfg.get("event_window_seconds")
    if event_window is not None:
        if (not isinstance(event_window, (list, tuple)) or len(event_window) != 2
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in event_window)
                or not np.isfinite(event_window).all() or event_window[0] < 0 or event_window[1] < event_window[0]):
            raise ValueError("event_window_seconds must be a finite ordered [start, end] interval")
    all_times = np.asarray(timestamps, dtype=float)
    positive_steps = np.diff(all_times)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
    nominal_step = float(np.median(positive_steps)) if len(positive_steps) else 1 / 30
    motion_window = max(2, min(8, int(math.ceil(.06 / nominal_step))))
    plateau_limit = max(2, min(18, int(math.ceil(.08 / nominal_step))))
    candidates = []
    for track in tracks or []:
        observations = []
        for source in track.get("observations", []):
            item = dict(source)
            try:
                item["frame_index"] = int(item["frame_index"])
                item["time"] = float(item["time"])
                item["x"], item["y"] = float(item["x"]), float(item["y"])
            except (ValueError, TypeError, KeyError):
                continue
            if all(math.isfinite(item[k]) for k in ("time", "x", "y")):
                observations.append(item)
        observations.sort(key=lambda o: (o["time"], o["frame_index"]))
        # One observation per timestamp prevents zero-dt velocities.
        observations = list({o["time"]: o for o in observations}.values())
        if len(observations) < min_observations:
            continue
        points = np.array([[o["x"], o["y"]] for o in observations], dtype=float)
        times = np.array([o["time"] for o in observations], dtype=float)
        dt = np.diff(times)
        if np.any(dt <= 0):
            continue
        velocity = np.diff(points, axis=0) / dt[:, None]
        speed = np.linalg.norm(velocity, axis=1)
        if not np.isfinite(speed).all() or float(np.max(speed)) < min_speed:
            continue
        hypotheses = []
        rebounds = []
        rebound_windows = {}
        for i in range(2, len(observations) - 2):
            previous = float(np.median(speed[max(0, i - 3):i]))
            following = float(np.median(speed[i:min(len(speed), i + 3)]))
            if previous >= min_speed and following <= max(stationary_speed, previous * 0.3):
                stable_count = int(np.sum(speed[i:min(len(speed), i + 4)] <= max(stationary_speed, previous * 0.35)))
                if stable_count >= 2:
                    strength = min(0.7, 0.35 + 0.15 * stable_count)
                    hypotheses.append((i, strength, "MOTION_SLOWDOWN_AND_PERSISTENCE"))
            before, after = velocity[i - 1], velocity[i]
            denom = float(np.linalg.norm(before) * np.linalg.norm(after))
            if denom > 0 and speed[i - 1] >= min_speed and speed[i] >= min_speed:
                cosine = float(np.dot(before, after) / denom)
                if cosine < -0.2:
                    hypotheses.append((i, min(0.6, 0.35 + 0.2 * -cosine), "DIRECTION_REVERSAL_CANDIDATE"))
            # A shuttle can keep moving sideways after it rebounds, so the
            # full 2-D velocity need not reverse. Preserve a separate image-y
            # approach/rebound hypothesis, rather than preferring its much
            # later final stop. This is an image-space inference only: a racket
            # hit or a player's movement can also produce this shape.
            before_window = slice(max(0, i - motion_window), i)
            radius = max(1., _value(observations[i], "radius", 2.))
            plateau_end = i
            # A distant shuttle may occupy the same quantized y-coordinate in
            # two frames at the turn. Keep the earliest plateau sample instead
            # of missing the event because neither instantaneous slope reverses.
            while (plateau_end + 1 < len(observations) - 1 and plateau_end - i < plateau_limit
                   and times[plateau_end + 1] - times[i] <= .08
                   and abs(points[plateau_end + 1, 1] - points[i, 1]) <= max(.25, radius * .25)):
                plateau_end += 1
            after_window = slice(plateau_end, min(len(velocity), plateau_end + motion_window))
            approach = float(np.median(velocity[before_window, 1]))
            rebound = float(np.median(velocity[after_window, 1]))
            contiguous = bool(np.all(dt[i - 1:plateau_end + 1] <= nominal_step * 2.5))
            if (contiguous and velocity[i - 1, 1] > 0 and velocity[plateau_end, 1] < 0
                    and approach >= max(12., min_speed * .5)
                    and rebound <= -max(12., min_speed * .5)
                    and points[i, 1] - points[max(0, i - motion_window), 1] >= max(2., radius * 1.5)
                    and points[i, 1] - points[min(len(points) - 1, plateau_end + motion_window), 1] >= max(1.5, radius)):
                # Within one causal track, a later bounce cannot be its first
                # observed approach/rebound. Keep it for review with lower rank.
                rebounds.append(i)
                rebound_windows[i] = plateau_end
                hypotheses.append((i, .88 if len(rebounds) == 1 else .44,
                                   "IMAGE_SPACE_DESCENT_AND_REBOUND"))
        if not hypotheses:
            # Track termination may be landing, a hit, or occlusion. Keep a draft.
            hypotheses.append((len(observations) - 1, 0.2, "TRACK_TERMINATED_WITHOUT_CONTACT"))
        accepted_times = []
        for index, motion_score, cue in sorted(hypotheses, key=lambda entry: -entry[1]):
            observation = observations[index]
            if event_window is not None and not event_window[0] <= observation["time"] <= event_window[1]:
                continue
            if any(abs(observation["time"] - t) < 0.2 for t in accepted_times):
                continue
            accepted_times.append(observation["time"])
            local = observations[max(0, index - 5):min(len(observations), index + 4)]
            target_score = _target_support(local, calibration or {})
            reasons = [cue, "FIRST_CONTACT_NOT_OBSERVED", "CORK_CONTACT_REGION_UNRESOLVED"]
            local_steps = np.diff([o["time"] for o in local])
            has_gap = bool(np.any(local_steps > nominal_step * 2.5))
            if has_gap:
                reasons.append("TEMPORAL_GAP_NEAR_CANDIDATE")
            ambiguous = bool(track.get("identity_ambiguous"))
            if ambiguous:
                reasons.append("TRACK_IDENTITY_AMBIGUOUS")
            if track.get("gaps"):
                reasons.append("TRACK_HAS_MISSING_OBSERVATIONS")
            selected = selected_track is not None and str(track.get("track_id")) == str(selected_track)
            if selected_track is not None and not selected:
                reasons.append("OTHER_TRACK_THAN_REQUESTED")
            if selected:
                identity = 0.9 if not ambiguous else 0.4
                identity_source = "operator_selected_track"
            else:
                identity = min(0.55, 0.15 + target_score * 0.4)
                identity_source = "unverified_image_space_consistency"
                reasons.append("TARGET_IDENTITY_UNVERIFIED")
            if not target_score:
                reasons.append("TARGET_COURT_IMAGE_SUPPORT_MISSING")
            if (calibration or {}).get("status") != "ok":
                reasons.append("CALIBRATION_NOT_CONFIRMED")
            if ambiguous:
                identity = min(identity, 0.25)
            if has_gap:
                identity *= 0.7
            radius = max(1.0, _value(observation, "radius", 2.0))
            nearest_speed = float(speed[min(index, len(speed) - 1)])
            largest_gap = float(np.max(local_steps)) if len(local_steps) else nominal_step
            uncertainty = max(radius, 1.5, nearest_speed * largest_gap / 2)
            if has_gap:
                uncertainty *= 2
            observation_confidence = np.mean([np.clip(_value(o, "confidence", 0.5), 0, 1) for o in local])
            confidence = motion_score * (0.4 + 0.6 * float(observation_confidence))
            if ambiguous or has_gap:
                confidence *= 0.6
            frame_index = observation["frame_index"]
            evidence_indices = sorted({o["frame_index"] for o in local
                                       if 0 <= o["frame_index"] < len(frames)})
            contiguous_start, contiguous_end = index, index
            while contiguous_start > 0 and times[contiguous_start] - times[contiguous_start - 1] <= nominal_step * 2.5:
                contiguous_start -= 1
            while contiguous_end < len(times) - 1 and times[contiguous_end + 1] - times[contiguous_end] <= nominal_step * 2.5:
                contiguous_end += 1
            candidate = {
                "track_id": track.get("track_id"), "frame_index": frame_index,
                "time": observation["time"], "candidate_px": [observation["x"], observation["y"]],
                "contact_observed": False, "contact_px": None,
                "contact_radius_px": radius, "uncertainty_px": round(uncertainty, 3),
                "identity_confidence": round(identity, 4), "identity_source": identity_source,
                "confidence": round(float(confidence), 4), "confidence_kind": "uncalibrated_candidate_score",
                "target_court_score": round(target_score, 4),
                "reason_codes": list(dict.fromkeys(reasons)), "evidence_frame_indices": evidence_indices,
                "backtrack_frame_indices": [o["frame_index"] for o in observations[max(0, index - 20):index + 1]],
                "observed_before_seconds": round(observation["time"] - observations[0]["time"], 4),
                "observed_after_seconds": round(observations[-1]["time"] - observation["time"], 4),
                "contiguous_observed_before_seconds": round(observation["time"] - times[contiguous_start], 4),
                "contiguous_observed_after_seconds": round(times[contiguous_end] - observation["time"], 4),
                "temporal_gap_seconds": round(largest_gap, 6),
                "appearance_evidence": _patch_evidence(frames, frame_index, observation["x"], observation["y"], radius),
                "method": "offline_motion_and_tracklet_candidate_search",
            }
            if cue == "IMAGE_SPACE_DESCENT_AND_REBOUND":
                first = index == rebounds[0]
                plateau_end = rebound_windows[index]
                candidate['image_motion_event'] = {
                    'kind': 'descent_and_rebound', 'first_in_track': first,
                    'point_kind': 'observed_object_center',
                    'frame_interval': [observations[index - 1]['frame_index'], observations[plateau_end + 1]['frame_index']],
                    'time_interval': [observations[index - 1]['time'], observations[plateau_end + 1]['time']],
                    'plateau_frame_interval': [observations[index]['frame_index'], observations[plateau_end]['frame_index']],
                    'inference_only': True, 'coordinate_system': 'analysis_pixels',
                    'trajectory': [{key: item[key] for key in ('frame_index', 'time', 'x', 'y')}
                                   for item in local],
                }
                if not first:
                    candidate['reason_codes'].append('LATER_REBOUND_AFTER_FIRST_CANDIDATE')
                else:
                    support = _flight_review_support(track, observations, index,
                        frames[0].shape[1] if frames else 0, nominal_step)
                    if support is not None:
                        candidate['flight_trajectory_support'] = support
            candidates.append(candidate)
        for candidate in _occluded_rebound_candidates(track, observations, timestamps, frames, calibration, nominal_step):
            if event_window is None or event_window[0] <= candidate['time'] <= event_window[1]:
                if not rebounds or candidate['time'] < observations[rebounds[0]]['time']:
                    candidates.append(candidate)
    candidates.sort(key=lambda item: (-item["confidence"], -item["identity_confidence"], item["time"]))
    # The global cap must not discard a reviewed region before the later native
    # evidence budget can see it. The cap and physical qualification stay intact.
    candidates = prioritize_review_candidates(candidates, regions)
    if regions:
        for candidate in candidates:
            _mark_review_guidance(candidate, regions)
    candidates = candidates[:max_candidates]
    if cfg.get("contact_model_path"):
        for candidate in candidates:
            _apply_contact_model(candidate, frames, timestamps, cfg["contact_model_path"],
                                 cfg.get("frame_preprocessing"))
    return candidates


def _apply_contact_model(candidate, frames, timestamps, checkpoint, preprocessing=None):
    """Keep learned contact proposals separate until the model qualifies them."""
    try:
        from .contact_model import infer_contact
        prediction = infer_contact(frames, timestamps, candidate, checkpoint,
                                   preprocessing=preprocessing)
        if not isinstance(prediction, dict):
            raise ValueError("contact model result must be an object")
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        candidate["contact_model"] = {"status": "failed", "error": str(error)[:1000]}
        candidate["reason_codes"].append("CONTACT_MODEL_ERROR")
        return
    height, width = frames[candidate["frame_index"]].shape[:2]
    _apply_contact_prediction(candidate, prediction, (width, height), preprocessing)


def _apply_contact_prediction(candidate, prediction, image_size, preprocessing):
    from .contact_model import SUPPORTED_PREPROCESSING
    candidate["contact_model"] = prediction
    candidate["model_origin"] = prediction.get("model_origin", "learned_temporal_contact")
    candidate["model_proposed_contact_px"] = prediction.get("proposed_contact_px")
    reasons = prediction.get("reason_codes", [])
    if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
        reasons = ["CONTACT_MODEL_INVALID_REASON_CODES"]
    provenance = prediction.get("provenance")
    preprocessing_matches = (preprocessing in SUPPORTED_PREPROCESSING
                             and isinstance(provenance, dict)
                             and provenance.get("preprocessing") == preprocessing
                             and provenance.get("runtime_preprocessing") == preprocessing)
    if not preprocessing_matches:
        reasons = list(dict.fromkeys(reasons + ["CONTACT_MODEL_PREPROCESSING_MISMATCH"]))
    candidate["reason_codes"] = list(dict.fromkeys(candidate["reason_codes"] + reasons))
    if prediction.get("contact_observed") is not True:
        return
    qualified = (isinstance(prediction.get("provenance"), dict)
                 and prediction["provenance"].get("automatic_contact_qualified") is True
                 and prediction.get("contact_evidence_source") == "qualified_learned_contact_localization"
                 and preprocessing_matches
                 and not reasons)
    try:
        point = [float(value) for value in prediction["contact_px"]]
        radius = float(prediction["contact_radius_px"])
        uncertainty = float(prediction["uncertainty_px"])
        score = float(prediction["confidence"])
        width, height = image_size
        valid_point = (len(point) == 2 and all(math.isfinite(value) for value in point)
                       and 0 <= point[0] < width and 0 <= point[1] < height)
        valid_values = all(math.isfinite(value) and value >= 0 for value in (radius, uncertainty, score)) and score <= 1
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        valid_point, valid_values = False, False
    if not qualified or not valid_point or not valid_values:
        candidate["reason_codes"].append("CONTACT_MODEL_OUTPUT_UNQUALIFIED_OR_INVALID")
        return
    # Only the contact fields are updated. Track identity and court geometry
    # must not inherit the confidence of an unrelated contact recognizer.
    candidate.update({"candidate_confidence": candidate["confidence"],
                      "contact_observed": True, "contact_px": point,
                      "contact_radius_px": radius, "uncertainty_px": uncertainty,
                      "confidence": score,
                      "confidence_kind": prediction.get("confidence_kind", "learned_contact_score"),
                      "contact_evidence_source": prediction["contact_evidence_source"],
                      "inference_origin": "learned_contact_model"})
    resolved = {"FIRST_CONTACT_NOT_OBSERVED", "CORK_CONTACT_REGION_UNRESOLVED"}
    candidate["reason_codes"] = [reason for reason in candidate["reason_codes"] if reason not in resolved]
    candidate["reason_codes"].append("QUALIFIED_LEARNED_CONTACT_OBSERVATION")


def refine_source_contacts(landings, source_frames, timestamps, analysis_size, config=None, output_dir=None):
    """Inspect native temporal ROIs and optionally run a native contact model.

    The fixed five-frame window preserves the original display frame indices.
    Source frames are requested in increasing order exactly once, and only
    bounded patches are retained. A strip or motion/appearance measurement is
    never evidence of observed first contact by itself. Native model outputs
    must pass the same held-out, domain, and provenance gates as legacy models.
    All geometry returned to the decision engine stays in analysis pixels.
    """
    from .contact_model import (SOURCE_PREPROCESSING, contact_checkpoint_metadata,
                                infer_contact_patches, native_contact_patch)
    cfg = dict(config or {})
    validate_contact_modes(cfg)
    width, height = map(int, analysis_size)
    times = np.asarray(timestamps, dtype=float)
    if (min(width, height) <= 0 or times.ndim != 1 or len(times) != len(source_frames)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Native contact analysis requires matching, ordered source timestamps and image bounds")
    patch_size = cfg.get("source_contact_patch_size", 96)
    maximum = cfg.get("source_contact_max_candidates", 6)
    if (not isinstance(patch_size, int) or isinstance(patch_size, bool) or not 24 <= patch_size <= 256 or patch_size % 2
            or not isinstance(maximum, int) or isinstance(maximum, bool) or not 0 <= maximum <= 64):
        raise ValueError("Native contact patch size must be even 24..256; candidate limit must be 0..64")
    results = copy.deepcopy(landings)
    regions = review_refinement_regions(cfg, (width, height), len(times))
    for candidate in results:
        _mark_review_guidance(candidate, regions)
        candidate["source_contact_evidence"] = {"status": "not_requested", "contact_observed": False,
                                                 "reason": "native_candidate_budget", "frames": []}
    positions = {id(candidate): i for i, candidate in enumerate(results)}
    selected = [positions[id(candidate)] for candidate in
                prioritize_review_candidates(results, regions, identity_first=True)[:maximum]]
    checkpoint = cfg.get("source_contact_model_path")
    model_metadata, model_error = None, None
    if checkpoint:
        try:
            model_metadata = contact_checkpoint_metadata(checkpoint)
            if model_metadata.get("preprocessing") != SOURCE_PREPROCESSING:
                raise ValueError("Native contact inference requires a native-source checkpoint; legacy crops are not interchangeable")
        except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            model_error = str(error)[:1000]
            model_metadata = None
    requested, patches, model_patches = defaultdict(list), {}, {}
    for candidate_index in selected:
        candidate = results[candidate_index]
        index, point = candidate.get("frame_index"), candidate.get("candidate_px")
        evidence = {"status": "unavailable", "coordinate_system": "source_pixels",
                    "analysis_size": {"width": width, "height": height}, "patch_size": patch_size,
                    "frames": [], "contact_observed": False,
                    "note": "Native-pixel temporal observations; neither image brightness nor a stop verifies cork/floor contact."}
        if regions:
            evidence.update(review_region_indices=review_region_indices(candidate, regions),
                            selection_reason="explicit_review_region" if review_region_indices(candidate, regions)
                            else "candidate_score_fallback", assistance_used=True, evaluation_eligible=False)
        candidate["source_contact_evidence"] = evidence
        if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(times)
                or not isinstance(point, (list, tuple)) or len(point) != 2
                or not all(isinstance(value, (int, float)) for value in point) or not np.isfinite(point).all()
                or not 0 <= point[0] < width or not 0 <= point[1] < height):
            evidence["reason"] = "invalid_candidate_location_or_frame"
            continue
        indices = list(range(max(0, index - 2), min(len(times), index + 3)))
        evidence.update(frame_indices=indices, times=[float(times[i]) for i in indices])
        patches[candidate_index], model_patches[candidate_index] = [], []
        for frame_index in indices:
            requested[frame_index].append(candidate_index)
    source_size, source_error = None, None
    for frame_index in sorted(requested):
        try:
            source = source_frames[frame_index]
            if source is None or source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
                raise ValueError("Source decoder did not return a uint8 BGR frame")
            sh, sw = source.shape[:2]
            if source_size is not None and source_size != (sw, sh):
                raise ValueError("Source dimensions changed inside the temporal interval")
            source_size = (sw, sh)
            scale = np.array([sw / width, sh / height])
            for candidate_index in requested[frame_index]:
                candidate = results[candidate_index]
                evidence = candidate["source_contact_evidence"]
                point = np.asarray(candidate["candidate_px"]) * scale
                patch = native_contact_patch(source, *point, patch_size)
                patches[candidate_index].append(patch)
                left, top = int(round(float(point[0]))) - patch_size // 2, int(round(float(point[1]))) - patch_size // 2
                roi = [max(0, left), max(0, top), min(sw, left + patch_size), min(sh, top + patch_size)]
                visible = (roi[2] - roi[0]) * (roi[3] - roi[1]) / patch_size ** 2
                gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                evidence.update(source_size={"width": sw, "height": sh}, scale_xy=scale.tolist(),
                                candidate_source_px=point.tolist())
                evidence["frames"].append({"frame_index": frame_index, "time": float(times[frame_index]),
                                           "source_roi": roi, "crop_origin_px": [left, top],
                                           "visible_fraction": round(float(visible), 6),
                                           "contrast_std": round(float(gray.std()), 4),
                                           "sharpness_laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4)})
                if model_metadata is not None:
                    model_patches[candidate_index].append(native_contact_patch(source, *point, model_metadata["patch_size"]))
        except (ValueError, TypeError, RuntimeError, OSError, IndexError, cv2.error) as error:
            source_error = str(error)[:1000]
            break
    for candidate_index in selected:
        candidate = results[candidate_index]
        evidence = candidate["source_contact_evidence"]
        crops = patches.get(candidate_index, [])
        if not crops:
            evidence.setdefault("reason", "source_frames_unavailable")
            if source_error:
                evidence["error"] = source_error
            continue
        complete = len(crops) == 5 and len(evidence["frame_indices"]) == 5
        steps = np.diff(evidence["times"])
        continuous = len(steps) == 4 and float(max(steps)) <= float(np.median(steps)) * 1.75
        evidence["status"] = "complete" if complete and continuous else "incomplete"
        if not complete:
            evidence["reason"] = "first_contact_window_truncated_or_unavailable"
        elif not continuous:
            evidence["reason"] = "timestamp_discontinuity"
        if source_error and not complete:
            evidence["error"] = source_error
        if output_dir is not None:
            relative = f"evidence/source-contact-{candidate_index:03d}.jpg"
            destination = Path(output_dir) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            strip = np.zeros((patch_size + 32, patch_size * len(crops), 3), np.uint8)
            for i, (crop, row) in enumerate(zip(crops, evidence["frames"])):
                strip[:patch_size, i*patch_size:(i+1)*patch_size] = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.putText(strip, f"f{row['frame_index']}", (i*patch_size + 3, patch_size + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1)
                cv2.putText(strip, f"{row['time']:.3f}s", (i*patch_size + 3, patch_size + 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (240, 240, 240), 1)
            if cv2.imwrite(str(destination), strip):
                evidence["image"] = relative
            else:
                evidence["image_error"] = "cannot_write_source_temporal_strip"
        if not checkpoint:
            continue
        if model_error:
            candidate["source_contact_model"] = {"status": "failed", "error": model_error}
            candidate["reason_codes"] = list(dict.fromkeys(candidate.get("reason_codes", []) + ["SOURCE_CONTACT_MODEL_ERROR"]))
            continue
        if evidence["status"] != "complete":
            candidate["source_contact_model"] = {"status": "incomplete_window", "contact_observed": False}
            continue
        try:
            native_candidate = {"frame_index": candidate["frame_index"], "candidate_px": evidence["candidate_source_px"]}
            prediction = infer_contact_patches(model_patches[candidate_index], evidence["times"], native_candidate,
                                                checkpoint, image_size=source_size, preprocessing=SOURCE_PREPROCESSING)
            candidate["source_contact_model"] = {**copy.deepcopy(prediction), "coordinate_system": "source_pixels"}
            scale = np.asarray(evidence["scale_xy"])
            mapped = copy.deepcopy(prediction)
            for key in ("contact_px", "proposed_contact_px"):
                if mapped.get(key) is not None:
                    mapped[key] = (np.asarray(mapped[key]) / scale).tolist()
            for key in ("contact_radius_px", "uncertainty_px"):
                if mapped.get(key) is not None:
                    mapped[key] = float(mapped[key]) / float(min(scale))
            mapped["coordinate_system"] = "analysis_pixels"
            mapped["source_to_analysis_scale_xy"] = (1 / scale).tolist()
            _apply_contact_prediction(candidate, mapped, (width, height), SOURCE_PREPROCESSING)
        except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            candidate["source_contact_model"] = {"status": "failed", "error": str(error)[:1000]}
            candidate["reason_codes"] = list(dict.fromkeys(candidate.get("reason_codes", []) + ["SOURCE_CONTACT_MODEL_ERROR"]))
    return results


def apply_contact_evidence(candidate, evidence, *, video_id, duration_seconds, image_size, timestamps=None):
    """Attach an explicit human observation without calling it model inference.

    The caller must supply bounds from the stored analysis result, not from the
    request body. A person must affirm both first contact and target identity;
    clicking an approximate shuttle centre alone is not sufficient evidence.
    The returned landing may be passed through the rule engine, but must remain
    separate from automatic predictions in accuracy reports.
    """
    if not isinstance(candidate, dict) or not isinstance(evidence, dict):
        raise ValueError("candidate and evidence must be objects")
    if evidence.get("source") != "human_review" or evidence.get("verified") is not True:
        raise ValueError("explicit verified human_review evidence is required")
    if not isinstance(video_id, str) or not video_id or evidence.get("video_id") != video_id:
        raise ValueError("contact evidence belongs to a different video")
    if candidate.get("video_id") is not None and candidate["video_id"] != video_id:
        raise ValueError("candidate belongs to a different video")
    if evidence.get("coordinate_system") != "analysis_pixels":
        raise ValueError("contact coordinates must use analysis_pixels")
    if evidence.get("first_contact_observed") is not True:
        raise ValueError("first physical contact must be explicitly observed")
    if evidence.get("target_identity_confirmed") is not True:
        raise ValueError("target shuttle identity must be explicitly confirmed")
    if evidence.get("track_id") is not None and str(evidence["track_id"]) != str(candidate.get("track_id")):
        raise ValueError("contact evidence belongs to a different track")
    try:
        duration = float(duration_seconds)
        width, height = (int(v) for v in image_size)
        contact_time = float(evidence["time"])
        point = [float(v) for v in evidence["contact_px"]]
        radius = float(evidence["contact_radius_px"])
        uncertainty = float(evidence["uncertainty_px"])
    except (KeyError, ValueError, TypeError, OverflowError) as error:
        raise ValueError("valid time, contact point, radius, uncertainty and trusted bounds are required") from error
    if not math.isfinite(duration) or duration <= 0 or min(width, height) <= 0:
        raise ValueError("trusted video bounds are invalid")
    if not math.isfinite(contact_time) or not 0 <= contact_time <= duration:
        raise ValueError("contact time is outside the video")
    if len(point) != 2 or not all(math.isfinite(value) for value in point):
        raise ValueError("contact_px must contain two finite coordinates")
    if not (0 <= point[0] < width and 0 <= point[1] < height):
        raise ValueError("contact point is outside the analysis image")
    if not all(math.isfinite(value) and 0 <= value <= math.hypot(width, height) for value in (radius, uncertainty)):
        raise ValueError("contact radius and uncertainty must be finite nonnegative image distances")
    reviewed_frame = candidate.get("frame_index") if candidate.get("time") == contact_time else None
    if timestamps is not None:
        trusted_times = np.asarray(timestamps, dtype=float)
        if trusted_times.ndim != 1 or not len(trusted_times) or not np.isfinite(trusted_times).all() or np.any(np.diff(trusted_times) < 0):
            raise ValueError("trusted frame timestamps must be a finite ordered sequence")
        reviewed_frame = int(np.argmin(np.abs(trusted_times - contact_time)))
    result = copy.deepcopy(candidate)
    result.update({"video_id": video_id, "candidate_time": candidate.get("time"),
                   "candidate_frame_index": candidate.get("frame_index"), "frame_index": reviewed_frame,
                   "time": contact_time, "contact_observed": True, "contact_px": point,
                   "contact_radius_px": radius, "uncertainty_px": uncertainty,
                   "identity_confidence": 1.0, "identity_source": "operator_confirmed_target",
                   "confidence": 1.0, "confidence_kind": "operator_asserted_not_model_probability",
                   "contact_evidence_source": "human_review", "inference_origin": "human_review",
                   "evaluation_eligible": False,
                   "evaluation_exclusion_reason": "A manually corrected decision is not an independent automatic prediction.",
                   "manual_review": {"source": "human_review", "verified": True,
                                     "first_contact_observed": True, "target_identity_confirmed": True,
                                     "coordinate_system": "analysis_pixels", "image_size": [width, height],
                                     "nearest_frame_index": reviewed_frame,
                                     "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                                     "reviewer": str(evidence.get("reviewer", "operator"))[:120]}})
    resolved = {"FIRST_CONTACT_NOT_OBSERVED", "CORK_CONTACT_REGION_UNRESOLVED", "TARGET_IDENTITY_UNVERIFIED",
                "TRACK_IDENTITY_AMBIGUOUS", "OTHER_TRACK_THAN_REQUESTED"}
    result["original_reason_codes"] = list(candidate.get("reason_codes", []))
    result["reason_codes"] = [reason for reason in candidate.get("reason_codes", []) if reason not in resolved]
    result["reason_codes"].append("HUMAN_CONFIRMED_CONTACT_EVIDENCE")
    return result
