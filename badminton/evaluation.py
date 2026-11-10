"""Evaluate predictions against independently reviewed annotations.

Missing labels yield unavailable metrics, never invented zero-error accuracy.
Machine proposals and UNKNOWN labels cannot serve as their own ground truth.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(v) for v in value)


def _position(obj):
    if _point(obj.get("point_px")):
        return obj["point_px"]
    if _number(obj.get("x")) and _number(obj.get("y")):
        return [obj["x"], obj["y"]]
    return None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return None
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("Expected 0 <= successes <= total")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _distribution(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    p95 = ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": p95, "max": max(values)}


def _prediction_label(result):
    decision = result.get("decision", {})
    if isinstance(decision, str):
        value = decision
    elif isinstance(decision, dict):
        value = decision.get("decision") or decision.get("label") or decision.get("verdict")
    else:
        value = None
    return value if value in ("IN", "OUT", "REVIEW") else "ERROR"


def _human_verified(annotation):
    return (annotation.get("status") == "verified"
            and annotation.get("source") == "human_review"
            and bool(str(annotation.get("reviewer") or "").strip()))


def _truth_exclusion(annotation):
    if not _human_verified(annotation):
        return "not_verified_human_ground_truth"
    if annotation.get("label", annotation.get("verdict")) not in ("IN", "OUT"):
        return "unknown_ground_truth"
    if annotation.get("contact_observed") is not True:
        return "contact_not_observed"
    timestamp = annotation.get("time", annotation.get("contact_time"))
    if not _number(timestamp) or timestamp < 0 or not _point(annotation.get("contact_px")):
        return "missing_contact_evidence"
    distance = annotation.get("near_line_distance_m")
    if _number(distance) and abs(distance) <= 0.05 and not annotation.get("independent_evidence"):
        return "near_line_without_independent_evidence"
    return None


def _context_exclusion(annotation, result):
    expected_hash = result.get("source_sha256")
    annotated_hash = annotation.get("source_sha256") or annotation.get("source_artifacts", {}).get("video_sha256")
    if expected_hash and annotated_hash and expected_hash != annotated_hash:
        return "annotation_source_hash_mismatch"
    analysis = result.get("analysis", {})
    annotated_size = annotation.get("analysis_size") or annotation.get("analysis") or {}
    width, height = analysis.get("width"), analysis.get("height")
    for dimension in ("width", "height"):
        recorded = annotated_size.get(dimension, annotation.get("analysis_" + dimension))
        if recorded is not None and analysis.get(dimension) is not None and recorded != analysis[dimension]:
            return "annotation_analysis_dimensions_mismatch"
    point = annotation.get("contact_px")
    if _point(point) and _number(width) and _number(height):
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            return "annotation_contact_outside_frame"
    duration = result.get("metadata", {}).get("duration", result.get("metadata", {}).get("duration_seconds"))
    timestamp = annotation.get("time", annotation.get("contact_time"))
    if _number(timestamp) and _number(duration) and timestamp > duration + 0.1:
        return "annotation_contact_outside_timeline"
    return None


def _decision_metrics(records):
    confusion = {truth: {prediction: 0 for prediction in ("IN", "OUT", "REVIEW", "ERROR")}
                 for truth in ("IN", "OUT")}
    for record in records:
        confusion[record["truth"]][record["prediction"]] += 1
    automated = [r for r in records if r["prediction"] in ("IN", "OUT")]
    correct = sum(r["prediction"] == r["truth"] for r in automated)
    reviewed = sum(r["prediction"] == "REVIEW" for r in records)
    return {"ground_truth_cases": len(records), "automatic_decisions": len(automated),
            "correct_automatic_decisions": correct, "wrong_automatic_decisions": len(automated) - correct,
            "automatic_accuracy": _rate(correct, len(automated)),
            "automatic_accuracy_wilson_95ci": wilson_interval(correct, len(automated)),
            "automatic_error_rate": _rate(len(automated) - correct, len(automated)),
            "automatic_coverage": _rate(len(automated), len(records)),
            "review_rate": _rate(reviewed, len(records)),
            "processing_error_rate": _rate(sum(r["prediction"] == "ERROR" for r in records), len(records)),
            "confusion": confusion,
            "available": bool(records),
            "reason": None if records else "No eligible independently reviewed IN/OUT labels.",
            "interpretation": "Accuracy is conditional on automatic IN/OUT decisions; REVIEW coverage is reported separately."}


def _selected_landing(result):
    selected = result.get("selected_landing")
    if isinstance(selected, dict):
        return selected
    candidates = [r for r in result.get("landings", []) if r.get("contact_observed") is True]
    # Selection never depends on the ground truth timestamp or position.
    return max(candidates, key=lambda r: r.get("confidence", 0), default={})


def _load_results(run_dir):
    index_path = run_dir / "index.json"
    errors = []
    records = []

    def attach_detection_artifact(result, result_path):
        artifact = result.get("artifacts", {}).get("detections")
        if "detections" in result or not artifact:
            return
        path = (result_path.parent / artifact).resolve()
        if not path.is_relative_to(run_dir):
            raise ValueError("Detection artifact path escapes run directory")
        try:
            detections = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(detections, list):
                raise ValueError("Detection artifact must contain frame records")
            result["detections"] = detections
        except (OSError, ValueError) as error:
            errors.append({"video_id": result.get("video_id"), "artifact": artifact, "error": str(error)})

    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        entries = index.get("videos", [])
        for entry in entries:
            identifier = entry.get("id") or entry.get("video_id")
            if entry.get("status") in ("failed", "error", "cancelled", "pending", "running"):
                # A prior successful result may remain on disk after a failed
                # reanalysis. The current run index takes precedence.
                error = entry.get("error") or f"Current run entry status is {entry['status']}"
                errors.append({"video_id": identifier, "error": error, "source": "current_run_index"})
                records.append({"video_id": identifier, "file": entry.get("file"),
                                "decision": {"decision": "ERROR"}, "analysis_status": entry["status"]})
                continue
            relative = entry.get("result_path") or f"videos/{identifier}/result.json"
            path = (run_dir / relative).resolve()
            if not path.is_relative_to(run_dir):
                raise ValueError("Result path escapes run directory")
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                result.setdefault("video_id", identifier)
                attach_detection_artifact(result, path)
                records.append(result)
            except (OSError, ValueError) as error:
                errors.append({"video_id": identifier, "error": str(error)})
                records.append({"video_id": identifier, "file": entry.get("file"), "decision": {"decision": "ERROR"}})
    else:
        for path in sorted((run_dir / "videos").glob("*/result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            attach_detection_artifact(result, path)
            records.append(result)
    identifiers = [r.get("video_id") or r.get("id") for r in records]
    if any(identifier is None for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Results require unique video IDs")
    return records, errors


def _matches(truth, prediction, radius):
    """Maximum-cardinality one-to-one spatial matching under a distance gate."""
    adjacency = []
    for target in truth:
        point = _position(target)
        candidates = []
        for j, candidate in enumerate(prediction):
            other = _position(candidate)
            if point is not None and other is not None:
                distance = math.dist(point, other)
                if distance <= radius:
                    candidates.append((distance, j))
        adjacency.append([j for _, j in sorted(candidates)])
    assigned = {}

    def augment(i, seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(truth)):
        augment(i, set())
    return [(i, j) for j, i in sorted(assigned.items())]


def _component_metrics(result_rows, annotations):
    detection_tp = detection_fp = detection_fn = detection_frames = 0
    missing_detection_predictions = 0
    labeled_track_frames = identity_switches = track_matches = 0
    reconnection_cases = correct_reconnections = 0
    calibration_cases = calibration_success = calibration_false_success = calibration_predicted_success = 0
    reprojection_errors = []
    target_cases = target_errors = 0
    for result in result_rows:
        identifier = result.get("video_id") or result.get("id")
        annotation = annotations.get(identifier, {})
        if not _human_verified(annotation) or _context_exclusion(annotation, result):
            continue
        predicted_frames = {row["frame_index"]: row.get("detections", []) for row in result.get("detections", []) if "frame_index" in row}
        track_frames = defaultdict(list)
        for track in result.get("tracks", []):
            for observation in track.get("observations", []):
                track_frames[observation.get("frame_index")].append({**observation, "track_id": track.get("track_id")})
        previous_ids, matched_identity_frames = {}, {}
        for frame in sorted(annotation.get("detection_frames", []), key=lambda r: r.get("frame_index", -1)):
            if frame.get("complete") is not True or not isinstance(frame.get("frame_index"), int):
                continue
            truth = frame.get("objects", [])
            if any(_position(obj) is None for obj in truth):
                continue
            radius = frame.get("matching_radius_px", 8.0)
            if not _number(radius) or radius <= 0:
                continue
            if "detections" in result:
                prediction = predicted_frames.get(frame["frame_index"], [])
                matches = _matches(truth, prediction, radius)
                detection_tp += len(matches)
                detection_fn += len(truth) - len(matches)
                detection_fp += len(prediction) - len(matches)
                detection_frames += 1
            else:
                missing_detection_predictions += 1
            identified = [obj for obj in truth if obj.get("track_id") is not None]
            if identified:
                labeled_track_frames += 1
                predicted_tracks = track_frames.get(frame["frame_index"], [])
                for i, j in _matches(identified, predicted_tracks, radius):
                    target_id, predicted_id = str(identified[i]["track_id"]), predicted_tracks[j]["track_id"]
                    if predicted_id is None:
                        continue
                    if target_id in previous_ids and previous_ids[target_id] != predicted_id:
                        identity_switches += 1
                    previous_ids[target_id] = predicted_id
                    matched_identity_frames[(target_id, frame["frame_index"])] = predicted_id
                    track_matches += 1
        for episode in annotation.get("reconnection_episodes", []):
            target_id = str(episode.get("track_id"))
            before = matched_identity_frames.get((target_id, episode.get("before_frame")))
            after = matched_identity_frames.get((target_id, episode.get("after_frame")))
            if before is not None and after is not None:
                reconnection_cases += 1
                correct_reconnections += before == after
        reference = annotation.get("calibration_reference", {})
        if isinstance(reference.get("valid"), bool):
            calibration_cases += 1
            predicted = result.get("calibration", {})
            is_auto = predicted.get("method") != "manual_reference" and predicted.get("source") != "manual_reference"
            success = predicted.get("status") == "ok" and is_auto
            calibration_predicted_success += success
            calibration_success += success and reference["valid"]
            calibration_false_success += success and not reference["valid"]
            matrix = predicted.get("homography")
            if success and reference["valid"] and reference.get("independent_reference") is True and matrix:
                for point in reference.get("reference_points", []):
                    image, court = point.get("image_px"), point.get("court_m")
                    if not _point(image) or not _point(court):
                        continue
                    try:
                        vector = [sum(matrix[i][j] * [*image, 1][j] for j in range(3)) for i in range(3)]
                        if abs(vector[2]) > 1e-12:
                            reprojection_errors.append(math.dist([vector[0] / vector[2], vector[1] / vector[2]], court) * 1000)
                    except (IndexError, TypeError):
                        pass
        if annotation.get("target_track_id") is not None:
            selected = _selected_landing(result)
            mapping = annotation.get("predicted_track_identity", {})
            predicted_target = mapping.get(str(selected.get("track_id")))
            if predicted_target is not None:
                target_cases += 1
                target_errors += str(predicted_target) != str(annotation["target_track_id"])
    return {
        "detection": {"available": detection_frames > 0, "exhaustively_labeled_frames": detection_frames,
                      "frames_excluded_missing_predictions": missing_detection_predictions,
                      "true_positives": detection_tp, "false_positives": detection_fp, "false_negatives": detection_fn,
                      "precision": _rate(detection_tp, detection_tp + detection_fp),
                      "recall": _rate(detection_tp, detection_tp + detection_fn),
                      "false_positives_per_frame": _rate(detection_fp, detection_frames),
                      "reason": None if detection_frames else "Requires complete per-frame human object annotations; partial training patches are not evaluation truth."},
        "tracking": {"available": labeled_track_frames > 0, "labeled_frames": labeled_track_frames,
                     "matched_track_observations": track_matches, "id_switches": identity_switches if labeled_track_frames else None,
                     "labeled_reconnection_cases": reconnection_cases,
                     "reconnection_accuracy": _rate(correct_reconnections, reconnection_cases),
                     "reconnection_reason": None if reconnection_cases else "Requires separately labeled occlusion/reconnection episodes and matched observations before/after; ID switches alone do not measure reconnection accuracy."},
        "calibration": {"available": calibration_cases > 0, "labeled_cases": calibration_cases,
                        "correct_automatic_successes": calibration_success,
                        "success_rate": _rate(calibration_success, calibration_cases),
                        "false_success_rate": _rate(calibration_false_success, calibration_predicted_success),
                        "independent_court_reprojection_error_mm": _distribution(reprojection_errors)},
        "target_identity": {"available": target_cases > 0, "labeled_cases": target_cases,
                            "wrong_target_rate": _rate(target_errors, target_cases)}}


def evaluate_run(run_dir, annotations_dir=None) -> dict:
    """Return metrics; writing/report publication remains the caller's choice."""
    run_dir = Path(run_dir).resolve()
    annotations_dir = Path(annotations_dir).resolve() if annotations_dir else run_dir / "annotations"
    results, result_errors = _load_results(run_dir)
    annotations, annotation_errors = {}, []
    for path in sorted(annotations_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            identifier = row.get("video_id")
            if not identifier or identifier in annotations:
                raise ValueError("Missing or duplicate annotation video_id")
            annotations[identifier] = row
        except (OSError, ValueError) as error:
            annotation_errors.append({"file": path.name, "error": str(error)})
    exclusions = Counter()
    eligible, boundary = [], []
    time_errors, position_errors = [], []
    position_exclusions = Counter()
    decision_counts = Counter(_prediction_label(result) for result in results)
    for result in results:
        identifier = result.get("video_id") or result.get("id")
        annotation = annotations.get(identifier)
        if annotation is None:
            exclusions["no_annotation"] += 1
            continue
        exclusion = _context_exclusion(annotation, result) or _truth_exclusion(annotation)
        if exclusion:
            exclusions[exclusion] += 1
            continue
        truth = annotation.get("label", annotation.get("verdict"))
        record = {"video_id": identifier, "truth": truth, "prediction": _prediction_label(result)}
        eligible.append(record)
        distance = annotation.get("near_line_distance_m")
        if _number(distance) and abs(distance) <= 0.05:
            boundary.append(record)
        landing = _selected_landing(result)
        predicted_time = landing.get("time")
        truth_time = annotation.get("time", annotation.get("contact_time"))
        if landing.get("contact_observed") is True and _number(predicted_time):
            time_errors.append(abs(predicted_time - truth_time) * 1000)
        calibration = result.get("calibration", {})
        decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
        predicted_position = decision.get("court_position_m") or landing.get("court_position_m")
        truth_position = annotation.get("court_position_m")
        same_court = annotation.get("court_id") and annotation["court_id"] == (calibration.get("court_id") or result.get("court_id"))
        if (landing.get("contact_observed") is True and _point(predicted_position) and _point(truth_position)
                and calibration.get("status") == "ok" and same_court
                and annotation.get("coordinate_system") == "badminton_court_m"
                and annotation.get("position_reference_verified") is True):
            position_errors.append(math.dist(predicted_position, truth_position) * 1000)
        else:
            position_exclusions["missing_compatible_independent_position_reference_or_observed_contact"] += 1
    total = len(results)
    known_ids = {r.get("video_id") or r.get("id") for r in results}
    components = _component_metrics(results, annotations)
    durations = []
    for result in results:
        timings = result.get("timings", {})
        elapsed = timings.get("total_seconds", timings.get("total"))
        if _number(elapsed):
            durations.append(elapsed)
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Real-video predictions compared only with eligible independently reviewed annotations. Synthetic unit tests are not real-world accuracy evidence.",
            "video_count": total, "annotation_count": len(annotations),
            "verified_human_annotation_count": sum(_human_verified(a) for key, a in annotations.items() if key in known_ids),
            "eligible_ground_truth_count": len(eligible), "excluded_ground_truth_counts": dict(exclusions),
            "annotations_without_matching_result": sorted(set(annotations) - known_ids),
            "result_read_errors": result_errors, "annotation_read_errors": annotation_errors,
            "operational": {"decision_counts": dict(decision_counts),
                            "automatic_coverage": _rate(decision_counts["IN"] + decision_counts["OUT"], total),
                            "review_rate": _rate(decision_counts["REVIEW"], total),
                            "processing_error_rate": _rate(decision_counts["ERROR"], total),
                            "clip_analysis_seconds": _distribution(durations),
                            "ground_truth_required": False},
            "decision": _decision_metrics(eligible), "near_line_within_5cm": _decision_metrics(boundary),
            "landing": {"contact_time_absolute_error_ms": _distribution(time_errors),
                        "contact_position_absolute_error_mm": _distribution(position_errors),
                        "position_exclusion_counts": dict(position_exclusions)},
            **components,
            "ground_truth_available": bool(eligible),
            "accuracy_verified": any(r["prediction"] in ("IN", "OUT") for r in eligible),
            "performance_acceptance_established": False,
            "acceptance_reason": "A measured score alone does not establish the target performance; independent venue/device holdout, sample size, coverage and confidence intervals must also be reviewed.",
            "ground_truth_policy": ["Machine drafts, model predictions and UNKNOWN labels are excluded from decision accuracy.",
                                    "Verified IN/OUT labels require a named human reviewer and observed physical-contact evidence.",
                                    "Near-line cases within 5 cm require independent evidence; the same low-frame-rate video alone is insufficient.",
                                    "Review rate is reported together with automatic accuracy to expose abstention.",
                                    "Millimeter errors are unavailable without compatible independently verified metric coordinates."]}
