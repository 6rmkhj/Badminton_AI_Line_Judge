#!/usr/bin/env python3
"""Audit a finalized PoC run against its real, unchanged source videos.

This does not assign ground truth or claim model accuracy. The analysis command
creates the results; this independent pass checks inventory, hashes, timestamps,
decoded previews, source evidence, and implementation provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

# Permit the documented ``python scripts/validate_integrated_poc.py`` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from badminton.media import file_hash, probe
from badminton.pipeline import implementation_fingerprint


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json(path):
    def parse_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite JSON number: {text}")
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant,
                      parse_float=parse_float)


def _path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Artifact requires a nonempty relative path")
    result = (root / relative).resolve()
    if Path(relative).is_absolute() or not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes its artifact directory: {relative}")
    return result


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _times_match(left, right, tolerance=0.0001):
    return (isinstance(left, list) and isinstance(right, list) and len(left) == len(right)
            and all(_finite(a) and _finite(b) and abs(a - b) <= tolerance for a, b in zip(left, right)))


def decoded_frames(path):
    """Count actual OpenCV decoder output, without trusting container nb_frames."""
    import cv2
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or not frame.size:
                raise ValueError("Decoder returned an empty image")
            count += 1
    finally:
        capture.release()
    return count


def _artifact_references(value):
    """Collect only output references; source paths and labels are not artifacts."""
    keys = {"preview", "detections", "overview", "image", "source_crop", "filmstrip",
            "filmstrip_path", "clip", "clip_path", "overlay_video", "temporal_strip",
            "path", "timing_path", "source_sequence"}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys and isinstance(item, str):
                yield item
            elif isinstance(item, (dict, list)):
                yield from _artifact_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _artifact_references(item)


def _audit_one(entry, source_row, video_root, run, expected_fingerprint, decode_source):
    identifier = entry.get("id")
    row = {"id": identifier, "file": entry.get("file"), "status": entry.get("status"),
           "errors": [], "warnings": []}
    fail = row["errors"].append
    if entry.get("status") != "completed":
        fail("ANALYSIS_NOT_COMPLETED")
        row["analysis_error"] = entry.get("error")
        return row
    try:
        result_path = _path(run, entry.get("result_path", f"videos/{identifier}/result.json"))
        result = read_json(result_path)
        folder = result_path.parent
        if not source_row:
            raise ValueError("Index video missing from source manifest")
        source = _path(video_root, source_row["file"])
        observed_hash = file_hash(source)
        row["source_sha256"] = observed_hash
        row["source_hash_matches"] = observed_hash == source_row.get("sha256") == result.get("source_sha256")
        if not row["source_hash_matches"]:
            fail("SOURCE_HASH_MISMATCH")
        if result.get("video_id") != identifier or result.get("file") != source.name:
            fail("RESULT_SOURCE_IDENTITY_MISMATCH")
        fingerprint = result.get("implementation_fingerprint")
        row["implementation_fingerprint"] = fingerprint
        if not fingerprint or fingerprint != expected_fingerprint:
            fail("STALE_OR_MISSING_IMPLEMENTATION_FINGERPRINT")
        source_metadata = probe(source)
        preview = _path(folder, result.get("artifacts", {}).get("preview", "preview.mp4"))
        preview_metadata = probe(preview)
        result_times = result.get("timestamps", [])
        count = result.get("analysis", {}).get("frame_count")
        preview_decoded = decoded_frames(preview)
        source_decoded = decoded_frames(source) if decode_source else None
        row.update(source_presented_frames=source_metadata["frame_count"],
                   preview_presented_frames=preview_metadata["frame_count"],
                   preview_decoded_frames=preview_decoded, source_decoded_frames=source_decoded,
                   analysis_frames=count, source_width=source_metadata["width"],
                   source_height=source_metadata["height"], source_fps=source_metadata["fps"],
                   source_duration_seconds=source_metadata["duration"])
        row["frame_counts_match"] = (count == len(result_times) == source_metadata["frame_count"]
                                     == preview_metadata["frame_count"] == preview_decoded
                                     and (source_decoded is None or source_decoded == count))
        if not row["frame_counts_match"]:
            fail("FRAME_COUNT_MISMATCH")
        row["timestamps_match"] = (_times_match(result_times, source_metadata["timestamps"])
                                    and _times_match(result_times, preview_metadata["timestamps"]))
        if not row["timestamps_match"]:
            fail("PRESENTATION_TIMESTAMP_MISMATCH")
        row["warnings"].extend(source_metadata.get("warnings", []))
        if result_times and preview_metadata["duration"] <= result_times[-1]:
            fail("PREVIEW_FINAL_FRAME_HAS_NO_DISPLAY_DURATION")
        detections = read_json(_path(folder, result.get("artifacts", {}).get("detections", "detections.json")))
        if not isinstance(detections, list) or len(detections) != count:
            fail("DETECTION_FRAME_COUNT_MISMATCH")
            detections = []
        elif any(frame.get("frame_index") != i or not _finite(frame.get("time"))
                 or abs(frame["time"] - result_times[i]) > 0.0001 for i, frame in enumerate(detections)):
            fail("DETECTION_FRAME_ALIGNMENT_MISMATCH")
        references = set(_artifact_references(result.get("artifacts", {})))
        references.update(_artifact_references([landing.get("source_contact_evidence", {})
                                               for landing in result.get("landings", [])]))
        checked = []
        for relative in sorted(references):
            artifact = _path(folder, relative)
            if not artifact.is_file() or artifact.stat().st_size == 0:
                fail(f"MISSING_OR_EMPTY_ARTIFACT:{relative}")
            elif artifact.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                import cv2
                decoded = cv2.imread(str(artifact))
                if decoded is None or not decoded.size:
                    fail(f"UNDECODABLE_IMAGE_ARTIFACT:{relative}")
            elif artifact.suffix.lower() == ".mp4" and artifact != preview:
                if probe(artifact)["frame_count"] <= 0:
                    fail(f"EMPTY_VIDEO_ARTIFACT:{relative}")
            checked.append(relative)
        row["checked_artifacts"] = checked
        annotated = result.get("artifacts", {}).get("annotated_clip")
        if isinstance(annotated, dict):
            replay = probe(_path(folder, annotated["path"]))
            timeline = read_json(_path(folder, annotated["timing_path"])).get("frames", [])
            if len(timeline) != replay["frame_count"] or len(timeline) != annotated.get("frame_count"):
                fail("ANNOTATED_REPLAY_FRAME_COUNT_MISMATCH")
            for frame, actual_time in zip(timeline, replay["timestamps"]):
                index = frame.get("source_frame_index")
                if (not isinstance(index, int) or not 0 <= index < len(result_times)
                        or not _finite(frame.get("source_time")) or not _finite(frame.get("replay_time"))
                        or abs(frame["source_time"] - result_times[index]) > 0.0001
                        or abs(actual_time - frame["replay_time"]) > 0.0001):
                    fail("ANNOTATED_REPLAY_SOURCE_ALIGNMENT_MISMATCH")
        for evidence in result.get("artifacts", {}).get("evidence", []):
            info = evidence.get("source_crop_info", {})
            if info.get("status") == "saved":
                index = info.get("decoded_frame_index")
                if (not isinstance(index, int) or not 0 <= index < len(result_times)
                        or not _finite(info.get("reported_time"))
                        or abs(info["reported_time"] - result_times[index]) > 0.0001):
                    fail("SOURCE_CROP_FRAME_ALIGNMENT_MISMATCH")
                box = info.get("source_box", [])
                if (len(box) != 4 or not all(_finite(value) for value in box)
                        or not (0 <= box[0] < box[2] <= source_metadata["width"]
                                and 0 <= box[1] < box[3] <= source_metadata["height"])):
                    fail("SOURCE_CROP_OUT_OF_BOUNDS")
        decision = result.get("decision", {})
        label = decision.get("decision")
        if label not in {"IN", "OUT", "REVIEW"}:
            fail("INVALID_DECISION")
        calibration = result.get("calibration", {})
        stability = result.get("camera_stability", calibration.get("camera_stability", {}))
        if label in {"IN", "OUT"} and (calibration.get("status") != "ok"
                or stability.get("status") not in (None, "ok")):
            fail("AUTOMATIC_DECISION_WITH_UNACCEPTED_GEOMETRY")
        if label in {"IN", "OUT"}:
            selected = result.get("selected_landing") or {}
            if selected.get("contact_observed") is not True:
                fail("AUTOMATIC_DECISION_WITHOUT_OBSERVED_CONTACT")
            if calibration.get("manual_reference") or calibration.get("method") == "manual_reference":
                fail("AUTOMATIC_DECISION_WITH_REFERENCE_ONLY_CALIBRATION")
        landings = result.get("landings", [])
        for landing in landings:
            model = landing.get("contact_model", {})
            if landing.get("contact_observed") and landing.get("inference_origin") == "learned_contact_model":
                if model.get("provenance", {}).get("automatic_contact_qualified") is not True:
                    fail("CONTACT_CLAIM_WITHOUT_QUALIFIED_MODEL")
        native = [frame.get("high_resolution", {}) for frame in detections]
        size = [source_metadata["width"], source_metadata["height"]]
        analysis = result.get("analysis", {})
        native_expected = (result.get("config", {}).get("detector", {}).get("highres_enabled", True)
                           and source_metadata["width"] > analysis.get("width", source_metadata["width"]))
        if native_expected and any(item.get("status") not in {"analyzed", "onnx_tiled"} for item in native):
            fail("EXPECTED_NATIVE_DETECTION_NOT_EXECUTED")
        for item in native:
            if item.get("status") == "analyzed":
                if item.get("source_size") != size:
                    fail("NATIVE_DETECTION_SOURCE_SIZE_MISMATCH")
                for roi in item.get("rois", []):
                    box = roi.get("bbox", [])
                    if (len(box) != 4 or not all(_finite(value) for value in box)
                            or not (0 <= box[0] < size[0] and 0 <= box[1] < size[1]
                                    and box[2] > 0 and box[3] > 0
                                    and box[0] + box[2] <= size[0] and box[1] + box[3] <= size[1])):
                        fail("NATIVE_DETECTION_ROI_OUT_OF_BOUNDS")
        for landing in landings:
            evidence = landing.get("source_contact_evidence", {})
            if not evidence or evidence.get("status") in {"not_requested", "unavailable"}:
                continue
            if evidence.get("contact_observed") is not False:
                fail("UNVERIFIED_SOURCE_STRIP_MISREPRESENTED_AS_CONTACT")
            if evidence.get("source_size") != {"width": size[0], "height": size[1]}:
                fail("SOURCE_CONTACT_EVIDENCE_SIZE_MISMATCH")
            for frame in evidence.get("frames", []):
                index = frame.get("frame_index")
                if (not isinstance(index, int) or not 0 <= index < len(result_times)
                        or not _finite(frame.get("time"))
                        or abs(frame["time"] - result_times[index]) > 0.0001):
                    fail("SOURCE_CONTACT_FRAME_ALIGNMENT_MISMATCH")
                box = frame.get("source_roi", [])
                if (len(box) != 4 or not all(_finite(value) for value in box)
                        or not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])):
                    fail("SOURCE_CONTACT_ROI_OUT_OF_BOUNDS")
        row.update(decision=label, decision_reasons=decision.get("reason_codes", []),
                   calibration_status=calibration.get("status"),
                   calibration_reasons=calibration.get("reason_codes", []),
                   camera_stability_status=stability.get("status", "not_recorded"),
                   camera_stability_reasons=stability.get("reason_codes", []),
                   highres_frame_records=sum(bool(item) for item in native),
                   highres_status_counts=dict(Counter(item.get("status", "not_recorded") for item in native)),
                   highres_roi_count=sum(item.get("roi_count", 0) for item in native),
                   highres_examined_pixels=sum(item.get("pixels_examined", 0) for item in native),
                   source_contact_candidates=sum(bool(landing.get("source_contact_evidence", {}).get("frames")) for landing in landings),
                   source_contact_frame_count=sum(len(landing.get("source_contact_evidence", {}).get("frames", [])) for landing in landings),
                   source_contact_status_counts=dict(Counter(landing.get("source_contact_evidence", {}).get("status", "not_recorded") for landing in landings)),
                   identity_sources=dict(Counter(landing.get("identity_source", "unknown") for landing in landings)),
                   observed_contact_candidates=sum(landing.get("contact_observed") is True for landing in landings),
                   trained_detector_used=result.get("detector", {}).get("trained_model_provided") is True,
                   detection_quality=result.get("detection_quality", {}),
                   total_analysis_seconds=result.get("timings", {}).get("total_seconds"))
    except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError, subprocess.SubprocessError) as error:
        fail(f"AUDIT_ERROR:{type(error).__name__}:{error}")
    return row


def audit_request_result(path):
    """Check saved request evidence without treating operator input as truth."""
    path = Path(path).resolve()
    row = {"path": str(path), "errors": []}
    try:
        request = read_json(path)
        row["request_id"] = request.get("request_id")
        row["status"] = request.get("status")
        decision = request.get("decision", {})
        label = decision.get("decision")
        row["decision"] = label
        if request.get("ground_truth_created") is not False or request.get("automatic_benchmark_eligible") is not False:
            row["errors"].append("REQUEST_ASSISTANCE_MISCLASSIFIED_AS_AUTOMATIC_GROUND_TRUTH")
        views = request.get("views", [])
        identifiers = [view.get("camera_id") for view in views]
        if len(set(identifiers)) != len(identifiers):
            row["errors"].append("DUPLICATE_REQUEST_CAMERA_ID")
        for view in views:
            if view.get("status") != "completed":
                if label != "REVIEW":
                    row["errors"].append("CONFIRMED_REQUEST_WITH_FAILED_VIEW")
                continue
            result = read_json(_path(path.parent, view["result_path"]))
            if result.get("source_sha256") != view.get("source_sha256"):
                row["errors"].append("REQUEST_VIEW_SOURCE_HASH_MISMATCH")
            window = view.get("event_window_seconds", [])
            if len(window) != 2 or not all(_finite(value) for value in window) or window[0] > window[1]:
                row["errors"].append("INVALID_REQUEST_EVENT_WINDOW")
            elif any(not window[0] <= landing.get("time", -1) <= window[1] for landing in result.get("landings", [])):
                row["errors"].append("REQUEST_LANDING_OUTSIDE_EVENT_WINDOW")
        if label not in {"IN", "OUT", "REVIEW"}:
            row["errors"].append("INVALID_REQUEST_DECISION")
        if label in {"IN", "OUT"}:
            if len(views) > 1 and request.get("same_event_confirmed") is not True:
                row["errors"].append("MULTIVIEW_DECISION_WITHOUT_EVENT_CORRESPONDENCE")
            if len(views) > 1 and request.get("same_court_coordinate_system_confirmed") is not True:
                row["errors"].append("MULTIVIEW_DECISION_WITHOUT_SHARED_COURT_COORDINATES")
            if not decision.get("responsible_camera_ids"):
                row["errors"].append("REQUEST_DECISION_WITHOUT_RESPONSIBLE_CAMERA")
        row["views"] = len(views)
    except (OSError, ValueError, TypeError, KeyError) as error:
        row["errors"].append(f"REQUEST_AUDIT_ERROR:{type(error).__name__}:{error}")
    return row


def audit_run(run_dir, video_dir, *, input_snapshot=None, decode_source=False, workers=2,
              request_results=None):
    run, video_root = Path(run_dir).resolve(), Path(video_dir).resolve()
    index = read_json(run / "index.json")
    manifest = read_json(run / "manifest.json")
    evaluation = read_json(run / "evaluation.json")
    errors = []
    if index.get("status") not in {"completed", "completed_with_failures"}:
        errors.append("RUN_NOT_FINALIZED")
    entries = index.get("videos", [])
    source_rows = manifest.get("videos", [])
    by_id = {row["id"]: row for row in source_rows}
    if len(by_id) != len(source_rows) or len({entry.get("id") for entry in entries}) != len(entries):
        errors.append("DUPLICATE_VIDEO_ID")
    inputs = sorted(path.relative_to(video_root).as_posix() for path in video_root.rglob("*")
                    if path.is_file() and path.suffix.lower() == ".mp4")
    if set(inputs) != {row.get("file") for row in source_rows}:
        errors.append("SOURCE_INVENTORY_MANIFEST_MISMATCH")
    if set(by_id) != {entry.get("id") for entry in entries}:
        errors.append("RUN_DOES_NOT_INCLUDE_ALL_INPUTS")
    if not inputs:
        errors.append("EMPTY_VIDEO_INVENTORY")
    if evaluation.get("video_count") != len(entries):
        errors.append("EVALUATION_VIDEO_COUNT_MISMATCH")
    expected_fingerprint = implementation_fingerprint()
    with ThreadPoolExecutor(max_workers=max(1, min(4, int(workers)))) as pool:
        rows = list(pool.map(lambda entry: _audit_one(entry, by_id.get(entry.get("id")), video_root,
                                                    run, expected_fingerprint, decode_source), entries))
    snapshot_status = {"provided": input_snapshot is not None, "matches": None, "mismatches": []}
    if input_snapshot is not None:
        before = read_json(input_snapshot)
        expected = {row["file"]: row.get("sha256") for row in before.get("videos", [])}
        observed = {row["file"]: row.get("source_sha256") for row in rows}
        snapshot_status["mismatches"] = sorted(name for name in set(expected) | set(observed)
                                                if expected.get(name) != observed.get(name))
        snapshot_status["matches"] = bool(expected) and not snapshot_status["mismatches"]
        if not snapshot_status["matches"]:
            errors.append("INPUTS_CHANGED_SINCE_SNAPSHOT")
    truth = evaluation.get("eligible_ground_truth_count", 0)
    metrics = evaluation.get("decision", {})
    if not truth and any(metrics.get(key) is not None for key in
                         ("automatic_accuracy", "automatic_accuracy_wilson_95ci", "automatic_error_rate")):
        errors.append("ACCURACY_REPORTED_WITHOUT_ELIGIBLE_GROUND_TRUTH")
    fingerprints = Counter(row.get("implementation_fingerprint") for row in rows
                           if row.get("implementation_fingerprint"))
    if len(fingerprints) != 1 or sum(fingerprints.values()) != len(entries):
        errors.append("NONUNIFORM_OR_INCOMPLETE_IMPLEMENTATION_PROVENANCE")
    failed = [row["file"] for row in rows if row["errors"]]
    source_seconds = sum(row.get("source_duration_seconds", 0) for row in rows)
    analysis_seconds = sum(row.get("total_analysis_seconds") or 0 for row in rows)
    decisions = Counter(row.get("decision", "ERROR") for row in rows)
    request_checks = [audit_request_result(path) for path in request_results or []]
    if any(row["errors"] for row in request_checks):
        errors.append("REQUEST_RESULT_AUDIT_FAILED")
    return {
        "schema_version": 1, "generated_at_seoul": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "scope": "Integrated PoC software execution, artifact integrity and source provenance; not ground-truth accuracy or competition readiness.",
        "software_integrity_passed": bool(rows) and not failed and not errors,
        "errors": errors, "failed_files": failed,
        "input_inventory": {"files": len(inputs), "manifest_files": len(source_rows),
                            "unique_content_count": manifest.get("unique_content_count"),
                            "unchanged_since_snapshot": snapshot_status},
        "execution": {"indexed_videos": len(entries), "completed_videos": sum(row.get("status") == "completed" for row in rows),
                      "audit_passed_videos": len(rows) - len(failed),
                      "source_presented_frames": sum(row.get("source_presented_frames", 0) for row in rows),
                      "preview_decoded_frames": sum(row.get("preview_decoded_frames", 0) for row in rows),
                      "analysis_frames": sum(row.get("analysis_frames") or 0 for row in rows),
                      "source_decode_independently_repeated": decode_source,
                      "source_duration_seconds": source_seconds,
                      "summed_clip_analysis_seconds": analysis_seconds,
                      "analysis_time_per_source_second": analysis_seconds / source_seconds if source_seconds else None},
        "provenance": {"current_implementation_fingerprint": expected_fingerprint,
                       "implementation_fingerprint_counts": dict(fingerprints)},
        "media": {"resolution_counts": dict(Counter(f"{row['source_width']}x{row['source_height']}" for row in rows if "source_width" in row)),
                  "cadence_fps_counts": dict(Counter(str(round(row["source_fps"], 2)) for row in rows if "source_fps" in row)),
                  "warnings": dict(Counter(reason for row in rows for reason in row["warnings"]))},
        "calibration": {"status_counts": dict(Counter(row.get("calibration_status", "unknown") for row in rows)),
                        "reason_counts": dict(Counter(reason for row in rows for reason in row.get("calibration_reasons", []))),
                        "camera_stability_counts": dict(Counter(row.get("camera_stability_status", "unknown") for row in rows))},
        "high_resolution": {"frame_records": sum(row.get("highres_frame_records", 0) for row in rows),
                            "roi_count": sum(row.get("highres_roi_count", 0) for row in rows),
                            "examined_pixels": sum(row.get("highres_examined_pixels", 0) for row in rows),
                            "source_contact_candidates": sum(row.get("source_contact_candidates", 0) for row in rows),
                            "source_contact_frame_count": sum(row.get("source_contact_frame_count", 0) for row in rows)},
        "decisions": {"counts": dict(decisions), "reason_counts": dict(Counter(reason for row in rows for reason in row.get("decision_reasons", []))),
                      "review_fraction": decisions["REVIEW"] / len(rows) if rows else None},
        "ground_truth": {"eligible_count": truth, "automatic_accuracy": metrics.get("automatic_accuracy"),
                         "automatic_accuracy_wilson_95ci": metrics.get("automatic_accuracy_wilson_95ci"),
                         "accuracy_reason": None if truth else "No independently reviewed eligible ground truth is available.",
                         "machine_drafts_are_ground_truth": False},
        "requests": request_checks,
        "videos": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--input-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-source", action="store_true", help="Independently decode each original again, in addition to all previews")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--request-result", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = audit_run(args.run, args.video_dir, input_snapshot=args.input_snapshot,
                       decode_source=args.decode_source, workers=args.workers, request_results=args.request_result)
    output = args.output or args.run / "integrated-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("software_integrity_passed", "errors", "failed_files", "execution", "ground_truth")}, ensure_ascii=False, indent=2))
    return 0 if report["software_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
