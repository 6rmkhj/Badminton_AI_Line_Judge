import json
from pathlib import Path
import subprocess

import pytest

from badminton.dataset import build_manifest
from badminton.media import load_frames, prepare_preview
from badminton.evidence import write_annotated_clip
from scripts import validate_integrated_poc as validation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def recorded_run(tmp_path, monkeypatch):
    """A real eight-frame stream, with explicit analysis records to audit."""
    video = tmp_path / "video"
    video.mkdir()
    source = video / "input.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=240x160:rate=30", "-frames:v", "8", "-c:v", "libx264",
                    "-threads", "1", "-pix_fmt", "yuv420p", str(source)], check=True)
    run = tmp_path / "run"
    manifest = build_manifest(video)
    entry = manifest["videos"][0]
    folder = run / "videos" / entry["id"]
    metadata, presentation, digest = prepare_preview(source, folder / "preview.mp4", 160)
    monkeypatch.setattr(validation, "implementation_fingerprint", lambda: "test-fingerprint")
    result = {"video_id": entry["id"], "file": source.name, "source_sha256": digest,
              "implementation_fingerprint": "test-fingerprint", "metadata": metadata,
              "analysis": {"width": presentation["width"], "height": presentation["height"], "frame_count": 8},
              "timestamps": metadata["timestamps"], "config": {"detector": {"highres_enabled": False}},
              "artifacts": {"preview": "preview.mp4", "detections": "detections.json", "evidence": []},
              "calibration": {"status": "review"}, "camera_stability": {"status": "ok"},
              "decision": {"decision": "REVIEW", "reason_codes": ["FIRST_CONTACT_NOT_OBSERVED"]},
              "tracks": [], "landings": [], "timings": {"total_seconds": 0.1}}
    write(folder / "result.json", result)
    write(folder / "detections.json", [{"frame_index": i, "time": t, "detections": []}
                                        for i, t in enumerate(metadata["timestamps"])])
    write(run / "manifest.json", manifest)
    write(run / "index.json", {"status": "completed", "videos": [
        {"id": entry["id"], "file": source.name, "status": "completed", "result_path": f"videos/{entry['id']}/result.json"}]})
    write(run / "evaluation.json", {"video_count": 1, "eligible_ground_truth_count": 0,
                                    "decision": {"automatic_accuracy": None, "automatic_accuracy_wilson_95ci": None}})
    snapshot = tmp_path / "before.json"
    write(snapshot, manifest)
    return run, video, folder, snapshot


def test_real_preview_and_source_counts_are_independently_audited(recorded_run):
    run, video, _, snapshot = recorded_run
    report = validation.audit_run(run, video, input_snapshot=snapshot, decode_source=True)
    assert report["software_integrity_passed"] is True
    assert report["execution"]["preview_decoded_frames"] == 8
    assert report["videos"][0]["source_decoded_frames"] == 8
    assert report["input_inventory"]["unchanged_since_snapshot"]["matches"] is True
    assert report["ground_truth"]["automatic_accuracy"] is None


def test_changed_source_cannot_pass_using_stale_manifest(recorded_run):
    run, video, _, snapshot = recorded_run
    with (video / "input.mp4").open("ab") as handle:
        handle.write(b"\0\0\0\0")
    report = validation.audit_run(run, video, input_snapshot=snapshot)
    assert report["software_integrity_passed"] is False
    assert "SOURCE_HASH_MISMATCH" in report["videos"][0]["errors"]
    assert "INPUTS_CHANGED_SINCE_SNAPSHOT" in report["errors"]


def test_timestamps_and_native_roi_are_checked_independently(recorded_run):
    run, video, folder, _ = recorded_run
    result = validation.read_json(folder / "result.json")
    result["timestamps"][3] += 0.01
    result["config"]["detector"]["highres_enabled"] = True
    write(folder / "result.json", result)
    detections = validation.read_json(folder / "detections.json")
    for frame in detections:
        frame["high_resolution"] = {"status": "analyzed", "source_size": [240, 160],
                                    "rois": [{"bbox": [239, 0, 10, 10]}], "roi_count": 1}
    write(folder / "detections.json", detections)
    report = validation.audit_run(run, video)
    errors = report["videos"][0]["errors"]
    assert "PRESENTATION_TIMESTAMP_MISMATCH" in errors
    assert "NATIVE_DETECTION_ROI_OUT_OF_BOUNDS" in errors


def test_source_contact_strip_cannot_claim_observation_or_wrong_frame(recorded_run):
    run, video, folder, _ = recorded_run
    result = validation.read_json(folder / "result.json")
    result["landings"] = [{"source_contact_evidence": {
        "status": "complete", "source_size": {"width": 240, "height": 160},
        "contact_observed": True,
        "frames": [{"frame_index": 7, "time": 0, "source_roi": [0, 0, 10, 10]}]}}]
    write(folder / "result.json", result)
    report = validation.audit_run(run, video)
    errors = report["videos"][0]["errors"]
    assert "UNVERIFIED_SOURCE_STRIP_MISREPRESENTED_AS_CONTACT" in errors
    assert "SOURCE_CONTACT_FRAME_ALIGNMENT_MISMATCH" in errors


def test_automatic_label_requires_an_observed_contact_not_only_good_geometry(recorded_run):
    run, video, folder, _ = recorded_run
    result = validation.read_json(folder / "result.json")
    result["decision"]["decision"] = "IN"
    result["calibration"]["status"] = "ok"
    result["selected_landing"] = {"contact_observed": False}
    write(folder / "result.json", result)
    report = validation.audit_run(run, video)
    assert "AUTOMATIC_DECISION_WITHOUT_OBSERVED_CONTACT" in report["videos"][0]["errors"]


def test_missing_artifact_stale_fingerprint_and_fake_accuracy_fail(recorded_run):
    run, video, folder, _ = recorded_run
    result = validation.read_json(folder / "result.json")
    result["implementation_fingerprint"] = "old-source"
    result["artifacts"]["overview"] = "missing.jpg"
    write(folder / "result.json", result)
    write(run / "evaluation.json", {"video_count": 1, "eligible_ground_truth_count": 0,
                                    "decision": {"automatic_accuracy": 1.0}})
    report = validation.audit_run(run, video)
    assert "STALE_OR_MISSING_IMPLEMENTATION_FINGERPRINT" in report["videos"][0]["errors"]
    assert "MISSING_OR_EMPTY_ARTIFACT:missing.jpg" in report["videos"][0]["errors"]
    assert "ACCURACY_REPORTED_WITHOUT_ELIGIBLE_GROUND_TRUTH" in report["errors"]


def test_annotated_replay_must_preserve_its_source_frame_timeline(recorded_run):
    run, video, folder, _ = recorded_run
    result = validation.read_json(folder / "result.json")
    result["artifacts"]["annotated_clip"] = write_annotated_clip(
        load_frames(folder / "preview.mp4"), result["timestamps"], result["calibration"], [], None,
        result["decision"], folder)
    write(folder / "result.json", result)
    timing_path = folder / result["artifacts"]["annotated_clip"]["timing_path"]
    timing = validation.read_json(timing_path)
    timing["frames"][2]["source_time"] += 0.01
    write(timing_path, timing)
    report = validation.audit_run(run, video)
    assert "ANNOTATED_REPLAY_SOURCE_ALIGNMENT_MISMATCH" in report["videos"][0]["errors"]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_result_numbers_are_rejected(tmp_path, value):
    path = tmp_path / "result.json"
    path.write_text('{"score":' + value + '}')
    with pytest.raises(ValueError, match="Non-finite"):
        validation.read_json(path)


def test_artifact_paths_cannot_escape_result_directory(recorded_run):
    run, video, folder, _ = recorded_run
    result = validation.read_json(folder / "result.json")
    result["artifacts"]["overview"] = "../../../outside.jpg"
    write(folder / "result.json", result)
    report = validation.audit_run(run, video)
    assert report["software_integrity_passed"] is False
    assert any("Path escapes" in error for error in report["videos"][0]["errors"])


def test_multiview_request_requires_correspondence_and_preserves_assistance(tmp_path):
    path = tmp_path / "request-result.json"
    write(path, {"request_id": "case-1", "status": "completed", "same_event_confirmed": False,
                 "ground_truth_created": False, "automatic_benchmark_eligible": True,
                 "views": [{"camera_id": "a", "status": "failed"}, {"camera_id": "b", "status": "failed"}],
                 "decision": {"decision": "IN", "responsible_camera_ids": []}})
    report = validation.audit_request_result(path)
    assert "MULTIVIEW_DECISION_WITHOUT_EVENT_CORRESPONDENCE" in report["errors"]
    assert "MULTIVIEW_DECISION_WITHOUT_SHARED_COURT_COORDINATES" in report["errors"]
    assert "REQUEST_ASSISTANCE_MISCLASSIFIED_AS_AUTOMATIC_GROUND_TRUTH" in report["errors"]
    assert "CONFIRMED_REQUEST_WITH_FAILED_VIEW" in report["errors"]
