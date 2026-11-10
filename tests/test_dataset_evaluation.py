import json
from pathlib import Path

import pytest

from badminton.dataset import build_manifest, draft_annotations, pair_identity
from badminton.evaluation import evaluate_run, wilson_interval, _matches


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def run_fixture(tmp_path, results, annotations=()):
    for result in results:
        write_json(tmp_path / "videos" / result["video_id"] / "result.json", result)
    write_json(tmp_path / "index.json", {"videos": [{"id": r["video_id"], "result_path": f"videos/{r['video_id']}/result.json"} for r in results]})
    for annotation in annotations:
        write_json(tmp_path / "annotations" / f"{annotation['video_id']}.json", annotation)
    return evaluate_run(tmp_path)


def truth(identifier="v1", label="IN", **extra):
    return {"video_id": identifier, "source": "human_review", "status": "verified",
            "reviewer": "test-reviewer", "label": label, "time": 1.0,
            "contact_observed": True, "contact_px": [10, 20], **extra}


def prediction(identifier="v1", label="REVIEW", **extra):
    return {"video_id": identifier, "decision": {"decision": label}, **extra}


def test_manifest_transitively_groups_pair_duplicates_and_is_deterministic(tmp_path):
    files = {"a시점 17.mp4": b"a17", "b시점 17.mp4": b"same",
             "a시점 18.mp4": b"a18", "b시점 18.mp4": b"same",
             "a시점 1.mp4": b"a1", "b시점 1.mp4": b"b1",
             "a시점 2.mp4": b"a2", "b시점 2.mp4": b"b2",
             "a시점 3.mp4": b"a3", "b시점 3.mp4": b"b3"}
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    manifest = build_manifest(tmp_path)
    assert manifest == build_manifest(tmp_path)
    assert manifest["unique_content_count"] == 9
    assert manifest["group_count"] == 4
    linked = [r for r in manifest["videos"] if "17" in r["file"] or "18" in r["file"]]
    assert len({r["group_id"] for r in linked}) == 1
    assert len({r["split"] for r in linked}) == 1
    assert set(r["split"] for r in manifest["videos"]) == {"train", "val", "test"}
    assert not manifest["generalization_evidence"]["independent_venues_verified"]
    for group in manifest["groups"]:
        assert len({r["split"] for r in manifest["videos"] if r["group_id"] == group["group_id"]}) == 1


@pytest.mark.parametrize("left,right", [
    ("8k 30 4k 24 a시점 1.mp4", "8k 30 4k 24 b시점 1.mp4"),
    ("a-1시점(uhd30 4k30).mp4", "b-1시점(uhd30 4k30).mp4"),
    ("a시점(uhd30 hd30) 1.mp4", "b시점(uhd30 hd30) 1.mp4"),
])
def test_pair_families(left, right):
    assert pair_identity(left)[0] == pair_identity(right)[0]
    assert pair_identity(left)[1] == "a"
    assert pair_identity(right)[1] == "b"


def test_drafts_never_become_truth_or_overwrite(tmp_path):
    result = prediction(label="IN", landings=[{"time": 1.0, "candidate_px": [20, 30], "confidence": 0.99}])
    output = draft_annotations([result], tmp_path)
    draft = json.loads((tmp_path / "v1.json").read_text())
    assert output["ground_truth_created"] is False
    assert draft["label"] == "UNKNOWN"
    assert draft["source"] == "machine_draft"
    assert draft["status"] == "needs_review"
    assert draft["contact_observed"] is False
    draft["notes"] = "review work in progress"
    write_json(tmp_path / "v1.json", draft)
    assert draft_annotations([result], tmp_path)["preserved_count"] == 1
    assert json.loads((tmp_path / "v1.json").read_text())["notes"] == "review work in progress"


def test_no_truth_has_null_accuracy_and_real_review_coverage(tmp_path):
    output = run_fixture(tmp_path, [prediction(), prediction("v2")])
    assert output["decision"]["automatic_accuracy"] is None
    assert output["decision"]["automatic_accuracy_wilson_95ci"] is None
    assert output["detection"]["recall"] is None
    assert output["tracking"]["id_switches"] is None
    assert output["operational"]["review_rate"] == 1.0
    assert not output["accuracy_verified"]
    assert not output["performance_acceptance_established"]


def test_machine_drafts_and_unknown_are_not_evaluable(tmp_path):
    output = run_fixture(tmp_path, [prediction(label="IN"), prediction("v2", "OUT")],
                         [truth(source="machine_draft"), truth("v2", "UNKNOWN")])
    assert output["eligible_ground_truth_count"] == 0
    assert output["excluded_ground_truth_counts"] == {"not_verified_human_ground_truth": 1, "unknown_ground_truth": 1}


def test_accuracy_coverage_and_wilson_interval_are_separate(tmp_path):
    output = run_fixture(tmp_path, [prediction(label="IN"), prediction("v2", "OUT"), prediction("v3")],
                         [truth(), truth("v2", "IN"), truth("v3", "OUT")])
    metrics = output["decision"]
    assert metrics["automatic_accuracy"] == 0.5
    assert metrics["automatic_coverage"] == pytest.approx(2 / 3)
    assert metrics["review_rate"] == pytest.approx(1 / 3)
    assert metrics["confusion"]["OUT"]["REVIEW"] == 1
    assert 0 < metrics["automatic_accuracy_wilson_95ci"][0] < 0.5 < metrics["automatic_accuracy_wilson_95ci"][1] < 1
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_near_line_requires_independent_evidence(tmp_path):
    output = run_fixture(tmp_path, [prediction(label="IN"), prediction("v2", "IN")],
                         [truth(near_line_distance_m=0.02),
                          truth("v2", near_line_distance_m=0.02, independent_evidence="independent high-speed clip reference")])
    assert output["eligible_ground_truth_count"] == 1
    assert output["near_line_within_5cm"]["ground_truth_cases"] == 1
    assert output["excluded_ground_truth_counts"]["near_line_without_independent_evidence"] == 1


def test_landing_metrics_require_observed_contact_and_metric_reference(tmp_path):
    observed = {"contact_observed": True, "time": 1.02, "contact_px": [10, 20], "court_position_m": [2.03, 4.04]}
    result = prediction(label="IN", selected_landing=observed, calibration={"status": "ok", "court_id": "court-1"})
    annotated = truth(court_id="court-1", court_position_m=[2, 4], coordinate_system="badminton_court_m", position_reference_verified=True)
    output = run_fixture(tmp_path, [result], [annotated])
    assert output["landing"]["contact_time_absolute_error_ms"]["mean"] == pytest.approx(20)
    assert output["landing"]["contact_position_absolute_error_mm"]["mean"] == pytest.approx(50)
    annotated["position_reference_verified"] = False
    output = run_fixture(tmp_path, [result], [annotated])
    assert output["landing"]["contact_position_absolute_error_mm"]["mean"] is None
    result["selected_landing"]["contact_observed"] = False
    output = run_fixture(tmp_path, [result], [annotated])
    assert output["landing"]["contact_time_absolute_error_ms"]["mean"] is None


def test_component_metrics_use_complete_frames_and_track_identities(tmp_path):
    result = prediction(detections=[
        {"frame_index": 0, "detections": [{"x": 10, "y": 10}, {"x": 90, "y": 90}]},
        {"frame_index": 1, "detections": [{"x": 12, "y": 10}]},
    ], tracks=[
        {"track_id": "p1", "observations": [{"frame_index": 0, "x": 10, "y": 10}]},
        {"track_id": "p2", "observations": [{"frame_index": 1, "x": 12, "y": 10}]},
    ], calibration={"status": "ok", "method": "automatic", "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    annotated = truth(detection_frames=[
        {"frame_index": 0, "complete": True, "objects": [{"x": 10, "y": 10, "track_id": "shuttle-1"}]},
        {"frame_index": 1, "complete": True, "objects": [{"x": 12, "y": 10, "track_id": "shuttle-1"}, {"x": 100, "y": 100, "track_id": "shuttle-2"}]},
        {"frame_index": 2, "complete": False, "objects": [{"x": 999, "y": 999}]},
    ], reconnection_episodes=[{"track_id": "shuttle-1", "before_frame": 0, "after_frame": 1}],
        calibration_reference={"valid": True, "independent_reference": True, "reference_points": [{"image_px": [1, 2], "court_m": [1, 2]}]})
    output = run_fixture(tmp_path, [result], [annotated])
    assert output["detection"]["precision"] == pytest.approx(2 / 3)
    assert output["detection"]["recall"] == pytest.approx(2 / 3)
    assert output["tracking"]["id_switches"] == 1
    assert output["tracking"]["reconnection_accuracy"] == 0
    assert output["calibration"]["success_rate"] == 1
    assert output["calibration"]["independent_court_reprojection_error_mm"]["mean"] == 0


def test_assignment_maximizes_matches_instead_of_greedy_nearest():
    # T0 can match either P0/P1, T1 can only match P0.
    truth_points = [{"x": 0, "y": 0}, {"x": 2, "y": 0}]
    predictions = [{"x": 1, "y": 0}, {"x": -1, "y": 0}]
    assert len(_matches(truth_points, predictions, 1.1)) == 2


def test_result_paths_cannot_escape_run(tmp_path):
    write_json(tmp_path / "index.json", {"videos": [{"id": "v1", "result_path": "../secret.json"}]})
    with pytest.raises(ValueError, match="escapes"):
        evaluate_run(tmp_path)


def test_failed_reanalysis_never_scores_stale_success(tmp_path):
    run_fixture(tmp_path, [prediction(label="IN")], [truth()])
    write_json(tmp_path / "index.json", {"videos": [{"id": "v1", "status": "failed",
                                                     "result_path": "videos/v1/result.json",
                                                     "error": "new source decoding failed"}]})
    output = evaluate_run(tmp_path)
    assert output["operational"]["automatic_coverage"] == 0
    assert output["operational"]["processing_error_rate"] == 1
    assert output["decision"]["automatic_accuracy"] is None
    assert output["decision"]["confusion"]["IN"]["ERROR"] == 1
    assert output["result_read_errors"][0]["source"] == "current_run_index"


def test_external_detection_artifact_is_evaluated(tmp_path):
    result = prediction(artifacts={"detections": "detections.json"})
    annotated = truth(detection_frames=[{"frame_index": 0, "complete": True,
                                         "objects": [{"x": 2, "y": 3}]}])
    run_fixture(tmp_path, [result], [annotated])
    write_json(tmp_path / "videos/v1/detections.json", [{"frame_index": 0, "detections": [{"x": 2, "y": 3}]}])
    output = evaluate_run(tmp_path)
    assert output["detection"]["precision"] == 1
    assert output["detection"]["recall"] == 1


def test_stale_source_labels_and_changed_coordinates_are_excluded(tmp_path):
    result = prediction(label="IN", source_sha256="new-hash", analysis={"width": 1280, "height": 720})
    annotated = truth(source_sha256="old-hash")
    output = run_fixture(tmp_path, [result], [annotated])
    assert output["excluded_ground_truth_counts"]["annotation_source_hash_mismatch"] == 1
    annotated["source_sha256"] = "new-hash"
    annotated["analysis_size"] = {"width": 640, "height": 360}
    output = run_fixture(tmp_path, [result], [annotated])
    assert output["excluded_ground_truth_counts"]["annotation_analysis_dimensions_mismatch"] == 1
