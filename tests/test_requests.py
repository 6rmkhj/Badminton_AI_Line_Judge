"""Request integration: distinct camera clocks, responsibilities, and failures."""
from copy import deepcopy
import json

import pytest

from badminton.requests import aggregate_request, analyze_request, validate_request, write_request_summary


def request(views=1, boundary="all"):
    return {"request_id": "line-call-01", "rules": {"game": "doubles", "phase": "rally"},
            "requested_boundary": boundary, "same_event_confirmed": views > 1,
            "same_court_coordinate_system_confirmed": views > 1,
            "views": [{"camera_id": f"camera-{i}", "video": f"camera-{i}.mp4",
                       "responsible_boundaries": ["left", "right", "near", "far"],
                       "event_window_seconds": [0.5 + i, 1.0 + i],
                       "hit_anchors": [{"time": 0.2 + i, "point_px": [100, 80]}]}
                      for i in range(views)]}


@pytest.fixture
def video_root(tmp_path, monkeypatch):
    root = tmp_path / "video"
    root.mkdir()
    for i in range(3):
        (root / f"camera-{i}.mp4").write_bytes(bytes([i]))
    monkeypatch.setattr("badminton.media.probe", lambda _: {
        "width": 1920, "height": 1080, "fps": 10, "timestamps": [i / 10 for i in range(30)]})
    return root


def row(camera, label="IN", point=(3, 3), time=0.7, assigned=None):
    return {"camera_id": camera, "status": "completed", "source_sha256": camera,
            "responsible_boundaries": assigned or ["left", "right", "near", "far"],
            "event_window_seconds": [0.5, 1.0],
            "decision": {"decision": label, "contact_time_seconds": time,
                         "contact_polygon_m": [list(point)], "uncertainty_m": .02}}


def test_validation_uses_display_frames_and_analysis_pixel_coordinates(video_root):
    prepared = validate_request(request(2), video_root, {"analysis_width": 640})
    assert prepared["views"][0]["analysis_size"] == [640, 360]
    assert prepared["views"][1]["window_frame_range"] == [15, 20]
    assert prepared["views"][1]["hit_anchors"][0]["time"] == 1.2
    bad = request()
    bad["views"][0]["hit_anchors"][0]["point_px"] = [641, 100]
    with pytest.raises(ValueError, match="analysis image"):
        validate_request(bad, video_root, {"analysis_width": 640})


@pytest.mark.parametrize("change, match", [
    (lambda r: r["views"][0].update(video="../outside.mp4"), "inside video_root"),
    (lambda r: r["views"][0].update(event_window_seconds=[2.0, 4.0]), "exceeds"),
    (lambda r: r["views"][0].update(event_window_seconds=[.01, .09]), "no displayed frames"),
    (lambda r: r["views"][0].update(responsible_boundaries=["net"]), "responsible_boundaries"),
    (lambda r: r["views"][0]["hit_anchors"][0].update(time=1.1), "precede"),
    (lambda r: r.update(same_event_confirmed="yes"), "boolean"),
])
def test_invalid_request_is_rejected_before_analysis(video_root, change, match):
    payload = request()
    change(payload)
    with pytest.raises(ValueError, match=match):
        validate_request(payload, video_root)


def test_symlink_escape_and_repeated_camera_rejected(video_root, tmp_path):
    external = tmp_path / "outside.mp4"
    external.write_bytes(b"outside")
    (video_root / "escape.mp4").symlink_to(external)
    payload = request()
    payload["views"][0]["video"] = "escape.mp4"
    with pytest.raises(ValueError, match="outside video_root"):
        validate_request(payload, video_root)
    payload = request(2)
    payload["views"][1]["camera_id"] = payload["views"][0]["camera_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_request(payload, video_root)


def test_missing_or_unresolved_responsible_view_cannot_be_masked():
    payload = request(boundary="near")
    result = aggregate_request(payload, [row("A", assigned=["far"])])
    assert result["decision"] == "REVIEW"
    assert "REQUESTED_BOUNDARY_WITHOUT_RESPONSIBLE_CAMERA" in result["reason_codes"]
    payload = request(2, boundary="near")
    result = aggregate_request(payload, [row("A", "REVIEW", assigned=["near"]), row("B", assigned=["far"])])
    assert result["decision"] == "REVIEW"
    assert "RESPONSIBLE_VIEW_UNRESOLVED" in result["reason_codes"]


def test_single_responsible_view_can_resolve_without_fusing_coordinates():
    result = aggregate_request(request(boundary="near"), [row("A", assigned=["near"])])
    assert result["decision"] == "IN"
    assert result["court_position_m"] is None
    assert result["automatic_benchmark_eligible"] is False


def test_an_outside_unassigned_line_is_not_an_authorized_line_result():
    payload = request(boundary="near")
    result = aggregate_request(payload, [row("A", "OUT", point=(-1, 3), assigned=["near"])])
    assert result["decision"] == "REVIEW"
    assert "OUT_NOT_SUPPORTED_BY_ASSIGNED_BOUNDARY" in result["reason_codes"]
    result = aggregate_request(payload, [row("A", "OUT", point=(3, -1), assigned=["near"])])
    assert result["decision"] == "OUT"
    assert result["outside_supported_boundaries"] == ["near"]


def test_multi_view_needs_explicit_correspondence_and_consistent_contact():
    payload = request(2)
    payload["same_event_confirmed"] = False
    result = aggregate_request(payload, [row("A"), row("B")])
    assert "CROSS_VIEW_EVENT_CORRESPONDENCE_UNCONFIRMED" in result["reason_codes"]
    payload["same_event_confirmed"] = True
    result = aggregate_request(payload, [row("A"), row("B", point=(3, 11))])
    assert "CROSS_VIEW_CONTACT_POSITION_DISAGREEMENT" in result["reason_codes"]
    result = aggregate_request(payload, [row("A"), row("B", "OUT", point=(3, -1))])
    assert "CONFLICTING_VIEW_DECISIONS" in result["reason_codes"]
    result = aggregate_request(payload, [row("A"), row("B", point=(3.01, 3.01))])
    assert result["decision"] == "IN"


def test_duplicate_source_and_contact_outside_window_remain_review():
    first, second = row("A"), row("B")
    second["source_sha256"] = first["source_sha256"]
    result = aggregate_request(request(2), [first, second])
    assert "DUPLICATE_SOURCE_IS_NOT_AN_INDEPENDENT_VIEW" in result["reason_codes"]
    result = aggregate_request(request(), [row("A", time=2.0)])
    assert "CONTACT_OUTSIDE_REQUESTED_EVENT_WINDOW" in result["reason_codes"]


def test_same_event_does_not_establish_shared_court_axes():
    payload = request(2)
    payload["same_court_coordinate_system_confirmed"] = False
    result = aggregate_request(payload, [row("A"), row("B", point=(3.1, 10.4))])
    assert result["decision"] == "REVIEW"
    assert "CROSS_VIEW_COURT_COORDINATE_SYSTEM_UNCONFIRMED" in result["reason_codes"]
    assert "CROSS_VIEW_CONTACT_POSITION_DISAGREEMENT" not in result["reason_codes"]
    assert result["cross_view_contact_checks"] == []


def test_request_routes_per_camera_windows_and_anchors_and_persists_errors(video_root, tmp_path, monkeypatch):
    calls = []
    def analyze(path, output, config):
        calls.append((path, output, deepcopy(config)))
        if config["request_context"]["camera_id"] == "camera-1":
            raise RuntimeError("decoder failed")
        return {"source_sha256": "a", "calibration": {"status": "ok"}, "landings": [],
                "decision": row("A")["decision"]}
    monkeypatch.setattr("badminton.pipeline.analyze_video", analyze)
    config = {"identity": {"hit_anchors": [{"time": 999}], "min_margin": .2},
              "events": {"target_track_id": "stale-other-video"}}
    before = deepcopy(config)
    output = tmp_path / "run"
    result = analyze_request(request(2), video_root, output, config)
    assert len(calls) == 2
    assert calls[0][2]["identity"]["hit_anchors"][0]["time"] == .2
    assert calls[1][2]["identity"]["event_window_seconds"] == [1.5, 2.0]
    assert calls[1][2]["identity"]["min_margin"] == .2
    assert "target_track_id" not in calls[0][2]["events"]
    assert config == before
    assert result["status"] == "completed_with_errors"
    assert result["decision"]["decision"] == "REVIEW"
    assert result["ground_truth_created"] is False
    assert json.loads((output / "request-result.json").read_text()) == result


def test_entire_request_validation_precedes_any_analysis(video_root, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("badminton.pipeline.analyze_video", lambda *args: calls.append(args))
    payload = request(2)
    payload["views"][1]["event_window_seconds"] = [3, 7]
    with pytest.raises(ValueError):
        analyze_request(payload, video_root, tmp_path / "run")
    assert not calls


def test_standalone_summary_escapes_labels_and_links_only_owned_evidence(tmp_path):
    folder = tmp_path / "views" / "camera_a"
    evidence = folder / "evidence"
    evidence.mkdir(parents=True)
    for path in (folder / "preview.mp4", folder / "annotated.mp4", evidence / "source contact.jpg"):
        path.write_bytes(b"artifact")
    (tmp_path / "views" / "private.jpg").write_bytes(b"outside this camera")
    artifacts = {"preview": "preview.mp4", "annotated_clip": {"path": "annotated.mp4"},
                 "evidence": [{"source_sequence": {"image": "evidence/source contact.jpg"}},
                              {"image": "../private.jpg"}]}
    (folder / "result.json").write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")
    view = row('<script>alert("bad")</script>')
    view.update(video="<video>.mp4", result_path="views/camera_a/result.json")
    result = {"request_id": "request <unsafe>", "same_event_confirmed": False,
              "views": [view], "decision": aggregate_request(request(), [view])}
    report = write_request_summary(result, tmp_path)
    text = report.read_text(encoding="utf-8")
    assert '<script>alert' not in text
    assert '&lt;script&gt;' in text and 'request &lt;unsafe&gt;' in text
    assert 'views/camera_a/annotated.mp4' in text
    assert 'views/camera_a/preview.mp4' in text
    assert 'views/camera_a/evidence/source%20contact.jpg' in text
    assert '../private.jpg' not in text and 'views/private.jpg' not in text
    assert 'http://' not in text and 'https://' not in text


def test_runtime_failure_does_not_link_a_previous_result(video_root, tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("incomplete decode")
    monkeypatch.setattr("badminton.pipeline.analyze_video", fail)
    result = analyze_request(request(), video_root, tmp_path / "run")
    assert result["views"][0]["result_path"] is None
    assert (tmp_path / "run/request-summary.html").is_file()


def test_full_clip_window_normalizes_to_existing_frames_for_identity(video_root, tmp_path, monkeypatch):
    from badminton.identity import apply_identity
    times = [i / 10 for i in range(30)]
    calls = []
    def analyze(path, output, config):
        # Exercise the actual consumer contract, not just a matching mock.
        assert apply_identity([], [], times, {}, config["identity"]) == []
        calls.append(deepcopy(config["identity"]))
        return {"source_sha256": "a", "calibration": {"status": "review"}, "landings": [],
                "decision": {"decision": "REVIEW", "reason_codes": ["NO_LANDING_CANDIDATE"]}}
    monkeypatch.setattr("badminton.pipeline.analyze_video", analyze)
    payload = request()
    payload["views"][0]["event_window_seconds"] = [0, 3.0]
    payload["views"][0]["hit_anchors"][0]["event_window_seconds"] = [.55, 3.0]
    result = analyze_request(payload, video_root, tmp_path / "full")
    assert result["status"] == "completed"
    assert result["views"][0]["event_window_seconds"] == [0.0, 3.0]
    assert result["views"][0]["analysis_event_window_seconds"] == [0.0, 2.9]
    assert calls[0]["event_window_seconds"] == [0.0, 2.9]
    assert calls[0]["hit_anchors"][0]["event_window_seconds"] == [.6, 2.9]
    assert calls[0]["hit_anchors"][0]["requested_event_window_seconds"] == [.55, 3.0]


def test_hit_anchor_in_final_frame_tail_is_not_an_observed_timestamp(video_root):
    payload = request()
    payload["views"][0]["event_window_seconds"] = [0, 3.0]
    payload["views"][0]["hit_anchors"][0]["time"] = 2.95
    with pytest.raises(ValueError, match="observed frame timestamps"):
        validate_request(payload, video_root)


def test_single_frame_scope_and_long_final_display_duration(video_root, monkeypatch):
    monkeypatch.setattr("badminton.media.probe", lambda _: {
        "width": 1920, "height": 1080, "fps": 10, "duration": 3.05,
        "timestamps": [i / 10 for i in range(30)]})
    payload = request()
    payload["views"][0]["event_window_seconds"] = [2.85, 3.05]
    payload["views"][0]["hit_anchors"][0]["event_window_seconds"] = [2.9, 2.9]
    view = validate_request(payload, video_root)["views"][0]
    assert view["event_window_seconds"] == [2.85, 3.05]
    assert view["analysis_event_window_seconds"] == [2.9, 2.9]
    assert view["hit_anchors"][0]["event_window_seconds"] == [2.9, 2.9]


@pytest.mark.parametrize("status, expected", [("completed", 0), ("completed_with_errors", 1), ("failed", 1)])
def test_judge_cli_reports_failed_views_as_nonzero(monkeypatch, capsys, status, expected):
    from badminton.__main__ import main
    monkeypatch.setattr("badminton.requests.analyze_request", lambda *args: {"status": status})
    monkeypatch.setattr("sys.argv", ["badminton", "judge", "request.json"])
    assert main() == expected
    assert status in capsys.readouterr().out
