"""Request integration: distinct camera clocks, responsibilities, and failures."""
from copy import deepcopy
import hashlib
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
    return {"camera_id": camera, "status": "completed", "source_sha256": hashlib.sha256(camera.encode()).hexdigest(),
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


def test_single_donor_does_not_require_linkage_but_linked_contacts_must_agree():
    payload = request(2)
    payload["same_event_confirmed"] = False
    result = aggregate_request(payload, [row("A"), row("B")])
    assert result["decision"] == "IN"
    assert result["decision_scope"] == "single_view"
    assert result["source_camera_ids"] == ["A"]
    assert "CROSS_VIEW_EVENT_CORRESPONDENCE_UNCONFIRMED" in result["warning_codes"]
    payload["same_event_confirmed"] = True
    result = aggregate_request(payload, [row("A"), row("B", point=(3, 11))])
    assert "CROSS_VIEW_CONTACT_POSITION_DISAGREEMENT" in result["reason_codes"]
    result = aggregate_request(payload, [row("A"), row("B", "OUT", point=(3, -1))])
    assert "CONFLICTING_VIEW_DECISIONS" in result["reason_codes"]
    result = aggregate_request(payload, [row("A"), row("B", point=(3.01, 3.01))])
    assert result["decision"] == "IN"


def test_duplicate_source_is_only_single_view_and_bad_contact_window_still_blocks():
    first, second = row("A"), row("B")
    second["source_sha256"] = first["source_sha256"]
    result = aggregate_request(request(2), [first, second])
    assert result["decision"] == "IN"
    assert result["decision_scope"] == "single_view"
    assert result["source_camera_ids"] == ["A"]
    assert "DUPLICATE_SOURCE_IS_NOT_AN_INDEPENDENT_VIEW" in result["warning_codes"]
    result = aggregate_request(request(), [row("A", time=2.0)])
    assert "CONTACT_OUTSIDE_REQUESTED_EVENT_WINDOW" in result["reason_codes"]


def test_same_event_does_not_establish_shared_court_axes():
    payload = request(2)
    payload["same_court_coordinate_system_confirmed"] = False
    result = aggregate_request(payload, [row("A"), row("B", point=(3.1, 10.4))])
    assert result["decision"] == "IN"
    assert result["source_camera_ids"] == ["A"]
    assert "CROSS_VIEW_COURT_COORDINATE_SYSTEM_UNCONFIRMED" in result["warning_codes"]
    assert "CROSS_VIEW_CONTACT_POSITION_DISAGREEMENT" not in result["reason_codes"]
    assert result["cross_view_contact_checks"] == []


@pytest.mark.parametrize("resolved_camera", ["A", "B"])
@pytest.mark.parametrize("label,point", [("IN", (3, 3)), ("OUT", (3, -1))])
@pytest.mark.parametrize("peer_status", ["completed", "failed"])
def test_one_valid_view_resolves_despite_an_unresolved_peer(resolved_camera, label, point, peer_status):
    payload = request(2)
    payload.update(same_event_confirmed=False, same_court_coordinate_system_confirmed=False)
    views = [row(camera, label if camera == resolved_camera else "REVIEW", point=point)
             for camera in ("A", "B")]
    peer = next(view for view in views if view["camera_id"] != resolved_camera)
    peer["status"] = peer_status
    original = deepcopy(views)
    result = aggregate_request(payload, views)
    assert result["decision"] == label
    assert result["source_camera_ids"] == [resolved_camera]
    assert result["decision_scope"] == "single_view"
    assert "RESPONSIBLE_VIEW_UNRESOLVED" in result["warning_codes"]
    assert views == original  # Do not relabel the unobserved peer.


def test_out_needs_one_assigned_boundary_not_complete_camera_coverage():
    result = aggregate_request(request(), [row("A", "OUT", point=(3, -1), assigned=["near"])])
    assert result["decision"] == "OUT"
    assert result["source_camera_ids"] == ["A"]
    assert result["outside_supported_boundaries"] == ["near"]
    assert "REQUESTED_BOUNDARY_WITHOUT_RESPONSIBLE_CAMERA" in result["warning_codes"]
    result = aggregate_request(request(), [row("A", "IN", assigned=["near"])])
    assert result["decision"] == "REVIEW"
    assert "IN_NOT_SUPPORTED_BY_ASSIGNED_BOUNDARIES" in result["reason_codes"]


def test_partial_in_responsibilities_cannot_be_combined_without_linkage():
    payload = request(2)
    first = row("A", assigned=["near", "left"])
    second = row("B", assigned=["far", "right"])
    payload["same_event_confirmed"] = False
    result = aggregate_request(payload, [first, second])
    assert result["decision"] == "REVIEW"
    payload["same_event_confirmed"] = True
    result = aggregate_request(payload, [first, second])
    assert result["decision"] == "IN"
    assert result["decision_scope"] == "linked_views"
    assert result["source_camera_ids"] == ["A", "B"]


@pytest.mark.parametrize("change,code", [
    ({"source_sha256": None}, "VIEW_SOURCE_UNVERIFIED"),
    ({"source_sha256": "invalid"}, "VIEW_SOURCE_UNVERIFIED"),
    ({"status": "failed"}, "RESPONSIBLE_VIEW_UNRESOLVED"),
])
def test_invalid_donor_cannot_determine_result(change, code):
    invalid = {**row("A"), **change}
    result = aggregate_request(request(), [invalid])
    assert result["decision"] == "REVIEW"
    assert code in result["reason_codes"]


def test_invalid_peer_label_does_not_create_a_false_conflict():
    invalid = row("B", "OUT", point=(3, -1), time=2)
    result = aggregate_request(request(2), [row("A"), invalid])
    assert result["decision"] == "IN"
    assert result["source_camera_ids"] == ["A"]
    assert "CONTACT_OUTSIDE_REQUESTED_EVENT_WINDOW" in result["warning_codes"]
    assert "CONFLICTING_VIEW_DECISIONS" not in result["reason_codes"]
    invalid = row("B", "IN", point=(3, -1))
    result = aggregate_request(request(2), [row("A", "OUT", point=(3, -1)), invalid])
    assert result["decision"] == "OUT"
    assert result["source_camera_ids"] == ["A"]


def test_rules_mismatch_cannot_reuse_a_doubles_verdict_for_singles():
    payload = request()
    payload["rules"]["game"] = "singles"
    donor = row("A")
    donor["decision"]["rules"] = {"game": "doubles", "phase": "rally"}
    result = aggregate_request(payload, [donor])
    assert result["decision"] == "REVIEW"
    assert "VIEW_RULES_MISMATCH" in result["reason_codes"]


def test_geometry_only_probe_cannot_be_used_as_a_verified_video_verdict():
    from badminton.decision import evaluate_contact
    donor = row("A")
    donor["decision"] = {**evaluate_contact([[3, 3]], uncertainty_m=.02), "contact_time_seconds": .7}
    result = aggregate_request(request(), [donor])
    assert result["decision"] == "REVIEW"
    assert "VIEW_CONTACT_GEOMETRY_UNAVAILABLE" in result["reason_codes"]


def primary_pixel_row(camera, point):
    from badminton.decision import decide
    primary = {"status": "selected", "scope": "near_half", "game": "doubles",
               "rules": {"game": "doubles", "phase": "rally"}, "confidence": .9,
               "region_supported": True,
               "polygon_px": [[200, 500], [800, 500], [650, 200], [350, 200]],
               "judgment_scope_polygon_px": [[100, 580], [900, 580], [720, 140], [280, 140]],
               "boundary_segments_px": [
                   {"role": role, "segment_px": segment, "observed": True,
                    "inside_point_px": [500, 350], "uncertainty_px": 3}
                   for role, segment in [("near", [200, 500, 800, 500]),
                                         ("left", [200, 500, 350, 200]),
                                         ("right", [800, 500, 650, 200])]]}
    calibration = {"status": "review", "image_width": 1000, "image_height": 700,
                   "camera_stability": {"status": "ok", "all_frames_sampled": True},
                   "primary_court": primary}
    contact = {"contact_observed": True, "contact_px": list(point), "confidence": .95,
               "identity_confidence": .95, "uncertainty_px": 1, "time": .7}
    value = row(camera)
    value["decision"] = decide(contact, calibration)
    return value


@pytest.mark.parametrize("point,label", [((500, 400), "IN"), ((500, 520), "OUT")])
def test_local_pixel_verdict_can_resolve_without_metric_geometry(point, label):
    donor = primary_pixel_row("A", point)
    assert donor["decision"]["decision"] == label
    result = aggregate_request(request(2), [donor, row("B", "REVIEW")])
    assert result["decision"] == label
    assert result["source_camera_ids"] == ["A"]
    assert result["court_position_m"] is None
    assert result["cross_view_contact_checks"] == []


def test_pixel_out_proof_must_match_responsible_boundary_and_observed_extent():
    donor = primary_pixel_row("A", (500, 520))
    donor["responsible_boundaries"] = ["left"]
    assert aggregate_request(request(boundary="left"), [donor])["decision"] == "REVIEW"
    donor["responsible_boundaries"] = ["near"]
    for edge in donor["decision"]["boundaries"]:
        edge["within_observed_extent"] = False
    assert aggregate_request(request(boundary="near"), [donor])["decision"] == "REVIEW"


def test_request_routes_per_camera_windows_and_anchors_and_persists_errors(video_root, tmp_path, monkeypatch):
    calls = []
    def analyze(path, output, config):
        calls.append((path, output, deepcopy(config)))
        if config["request_context"]["camera_id"] == "camera-1":
            raise RuntimeError("decoder failed")
        return {"source_sha256": "a" * 64, "calibration": {"status": "ok"}, "landings": [],
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
    assert result["decision"]["decision"] == "IN"
    assert result["decision"]["source_camera_ids"] == ["camera-0"]
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
