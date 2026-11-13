import http.client
import json
from pathlib import Path
import threading
import time
from unittest.mock import patch

import pytest

from badminton.web import create_server


@pytest.fixture
def review_server(tmp_path):
    rows = []
    for video_id in ("v_a", "v_b"):
        folder = tmp_path / "videos" / video_id
        folder.mkdir(parents=True)
        result = {
            "video_id": video_id, "source_sha256": "a" * 64,
            "metadata": {"width": 1920, "height": 1080, "duration": 2.0, "fps": 10},
            "analysis": {"width": 1280, "height": 720},
            "timestamps": [i / 10 for i in range(20)],
            "decision": {"decision": "REVIEW"}, "landings": [], "tracks": [],
            "calibration": {"status": "failed", "reason_codes": ["NO_CALIBRATION"]},
            "config": {"rules": {"format": "singles"}, "analysis_width": 1280},
        }
        (folder / "result.json").write_text(json.dumps(result))
        (folder / "preview.mp4").write_bytes(b"0123456789abcdef")
        rows.append({"id": video_id, "file": f"{video_id}.mp4", "result_path": f"videos/{video_id}/result.json", "preview_path": f"videos/{video_id}/preview.mp4"})
    (tmp_path / "index.json").write_text(json.dumps({"run_id": "test", "videos": rows,
                                                    "pairs": [{"a_id": "v_a", "b_id": "v_b", "sync": {"status": "review"}}]}))
    server = create_server(tmp_path, port=0)
    server.configure_resources(server.resources_status()["settings"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(server, path, method="GET", body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=4)
    request_headers = dict(headers or {})
    encoded = None
    if body is not None:
        encoded = json.dumps(body).encode()
        request_headers.setdefault("Content-Type", "application/json")
    conn.request(method, path, encoded, request_headers)
    response = conn.getresponse()
    status, response_headers, payload = response.status, dict(response.getheaders()), response.read()
    conn.close()
    if response_headers.get("Content-Type", "").startswith("application/json") and payload:
        payload = json.loads(payload)
    return status, response_headers, payload


def test_index_result_and_static_are_served(review_server):
    server, root = review_server
    status, headers, payload = request(server, "/api/index")
    assert status == 200 and len(payload["videos"]) == 2
    assert payload["review_counts"] == {"verified": 0, "needs_review": 2}
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert request(server, "/api/video/v_a")[2]["video_id"] == "v_a"
    assert request(server, "/")[0] == 200
    assert b"COURT" in request(server, "/")[2]
    assert request(server, "/api/video/not-listed")[0] == 404


def test_media_ranges_head_and_unsatisfiable_ranges(review_server):
    server, _ = review_server
    path = "/media/videos/v_a/preview.mp4"
    status, headers, body = request(server, path, headers={"Range": "bytes=2-5"})
    assert (status, body, headers["Content-Range"]) == (206, b"2345", "bytes 2-5/16")
    assert request(server, path, headers={"Range": "bytes=-3"})[2] == b"def"
    assert request(server, path, headers={"Range": "bytes=14-100"})[2] == b"ef"
    assert request(server, path, headers={"Range": "bytes=100-"})[0] == 416
    assert request(server, path, headers={"Range": "bytes=0-1,3-4"})[0] == 416
    assert request(server, path, headers={"Range": "bytes=-0"})[0] == 416
    status, headers, body = request(server, path, method="HEAD")
    assert status == 200 and headers["Content-Length"] == "16" and body == b""


def test_media_traversal_and_symlink_escape_blocked(review_server, tmp_path_factory):
    server, root = review_server
    other = tmp_path_factory.mktemp("external") / "secret.txt"
    other.write_text("private")
    (root / "escape").symlink_to(other)
    assert request(server, "/media/%2e%2e/secret.txt")[0] == 403
    assert request(server, "/media/escape")[0] == 403
    assert request(server, "/api/annotations/%2e%2e")[0] == 400


def test_annotations_stored_separately_with_explicit_verification(review_server):
    server, root = review_server
    original = (root / "videos/v_a/result.json").read_bytes()
    body = {"label": "IN", "reviewer": "reviewer", "contact_observed": True,
            "time": 0.8, "contact_px": [400, 300], "confirmed": True}
    status, _, saved = request(server, "/api/annotations/v_a", "POST", body)
    assert status == 200 and saved["status"] == "verified"
    assert saved["source"] == "human_review" and saved["source_sha256"] == "a" * 64
    assert saved["analysis_size"] == {"width": 1280, "height": 720}
    assert saved["assisted_decision"]["decision"] == "REVIEW"
    assert saved["assisted_decision"]["evaluation_eligible"] is False
    assert (root / "annotations/v_a.json").exists()
    assert (root / "videos/v_a/result.json").read_bytes() == original
    assert request(server, "/api/index")[2]["review_counts"]["verified"] == 1
    assert request(server, "/api/annotations/v_a")[2]["label"] == "IN"


def test_contact_evidence_applies_rules_without_overwriting_model(review_server):
    server, root = review_server
    body = {"label": "IN", "reviewer": "reviewer", "contact_observed": True,
            "time": 0.8, "contact_px": [400, 300], "confirmed": True,
            "target_identity_confirmed": True, "contact_radius_px": 2, "uncertainty_px": 4}
    status, _, saved = request(server, "/api/annotations/v_a", "POST", body)
    assert status == 200
    assert saved["assisted_landing"]["contact_observed"] is True
    assert saved["assisted_landing"]["frame_index"] == 8
    assert saved["assisted_decision"]["decision"] == "REVIEW"  # failed calibration still gates it
    assert saved["assisted_decision"]["source"] == "human_review"
    assert request(server, "/api/video/v_a")[2]["decision"] == {"decision": "REVIEW"}


def test_unknown_and_incomplete_labels_are_not_promoted(review_server):
    server, _ = review_server
    status, _, saved = request(server, "/api/annotations/v_a", "POST", {"label": "UNKNOWN", "reviewer": "r", "confirmed": True})
    assert status == 200 and saved["status"] == "needs_review" and saved["reviewed"]
    assert saved["verification_reason"] == "ground_truth_unknown"
    status, _, saved = request(server, "/api/annotations/v_a", "POST", {"label": "IN", "status": "verified"})
    assert status == 200 and saved["status"] == "needs_review"
    assert request(server, "/api/annotations/v_a", "POST", {"label": "OUT", "confirmed": True, "reviewer": "r"})[0] == 400
    assert request(server, "/api/annotations/v_a", "POST", {"label": "UNKNOWN", "confirmed": True})[0] == 400


def test_near_line_requires_independent_evidence(review_server):
    server, _ = review_server
    body = {"label": "IN", "reviewer": "r", "confirmed": True, "contact_observed": True,
            "time": 0.5, "contact_px": [100, 200], "near_line_distance_m": 0.02}
    status, _, saved = request(server, "/api/annotations/v_a", "POST", body)
    assert status == 200 and saved["status"] == "needs_review"
    body["independent_evidence"] = "Independent high-speed reference clip #2"
    assert request(server, "/api/annotations/v_a", "POST", body)[2]["status"] == "verified"
    body["independent_evidence"] = False
    assert request(server, "/api/annotations/v_a", "POST", body)[0] == 400


def test_patch_labels_can_be_verified_without_forcing_in_out(review_server):
    server, _ = review_server
    body = {"label": "UNKNOWN", "reviewer": "r", "detection_labels_verified": True,
            "detection_labels": [{"frame_index": 4, "x": 50, "y": 50, "label": "shuttle"}]}
    status, _, saved = request(server, "/api/annotations/v_a", "POST", body)
    assert status == 200 and saved["status"] == "needs_review"
    assert saved["detection_labels_verified"] is True
    assert saved["analysis_size"]["width"] == 1280
    body["detection_labels"][0]["frame_index"] = 500
    assert request(server, "/api/annotations/v_a", "POST", body)[0] == 400


def test_temporal_labels_require_explicit_cork_contact_point(review_server):
    server, _ = review_server
    item = {"frame_index": 4, "x": 50, "y": 50, "label": "first_contact", "contact_observed": True}
    body = {"label": "UNKNOWN", "reviewer": "r", "contact_sequences_verified": True, "contact_sequences": [item]}
    assert request(server, "/api/annotations/v_a", "POST", body)[0] == 400
    item.update(contact_px=[51, 54], contact_point_verified=True, independent_evidence="high-speed reference")
    status, _, saved = request(server, "/api/annotations/v_a", "POST", body)
    assert status == 200 and saved["status"] == "needs_review"
    assert saved["contact_sequences_verified"] is True
    assert saved["contact_sequences"][0]["contact_px"] == [51, 54]
    assert saved["contact_sequences"][0]["x"] == 50
    item["contact_observed"] = False
    assert request(server, "/api/annotations/v_a", "POST", body)[0] == 400


@pytest.mark.parametrize("body", [{"contact_px": [-1, 3]}, {"contact_px": [1280, 1]}, {"contact_px": [1]},
                                  {"time": 100}, {"time": True}, {"contact_radius_px": -1}, {"label": "maybe"}])
def test_bad_annotation_values_rejected(review_server, body):
    assert request(review_server[0], "/api/annotations/v_a", "POST", body)[0] == 400


def test_cross_origin_writes_and_oversize_requests_rejected(review_server):
    server, _ = review_server
    assert request(server, "/api/annotations/v_a", "POST", {"label": "UNKNOWN"}, {"Origin": "https://other.example"})[0] == 403
    assert request(server, "/api/annotations/v_a", "POST", {"label": "UNKNOWN"}, {"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert request(server, "/api/annotations/v_a", "POST", {"label": "UNKNOWN"}, {"Content-Type": "text/plain"})[0] == 415
    assert request(server, "/api/annotations/v_a", "POST", {"notes": "x" * 270000})[0] == 413
    assert request(server, "/api/index", headers={"Host": "attacker.example"})[0] == 403


def test_draft_fallback_and_stale_source_invalidation(review_server):
    server, root = review_server
    (root / "annotation-drafts").mkdir()
    draft = {"video_id": "v_a", "source": "machine_draft", "label": "UNKNOWN", "status": "needs_review", "model_suggestions": []}
    (root / "annotation-drafts/v_a.json").write_text(json.dumps(draft))
    assert request(server, "/api/annotations/v_a")[2]["source"] == "machine_draft"
    (root / "annotations").mkdir()
    (root / "annotations/v_a.json").write_text(json.dumps({"source_sha256": "b" * 64, "source": "human_review", "status": "verified", "label": "IN", "reviewed": True}))
    loaded = request(server, "/api/annotations/v_a")[2]
    assert loaded["status"] == "needs_review" and not loaded["reviewed"]
    assert loaded["verification_reason"] == "source_video_changed"


def test_pair_anchors_keep_automatic_sync_and_manual_provenance(review_server):
    server, root = review_server
    body = {"reviewer": "operator", "event_correspondences": [
        {"time_a": .1, "time_b": .3, "verified": True},
        {"time_a": 1.1, "time_b": 1.3, "verified": True},
    ]}
    status, _, saved = request(server, "/api/pairs/0/sync", "POST", body)
    assert status == 200 and saved["sync"]["status"] == "ok"
    assert saved["sync"]["offset_seconds"] == pytest.approx(.2)
    assert saved["automatic_sync"] == {"status": "review"}
    assert saved["source"] == "human_review"
    assert (root / "pair-annotations/0.json").exists()
    assert request(server, "/api/index")[2]["pairs"][0]["sync"]["status"] == "ok"
    assert request(server, "/api/pairs/20/sync", "POST", body)[0] == 404


def test_reanalysis_preserves_trusted_configuration(review_server):
    server, root = review_server
    source = root / "source.mp4"
    source.write_bytes(b"source")
    from badminton.media import file_hash
    stored = json.loads((root / "videos/v_a/result.json").read_text())
    stored["source_sha256"] = file_hash(source)
    (root / "videos/v_a/result.json").write_text(json.dumps(stored))
    index = json.loads((root / "index.json").read_text())
    index["videos"][0]["source_path"] = str(source)
    (root / "index.json").write_text(json.dumps(index))
    captured = []
    def analyze(path, output_dir, config):
        captured.append(config)
    with patch("badminton.pipeline.analyze_video", analyze):
        status, _, job = request(server, "/api/analyze/v_a", "POST", {"config": {"rules": {"format": "doubles"}}})
        assert status == 202
        for _ in range(100):
            state = request(server, "/api/jobs/" + job["job_id"])[2]
            if state["status"] == "completed":
                break
            time.sleep(.01)
    assert captured and captured[0]["rules"] == {"format": "singles"}
    assert captured[0]["video_id"] == "v_a"
    assert captured[0]["resource_settings"] == server.resources_status()["settings"]
    assert state["resource_settings"] == captured[0]["resource_settings"]
