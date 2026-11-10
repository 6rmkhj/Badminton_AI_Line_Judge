"""Integration contracts: requests select identity without inventing contact.

The module boundaries are stubbed so these tests can distinguish genuine
decision gating from the accidental all-REVIEW behaviour of an untrained model.
These synthetic contracts make no empirical accuracy claim.
"""
import copy

import numpy as np
import pytest

from badminton import calibration, detection, events, pipeline, tracking
from badminton.evaluation import automatically_inferred


def _track(key, *, start=0, end=20, x=20):
    return {"track_id": key, "identity_ambiguous": False, "source_track_ids": [key],
            "reconnection_hypotheses": [], "reconnections": [], "gaps": [],
            "observations": [{"frame_index": i, "time": i / 10, "x": x + i, "y": 20,
                              "observed": True, "confidence": .9} for i in range(start, end + 1)]}


def _event(key=1, *, time=1.5, contact=True, confidence=.95):
    return {"track_id": key, "time": time, "frame_index": round(time * 10),
            "candidate_px": [30, 30], "contact_observed": contact,
            "contact_px": [30, 30] if contact else None,
            "contact_radius_px": 0, "uncertainty_px": .1,
            "confidence": confidence, "identity_confidence": .4,
            "contact_evidence_source": "qualified_learned_contact_localization" if contact else None,
            "reason_codes": [] if contact else ["FIRST_CONTACT_NOT_OBSERVED"]}


def _run(monkeypatch, tmp_path, candidates, tracks, *, identity=None, stability=None):
    timestamps = [i / 10 for i in range(31)]
    frames = [np.zeros((100, 200, 3), np.uint8) for _ in timestamps]
    metadata = {"width": 200, "height": 100, "frame_count": len(frames), "duration": 3.1,
                "fps": 10, "warnings": [], "timestamps": timestamps}
    source = tmp_path / "source.mp4"
    source.touch()
    def preview(*args):
        return copy.deepcopy(metadata), copy.deepcopy(metadata), "a" * 64
    monkeypatch.setattr(pipeline, "prepare_preview", preview)
    monkeypatch.setattr(pipeline, "load_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(calibration, "calibrate", lambda *args: {
        "status": "ok", "confidence": .99, "method": "automatic_test_geometry",
        "homography": [[.05, 0, 0], [0, .05, 0], [0, 0, 1]],
        "line_fit_error_px": .1, "image_width": 200, "image_height": 100,
        "reason_codes": []})
    monkeypatch.setattr(calibration, "assess_camera_stability", lambda *args: stability or {"status": "ok"})
    monkeypatch.setattr(detection, "detect_sequence", lambda *args: [
        {"frame_index": i, "time": time, "detections": []} for i, time in enumerate(timestamps)])
    monkeypatch.setattr(tracking, "track_detections", lambda *args: copy.deepcopy(tracks))
    monkeypatch.setattr(events, "estimate_landings", lambda *args: copy.deepcopy(candidates))
    config = {"source_crop_count": 0, "detector": {"highres_enabled": False},
              "events": {"source_contact_enabled": False}, "evidence": {"annotated_clip": False},
              "identity": identity or {}}
    return pipeline.analyze_video(source, tmp_path / "analysis", config)


def _identity():
    return {"hit_anchors": [{"time": .5, "point_px": [25, 20], "radius_px": 8,
                             "time_tolerance_seconds": .05}], "event_window_seconds": [1, 2]}


def test_request_selects_target_instead_of_highest_score_and_excludes_assisted_metrics(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, [_event(1, confidence=.9), _event(2, confidence=.99)],
                  [_track(1), _track(2, x=100)], identity=_identity())
    assert result["decision"]["decision"] == "IN"
    assert len(result["landings"]) == 2
    assert result["selected_landing"]["track_id"] == 1
    assert result["identity"]["requested_event_count"] == 1
    assert result["decision"]["evaluation_eligible"] is False
    assert result["assistance_used"] is True
    assert automatically_inferred(result) is False


def test_target_hit_does_not_create_first_contact(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, [_event(contact=False)], [_track(1)], identity=_identity())
    assert result["decision"]["decision"] == "REVIEW"
    assert "FIRST_CONTACT_NOT_OBSERVED" in result["decision"]["reason_codes"]
    assert result["selected_landing"]["contact_px"] is None


def test_competing_anchor_hypotheses_never_choose_highest_candidate_score(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, [_event(1, confidence=.9), _event(2, confidence=.99)],
                  [_track(1), _track(2, x=21)], identity=_identity())
    assert result["decision"]["decision"] == "REVIEW"
    assert all(item["request_match"] == "ambiguous" for item in result["landings"])


def test_unique_identity_still_keeps_two_landings_unresolved(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, [_event(time=1.2), _event(time=1.8)],
                  [_track(1)], identity=_identity())
    assert result["decision"]["decision"] == "REVIEW"
    assert "MULTIPLE_LANDING_CANDIDATES" in result["decision"]["reason_codes"]


def test_window_never_falls_back_to_candidate_outside_requested_event(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, [_event(time=.8)], [_track(1)], identity=_identity())
    assert result["decision"]["decision"] == "REVIEW"
    assert result["landings"] == [] and result["selected_landing"] is None


def test_camera_motion_invalidates_otherwise_qualified_assisted_contact(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, [_event()], [_track(1)], identity=_identity(),
                  stability={"status": "review", "reason_codes": ["CAMERA_MOVED"]})
    assert result["calibration"]["status"] == "review"
    assert result["decision"]["decision"] == "REVIEW"
    assert "CALIBRATION_NOT_ACCEPTED" in result["decision"]["reason_codes"]


def test_offline_tracklet_merge_preserves_observed_occlusion_evidence():
    a, b = _track(1, start=0, end=2), _track(2, start=6, end=8)
    a["gaps"] = [{"start_frame": 3, "end_frame": 5, "resolved": False, "possible_occlusion": True}]
    result = tracking._offline_links([a, b], {"offline_max_gap_seconds": .8, "offline_max_residual_px": 28})
    assert len(result) == 1
    assert result[0]["gaps"][0]["resolved"] is True
    assert result[0]["gaps"][0]["possible_occlusion"] is True
    assert result[0]["gaps"][0]["observations_interpolated"] is False
    assert result[0]["reconnections"][0]["verified_identity"] is False
