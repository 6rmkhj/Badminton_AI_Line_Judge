import numpy as np
import pytest

from badminton.events import estimate_landings, apply_contact_evidence


def track(points, *, ambiguous=False, times=None):
    times = times or [i / 30 for i in range(len(points))]
    return {"track_id": 7, "identity_ambiguous": ambiguous, "gaps": [],
            "observations": [{"frame_index": i, "time": t, "x": x, "y": y,
                              "radius": 2, "confidence": 0.9}
                             for i, ((x, y), t) in enumerate(zip(points, times))]}


def test_slowdown_proposes_review_not_physical_contact():
    points = [(10 + i * 4, 10 + i * 3) for i in range(7)] + [(34, 28)] * 5
    frames = [np.zeros((100, 100, 3), np.uint8) for _ in points]
    times = [i / 30 for i in range(len(points))]
    events = estimate_landings([track(points)], frames, times,
                              {"status": "ok", "polygon_px": [[0, 0], [99, 0], [99, 99], [0, 99]]})
    assert events
    candidate = events[0]
    assert "MOTION_SLOWDOWN_AND_PERSISTENCE" in candidate["reason_codes"]
    assert candidate["contact_observed"] is False
    assert candidate["contact_px"] is None
    assert "FIRST_CONTACT_NOT_OBSERVED" in candidate["reason_codes"]
    assert candidate["target_court_score"] == 1
    assert candidate["identity_confidence"] <= 0.55


def test_stationary_background_is_not_a_landing():
    points = [(20, 20)] * 10
    assert estimate_landings([track(points)], [], [i / 30 for i in range(10)], {}) == []


def test_missing_observations_and_crossing_id_remain_explicit():
    points = [(10 + i * 4, 10 + i * 3) for i in range(7)] + [(34, 28)] * 5
    times = [i / 30 if i < 7 else i / 30 + 0.3 for i in range(len(points))]
    tr = track(points, ambiguous=True, times=times)
    tr["gaps"] = [{"duration": 0.3}]
    events = estimate_landings([tr], [np.zeros((80, 80, 3), np.uint8) for _ in points],
                              [i / 30 for i in range(30)], {})
    assert any("TRACK_IDENTITY_AMBIGUOUS" in e["reason_codes"] for e in events)
    assert all(e["contact_observed"] is False for e in events)
    assert all(e["identity_confidence"] <= 0.25 for e in events)


def test_selected_track_cannot_override_unobserved_contact():
    points = [(10 + i * 3, 20 + i * 2) for i in range(6)] + [(25, 30)] * 5
    events = estimate_landings([track(points)], [np.zeros((80, 80, 3), np.uint8) for _ in points],
                              [i / 30 for i in range(len(points))], {}, {"target_track_id": 7})
    assert events and events[0]["identity_source"] == "operator_selected_track"
    assert events[0]["contact_px"] is None


def test_track_termination_is_low_confidence_draft():
    points = [(10 + i * 4, 10 + i * 2) for i in range(8)]
    events = estimate_landings([track(points)], [], [i / 30 for i in range(8)], {})
    assert events[0]["confidence"] <= 0.2
    assert "TRACK_TERMINATED_WITHOUT_CONTACT" in events[0]["reason_codes"]


def contact_evidence():
    return {"video_id": "clip-a", "source": "human_review", "verified": True,
            "coordinate_system": "analysis_pixels", "time": 1.2, "contact_px": [45, 60],
            "contact_radius_px": 2, "uncertainty_px": 3,
            "first_contact_observed": True, "target_identity_confirmed": True}


def test_human_contact_is_separate_from_automatic_predictions():
    original = {"time": 1.3, "contact_observed": False, "contact_px": None,
                "reason_codes": ["FIRST_CONTACT_NOT_OBSERVED", "CALIBRATION_NOT_CONFIRMED"]}
    result = apply_contact_evidence(original, contact_evidence(), video_id="clip-a", duration_seconds=3,
                                    image_size=(100, 80))
    assert result["contact_observed"] is True
    assert result["inference_origin"] == "human_review"
    assert result["evaluation_eligible"] is False
    assert result["contact_px"] == [45, 60]
    assert "CALIBRATION_NOT_CONFIRMED" in result["reason_codes"]
    assert original["contact_observed"] is False


def test_reviewed_time_uses_trusted_frame_timestamps():
    candidate = {"time": 1.5, "frame_index": 45}
    result = apply_contact_evidence(candidate, contact_evidence(), video_id="clip-a", duration_seconds=3,
                                    image_size=(100, 80), timestamps=[i / 30 for i in range(90)])
    assert result["frame_index"] == 36
    assert result["candidate_frame_index"] == 45
    unknown = apply_contact_evidence(candidate, contact_evidence(), video_id="clip-a", duration_seconds=3,
                                     image_size=(100, 80))
    assert unknown["frame_index"] is None


def model_events(monkeypatch, response, preprocessing="opencv_source_bgr_area_v1"):
    from badminton import contact_model
    def infer(*args, **kwargs):
        assert kwargs.get("preprocessing") == preprocessing
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(contact_model, "infer_contact", infer)
    points = [(10 + i * 4, 10 + i * 3) for i in range(7)] + [(34, 28)] * 5
    frames = [np.zeros((100, 100, 3), np.uint8) for _ in points]
    return estimate_landings([track(points)], frames, [i / 30 for i in range(len(points))],
                             {"status": "ok", "polygon_px": [[0, 0], [99, 0], [99, 99], [0, 99]]},
                             {"contact_model_path": "mock-checkpoint.pt", "frame_preprocessing": preprocessing})


def qualified_contact_response():
    return {"contact_observed": True, "contact_px": [35, 30], "proposed_contact_px": [35, 30],
            "contact_radius_px": 0, "uncertainty_px": 2.5, "confidence": 0.995,
            "contact_evidence_source": "qualified_learned_contact_localization",
            "model_origin": "learned_temporal_contact", "reason_codes": [],
            "provenance": {"automatic_contact_qualified": True,
                           "preprocessing": "opencv_source_bgr_area_v1",
                           "runtime_preprocessing": "opencv_source_bgr_area_v1"}}


def test_qualified_contact_model_keeps_target_identity_separate(monkeypatch):
    events = model_events(monkeypatch, qualified_contact_response())
    result = events[0]
    assert result["contact_observed"] is True
    assert result["contact_px"] == [35, 30]
    assert result["candidate_px"] != result["contact_px"]
    assert result["identity_confidence"] <= 0.55
    assert "TARGET_IDENTITY_UNVERIFIED" in result["reason_codes"]
    assert result["inference_origin"] == "learned_contact_model"


@pytest.mark.parametrize("changes", [
    {"provenance": {"automatic_contact_qualified": False}},
    {"provenance": {"automatic_contact_qualified": True,
                    "preprocessing": "opencv_source_bgr_area_v1", "runtime_preprocessing": "h264_proxy"}},
    {"contact_px": [200, 30]}, {"uncertainty_px": -1},
    {"reason_codes": ["CONTACT_MODEL_CAPTURE_DOMAIN_MISMATCH"]},
])
def test_unqualified_or_invalid_model_cannot_assert_contact(monkeypatch, changes):
    result = model_events(monkeypatch, {**qualified_contact_response(), **changes})[0]
    assert result["contact_observed"] is False
    assert result["contact_px"] is None
    assert "FIRST_CONTACT_NOT_OBSERVED" in result["reason_codes"]


def test_model_error_is_visible_and_fails_closed(monkeypatch):
    result = model_events(monkeypatch, ValueError("invalid checkpoint"))[0]
    assert result["contact_model"]["status"] == "failed"
    assert "CONTACT_MODEL_ERROR" in result["reason_codes"]
    assert result["contact_observed"] is False


@pytest.mark.parametrize("preprocessing", [None, "h264_proxy_bgr_resize", "unknown"])
def test_contact_model_requires_known_matching_frame_preprocessing(monkeypatch, preprocessing):
    result = model_events(monkeypatch, qualified_contact_response(), preprocessing=preprocessing)[0]
    assert result["contact_observed"] is False
    assert result["contact_px"] is None
    assert result["model_proposed_contact_px"] == [35, 30]
    assert "CONTACT_MODEL_PREPROCESSING_MISMATCH" in result["reason_codes"]


@pytest.mark.parametrize("changes", [
    {"verified": False}, {"first_contact_observed": False}, {"target_identity_confirmed": False},
    {"video_id": "wrong-video"}, {"coordinate_system": "source_pixels"}, {"time": 5},
    {"contact_px": [-2, 10]}, {"contact_px": [100, 40]}, {"uncertainty_px": -1},
    {"contact_px": [float("nan"), 5]},
])
def test_unverified_or_out_of_bounds_contact_is_rejected(changes):
    evidence = {**contact_evidence(), **changes}
    with pytest.raises(ValueError):
        apply_contact_evidence({}, evidence, video_id="clip-a", duration_seconds=3, image_size=(100, 80))
