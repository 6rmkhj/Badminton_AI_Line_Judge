"""Guidance changes which evidence is shown, not physical contact or identity."""
from copy import deepcopy

import numpy as np
import pytest

from badminton.events import (estimate_landings, prioritize_review_candidates,
    refine_source_contacts, review_refinement_regions, review_region_indices)


REGION = {"frame_start": 8, "frame_end": 12, "bbox": [20, 15, 15, 10]}


def landing(index=3, point=(5, 5), confidence=.9, track_id=1):
    return {"frame_index": index, "time": index/30, "candidate_px": list(point),
            "contact_observed": False, "contact_px": None, "identity_confidence": .4,
            "confidence": confidence, "track_id": track_id, "reason_codes": ["FIRST_CONTACT_NOT_OBSERVED"]}


class SourceFrames:
    def __init__(self):
        self.requested = []

    def __len__(self):
        return 20

    def __getitem__(self, index):
        assert not self.requested or self.requested[-1] < index
        self.requested.append(index)
        return np.zeros((60, 80, 3), np.uint8)


def test_native_budget_reserves_guided_candidate_without_score_or_contact_promotion(tmp_path):
    candidates = [landing(confidence=.99-i*.01, track_id=i) for i in range(8)]
    candidates.append(landing(10, (25, 20), .1, 8))
    before = deepcopy(candidates)
    source = SourceFrames()
    result = refine_source_contacts(candidates, source, [i/30 for i in range(20)], (40, 30),
        {"review_refinement_regions": [REGION], "source_contact_max_candidates": 2,
         "source_contact_patch_size": 24}, tmp_path)
    assert candidates == before
    assert len(result) == len(candidates)
    selected = [row for row in result if row["source_contact_evidence"]["status"] == "complete"]
    assert {row["track_id"] for row in selected} == {0, 8}
    assert result[-1]["source_contact_evidence"]["selection_reason"] == "explicit_review_region"
    assert len(source.requested) <= 2*5
    for original, current in zip(before, result):
        for field in ("candidate_px", "frame_index", "confidence", "identity_confidence", "contact_observed", "contact_px"):
            assert current[field] == original[field]
        assert current["assistance_used"] is True
        assert current["evaluation_eligible"] is False


def test_event_cap_preserves_focused_low_score_track_before_native_evidence():
    tracks = []
    for track_id, first, origin, score in ((1, 0, (5, 5), .99), (2, 20, (70, 50), .1)):
        points = [(origin[0]+2*i, origin[1]+2*i) for i in range(7)] + [(origin[0]+12, origin[1]+12)]*5
        tracks.append({"track_id": track_id, "observations": [
            {"frame_index": first+i, "time": (first+i)/60, "x": x, "y": y,
             "confidence": score, "radius": 1} for i, (x, y) in enumerate(points)]})
    frames = [np.zeros((100, 100, 3), np.uint8)]*40
    times = [i/60 for i in range(40)]
    baseline = estimate_landings(tracks, frames, times, {}, {"max_candidates": 1})
    full = estimate_landings(tracks, frames, times, {}, {"max_candidates": 100})
    cfg = {"max_candidates": 1, "review_refinement_regions": [
        {"frame_start": 20, "frame_end": 31, "bbox": [68, 48, 20, 20]}]}
    guided = estimate_landings(tracks, frames, times, {}, cfg)
    assert baseline[0]["track_id"] == 1
    assert len(guided) == 1 and guided[0]["track_id"] == 2
    original = next(row for row in full if row["track_id"] == 2 and row["frame_index"] == guided[0]["frame_index"])
    for field in ("confidence", "identity_confidence", "contact_observed", "contact_px"):
        assert guided[0][field] == original[field]
    assert guided[0]["review_refinement"]["prioritization_only"] is True


@pytest.mark.parametrize("candidate", [landing(7, (25, 20)), landing(13, (25, 20)),
    landing(10, (19, 20)), landing(10, (35, 20)), landing(10, (25, 25)),
    {"frame_index": True, "candidate_px": [25, 20]},
    {"frame_index": 10, "candidate_px": [float('nan'), 20]},
    {"frame_index": 10, "candidate_px": [True, 20]},
    {"frame_index": 10, "candidate_px": None}])
def test_guidance_requires_valid_frame_and_point_inside_both_bounds(candidate):
    assert review_region_indices(candidate, [REGION]) == []


@pytest.mark.parametrize("regions", [None, {}, [None],
    [{"frame_start": True, "frame_end": 12, "bbox": [20, 15, 15, 10]}],
    [{"frame_start": 8, "frame_end": 20, "bbox": [20, 15, 15, 10]}],
    [{"frame_start": 8, "frame_end": 12, "bbox": [20, 15, 30, 10]}],
    [{"frame_start": 8, "frame_end": 12, "bbox": [20, 15, float('nan'), 10]}]])
def test_invalid_guidance_fails_before_reading_native_frames(regions):
    source = SourceFrames()
    with pytest.raises(ValueError):
        refine_source_contacts([landing()], source, [i/30 for i in range(20)], (40, 30),
                               {"review_refinement_regions": regions})
    assert not source.requested


def test_no_matching_region_preserves_original_order_and_all_candidates():
    original = [landing(confidence=.2), landing(confidence=.8)]
    result = prioritize_review_candidates(original, [REGION])
    assert result == [original[1], original[0]]
    assert len(result) == len(original)
    assert review_refinement_regions({}, (40, 30), 20) == []


@pytest.mark.parametrize("has_match", [True, False])
def test_pipeline_prioritizes_preview_and_evidence_but_aggregates_every_landing(tmp_path, monkeypatch, has_match):
    from badminton import calibration, detection, events, identity, pipeline, tracking, decision
    frames = [np.zeros((30, 40, 3), np.uint8)]*20
    times = [i/30 for i in range(20)]
    candidates = [landing(confidence=.99-i*.01, track_id=i) for i in range(7)]
    candidates.append(landing(10, (25, 20) if has_match else (5, 5), .1, 7))
    monkeypatch.setattr(pipeline, "prepare_preview", lambda *args: (
        {"width": 80, "height": 60, "warnings": []},
        {"width": 40, "height": 30, "frame_count": 20, "timestamps": times.copy(), "warnings": []}, '0'*64))
    monkeypatch.setattr(pipeline, "load_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr(calibration, "calibrate", lambda *args: {"status": "unconfirmed"})
    monkeypatch.setattr(calibration, "assess_camera_stability", lambda *args: {})
    monkeypatch.setattr(calibration, "guard_calibration_stability", lambda result, *args: result)
    monkeypatch.setattr(detection, "detect_sequence", lambda *args: [])
    monkeypatch.setattr(detection, "summarize_detection_quality", lambda *args: {})
    monkeypatch.setattr(tracking, "track_detections", lambda *args: [])
    def estimated(*args):
        assert args[-1]["review_refinement_regions"] == [REGION]
        return deepcopy(candidates)
    monkeypatch.setattr(events, "estimate_landings", estimated)
    monkeypatch.setattr(identity, "apply_identity", lambda values, *args: values)
    monkeypatch.setattr(decision, "decide", lambda row, *args: {
        "decision": "IN" if row["track_id"] == 7 else "OUT", "reason_codes": []})
    result = pipeline.analyze_video(tmp_path/'input.mp4', tmp_path/'run', {
        "detector": {"highres_enabled": False, "refinement_regions": [REGION]},
        "events": {"source_contact_enabled": False}, "evidence": {"annotated_clip": False},
        "source_crop_count": 0})
    assert len(result['landings']) == 8 and result['identity']['requested_event_count'] == 8
    assert result['decision']['decision'] == 'REVIEW'
    assert 'MULTIPLE_LANDING_CANDIDATES' in result['decision']['reason_codes']
    assert result['automatic_evaluation_eligible'] is False
    expected = 'event_007' if has_match else 'event_000'
    assert result['selected_landing']['event_id'] == expected
    assert result['artifacts']['evidence'][0]['event_id'] == expected
    assert len(result['artifacts']['evidence']) == 6
    assert result['review_refinement']['selected_in_requested_region'] is has_match
    assert result['review_refinement']['status'] == ('matched_candidates' if has_match else 'no_matching_candidate')
    assert result['review_refinement']['matching_event_ids'] == (['event_007'] if has_match else [])
