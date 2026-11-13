"""Synthetic scene contracts, not semantic floor or shuttle accuracy tests."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

import badminton.scene_context as context


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(35)
    frame = np.empty((270, 480, 3), np.uint8)
    frame[:135] = [45, 120, 35]
    frame[135:] = [90, 135, 175]
    noise = rng.integers(-4, 5, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return [frame.copy() for _ in range(17)]


def candidate(**extra):
    return {"frame_index": 1, "time": .04, "candidate_px": [240., 20.],
            "contact_observed": False, "identity_source": "unverified_image_space_consistency",
            "flight_trajectory_support": {"supported_for_review": True}, **extra}


@pytest.fixture(scope="module")
def annotated(frames):
    result = context.annotate_scene_context([candidate()], frames)[0]
    assert result["scene_context"]["evidence_valid"]
    assert context.scene_rank_factor(result) == .5
    return result


def test_actual_pixels_produce_only_bounded_ambiguous_context_without_mutation(frames, annotated):
    original = candidate()
    result = context.annotate_scene_context([original], frames)[0]
    assert original == candidate() and "scene_context" not in original
    assert {key: result[key] for key in original} == original
    evidence = result["scene_context"]
    assert evidence["status"] == "ambiguous_image_context"
    assert not any(evidence[key] for key in ("floor_verified", "identity_verified", "contact_verified",
                                           "coordinates_modified", "flight_eligibility_modified"))
    assert evidence["reason_codes"] == ["BOTTOM_CONNECTED_APPEARANCE_ONLY", "NOT_FLOOR_OR_CONTACT_PROOF"]
    assert annotated["candidate_px"] == [240., 20.]


def test_inside_region_and_small_boundary_uncertainty_are_neutral(frames):
    inside, boundary = context.annotate_scene_context([
        candidate(candidate_px=[240., 210.]), candidate(candidate_px=[240., 130.])], frames)
    assert context.scene_rank_factor(inside) == context.scene_rank_factor(boundary) == 1.
    assert inside["scene_context"]["median_distance_px"] == 0


def test_continuous_penalty_has_fixed_margin_and_fifty_percent_ceiling():
    assert context._factor(13.5, 270) == 1
    assert context._factor(20.25, 270) == .75
    assert context._factor(27, 270) == context._factor(10000, 270) == .5


@pytest.mark.parametrize("extra", [
    {"identity_source": "operator_selected_track"}, {"request_match": "selected"},
    {"contact_observed": True, "contact_evidence_source": "human_review"},
    {"contact_observed": True, "contact_evidence_source": "qualified_learned_contact_localization"},
])
def test_explicit_guidance_or_qualified_contact_is_always_neutral(frames, extra):
    result = context.annotate_scene_context([candidate(**extra)], frames)[0]
    assert context.scene_rank_factor(result) == result["scene_context"]["rank_factor"] == 1


def test_unqualified_contact_flag_does_not_disable_soft_context(frames):
    result = context.annotate_scene_context([candidate(contact_observed=True, contact_evidence_source="motion_heuristic")], frames)[0]
    assert context.scene_rank_factor(result) == .5


@pytest.mark.parametrize("change", [
    {"frame_index": 2}, {"frame_index": True}, {"candidate_px": [241., 20.]},
    {"candidate_px": [240., float("nan")]}, {"time": .05}, {"time": -1},
])
def test_geometry_and_source_time_binding_rejects_stale_evidence(annotated, change):
    assert context.scene_rank_factor({**annotated, **change}) == 1


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("method", "other"),
    ("parameters_sha256", "0" * 64), ("evidence_valid", False), ("status", "floor"),
    ("floor_verified", True), ("rank_factor", .1), ("rank_factor", .8),
    ("rank_factor", float("nan")), ("median_distance_px", -1),
    ("distance_px_each_source_view", [100.]), ("margin_px", 100),
    ("source_frame_binding_sha256", "0" * 64),
])
def test_invalid_or_modified_diagnostics_are_neutral(annotated, field, value):
    result = deepcopy(annotated)
    result["scene_context"][field] = value
    assert context.scene_rank_factor(result) == 1


def test_source_signature_and_query_pixel_binding_cannot_silently_change(annotated):
    result = deepcopy(annotated)
    result["scene_context"]["source_scene"]["sample_frame_sha256"][0] = "0" * 64
    assert context.scene_rank_factor(result) == 1
    result = deepcopy(annotated)
    result["scene_context"]["source_frame_binding"]["frame_sha256"] = "0" * 64
    assert context.scene_rank_factor(result) == 1


def test_scene_is_shared_and_each_unique_source_frame_is_hashed_once(frames, monkeypatch):
    calls = []
    original = context._frame_digest
    def counted(frame):
        calls.append(id(frame))
        return original(frame)
    monkeypatch.setattr(context, "_frame_digest", counted)
    results = context.annotate_scene_context([candidate() for _ in range(50)] + [candidate(frame_index=8)], frames)
    assert len(calls) == 8  # seven sample indices plus candidate frame 1; frame 8 is sampled
    assert len(set(calls)) == 8
    assert len({id(result["scene_context"]["source_scene"]) for result in results}) == 1
    assert all(context.scene_rank_factor(result) == .5 for result in results)


@pytest.mark.parametrize("bad_point", [None, 42, [], [1], [1, 2, 3], [[1], [2]], np.array(1), [True, 20]])
def test_malformed_candidate_geometry_is_neutral(frames, bad_point):
    result = context.annotate_scene_context([candidate(candidate_px=bad_point)], frames)[0]
    assert not result["scene_context"]["evidence_valid"]
    assert context.scene_rank_factor(result) == 1


def test_missing_or_wrong_source_images_remain_unknown(frames):
    for source in (None, [], frames[:6], [np.zeros((10, 10), np.uint8)] * 7,
                   [frames[0].astype(np.float32)] * 7):
        result = context.annotate_scene_context([candidate()], source)[0]
        assert result["scene_context"]["status"] == "unknown"
        assert context.scene_rank_factor(result) == 1


def test_overbroad_palette_region_cannot_claim_useful_context():
    frame = np.full((270, 480, 3), 100, np.uint8)
    result = context.annotate_scene_context([candidate()], [frame] * 7)[0]
    assert result["scene_context"]["status"] == "unknown"
    assert context.scene_rank_factor(result) == 1


def test_large_camera_translation_is_unknown(frames):
    shifted = [frame.copy() for frame in frames]
    shifted[0] = cv2.warpAffine(frames[0], np.float32([[1, 0, 30], [0, 1, 0]]),
                                (480, 270), borderMode=cv2.BORDER_REFLECT)
    result = context.annotate_scene_context([candidate()], shifted)[0]
    assert result["scene_context"]["status"] == "unknown"
    assert context.scene_rank_factor(result) == 1


def test_sampled_actual_pixel_change_has_new_source_signature(frames, annotated):
    changed = list(frames)
    changed[0] = frames[0].copy()
    changed[0][200, 200] = 255
    result = context.annotate_scene_context([candidate()], changed)[0]
    assert result["scene_context"]["source_scene"]["signature_sha256"] != annotated["scene_context"]["source_scene"]["signature_sha256"]


def test_reader_is_neutral_for_absent_or_malformed_metadata():
    for evidence in (None, {}, [], {"schema_version": 1, "rank_factor": .5}):
        assert context.scene_rank_factor({**candidate(), "scene_context": evidence}) == 1
    assert context.scene_rank_factor(candidate(contact_observed=True, contact_evidence_source=[])) == 1
