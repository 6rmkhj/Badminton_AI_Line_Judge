import copy

import pytest

from badminton.decision import decide
from badminton.identity import apply_identity


TIMES = [i / 10 for i in range(31)]
CALIBRATION = {"image_width": 200, "image_height": 100}


def track(key, start=0, end=20, *, x=20, ambiguous=False, source_ids=None):
    return {"track_id": key, "source_track_ids": source_ids or [key],
            "identity_ambiguous": ambiguous, "gaps": [], "reconnection_hypotheses": [],
            "observations": [{"frame_index": i, "time": i / 10, "x": x + i, "y": 20,
                              "observed": True, "confidence": .9} for i in range(start, end + 1)]}


def landing(key=1, time=1.5):
    return {"track_id": key, "frame_index": round(time * 10), "time": time,
            "contact_observed": False, "contact_px": None, "confidence": .6,
            "identity_confidence": .45, "reason_codes": ["FIRST_CONTACT_NOT_OBSERVED", "TARGET_IDENTITY_UNVERIFIED"]}


def config(**extra):
    return {"hit_anchors": [{"time": .5, "point_px": [25, 20], "radius_px": 8,
                            "time_tolerance_seconds": .06}], **extra}


def test_manual_hit_uniquely_selects_target_without_creating_contact_or_ground_truth():
    original = [landing(), landing(2)]
    snapshot = copy.deepcopy(original)
    result = apply_identity(original, [track(1), track(2, x=100)], TIMES, CALIBRATION, config())
    selected, other = result
    assert selected["identity_confidence"] >= .8
    assert selected["identity_evidence"]["status"] == "matched"
    assert selected["request_match"] == "selected"
    assert other["request_match"] == "rejected"
    assert selected["contact_observed"] is False and selected["contact_px"] is None
    assert selected["evaluation_eligible"] is False and selected["assistance_used"] is True
    assert selected["identity_evidence"]["provides_ground_truth"] is False
    assert decide(selected, {})["decision"] == "REVIEW"
    assert "FIRST_CONTACT_NOT_OBSERVED" in selected["reason_codes"]
    assert "TARGET_IDENTITY_UNVERIFIED" not in selected["reason_codes"]
    assert original == snapshot


def test_event_window_filters_only_requested_interval_and_is_inclusive():
    events = [landing(time=.5), landing(time=1), landing(time=2), landing(time=2.5)]
    result = apply_identity(events, [], TIMES, CALIBRATION, {"event_window_seconds": [1, 2]})
    assert [event["time"] for event in result] == [1, 2]
    assert all(event["requested_event_window_seconds"] == [1, 2] for event in result)
    assert apply_identity(events, [], TIMES, CALIBRATION, {"event_window_seconds": [1.1, 1.2]}) == []


@pytest.mark.parametrize("bad", [[2, 1], [-1, 1], [0, 3.1], [1], [0, float("nan")]])
def test_invalid_requested_window_fails_closed(bad):
    with pytest.raises(ValueError):
        apply_identity([landing()], [], TIMES, CALIBRATION, {"event_window_seconds": bad})


@pytest.mark.parametrize("change", [{"time": 4}, {"point_px": [-1, 20]}, {"point_px": [200, 20]},
                                    {"radius_px": 0}, {"time_tolerance_seconds": -1},
                                    {"coordinate_system": "source_pixels"}, {"time": float("inf")}])
def test_malformed_anchors_cannot_be_accepted(change):
    cfg = config()
    cfg["hit_anchors"][0].update(change)
    with pytest.raises(ValueError):
        apply_identity([landing()], [track(1)], TIMES, CALIBRATION, cfg)


def test_two_nearby_shuttles_preserve_ambiguous_identity():
    result = apply_identity([landing(), landing(2)], [track(1), track(2, x=21)], TIMES, CALIBRATION, config())
    assert all(item["identity_confidence"] < .8 for item in result)
    assert all(item["request_match"] == "ambiguous" for item in result)
    assert all("TARGET_HIT_MATCH_AMBIGUOUS" in item["reason_codes"] for item in result)
    assert len(result[0]["identity_evidence"]["anchors"][0]["matches"]) == 2


def test_anchor_after_landing_cannot_retroactively_select_shuttle():
    result = apply_identity([landing(time=.3)], [track(1)], TIMES, CALIBRATION, config())[0]
    assert result["identity_confidence"] == 0
    assert "NO_PRECEDING_TARGET_HIT_ANCHOR" in result["reason_codes"]


def edge(before, after, score=0):
    return {"from_track_id": before, "to_track_id": after, "score": score}


def test_reverse_tracklet_graph_links_hit_to_landing_and_exports_observed_indices():
    a, b = track(1, 0, 7), track(2, 10, 20)
    a["reconnection_hypotheses"] = [edge(1, 2)]
    result = apply_identity([landing(2)], [a, b], TIMES, CALIBRATION, config())[0]
    assert result["identity_evidence"]["status"] == "matched"
    assert result["identity_evidence"]["anchors"][0]["hypotheses"][0]["track_ids"] == ["1", "2"]
    assert result["backtrack_frame_indices"] == [*range(8), *range(10, 16)]
    assert result["contact_observed"] is False


def test_competing_graph_paths_are_preserved_and_block_confirmation():
    a, b, c, d = track(1, 0, 6), track(2, 8, 10), track(3, 8, 10), track(4, 12, 20)
    a["reconnection_hypotheses"] = [edge(1, 2), edge(1, 3)]
    b["reconnection_hypotheses"] = [edge(2, 4)]
    c["reconnection_hypotheses"] = [edge(3, 4)]
    result = apply_identity([landing(4)], [a, b, c, d], TIMES, CALIBRATION, config())[0]
    assert "TARGET_IDENTITY_MULTIPLE_GRAPH_PATHS" in result["reason_codes"]
    assert result["identity_confidence"] <= .4
    assert len(result["identity_evidence"]["anchors"][0]["hypotheses"]) == 2


def test_graph_rival_without_a_landing_still_blocks_identity():
    a, b, c = track(1, 0, 7), track(2, 10, 20), track(3, 10, 20, x=100)
    a["reconnection_hypotheses"] = [edge(1, 2), edge(1, 3)]
    result = apply_identity([landing(2)], [a, b, c], TIMES, CALIBRATION, config())[0]
    assert result["identity_confidence"] <= .4
    assert "TARGET_IDENTITY_UNRESOLVED_GRAPH_BRANCH" in result["reason_codes"]


def test_anchor_event_windows_keep_separate_rallies_independent():
    cfg = config()
    cfg["hit_anchors"][0]["event_window_seconds"] = [1, 1.5]
    cfg["hit_anchors"].append({"time": 1.8, "point_px": [118, 20], "radius_px": 5,
                               "event_window_seconds": [2, 2.5]})
    result = apply_identity([landing(1, 1.2), landing(2, 2.2)],
                            [track(1, 0, 15), track(2, 16, 30, x=100)], TIMES, CALIBRATION, cfg)
    assert all(item["identity_evidence"]["status"] == "matched" for item in result)
    assert [item["identity_evidence"]["anchors"][0]["anchor_index"] for item in result] == [0, 1]


def test_anchor_cannot_upgrade_a_temporal_gap_near_contact():
    event = landing()
    event["reason_codes"].append("TEMPORAL_GAP_NEAR_CANDIDATE")
    result = apply_identity([event], [track(1)], TIMES, CALIBRATION, config())[0]
    assert result["identity_confidence"] <= .4
    assert "TARGET_IDENTITY_GAP_NEAR_EVENT" in result["reason_codes"]


@pytest.mark.parametrize("edge_value", [edge(2, 1), edge(1, 2, float("nan")), edge(1, 2, -1)])
def test_noncausal_or_invalid_graph_edges_do_not_prove_identity(edge_value):
    a, b = track(1, 0, 7), track(2, 10, 20)
    a["reconnection_hypotheses"] = [edge_value]
    result = apply_identity([landing(2)], [a, b], TIMES, CALIBRATION, config())[0]
    assert result["request_match"] == "rejected"


def test_merged_original_track_ids_resolve_to_current_owner():
    a, b = track(10, 0, 7, source_ids=[1, 10]), track(20, 10, 20, source_ids=[2, 20])
    a["reconnection_hypotheses"] = [edge(1, 2)]
    result = apply_identity([landing(20)], [a, b], TIMES, CALIBRATION, config())[0]
    assert result["identity_evidence"]["status"] == "matched"
    assert result["identity_evidence"]["anchors"][0]["hypotheses"][0]["track_ids"] == ["10", "20"]


def test_ambiguous_track_or_unresolved_gap_cannot_be_overridden_by_anchor():
    for current in [track(1, ambiguous=True), track(1)]:
        if not current["identity_ambiguous"]:
            current["gaps"] = [{"start_frame": 8, "resolved": False}]
        result = apply_identity([landing()], [current], TIMES, CALIBRATION, config())[0]
        assert result["identity_confidence"] <= .4
        assert result["request_match"] == "ambiguous"


def test_disagreeing_earlier_hit_anchors_do_not_cherry_pick_one_match():
    cfg = config()
    cfg["hit_anchors"].append({"time": 1, "point_px": [110, 20], "radius_px": 5})
    result = apply_identity([landing()], [track(1), track(2, x=100)], TIMES, CALIBRATION, cfg)[0]
    assert result["identity_confidence"] <= .4
    assert "TARGET_HIT_NOT_LINKED_TO_LANDING" in result["reason_codes"]


def test_no_identity_request_preserves_legacy_selection_and_contact_fields():
    event = {**landing(), "identity_source": "operator_selected_track", "identity_confidence": .9,
             "contact_observed": True, "contact_px": [30, 40]}
    assert apply_identity([event], [], TIMES, CALIBRATION) == [event]
