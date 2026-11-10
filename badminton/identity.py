"""Requested-event filtering and conservative target identity from manual hits.

A target-court hit is permitted as an initial experimental anchor by the plan.
It is identity assistance, never observed floor contact or a ground-truth label.
Tracklet links are traversed backwards from a landing; competing paths remain
explicit and cannot increase identity confidence above the decision threshold.
"""
from __future__ import annotations

import copy
import math


def _finite(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return number


def _configuration(config, timestamps, calibration):
    cfg = dict(config or {})
    times = [_finite(time, "timestamp") for time in timestamps]
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("identity requires nonempty strictly increasing timestamps")
    window = cfg.get("event_window_seconds")
    if window is not None:
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError("event_window_seconds must be [start, end]")
        window = [_finite(value, "event window") for value in window]
        if not times[0] <= window[0] <= window[1] <= times[-1]:
            raise ValueError("event window must be ordered and within video timestamps")
    anchors = cfg.get("hit_anchors", [])
    if not isinstance(anchors, list):
        raise ValueError("hit_anchors must be a list")
    if len(anchors) > 100:
        raise ValueError("at most 100 hit anchors are supported")
    parsed = []
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, dict):
            raise ValueError("each hit anchor must be an object")
        if anchor.get("coordinate_system", "analysis_pixels") != "analysis_pixels":
            raise ValueError("hit anchors must use analysis_pixels")
        if anchor.get("source", "manual_hit_anchor") != "manual_hit_anchor":
            raise ValueError("hit anchor source must be manual_hit_anchor")
        point = anchor.get("point_px")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("hit anchor point_px must contain [x, y]")
        point = [_finite(value, "anchor coordinate") for value in point]
        time = _finite(anchor.get("time"), "anchor time")
        if not times[0] <= time <= times[-1] or min(point) < 0:
            raise ValueError("hit anchor is outside video time/image bounds")
        for value, key in zip(point, ("image_width", "image_height")):
            if calibration.get(key) is not None and value >= _finite(calibration[key], key, positive=True):
                raise ValueError("hit anchor is outside analysis image")
        scope = anchor.get("event_window_seconds")
        if scope is not None:
            if not isinstance(scope, (list, tuple)) or len(scope) != 2:
                raise ValueError("anchor event_window_seconds must be [start, end]")
            scope = [_finite(value, "anchor event window") for value in scope]
            if not times[0] <= scope[0] <= scope[1] <= times[-1]:
                raise ValueError("anchor event window must be ordered and within video timestamps")
        parsed.append({"anchor_index": index, "time": time, "point_px": point,
                       "event_window_seconds": scope,
                       "radius_px": _finite(anchor.get("radius_px", 30), "anchor radius", positive=True),
                       "time_tolerance_seconds": _finite(anchor.get("time_tolerance_seconds", .08), "anchor time tolerance", positive=True)})
    gap = _finite(cfg.get("max_graph_gap_seconds", .75), "graph gap", positive=True)
    margin = _finite(cfg.get("ambiguity_margin", .08), "ambiguity margin")
    if not 0 <= margin <= 1:
        raise ValueError("ambiguity_margin must be between zero and one")
    return window, parsed, gap, margin


def _graph(tracks, max_gap):
    by_id, owners = {}, {}
    for track in tracks:
        key = str(track["track_id"])
        if key in by_id:
            raise ValueError("duplicate track_id")
        by_id[key] = track
        for source_id in [track["track_id"], *track.get("source_track_ids", [])]:
            source = str(source_id)
            if source in owners and owners[source] != key:
                raise ValueError("source track belongs to multiple merged tracklets")
            owners[source] = key
    edges = {}
    for track in tracks:
        for edge in track.get("reconnection_hypotheses", []):
            before, after = owners.get(str(edge.get("from_track_id"))), owners.get(str(edge.get("to_track_id")))
            if before is None or after is None or before == after:
                continue
            left, right = by_id[before].get("observations", []), by_id[after].get("observations", [])
            if not left or not right:
                continue
            gap = float(right[0]["time"]) - float(left[-1]["time"])
            score = edge.get("score")
            if (not math.isfinite(gap) or not 0 < gap <= max_gap
                    or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
                continue
            # The graph supplies geometric hypotheses, never verified identity.
            quality = 1 - .15 * score - .10 * gap / max_gap
            key = (before, after)
            edges[key] = max(edges.get(key, 0), quality)
    incoming = {}
    for (before, after), quality in edges.items():
        incoming.setdefault(after, []).append((before, quality))
    return by_id, incoming, len(edges)


def _paths_backwards(target, source, incoming, *, max_paths=16):
    paths, overflow = [], False
    pending = [(target, [target], 1.0)]
    explored = 0
    while pending:
        node, reverse_path, score = pending.pop()
        explored += 1
        if explored > 4096:
            return paths, True
        if node == source:
            paths.append({"track_ids": list(reversed(reverse_path)), "graph_score": score})
            if len(paths) >= max_paths:
                overflow = bool(pending)
                break
            continue
        for predecessor, quality in incoming.get(node, []):
            if predecessor not in reverse_path:
                pending.append((predecessor, [*reverse_path, predecessor], min(score, quality)))
    return sorted(paths, key=lambda path: -path["graph_score"]), overflow


def _anchor_matches(anchor, by_id, event_time):
    matches = []
    for key, track in by_id.items():
        best = None
        for observation in track.get("observations", []):
            time = float(observation["time"])
            if time > event_time or observation.get("observed") is False:
                continue
            delta = abs(time - anchor["time"])
            distance = math.hypot(float(observation["x"]) - anchor["point_px"][0],
                                  float(observation["y"]) - anchor["point_px"][1])
            if not math.isfinite(delta) or not math.isfinite(distance) or delta > anchor["time_tolerance_seconds"] or distance > anchor["radius_px"]:
                continue
            score = .95 - .10 * distance / anchor["radius_px"] - .05 * delta / anchor["time_tolerance_seconds"]
            match = {"track_id": key, "frame_index": int(observation["frame_index"]),
                     "distance_px": round(distance, 5), "time_delta_seconds": round(delta, 6), "score": score}
            if best is None or score > best["score"]:
                best = match
        if best:
            matches.append(best)
    return sorted(matches, key=lambda match: (-match["score"], match["track_id"]))


def apply_identity(landings, tracks, timestamps, calibration, config=None):
    """Return copied, optionally filtered landings with auditable identity evidence.

    ``event_window_seconds`` limits the requested event, without creating an
    event when no candidate exists. ``hit_anchors`` contains operator-provided
    target-court hits as ``time``, ``point_px``, optional ``radius_px`` and
    ``time_tolerance_seconds``. All image coordinates are analysis pixels.
    Every preceding anchor must agree with one unambiguous causal tracklet
    path. Contact fields are never modified. Assisted predictions carry
    ``evaluation_eligible=False`` and must be excluded from automatic metrics.
    """
    calibration = calibration or {}
    window, anchors, max_gap, margin = _configuration(config, timestamps, calibration)
    result = copy.deepcopy(landings)
    if window is not None:
        result = [item for item in result if window[0] <= _finite(item.get("time"), "landing time") <= window[1]]
        for item in result:
            item["requested_event_window_seconds"] = list(window)
    if not anchors:
        return result
    by_id, incoming, edge_count = _graph(tracks, max_gap)
    outgoing = {}
    for successor, predecessors in incoming.items():
        for predecessor, quality in predecessors:
            outgoing.setdefault(predecessor, []).append((successor, quality))
    for landing in result:
        event_time = _finite(landing.get("time"), "landing time")
        target = str(landing.get("track_id"))
        preceding = [anchor for anchor in anchors if anchor["time"] <= event_time
                     and (anchor["event_window_seconds"] is None
                          or anchor["event_window_seconds"][0] <= event_time <= anchor["event_window_seconds"][1])]
        evidence = {"source": "manual_hit_anchor", "coordinate_system": "analysis_pixels",
                    "confidence_kind": "assisted_identity_support_not_calibrated_probability",
                    "graph_edge_count": edge_count, "anchors": [],
                    "provides_contact_evidence": False, "provides_ground_truth": False}
        reasons, scores, support_paths = [], [], []
        if target not in by_id:
            reasons.append("LANDING_TRACK_NOT_FOUND")
        if not preceding:
            reasons.append("NO_PRECEDING_TARGET_HIT_ANCHOR")
        if "TEMPORAL_GAP_NEAR_CANDIDATE" in landing.get("reason_codes", []):
            reasons.append("TARGET_IDENTITY_GAP_NEAR_EVENT")
        for anchor in preceding:
            matches = _anchor_matches(anchor, by_id, event_time)
            hypotheses, overflow = [], False
            for match in matches:
                paths, limited = _paths_backwards(target, match["track_id"], incoming)
                overflow |= limited
                for path in paths:
                    hypotheses.append({**match, **path, "score": min(match["score"], path["graph_score"])})
            hypotheses.sort(key=lambda entry: -entry["score"])
            entry = {**anchor, "matches": matches[:16], "hypotheses": hypotheses[:16],
                     "search_truncated": overflow or len(matches) > 16 or len(hypotheses) > 16}
            evidence["anchors"].append(entry)
            if not hypotheses:
                reasons.append("TARGET_HIT_NOT_LINKED_TO_LANDING")
                continue
            best = hypotheses[0]
            support_paths.append(best["track_ids"])
            scores.append(best["score"])
            if entry["search_truncated"]:
                reasons.append("TARGET_IDENTITY_SEARCH_TRUNCATED")
            if len(hypotheses) > 1 and best["score"] - hypotheses[1]["score"] <= margin:
                reasons.append("TARGET_IDENTITY_MULTIPLE_GRAPH_PATHS")
            competing = [match for match in matches if match["track_id"] != best["track_id"]]
            if competing and best["score"] - competing[0]["score"] <= margin:
                reasons.append("TARGET_HIT_MATCH_AMBIGUOUS")
            for path_index, key in enumerate(best["track_ids"]):
                track = by_id[key]
                if track.get("identity_ambiguous"):
                    reasons.append("TARGET_IDENTITY_AMBIGUOUS_TRACKLET")
                if any(not gap.get("resolved", False) and float(gap.get("start_frame", -1)) <= landing.get("frame_index", math.inf)
                       for gap in track.get("gaps", [])):
                    reasons.append("TARGET_IDENTITY_UNRESOLVED_GAP")
                if path_index + 1 < len(best["track_ids"]):
                    chosen = best["track_ids"][path_index + 1]
                    selected_quality = next(quality for successor, quality in outgoing[key] if successor == chosen)
                    if any(successor != chosen and selected_quality - quality <= margin
                           and float(by_id[successor]["observations"][0]["time"]) <= event_time
                           for successor, quality in outgoing[key]):
                        reasons.append("TARGET_IDENTITY_UNRESOLVED_GRAPH_BRANCH")
        raw_score = min(scores) if scores else 0.0
        if raw_score < .8 and scores:
            reasons.append("TARGET_IDENTITY_ANCHOR_SUPPORT_LOW")
        accepted = not reasons and bool(scores)
        evidence.update(status="matched" if accepted else "review", reason_codes=list(dict.fromkeys(reasons)),
                        support_score=round(raw_score, 5))
        landing.update(identity_confidence=round(raw_score if accepted else min(raw_score, .4), 5),
                       identity_source="manual_hit_anchor_tracklet_backtracking", identity_evidence=evidence,
                       identity_assisted=True, assistance_used=True, evaluation_eligible=False,
                       request_match="selected" if accepted else ("ambiguous" if scores else "rejected"),
                       automatic_evaluation_eligible=False)
        existing = list(landing.get("reason_codes", []))
        if accepted:
            existing = [reason for reason in existing if reason not in {"TARGET_IDENTITY_UNVERIFIED", "OTHER_TRACK_THAN_REQUESTED"}]
            existing.append("TARGET_IDENTITY_SUPPORTED_BY_MANUAL_HIT")
        landing["reason_codes"] = list(dict.fromkeys([*existing, *reasons]))
        relevant = {key for path in support_paths for key in path}
        landing["backtrack_frame_indices"] = sorted({int(observation["frame_index"])
            for key in relevant for observation in by_id[key].get("observations", [])
            if float(observation["time"]) <= event_time and observation.get("observed") is not False})
    return result
