"""Conservative multi-hypothesis shuttle tracklets in image coordinates."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def _inside(point, region):
    x, y, width, height = region.get("bbox", [0, 0, 0, 0])
    return x <= point[0] <= x + width and y <= point[1] <= y + height


def _velocity(observations, tail=True):
    observations = observations[-4:] if tail else observations[:4]
    if len(observations) < 2:
        return np.zeros(2)
    times = np.array([observation["time"] for observation in observations])
    times = times - times.mean()
    xy = np.array([[observation["x"], observation["y"]] for observation in observations])
    denominator = float(times @ times)
    return (times[:, None] * (xy - xy.mean(axis=0))).sum(axis=0) / denominator if denominator > 1e-12 else np.zeros(2)


class _Track:
    def __init__(self, track_id, observation):
        self.track_id = track_id
        self.state = np.array([observation["x"], observation["y"], 0.0, 0.0], dtype=float)
        self.covariance = np.diag([16.0, 16.0, 40000.0, 40000.0])
        self.time = observation["time"]
        self.observations = [observation]
        self.misses = []
        self.gaps = []
        self.ambiguous = False
        self.ambiguity_events = []
        self.reconnections = []

    def predict(self, time, acceleration):
        dt = time - self.time
        transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        noise_gain = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + noise_gain @ noise_gain.T * acceleration ** 2
        self.time = time

    def observe(self, observation):
        measurement = np.array([observation["x"], observation["y"]])
        projection = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        variance = max(2.0, float(observation.get("radius", 2))) ** 2
        innovation = measurement - projection @ self.state
        innovation_covariance = projection @ self.covariance @ projection.T + np.eye(2) * variance
        gain = self.covariance @ projection.T @ np.linalg.inv(innovation_covariance)
        self.state += gain @ innovation
        self.covariance = (np.eye(4) - gain @ projection) @ self.covariance
        if len(self.observations) == 1:
            dt = observation["time"] - self.observations[-1]["time"]
            if dt > 0:
                self.state[2:] = (measurement - np.array([self.observations[-1]["x"], self.observations[-1]["y"]])) / dt
        if self.misses:
            gap = {"start_frame": self.misses[0]["frame_index"], "end_frame": self.misses[-1]["frame_index"],
                   "missing_frame_count": len(self.misses),
                   "duration_s": observation["time"] - self.observations[-1]["time"],
                   "possible_occlusion": any(m["possible_occlusion"] for m in self.misses),
                   "resolved": True, "observations_interpolated": False}
            self.gaps.append(gap)
            self.reconnections.append({"from_frame": self.observations[-1]["frame_index"],
                                       "to_frame": observation["frame_index"],
                                       "method": "unique_gated_kalman_assignment", "verified_identity": False})
            self.misses = []
        self.observations.append(observation)

    def output(self):
        gaps = list(self.gaps)
        if self.misses:
            gaps.append({"start_frame": self.misses[0]["frame_index"], "end_frame": self.misses[-1]["frame_index"],
                         "missing_frame_count": len(self.misses),
                         "duration_s": self.misses[-1]["time"] - self.observations[-1]["time"],
                         "possible_occlusion": any(m["possible_occlusion"] for m in self.misses),
                         "resolved": False, "observations_interpolated": False})
        return {"track_id": self.track_id, "observations": self.observations, "gaps": gaps,
                "identity_ambiguous": self.ambiguous, "ambiguity_events": self.ambiguity_events,
                "reconnections": self.reconnections, "source_track_ids": [self.track_id],
                "reconnection_hypotheses": [], "method": "gated_kalman_hungarian_tracklets"}


def _offline_links(tracks, config):
    """Keep forward/backward-consistent gap hypotheses; merge only unique pairs."""
    max_gap = float(config.get("offline_max_gap_seconds", 0.6))
    max_residual = float(config.get("offline_max_residual_px", 28.0))
    links = []
    for before in tracks:
        if len(before["observations"]) < 3 or before["identity_ambiguous"]:
            continue
        end = before["observations"][-1]
        velocity_before = _velocity(before["observations"])
        for after in tracks:
            if before is after or len(after["observations"]) < 3 or after["identity_ambiguous"]:
                continue
            start = after["observations"][0]
            dt = start["time"] - end["time"]
            if not 0 < dt <= max_gap or start["frame_index"] <= end["frame_index"] + 1:
                continue
            velocity_after = _velocity(after["observations"], tail=False)
            displacement = np.array([start["x"] - end["x"], start["y"] - end["y"]])
            forward = float(np.linalg.norm(displacement - velocity_before * dt))
            backward = float(np.linalg.norm(displacement - velocity_after * dt))
            speed_product = float(np.linalg.norm(velocity_before) * np.linalg.norm(velocity_after))
            cosine = float(velocity_before @ velocity_after / speed_product) if speed_product > 1 else 1.0
            if max(forward, backward) <= max_residual and cosine > 0.4:
                links.append({"from_track_id": before["track_id"], "to_track_id": after["track_id"],
                              "gap_seconds": dt, "forward_residual_px": forward,
                              "backward_residual_px": backward, "direction_cosine": cosine,
                              "score": max(forward, backward) / max_residual,
                              "verified_identity": False, "method": "forward_backward_velocity_consistency"})
    outgoing, incoming = {}, {}
    for edge in links:
        outgoing.setdefault(edge["from_track_id"], []).append(edge)
        incoming.setdefault(edge["to_track_id"], []).append(edge)
    by_id = {track["track_id"]: track for track in tracks}
    for edge in links:
        by_id[edge["from_track_id"]]["reconnection_hypotheses"].append(dict(edge))
        by_id[edge["to_track_id"]]["reconnection_hypotheses"].append(dict(edge))
    # Ambiguous graph links are retained as hypotheses and never collapsed.
    for track in tracks:
        if len(outgoing.get(track["track_id"], [])) > 1 or len(incoming.get(track["track_id"], [])) > 1:
            track["identity_ambiguous"] = True
    selected = [edge for edge in links if len(outgoing[edge["from_track_id"]]) == 1 and len(incoming[edge["to_track_id"]]) == 1]
    if not config.get("offline_merge", True):
        return tracks
    parent = {track["track_id"]: track["track_id"] for track in tracks}

    def leader(key):
        while parent[key] != key:
            key = parent[key]
        return key

    for edge in sorted(selected, key=lambda e: by_id[e["to_track_id"]]["observations"][0]["time"]):
        left, right = leader(edge["from_track_id"]), leader(edge["to_track_id"])
        if left == right or by_id[left]["identity_ambiguous"] or by_id[right]["identity_ambiguous"]:
            continue
        before, after = by_id[left], by_id[right]
        end, start = before["observations"][-1], after["observations"][0]
        if end["time"] >= start["time"]:
            continue
        before["gaps"] = [gap for gap in before["gaps"] if gap["resolved"]]
        before["gaps"].append({"start_frame": end["frame_index"] + 1, "end_frame": start["frame_index"] - 1,
                               "missing_frame_count": start["frame_index"] - end["frame_index"] - 1,
                               "duration_s": start["time"] - end["time"], "resolved": True,
                               "possible_occlusion": False, "observations_interpolated": False})
        before["observations"].extend(after["observations"])
        before["gaps"].extend(after["gaps"])
        before["reconnections"].extend([dict(edge), *after["reconnections"]])
        before["source_track_ids"].extend(after["source_track_ids"])
        before["reconnection_hypotheses"].extend(after["reconnection_hypotheses"])
        parent[right] = left
    return [track for track in tracks if leader(track["track_id"]) == track["track_id"]]


def track_detections(frame_records, config=None):
    """Track all viable candidates without treating prediction as observation.

    Hungarian matches are geometrically gated. Two plausible choices after a
    missing interval are left unresolved. Forward/backward graph links preserve
    uncertainty; none establishes physical shuttle identity or first contact.
    """
    config = dict(config or {})
    max_gap = float(config.get("max_gap_seconds", 0.22))
    max_misses = int(config.get("max_missed_frames", 10))
    base_gate = float(config.get("base_gate_px", 35.0))
    max_gate = float(config.get("max_gate_px", 150.0))
    acceleration = float(config.get("acceleration_noise", 900.0))
    ambiguity_margin = float(config.get("ambiguity_margin", 0.16))
    min_observations = int(config.get("min_observations", 3))
    if not all(math.isfinite(value) for value in (max_gap, base_gate, max_gate, acceleration, ambiguity_margin)) or max_gap <= 0 or max_misses < 0 or base_gate <= 0 or max_gate < base_gate or acceleration <= 0 or min_observations < 1:
        raise ValueError("Invalid tracker time, gate, or acceleration configuration")
    active, finished = [], []
    next_id = 1
    previous_time, previous_frame = None, None
    for record in frame_records:
        timestamp, frame_index = float(record["time"]), int(record["frame_index"])
        if not math.isfinite(timestamp) or previous_time is not None and timestamp <= previous_time:
            raise ValueError("Frame timestamps must be finite and strictly increasing")
        if previous_frame is not None and frame_index <= previous_frame:
            raise ValueError("Frame indices must be strictly increasing")
        previous_time, previous_frame = timestamp, frame_index
        candidates = []
        for raw in record.get("detections", []):
            if not all(math.isfinite(float(raw[k])) for k in ("x", "y", "confidence")):
                continue
            if not 0 <= float(raw["confidence"]) <= 1:
                continue
            if not math.isfinite(float(raw.get("radius", 2))) or float(raw.get("radius", 2)) <= 0:
                continue
            candidates.append({**raw, "x": float(raw["x"]), "y": float(raw["y"]),
                               "confidence": float(raw["confidence"]), "radius": float(raw.get("radius", 2.0)),
                               "frame_index": frame_index, "time": timestamp, "observed": True})
        surviving = []
        for track in active:
            if timestamp - track.observations[-1]["time"] > max_gap or len(track.misses) > max_misses:
                finished.append(track.output())
            else:
                track.predict(timestamp, acceleration)
                surviving.append(track)
        active = surviving
        costs = np.full((len(active), len(candidates)), 1e6, dtype=float)
        for i, track in enumerate(active):
            gap = timestamp - track.observations[-1]["time"]
            speed = float(np.linalg.norm(track.state[2:]))
            gate = min(max_gate, base_gate + speed * gap * 0.45 + 2 * math.sqrt(max(0, float(np.trace(track.covariance[:2, :2])))))
            if track.misses:
                gate = min(gate, float(config.get("reconnect_gate_px", 65.0)))
            last_radius = max(1.0, float(track.observations[-1].get("radius", 2)))
            for j, candidate in enumerate(candidates):
                distance = math.hypot(candidate["x"] - track.state[0], candidate["y"] - track.state[1])
                if distance > gate:
                    continue
                size_penalty = min(0.35, abs(math.log(max(0.1, candidate["radius"]) / last_radius)) * 0.08)
                costs[i, j] = distance / gate + size_penalty + (1 - candidate["confidence"]) * 0.15
            plausible = np.sort(costs[i][costs[i] < 1.1])
            if len(plausible) > 1 and plausible[1] - plausible[0] < ambiguity_margin:
                track.ambiguous = True
                track.ambiguity_events.append({"frame_index": frame_index, "time": timestamp,
                                               "reason": "multiple_plausible_candidates", "after_gap": bool(track.misses)})
                if track.misses:
                    costs[i, :] = 1e6
        assigned_tracks, assigned_candidates = set(), set()
        if costs.size:
            track_indices, candidate_indices = linear_sum_assignment(costs)
            for i, j in zip(track_indices, candidate_indices):
                if costs[i, j] > 1.1:
                    continue
                competing = sorted(costs[:, j][costs[:, j] < 1.1])
                if len(competing) > 1 and competing[1] - competing[0] < ambiguity_margin:
                    active[i].ambiguous = True
                    active[i].ambiguity_events.append({"frame_index": frame_index, "time": timestamp,
                                                       "reason": "competing_tracks", "after_gap": bool(active[i].misses)})
                    if active[i].misses:
                        continue
                active[i].observe(candidates[j])
                assigned_tracks.add(i)
                assigned_candidates.add(j)
        for i, track in enumerate(active):
            if i not in assigned_tracks:
                track.misses.append({"frame_index": frame_index, "time": timestamp,
                                     "possible_occlusion": any(_inside(track.state[:2], region) for region in record.get("occlusion_regions", []))})
        for j, candidate in enumerate(candidates):
            if j not in assigned_candidates:
                active.append(_Track(next_id, candidate))
                next_id += 1
    finished.extend(track.output() for track in active)
    finished = [track for track in finished if len(track["observations"]) >= min_observations]
    finished = _offline_links(finished, config)
    for track in finished:
        observations = track["observations"]
        count = len(observations)
        span = observations[-1]["frame_index"] - observations[0]["frame_index"] + 1
        coverage = count / span
        average_confidence = float(np.mean([observation["confidence"] for observation in observations]))
        identity_score = min(0.95, average_confidence * coverage * min(1.0, count / 6))
        if track["identity_ambiguous"]:
            identity_score *= 0.45
        xy = np.array([[observation["x"], observation["y"]] for observation in observations])
        track.update({"identity_confidence": round(identity_score, 4), "confidence_kind": "heuristic_rank_not_probability",
                      "observation_count": count, "observed_frame_fraction": round(coverage, 4),
                      "motion_span_px": round(float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0))), 3),
                      "target_court_confirmed": False})
    return sorted(finished, key=lambda track: track["track_id"])
