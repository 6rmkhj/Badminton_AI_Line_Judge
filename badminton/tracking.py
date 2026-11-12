"""Conservative multi-hypothesis shuttle tracklets in image coordinates."""
from __future__ import annotations

import math
from collections import defaultdict

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
        self.motion_model_changes = []

    def predict(self, time, acceleration):
        dt = time - self.time
        transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        noise_gain = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + noise_gain @ noise_gain.T * acceleration ** 2
        self.time = time

    def observe(self, observation, adapt_maneuver=True):
        measurement = np.array([observation["x"], observation["y"]])
        last = self.observations[-1]
        dt = observation["time"] - last["time"]
        displacement = measurement - np.array([last["x"], last["y"]])
        incoming = _velocity(self.observations)
        minimum_motion = max(3., 2 * max(float(last.get("radius", 2)), float(observation.get("radius", 2))))
        # A constant-velocity posterior otherwise keeps moving through a clear
        # observed reversal and can attach the *next* frame to a distractor on
        # its old path. Adapt only with consecutive, sufficiently resolved
        # observations. Missing-interval reconnections retain their old rules.
        reverse = (adapt_maneuver and not self.misses and len(self.observations) >= 3
                   and observation["frame_index"] == last["frame_index"] + 1 and dt > 0
                   and float(np.linalg.norm(displacement)) >= minimum_motion
                   and float(np.linalg.norm(incoming)) * dt >= 2 * minimum_motion
                   and float(displacement @ self.state[2:]) < 0
                   and float(displacement @ incoming) < -0.5 * float(np.linalg.norm(displacement) * np.linalg.norm(incoming)))
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
        if reverse:
            self.state[:2] = measurement
            self.state[2:] = displacement / dt
            velocity_variance = 4 * variance / dt ** 2
            self.covariance = np.diag([variance, variance, velocity_variance, velocity_variance])
            self.motion_model_changes.append({"frame_index": observation["frame_index"], "time": observation["time"],
                "reason": "observed_direction_reversal", "incoming_velocity_px_per_second": incoming.tolist(),
                "updated_velocity_px_per_second": self.state[2:].tolist(),
                "verified_identity": False, "contact_confirmed": False})
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
                "motion_model_changes": self.motion_model_changes,
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
        occluded_gap = any(gap.get("possible_occlusion", False)
                           and gap.get("end_frame", -1) >= end["frame_index"] + 1
                           and gap.get("start_frame", math.inf) <= start["frame_index"] - 1
                           for gap in before["gaps"])
        before["gaps"] = [gap for gap in before["gaps"] if gap["resolved"]]
        before["gaps"].append({"start_frame": end["frame_index"] + 1, "end_frame": start["frame_index"] - 1,
                               "missing_frame_count": start["frame_index"] - end["frame_index"] - 1,
                               "duration_s": start["time"] - end["time"], "resolved": True,
                               "possible_occlusion": occluded_gap, "observations_interpolated": False})
        before["observations"].extend(after["observations"])
        before["gaps"].extend(after["gaps"])
        before["reconnections"].extend([dict(edge), *after["reconnections"]])
        before.setdefault("motion_model_changes", []).extend(after.get("motion_model_changes", []))
        before["source_track_ids"].extend(after["source_track_ids"])
        before["reconnection_hypotheses"].extend(after["reconnection_hypotheses"])
        parent[right] = left
    return [track for track in tracks if leader(track["track_id"]) == track["track_id"]]


def _flight_fit(observations, timestamp, window=6):
    recent = observations[-window:]
    xy = np.asarray([[o["x"], o["y"]] for o in recent], dtype=float)
    times = np.asarray([o["time"] for o in recent], dtype=float)
    times -= times[-1]
    degree = 2 if len(recent) >= 4 else 1
    design = np.column_stack([times ** i for i in range(degree + 1)])
    coefficients = np.linalg.lstsq(design, xy, rcond=None)[0]
    dt = timestamp - recent[-1]["time"]
    predicted = np.asarray([dt ** i for i in range(degree + 1)]) @ coefficients
    residual = float(np.sqrt(np.mean(np.sum((design @ coefficients - xy) ** 2, axis=1))))
    return predicted, coefficients[1], residual


def _path_travel(observations):
    return sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(observations, observations[1:]))


def _flight_rank(beam):
    observations, error, travel = beam
    span = observations[-1]["frame_index"] - observations[0]["frame_index"] + 1
    coverage = len(observations) / max(1, span)
    return (len(observations) + min(travel, 300.) * .06 * coverage
            - error / max(1, len(observations) - 2) * 4 - (span - len(observations)) * .6)


def _rebound_extensions(approach, rows):
    """Retain bounded observed alternatives near a free-flight endpoint.

    This proposes a piecewise motion path, not an identity reconnection or a
    physical bounce. The event stage independently checks the two segments.
    """
    end = approach[-1]
    _, velocity, _ = _flight_fit(approach, end["time"])
    if velocity[1] < 12:
        return []
    radius = max(1., float(np.median([o.get("radius", 2.) for o in approach[-5:]])))
    nearby = [row for row in rows if 0 < row[0] - end["time"] <= .5][:64]
    active, finished = [], []
    for timestamp, points in nearby:
        fresh = []
        for observations, error in active:
            gap = timestamp - observations[-1]["time"]
            if gap > .13:
                continue
            if len(observations) < 2:
                prediction = np.array([observations[-1]["x"], observations[-1]["y"]])
            else:
                prediction, _, _ = _flight_fit(observations, timestamp)
            distances = [math.hypot(p["x"] - prediction[0], p["y"] - prediction[1]) for p in points]
            for j in np.argsort(distances)[:2]:
                if distances[j] <= max(7., radius * 3):
                    fresh.append((observations + [points[j]], error + distances[j] ** 2))
            fresh.append((observations, error))
        if timestamp - end["time"] <= .35:
            dt = timestamp - end["time"]
            predicted_x = end["x"] + velocity[0] * dt
            seeds = [p for p in points if abs(p["x"] - predicted_x) <= max(6., radius * 3)
                     and abs(p["y"] - end["y"]) <= max(12., min(60., abs(velocity[1]) * .15))]
            seeds.sort(key=lambda p: abs(p["x"] - predicted_x) + abs(p["y"] - end["y"]) * .2)
            fresh.extend(([p], 0.) for p in seeds[:3])
        # Both working history and branch count are bounded independently of
        # video length. Original observations are never interpolated.
        unique = {}
        for observations, error in fresh:
            observations = observations[:6]
            key = tuple((p["frame_index"], p["x"], p["y"]) for p in observations)
            unique[key] = (observations, error)
        active = sorted(unique.values(), key=lambda b: (len(b[0]) - b[1] * .03), reverse=True)[:24]
        for observations, error in active:
            if len(observations) >= 3 and observations[-1]["y"] < observations[0]["y"] - .5:
                _, _, residual = _flight_fit(observations, observations[-1]["time"], window=6)
                if residual <= max(3., radius * 1.5):
                    finished.append((observations, error, residual))
    finished.sort(key=lambda b: (-len(b[0]), b[1], b[0][0]["time"]))
    selected = []
    for observations, _, residual in finished:
        if any(abs(observations[0]["time"] - old[0][0]["time"]) < .08
               and math.hypot(observations[0]["x"] - old[0][0]["x"], observations[0]["y"] - old[0][0]["y"]) < 3
               for old in selected):
            continue
        selected.append((observations, residual))
        if len(selected) == 2:
            break
    return selected


def _flight_hypotheses(frame_records, config, first_id, progress_callback=None):
    """Bounded offline alternatives to mixed constant-velocity track histories.

    These paths rank coherent image motion for review. They do not establish
    which object is the shuttle, first contact, or continuity through a gap.
    """
    max_beams = min(2048, max(32, int(config.get("flight_max_beams", 1600))))
    max_hypotheses = min(160, max(1, int(config.get("flight_max_hypotheses", 96))))
    steps = np.diff([float(record["time"]) for record in frame_records])
    nominal_step = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 1 / 30
    history_limit = min(64, max(18, int(math.ceil(.35 / nominal_step)) + 1))
    rows = []
    for record in frame_records:
        points = []
        for raw in record.get("detections", []):
            if not all(math.isfinite(float(raw.get(key, math.nan))) for key in ("x", "y", "confidence")):
                continue
            radius = float(raw.get("radius", 2.))
            confidence = float(raw["confidence"])
            if not math.isfinite(radius) or radius <= 0 or not 0 <= confidence <= 1:
                continue
            points.append({**raw, "x": float(raw["x"]), "y": float(raw["y"]), "confidence": confidence,
                           "radius": radius, "frame_index": int(record["frame_index"]),
                           "time": float(record["time"]), "observed": True})
        rows.append((float(record["time"]), points[:256]))
    beams, archive = [], {}
    from .media import _FrameProgress
    reporter = _FrameProgress(progress_callback, len(rows), '일관된 비행 궤적 탐색')
    reporter.report(0, force=True)
    for order, (timestamp, points) in enumerate(rows):
        coordinates = np.asarray([[p["x"], p["y"]] for p in points], dtype=float).reshape(-1, 2)
        fresh = []
        for observations, error, travel in beams:
            gap = int(frame_records[order]["frame_index"]) - observations[-1]["frame_index"]
            if not 1 <= gap <= 3:
                continue
            prediction, _, _ = _flight_fit(observations, timestamp)
            tolerance = 4 + min(4., float(np.median([o["radius"] for o in observations[-4:]]))) * 1.3 + (gap - 1) * 2
            distances = np.linalg.norm(coordinates - prediction, axis=1)
            for j in np.argsort(distances)[:2]:
                if distances[j] <= tolerance:
                    extended = (observations + [points[j]])[-history_limit:]
                    fresh.append((extended, error * .94 + float((distances[j] / tolerance) ** 2), _path_travel(extended)))
            if gap < 3:
                fresh.append((observations, error, travel))
        if order and points and rows[order - 1][1]:
            previous = rows[order - 1][1]
            previous_xy = np.asarray([[p["x"], p["y"]] for p in previous])
            for j, point in enumerate(points):
                distances = np.linalg.norm(previous_xy - coordinates[j], axis=1)
                eligible = np.where((distances >= 2) & (distances <= 80))[0]
                for k in eligible[np.argsort(distances[eligible])[:6]]:
                    fresh.append(([previous[k], point], 0., float(distances[k])))
        buckets = {}
        for beam in fresh:
            a, b = beam[0][-2:]
            vx, vy = b["x"] - a["x"], b["y"] - a["y"]
            key = (b["frame_index"], b["x"], b["y"], int((math.atan2(vy, vx) + math.pi) / (math.pi / 4)),
                   int(math.log2(max(1., math.hypot(vx, vy)))))
            if key not in buckets or _flight_rank(beam) > _flight_rank(buckets[key]):
                buckets[key] = beam
        regions = defaultdict(list)
        for beam in sorted(buckets.values(), key=_flight_rank, reverse=True):
            point = beam[0][-1]
            regions[(int(point["x"] / 48), int(point["y"] / 48))].append(beam)
        # Fair spatial allocation keeps old clothing paths from monopolizing
        # the finite search budget before a new free-flight object appears.
        beams = []
        for depth in range(64):
            for region in regions.values():
                if len(region) > depth:
                    beams.append(region[depth])
                    if len(beams) == max_beams:
                        break
            if len(beams) == max_beams:
                break
        for observations, error, _ in sorted(beams, key=_flight_rank, reverse=True)[:400]:
            direction = observations[-1]["y"] - observations[-2]["y"]
            ys = [o["y"] for o in observations]
            extremum = int(np.argmin(ys)) if direction > 0 else int(np.argmax(ys))
            observations = observations[extremum:]
            travel = _path_travel(observations)
            if len(observations) < 6 or observations[-1]["time"] - observations[0]["time"] < .16 or travel < 18:
                continue
            net = math.hypot(observations[-1]["x"] - observations[0]["x"], observations[-1]["y"] - observations[0]["y"])
            if net / travel < .8:
                continue
            key = tuple((o["frame_index"], o["x"], o["y"]) for o in (observations[0], observations[-1]))
            candidate = (observations, error, travel)
            if key not in archive or _flight_rank(candidate) > _flight_rank(archive[key]):
                archive[key] = candidate
        if len(archive) > 2048:
            archive = dict(sorted(archive.items(), key=lambda item: _flight_rank(item[1]), reverse=True)[:2048])
        reporter.report(order + 1, force=order + 1 == len(rows))
    selected = []
    for beam in sorted(archive.values(), key=_flight_rank, reverse=True):
        observations = beam[0]
        identifiers = {(o["frame_index"], o["x"], o["y"]) for o in observations}
        if any(len(identifiers & old_ids) / min(len(identifiers), len(old_ids)) > .65
               and abs(observations[-1]["frame_index"] - old[0][-1]["frame_index"]) <= 1
               and math.hypot(observations[-1]["x"] - old[0][-1]["x"], observations[-1]["y"] - old[0][-1]["y"]) < 5
               for old, old_ids in selected):
            continue
        selected.append((beam, identifiers))
        if len(selected) >= max_hypotheses:
            break
    paths = []
    if progress_callback is not None:
        progress_callback({'detail': '비행·반동 궤적 연결 검토'})
    for beam, _ in selected:
        approach = beam[0]
        _, _, residual = _flight_fit(approach, approach[-1]["time"], window=history_limit)
        paths.append((approach, len(approach), residual))
        if residual <= max(3., float(np.median([o["radius"] for o in approach])) * 1.5):
            for rebound, _ in _rebound_extensions(approach, rows):
                paths.append((approach + rebound, len(approach), residual))
    output = []
    for offset, (observations, approach_count, residual) in enumerate(paths):
        identifier = first_id + offset
        gaps = [{"start_frame": a["frame_index"] + 1, "end_frame": b["frame_index"] - 1,
                 "missing_frame_count": b["frame_index"] - a["frame_index"] - 1,
                 "duration_s": b["time"] - a["time"], "resolved": False,
                 "possible_occlusion": False, "cause": "unresolved_candidate_association",
                 "observations_interpolated": False}
                for a, b in zip(observations, observations[1:]) if b["frame_index"] > a["frame_index"] + 1]
        output.append({"track_id": identifier, "observations": observations, "gaps": gaps,
            "identity_ambiguous": True, "ambiguity_events": [{"reason": "unverified_free_flight_hypothesis"}],
            "reconnections": [], "reconnection_hypotheses": [], "source_track_ids": [identifier],
            "method": "coherent_free_flight_hypothesis", "motion_model_changes": [],
            "flight_hypothesis": {"method": "coherent_free_flight_search", "fit_residual_px": residual,
                "inlier_count": approach_count, "observed_span_seconds": observations[approach_count - 1]["time"] - observations[0]["time"],
                "travel_px": _path_travel(observations[:approach_count]), "approach_count": approach_count,
                "rebound_count": len(observations) - approach_count, "verified_identity": False,
                "physical_contact_confirmed": False, "observations_interpolated": False}})
    return output


def track_detections(frame_records, config=None, *, progress_callback=None):
    """Track all viable candidates without treating prediction as observation.

    Hungarian matches are geometrically gated. Two plausible choices after a
    missing interval are left unresolved. Forward/backward graph links preserve
    uncertainty; none establishes physical shuttle identity or first contact.
    """
    config = dict(config or {})
    if config.get("flight_search_enabled", False) or progress_callback is not None:
        frame_records = list(frame_records)
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
    from .media import _FrameProgress
    reporter = _FrameProgress(progress_callback, len(frame_records) if progress_callback is not None else 0,
                              '프레임 사이의 후보 궤적 연결')
    reporter.report(0, force=True)
    next_id = 1
    previous_time, previous_frame = None, None
    for processed_count, record in enumerate(frame_records, 1):
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
                active[i].observe(candidates[j], adapt_maneuver=config.get("maneuver_adaptation", True))
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
        if progress_callback is not None:
            reporter.report(processed_count)
    finished.extend(track.output() for track in active)
    finished = [track for track in finished if len(track["observations"]) >= min_observations]
    finished = _offline_links(finished, config)
    if progress_callback is not None:
        reporter.report(len(frame_records), force=True)
    if config.get("flight_search_enabled", False):
        finished.extend(_flight_hypotheses(frame_records, config, next_id, progress_callback))
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
