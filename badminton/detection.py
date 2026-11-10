"""Multi-candidate shuttle proposals with explicit detector provenance.

The default backend is a deterministic computer-vision baseline, not a trained
shuttle detector. Its scores are ranking heuristics, not calibrated probability.
An explicitly supplied YOLO ONNX model or learned patch scorer can replace or
rescore proposals. No weights are silently downloaded.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np


def _validate_inputs(frames, timestamps):
    if len(frames) != len(timestamps):
        raise ValueError("frames and timestamps must have equal length")
    previous = None
    shape = None
    for frame, stamp in zip(frames, timestamps):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim not in (2, 3):
            raise ValueError("frames must be uint8 grayscale or BGR arrays")
        if frame.ndim == 3 and frame.shape[2] != 3:
            raise ValueError("colour frames must have three BGR channels")
        if not frame.size or min(frame.shape[:2]) < 8:
            raise ValueError("frames must be nonempty and at least 8x8 pixels")
        if shape is not None and frame.shape != shape:
            raise ValueError("all frames must have the same dimensions")
        if not math.isfinite(float(stamp)) or previous is not None and float(stamp) <= previous:
            raise ValueError("timestamps must be finite and strictly increasing")
        shape, previous = frame.shape, float(stamp)


def _bgr(frame):
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame


def _roi_mask(shape, config):
    height, width = shape[:2]
    mask = np.full((height, width), 255, np.uint8)
    polygon = config.get("roi_polygon")
    if polygon is not None:
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("roi_polygon must contain at least three finite [x,y] points")
        mask.fill(0)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


def _nms(detections, distance=3.0, limit=64):
    accepted = []
    for item in sorted(detections, key=lambda d: (-d["confidence"], d["x"], d["y"])):
        if not any(math.hypot(item["x"] - old["x"], item["y"] - old["y"]) <= max(distance, min(item["radius"], old["radius"])) for old in accepted):
            accepted.append(item)
        if len(accepted) >= limit:
            break
    return accepted


class _OnnxDetector:
    """Explicit YOLO v5/v8 ONNX inference, including optional source-size tiling."""

    def __init__(self, config):
        path = Path(config.get("model_path", ""))
        if not path.is_file():
            raise ValueError("ONNX backend requires an existing model_path")
        self.format = config.get("onnx_format", "yolo_v8")
        if self.format not in ("yolo_v5", "yolo_v8"):
            raise ValueError("onnx_format must be yolo_v5 or yolo_v8")
        self.size = int(config.get("model_input_size", 640))
        self.class_id = int(config.get("shuttle_class_id", 0))
        self.threshold = float(config.get("confidence_threshold", 0.25))
        self.tile_size = int(config.get("tile_size", 960))
        self.tile_overlap = float(config.get("tile_overlap", 0.2))
        if self.size < 32 or self.tile_size < 32 or not 0 <= self.tile_overlap < 0.75 or self.class_id < 0:
            raise ValueError("Invalid ONNX size, tiling, or class configuration")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.provenance = {
            "backend": "onnx", "model_file": path.name,
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "format": self.format, "shuttle_class_id": self.class_id,
            "weights_validation": "user_supplied_not_independently_validated",
        }

    def _tile(self, tile, offset):
        height, width = tile.shape[:2]
        factor = min(self.size / width, self.size / height)
        scaled_width, scaled_height = max(1, round(width * factor)), max(1, round(height * factor))
        resized = cv2.resize(tile, (scaled_width, scaled_height))
        pad_x, pad_y = (self.size - scaled_width) // 2, (self.size - scaled_height) // 2
        padded = np.full((self.size, self.size, 3), 114, np.uint8)
        padded[pad_y:pad_y + scaled_height, pad_x:pad_x + scaled_width] = resized
        self.net.setInput(cv2.dnn.blobFromImage(padded, 1 / 255, swapRB=True, crop=False))
        output = np.asarray(self.net.forward())
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError("Expected unpostprocessed YOLO ONNX output of shape [1,C,N] or [1,N,C]")
        predictions = output[0].T if self.format == "yolo_v8" else output[0]
        class_start = 4 if self.format == "yolo_v8" else 5
        if predictions.shape[1] <= class_start + self.class_id:
            raise ValueError("shuttle_class_id is outside the ONNX output class range")
        detections = []
        for prediction in predictions:
            confidence = float(prediction[class_start + self.class_id])
            if self.format == "yolo_v5":
                confidence *= float(prediction[4])
            if not math.isfinite(confidence) or confidence < self.threshold:
                continue
            cx, cy, bw, bh = map(float, prediction[:4])
            if not all(math.isfinite(v) for v in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                continue
            x1 = min(width, max(0.0, (cx - bw / 2 - pad_x) / factor))
            y1 = min(height, max(0.0, (cy - bh / 2 - pad_y) / factor))
            x2 = min(width, max(0.0, (cx + bw / 2 - pad_x) / factor))
            y2 = min(height, max(0.0, (cy + bh / 2 - pad_y) / factor))
            if x2 <= x1 or y2 <= y1:
                continue
            x1, x2 = x1 + offset[0], x2 + offset[0]
            y1, y2 = y1 + offset[1], y2 + offset[1]
            detections.append({"x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                               "radius": max(x2 - x1, y2 - y1) / 2,
                               "bbox": [x1, y1, x2 - x1, y2 - y1],
                               "confidence": min(1.0, confidence), "method": self.format + "_onnx",
                               "confidence_kind": "model_score_not_calibrated"})
        return detections

    def detect(self, frame, output_shape):
        height, width = frame.shape[:2]
        stride = max(1, round(self.tile_size * (1 - self.tile_overlap)))
        xs = sorted(set(list(range(0, max(1, width - self.tile_size + 1), stride)) + [max(0, width - self.tile_size)]))
        ys = sorted(set(list(range(0, max(1, height - self.tile_size + 1), stride)) + [max(0, height - self.tile_size)]))
        detections = []
        for y in ys:
            for x in xs:
                detections.extend(self._tile(frame[y:y + self.tile_size, x:x + self.tile_size], (x, y)))
        scale_x, scale_y = output_shape[1] / width, output_shape[0] / height
        for item in detections:
            item["x"] *= scale_x
            item["y"] *= scale_y
            item["radius"] *= max(scale_x, scale_y)
            item["bbox"] = [item["bbox"][0] * scale_x, item["bbox"][1] * scale_y,
                            item["bbox"][2] * scale_x, item["bbox"][3] * scale_y]
        return _nms(detections)


def detect_sequence(frames, timestamps, config=None):
    """Return frame-wise observations in the supplied frames' coordinate system.

    ``backend='opencv_candidates'`` is the default. ``backend='onnx'`` requires
    explicit weights and class mapping. Optional ``source_frames`` can provide
    higher-resolution frames for tiled ONNX inference. A ``scorer_path`` loads
    the separately trained patch classifier; learned scores change ranking only
    when ``use_learned_confidence=True`` is explicitly selected.
    """
    config = dict(config or {})
    _validate_inputs(frames, timestamps)
    if not frames:
        return []
    backend = config.get("backend", "opencv_candidates")
    if backend not in ("opencv_candidates", "onnx"):
        raise ValueError("backend must be opencv_candidates or onnx")
    height, width = frames[0].shape[:2]
    scale = width / 1280.0
    roi = _roi_mask(frames[0].shape, config)
    max_candidates = int(config.get("max_candidates", 64))
    if not 1 <= max_candidates <= 512:
        raise ValueError("max_candidates must be between 1 and 512")
    sample_ids = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), 9), dtype=int))
    background = np.median(np.stack([cv2.cvtColor(_bgr(frames[i]), cv2.COLOR_BGR2GRAY) for i in sample_ids]), axis=0).astype(np.uint8)
    onnx = _OnnxDetector(config) if backend == "onnx" else None
    source_frames = config.get("source_frames", frames)
    if len(source_frames) != len(frames):
        raise ValueError("source_frames must match frames in length")
    scorer = None
    if config.get("scorer_path"):
        from .learning import load_candidate_scorer
        scorer = load_candidate_scorer(config["scorer_path"])
    provenance = onnx.provenance if onnx else {
        "backend": "opencv_candidates", "trained_model": False,
        "method": "bright_small_components_with_background_and_temporal_difference",
        "confidence_kind": "heuristic_rank_not_probability",
    }
    min_area = float(config.get("min_area", max(1.0, 2 * scale * scale)))
    max_area = float(config.get("max_area", max(40.0, 180 * scale * scale)))
    if not math.isfinite(min_area) or not math.isfinite(max_area) or min_area <= 0 or max_area < min_area:
        raise ValueError("min_area and max_area must be positive finite ordered values")
    previous_gray = None
    recent_motion = np.zeros((height, width), np.float32)
    background_low = cv2.erode(background, np.ones((3, 3), np.uint8))
    background_high = cv2.dilate(background, np.ones((3, 3), np.uint8))
    phase_window = cv2.createHanningWindow((max(8, width // 2), max(8, height // 2)), cv2.CV_32F)
    texture_points = cv2.goodFeaturesToTrack(background, maxCorners=120, qualityLevel=0.02, minDistance=max(4, round(width / 80)))
    global_motion_supported = bool(texture_points is not None and len(texture_points) >= 24
                                   and cv2.contourArea(cv2.convexHull(texture_points)) >= width * height * 0.2)
    records = []
    for index, (raw_frame, timestamp) in enumerate(zip(frames, timestamps)):
        frame = _bgr(raw_frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # A one-pixel range tolerates codec noise and minor geometric jitter at
        # static edges. Translation compensation prevents static court/wall
        # features becoming candidates merely because the phone moved slightly.
        foreground = np.maximum(cv2.subtract(gray, background_high), cv2.subtract(background_low, gray))
        camera_motion = {"translation_px": [0.0, 0.0], "phase_response": None, "compensated": False}
        if previous_gray is not None:
            phase_size = (phase_window.shape[1], phase_window.shape[0])
            old_small = cv2.resize(previous_gray, phase_size).astype(np.float32)
            new_small = cv2.resize(gray, phase_size).astype(np.float32)
            (dx, dy), response = cv2.phaseCorrelate(old_small, new_small, phase_window)
            dx, dy = dx * width / phase_size[0], dy * height / phase_size[1]
            if global_motion_supported and math.isfinite(dx) and math.isfinite(dy) and math.isfinite(response) and response >= 0.25 and max(abs(dx), abs(dy)) <= 20:
                transform = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned_previous = cv2.warpAffine(previous_gray, transform, (width, height), borderMode=cv2.BORDER_REFLECT)
                recent_motion = cv2.warpAffine(recent_motion, transform, (width, height), borderMode=cv2.BORDER_CONSTANT)
                camera_motion = {"translation_px": [round(dx, 4), round(dy, 4)], "phase_response": round(response, 4), "compensated": True}
            else:
                aligned_previous = previous_gray
            temporal = cv2.absdiff(gray, aligned_previous)
        else:
            temporal = foreground
        persistence = math.exp(-(float(timestamp) - float(timestamps[index - 1])) / 0.12) if index else 0.0
        recent_motion = np.maximum(temporal.astype(np.float32), recent_motion * persistence)
        motion = recent_motion
        movement = np.where(foreground >= 24, 255, 0).astype(np.uint8)
        movement = cv2.morphologyEx(movement, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(movement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        occlusion_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) >= max(200, width * height * 0.0025) and h >= max(12, height * 0.035):
                occlusion_regions.append({"bbox": [int(x), int(y), int(w), int(h)],
                                          "kind": "moving_foreground_possible_occluder",
                                          "person_confirmed": False})
        occlusion_regions.sort(key=lambda item: item["bbox"][2] * item["bbox"][3], reverse=True)
        occlusion_regions = occlusion_regions[:24]
        if onnx:
            detections = onnx.detect(_bgr(source_frames[index]), frame.shape)
            detections = [d for d in detections if roi[min(height - 1, max(0, round(d["y"]))), min(width - 1, max(0, round(d["x"])))] > 0]
            proposal_count = len(detections)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bright = cv2.inRange(hsv, np.array([0, 0, int(config.get("min_brightness", 155))], np.uint8),
                                 np.array([179, int(config.get("max_saturation", 115)), 255], np.uint8))
            # Segment broad white objects before applying temporal/local-edge
            # masks. Otherwise a moving white shirt is fragmented into dozens
            # of misleading shuttle-sized edge components.
            broad_white = np.where(frame.min(axis=2) >= int(config.get("broad_white_min", 205)), 255, 0).astype(np.uint8)
            white_count, white_labels, white_stats, _ = cv2.connectedComponentsWithStats(broad_white, 8)
            large_labels = np.where(white_stats[:, cv2.CC_STAT_AREA] > max_area * 1.5)[0]
            large_labels = large_labels[large_labels != 0]
            excluded_white = np.isin(white_labels, large_labels).astype(np.uint8) * 255 if len(large_labels) else np.zeros_like(gray)
            excluded_white = cv2.dilate(excluded_white, np.ones((3, 3), np.uint8))
            local_base = cv2.GaussianBlur(gray, (0, 0), max(1.0, 4 * scale))
            contrast = cv2.subtract(gray, local_base)
            cue = ((motion >= float(config.get("motion_threshold", 16)))
                   & (foreground >= float(config.get("background_threshold", 8)))
                   & (contrast >= float(config.get("contrast_threshold", 8))))
            mask = cv2.bitwise_and(bright, np.where(cue, 255, 0).astype(np.uint8))
            mask = cv2.bitwise_and(mask, roi)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(excluded_white))
            count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
            detections = []
            for label in range(1, count):
                x, y, w, h, area = map(int, stats[label])
                if not min_area <= area <= max_area or max(w, h) / max(1, min(w, h)) > 4.0:
                    continue
                if max(w, h) > max(18, 34 * scale) or area / (w * h) < 0.12:
                    continue
                pixels = labels[y:y + h, x:x + w] == label
                local_motion = float(motion[y:y + h, x:x + w][pixels].mean())
                background_difference = float(foreground[y:y + h, x:x + w][pixels].mean())
                local_contrast = float(contrast[y:y + h, x:x + w][pixels].mean())
                compactness = area / (w * h)
                score = 0.2 + 0.35 * min(1.0, local_motion / 90) + 0.3 * min(1.0, local_contrast / 65) + 0.15 * compactness
                cx, cy = map(float, centers[label])
                overlaps = any(b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]
                               for b in (region["bbox"] for region in occlusion_regions))
                if overlaps:
                    score *= 0.85
                detections.append({"x": round(cx, 3), "y": round(cy, 3),
                                   "radius": round(max(1.0, math.sqrt(area / math.pi)), 3),
                                   "confidence": round(min(0.95, max(0.0, score)), 4),
                                   "bbox": [x, y, w, h], "method": "opencv_motion_bright_component",
                                   "confidence_kind": "heuristic_rank_not_probability",
                                   "foreground_difference": round(local_motion, 3),
                                   "background_difference": round(background_difference, 3),
                                   "local_contrast": round(local_contrast, 3),
                                   "possible_occluder_overlap": overlaps})
            proposal_count = len(detections)
            detections = _nms(detections, max(1.0, 3 * scale), max_candidates)
        if scorer:
            rescored = scorer.score(frame, detections, preprocessing=config.get("frame_preprocessing"))
            for old, new in zip(detections, rescored):
                new["heuristic_confidence"] = old["confidence"]
                if config.get("use_learned_confidence") and new.get("learned_confidence") is not None:
                    new["confidence"] = float(new["learned_confidence"])
                    new["confidence_kind"] = "learned_score_not_calibrated"
                else:
                    new["confidence"] = old["confidence"]
            detections = _nms(rescored, max(1.0, 3 * scale), max_candidates)
        record = {"frame_index": index, "time": float(timestamp), "detections": detections,
                  "occlusion_regions": occlusion_regions, "detector": dict(provenance),
                  "camera_motion": camera_motion, "proposal_count_before_limit": proposal_count,
                  "candidate_limit_reached": proposal_count > max_candidates}
        if scorer:
            record["candidate_scorer"] = getattr(scorer, "provenance", {"path": str(config["scorer_path"])})
        records.append(record)
        previous_gray = gray
    return records


def summarize_detection_quality(frame_records, tracks):
    """Operational diagnostics only; no precision/recall claim without labels."""
    counts = [len(record.get("detections", [])) for record in frame_records]
    capped = sum(bool(record.get("candidate_limit_reached")) for record in frame_records)
    ambiguous = sum(bool(track.get("identity_ambiguous")) for track in tracks)
    warnings = []
    if capped:
        warnings.append("CANDIDATE_BUDGET_EXHAUSTED")
    if tracks and ambiguous / len(tracks) > 0.5:
        warnings.append("MANY_AMBIGUOUS_TRACKLETS")
    backends = sorted({record.get("detector", {}).get("backend", "unknown") for record in frame_records})
    if "opencv_candidates" in backends:
        warnings.append("HEURISTIC_DETECTOR_NOT_VALIDATED")
    return {"frames": len(frame_records), "detection_count": sum(counts),
            "detections_per_frame_mean": round(float(np.mean(counts)), 3) if counts else 0.0,
            "detections_per_frame_p95": round(float(np.percentile(counts, 95)), 3) if counts else 0.0,
            "candidate_budget_hit_frames": capped,
            "candidate_budget_hit_fraction": round(capped / len(frame_records), 4) if frame_records else 0.0,
            "track_count": len(tracks), "ambiguous_track_count": ambiguous,
            "ambiguous_track_fraction": round(ambiguous / len(tracks), 4) if tracks else 0.0,
            "backends": backends, "warnings": warnings,
            "precision_measured": False, "recall_measured": False,
            "note": "Candidate volume and track ambiguity describe this run; they do not measure detection accuracy."}
