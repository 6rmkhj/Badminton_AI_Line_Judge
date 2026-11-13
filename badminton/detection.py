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
from .media import _FrameProgress


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
    # Bucket accepted centres so a larger proposal pool does not require an
    # all-pairs scan. The suppression radius is at most this cell width, so the
    # nine neighbouring cells are exactly equivalent to the old linear test.
    cell = max(float(distance), max((float(item["radius"]) for item in detections), default=0), 1e-9)
    cells = {}
    accepted = []
    for item in sorted(detections, key=lambda d: (-d["confidence"], d["x"], d["y"])):
        key = (math.floor(item["x"] / cell), math.floor(item["y"] / cell))
        neighbours = (old for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      for old in cells.get((key[0] + dx, key[1] + dy), ()))
        if not any(math.hypot(item["x"] - old["x"], item["y"] - old["y"])
                   <= max(distance, min(item["radius"], old["radius"])) for old in neighbours):
            accepted.append(item)
            cells.setdefault(key, []).append(item)
        if len(accepted) >= limit:
            break
    return accepted


def _candidate_budget(detections, shape, limit, *, mode="ranked", distance=3.0):
    """Preserve spatially separated proposals under the same finite budget.

    Spatial mode takes the best remaining observation from each image cell per
    pass. It changes proposal ranking only, never identity/contact confidence.
    Use it explicitly for difficult clips; broad spatial coverage is not a
    substitute for an independently validated detector.
    """
    if mode not in {"ranked", "spatial"}:
        raise ValueError("candidate_selection must be ranked or spatial")
    if mode == "ranked":
        return _nms(detections, distance, limit)
    height, width = shape[:2]
    pool = _nms(detections, distance, 512)
    cells = {}
    for item in pool:
        key = (min(7, max(0, int(item["x"] * 8 / width))),
               min(4, max(0, int(item["y"] * 5 / height))))
        cells.setdefault(key, []).append(item)
    selected = []
    for level in range(max((len(group) for group in cells.values()), default=0)):
        tier = sorted((group[level] for group in cells.values() if len(group) > level),
                      key=lambda item: (-item["confidence"], item["x"], item["y"]))
        selected.extend(tier[:limit - len(selected)])
        if len(selected) >= limit:
            break
    return selected


def _refinement_regions(config, shape, frame_count):
    regions = config.get("refinement_regions", [])
    if not isinstance(regions, list) or len(regions) > 16:
        raise ValueError("refinement_regions must be a list of at most 16 regions")
    height, width = shape[:2]
    parsed = []
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("each refinement region must be an object")
        first, last, box = region.get("frame_start"), region.get("frame_end"), region.get("bbox")
        if (any(not isinstance(value, int) or isinstance(value, bool) for value in (first, last))
                or not 0 <= first <= last < frame_count):
            raise ValueError("refinement frame bounds must be inclusive indices within the video")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(value) for value in box)):
            raise ValueError("refinement bbox must contain four finite analysis-pixel coordinates")
        x, y, w, h = map(float, box)
        if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
            raise ValueError("refinement bbox must lie inside the analysis image")
        parsed.append({"frame_start": first, "frame_end": last, "bbox": [x, y, w, h]})
    return parsed


def _inside_refinement(item, regions):
    return any(x <= item["x"] <= x + w and y <= item["y"] <= y + h
               for x, y, w, h in (region["bbox"] for region in regions))


def _refined_budget(detections, shape, limit, mode, distance, regions):
    if not regions:
        return _candidate_budget(detections, shape, limit, mode=mode, distance=distance), 0
    # Reserve part, not all, of the finite candidate budget for the explicitly
    # selected region. Do not manufacture candidates or increase their scores.
    preferred = _candidate_budget([item for item in detections if _inside_refinement(item, regions)],
                                  shape, max(1, limit // 4), mode=mode, distance=distance)
    ordinary = _candidate_budget(detections, shape, min(512, limit + len(preferred)),
                                mode=mode, distance=distance)
    result = list(preferred)
    if len(result) >= limit:
        return result[:limit], min(limit, len(preferred))
    for item in ordinary:
        if any(math.hypot(item["x"] - old["x"], item["y"] - old["y"])
               <= max(distance, min(item["radius"], old["radius"])) for old in result):
            continue
        result.append(item)
        if len(result) >= limit:
            break
    return result, len(preferred)


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


class _HogPeople:
    """Sample the bundled OpenCV pedestrian model, without claiming court identity.

    A model box is only a possible occluder. Cached boxes carry their age so that
    an old detection is never presented as a fresh observation or verified player.
    """

    def __init__(self, config):
        self.backend = config.get("person_backend", "hog")
        if self.backend not in {"hog", "none"}:
            raise ValueError("person_backend must be hog or none")
        self.interval = int(config.get("person_interval", 16))
        self.width = int(config.get("person_width", 960))
        if not 1 <= self.interval <= 120 or not 128 <= self.width <= 1280:
            raise ValueError("invalid person detector interval or width")
        self.hog = None
        if self.backend == "hog":
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.boxes = []
        self.observed_index = None

    def detect(self, frame, index):
        if self.hog is None:
            return [], {"backend": "none", "sampled": False}
        height, width = frame.shape[:2]
        ratio = min(1.0, self.width / width)
        sample_width, sample_height = round(width * ratio), round(height * ratio)
        sampled = index % self.interval == 0
        if sampled:
            self.boxes = []
            self.observed_index = index
            if sample_width >= 64 and sample_height >= 128:
                sample = cv2.resize(frame, (sample_width, sample_height), interpolation=cv2.INTER_AREA)
                boxes, weights = self.hog.detectMultiScale(
                    sample, winStride=(8, 8), padding=(8, 8), scale=1.08,
                    hitThreshold=0.0, groupThreshold=2.0)
                for box, score in zip(boxes, np.asarray(weights).reshape(-1)):
                    x, y, w, h = (float(value) / ratio for value in box)
                    if not math.isfinite(float(score)):
                        continue
                    self.boxes.append({"bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                                       "model_margin": round(float(score), 4)})
                self.boxes = sorted(self.boxes, key=lambda item: -item["model_margin"])[:16]
        age = index - self.observed_index if self.observed_index is not None else None
        regions = [{**box, "kind": "hog_person_hypothesis", "person_confirmed": False,
                    "person_model_detected": True, "target_court_player_confirmed": False,
                    "observation_frame_index": self.observed_index, "age_frames": age,
                    "interpolated_observation": False} for box in self.boxes]
        return regions, {"backend": "opencv_hog_default_people", "sampled": sampled,
                         "observation_frame_index": self.observed_index, "age_frames": age,
                         "model_box_count": len(regions), "independently_validated": False}


class _NativeRoiDetector:
    """Bounded source-resolution re-detection, not resized evidence cropping.

    Windows follow global candidates and recent native proposals. Two rotating
    exploration windows also examine pixels missed by the low-resolution scan.
    This deliberately reports incomplete source coverage; it is not exhaustive
    tiled detection and does not certify shuttle identity or contact.
    """

    def __init__(self, config, output_shape, background, roi, max_area):
        self.config = config
        self.height, self.width = output_shape[:2]
        self.background, self.roi = background, roi
        self.max_area = max_area
        self.size = int(config.get("highres_roi_size", 192))
        self.budget = int(config.get("highres_max_rois", 8))
        self.explore = int(config.get("highres_exploration_rois", min(2, self.budget)))
        if not 32 <= self.size <= 1024 or not 1 <= self.budget <= 64 or not 0 <= self.explore <= self.budget:
            raise ValueError("invalid high-resolution ROI size, budget, or exploration count")
        self.previous = None
        self.recent = []
        self.source_shape = None

    def _boxes(self, detections, shape, index, regions=()):
        height, width = shape[:2]
        sx, sy = width / self.width, height / self.height
        boxes = []

        def append(cx, cy, origin):
            if not (0 <= cx < width and 0 <= cy < height):
                return
            low_x, low_y = min(self.width - 1, int(cx / sx)), min(self.height - 1, int(cy / sy))
            if self.roi[low_y, low_x] == 0:
                return
            left = max(0, min(max(0, width - self.size), round(cx - self.size / 2)))
            top = max(0, min(max(0, height - self.size), round(cy - self.size / 2)))
            box = [left, top, min(self.size, width - left), min(self.size, height - top)]
            # Nearby windows mostly read the same pixels. Keep separate objects
            # inside the first window, instead of spending the budget twice.
            if any(abs(left - old["bbox"][0]) < box[2] * .55
                   and abs(top - old["bbox"][1]) < box[3] * .55 for old in boxes):
                return
            boxes.append({"bbox": box, "origin": origin})

        focus_budget = min(self.budget, max(1, self.budget // 2)) if regions else 0
        # Start at each reviewed region's centre before rotating across larger
        # regions. These are search windows, not observations or contact labels.
        for region in regions:
            x, y, w, h = region["bbox"]
            append((x + w / 2) * sx, (y + h / 2) * sy, "review_refinement_region")
            if len(boxes) >= focus_budget:
                break
        for region in regions:
            if len(boxes) >= focus_budget:
                break
            x, y, w, h = region["bbox"]
            columns, rows = max(1, math.ceil(w * sx / self.size)), max(1, math.ceil(h * sy / self.size))
            grid = [(x + w * (col + .5) / columns, y + h * (row + .5) / rows)
                    for row in range(rows) for col in range(columns)]
            for offset in range(len(grid)):
                cx, cy = grid[(index - region["frame_start"] + offset) % len(grid)]
                append(cx * sx, cy * sy, "review_refinement_region")
                if len(boxes) >= focus_budget:
                    break
        proposal_budget = max(len(boxes), self.budget - self.explore)
        # Alternate global and previous-native proposals so neither monopolizes
        # the entire bounded native-resolution budget.
        ranked = (_candidate_budget(detections, (self.height, self.width), 512, mode="spatial")
                  if self.config.get("candidate_selection") == "spatial"
                  else sorted(detections, key=lambda item: -item["confidence"]))
        for position in range(max(len(ranked), len(self.recent))):
            for proposals, origin in ((ranked, "global_candidate"), (self.recent, "previous_native_candidate")):
                if len(boxes) >= proposal_budget:
                    break
                if position < len(proposals):
                    point = proposals[position]
                    append(point["x"] * sx, point["y"] * sy, origin)
            if len(boxes) >= proposal_budget:
                break
        if self.explore == 0:
            return boxes
        xs = list(range(0, width, self.size))
        ys = list(range(0, height, self.size))
        grid = [(min(width - 1, x + self.size / 2), min(height - 1, y + self.size / 2))
                for y in ys for x in xs]
        # Fill unused proposal slots as well, making source-only detections
        # possible even when the low-resolution scan has no observations.
        for offset in range(len(grid)):
            if len(boxes) >= self.budget:
                break
            x, y = grid[(index * max(1, self.explore) + offset) % len(grid)]
            append(x, y, "rotating_exploration")
        return boxes

    def detect(self, raw_source, detections, index, camera_motion, regions=()):
        if (not isinstance(raw_source, np.ndarray) or raw_source.dtype != np.uint8
                or raw_source.ndim not in (2, 3) or not raw_source.size
                or raw_source.ndim == 3 and raw_source.shape[2] != 3):
            raise ValueError("source_frames must contain uint8 BGR or grayscale frames")
        source = _bgr(raw_source)
        if self.source_shape is not None and source.shape != self.source_shape:
            raise ValueError("source_frames dimensions must remain constant")
        self.source_shape = source.shape
        sh, sw = source.shape[:2]
        sx, sy = sw / self.width, sh / self.height
        if sw <= self.width and sh <= self.height:
            return [], {"status": "not_higher_resolution", "source_size": [sw, sh], "rois": []}
        if abs(sx / sy - 1) > .03:
            raise ValueError("source_frames and analysis frames must have the same displayed aspect ratio")
        windows = self._boxes(detections, source.shape, index, regions)
        native = []
        area_limit = self.max_area * sx * sy
        for window in windows:
            x, y, w, h = window["bbox"]
            patch = source[y:y + h, x:x + w]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            left, top = int(x / sx), int(y / sy)
            right, bottom = min(self.width, math.ceil((x + w) / sx)), min(self.height, math.ceil((y + h) / sy))
            background = cv2.resize(self.background[top:bottom, left:right], (w, h), interpolation=cv2.INTER_LINEAR)
            foreground = cv2.absdiff(gray, background)
            if self.previous is None:
                temporal = foreground
            else:
                old = cv2.cvtColor(self.previous[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                if camera_motion.get("compensated"):
                    dx, dy = camera_motion["translation_px"]
                    old = cv2.warpAffine(old, np.float32([[1, 0, dx * sx], [0, 1, dy * sy]]),
                                         (w, h), borderMode=cv2.BORDER_REFLECT)
                temporal = cv2.absdiff(gray, old)
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            bright = cv2.inRange(hsv, np.array([0, 0, int(self.config.get("min_brightness", 155))], np.uint8),
                                 np.array([179, int(self.config.get("max_saturation", 115)), 255], np.uint8))
            local_base = cv2.GaussianBlur(gray, (0, 0), max(1.0, min(12.0, 4 * sw / 1280)))
            contrast = (cv2.absdiff(gray, local_base) if self.config.get('include_dark_components')
                        else cv2.subtract(gray, local_base))
            local_components = self.config.get("native_local_components", bool(self.config.get("refinement_regions")))
            if local_components:
                # Bright wooden flooring may meet the colour threshold too.
                # Without local structure segmentation, adding a dark cork to
                # that mask joins it to the entire floor and the area filter
                # rejects both. Preserve local objects before testing motion.
                bright = cv2.bitwise_and(bright, np.where(
                    cv2.subtract(gray, local_base) >= float(self.config.get("contrast_threshold", 8)),
                    255, 0).astype(np.uint8))
            if self.config.get('include_dark_components'):
                # The cork is often darker than the wooden floor at impact.
                # These remain motion proposals; a dark component is not contact evidence.
                dark = ((cv2.subtract(local_base, gray) >= 18) & (hsv[:, :, 1] <= 115))
                bright = np.maximum(bright, dark.astype(np.uint8) * 255)
            cue = ((temporal >= float(self.config.get("motion_threshold", 16)))
                   & (foreground >= float(self.config.get("background_threshold", 8)))
                   & (contrast >= float(self.config.get("contrast_threshold", 8))))
            # Reject complete broad bright objects before edge differencing;
            # otherwise a white shirt becomes many apparent small shuttles.
            broad = np.where(patch.min(axis=2) >= int(self.config.get("broad_white_min", 205)), 255, 0).astype(np.uint8)
            _, broad_labels, broad_stats, _ = cv2.connectedComponentsWithStats(broad, 8)
            large = np.flatnonzero(broad_stats[:, cv2.CC_STAT_AREA] > area_limit * 1.5)
            large = large[large != 0]
            exclusion = np.isin(broad_labels, large) if len(large) else np.zeros_like(gray, dtype=bool)
            # Segment the whole small bright object, then use motion/contrast
            # as evidence for that component. Segmenting only changed pixels
            # would move the centroid to a leading edge between frames.
            mask = np.where((bright > 0) & ~exclusion, 255, 0).astype(np.uint8)
            count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
            for label in range(1, count):
                bx, by, bw, bh, area = map(int, stats[label])
                if not 1 <= area <= area_limit or max(bw, bh) / max(1, min(bw, bh)) > 4:
                    continue
                if max(bw, bh) > max(18.0, 34 * sw / 1280) or area / (bw * bh) < .12:
                    continue
                if ((bx == 0 and x > 0) or (by == 0 and y > 0)
                        or (bx + bw >= w and x + w < sw) or (by + bh >= h and y + h < sh)):
                    continue  # the ROI cut through this object
                pixels = labels[by:by + bh, bx:bx + bw] == label
                cue_fraction = float(cue[by:by + bh, bx:bx + bw][pixels].mean())
                if cue_fraction < .15:
                    continue
                source_x, source_y = float(centers[label][0] + x), float(centers[label][1] + y)
                cx, cy = source_x / sx, source_y / sy
                if not self.roi[min(self.height - 1, int(cy)), min(self.width - 1, int(cx))]:
                    continue
                motion = float(temporal[by:by + bh, bx:bx + bw][pixels].mean())
                local_contrast = float(contrast[by:by + bh, bx:bx + bw][pixels].mean())
                score = .2 + .35 * min(1.0, motion / 90) + .3 * min(1.0, local_contrast / 65) + .15 * area / (bw * bh)
                native.append({"x": round(cx, 3), "y": round(cy, 3),
                               "radius": round(max(.5, math.sqrt(area / math.pi) / min(sx, sy)), 3),
                               "bbox": [(x + bx) / sx, (y + by) / sy, bw / sx, bh / sy],
                               "confidence": round(min(.95, score), 4),
                               "confidence_kind": "heuristic_rank_not_probability",
                               "method": "opencv_source_roi_component", "native_resolution_observation": True,
                               "source_px": [round(source_x, 3), round(source_y, 3)],
                               "source_bbox": [x + bx, y + by, bw, bh],
                               "source_size": [sw, sh], "roi_origin": window["origin"],
                               "motion_supported_fraction": round(cue_fraction, 4),
                               "foreground_difference": round(motion, 3), "local_contrast": round(local_contrast, 3)})
        self.previous = source  # one additional original frame, never the whole clip
        native = _nms(native, max(.5, self.width / 1280), 512)
        self.recent = (_candidate_budget(native, (self.height, self.width), 16, mode="spatial",
                                        distance=max(.5, self.width / 1280))
                       if self.config.get("candidate_selection") == "spatial" else native[:16])
        return native, {"status": "analyzed", "source_size": [sw, sh], "rois": windows,
                        "roi_count": len(windows), "proposal_count": len(native),
                        "pixels_examined": sum(row["bbox"][2] * row["bbox"][3] for row in windows),
                        "full_source_scan": False, "score_kind": "heuristic_rank_not_probability",
                        "local_component_segmentation": bool(self.config.get(
                            "native_local_components", bool(self.config.get("refinement_regions"))))}


def _merge_native(global_candidates, native_candidates, distance):
    """Prefer measured native location when both scales support one candidate."""
    pending = list(global_candidates)
    combined = []
    for native in native_candidates:
        matches = [item for item in pending
                   if math.hypot(item["x"] - native["x"], item["y"] - native["y"])
                   <= max(distance, min(item["radius"], native["radius"]))]
        refined = dict(native)
        if matches:
            best = max(matches, key=lambda item: item["confidence"])
            refined["global_confidence"] = best["confidence"]
            refined["native_confidence"] = native["confidence"]
            refined["confidence"] = max(best["confidence"], native["confidence"])
            refined["method"] = "opencv_source_roi_with_global_support"
            pending = [item for item in pending if item not in matches]
        combined.append(refined)
    return pending + combined


def detect_sequence(frames, timestamps, config=None, *, progress_callback=None):
    """Return frame-wise observations in the supplied frames' coordinate system.

    ``backend='opencv_candidates'`` is the default. ``backend='onnx'`` requires
    explicit weights and class mapping. Optional ``source_frames`` provides
    native ROI re-detection by default, or tiled ONNX inference. ``highres_enabled``
    can disable native ROI work. Sampled ``person_backend='hog'`` boxes are model
    hypotheses only. A ``scorer_path`` loads
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
    selection = config.get("candidate_selection", "ranked")
    if selection not in {"ranked", "spatial"}:
        raise ValueError("candidate_selection must be ranked or spatial")
    regions = _refinement_regions(config, frames[0].shape, len(frames))
    sample_ids = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), 9), dtype=int))
    background = np.median(np.stack([cv2.cvtColor(_bgr(frames[i]), cv2.COLOR_BGR2GRAY) for i in sample_ids]), axis=0).astype(np.uint8)
    onnx = _OnnxDetector(config) if backend == "onnx" else None
    source_frames = config.get("source_frames", frames)
    if len(source_frames) != len(frames):
        raise ValueError("source_frames must match frames in length")
    people = _HogPeople(config)
    scorer = None
    if config.get("scorer_path"):
        from .learning import load_candidate_scorer
        scorer = load_candidate_scorer(config["scorer_path"])
    provenance = onnx.provenance if onnx else {
        "backend": "opencv_candidates", "trained_model": False,
        "method": "bright_small_components_with_background_and_temporal_difference",
        "confidence_kind": "heuristic_rank_not_probability",
        "include_dark_components": bool(config.get('include_dark_components')),
    }
    provenance.update(candidate_selection=selection, refinement_region_count=len(regions),
                      refinement_region_source="operator_or_review_supplied" if regions else None,
                      refinement_provides_contact_evidence=False)
    min_area = float(config.get("min_area", max(1.0, 2 * scale * scale)))
    max_area = float(config.get("max_area", max(40.0, 180 * scale * scale)))
    if not math.isfinite(min_area) or not math.isfinite(max_area) or min_area <= 0 or max_area < min_area:
        raise ValueError("min_area and max_area must be positive finite ordered values")
    native_detector = (_NativeRoiDetector(config, frames[0].shape, background, roi, max_area)
                       if not onnx and config.get("highres_enabled", True) and "source_frames" in config else None)
    previous_gray = None
    recent_motion = np.zeros((height, width), np.float32)
    background_low = cv2.erode(background, np.ones((3, 3), np.uint8))
    background_high = cv2.dilate(background, np.ones((3, 3), np.uint8))
    phase_window = cv2.createHanningWindow((max(8, width // 2), max(8, height // 2)), cv2.CV_32F)
    texture_points = cv2.goodFeaturesToTrack(background, maxCorners=120, qualityLevel=0.02, minDistance=max(4, round(width / 80)))
    global_motion_supported = bool(texture_points is not None and len(texture_points) >= 24
                                   and cv2.contourArea(cv2.convexHull(texture_points)) >= width * height * 0.2)
    records = []
    progress = _FrameProgress(progress_callback, len(frames), '프레임별 셔틀콕 후보 탐지')
    progress.report(0, force=True)
    for index, (raw_frame, timestamp) in enumerate(zip(frames, timestamps)):
        active_regions = [region for region in regions if region["frame_start"] <= index <= region["frame_end"]]
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
        people_regions, person_diagnostics = people.detect(frame, index)
        occlusion_regions.extend(people_regions)
        high_resolution = {"status": "disabled" if not config.get("highres_enabled", True) else "source_frames_not_supplied",
                           "roi_count": 0, "proposal_count": 0, "full_source_scan": False}
        if onnx:
            source_frame = _bgr(source_frames[index])
            detections = onnx.detect(source_frame, frame.shape)
            detections = [d for d in detections if roi[min(height - 1, max(0, round(d["y"]))), min(width - 1, max(0, round(d["x"])))] > 0]
            proposal_count = len(detections)
            high_resolution = {"status": "onnx_tiled", "source_size": [source_frame.shape[1], source_frame.shape[0]],
                               "full_source_scan": True, "proposal_count": proposal_count}
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
            contrast = (cv2.absdiff(gray, local_base) if config.get('include_dark_components')
                        else cv2.subtract(gray, local_base))
            if config.get('include_dark_components'):
                dark = ((cv2.subtract(local_base, gray) >= 18) & (hsv[:, :, 1] <= 115))
                bright = np.maximum(bright, dark.astype(np.uint8) * 255)
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
            # Give the learned scorer a bounded proposal pool before the final
            # cutoff, otherwise discarded target proposals can never be rescued.
            detections = _nms(detections, max(1.0, 3 * scale),
                              512 if scorer or active_regions or selection == "spatial" else max_candidates)
        if native_detector is not None:
            native, high_resolution = native_detector.detect(source_frames[index], detections, index, camera_motion, active_regions)
            for item in native:
                item["possible_occluder_overlap"] = any(
                    b[0] <= item["x"] <= b[0] + b[2] and b[1] <= item["y"] <= b[1] + b[3]
                    for b in (region["bbox"] for region in occlusion_regions))
                if item["possible_occluder_overlap"]:
                    item["confidence"] = round(item["confidence"] * .85, 4)
            global_count = len(detections)
            detections = _merge_native(detections, native, max(1.0, 3 * scale))
            proposal_count += max(0, len(detections) - global_count)
            detections = _nms(detections, max(1.0, 3 * scale),
                              512 if scorer or active_regions or selection == "spatial" else max_candidates)
        if scorer:
            rescored = scorer.score(frame, detections, preprocessing=config.get("frame_preprocessing"))
            for old, new in zip(detections, rescored):
                new["heuristic_confidence"] = old["confidence"]
                if config.get("use_learned_confidence") and new.get("learned_confidence") is not None:
                    new["confidence"] = float(new["learned_confidence"])
                    new["confidence_kind"] = "learned_score_not_calibrated"
                else:
                    new["confidence"] = old["confidence"]
            detections = rescored
        detections, preferred_count = _refined_budget(detections, frame.shape, max_candidates,
                                                       selection, max(1.0, 3 * scale), active_regions)
        record = {"frame_index": index, "time": float(timestamp), "detections": detections,
                  "occlusion_regions": occlusion_regions, "detector": dict(provenance),
                  "person_detector": person_diagnostics, "high_resolution": high_resolution,
                  "camera_motion": camera_motion, "proposal_count_before_limit": proposal_count,
                  "candidate_limit_reached": proposal_count > max_candidates,
                  "refinement": {"active_regions": active_regions, "preferred_candidates": preferred_count,
                                 "assistance_used": bool(active_regions), "coordinate_system": "analysis_pixels",
                                 "provides_contact_evidence": False, "provides_target_identity": False}}
        if scorer:
            record["candidate_scorer"] = getattr(scorer, "provenance", {"path": str(config["scorer_path"])})
        records.append(record)
        previous_gray = gray
        progress.report(index + 1)
    progress.report(len(records), force=True)
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
    native_rows = [record.get("high_resolution", {}) for record in frame_records]
    native_frames = sum(row.get("status") == "analyzed" for row in native_rows)
    native_detections = sum(bool(item.get("native_resolution_observation"))
                            for record in frame_records for item in record.get("detections", []))
    person_sampled = sum(bool(record.get("person_detector", {}).get("sampled")) for record in frame_records)
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
            "native_roi_frames": native_frames,
            "native_roi_count": sum(row.get("roi_count", 0) for row in native_rows),
            "native_proposal_count": sum(row.get("proposal_count", 0) for row in native_rows if row.get("status") == "analyzed"),
            "native_retained_detection_count": native_detections,
            "person_detector_sampled_frames": person_sampled,
            "precision_measured": False, "recall_measured": False,
            "note": "Candidate volume and track ambiguity describe this run; they do not measure detection accuracy."}
