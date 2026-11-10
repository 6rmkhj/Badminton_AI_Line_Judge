"""Court-line segmentation and conservative projective template matching.

This is a deterministic OpenCV baseline, not a trained segmentation model.
Confidence is geometric image support, not a calibrated probability or a proven
metric error bound. Partial courts and competing fits remain REVIEW. Four manual
reference corners are available for annotation, never automatic acceptance.
"""
from __future__ import annotations

import itertools
import math

import cv2
import numpy as np


def lens_geometry(config, width, height):
    """Validate known camera intrinsics and scale them to analysis coordinates.

    Intrinsics must come from an independent lens calibration. A court fit is
    not enough to estimate lens distortion. ``image_size`` is [width, height]
    of the images used to measure the supplied camera matrix.
    """
    if not config:
        return None
    if not isinstance(config, dict) or config.get("model", "opencv_radial_tangential") != "opencv_radial_tangential":
        raise ValueError("known lens calibration currently requires the OpenCV radial/tangential model")
    matrix = np.asarray(config.get("camera_matrix"), dtype=np.float64)
    distortion = np.asarray(config.get("dist_coeffs"), dtype=np.float64).reshape(-1)
    size = np.asarray(config.get("image_size", [width, height]), dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose([matrix[0, 1], matrix[1, 0]], [0, 0])
            or distortion.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()
            or size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any()):
        raise ValueError("lens needs a finite camera_matrix, 4/5/8/12/14 dist_coeffs and positive image_size")
    scaled = np.diag([width / size[0], height / size[1], 1.0]) @ matrix
    return {"camera_matrix": scaled.tolist(), "dist_coeffs": distortion.tolist(),
            "image_size": [width, height], "reference_image_size": size.tolist(),
            "source": "provided_independent_camera_intrinsics", "model": "opencv_radial_tangential"}


def points_for_homography(points, calibration):
    """Map raw analysis-image points to the pixel space used to fit H.

    Detections and evidence remain in the original, distorted image. Consumers
    must call this before applying a homography fitted on rectified imagery.
    No airborne-point-to-floor conversion is performed by this helper.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("image points must be finite Nx2 coordinates")
    if calibration.get("homography_pixel_space", "raw_analysis_pixels") == "raw_analysis_pixels":
        return array.copy()
    if calibration.get("homography_pixel_space") != "undistorted_analysis_pixels":
        raise ValueError("unknown homography pixel space")
    lens = calibration.get("lens")
    if not isinstance(lens, dict):
        raise ValueError("rectified homography requires lens metadata")
    matrix = np.asarray(lens["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(lens["dist_coeffs"], dtype=np.float64)
    return cv2.undistortPoints(array.reshape(-1, 1, 2), matrix, distortion, P=matrix).reshape(-1, 2)


def _raw_points(points, lens):
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if lens is None:
        return array
    matrix = np.asarray(lens["camera_matrix"], dtype=np.float64)
    normalized = np.column_stack((array, np.ones(len(array)))) @ np.linalg.inv(matrix).T
    projected, _ = cv2.projectPoints(normalized, np.zeros(3), np.zeros(3), matrix,
                                     np.asarray(lens["dist_coeffs"], dtype=np.float64))
    return projected.reshape(-1, 2)


def assess_camera_stability(frames, config=None):
    """Measure robust background motion before reusing one court homography.

    Sparse optical flow is fitted with RANSAC against the first frame. Inlier
    coverage is required, so a moving player cannot alone certify camera
    stability. Unobservable or inconsistent estimates remain REVIEW. Motion is
    expressed in the same analysis pixels as the court and contact coordinates.
    """
    cfg = dict(config or {})
    if not frames:
        return {"status": "review", "reason_codes": ["CAMERA_STABILITY_NO_FRAMES"], "observations": []}
    height, width = frames[0].shape[:2]
    reduced_width = min(width, max(160, int(cfg.get("analysis_width", 640))))
    scale = reduced_width / width
    size = (reduced_width, max(1, round(height * scale)))
    threshold = float(cfg.get("max_displacement_px", 3.0))
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("max_displacement_px must be finite and positive")
    # A brief camera bump at the contact frame can invalidate H even when all
    # neighbouring samples are stable. The default therefore checks every
    # analysed frame. Explicit subsampling remains a diagnostic, never a basis
    # for automatic metric acceptance.
    count = min(len(frames), max(2, int(cfg.get("max_sample_frames", len(frames)))))
    indices = np.unique(np.linspace(0, len(frames) - 1, count).astype(int)).tolist()
    def gray(frame):
        return cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    reference = gray(frames[0])
    corners = cv2.goodFeaturesToTrack(reference, maxCorners=500, qualityLevel=.015,
                                     minDistance=8, blockSize=5)
    observations = []
    if corners is not None and len(corners) >= 12:
        for index in indices[1:]:
            current = gray(frames[index])
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(reference, current, corners, None,
                                                        winSize=(21, 21), maxLevel=3)
            if tracked is None or status is None:
                observations.append({"frame_index": index, "status": "unobservable"})
                continue
            good = status.ravel().astype(bool) & np.isfinite(tracked.reshape(-1, 2)).all(axis=1)
            before, after = corners.reshape(-1, 2)[good], tracked.reshape(-1, 2)[good]
            if len(before) < 12:
                observations.append({"frame_index": index, "status": "unobservable"})
                continue
            affine, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC,
                                                         ransacReprojThreshold=1.5, maxIters=1000)
            if affine is None or inliers is None:
                observations.append({"frame_index": index, "status": "unobservable"})
                continue
            accepted = inliers.ravel().astype(bool)
            support = int(accepted.sum())
            cells = np.floor(before[accepted] / np.array(size) * [4, 3]).astype(int)
            coverage = len(set(map(tuple, cells)))
            measured = support >= 12 and support / len(before) >= .5 and coverage >= 4
            boundary = np.array([[0, 0], [size[0]-1, 0], [size[0]-1, size[1]-1], [0, size[1]-1]], float)
            shifted = np.column_stack((boundary, np.ones(4))) @ affine.T
            displacement = float(np.linalg.norm(shifted-boundary, axis=1).max() / scale)
            observations.append({"frame_index": index, "status": "measured" if measured else "unobservable",
                                 "inlier_count": support, "inlier_fraction": round(support / len(before), 4),
                                 "spatial_grid_cells": coverage, "displacement_px": round(displacement, 4)})
    measured = [o for o in observations if o["status"] == "measured"]
    max_displacement = max((o["displacement_px"] for o in measured), default=None)
    reasons = []
    if len(indices) != len(frames):
        reasons.append("CAMERA_STABILITY_PARTIAL_COVERAGE")
    if max_displacement is not None and max_displacement > threshold:
        reasons.append("CAMERA_MOVED_FROM_CALIBRATION_REFERENCE")
    if len(measured) != len(indices) - 1 or not measured:
        reasons.append("CAMERA_STABILITY_UNOBSERVABLE")
    return {"status": "review" if reasons else "ok", "reason_codes": reasons or ["ALL_ANALYZED_FRAMES_BACKGROUND_STABLE"],
            "max_displacement_px": max_displacement, "threshold_px": threshold,
            "reference_frame_index": 0, "sampled_frame_indices": indices,
            "all_frames_sampled": len(indices) == len(frames), "observations": observations,
            "method": "reference_optical_flow_ransac_background_motion"}


def guard_calibration_stability(calibration, stability):
    """Invalidate metric acceptance without discarding useful fit hypotheses."""
    result = dict(calibration)
    result["camera_stability"] = stability
    incomplete = stability.get("all_frames_sampled") is False
    if stability.get("status") != "ok" or incomplete:
        if result.get("status") == "ok":
            result["status"] = "review"
        reasons = stability.get("reason_codes", ["CAMERA_STABILITY_UNCONFIRMED"])
        if incomplete:
            reasons = reasons + ["CAMERA_STABILITY_PARTIAL_COVERAGE"]
        result["reason_codes"] = list(dict.fromkeys(result.get("reason_codes", []) + reasons))
        result["judgment_region_px"] = None
        result["metric_coordinates_available"] = False
    return result


def _segment_model(background, config, roi_polygon=None):
    """Optional ONNX binary line segmentation, with explicit preprocessing."""
    size = config.get("line_model_input_size", [512, 512])
    if len(size) != 2 or min(map(int, size)) < 16 or max(map(int, size)) > 4096:
        raise ValueError("line_model_input_size must be [width, height] between 16 and 4096")
    rgb = cv2.cvtColor(cv2.resize(background, tuple(map(int, size))), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    mean = np.asarray(config.get("line_model_mean_rgb", [0, 0, 0]), np.float32)
    std = np.asarray(config.get("line_model_std_rgb", [1, 1, 1]), np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("line-model RGB mean/std must be finite three-vectors, with positive std")
    network = cv2.dnn.readNetFromONNX(str(config["line_model_path"]))
    network.setInput(np.transpose((rgb-mean)/std, (2, 0, 1))[None])
    output = np.asarray(network.forward(), dtype=float)
    if output.ndim == 4 and output.shape[0] == 1:
        output = output[0]
    if output.ndim == 3:
        channel = int(config.get("line_model_output_channel", 0))
        if not 0 <= channel < len(output):
            raise ValueError("line_model_output_channel out of range")
        output = output[channel]
    if output.ndim != 2 or min(output.shape) < 2 or not np.isfinite(output).all():
        raise ValueError("line model must return a finite [1,C,H,W], [C,H,W], or [H,W] mask")
    output_kind = config.get("line_model_output_kind", "probability")
    if output_kind == "logit":
        output = 1 / (1 + np.exp(-np.clip(output, -60, 60)))
    elif output_kind != "probability" or output.min() < 0 or output.max() > 1:
        raise ValueError("line output must be probabilities in [0,1] or declared logit")
    threshold = float(config.get("line_model_threshold", .5))
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("line_model_threshold must be in (0,1)")
    mask = (cv2.resize(output.astype(np.float32), (background.shape[1], background.shape[0])) >= threshold).astype(np.uint8) * 255
    if roi_polygon is not None:
        roi = np.zeros_like(mask)
        cv2.fillPoly(roi, [np.asarray(roi_polygon, np.int32)], 255)
        mask = cv2.bitwise_and(mask, roi)
    return mask


COURT_CORNERS_M = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], np.float32)
TEMPLATE_LINES_M = [
    ((0, 0), (0, 13.4)), ((6.1, 0), (6.1, 13.4)),
    ((0, 0), (6.1, 0)), ((0, 13.4), (6.1, 13.4)),
    ((0.46, 0), (0.46, 13.4)), ((5.64, 0), (5.64, 13.4)),
    ((0, 0.76), (6.1, 0.76)), ((0, 12.64), (6.1, 12.64)),
    ((0, 4.72), (6.1, 4.72)), ((0, 8.68), (6.1, 8.68)),
    ((3.05, 0), (3.05, 4.72)), ((3.05, 8.68), (3.05, 13.4)),
]


def court_template_lines():
    return [[list(a), list(b)] for a, b in TEMPLATE_LINES_M]


def segment_lines(background: np.ndarray, roi_polygon=None):
    """Return a uint8 mask of blue tape, or white tape when blue is absent."""
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([85, 65, 35]), np.array([140, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([179, 65, 255]))
    if roi_polygon is not None:
        roi = np.zeros(background.shape[:2], np.uint8)
        cv2.fillPoly(roi, [np.asarray(roi_polygon, np.int32)], 255)
        blue = cv2.bitwise_and(blue, roi)
        white = cv2.bitwise_and(white, roi)
    chosen, colour = (blue, "blue") if np.count_nonzero(blue) > background.shape[0] * background.shape[1] * .0015 else (white, "white")
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(chosen)
    clean = np.zeros_like(chosen)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= 20:
            clean[labels == component] = 255
    return clean, colour


def _line(segment):
    x1, y1, x2, y2 = map(float, segment)
    length = math.hypot(x2-x1, y2-y1)
    if length < 1e-6:
        return None
    vector = np.array([y1-y2, x2-x1, x1*y2-x2*y1], float)
    vector /= np.linalg.norm(vector[:2])
    if vector[0] < 0 or (abs(vector[0]) < 1e-8 and vector[1] < 0):
        vector = -vector
    angle = math.atan2(y2-y1, x2-x1) % np.pi
    return {"line": vector, "segment": [x1, y1, x2, y2], "length": length, "angle": angle}


def _angle_distance(a, b):
    distance = abs(a-b) % np.pi
    return min(distance, np.pi-distance)


def _merged_lines(mask):
    height, width = mask.shape
    raw = cv2.HoughLinesP(mask, 1, np.pi/360, max(35, int(width*.055)),
                          minLineLength=max(40, int(width*.10)), maxLineGap=max(8, int(width*.025)))
    if raw is None:
        return []
    candidates = [_line(segment) for segment in raw[:, 0]]
    candidates = sorted((r for r in candidates if r), key=lambda r: r["length"], reverse=True)
    lines = []
    for candidate in candidates:
        midpoint = np.array([(candidate["segment"][0]+candidate["segment"][2])/2,
                             (candidate["segment"][1]+candidate["segment"][3])/2, 1])
        if any(_angle_distance(candidate["angle"], old["angle"]) < np.deg2rad(2.5)
               and abs(float(old["line"] @ midpoint)) < max(5, width*.006) for old in lines):
            continue
        lines.append(candidate)
        if len(lines) >= 32:
            break
    return lines


def _families(lines):
    """Deterministic circular two-cluster orientation grouping."""
    if len(lines) < 4:
        return [], []
    vectors = np.array([[math.cos(2*r["angle"]), math.sin(2*r["angle"])] for r in lines])
    centres = np.array([vectors[0], vectors[np.argmin(vectors @ vectors[0])]])
    weights = np.array([r["length"] for r in lines])
    assignment = None
    for _ in range(12):
        new = np.argmax(vectors @ centres.T, axis=1)
        if assignment is not None and np.array_equal(new, assignment):
            break
        assignment = new
        for i in range(2):
            if np.any(new == i):
                centre = np.sum(vectors[new == i] * weights[new == i, None], axis=0)
                norm = np.linalg.norm(centre)
                if norm:
                    centres[i] = centre/norm
    groups = [[row for i, row in enumerate(lines) if assignment[i] == k][:12] for k in range(2)]
    return groups


def _intersection(a, b):
    cross = np.cross(a["line"], b["line"])
    if abs(cross[2]) < .025:
        return None
    return cross[:2] / cross[2]


def _orientations(points):
    hull = cv2.convexHull(np.asarray(points, np.float32)).reshape(-1, 2)
    if len(hull) != 4:
        return []
    variants = []
    for parity in range(2):
        edge = max((parity, parity+2), key=lambda i: (hull[i][1]+hull[(i+1)%4][1])/2)
        ordered = np.array([hull[(edge+i)%4] for i in range(4)], np.float32)
        if ordered[0, 0] > ordered[1, 0]:
            ordered = ordered[[1, 0, 3, 2]]
        variants.append(ordered)
    return variants


def _proposals(lines, width, height):
    groups = _families(lines)
    if min(map(len, groups), default=0) < 2:
        return []
    candidates = []
    for aa in itertools.combinations(groups[0], 2):
        for bb in itertools.combinations(groups[1], 2):
            corners = [_intersection(a, b) for a, b in ((aa[0], bb[0]), (aa[0], bb[1]), (aa[1], bb[1]), (aa[1], bb[0]))]
            if any(c is None for c in corners):
                continue
            points = np.array(corners, np.float32)
            if not cv2.isContourConvex(points):
                continue
            area = abs(cv2.contourArea(points))
            if not width*height*.06 <= area <= width*height*1.25:
                continue
            # Outside corners are retained only as partial-court evidence, never
            # as sufficient full-court calibration.
            if (points[:, 0] < -.12*width).any() or (points[:, 0] > 1.12*width).any() or (points[:, 1] < -.12*height).any() or (points[:, 1] > 1.12*height).any():
                continue
            if np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).min() < min(width, height)*.08:
                continue
            strength = sum(r["length"] for r in (*aa, *bb))
            candidates.append((strength, area, points))
    # Limit template evaluations so processing remains bounded for cluttered halls.
    candidates.sort(key=lambda r: (r[0], r[1]), reverse=True)
    return [p for _, _, points in candidates[:400] for p in _orientations(points)]


_TEMPLATE_SAMPLES = [np.linspace(a, b, 48, dtype=np.float32) for a, b in TEMPLATE_LINES_M]
_TEMPLATE_SAMPLES_ARRAY = np.concatenate(_TEMPLATE_SAMPLES)


def _score(polygon, distance, mask):
    height, width = mask.shape
    homography = cv2.getPerspectiveTransform(polygon, COURT_CORNERS_M)
    inverse = cv2.getPerspectiveTransform(COURT_CORNERS_M, polygon)
    projected = cv2.perspectiveTransform(_TEMPLATE_SAMPLES_ARRAY[None], inverse)[0]
    good = ((projected[:, 0] >= 0) & (projected[:, 0] < width) & (projected[:, 1] >= 0) & (projected[:, 1] < height))
    coords = np.rint(projected).astype(int)
    coords[:, 0] = coords[:, 0].clip(0, width-1)
    coords[:, 1] = coords[:, 1].clip(0, height-1)
    errors = distance[coords[:, 1], coords[:, 0]]
    tolerance = max(2.5, width*.0035)
    supports = ((errors <= tolerance) & good).reshape(len(TEMPLATE_LINES_M), -1).mean(axis=1)
    visible = good.reshape(len(TEMPLATE_LINES_M), -1).mean(axis=1)
    interior = supports[4:]
    score = float(.30*supports[:4].mean() + .70*interior.mean())
    count = int(np.count_nonzero(supports >= .65))
    return {"polygon": polygon, "homography": homography, "score": score,
            "line_support": supports.tolist(), "line_visibility": visible.tolist(),
            "supported_line_count": count, "visible_fraction": float(good.mean()),
            "line_fit_error_px": float(np.mean(np.minimum(errors[good], tolerance*4))) if good.any() else tolerance*4}


def _iou(a, b):
    intersection, _ = cv2.intersectConvexConvex(np.asarray(a, np.float32), np.asarray(b, np.float32))
    union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - intersection
    return intersection/union if union > 0 else 0.0


def _manual_reference(hint, width, height):
    points = np.asarray(hint.get("reference_points_px", hint.get("corners_px")), dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all() or not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 100:
        raise ValueError("manual reference requires four ordered convex corners")
    target = np.asarray(hint.get("reference_points_m", COURT_CORNERS_M), np.float32)
    if target.shape != (4, 2) or not np.isfinite(target).all():
        raise ValueError("manual reference metric points must be four Nx2 points")
    homography = cv2.getPerspectiveTransform(points, target)
    if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
        raise ValueError("singular manual reference")
    return {"status": "review", "homography": homography.tolist(), "polygon_px": points.tolist(),
            "confidence": 0.0, "reason_codes": ["MANUAL_CALIBRATION_REFERENCE_ONLY"],
            "method": "manual_reference", "manual_reference": True,
            "image_width": width, "image_height": height, "line_fit_error_px": None,
            "independent_control_point_error_px": None}


def calibrate(frames: list[np.ndarray], court_hint: dict | None = None, config: dict | None = None) -> dict:
    """Estimate the full doubles-court plane from several stationary-camera frames.

    Optional ``court_hint['roi_polygon_px']`` restricts a target area without
    supplying its metric geometry. ``reference_points_px``/``corners_px`` are
    ordered [near-left, near-right, far-right, far-left] manual references and
    always return REVIEW. A homography on a REVIEW result is a hypothesis only.
    """
    cfg = dict(config or {})
    base = {"status": "failed", "homography": None, "polygon_px": None, "confidence": 0.0,
            "reason_codes": [], "method": "colour_hough_court_template",
            "manual_reference": False, "confidence_kind": "geometric support score, not a calibrated probability",
            "image_width": None, "image_height": None, "line_segments_px": [],
            "candidate_polygons_px": [], "line_fit_error_px": None,
            "independent_control_point_error_px": None, "lens_distortion_corrected": False,
            "coordinate_convention": "x=0..6.1 m left-to-right across near baseline, y=0..13.4 m near-to-far",
            "metric_accuracy_validated": False, "metric_coordinates_available": False,
            "homography_pixel_space": "raw_analysis_pixels", "line_fit_error_pixel_space": "raw_analysis_pixels",
            "judgment_region_px": None,
            "supported_line_regions_px": [], "line_roles_confirmed": False}
    if not frames:
        return {**base, "reason_codes": ["NO_VIDEO_FRAMES"]}
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        return {**base, "reason_codes": ["INVALID_VIDEO_FRAME_SHAPE"]}
    height, width = first.shape[:2]
    base.update(image_width=width, image_height=height)
    selected = [np.asarray(frames[int(i)]) for i in np.linspace(0, len(frames)-1, min(5, len(frames)))]
    if any(f.shape != first.shape or f.dtype != np.uint8 for f in selected):
        return {**base, "reason_codes": ["INCONSISTENT_VIDEO_FRAME_FORMAT"]}
    try:
        lens = lens_geometry(cfg.get("lens"), width, height)
    except (ValueError, TypeError, KeyError):
        return {**base, "reason_codes": ["INVALID_KNOWN_LENS_CALIBRATION"]}
    if lens:
        base.update(lens=lens, lens_distortion_corrected=True,
                    homography_pixel_space="undistorted_analysis_pixels",
                    line_fit_error_pixel_space="undistorted_analysis_pixels")
        matrix, distortion = np.asarray(lens["camera_matrix"]), np.asarray(lens["dist_coeffs"])
        selected = [cv2.undistort(frame, matrix, distortion, None, matrix) for frame in selected]
    hint = court_hint or {}
    if "reference_points_px" in hint or "corners_px" in hint:
        try:
            transformed = dict(hint)
            key = "reference_points_px" if "reference_points_px" in hint else "corners_px"
            transformed[key] = points_for_homography(hint[key], base).tolist()
            result = {**base, **_manual_reference(transformed, width, height)}
            result["polygon_px"] = _raw_points(result["polygon_px"], lens).tolist()
            return result
        except (ValueError, TypeError, cv2.error):
            return {**base, "reason_codes": ["INVALID_MANUAL_REFERENCE"]}
    roi = hint.get("roi_polygon_px", hint.get("roi"))
    if roi is not None:
        try:
            roi = np.asarray(roi, float)
            if roi.ndim != 2 or roi.shape[1] != 2 or len(roi) < 3 or not np.isfinite(roi).all():
                raise ValueError("invalid ROI")
            roi = points_for_homography(roi, base)
        except (ValueError, TypeError):
            return {**base, "reason_codes": ["INVALID_TARGET_COURT_ROI"]}
    background = np.median(np.stack(selected), axis=0).astype(np.uint8)
    if cfg.get("line_model_path"):
        try:
            mask, colour = _segment_model(background, cfg, roi), "learned_line_mask"
            base.update(method="onnx_line_segmentation_hough_court_template",
                        line_model_path=str(cfg["line_model_path"]), line_model_accuracy_validated=False)
        except (ValueError, TypeError, KeyError, OSError, cv2.error) as error:
            return {**base, "reason_codes": ["LINE_SEGMENTATION_MODEL_FAILED"], "model_error": str(error)[:500]}
    else:
        mask, colour = segment_lines(background, roi)
    lines = _merged_lines(mask)
    raw_segments = [_raw_points(np.asarray(r["segment"]).reshape(2, 2), lens).reshape(-1).tolist() for r in lines]
    regions = []
    for segment in raw_segments:
        a, b = np.asarray(segment).reshape(2, 2)
        direction = b-a
        normal = np.array([-direction[1], direction[0]]) / max(np.linalg.norm(direction), 1e-9)
        polygon = np.array([a+normal*3, b+normal*3, b-normal*3, a-normal*3])
        polygon[:, 0] = polygon[:, 0].clip(0, width-1)
        polygon[:, 1] = polygon[:, 1].clip(0, height-1)
        regions.append({"polygon_px": polygon.tolist(), "line_segment_px": segment,
                        "court_line_role": None, "metric_coordinates_available": False})
    base.update(line_segments_px=raw_segments, line_colour=colour,
                supported_line_regions_px=regions,
                background_frame_count=len(selected), mask_pixel_count=int(np.count_nonzero(mask)))
    if len(lines) < 4:
        return {**base, "reason_codes": ["INSUFFICIENT_COURT_LINES"],
                "partial_line_mode": bool(lines)}
    proposals = _proposals(lines, width, height)
    if not proposals:
        return {**base, "status": "review", "reason_codes": ["PARTIAL_LINES_NO_UNIQUE_COURT_PLANE"],
                "partial_line_mode": True}
    distance = cv2.distanceTransform(255-mask, cv2.DIST_L2, 3)
    scored = sorted((_score(polygon, distance, mask) for polygon in proposals), key=lambda r: r["score"], reverse=True)
    distinct = []
    for candidate in scored:
        if not any(_iou(candidate["polygon"], existing["polygon"]) > .90 for existing in distinct):
            distinct.append(candidate)
        if len(distinct) == 6:
            break
    best = scored[0]
    runner_up = next((r for r in distinct if _iou(r["polygon"], best["polygon"]) < .75), None)
    margin = best["score"] - runner_up["score"] if runner_up else best["score"]
    corners_inside = bool((best["polygon"][:, 0] >= 2).all() and (best["polygon"][:, 0] < width-2).all()
                          and (best["polygon"][:, 1] >= 2).all() and (best["polygon"][:, 1] < height-2).all())
    reasons = []
    if best["score"] < .74 or best["supported_line_count"] < 9 or min(best["line_support"][:4]) < .60:
        reasons.append("COURT_TEMPLATE_SUPPORT_INSUFFICIENT")
    if not corners_inside or best["visible_fraction"] < .97:
        reasons.append("PARTIAL_COURT_VISIBLE")
    if runner_up and margin < .08:
        reasons.append("MULTIPLE_COURT_FITS_AMBIGUOUS")
    # A camera cannot establish a surveyed metric error bound from image support
    # alone. The support score merely determines whether this baseline fit can be
    # passed to downstream evidence gates, not official judging accuracy.
    raw_polygon = _raw_points(best["polygon"], lens).tolist()
    base.update(status="review" if reasons else "ok", homography=best["homography"].tolist(),
                polygon_px=raw_polygon, polygon_homography_space_px=best["polygon"].tolist(),
                confidence=round(best["score"], 6),
                reason_codes=reasons or ["FULL_COURT_TEMPLATE_SUPPORTED"],
                line_fit_error_px=round(best["line_fit_error_px"], 4),
                line_support=best["line_support"], supported_line_count=best["supported_line_count"],
                visible_fraction=best["visible_fraction"], competing_fit_margin=round(margin, 6),
                partial_line_mode=not corners_inside,
                judgment_region_px=raw_polygon if not reasons else None,
                metric_coordinates_available=not reasons, line_roles_confirmed=not reasons,
                candidate_polygons_px=[{"polygon_px": _raw_points(r["polygon"], lens).tolist(), "support_score": round(r["score"], 6)} for r in distinct],
                template_lines_m=court_template_lines())
    return base
