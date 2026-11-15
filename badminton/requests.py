"""Explicit, line-responsible requests over independently analysed camera clips.

A shared file number or a matching model decision does not establish event
correspondence. One validated view may resolve its own request; combining partial
responsibilities requires confirmed event correspondence. No clock synchronization, 3D
reconstruction, annotation, or ground-truth label is generated here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.parse import quote

from .decision import evaluate_contact, rule_region


BOUNDARIES = frozenset(("left", "right", "near", "far"))
_SOURCE_HASH = re.compile(r"[0-9a-f]{64}")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(f"{name} must be a nonempty string of at most 128 characters")
    return value


def _window(value, *, allow_single_time=False):
    if (not isinstance(value, list) or len(value) != 2 or
            not all(_number(v) for v in value) or value[0] < 0 or value[1] < value[0] or
            (value[0] == value[1] and not allow_single_time)):
        operator = "<=" if allow_single_time else "<"
        raise ValueError(f"event_window_seconds must be [start, end] with 0 <= start {operator} end")
    return [float(value[0]), float(value[1])]


def _observed_window(window, timestamps, presentation_end, description):
    """Map a requested display interval to the frames that actually exist.

    A video remains displayed until its final frame ends, whereas event/identity
    algorithms accept frame timestamps only. Preserve the request externally,
    but never send the final frame's *end time* as an observed frame timestamp.
    """
    if window[1] > presentation_end + 1e-6:
        raise ValueError(f"{description} exceeds displayed timeline")
    indices = [i for i, t in enumerate(timestamps) if window[0] <= t <= window[1]]
    if not indices:
        raise ValueError(f"{description} has no displayed frames")
    return [float(timestamps[indices[0]]), float(timestamps[indices[-1]])], [indices[0], indices[-1]]


def _video_path(value, root):
    if not isinstance(value, str) or not value:
        raise ValueError("Every view requires a relative video path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError("video must be a relative path inside video_root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Video is missing or outside video_root: {value}")
    return path


def validate_request(request, video_root, base_config=None):
    """Validate the entire request before starting any expensive analysis.

    Window and anchor times are relative to each video's first displayed frame.
    Anchor positions use the analysis image dimensions, after display rotation.
    """
    from .media import probe

    if not isinstance(request, dict) or request.get("schema_version", 1) != 1:
        raise ValueError("Expected a version 1 request object")
    root = Path(video_root).resolve()
    if not root.is_dir():
        raise ValueError("video_root must be an existing directory")
    if base_config is not None and not isinstance(base_config, dict):
        raise ValueError("base_config must be an object")
    cfg = base_config or {}
    identifier = _identifier(request.get("request_id"), "request_id")
    rules = request.get("rules")
    if not isinstance(rules, dict):
        raise ValueError("A request requires explicit rules")
    rule_region(rules)
    requested = request.get("requested_boundary", "all")
    if not isinstance(requested, str) or requested not in BOUNDARIES | {"all"}:
        raise ValueError("requested_boundary must be left, right, near, far, or all")
    same_event = request.get("same_event_confirmed", False)
    if not isinstance(same_event, bool):
        raise ValueError("same_event_confirmed must be a boolean")
    shared_coordinates = request.get("same_court_coordinate_system_confirmed", False)
    if not isinstance(shared_coordinates, bool):
        raise ValueError("same_court_coordinate_system_confirmed must be a boolean")
    views = request.get("views")
    if not isinstance(views, list) or not 1 <= len(views) <= 2:
        raise ValueError("The two-camera PoC accepts one or two views")
    normalized, camera_ids, paths = [], set(), set()
    for view in views:
        if not isinstance(view, dict):
            raise ValueError("Every view must be an object")
        camera = _identifier(view.get("camera_id"), "camera_id")
        path = _video_path(view.get("video"), root)
        if camera in camera_ids or path in paths:
            raise ValueError("Camera IDs and source paths must be unique within a request")
        camera_ids.add(camera)
        paths.add(path)
        responsibility = view.get("responsible_boundaries")
        if (not isinstance(responsibility, list) or not responsibility or
                not all(isinstance(v, str) and v in BOUNDARIES for v in responsibility) or
                len(set(responsibility)) != len(responsibility)):
            raise ValueError("responsible_boundaries must name distinct left/right/near/far boundaries")
        window = _window(view.get("event_window_seconds"))
        metadata = probe(path)
        timestamps = metadata["timestamps"]
        last_timestamp = float(timestamps[-1])
        end = last_timestamp + 1 / float(metadata["fps"])
        if _number(metadata.get("duration")):
            end = max(end, float(metadata["duration"]))
        analysis_window, frame_range = _observed_window(window, timestamps, end, f"Event window for {camera}")
        width = min(int(cfg.get("analysis_width", 1280)), metadata["width"])
        width -= width % 2
        if width < 160:
            raise ValueError("Analysis width must be at least 160")
        height = max(2, round(metadata["height"] * width / metadata["width"] / 2) * 2)
        anchors = view.get("hit_anchors", [])
        if not isinstance(anchors, list) or len(anchors) > 100:
            raise ValueError("hit_anchors must be a list of at most 100 anchors")
        normalized_anchors = []
        for anchor in anchors:
            if not isinstance(anchor, dict):
                raise ValueError("Every hit anchor must be an object")
            if anchor.get("coordinate_system", "analysis_pixels") != "analysis_pixels":
                raise ValueError("Hit anchors must use coordinate_system=analysis_pixels")
            if anchor.get("source", "manual_hit_anchor") != "manual_hit_anchor":
                raise ValueError("Hit anchors must have source=manual_hit_anchor")
            t, point = anchor.get("time"), anchor.get("point_px")
            if not _number(t) or not timestamps[0] <= t <= min(last_timestamp, window[1]):
                raise ValueError("Hit anchor must precede the requested landing window end and lie within observed frame timestamps")
            if (not isinstance(point, list) or len(point) != 2 or not all(_number(v) for v in point) or
                    not 0 <= point[0] < width or not 0 <= point[1] < height):
                raise ValueError("Hit anchor point_px must lie inside the analysis image")
            radius = anchor.get("radius_px", 24.0)
            tolerance = anchor.get("time_tolerance_seconds", 0.1)
            if not _number(radius) or radius <= 0 or not _number(tolerance) or tolerance <= 0:
                raise ValueError("Hit anchor radius and time tolerance must be positive and finite")
            normalized_anchor = {"time": float(t), "point_px": list(point),
                                 "radius_px": float(radius), "time_tolerance_seconds": float(tolerance),
                                 "coordinate_system": "analysis_pixels", "source": "manual_hit_anchor"}
            if anchor.get("event_window_seconds") is not None:
                scope = _window(anchor["event_window_seconds"], allow_single_time=True)
                observed_scope, _ = _observed_window(scope, timestamps, end, "Hit anchor event window")
                normalized_anchor.update(event_window_seconds=observed_scope,
                                         requested_event_window_seconds=scope)
            normalized_anchors.append(normalized_anchor)
        normalized.append({"camera_id": camera, "video": path.relative_to(root).as_posix(),
                           "source_path": str(path), "event_window_seconds": window,
                           "analysis_event_window_seconds": analysis_window,
                           "window_frame_range": frame_range,
                           "analysis_size": [width, height], "hit_anchors": normalized_anchors,
                           "responsible_boundaries": list(responsibility)})
    return {"schema_version": 1, "request_id": identifier, "rules": deepcopy(rules),
            "requested_boundary": requested, "same_event_confirmed": same_event,
            "same_court_coordinate_system_confirmed": shared_coordinates, "views": normalized}


def _geometry(decision):
    polygon, error = decision.get("contact_polygon_m"), decision.get("uncertainty_m")
    if (not isinstance(polygon, list) or not polygon or not _number(error) or error < 0 or
            any(not isinstance(p, list) or len(p) != 2 or not all(_number(v) for v in p) for p in polygon)):
        return None
    center = [sum(p[i] for p in polygon) / len(polygon) for i in (0, 1)]
    radius = max(math.dist(center, p) for p in polygon)
    return polygon, float(error), center, radius


def _pixel_decision_geometry(decision, region):
    """Recognize an already gated local-boundary decision, without inventing metres."""
    if (decision.get("decision_method") != "primary_court_observed_boundaries"
            or decision.get("coordinate_system") != "analysis_pixels"
            or decision.get("geometry_only") is not False):
        return None
    polygon, error = decision.get("contact_polygon_px"), decision.get("uncertainty_px")
    if (not isinstance(polygon, list) or not polygon or not _number(error) or error < 0
            or any(not isinstance(p, list) or len(p) != 2 or not all(_number(v) for v in p) for p in polygon)):
        return None
    rules = decision.get("rules")
    if (not isinstance(rules, dict)
            or any(rules.get(key) != region[key] for key in ("game", "phase", "xmin", "xmax", "ymin", "ymax"))):
        return None
    boundaries = decision.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        return None
    outside, roles = set(), set()
    for edge in boundaries:
        if (not isinstance(edge, dict) or edge.get("role") not in BOUNDARIES or edge["role"] in roles
                or not all(_number(edge.get(key)) for key in
                           ("min_inside_margin_px", "max_inside_margin_px", "uncertainty_px"))
                or edge["uncertainty_px"] < 0
                or edge["min_inside_margin_px"] > edge["max_inside_margin_px"]):
            return None
        roles.add(edge["role"])
        if (edge.get("robust_outside") is True and edge.get("within_observed_extent") is True
                and edge["max_inside_margin_px"] < -edge["uncertainty_px"]):
            outside.add(edge["role"])
    if decision.get("decision") == "IN" and (outside or not {"left", "right", "near"} <= roles):
        return None
    return outside


def _decision_rules_match(decision, region):
    """Legacy request decisions inherit their submitted rules; explicit rules must agree."""
    if "rules" not in decision:
        return True
    rules = decision["rules"]
    if not isinstance(rules, dict):
        return False
    keys = ("game", "phase", "xmin", "xmax", "ymin", "ymax")
    if not all(key in rules for key in keys):
        try:
            rules = rule_region(rules)
        except (ValueError, TypeError):
            return False
    return all(rules.get(key) == region[key] for key in keys)


def aggregate_request(request, views):
    """Use one validated responsible view when another view is unresolved.

    IN needs a donor covering the requested boundaries; OUT needs one observed
    assigned boundary proving exclusion. Partial IN responsibilities may only be
    combined with confirmed common event/metric coordinates. Pixel coordinates
    from separate cameras are never compared, merged, or transferred.
    """
    required = BOUNDARIES if request["requested_boundary"] == "all" else {request["requested_boundary"]}
    region = rule_region(request["rules"])
    shared_coordinates = request.get("same_court_coordinate_system_confirmed") is True
    same_event = request.get("same_event_confirmed") is True
    covered, relevant, donors, warnings, reasons = set(), [], [], [], []
    if len(views) > 1 and not same_event:
        warnings.append("CROSS_VIEW_EVENT_CORRESPONDENCE_UNCONFIRMED")
    if len(views) > 1 and not shared_coordinates:
        warnings.append("CROSS_VIEW_COURT_COORDINATE_SYSTEM_UNCONFIRMED")
    hashes = [view.get("source_sha256") for view in views if view.get("source_sha256")]
    duplicates = len(hashes) != len(set(hashes))
    if duplicates:
        warnings.append("DUPLICATE_SOURCE_IS_NOT_AN_INDEPENDENT_VIEW")
    for view in views:
        assigned = set(view["responsible_boundaries"]) & required
        if not assigned:
            continue
        covered.update(assigned)
        relevant.append(view)
        decision = view.get("decision", {})
        label = decision.get("decision", "REVIEW")
        if view.get("status") != "completed" or label not in {"IN", "OUT"}:
            warnings.append("RESPONSIBLE_VIEW_UNRESOLVED")
            continue
        digest = view.get("source_sha256")
        if not isinstance(digest, str) or not _SOURCE_HASH.fullmatch(digest):
            warnings.append("VIEW_SOURCE_UNVERIFIED")
            continue
        if not _decision_rules_match(decision, region):
            warnings.append("VIEW_RULES_MISMATCH")
            continue
        if decision.get("geometry_only") is True:
            warnings.append("VIEW_CONTACT_GEOMETRY_UNAVAILABLE")
            continue
        contact_time = decision.get("contact_time_seconds")
        start, end = view["event_window_seconds"]
        if not _number(contact_time) or not start <= contact_time <= end:
            warnings.append("CONTACT_OUTSIDE_REQUESTED_EVENT_WINDOW")
            continue
        geometry = _geometry(decision)
        outside = set()
        if geometry is not None:
            polygon, error, _, _ = geometry
            # Imported labels cannot override their own contact geometry.
            if evaluate_contact(polygon, request["rules"], error)["decision"] != label:
                warnings.append("VIEW_CONTACT_GEOMETRY_UNAVAILABLE")
                continue
            x, y = [p[0] for p in polygon], [p[1] for p in polygon]
            if max(x) + error < region["xmin"]:
                outside.add("left")
            if min(x) - error > region["xmax"]:
                outside.add("right")
            if max(y) + error < region["ymin"]:
                outside.add("near")
            if min(y) - error > region["ymax"]:
                outside.add("far")
        else:
            outside = _pixel_decision_geometry(decision, region)
            if outside is None:
                warnings.append("VIEW_CONTACT_GEOMETRY_UNAVAILABLE")
                continue
        outside &= assigned
        if label == "OUT" and not outside:
            warnings.append("OUT_NOT_SUPPORTED_BY_ASSIGNED_BOUNDARY")
            continue
        donors.append({"view": view, "label": label, "assigned": assigned,
                       "geometry": geometry, "outside": outside})
    if required - covered:
        warnings.append("REQUESTED_BOUNDARY_WITHOUT_RESPONSIBLE_CAMERA")
    if len({donor["label"] for donor in donors}) > 1:
        reasons.append("CONFLICTING_VIEW_DECISIONS")
    agreements = []
    linked = same_event and shared_coordinates and not duplicates
    if linked:
        for index, first in enumerate(donors):
            for second in donors[index+1:]:
                a, b = first["geometry"], second["geometry"]
                if a is None or b is None:
                    continue  # Local camera pixels are not a shared court plane.
                distance = math.dist(a[2], b[2])
                allowance = a[1] + b[1] + a[3] + b[3] + .02
                agreements.append({"camera_a": first["view"]["camera_id"],
                                   "camera_b": second["view"]["camera_id"],
                                   "distance_m": distance, "allowance_m": allowance})
                if distance > allowance:
                    reasons.append("CROSS_VIEW_CONTACT_POSITION_DISAGREEMENT")
    selected = []
    if donors and not reasons:
        independent = [donor for donor in donors
                       if donor["label"] == "OUT" or required <= donor["assigned"]]
        if independent:
            selected = independent if linked and all(d["geometry"] is not None for d in independent) else independent[:1]
        elif (linked and all(d["geometry"] is not None for d in donors)
              and required <= set().union(*(d["assigned"] for d in donors))):
            selected = donors
        else:
            reasons.append("IN_NOT_SUPPORTED_BY_ASSIGNED_BOUNDARIES")
    if not selected:
        reasons.extend(warnings)
        reasons.append("NO_UNIQUE_RESOLVED_REQUEST_DECISION")
    label = selected[0]["label"] if selected else "REVIEW"
    outside_supported = set().union(*(donor["outside"] for donor in selected)) if selected else set()
    source_ids = [donor["view"]["camera_id"] for donor in selected]
    return {"decision": label,
            "reason_codes": sorted(set(reasons)) or ["VERDICT_FROM_QUALIFIED_VIEW" if len(selected) == 1 else "RESPONSIBLE_VIEWS_AGREE"],
            "warning_codes": sorted(set(warnings)),
            "decision_scope": "single_view" if len(selected) == 1 else "linked_views" if selected else "unresolved",
            "source_camera_ids": source_ids,
            "requested_boundary": request["requested_boundary"],
            "responsible_camera_ids": [v["camera_id"] for v in relevant],
            "covered_boundaries": sorted(covered), "outside_supported_boundaries": sorted(outside_supported),
            "cross_view_contact_checks": agreements, "court_position_m": None,
            "same_court_coordinate_system_confirmed": shared_coordinates,
            "automatic_benchmark_eligible": False,
            "scope": "Operator-assisted event request; no synchronized fusion or 3D reconstruction."}


def analyze_request(request_path, video_root, output_dir, base_config=None):
    """Validate, analyse, and persist an explicit one-/two-camera request.

    ``request_path`` may be a JSON filename or an already decoded request dict.
    Validation errors raise before processing; per-view runtime failures are
    retained in request-result.json; a valid responsible peer can still decide.
    """
    from .pipeline import analyze_video, write_json

    payload = (deepcopy(request_path) if isinstance(request_path, dict) else
               json.loads(Path(request_path).read_text(encoding="utf-8-sig")))
    request = validate_request(payload, video_root, base_config)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "request.json", request)
    rows = []
    for view in request["views"]:
        token = "camera_" + hashlib.sha256(view["camera_id"].encode()).hexdigest()[:16]
        config = deepcopy(base_config or {})
        identity = dict(config.get("identity") or {})
        identity.update(hit_anchors=view["hit_anchors"], event_window_seconds=view["analysis_event_window_seconds"])
        events = dict(config.get("events") or {})
        events.pop("target_track_id", None)  # A prior video's track ID is not portable.
        config.update(rules=request["rules"], identity=identity, events=events, video_id=token,
                      request_context={"request_id": request["request_id"], "camera_id": view["camera_id"],
                                       "responsible_boundaries": view["responsible_boundaries"]})
        row = {key: deepcopy(value) for key, value in view.items() if key != "source_path"}
        row["result_path"] = f"views/{token}/result.json"
        try:
            result = analyze_video(view["source_path"], output / "views" / token, config)
            row.update(status="completed", source_sha256=result.get("source_sha256"),
                       decision=result["decision"], calibration_status=result["calibration"]["status"],
                       landing_count=len(result.get("landings", [])))
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}",
                       result_path=None,
                       decision={"decision": "REVIEW", "reason_codes": ["VIEW_ANALYSIS_FAILED"]})
        rows.append(row)
    result = {"schema_version": 1, "request_id": request["request_id"], "rules": request["rules"],
              "status": "completed" if all(r["status"] == "completed" for r in rows) else "completed_with_errors",
              "same_event_confirmed": request["same_event_confirmed"],
              "same_court_coordinate_system_confirmed": request["same_court_coordinate_system_confirmed"],
              "event_correspondence_source": "operator_request" if request["same_event_confirmed"] else None,
              "views": rows, "decision": aggregate_request(request, rows),
              "ground_truth_created": False, "automatic_benchmark_eligible": False,
              "artifacts": {"summary": "request-summary.html", "result": "request-result.json"}}
    write_request_summary(result, output)
    write_json(output / "request-result.json", result)
    return result


_REASONS_KO = {
    "REQUESTED_BOUNDARY_WITHOUT_RESPONSIBLE_CAMERA": "요청한 라인을 담당하는 카메라가 없습니다.",
    "CROSS_VIEW_EVENT_CORRESPONDENCE_UNCONFIRMED": "두 영상의 지정 구간이 같은 착지인지 확인되지 않았습니다.",
    "CROSS_VIEW_COURT_COORDINATE_SYSTEM_UNCONFIRMED": "두 카메라의 코트 좌표축과 담당 라인이 같은 기준인지 확인되지 않았습니다.",
    "DUPLICATE_SOURCE_IS_NOT_AN_INDEPENDENT_VIEW": "같은 원본 영상이 반복되어 독립된 시점으로 사용할 수 없습니다.",
    "CONFLICTING_VIEW_DECISIONS": "카메라별 IN·OUT 결과가 서로 다릅니다.",
    "RESPONSIBLE_VIEW_UNRESOLVED": "담당 카메라의 판정 근거가 부족하거나 영상 처리가 실패했습니다.",
    "CONTACT_OUTSIDE_REQUESTED_EVENT_WINDOW": "접촉 시점이 요청한 영상 구간 안에서 확인되지 않았습니다.",
    "VIEW_CONTACT_GEOMETRY_UNAVAILABLE": "확인된 접촉 영역과 위치 오차가 없습니다.",
    "CROSS_VIEW_CONTACT_POSITION_DISAGREEMENT": "두 영상의 접촉 위치가 추정 오차 범위 안에서 일치하지 않습니다.",
    "OUT_NOT_SUPPORTED_BY_ASSIGNED_BOUNDARY": "OUT 근거가 요청한 담당 라인에서 확인되지 않았습니다.",
    "NO_UNIQUE_RESOLVED_REQUEST_DECISION": "요청한 사건에 대해 하나의 확정 가능한 결과가 없습니다.",
    "RESPONSIBLE_VIEWS_AGREE": "담당 카메라의 판정 근거가 일치합니다.",
    "VERDICT_FROM_QUALIFIED_VIEW": "유효한 한 시점의 판정을 최종 결과에 반영했습니다.",
    "IN_NOT_SUPPORTED_BY_ASSIGNED_BOUNDARIES": "IN 판정에 필요한 요청 경계 전체를 확인할 수 없습니다.",
    "VIEW_SOURCE_UNVERIFIED": "판정 근거 영상의 원본 식별값을 확인할 수 없습니다.",
    "VIEW_RULES_MISMATCH": "시점의 판정 규칙이 요청한 단식·복식 규칙과 다릅니다.",
}
_BOUNDARIES_KO = {"left": "왼쪽", "right": "오른쪽", "near": "가까운 쪽", "far": "먼 쪽", "all": "전체"}


def _asset_url(value, folder, output):
    """Only embed an existing artifact within this particular view folder."""
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    asset = (folder / relative).resolve()
    if not asset.is_relative_to(folder) or not asset.is_file() or not asset.is_relative_to(output):
        return None
    return quote(asset.relative_to(output).as_posix(), safe="/")


def _nested_assets(value, folder, output):
    if isinstance(value, dict):
        for item in value.values():
            yield from _nested_assets(item, folder, output)
    elif isinstance(value, list):
        for item in value:
            yield from _nested_assets(item, folder, output)
    elif isinstance(value, str) and Path(value).suffix.lower() in {".jpg", ".jpeg", ".png", ".mp4", ".json"}:
        url = _asset_url(value, folder, output)
        if url:
            yield value, url


def write_request_summary(result, output_dir):
    """Write a local, server-free HTML view with links to real evidence assets."""
    output = Path(output_dir).resolve()
    escape = lambda value: html.escape(str(value), quote=True)
    decision = result["decision"]
    reasons = "".join(f"<li>{escape(_REASONS_KO.get(reason, reason))}</li>" for reason in decision["reason_codes"])
    warnings = "".join(f"<li>{escape(_REASONS_KO.get(reason, reason))}</li>" for reason in decision.get("warning_codes", []))
    warnings_html = f'<details><summary>다른 시점·연결 상태</summary><ul>{warnings}</ul></details>' if warnings else ''
    source_cameras = ", ".join(decision.get("source_camera_ids", []))
    source_detail = (f'최종 판정 근거: {escape(source_cameras)} · 다른 시점의 개별 결과는 그대로 유지됩니다.'
                     if source_cameras else '최종 판정에 사용할 유효한 시점이 없습니다.')
    views_html = []
    for row in result["views"]:
        responsibility = ", ".join(_BOUNDARIES_KO[b] for b in row["responsible_boundaries"])
        start, end = row["event_window_seconds"]
        label = row["decision"].get("decision", "REVIEW")
        content = [f'<article><h2>카메라 {escape(row["camera_id"])} <span class="badge {escape(label)}">{escape(label)}</span></h2>',
                   f'<p>{escape(row["video"])}</p><p>담당 라인: {escape(responsibility)} · 요청 구간: {start:.3f}–{end:.3f}초</p>']
        observed = row.get("analysis_event_window_seconds")
        if observed is not None and list(observed) != [start, end]:
            content.append(f'<p class="note">실제 표시 프레임의 분석 구간: {observed[0]:.3f}–{observed[1]:.3f}초</p>')
        if row.get("status") != "completed":
            content.append(f'<p class="error">분석 실패: {escape(row.get("error", "영상 결과를 확인할 수 없습니다."))}</p>')
        else:
            result_path = row.get("result_path")
            path = (output / str(result_path)).resolve()
            if path.is_relative_to(output) and path.is_file():
                view_result = json.loads(path.read_text(encoding="utf-8"))
                folder = path.parent
                artifacts = view_result.get("artifacts", {})
                annotated = artifacts.get("annotated_clip", {})
                annotated_path = annotated.get("path") if isinstance(annotated, dict) else annotated
                video_links = []
                for title, asset in (("궤적·착지 후보 근거 영상", annotated_path), ("검수용 원본 미리보기", artifacts.get("preview", "preview.mp4"))):
                    url = _asset_url(asset, folder, output)
                    if url:
                        video_links.append(f'<figure><video controls preload="metadata" src="{url}"></video><figcaption>{title}</figcaption></figure>')
                content.extend(video_links)
                images, other_links = [], []
                seen = set()
                evidence = {"overview": artifacts.get("overview"), "evidence": artifacts.get("evidence", []),
                            "native_contact": [landing.get("source_contact_evidence", {}) for landing in view_result.get("landings", [])]}
                for name, url in _nested_assets(evidence, folder, output):
                    if url in seen:
                        continue
                    seen.add(url)
                    if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        images.append(f'<figure><a href="{url}"><img loading="lazy" src="{url}" alt="{escape(name)}"></a><figcaption>{escape(name)}</figcaption></figure>')
                    else:
                        other_links.append(f'<li><a href="{url}">{escape(name)}</a></li>')
                if images:
                    content.append('<details open><summary>착지 후보·원본 해상도 근거 이미지</summary><div class="evidence">' + ''.join(images) + '</div></details>')
                if other_links:
                    content.append('<ul>' + ''.join(other_links) + '</ul>')
                result_url = quote(path.relative_to(output).as_posix(), safe="/")
                content.append(f'<p><a href="{result_url}">이 카메라의 상세 분석 JSON</a></p>')
            else:
                content.append('<p class="error">상세 분석 파일을 찾을 수 없습니다.</p>')
        codes = "\n".join(row["decision"].get("reason_codes", []))
        if codes:
            content.append(f'<details><summary>카메라 판정 사유 코드</summary><pre>{escape(codes)}</pre></details>')
        content.append('</article>')
        views_html.append(''.join(content))
    correspondence = "사용자가 같은 착지 사건으로 지정함" if result["same_event_confirmed"] else "같은 착지 사건인지 미확인"
    orientation = ("코트 좌표축·담당 라인이 동일 기준임을 사용자가 확인함"
                   if result.get("same_court_coordinate_system_confirmed") else "카메라 간 코트 좌표축·담당 라인 기준 미확인")
    document = f'''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>판정 요청 {escape(result["request_id"])}</title>
<style>body{{margin:0;background:#f3f5f7;color:#182535;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:24px}}h1,h2{{line-height:1.3}}header,article{{background:white;padding:24px;margin-bottom:20px;border-radius:12px;border:1px solid #dce2e9}}.badge{{display:inline-block;padding:4px 12px;border-radius:6px;background:#fff2c4;color:#805800}}.IN{{background:#d7f2e4;color:#126141}}.OUT{{background:#fbe0dc;color:#9e2822}}.views{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}}figure{{margin:16px 0}}video,img{{max-width:100%;width:100%;height:auto;border-radius:6px;background:#111}}figcaption{{font-size:13px;overflow-wrap:anywhere;color:#5b6674}}a{{color:#145fab}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}.error{{color:#a32620}}details{{margin-top:16px}}summary{{cursor:pointer}}p{{overflow-wrap:anywhere}}.note{{font-size:14px;color:#526070}}@media(max-width:600px){{main{{padding:12px}}header,article{{padding:16px}}}}</style>
<main><header><p>배드민턴 · 판정 요청 검토</p><h1>{escape(result["request_id"])}</h1>
<p><span class="badge {escape(decision["decision"])}">{escape(decision["decision"])}</span> · 요청 라인: {escape(_BOUNDARIES_KO[decision["requested_boundary"]])}</p>
<ul>{reasons}</ul><p>{source_detail}</p>{warnings_html}<p>시점 연결: {correspondence}</p><p>좌표 방향: {orientation}</p>
<p class="note">영상별 시간은 각 영상의 첫 표시 프레임부터의 시간입니다. 카메라 좌표를 합쳐 더 정밀한 위치로 보고하지 않습니다. 요청 구간·타격 표시를 사용한 보조 분석이며, 정답 라벨이나 독립 정확도 검증 결과가 아닙니다.</p>
<p><a href="request-result.json">요청 결과 JSON</a> · <a href="request.json">요청 설정</a></p></header>
<section class="views">{''.join(views_html)}</section></main></html>'''
    path = output / "request-summary.html"
    path.write_text(document, encoding="utf-8")
    return path
