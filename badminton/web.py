"""Local evidence review server. Human labels are stored separately from inference."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path, PureWindowsPath
import re
import threading
from urllib.parse import unquote, urlsplit
import uuid
from copy import deepcopy


class RequestError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


def _safe_path(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or ".." in Path(relative).parts:
        raise RequestError(403, "허용되지 않는 파일 경로입니다.")
    return candidate


def _read_json(path, default=None):
    if not path.is_file():
        if default is not None:
            return default
        raise RequestError(404, "아직 준비되지 않은 결과입니다.")
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, root, video_dir=None):
        self.root = Path(root).resolve()
        self.video_dir = Path(video_dir if video_dir is not None else "video").resolve()
        self.static = Path(__file__).with_name("static")
        self.annotation_lock = threading.Lock()
        self.jobs = {}
        self.job_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="review-analysis")
        self.pair_workspace = None
        self.pair_workspace_lock = threading.Lock()
        self.resource_lock = threading.RLock()
        self.resource_environment = None
        self.resource_settings = None
        self.resources_confirmed = False
        super().__init__(address, ReviewHandler)

    def _load_resources(self):
        if self.resource_environment is not None:
            return
        from .resources import detect_environment, validate_settings
        self.resource_environment = detect_environment()
        self.resource_settings = deepcopy(self.resource_environment["defaults"])
        try:
            saved = _read_json(self.root / "resource-settings.json", {})
            if saved.get("fingerprint") == self.resource_environment["fingerprint"]:
                self.resource_settings = validate_settings(saved["settings"], self.resource_environment)
        except (ValueError, KeyError, TypeError, AttributeError, OSError):
            pass

    def resources_busy(self):
        with self.job_lock:
            if any(job["status"] in {"queued", "running"} for job in self.jobs.values()):
                return True
        workspace = self.pair_workspace
        if workspace is not None and hasattr(workspace, "lock"):
            with workspace.lock:
                return any(job["status"] in {"queued", "running"} for job in workspace.jobs.values())
        return False

    def resources_status(self):
        with self.resource_lock:
            self._load_resources()
            return {"environment": deepcopy(self.resource_environment),
                    "settings": deepcopy(self.resource_settings),
                    "confirmed": self.resources_confirmed, "busy": self.resources_busy()}

    def configure_resources(self, body):
        from .resources import validate_settings, apply_runtime, ResourceSettingsError
        from .pair_workspace import _write
        with self.resource_lock:
            self._load_resources()
            if self.resources_busy():
                raise RequestError(409, "진행 중이거나 대기 중인 분석이 끝난 뒤 장치 설정을 변경하세요.")
            try:
                selected = validate_settings(body, self.resource_environment)
                apply_runtime(selected)
            except ResourceSettingsError as error:
                raise RequestError(400, str(error)) from error
            try:
                _write(self.root / "resource-settings.json", {
                    "fingerprint": self.resource_environment["fingerprint"], "settings": selected})
            except OSError as error:
                apply_runtime(self.resource_settings)
                raise RequestError(500, "장치 설정을 저장하지 못했습니다. 저장 공간과 폴더 권한을 확인하세요.") from error
            self.resource_settings = selected
            self.resources_confirmed = True
            return self.resources_status()

    def require_resources(self):
        with self.resource_lock:
            if not self.resources_confirmed:
                raise RequestError(409, "분석 전에 실행 환경과 장치 사용 설정을 확인해 주세요.")
            return deepcopy(self.resource_settings)

    def server_close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.pair_workspace is not None:
            self.pair_workspace.close()
        super().server_close()

    def get_pair_workspace(self):
        """Keep paired jobs and uploads separate from the historical analysis."""
        with self.pair_workspace_lock:
            if self.pair_workspace is None:
                from .pair_workspace import PairWorkspace
                project = Path(__file__).resolve().parents[1]
                library = self.root if (self.root / "index.json").is_file() else project / "runs/integrated-poc"
                self.pair_workspace = PairWorkspace(project_root=project,
                    workspace_root=self.root / "paired", library_root=library,
                    resource_settings_provider=self.require_resources)
            return self.pair_workspace

    def video(self, video_id):
        if not re.fullmatch(r"[\w.-]{1,140}", video_id) or video_id in (".", ".."):
            raise RequestError(400, "잘못된 영상 ID입니다.")
        index = _read_json(self.root / "index.json", {"videos": []})
        for row in index.get("videos", []):
            if str(row.get("id", row.get("video_id"))) == video_id:
                return row
        raise RequestError(404, "영상이 목록에 없습니다.")

    def result(self, video_id):
        row = self.video(video_id)
        return _read_json(_safe_path(self.root, row.get("result_path", f"videos/{video_id}/result.json")))

    def resolve_original(self, row, result):
        """Locate the same source after a run is copied to another computer.

        A stored absolute path may remain usable. Otherwise a relative filename
        is looked up only inside the explicitly configured local video folder.
        Every accepted file must match the persisted original content hash.
        """
        from .media import file_hash

        recorded_hashes = [value for value in (row.get("sha256"), result.get("source_sha256"))
                           if value is not None]
        if (not recorded_hashes or any(not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value)
                                       for value in recorded_hashes)):
            raise RequestError(409, "원본 SHA256 기록이 없어 재분석할 영상을 확인할 수 없습니다. CLI에서 새로 분석하세요.")
        expected = recorded_hashes[0].lower()
        if any(value.lower() != expected for value in recorded_hashes):
            raise RequestError(409, "영상 목록과 분석 결과의 원본 SHA256이 다릅니다. 동일한 실행 결과인지 확인하세요.")
        name = row.get("file", result.get("file"))
        fallback = None
        if name is not None:
            if (not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name or
                    Path(name).is_absolute() or PureWindowsPath(name).drive or ".." in PureWindowsPath(name).parts):
                raise RequestError(403, "원본 파일 이름은 --video-dir 안의 상대 경로여야 합니다.")
            fallback = _safe_path(self.video_dir, name)
        candidates = []
        original = row.get("source_path", result.get("source_path"))
        if isinstance(original, str) and "\x00" not in original:
            path = Path(original)
            if path.is_absolute():
                candidates.append(path.resolve())
        if fallback is not None and fallback not in candidates:
            candidates.append(fallback)
        mismatched = False
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                digest = file_hash(candidate)
            except OSError:
                raise RequestError(403, "원본 영상을 읽을 수 없습니다. 파일 접근 권한을 확인하세요.")
            if digest == expected:
                return candidate, expected
            mismatched = True
        if mismatched:
            raise RequestError(409, "찾은 영상의 SHA256이 기존 분석의 원본과 다릅니다. 동일한 원본을 --video-dir 폴더에 넣으세요.")
        raise RequestError(404, "원본 영상을 찾을 수 없습니다. 원본 파일이 있는 폴더를 serve --video-dir로 지정하세요.")

    def annotate(self, video_id, body):
        self.video(video_id)
        result = self.result(video_id)
        label = body.get("label", "UNKNOWN")
        if label not in ("IN", "OUT", "UNKNOWN"):
            raise RequestError(400, "정답은 IN, OUT 또는 UNKNOWN이어야 합니다.")
        reviewer = str(body.get("reviewer", "")).strip()[:120]
        notes = str(body.get("notes", ""))[:8000]
        point, timestamp = body.get("contact_px"), body.get("time")
        analysis = result.get("analysis", {})
        metadata = result.get("metadata", {})
        width = analysis.get("width", metadata.get("width", 0))
        height = analysis.get("height", metadata.get("height", 0))
        if point is not None:
            if not isinstance(point, list) or len(point) != 2 or not all(_finite(v) for v in point):
                raise RequestError(400, "접촉 위치는 유한한 x, y 좌표여야 합니다.")
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise RequestError(400, "접촉 위치가 영상 범위를 벗어났습니다.")
        duration = metadata.get("duration", metadata.get("duration_seconds"))
        if timestamp is not None and (not _finite(timestamp) or timestamp < 0 or (_finite(duration) and timestamp > duration + 0.1)):
            raise RequestError(400, "접촉 시각이 영상 범위를 벗어났습니다.")
        contact = body.get("contact_observed") is True
        confirmed = body.get("confirmed") is True
        if confirmed and not reviewer:
            raise RequestError(400, "검수 완료를 기록하려면 검수자 이름을 입력하세요.")
        if confirmed and label != "UNKNOWN" and (not contact or point is None or timestamp is None):
            raise RequestError(400, "IN/OUT 정답 확정에는 실제 최초 접촉 확인, 시각, 위치가 필요합니다.")
        radius, uncertainty = body.get("contact_radius_px"), body.get("uncertainty_px")
        for value in (radius, uncertainty):
            if value is not None and (not _finite(value) or value < 0 or value > max(width, height)):
                raise RequestError(400, "접촉 반경과 위치 오차는 영상 범위 내 0 이상의 픽셀 값이어야 합니다.")
        distance = body.get("near_line_distance_m")
        if distance is not None and (not _finite(distance) or distance < 0):
            raise RequestError(400, "라인 거리는 0 이상의 미터 값이어야 합니다.")
        evidence = body.get("independent_evidence", "")
        if evidence is None:
            evidence = ""
        if not isinstance(evidence, str):
            raise RequestError(400, "독립 관측 근거는 설명 또는 파일 위치를 입력하세요.")
        evidence = evidence.strip()[:4000]
        detection_labels = body.get("detection_labels", [])
        if not isinstance(detection_labels, list) or len(detection_labels) > 2000:
            raise RequestError(400, "탐지 라벨은 최대 2,000개입니다.")
        frame_count = len(result.get("timestamps", []))
        clean_detection_labels = []
        for item in detection_labels:
            if not isinstance(item, dict) or item.get("label") not in ("shuttle", "background"):
                raise RequestError(400, "탐지 라벨 종류가 잘못되었습니다.")
            frame = item.get("frame_index")
            x, y = item.get("x"), item.get("y")
            if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 or (frame_count and frame >= frame_count):
                raise RequestError(400, "탐지 라벨의 프레임 번호가 잘못되었습니다.")
            if not _finite(x) or not _finite(y) or not (0 <= x < width and 0 <= y < height):
                raise RequestError(400, "탐지 라벨 좌표가 영상 범위를 벗어났습니다.")
            clean_detection_labels.append({"frame_index": frame, "x": x, "y": y, "label": item["label"]})
        detection_verified = body.get("detection_labels_verified") is True
        if detection_verified and (not reviewer or not clean_detection_labels):
            raise RequestError(400, "탐지 라벨 검수 완료에는 검수자와 최소 한 개의 라벨이 필요합니다.")
        sequences = body.get("contact_sequences", [])
        if not isinstance(sequences, list) or len(sequences) > 1000:
            raise RequestError(400, "착지 시계열 라벨은 최대 1,000개입니다.")
        clean_sequences = []
        for item in sequences:
            if not isinstance(item, dict) or item.get("label") not in ("airborne", "first_contact", "after_contact"):
                raise RequestError(400, "착지 시계열 라벨 종류가 잘못되었습니다.")
            frame, x, y = item.get("frame_index"), item.get("x"), item.get("y")
            if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 or (frame_count and frame >= frame_count):
                raise RequestError(400, "착지 시계열 라벨의 프레임 번호가 잘못되었습니다.")
            if not _finite(x) or not _finite(y) or not (0 <= x < width and 0 <= y < height):
                raise RequestError(400, "착지 시계열 라벨 좌표가 영상 범위를 벗어났습니다.")
            observed = item.get("contact_observed") is True
            if item["label"] == "first_contact" and not observed:
                raise RequestError(400, "최초 접촉 학습 라벨은 실제 접촉을 관측한 프레임에만 기록하세요.")
            contact_point = item.get("contact_px")
            if item["label"] == "first_contact":
                if (not isinstance(contact_point, list) or len(contact_point) != 2
                        or not all(_finite(v) for v in contact_point)
                        or not (0 <= contact_point[0] < width and 0 <= contact_point[1] < height)
                        or item.get("contact_point_verified") is not True):
                    raise RequestError(400, "최초 접촉 라벨에는 후보 중심과 별도로 직접 확인한 코르크 접촉점이 필요합니다.")
            independent = item.get("independent_evidence", "")
            if not isinstance(independent, str):
                raise RequestError(400, "착지 시계열 독립 관측 근거는 문자열이어야 합니다.")
            clean_sequences.append({"frame_index": frame, "x": x, "y": y, "label": item["label"],
                                    "contact_observed": observed, "independent_evidence": independent[:4000],
                                    "contact_px": contact_point if item["label"] == "first_contact" else None,
                                    "contact_point_verified": item["label"] == "first_contact"})
        sequences_verified = body.get("contact_sequences_verified") is True
        if sequences_verified and (not reviewer or not clean_sequences):
            raise RequestError(400, "착지 시계열 검수 완료에는 검수자와 최소 한 개의 라벨이 필요합니다.")
        verified = confirmed and label != "UNKNOWN"
        reason = None
        if confirmed and label == "UNKNOWN":
            reason = "ground_truth_unknown"
        if verified and distance is not None and distance <= 0.05 and not evidence:
            verified, reason = False, "near_line_independent_evidence_required"
        annotation = {
            "schema_version": 1, "video_id": video_id, "source": "human_review",
            "status": "verified" if verified else "needs_review", "reviewed": confirmed,
            "verification_reason": reason, "label": label, "reviewer": reviewer,
            "contact_observed": contact, "time": timestamp, "contact_px": point,
            "court_id": str(body.get("court_id", "")).strip()[:160] or None,
            "notes": notes, "independent_evidence": evidence,
            "target_identity_confirmed": body.get("target_identity_confirmed") is True,
            "contact_radius_px": radius, "uncertainty_px": uncertainty,
            "near_line_distance_m": distance,
            "detection_labels": clean_detection_labels, "detection_labels_verified": detection_verified,
            "contact_sequences": clean_sequences, "contact_sequences_verified": sequences_verified,
            "analysis_size": {"width": width, "height": height},
            "coordinate_space": "analysis_pixels", "analysis_width": width, "analysis_height": height,
            "source_sha256": result.get("source_sha256", metadata.get("sha256")),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_artifacts": {"result_path": self.video(video_id).get("result_path"),
                                 "video_sha256": metadata.get("sha256", result.get("sha256"))},
        }
        if verified:
            assisted = {"decision": "REVIEW", "reason_codes": ["MANUAL_CONTACT_EVIDENCE_INCOMPLETE"]}
            if body.get("target_identity_confirmed") is True and radius is not None and uncertainty is not None:
                from .events import apply_contact_evidence
                from .decision import decide
                candidate = result.get("selected_landing") or next(iter(result.get("landings", [])), {})
                human_evidence = {"source": "human_review", "verified": True, "video_id": video_id,
                                  "coordinate_system": "analysis_pixels", "time": timestamp,
                                  "contact_px": point, "first_contact_observed": contact,
                                  "target_identity_confirmed": True,
                                  "contact_radius_px": radius, "uncertainty_px": uncertainty}
                try:
                    landing = apply_contact_evidence(candidate, human_evidence, video_id=video_id,
                                                     duration_seconds=duration, image_size=(width, height),
                                                     timestamps=result.get("timestamps"))
                    assisted = decide(landing, result.get("calibration", {}), result.get("config", {}).get("rules"))
                    annotation["assisted_landing"] = landing
                except ValueError as error:
                    raise RequestError(400, "접촉 근거를 적용할 수 없습니다: " + str(error))
            assisted.update(source="human_review", evaluation_eligible=False)
            annotation["assisted_decision"] = assisted
        folder = self.root / "annotations"
        folder.mkdir(parents=True, exist_ok=True)
        with self.annotation_lock:
            target = folder / f"{video_id}.json"
            temporary = folder / f".{video_id}.{uuid.uuid4().hex}.tmp"
            temporary.write_text(json.dumps(annotation, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(target)
        return annotation


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "BadmintonReview/1.0"

    def log_message(self, format, *args):
        # Avoid logging annotation content, query strings, or filesystem paths.
        return

    def _headers(self, status, content_type, length=None, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        if length is not None:
            self.send_header("Content-Length", str(length))
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()

    def _json(self, value, status=200):
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self._headers(status, "application/json; charset=utf-8", len(encoded))
        if self.command != "HEAD":
            self.wfile.write(encoded)

    def _file(self, path):
        if not path.is_file():
            raise RequestError(404, "파일을 찾을 수 없습니다.")
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        extra = {"Accept-Ranges": "bytes"}
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            if not match or not any(match.groups()) or not size:
                self._headers(416, "application/octet-stream", 0, {"Content-Range": f"bytes */{size}"})
                return
            low, high = match.groups()
            if low:
                start = int(low)
                end = min(int(high), size - 1) if high else size - 1
            else:
                suffix = int(high)
                start, end = max(0, size - suffix), size - 1
                if suffix == 0:
                    start = size
            if start >= size or start > end:
                self._headers(416, "application/octet-stream", 0, {"Content-Range": f"bytes */{size}"})
                return
            status = 206
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = max(0, end - start + 1)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime in ("application/javascript", "application/json"):
            mime += "; charset=utf-8"
        self._headers(status, mime, length, extra)
        if self.command == "HEAD":
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _check_write_origin(self):
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc.lower() != self.headers.get("Host", "").lower():
            raise RequestError(403, "다른 사이트에서의 저장 요청은 허용되지 않습니다.")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise RequestError(403, "다른 사이트에서의 저장 요청은 허용되지 않습니다.")

    def _write_body(self):
        self._check_write_origin()
        if self.headers.get_content_type() != "application/json":
            raise RequestError(415, "JSON 요청만 허용됩니다.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise RequestError(400, "잘못된 요청 길이입니다.")
        if length < 1 or length > 256 * 1024:
            raise RequestError(413, "요청 크기는 256KB 이하여야 합니다.")
        try:
            body = json.loads(self.rfile.read(length), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, UnicodeError):
            raise RequestError(400, "올바른 JSON이 아닙니다.")
        if not isinstance(body, dict):
            raise RequestError(400, "JSON 객체가 필요합니다.")
        return body

    def _get(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/resources":
            return self._json(self.server.resources_status())
        if path in ("/resources.js", "/resources.css"):
            return self._file(self.server.static / path[1:])
        if path in ("/pair", "/pair/", "/pair.html"):
            return self._file(self.server.static / "pair.html")
        if path in ("/pair.js", "/pair.css"):
            return self._file(self.server.static / path[1:])
        if path == "/api/pair-library":
            return self._json(self.server.get_pair_workspace().library())
        if path == "/api/pair-jobs":
            return self._json(self.server.get_pair_workspace().list_jobs())
        if path.startswith("/api/pair-jobs/"):
            return self._json(self.server.get_pair_workspace().get_job(path.removeprefix("/api/pair-jobs/")))
        if path.startswith("/pair-media/"):
            return self._file(self.server.get_pair_workspace().asset_path(path.removeprefix("/pair-media/")))
        if path.startswith("/pair-review-assets/"):
            return self._file(self.server.get_pair_workspace().review_asset_path(path.removeprefix("/pair-review-assets/")))
        if path.startswith("/pair-library-assets/"):
            return self._file(self.server.get_pair_workspace().library_asset_path(path.removeprefix("/pair-library-assets/")))
        if path in ("/", "/index.html"):
            return self._file(self.server.static / "index.html")
        if path in ("/app.js", "/style.css"):
            return self._file(self.server.static / path[1:])
        if path == "/api/index":
            index = _read_json(self.server.root / "index.json", {"videos": [], "pairs": []})
            for i, pair in enumerate(index.get("pairs", [])):
                review = _read_json(self.server.root / "pair-annotations" / f"{i}.json", {})
                pair.update(review)
            index["review_counts"] = {"verified": 0, "needs_review": 0}
            for video in index.get("videos", []):
                video_id = str(video.get("id", video.get("video_id")))
                annotation = _read_json(self.server.root / "annotations" / f"{video_id}.json", {})
                status = annotation.get("status", "needs_review")
                if annotation.get("source_sha256") and video.get("sha256") and annotation["source_sha256"] != video["sha256"]:
                    status = "needs_review"
                video["annotation_status"] = status
                index["review_counts"]["verified" if status == "verified" else "needs_review"] += 1
            return self._json(index)
        if path.startswith("/api/video/"):
            return self._json(self.server.result(path.removeprefix("/api/video/")))
        if path.startswith("/api/annotations/"):
            video_id = path.removeprefix("/api/annotations/")
            self.server.video(video_id)
            saved = self.server.root / "annotations" / f"{video_id}.json"
            draft = self.server.root / "annotation-drafts" / f"{video_id}.json"
            annotation = _read_json(saved, _read_json(draft, {"video_id": video_id, "label": "UNKNOWN", "status": "needs_review", "source": "machine_draft", "contact_observed": False}))
            current_hash = self.server.result(video_id).get("source_sha256")
            if annotation.get("source_sha256") and current_hash and annotation["source_sha256"] != current_hash:
                annotation.update(status="needs_review", reviewed=False, detection_labels_verified=False,
                                  contact_sequences_verified=False, verification_reason="source_video_changed")
            return self._json(annotation)
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/")
            with self.server.job_lock:
                job = dict(self.server.jobs.get(job_id, {}))
            if not job:
                raise RequestError(404, "작업을 찾을 수 없습니다.")
            return self._json(job)
        if path.startswith("/media/"):
            return self._file(_safe_path(self.server.root, path.removeprefix("/media/")))
        raise RequestError(404, "경로를 찾을 수 없습니다.")

    def _post(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/pair-uploads":
            self._check_write_origin()
            if self.headers.get_content_type() != "application/octet-stream":
                raise RequestError(415, "영상 파일을 application/octet-stream 형식으로 전송하세요.")
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
                raise RequestError(400, "압축하지 않은 파일과 정확한 Content-Length가 필요합니다.")
            value = self.headers.get("Content-Length", "")
            if not re.fullmatch(r"[0-9]+", value):
                raise RequestError(400, "정확한 파일 크기(Content-Length)가 필요합니다.")
            length = int(value)
            if length < 1 or length > 1024 * 1024 * 1024:
                raise RequestError(413, "영상 한 개의 크기는 1바이트 이상 1GiB 이하여야 합니다.")
            filename = unquote(self.headers.get("X-Filename", ""))
            self.connection.settimeout(60)
            return self._json(self.server.get_pair_workspace().stage_upload(self.rfile, length, filename), 201)
        body = self._write_body()
        if path == "/api/resources":
            return self._json(self.server.configure_resources(body))
        if path == "/api/pair-jobs":
            with self.server.resource_lock:
                self.server.require_resources()
                with self.server.job_lock:
                    if any(job["status"] in {"queued", "running"} for job in self.server.jobs.values()):
                        raise RequestError(409, "개별 영상 분석이 끝난 뒤 A/B 분석을 시작하세요.")
                return self._json(self.server.get_pair_workspace().submit(body), 202)
        if path.startswith("/api/annotations/"):
            return self._json(self.server.annotate(path.removeprefix("/api/annotations/"), body))
        match = re.fullmatch(r"/api/pairs/(\d+)/sync", path)
        if match:
            pair_index = int(match[1])
            index = _read_json(self.server.root / "index.json")
            pairs = index.get("pairs", [])
            if pair_index >= len(pairs):
                raise RequestError(404, "시점 짝을 찾을 수 없습니다.")
            pair = pairs[pair_index]
            def find_id(camera):
                value = pair.get(f"{camera}_id", pair.get(f"video_{camera}_id"))
                if value is not None:
                    return str(value)
                name = pair.get(f"{camera}_file", pair.get(f"video_{camera}"))
                for video in index.get("videos", []):
                    if video.get("file") == name:
                        return str(video.get("id", video.get("video_id")))
                raise RequestError(400, "짝 영상 ID를 확인할 수 없습니다.")
            reviewer = body.get("reviewer")
            anchors = body.get("event_correspondences")
            if not isinstance(reviewer, str) or not reviewer.strip():
                raise RequestError(400, "동일 사건을 확인한 검수자 이름이 필요합니다.")
            if not isinstance(anchors, list) or len(anchors) > 50:
                raise RequestError(400, "시점 연결 앵커는 최대 50개입니다.")
            clean = []
            for anchor in anchors:
                if not isinstance(anchor, dict) or not _finite(anchor.get("time_a")) or not _finite(anchor.get("time_b")) or anchor.get("verified") is not True:
                    raise RequestError(400, "양쪽 영상 시각과 동일 사건 확인이 필요합니다.")
                clean.append({"time_a": anchor["time_a"], "time_b": anchor["time_b"], "verified": True})
            result_a, result_b = self.server.result(find_id("a")), self.server.result(find_id("b"))
            meta_a, meta_b = result_a.get("metadata", {}), result_b.get("metadata", {})
            from .sync import synchronize_event_anchors
            sync = synchronize_event_anchors(clean, duration_a=meta_a.get("duration", meta_a.get("duration_seconds")),
                                            duration_b=meta_b.get("duration", meta_b.get("duration_seconds")))
            record = {"source": "human_review", "reviewer": reviewer.strip()[:120],
                      "event_correspondences": clean, "sync": sync,
                      "automatic_sync": pair.get("automatic_sync", pair.get("sync", pair.get("synchronization"))),
                      "updated_at_utc": datetime.now(timezone.utc).isoformat()}
            folder = self.server.root / "pair-annotations"
            folder.mkdir(parents=True, exist_ok=True)
            with self.server.annotation_lock:
                target = folder / f"{pair_index}.json"
                temporary = folder / f".{pair_index}.{uuid.uuid4().hex}.tmp"
                temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                temporary.replace(target)
            return self._json(record)
        if path.startswith("/api/analyze/"):
            self.server.require_resources()
            video_id = path.removeprefix("/api/analyze/")
            row = self.server.video(video_id)
            existing = self.server.result(video_id)
            source, expected_source_hash = self.server.resolve_original(row, existing)
            configuration = dict(existing.get("config") or {})
            configuration["video_id"] = video_id
            with self.server.resource_lock:
                if self.server.resources_busy():
                    raise RequestError(409, "분석이 이미 진행 중입니다.")
                configuration["resource_settings"] = self.server.require_resources()
                job_id = uuid.uuid4().hex
                with self.server.job_lock:
                    self.server.jobs[job_id] = {"id": job_id, "video_id": video_id, "status": "queued",
                                                "resource_settings": configuration["resource_settings"]}
            def work():
                with self.server.job_lock:
                    self.server.jobs[job_id]["status"] = "running"
                try:
                    from .pipeline import analyze_video
                    from .media import file_hash
                    if file_hash(source) != expected_source_hash:
                        raise ValueError("Original video changed after the reanalysis request")
                    analyze_video(source, self.server.root / "videos" / video_id, config=configuration)
                    with self.server.job_lock:
                        self.server.jobs[job_id]["status"] = "completed"
                except Exception as error:
                    from .resources import ResourceSettingsError
                    with self.server.job_lock:
                        message = str(error) if isinstance(error, ResourceSettingsError) else f"분석 실패: {type(error).__name__}"
                        self.server.jobs[job_id].update(status="failed", error=message)
            self.server.executor.submit(work)
            return self._json({"job_id": job_id, "status": "queued"}, 202)
        raise RequestError(404, "경로를 찾을 수 없습니다.")

    def _dispatch(self, callback):
        from .pair_workspace import PairWorkspaceError
        try:
            hostname = urlsplit("http://" + self.headers.get("Host", "")).hostname
            allowed_hosts = {"127.0.0.1", "localhost", "::1", self.server.server_address[0]}
            if hostname not in allowed_hosts:
                raise RequestError(403, "로컬 서버 주소로 접속하세요.")
            callback()
        except RequestError as error:
            self._json({"error": error.message}, error.status)
        except PairWorkspaceError as error:
            self._json({"error": error.message}, error.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._json({"error": "요청 처리 중 오류가 발생했습니다."}, 500)

    def do_GET(self):
        self._dispatch(self._get)

    def do_HEAD(self):
        self._dispatch(self._get)

    def do_POST(self):
        self._dispatch(self._post)


def create_server(root, host="127.0.0.1", port=8765, video_dir=None):
    return ReviewServer((host, port), root, video_dir=video_dir)


def serve(root, host="127.0.0.1", port=8765, video_dir=None):
    server = create_server(root, host, port, video_dir=video_dir)
    print(f"Review UI: http://{host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
