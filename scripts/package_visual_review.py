"""Build a portable AI-assisted review package after the 74-video report is frozen.

Only the standard library is required. Original recordings and old analysis runs
are deliberately excluded; copied report assets are the portable viewing data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile


VIDEO_COUNT = 74
PREFIX = "badminton-reviewed-poc"
SOURCE_DIRS = (
    "badminton", "configs", "docs", "examples", "scripts", "tests",
    "annotations/ai-visual", "annotations/ai-training-v1", "annotations/ai-training-v2",
)
OPTIONAL_DIRS = ("runs/pair-validation", "runs/review-remediation/verification",
                 "runs/review-remediation/pipeline", "runs/review-remediation/browser",
                 "runs/review-remediation/baseline")
OPTIONAL_FILES = tuple(f"runs/review-remediation/{name}/inspection.json"
                       for name in ("boundary", "occlusion", "target", "uhd")) + (
    "runs/review-remediation/guided-evidence/fresh-validation.json",)
REQUIRED_FILES = (
    "README.md", "pyproject.toml", "start.py", "start-review-windows.bat", "start-pair-windows.bat",
    "models/ai-candidate-v1.pt", "models/ai-candidate-v1.pt.json",
    "models/ai-candidate-v2.pt", "models/ai-candidate-v2.pt.json",
    "runs/inspection/input-manifest-before.json",
)
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """Accept portable relative paths only, rejecting all symlink components."""
    posix = PurePosixPath(relative)
    if (not relative or "\\" in relative or posix.is_absolute() or posix.as_posix() != relative
            or any(part in {".", ".."} or ":" in part for part in relative.split("/"))):
        raise ValueError(f"Unsafe package path: {relative!r}")
    path = root
    for part in posix.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlinks are not package inputs: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Package input escapes root: {relative}")
    return path


def validate_report(root: Path, report_dir: Path, *, rehash_sources: bool = False) -> dict:
    report = read_json(report_dir / "visual-review.json")
    for name in ("video_count", "reviewed_count", "valid_review_count", "source_hash_verified_count"):
        if report.get(name) != VIDEO_COUNT:
            raise ValueError(f"Report {name} must be {VIDEO_COUNT}: {report.get(name)!r}")
    if report.get("status") != "completed" or report.get("review_source") != "ai_visual_review":
        raise ValueError("A completed AI visual review report is required")
    if (report.get("ground_truth_eligible") is not False
            or report.get("automatic_accuracy") is not None
            or report.get("independent_human_ground_truth_count") != 0):
        raise ValueError("AI review must remain distinct from independent human ground truth")
    videos = report.get("videos", [])
    if len(videos) != VIDEO_COUNT:
        raise ValueError(f"Expected {VIDEO_COUNT} video records")
    identifiers = [video.get("video_id") for video in videos]
    if (any(not isinstance(identifier, str) or not re.fullmatch(r"v_[0-9a-f]+", identifier)
            for identifier in identifiers) or len(set(identifiers)) != VIDEO_COUNT):
        raise ValueError("Video identifiers must be valid and unique")
    annotations = root / "annotations/ai-visual"
    if {path.stem for path in annotations.glob("v_*.json")} != set(identifiers):
        raise ValueError("Report and AI visual annotation video sets differ")
    for video in videos:
        identifier = video["video_id"]
        review = video.get("review", {})
        integrity = video.get("source_integrity", {})
        annotation = video.get("annotation", {})
        source_hash = video.get("source_sha256", "")
        if review.get("valid") is not True or review.get("errors"):
            raise ValueError(f"Invalid review: {identifier}")
        if (not isinstance(source_hash, str) or not SHA256.fullmatch(source_hash)
                or integrity.get("verified") is not True
                or integrity.get("sha256") != source_hash
                or annotation.get("source_sha256") != source_hash):
            raise ValueError(f"Unverified or mismatched source SHA256: {identifier}")
        if (annotation.get("video_id") != identifier
                or annotation.get("source") != "ai_visual_review"
                or annotation.get("status") != "reviewed_by_ai"
                or annotation.get("human_verified") is not False):
            raise ValueError(f"Incorrect annotation provenance: {identifier}")
        annotation_path = safe_path(root, f"annotations/ai-visual/{identifier}.json")
        if read_json(annotation_path) != annotation:
            raise ValueError(f"Report is stale relative to annotation: {identifier}; rebuild it first")
        for key in ("automatic_decision", "assisted_decision", "direct_assisted_decision"):
            if video.get(key) not in {"IN", "OUT", "REVIEW"}:
                raise ValueError(f"Invalid {key}: {identifier}")
        assets = video.get("assets", {})
        paths = [assets.get("preview")]
        evidence = assets.get("evidence", [])
        if not evidence:
            raise ValueError(f"Missing visual evidence: {identifier}")
        paths.extend(item.get("path") for item in evidence)
        for relative in paths:
            if not isinstance(relative, str) or not relative.startswith("assets/"):
                raise ValueError(f"Missing portable asset path: {identifier}")
            asset = safe_path(report_dir, relative)
            if not asset.is_file() or asset.stat().st_size == 0:
                raise ValueError(f"Missing or empty report asset: {asset}")
        if rehash_sources:
            source = safe_path(root, integrity.get("path", ""))
            if not source.is_file() or digest_file(source) != source_hash:
                raise ValueError(f"Original source SHA256 no longer matches: {identifier}")
    for counter_key, decision_key in (
        ("assisted_counts", "assisted_decision"),
        ("automatic_counts", "automatic_decision"),
        ("direct_assisted_counts", "direct_assisted_decision"),
    ):
        actual = Counter(video[decision_key] for video in videos)
        expected = report.get(counter_key, {})
        if any(actual[label] != expected.get(label, 0) for label in {"IN", "OUT", "REVIEW"}):
            raise ValueError(f"Report {counter_key} disagrees with its video records")
    if not (report_dir / "index.html").is_file():
        raise ValueError("Missing portable review index.html")
    return report


def collect_files(root: Path, report_relative: str) -> list[Path]:
    selected: dict[str, Path] = {}

    def add(path: Path):
        relative = path.relative_to(root).as_posix()
        safe_path(root, relative)
        if not path.is_file():
            raise ValueError(f"Required input is not a file: {relative}")
        key = relative.casefold()
        if key in selected:
            raise ValueError(f"Duplicate or case-insensitive package path: {relative}")
        if path.stat().st_size == 0 and relative.startswith("models/"):
            raise ValueError(f"Empty model input: {relative}")
        selected[key] = path

    for relative in REQUIRED_FILES:
        add(safe_path(root, relative))
    for relative in OPTIONAL_FILES:
        path = safe_path(root, relative)
        if path.exists():
            add(path)
    for version in ("v1", "v2"):
        read_json(root / f"models/ai-candidate-{version}.pt.json")
    pdfs = sorted(root.glob("*.pdf"))
    if not pdfs:
        raise ValueError("The original development-plan PDF is required")
    for pdf in pdfs:
        add(pdf)
    optional = tuple(relative for relative in OPTIONAL_DIRS if safe_path(root, relative).exists())
    for relative in (*SOURCE_DIRS, report_relative, *optional):
        directory = safe_path(root, relative)
        if not directory.is_dir():
            raise ValueError(f"Required directory is missing: {relative}")
        for parent, folders, names in os.walk(directory, followlinks=False):
            parent = Path(parent)
            for folder in folders:
                if (parent / folder).is_symlink():
                    raise ValueError(f"Symlink directory is not portable: {parent / folder}")
            folders[:] = sorted(folder for folder in folders
                                 if folder not in SKIP_DIRS and not folder.startswith("."))
            for name in sorted(names):
                if name.startswith(".") or name.endswith((".pyc", ".pyo")):
                    continue
                add(parent / name)
    return sorted(selected.values(), key=lambda path: path.relative_to(root).as_posix())


def reference_manifest(root: Path, report_relative: str, report: dict, files: list[Path]) -> dict:
    included = {path.relative_to(root).as_posix() for path in files}
    remapping: dict[str, set[str]] = {}
    unresolved = []
    for video in report["videos"]:
        for asset in video["assets"]["evidence"]:
            source = asset.get("source")
            if isinstance(source, str):
                remapping.setdefault(source, set()).add(f"{report_relative}/{asset['path']}")
        for kind, source in (("original_video", video["source_integrity"].get("path")),
                             ("original_analysis_result", video.get("original_result"))):
            if source and source not in included:
                unresolved.append({"video_id": video["video_id"], "kind": kind, "path": source})
    # Keep authoritative annotation bytes unchanged. Record original provenance
    # references and where their portable report copies can be found instead.
    for video in report["videos"]:
        annotation = video["annotation"]
        evidence = list(annotation.get("evidence_paths", []))
        match = annotation.get("cross_view_match", {})
        if isinstance(match, dict):
            evidence.extend(match.get("evidence_paths", []))
        for source in dict.fromkeys(evidence):
            if source not in included and source not in remapping:
                unresolved.append({"video_id": video["video_id"], "kind": "provenance_evidence", "path": source})
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report": f"{report_relative}/visual-review.json",
        "video_count": VIDEO_COUNT,
        "valid_review_count": report["valid_review_count"],
        "source_hash_verified_count": report["source_hash_verified_count"],
        "source_hash_check": "Recorded per-video report verification; use --rehash-sources to re-read originals at packaging time.",
        "assisted_counts": report["assisted_counts"],
        "automatic_counts": report["automatic_counts"],
        "ground_truth_eligible": False,
        "original_videos_included": False,
        "metadata_policy": "Authoritative JSON and source hashes are preserved. Report assets are relative and portable. Original source/evidence/model-training provenance paths may refer to excluded originals or the build workspace; use evidence_path_remapping for available copies. Reanalysis requires original videos and a new analysis run.",
        "evidence_path_remapping": {source: sorted(paths) for source, paths in sorted(remapping.items())},
        "excluded_or_unresolved_provenance_references": unresolved,
        "files": [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
                  for path in files],
    }


def start_here(report: dict, report_relative: str) -> str:
    counts = report["assisted_counts"]
    return f"""배드민턴 AI 시각 검수 통합 PoC — 먼저 읽어 주세요

새 A/B 입력·통합 판정 화면:
Windows: start-pair-windows.bat
macOS/Linux: python3 start.py --pair
A와 B를 함께 선택하거나 새 파일을 올린 뒤 하나의 최종 보조 판정을 확인합니다.
처음에는 기본 Python 의존성을 설치할 수 있으며, 새 영상 분석에는 FFmpeg/ffprobe가 필요합니다.
저장된 검수 활용과 새 자동 분석은 구분됩니다. 자세한 사용법은 docs/PAIR_WORKSPACE.md를 보세요.
0.5.0의 REVIEW 원인·집중 재분석 개선과 15쌍 재검수 결과는 docs/REVIEW_REMEDIATION.md에 있습니다.
원본 없는 이 패키지에서 집중 재분석을 하려면 해당 A/B 원본 파일을 업로드하세요.

1. ZIP을 폴더에 모두 압축 해제하고 Python 3.11 이상을 설치하세요.
2. Windows: start-review-windows.bat 파일을 더블클릭하세요.
   macOS/Linux: 이 폴더에서 python3 start.py --review 를 실행하세요.
   이미 생성된 검수 화면 열람에는 외부 Python 패키지, FFmpeg, 원본 영상이 필요하지 않습니다.
   실행 창을 닫으면 로컬 열람 서버가 종료됩니다.

74개 영상 모두 AI 직접 시각 검수 기록과 미리보기·근거가 포함되어 있습니다.
보고서: {report_relative}/index.html
AI 보조 결과: IN {counts.get('IN', 0)} / OUT {counts.get('OUT', 0)} / REVIEW {counts.get('REVIEW', 0)}
확정할 수 없는 영상은 REVIEW로 유지했습니다. 전부 IN/OUT으로 판정했다는 뜻이 아닙니다.
AI 검수 라벨과 같은 랠리의 다른 시점에서 전이한 보조 결과는 독립된 사람 정답이 아닙니다.
포함된 AI 학습 모델의 개발 지표도 실제 자동 라인 판정 정확도의 증거가 아닙니다.
자동 판정 결과와 AI 보조 결과를 화면과 JSON에서 구분해 확인하세요.

포함: 소스, 문서, 설정, 예제, 테스트, 스크립트, 원본 개발계획서 PDF,
AI 검수 라벨, v1/v2 학습 라벨·후보 분류 모델과 메타데이터, 최종 보고서 전체 및 assets.
A/B 실행 검증 기록이 있으면 runs/pair-validation/에 함께 포함됩니다.
미포함: 원본 video/ 영상과 과거의 대용량 분석 결과. 미리보기는 원본 영상의 대체 원본이 아닙니다.

보고서의 미리보기·근거 링크는 assets/ 상대 경로로 복사되어 함께 이동합니다.
원본 annotation/학습 JSON의 출처 경로는 무결성 보존을 위해 변경하지 않았습니다.
그 경로와 과거 result.json, 빌드 환경의 절대 경로는 이 패키지에서 열리지 않을 수 있습니다.
PACKAGE_MANIFEST.json의 evidence_path_remapping에서 포함된 근거 사본을 찾을 수 있습니다.
excluded_or_unresolved_provenance_references에는 포함하지 않은 주요 출처 경로를 기록했습니다.

원본 재분석·모델 실행/재학습: README.md와 docs/를 따르세요.
원본 영상을 video/에 별도로 준비하고 FFmpeg/ffprobe 및 프로젝트 의존성을 설치해야 합니다.
학습 모델 실행에는 learning 선택 의존성(PyTorch 등)이 필요합니다.
예: python -m pip install -e ".[test,learning]"
AI 시각 검수 화면은 저장된 결과 열람용이며 새 영상 자동 추론을 실행하지 않습니다.

ZIP 옆 .sha256 파일은 전체 압축파일의 SHA256입니다.
PACKAGE_MANIFEST.json에 패키지 검증 범위와 파일 목록을 기록했습니다.
"""


def snapshot(files: list[Path]) -> dict[Path, tuple[int, int, int]]:
    return {path: (stat.st_size, stat.st_mtime_ns, stat.st_ino)
            for path in files for stat in [path.stat()]}


def package(root: Path, report_relative: str, output: Path, *, rehash_sources: bool = False,
            check_only: bool = False) -> dict:
    report_dir = safe_path(root, report_relative)
    report = validate_report(root, report_dir, rehash_sources=rehash_sources)
    files = collect_files(root, report_relative)
    manifest = reference_manifest(root, report_relative, report, files)
    if rehash_sources:
        manifest["source_hash_check"] = "All 74 original videos were re-read and SHA256 verified at packaging time."
    summary = {"video_count": VIDEO_COUNT, "valid_review_count": VIDEO_COUNT,
               "source_hash_verified_count": VIDEO_COUNT,
               "assisted_counts": report["assisted_counts"],
               "original_videos_included": False, "input_files": len(files)}
    if check_only:
        return {"status": "validated_without_packaging", **summary}
    if output in files or output.is_symlink():
        raise ValueError("The output must not replace a package input or symlink")
    if any(output.is_relative_to(root / relative) for relative in (*SOURCE_DIRS, report_relative, *OPTIONAL_DIRS)):
        raise ValueError("Place the output outside the selected source and report directories")
    for parent in output.parents:
        if parent.is_symlink():
            raise ValueError(f"Output parent is a symlink: {parent}")
    output.parent.mkdir(parents=True, exist_ok=True)
    before = snapshot(files)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".review-package-", suffix=".zip", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                safe_path(root, relative)
                archive.write(path, f"{PREFIX}/{relative}")
            archive.writestr(f"{PREFIX}/START_HERE.txt", start_here(report, report_relative))
            archive.writestr(f"{PREFIX}/PACKAGE_MANIFEST.json",
                             json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if (snapshot(files) != before or collect_files(root, report_relative) != files
                or validate_report(root, report_dir) != report):
            raise ValueError("Inputs changed while packaging; freeze and rebuild the report before retrying")
        with zipfile.ZipFile(temporary) as archive:
            names = archive.namelist()
            if len(names) != len({name.casefold() for name in names}):
                raise ValueError("Duplicate ZIP entries")
            corrupt = archive.testzip()
            if corrupt:
                raise ValueError(f"ZIP integrity check failed: {corrupt}")
        digest = digest_file(temporary)
        os.replace(temporary, output)
        checksum = output.with_suffix(".sha256")
        checksum.write_text(f"{digest}  {output.name}\n", encoding="ascii")
        return {"status": "packaged", "archive": str(output), "bytes": output.stat().st_size,
                "sha256": digest, "checksum_file": str(checksum), **summary}
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", default="runs/ai-reviewed-poc", help="Project-relative frozen report directory")
    parser.add_argument("--output", default="dist/badminton-reviewed-poc.zip", help="Project-relative output ZIP")
    parser.add_argument("--rehash-sources", action="store_true", help="Additionally re-read and hash all original videos")
    parser.add_argument("--check-only", action="store_true", help="Validate frozen inputs without creating an archive")
    args = parser.parse_args()
    try:
        root = args.project_root.resolve(strict=True)
        result = package(root, args.report, safe_path(root, args.output),
                         rehash_sources=args.rehash_sources, check_only=args.check_only)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
