#!/usr/bin/env python3
"""Render a Korean HTML report from completed video validation evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import html
import json
from pathlib import Path
import re
import statistics
import subprocess
from zoneinfo import ZoneInfo


def escape(value):
    return html.escape(str(value), quote=True)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f"<th>{escape(h)}</th>" for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f"<td>{escape(cell)}</td>" for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def family(filename):
    camera = re.search(r"([ab])(?:-\d+)?시점", filename)
    camera = camera.group(1).upper() if camera else "미상"
    label = ("8k 30 4k 24" if filename.startswith("8k") else
             "uhd30 4k30" if "uhd30 4k30" in filename else
             "uhd30 hd30" if "uhd30 hd30" in filename else "기본")
    return label, camera


def provenance(run_dir, rows, commit):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit or ""):
        raise ValueError("timing_audit.source_commit must be a full Git SHA-1")
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=run_dir, text=True).strip())
    listing = subprocess.check_output(["git", "ls-tree", "-rz", commit, "--", "video"], cwd=repo)
    expected = {}
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, blob_hash = metadata.decode("ascii").split()
        path = name.decode("utf-8")
        if kind == "blob" and path.lower().endswith(".mp4"):
            expected[path.removeprefix("video/")] = blob_hash
    observed = {row["file"]: row.get("git_blob_sha1") for row in rows}
    repeated = [name for name, count in Counter(row["file"] for row in rows).items() if count > 1]
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    mismatches = [{"file": name, "expected_git_blob_sha1": expected[name], "observed_git_blob_sha1": observed[name]}
                  for name in sorted(set(expected) & set(observed)) if expected[name] != observed[name]]
    return {"source_commit": commit,
            "method": "Compare every validated file's recorded Git blob SHA-1 with git ls-tree at the pinned source commit; no remote update or publish",
            "expected_github_video_files": len(expected), "validated_file_rows": len(rows),
            "matching_files": sum(expected[name] == observed[name] for name in set(expected) & set(observed)),
            "missing_files": missing, "extra_files": extra, "hash_mismatches": mismatches,
            "duplicate_result_rows": repeated,
            "passed": bool(expected) and not missing and not extra and not mismatches and not repeated}


def build_result(run_dir):
    summary = load(run_dir / "final" / "summary.json")
    timing = load(run_dir / "timing_audit.json")
    visual = load(run_dir / "visual" / "frame_summary.json")
    rows = summary["files"]
    proof = provenance(run_dir, rows, timing["source_commit"])
    timestamps = {row["file"]: row for row in timing["files"]}
    failed = [row["file"] for row in rows
              if row.get("probe_status") != "passed" or row.get("decode_status") != "passed"
              or row.get("error") or row.get("presented_frame_count_matches") is not True]
    hashes = defaultdict(list)
    groups = defaultdict(list)
    for row in rows:
        if row.get("sha256"):
            hashes[row["sha256"]].append(row["file"])
        groups[family(row["file"])].append(row)
    duplicate_groups = [{"sha256": key, "files": names} for key, names in sorted(hashes.items()) if len(names) > 1]
    capture_groups = []
    for (label, camera), members in sorted(groups.items()):
        intervals = [timestamps[r["file"]].get("presentable_interval_median_ms") for r in members if r["file"] in timestamps]
        intervals = [value for value in intervals if value]
        median_ms = statistics.median(intervals) if intervals else None
        capture_groups.append({"filename_family": label, "view": camera, "files": len(members),
                               "resolutions": sorted({f"{r['width']}×{r['height']}" for r in members}),
                               "median_presentable_frame_interval_ms": median_ms,
                               "cadence_fps_from_median_interval": round(1000 / median_ms, 3) if median_ms else None})
    gaps = [{"file": row["file"], **gap} for row in timing["files"] for gap in row.get("presentable_intervals_over_1_5x_median", [])]
    pairs = timing.get("candidate_pairs", [])
    unaccounted = [row["file"] for row in rows if row["file"] not in timestamps]
    input_passed = bool(rows) and not failed and proof["passed"] and not unaccounted
    samples_passed = visual.get("successful_samples") == visual.get("expected_samples") and not visual.get("failed_samples")
    initial_attempt = None
    initial_path = run_dir / "integrity" / "summary.json"
    if initial_path.exists():
        initial = load(initial_path)
        initial_failed = [r for r in initial["files"] if r.get("decode_status") != "passed"]
        initial_timestamp_errors = [r for r in initial_failed if "non monotonically increasing dts to muxer" in (r.get("error") or "")]
        final_by_name = {r["file"]: r for r in rows}
        corrected = [r["file"] for r in initial_timestamp_errors
                     if final_by_name.get(r["file"], {}).get("decode_status") == "passed"
                     and "-enc_time_base" in final_by_name.get(r["file"], {}).get("decode_command", [])]
        initial_attempt = {"summary": "integrity/summary.json", "initial_failed_decode_files": len(initial_failed),
                           "initial_output_timestamp_error_files": len(initial_timestamp_errors),
                           "corrected_timebase_rerun_passed_files": len(corrected),
                           "corrected_timebase_rerun_files": corrected,
                           "note": "Initial null-muxer timestamp failures are retained. Affected videos were rerun preserving the demuxer time base; final decoder status comes from the corrected execution."}
    result = {
        "scope": "실제 영상 입력·전처리 검증. AI 라인 판정 정확도, 코트 보정 성공률, 셔틀콕 추적 성능 또는 카메라 동기화 검증이 아님.",
        "generated_at_seoul": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "source_commit": timing["source_commit"],
        "input_integrity_passed": input_passed,
        "sample_preprocessing_passed": samples_passed,
        "ai_accuracy_evaluated": False,
        "provenance": proof,
        "file_count": len(rows), "unique_content_count": len(hashes),
        "total_bytes": sum(r.get("bytes", 0) for r in rows),
        "total_decoded_frames": sum(r.get("decoded_frames") or 0 for r in rows),
        "decode_passed_files": sum(r.get("decode_status") == "passed" for r in rows),
        "failed_files": failed, "missing_timing_files": unaccounted,
        "presented_frame_count_matched_files": sum(r.get("presented_frame_count_matches") is True for r in rows),
        "discard_preroll_packets": sum(r.get("discard_flagged_packets") or 0 for r in timing["files"]),
        "discard_preroll_files": sum(bool(r.get("discard_flagged_packets")) for r in timing["files"]),
        "editlist_excluded_packets": sum(r.get("editlist_excluded_packets") or 0 for r in timing["files"]),
        "editlist_exclusions": [{"file": r["file"], "default_packet_count": r["packet_count"],
                                  "ignored_editlist_packet_count": r.get("ignored_editlist_packet_count"),
                                  "editlist_excluded_packets": r.get("editlist_excluded_packets"),
                                  "default_stream_duration_seconds": r.get("stream_duration_seconds"),
                                  "ignored_editlist_stream_duration_seconds": r.get("ignored_editlist_stream_duration_seconds")}
                                 for r in timing["files"] if r.get("editlist_excluded_packets")],
        "resolution_counts": dict(Counter(f"{r['width']}×{r['height']}" for r in rows)),
        "capture_groups": capture_groups, "duplicate_groups": duplicate_groups,
        "presentable_cadence_gaps": gaps,
        "candidate_camera_pairs": len(pairs),
        "pairs_with_duration_difference_gt_0_5_seconds": sum(abs(p["a_minus_b_duration_seconds"]) > 0.5 for p in pairs),
        "max_pair_duration_difference_seconds": max((abs(p["a_minus_b_duration_seconds"]) for p in pairs), default=None),
        "camera_synchronization_verified": False,
        "sample_frames": visual.get("successful_samples"), "expected_sample_frames": visual.get("expected_samples"),
        "samples_with_generic_line_candidates": visual.get("samples_with_line_candidates"),
        "sample_failures": visual.get("failed_samples", []),
        "initial_attempt": initial_attempt,
        "notes": ["전체 파일의 첫 번째 영상 스트림을 끝까지 디코딩했다. 오디오와 부가 스트림은 검증 범위 밖이다.",
                  "MP4 nb_frames에는 디코딩 준비용 discard/preroll 패킷과 편집 목록(edit list)에서 재생 대상에서 제외한 패킷이 포함될 수 있다. 두 경우를 별도로 확인해 예상 출력 프레임 수와 비교했다.",
                  "편집 목록 제외는 nb_frames와 기본 패킷 수가 다른 파일에만 -ignore_editlist 1 메타데이터 조회를 추가해 확인했다. 원본 영상이나 편집 목록을 변경하지 않았다.",
                  "FPS 표는 표시 대상 패킷의 중앙 프레임 간격에서 계산했다. r_frame_rate 또는 파일명만으로 실제 촬영 FPS를 판단하지 않았다.",
                  "타임스탬프 간격은 파일의 재생 시각을 나타내며 센서 촬영 시각 정밀도나 롤링셔터 성능을 보증하지 않는다.",
                  "A/B 짝은 파일명으로만 추정했다. 길이가 같거나 다르다는 사실만으로 동일 랠리 또는 동기화를 확정할 수 없다.",
                  "Canny/Hough 후보에는 코트 선 외에 벽·기구·다른 코트 선이 포함된다. 후보 검출은 자동 코트 보정 성공을 뜻하지 않는다."]}
    return result, rows


def render(result, rows):
    proof = result["provenance"]
    ok = result["input_integrity_passed"]
    headline = "영상 입력 검증 통과" if ok else "영상 입력 검증 확인 필요"
    status_class = "pass" if ok else "warn"
    stats = [("검증 영상", f"{result['file_count']}개"),
             ("전체 디코딩 성공", f"{result['decode_passed_files']}개"),
             ("디코딩 프레임", f"{result['total_decoded_frames']:,}"),
             ("샘플 프레임", f"{result['sample_frames']}/{result['expected_sample_frames']}")]
    cards = ''.join(f'<div class="stat"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>' for label, value in stats)
    groups_table = table(["파일명 그룹", "시점", "파일 수", "실제 해상도", "중앙 간격(ms)", "간격 기반 FPS"],
                         [[g["filename_family"], g["view"], g["files"], ", ".join(g["resolutions"]),
                           f"{g['median_presentable_frame_interval_ms']:.3f}" if g['median_presentable_frame_interval_ms'] else "—",
                           f"{g['cadence_fps_from_median_interval']:.2f}" if g['cadence_fps_from_median_interval'] else "—"] for g in result["capture_groups"]])
    duplicates = ''.join(f'<li>{" / ".join(escape(name) for name in group["files"])} — 파일 내용이 완전히 같습니다.</li>' for group in result["duplicate_groups"])
    if not duplicates:
        duplicates = "<li>동일한 SHA-256을 가진 중복 파일이 없습니다.</li>"
    gap_table = table(["영상", "이전 PTS(초)", "다음 PTS(초)", "간격(ms)"],
                      [[g["file"], f"{g['previous_pts_seconds']:.6f}", f"{g['next_pts_seconds']:.6f}", f"{g['interval_ms']:.3f}"] for g in result["presentable_cadence_gaps"]]) if result["presentable_cadence_gaps"] else "<p>중앙 간격의 1.5배를 넘는 표시 대상 프레임 간격은 없었습니다.</p>"
    notes = ''.join(f'<li>{escape(note)}</li>' for note in result["notes"])
    source_findings = []
    for key, label in (("missing_files", "누락"), ("extra_files", "추가"), ("hash_mismatches", "해시 불일치"), ("duplicate_result_rows", "중복 결과 행")):
        source_findings.append(f'{label} {len(proof[key])}개')
    provenance_table = table(["항목", "결과"], [
        ["GitHub 소스 커밋", result["source_commit"]],
        ["커밋에 기록된 MP4", proof["expected_github_video_files"]],
        ["검증 결과와 Git blob 일치", proof["matching_files"]],
        ["소스 대조", " · ".join(source_findings)],
        ["총 크기", f"{result['total_bytes'] / 1024**3:.3f} GiB"],
        ["고유 파일 내용", result["unique_content_count"]]])
    file_table = table(["영상", "디코딩", "프레임", "예상 출력 프레임 일치", "해상도"],
                       [[r["file"], "통과" if r.get("decode_status") == "passed" else r.get("decode_status"),
                         r.get("decoded_frames"), "일치" if r.get("presented_frame_count_matches") is True else "확인 필요",
                         f"{r.get('width')}×{r.get('height')}"] for r in rows])
    editlist_html = ""
    if result["editlist_exclusions"]:
        editlist_html = f'<p>별도로 <strong>{len(result["editlist_exclusions"])}개 파일의 편집 목록(edit list)이 {result["editlist_excluded_packets"]}개 패킷을 재생 대상에서 제외</strong>하는 것을 확인했습니다. 이는 위 discard/preroll 수와 별개입니다. 편집 목록을 무시한 메타데이터 조회와 기본 조회를 비교한 근거:</p>'
        editlist_html += table(["영상", "편집 목록 무시 패킷", "기본 패킷", "제외 패킷", "기본 / 편집 목록 무시 길이(초)"],
                               [[e["file"], e["ignored_editlist_packet_count"], e["default_packet_count"], e["editlist_excluded_packets"],
                                 f"{e['default_stream_duration_seconds']:.6f} / {e['ignored_editlist_stream_duration_seconds']:.6f}"] for e in result["editlist_exclusions"]])
    failures_html = ""
    if result["failed_files"]:
        failures_html = '<div class="callout warn"><strong>확인할 파일</strong><ul>' + ''.join(f"<li>{escape(name)}</li>" for name in result["failed_files"]) + '</ul></div>'
    initial_html = ""
    attempt = result.get("initial_attempt")
    if attempt and attempt["initial_output_timestamp_error_files"]:
        initial_html = f'''<div class="callout scope"><strong>검증 명령 수정 및 재실행:</strong> 최초 실행에서 {attempt['initial_output_timestamp_error_files']}개 파일에 null 출력 장치의 타임스탬프 오류가 기록됐습니다. 원본 시간 단위를 유지하는 <code>-enc_time_base demux</code> 설정으로 재실행하여 {attempt['corrected_timebase_rerun_passed_files']}개 파일의 디코딩 통과를 확인했습니다. 이 보고서는 수정된 최종 실행 결과를 사용하며 <a href="integrity/summary.json">최초 실행 결과</a>도 보존했습니다.</div>'''
    return f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GitHub 영상 검증 결과</title>
<style>
:root{{font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;color:#172033;background:#f3f5f8;line-height:1.65}}body{{margin:0}}main{{max-width:1100px;margin:40px auto;padding:0 24px 60px}}h1{{font-size:32px;letter-spacing:-1px;margin:8px 0}}h2{{font-size:22px;margin:0 0 14px}}p{{margin:10px 0}}a{{color:#155baf}}.eyebrow{{font-size:13px;color:#64748b;letter-spacing:.04em}}.subtitle{{color:#526071;max-width:900px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.stat,section{{background:white;border:1px solid #e1e6ee;border-radius:12px;padding:22px}}.stat span{{display:block;color:#64748b;font-size:13px}}.stat strong{{font-size:26px}}section{{margin:18px 0}}.pill{{display:inline-block;border-radius:30px;padding:4px 11px;font-size:13px}}.pass{{background:#e3f4eb;color:#166442}}.warn{{background:#fff3d7;color:#805600}}.callout{{border-radius:8px;padding:14px 18px;margin:16px 0}}.scope{{background:#eaf0fa;color:#23436d}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px 12px;border-bottom:1px solid #e6ebf2;text-align:left;vertical-align:top}}th{{background:#f8fafc;font-weight:600;white-space:nowrap}}td:first-child{{word-break:break-word}}ul{{padding-left:22px}}li{{margin:6px 0}}img{{width:100%;height:auto;display:block;border-radius:8px}}summary{{cursor:pointer;font-weight:600}}.muted{{color:#64748b;font-size:13px}}.links{{display:flex;gap:16px;flex-wrap:wrap}}footer{{font-size:12px;color:#64748b;margin-top:24px}}@media(max-width:700px){{main{{margin:20px auto;padding:0 14px 40px}}.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:27px}}section{{padding:16px}}}}
</style></head><body><main>
<header><div class="eyebrow">BADMINTON AI LINE JUDGE · REAL VIDEO VALIDATION</div><h1>{headline}</h1>
<span class="pill {status_class}">{"입력 무결성 통과" if ok else "입력 무결성 확인 필요"}</span>
<p class="subtitle">GitHub에 업로드된 원본 영상의 파일 동일성, 전체 디코딩, 프레임 수, 시간 간격과 OpenCV 전처리를 확인했습니다.</p>
<div class="callout scope"><strong>검증 범위:</strong> {escape(result['scope'])} 저장소에 판정 모델·정답 라벨이 없어 인·아웃 정확도 수치는 산출하지 않았습니다.</div>
<p class="muted">작성 시각: {escape(result['generated_at_seoul'])} (Asia/Seoul)</p></header>
<div class="cards">{cards}</div>{failures_html}
<section><h2>원본과 실행 결과</h2>{provenance_table}
<p>예상 출력 프레임 수와 일치한 파일은 <strong>{result['presented_frame_count_matched_files']}/{result['file_count']}개</strong>입니다. {result['discard_preroll_files']}개 파일의 디코딩 준비용 discard/preroll 패킷 {result['discard_preroll_packets']}개는 표시 프레임 계산에서 제외했습니다.</p>
{editlist_html}
<p class="muted">Git 커밋의 영상 blob 해시와 전체 디코딩 실행에서 기록한 파일 해시를 대조했습니다. 오디오 및 부가 스트림은 검증하지 않았습니다.</p>{initial_html}</section>
<section><h2>실제 영상 규격</h2>{groups_table}
<p class="muted">FPS는 각 영상의 표시 대상 패킷 간격 중앙값을 그룹별로 모아 계산했습니다. 파일명 및 FFprobe의 r_frame_rate 추정치와 다를 수 있으며, 센서 촬영 속도를 직접 측정한 값은 아닙니다.</p></section>
<section><h2>데이터 확인 사항</h2><ul>{duplicates}</ul>
<p>파일명 기준 A/B 후보 <strong>{result['candidate_camera_pairs']}쌍</strong> 중 <strong>{result['pairs_with_duration_difference_gt_0_5_seconds']}쌍</strong>은 영상 길이가 0.5초 넘게 다릅니다. 최대 차이는 <strong>{result['max_pair_duration_difference_seconds']:.6f}초</strong>입니다. 같은 랠리인지와 카메라 동기화 여부는 확인하지 않았습니다.</p>
<p>표시 대상 타임스탬프에서 중앙 간격의 1.5배를 넘는 구간:</p>{gap_table}
<p class="muted">이 간격은 타임스탬프 관측값입니다. 촬영 중 누락인지 편집 결과인지는 추가 확인이 필요합니다. 음수 PTS의 discard/preroll 구간은 위 표에서 제외했습니다.</p></section>
<section><h2>실제 프레임 전처리</h2>
<p>각 영상의 25%·50%·75% 지점을 샘플링하여 <strong>{result['sample_frames']}/{result['expected_sample_frames']}프레임</strong>을 읽었습니다. 일반 직선 후보가 나온 샘플은 <strong>{result['samples_with_generic_line_candidates']}개</strong>입니다.</p>
<p class="muted">왼쪽은 원본 샘플, 오른쪽은 Canny/Hough 직선 후보입니다. 후보에는 벽과 기구 등이 포함될 수 있으므로 자동 코트 보정 또는 셔틀콕 검출 성공으로 해석하지 않습니다.</p>
<a href="visual/representative_frames.jpg"><img src="visual/representative_frames.jpg" alt="촬영 그룹 및 시점별 중앙 프레임 원본과 일반 직선 후보"></a></section>
<section><details><summary>파일별 디코딩 결과 {result['file_count']}개</summary>{file_table}</details></section>
<section><h2>근거 파일</h2><div class="links"><a href="final/files.csv">파일별 CSV</a><a href="final/summary.json">전체 검증 JSON</a><a href="timing_audit.json">타임스탬프 검증 JSON</a><a href="visual/frame_summary.json">전처리 JSON</a><a href="result.json">요약 JSON</a></div><ul>{notes}</ul></section>
<footer>소스 커밋 {escape(result['source_commit'])} · 이 보고서는 입력 데이터와 실행 결과를 기록하며 AI 판정 성능 인증을 제공하지 않습니다.</footer>
</main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result, rows = build_result(run_dir)
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.html").write_text(render(result, rows), encoding="utf-8")
    print(json.dumps({"report": str(run_dir / "report.html"), "input_integrity_passed": result["input_integrity_passed"],
                      "sample_preprocessing_passed": result["sample_preprocessing_passed"], "provenance": result["provenance"]}, ensure_ascii=False))
    return 0 if result["input_integrity_passed"] and result["sample_preprocessing_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
