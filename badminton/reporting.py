"""Write a Korean, evidence-based summary of a finalized research run."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_relative(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Run artifact path escapes run directory")
    return path


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _pair_sync(pair):
    """Prefer the current index schema; retain older run compatibility."""
    current = pair.get("sync")
    if isinstance(current, dict):
        return current
    legacy = pair.get("synchronization")
    return legacy if isinstance(legacy, dict) else {}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _escape(value):
    return html.escape(str(value), quote=True)


def _percent(value):
    return "미측정" if value is None else f"{100 * value:.1f}%"


def _link(relative, label):
    return f'<a href="{_escape(quote(str(relative).replace(os.sep, "/"), safe="/"))}">{_escape(label)}</a>'


def _table(headers, rows):
    head = "".join(f"<th>{_escape(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _summarize(run):
    index, evaluation, manifest = (_read(run / name) for name in ("index.json", "evaluation.json", "manifest.json"))
    if index.get("status") not in ("completed", "completed_with_failures"):
        raise ValueError("Refusing to publish a final summary while the run is not finalized")
    entries = index.get("videos", [])
    if evaluation.get("video_count") != len(entries):
        raise ValueError("evaluation.json does not match the finalized run; run badminton evaluate first")
    manifest_by_id = {row["id"]: row for row in manifest.get("videos", [])}
    status_counts = Counter(row.get("status", "unknown") for row in entries)
    decisions, calibration_status, sync_status, detector_methods = Counter(), Counter(), Counter(), Counter()
    implementation_fingerprints = Counter()
    source_errors, rows, read_errors = [], [], []
    total_frames = total_detections = budget_frames = total_tracks = ambiguous_tracks = 0
    budget_videos = ambiguous_videos = trained_model_videos = automatic_calibration_accepted = 0
    independent_contacts = 0
    analysis_seconds = 0.0
    reason_counts = Counter()
    for entry in entries:
        identifier = entry.get("id") or entry.get("video_id")
        row = {"id": identifier, "file": entry.get("file"), "status": entry.get("status"),
               "result_path": entry.get("result_path", f"videos/{identifier}/result.json"),
               "preview_path": entry.get("preview_path", f"videos/{identifier}/preview.mp4")}
        if entry.get("status") != "completed":
            row.update(decision="ERROR", error=entry.get("error"), calibration_status=None,
                       track_count=None, ambiguous_track_count=None, landing_count=None, budget_hit_frames=None)
            decisions["ERROR"] += 1
            rows.append(row)
            continue
        try:
            result = _read(_safe_relative(run, row["result_path"]))
        except (OSError, ValueError) as error:
            row.update(decision="ERROR", error=str(error), calibration_status=None,
                       track_count=None, ambiguous_track_count=None, landing_count=None, budget_hit_frames=None)
            read_errors.append({"video_id": identifier, "error": str(error)})
            decisions["ERROR"] += 1
            rows.append(row)
            continue
        decision = result.get("decision", {})
        label = decision.get("decision", decision.get("label", "ERROR")) if isinstance(decision, dict) else decision
        label = label if label in ("IN", "OUT", "REVIEW") else "ERROR"
        decisions[label] += 1
        if isinstance(decision, dict):
            reason_counts.update(decision.get("reason_codes", []))
        calibration = result.get("calibration", {})
        calibration_status[calibration.get("status", "unknown")] += 1
        auto_calibrated = (calibration.get("status") == "ok" and not calibration.get("manual_reference")
                           and calibration.get("method") != "manual_reference")
        automatic_calibration_accepted += auto_calibrated
        quality = result.get("detection_quality", {})
        tracks = result.get("tracks", [])
        frames = quality.get("frames", result.get("analysis", {}).get("frame_count", 0))
        budget = quality.get("candidate_budget_hit_frames", 0)
        ambiguous = quality.get("ambiguous_track_count", sum(bool(track.get("identity_ambiguous")) for track in tracks))
        total_frames += frames or 0
        total_detections += quality.get("detection_count", 0) or 0
        budget_frames += budget or 0
        budget_videos += bool(budget)
        total_tracks += len(tracks)
        ambiguous_tracks += ambiguous or 0
        ambiguous_videos += bool(ambiguous)
        model = result.get("detector", {})
        trained_model_videos += model.get("trained_model_provided") is True
        detector_methods[model.get("method", "unknown")] += 1
        independent_contacts += sum(landing.get("contact_observed") is True for landing in result.get("landings", []))
        elapsed = result.get("timings", {}).get("total_seconds")
        if _finite(elapsed):
            analysis_seconds += elapsed
        fingerprint = result.get("implementation_fingerprint")
        if fingerprint:
            # Keep JSON-shaped fingerprints reproducible without assuming a type.
            fingerprint = fingerprint if isinstance(fingerprint, str) else json.dumps(fingerprint, sort_keys=True)
            implementation_fingerprints[fingerprint] += 1
        source = manifest_by_id.get(identifier)
        source_hash = result.get("source_sha256")
        if source is None or not source_hash or source_hash != source.get("sha256"):
            source_errors.append({"video_id": identifier, "expected_sha256": source.get("sha256") if source else None,
                                  "result_source_sha256": source_hash})
        row.update(decision=label, calibration_status=calibration.get("status"),
                   track_count=len(tracks), ambiguous_track_count=ambiguous,
                   landing_count=len(result.get("landings", [])), budget_hit_frames=budget,
                   trained_model_provided=model.get("trained_model_provided") is True,
                   reason_codes=decision.get("reason_codes", []) if isinstance(decision, dict) else [],
                   source_sha256=source_hash, analysis_seconds=elapsed)
        rows.append(row)
    for pair in index.get("pairs", []):
        sync_status[_pair_sync(pair).get("status", "unknown")] += 1
    completed = status_counts["completed"] - len(read_errors)
    failed = len(entries) - completed
    automatic = decisions["IN"] + decisions["OUT"]
    ground_truth = evaluation.get("eligible_ground_truth_count", 0)
    accuracy = evaluation.get("decision", {}).get("automatic_accuracy")
    training_file = next((path for path in (run / "training-readiness.json", run / "training_readiness.json") if path.exists()), None)
    training = _read(training_file) if training_file else None
    all_weights = completed > 0 and trained_model_videos == completed
    automatic_accuracy_ready = (ground_truth > 0 and accuracy is not None and automatic > 0 and all_weights)
    modules = ["media", "calibration", "detection", "tracking", "events", "contact_model", "decision", "sync", "web", "dataset", "evaluation", "learning"]
    package = Path(__file__).resolve().parent
    module_presence = {name: (package / f"{name}.py").exists() for name in modules}
    software_available = all(module_presence.values())
    acceptance = evaluation.get("performance_acceptance_established") is True
    overall = software_available and failed == 0 and automatic_accuracy_ready and acceptance and not source_errors
    screenshot = next((name for name in ("review-ui.png", "review-screenshot.png", "screenshots/review-ui.png") if (run / name).exists()), None)
    return {
        "schema_version": 1, "generated_at_seoul": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "run_id": index.get("run_id", run.name), "pipeline_version": index.get("pipeline_version"),
        "scope": "구현된 연구용 분석·검수·평가 소프트웨어의 실행 결과입니다. 자동 판정 품질이 검증되었다는 뜻이 아닙니다.",
        "implementation": {"software_workflow_available": software_available, "modules_present": module_presence,
                           "implementation_fingerprints": dict(implementation_fingerprints)},
        "goal_status": {"overall_goal_complete": overall,
                        "real_automatic_accuracy_ready": automatic_accuracy_ready,
                        "accuracy_measurement_available": accuracy is not None,
                        "trained_weights_used_for_all_completed_clips": all_weights,
                        "independent_performance_acceptance_established": acceptance,
                        "reason": "소프트웨어 구현과 실제 자동 판정 성능 입증은 별도입니다. 정답 검수·학습·독립 시험이 충족돼야 전체 목표 달성으로 볼 수 있습니다."},
        "execution": {"status": index["status"], "indexed_videos": len(entries), "completed_videos": completed,
                      "failed_videos": failed, "manifest_videos": manifest.get("file_count"),
                      "all_manifest_videos_included": {row.get("id") for row in entries} == set(manifest_by_id),
                      "processed_frames": total_frames, "summed_clip_analysis_seconds": analysis_seconds,
                      "result_read_errors": read_errors},
        "decisions": {"counts": dict(decisions), "automatic_decisions": automatic,
                      "automatic_coverage": _ratio(automatic, len(entries)),
                      "review_rate": _ratio(decisions["REVIEW"], len(entries)),
                      "all_successfully_processed_clips_review": completed > 0 and decisions["REVIEW"] == completed,
                      "reason_counts": dict(reason_counts.most_common())},
        "calibration": {"status_counts": dict(calibration_status),
                        "automatic_accepted_clips": automatic_calibration_accepted,
                        "acceptance_rate": _ratio(automatic_calibration_accepted, completed),
                        "measured_accuracy": evaluation.get("calibration", {}).get("success_rate"),
                        "note": "ok는 알고리즘의 기하 기준 수락이며 정답 대비 보정 정확도 또는 실제 성공률이 아닙니다."},
        "synchronization": {"pairs": len(index.get("pairs", [])), "status_counts": dict(sync_status),
                            "accepted_pairs": sync_status["ok"],
                            "same_rally_verified_pairs": sum(_pair_sync(pair).get("same_rally_verified") is True for pair in index.get("pairs", [])),
                            "note": "정렬 후보 수락은 독립적으로 측정한 동기화 정확도가 아니며, 동일 랠리·최초 접촉 관측을 보증하지 않습니다."},
        "detection_tracking": {"candidate_count": total_detections, "candidate_budget_hit_frames": budget_frames,
                               "candidate_budget_hit_fraction": _ratio(budget_frames, total_frames),
                               "videos_with_candidate_budget_hits": budget_videos,
                               "track_count": total_tracks, "ambiguous_track_count": ambiguous_tracks,
                               "ambiguous_track_fraction": _ratio(ambiguous_tracks, total_tracks),
                               "videos_with_identity_ambiguity": ambiguous_videos,
                               "detector_methods": dict(detector_methods),
                               "clips_with_trained_weights_provided": trained_model_videos,
                               "contact_observed_candidates": independent_contacts,
                               "precision": evaluation.get("detection", {}).get("precision"),
                               "recall": evaluation.get("detection", {}).get("recall"),
                               "note": "후보 수·예산 소진·ID 모호성은 실행 진단입니다. 후보 수가 많다는 사실은 탐지 성능 향상을 의미하지 않습니다."},
        "ground_truth": {"annotation_count": evaluation.get("annotation_count", 0),
                         "verified_human_annotation_count": evaluation.get("verified_human_annotation_count", 0),
                         "eligible_ground_truth_count": ground_truth,
                         "excluded_ground_truth_counts": evaluation.get("excluded_ground_truth_counts", {}),
                         "accuracy": accuracy, "accuracy_wilson_95ci": evaluation.get("decision", {}).get("automatic_accuracy_wilson_95ci"),
                         "near_line_within_5cm": evaluation.get("near_line_within_5cm"),
                         "draft_count": len(list((run / "annotation-drafts").glob("*.json"))),
                         "drafts_are_ground_truth": False},
        "dataset": {"file_count": manifest.get("file_count"), "unique_content_count": manifest.get("unique_content_count"),
                    "group_count": manifest.get("group_count"), "split_group_counts": manifest.get("split_group_counts"),
                    "dataset_fingerprint": manifest.get("dataset_fingerprint"),
                    "duplicate_groups": manifest.get("duplicate_groups", []),
                    "generalization_evidence": manifest.get("generalization_evidence", {}),
                    "recorded_source_hash_matches": completed - len(source_errors),
                    "recorded_source_hash_mismatches": source_errors,
                    "provenance_note": "실행 결과가 기록한 원본 SHA256을 매니페스트와 비교했습니다. 새로운 체육관·기기의 독립 성능 검증을 의미하지 않습니다."},
        "training_readiness": training,
        "training_readiness_path": training_file.name if training_file else None,
        "review_screenshot_path": screenshot,
        "videos": rows,
        "links": {"index": "index.json", "evaluation": "evaluation.json", "manifest": "manifest.json",
                  "annotation_drafts": "annotation-drafts/", "readme": os.path.relpath(package.parent / "README.md", run),
                  "evaluation_guide": os.path.relpath(package.parent / "docs/evaluation.md", run)},
    }


def _render(data, run):
    execution, decision, truth = data["execution"], data["decisions"], data["ground_truth"]
    calibration, sync, diagnostics = data["calibration"], data["synchronization"], data["detection_tracking"]
    dataset, goal = data["dataset"], data["goal_status"]
    all_review = decision["all_successfully_processed_clips_review"]
    headline = "분석·검수 도구 구현 — 실제 자동 판정 성능은 아직 미입증"
    if goal["overall_goal_complete"]:
        headline = "구현 및 독립 성능 기준 충족 근거가 기록되었습니다"
    if all_review:
        important = f"처리된 {execution['completed_videos']}개 영상 모두 REVIEW입니다. 자동 IN/OUT 판정 성공을 입증한 결과가 아닙니다."
    elif truth["eligible_ground_truth_count"] == 0:
        important = "적격 정답 라벨이 없어 실제 판정 정확도를 계산할 수 없습니다. 처리 성공과 정확도 검증을 구분해야 합니다."
    else:
        important = "아래 정확도는 적격 정답 사례 중 자동 IN/OUT을 낸 사례에 한정됩니다. REVIEW 비율과 신뢰구간을 함께 확인하세요."
    cards = [("영상 처리", f"{execution['completed_videos']} / {execution['indexed_videos']}", f"실패 {execution['failed_videos']}개"),
             ("자동 IN/OUT", str(decision["automatic_decisions"]), f"자동 판정 비율 {_percent(decision['automatic_coverage'])}"),
             ("REVIEW", str(decision["counts"].get("REVIEW", 0)), f"REVIEW 비율 {_percent(decision['review_rate'])}"),
             ("실제 정답", str(truth["eligible_ground_truth_count"]), f"자동 판정 정확도 {_percent(truth['accuracy'])}")]
    cards_html = "".join(f'<div class="metric"><span>{_escape(label)}</span><strong>{_escape(value)}</strong><small>{_escape(note)}</small></div>' for label, value, note in cards)
    implementation_rows = [
        ["분석·검수·평가 모듈", "코드 제공" if data["implementation"]["software_workflow_available"] else "일부 파일 누락", "영상 처리 결과와 모듈별 근거를 저장합니다."],
        ["실제 영상 실행", f"{execution['completed_videos']}개 완료 / {execution['failed_videos']}개 실패", "프로그램 실행 여부이며 정확도 통과가 아닙니다."],
        ["학습된 탐지 가중치 사용", f"{diagnostics['clips_with_trained_weights_provided']} / {execution['completed_videos']}개 영상", "현재 사용된 검출 방식과 모델 출처를 구분합니다."],
        ["실제 자동 판정 정확도", _percent(truth["accuracy"]), "독립 검수 라벨과 자동 판정 사례가 필요합니다."],
        ["전체 목표 달성", "충족 근거 있음" if goal["overall_goal_complete"] else "미완료", "소프트웨어 구현만으로 실제 대회 판정 성능을 확정하지 않습니다."],
    ]
    overview = _table(["구분", "현재 결과", "해석"], implementation_rows)
    limits = _table(["항목", "관측값", "주의"], [
        ["자동 코트 기하 수락", f"{calibration['automatic_accepted_clips']} / {execution['completed_videos']} ({_percent(calibration['acceptance_rate'])})", "정답 대비 보정 성공률이 아닙니다."],
        ["시점 정렬 수락", f"{sync['accepted_pairs']} / {sync['pairs']}쌍", "동일 랠리 검증 및 정밀 동기화 정확도와 구분합니다."],
        ["후보 예산 소진", f"{diagnostics['candidate_budget_hit_frames']}프레임 / {diagnostics['videos_with_candidate_budget_hits']}개 영상", "제한에 걸린 프레임에서는 일부 후보가 제외될 수 있습니다."],
        ["추적 ID 모호성", f"{diagnostics['ambiguous_track_count']} / {diagnostics['track_count']} 궤적", "많은 후보·궤적은 높은 재현율의 증거가 아닙니다."],
        ["확인된 접촉 후보", diagnostics["contact_observed_candidates"], "운동 변화만으로 물리적 최초 접촉을 확정하지 않습니다."],
        ["탐지 precision / recall", f"{_percent(diagnostics['precision'])} / {_percent(diagnostics['recall'])}", "완전한 프레임별 검수 정답이 없으면 미측정입니다."],
    ])
    duplicate_html = _table(["같은 내용의 파일", "누수 방지 그룹"], [[" / ".join(group["files"]), group.get("group_id", "")] for group in dataset["duplicate_groups"]]) if dataset["duplicate_groups"] else "<p>중복 내용 그룹이 없습니다.</p>"
    split = dataset.get("split_group_counts") or {}
    source = f"<p>원본 {dataset['file_count']}개, 고유 내용 {dataset['unique_content_count']}개, 연결 그룹 {dataset['group_count']}개입니다. 그룹 분할은 train {split.get('train', 0)} / val {split.get('val', 0)} / test {split.get('test', 0)}입니다. A/B 후보와 중복 영상은 같은 그룹으로 묶어 분할 누수를 방지했습니다.</p>"
    source += f"<p>결과에 기록된 원본 SHA256 일치: {dataset['recorded_source_hash_matches']}개. 불일치: {len(dataset['recorded_source_hash_mismatches'])}개. 체육관·날짜·기기의 독립성이 확인되지 않았다면 이 분할을 외부 환경 성능 검증으로 해석할 수 없습니다.</p>"
    ci = truth["accuracy_wilson_95ci"]
    ci_text = "미측정" if ci is None else f"{_percent(ci[0])} ~ {_percent(ci[1])}"
    evaluation_html = f"<p>사람이 검수한 기록 {truth['verified_human_annotation_count']}개 중 정확도 평가에 적격한 정답은 {truth['eligible_ground_truth_count']}개입니다. 자동 판정 정확도: <b>{_percent(truth['accuracy'])}</b>, Wilson 95% 신뢰구간: {ci_text}.</p>"
    evaluation_html += f"<p>검수 초안 {truth['draft_count']}개는 정답이 아닙니다. UNKNOWN·머신 초안·미확인 접촉은 정확도에서 제외하며, 경계선 5cm 이내 사례는 독립 관측 근거가 필요합니다. 확인할 수 없는 수치는 0%나 100%가 아니라 <code>null</code>로 기록합니다. 목표치 99% 달성은 주장하지 않습니다.</p>"
    training = data["training_readiness"]
    if training is None:
        training_html = "<p>저장된 학습 준비도 보고서가 없습니다. 준비 완료로 간주하지 않습니다.</p>"
    else:
        training_html = f"<p>학습 실행 전제 충족: {'예' if training.get('ready') else '아니오'}. 검수된 분류 패치: {training.get('verified_patch_count', 0)}개. 패치 분류 학습은 셔틀콕 재현율이나 인·아웃 정확도 자체가 아닙니다.</p>"
        reasons = training.get("reasons", [])
        if reasons:
            training_html += "<ul>" + "".join(f"<li>{_escape(reason)}</li>" for reason in reasons) + "</ul>"
    video_rows = []
    for row in data["videos"]:
        links = []
        for key, label in (("result_path", "결과 JSON"), ("preview_path", "재생 영상")):
            if (run / row[key]).is_file():
                links.append(_link(row[key], label))
        draft = f"annotation-drafts/{row['id']}.json"
        if (run / draft).is_file():
            links.append(_link(draft, "검수 초안"))
        values = [row["file"], row["status"], row["decision"], row.get("calibration_status") or "—",
                  row.get("budget_hit_frames") if row.get("budget_hit_frames") is not None else "—",
                  row.get("ambiguous_track_count") if row.get("ambiguous_track_count") is not None else "—"]
        video_rows.append("<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in values) + f"<td>{' · '.join(links)}</td></tr>")
    videos_html = '<div class="table-scroll"><table><thead><tr><th>영상</th><th>실행</th><th>판정</th><th>보정 상태</th><th>예산 소진 프레임</th><th>모호한 궤적</th><th>근거</th></tr></thead><tbody>' + "".join(video_rows) + "</tbody></table></div>"
    links = [_link(relative, label) for key, label in (("evaluation", "평가 JSON"), ("manifest", "데이터 매니페스트"), ("index", "실행 인덱스"), ("readme", "실행 안내"), ("evaluation_guide", "정답·평가 기준")) if (relative := data["links"][key]) and (run / relative).exists()]
    if data.get("training_readiness_path"):
        links.append(_link(data["training_readiness_path"], "학습 준비도"))
    links.append(_link("summary.json", "요약 JSON"))
    screenshot = ""
    if data.get("review_screenshot_path"):
        screenshot = f'<figure><img src="{_escape(data["review_screenshot_path"])}" alt="영상·근거·정답 라벨 검수 화면"><figcaption>검수 화면 캡처. 화면 기능 제공은 실제 판정 정확도의 증거가 아닙니다.</figcaption></figure>'
    style = """body{margin:0;background:#f3f5f8;color:#172033;font:15px/1.65 system-ui,-apple-system,sans-serif}main{max-width:1200px;margin:auto;padding:36px 24px 64px}header{margin-bottom:24px}h1{font-size:29px;line-height:1.35;margin:8px 0 12px}h2{font-size:21px;margin:0 0 14px}small,.muted{color:#5d687d}.badge{display:inline-block;border-radius:5px;background:#dce6f7;padding:4px 10px;font-weight:650;color:#214273}.warning{background:#fff1da;border-left:5px solid #b46800;padding:18px 20px;margin:24px 0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.metric,section{background:#fff;border:1px solid #dce1e8;border-radius:10px;padding:20px}.metric span,.metric strong,.metric small{display:block}.metric strong{font-size:31px;line-height:1.5}section{margin-top:20px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border-bottom:1px solid #e1e5eb;text-align:left;vertical-align:top;padding:10px 12px}th{background:#f0f3f7;white-space:nowrap}td:first-child{overflow-wrap:anywhere}.table-scroll{overflow-x:auto}a{color:#2055a2;text-decoration:none}a:hover{text-decoration:underline}.links{display:flex;flex-wrap:wrap;gap:14px}code{background:#eef1f5;padding:2px 5px}figure{margin:16px 0 0}img{display:block;max-width:100%;height:auto;border:1px solid #dce1e8;border-radius:8px}figcaption{font-size:13px;color:#5d687d;margin-top:8px}ul{padding-left:22px}@media(max-width:720px){main{padding:20px 12px}h1{font-size:24px}.metrics{grid-template-columns:repeat(2,1fr)}section{padding:16px}}"""
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>배드민턴 영상 분석 실행 보고서</title><style>{style}</style></head><body><main>
<header><span class="badge">연구용 PoC 실행 보고서</span><h1>{_escape(headline)}</h1><p class="muted">실행 {_escape(data['run_id'])} · 버전 {_escape(data['pipeline_version'])} · 작성 {_escape(data['generated_at_seoul'])}</p><p>{_escape(data['scope'])}</p></header>
<div class="warning"><strong>{_escape(important)}</strong></div><div class="metrics">{cards_html}</div>
<section><h2>구현 상태와 실제 성능</h2>{overview}</section>
<section><h2>이번 영상에서 드러난 한계</h2>{limits}<p>{_escape(sync['note'])}</p></section>
<section><h2>정답 라벨과 평가</h2>{evaluation_html}{training_html}</section>
<section><h2>원본과 데이터 분할</h2>{source}{duplicate_html}</section>
<section><h2>검수 화면과 산출물</h2><p class="links">{' '.join(links)}</p><p>실행 명령: <code>python -m badminton serve --run {_escape(str(run))}</code></p>{screenshot}</section>
<section><h2>영상별 실행 결과</h2>{videos_html}</section>
</main></body></html>"""


def write_run_report(run_dir):
    """Write ``summary.json`` and ``summary.html`` after finalized analysis/evaluation."""
    run = Path(run_dir).resolve()
    data = _summarize(run)
    document = _render(data, run)
    targets = {"json": run / "summary.json", "html": run / "summary.html"}
    for kind, content in (("json", json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"), ("html", document)):
        target = targets[kind]
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    return {key: str(path) for key, path in targets.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/latest"))
    args = parser.parse_args()
    print(json.dumps(write_run_report(args.run), ensure_ascii=False))


if __name__ == "__main__":
    main()
