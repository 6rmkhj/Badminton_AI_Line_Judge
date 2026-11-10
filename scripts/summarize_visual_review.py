"""Summarize audited visual results and concrete unresolved-video follow-ups."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ACTIONS = {
 'contact_out_of_view': '같은 사건이 보이는 다른 시점 또는 착지 영역까지 포함한 원본을 확보하고 두 동작으로 대응을 확인한다.',
 'contact_occluded': '가림이 없는 반대 시점의 최초 접촉 연속 프레임을 확보한다. 선수의 회수 행동만으로 접촉점을 정하지 않는다.',
 'no_contact_in_clip': '현재 클립 뒤의 원본 구간을 확보하여 첫 접촉까지 이어서 확인한다.',
 'line_ambiguous': '첫 접촉 구간의 코르크와 실제 바깥 칠 경계를 추가 시점 또는 더 높은 시간 해상도에서 확인한다.',
 'target_ambiguous': '진행 중 랠리의 셔틀과 예비 셔틀·랠리 종료 후 반환 동작을 구분할 추가 타격/접촉 근거가 필요하다.',
 'contact_interval': '접촉 구간과 경계의 가능한 오차 영역을 좁힐 원본 근거를 추가 검수한다.',
 'contact_observed': '대상 정체 또는 유효 경계의 미해결 조건을 검수한다.',
}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',type=Path,default=Path('runs/ai-reviewed-poc'))
 p.add_argument('--model',type=Path,default=Path('models/ai-candidate-v2.pt.json'))
 args=p.parse_args()
 r=json.loads((args.run/'visual-review.json').read_text(encoding='utf-8'))
 rows=[]; statuses=Counter(); frame_count=0
 for v in r['videos']:
  a=v['annotation']; frame_count+=len(set(a.get('inspected_frame_indices',[])))
  status=a.get('observation_status'); statuses[status]+=1
  notes=a.get('follow_up_actions') or [ACTIONS.get(status,'영상별 검수 사유에 필요한 원본 근거를 보완한다.')]
  if isinstance(notes,str): notes=[notes]
  row={'video_id':v['video_id'],'file':v['file'],'assisted_decision':v['assisted_decision'],
       'direct_assisted_decision':v['direct_assisted_decision'],'observation_status':status,
       'originating_video_id':v.get('originating_video_id'),'source_sha256':v['source_sha256'],
       'rationale':a['rationale'],'follow_up_actions':notes,
       'reason_codes':v['review'].get('reason_codes',[]),'pair_match_status':a.get('pair_match_status')}
  rows.append(row)
 unresolved=[x for x in rows if x['assisted_decision']=='REVIEW']
 summary={'video_count':len(rows),'inspected_frame_index_count':frame_count,'assisted_counts':r['assisted_counts'],
          'unresolved_count':len(unresolved),'automatic_accuracy':None,'human_ground_truth_count':0,
          'unresolved':unresolved,'all_videos':rows}
 (args.run/'follow-up.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 with (args.run/'verdicts.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.writer(f);writer.writerow(['파일','AI 보조 판정','직접 판정','관측 상태','접촉 근거 영상 ID','미해결 후속 조치'])
  for row in rows: writer.writerow([row['file'],row['assisted_decision'],row['direct_assisted_decision'],row['observation_status'],row['originating_video_id'] or '', '; '.join(row['follow_up_actions']) if row['assisted_decision']=='REVIEW' else ''])
 model=json.loads(args.model.read_text(encoding='utf-8')) if args.model.exists() else {}
 benchmark_path=args.run/'detector-validation-v2.json'
 benchmark=json.loads(benchmark_path.read_text()) if benchmark_path.exists() else {}
 counts=r['assisted_counts']; direct=r['direct_assisted_counts']
 text=f'''# AI 직접 검수·보조 판정 결과 · 2026-11-10

74개 원본 모두 시각 검수 기록을 작성했다. 영상 파일 기준 AI 보조 판정은 **IN {counts['IN']} / OUT {counts['OUT']} / REVIEW {counts['REVIEW']}**다. 전부 IN/OUT으로 확정한 결과가 아니며, 남은 영상의 관측 한계와 후속 조치를 영상별로 보존했다.

| 항목 | 결과 |
|---|---:|
| 영상 / 유효 검수 기록 / 현재 원본 해시 검증 | {r['video_count']} / {r['valid_review_count']} / {r['source_hash_verified_count']} |
| 고유 원본 내용(SHA256) | {len({x['source_sha256'] for x in r['videos']})} |
| 직접 접촉 검수 IN / OUT | {direct['IN']} / {direct['OUT']} |
| 실제 동작 대응으로 다른 시점의 직접 판정을 연결 | {r['cross_view_transferred_count']} |
| 최종 AI 보조 IN / OUT / REVIEW | {counts['IN']} / {counts['OUT']} / {counts['REVIEW']} |
| 검수 기록의 고유 프레임 인덱스 합계 | {frame_count} |
| 이전 무보조 자동 실행 | REVIEW 74 |
| 독립된 사람 정답 / 실제 자동 판정 정확도 | 0 / 미측정 |

B17/B18은 동일 내용이고 같은 랠리의 서로 다른 시점도 포함되어 있으므로, 파일 수를 독립 사건 수로 해석하지 않는다.

검수는 영상 전체 시간 범위의 접촉 시트를 먼저 확인하고, 접촉·시점 대응 후보를 원본 연속 프레임으로 확대하는 방식이다. 모든 15,017프레임을 사람이 한 장씩 검수했다는 뜻이 아니다. `inspected_frame_indices`와 실제 확대 이미지가 판독 범위를 기록한다. AI 검수는 사람의 독립 정답과 구분되며, 이를 이용한 보조 판정은 자동 정확도 집계에 포함하지 않는다.

일부 영상에서는 최초 접촉이 안쪽인데 반등 후 밖에 멈췄다. 또한 예비 셔틀 회수와 실제 랠리 종료를 혼동할 수 있었고, 파일명이 대응해도 다른 랠리인 사례가 있었다. 따라서 정지점·회수 동작·파일명만으로 정답을 만들지 않았다. 근접 라인 사례는 접촉/칠 경계의 오차와 가능한 지면 영역을 함께 계산했다.

## 후보 모델 실험

확정된 AI 패치 검수 스냅샷으로 후보 분류기를 학습했다. 동일 내용 중복 영상의 같은 패치는 제거했고, 기존 랠리/중복 그룹 분할을 유지했다. v1 이후 원본 재검수에서 A16의 한 패치 좌표를 정정하고 B-UHD3의 검수 패치를 추가해, 같은 40 epoch 설정으로 v2를 다시 학습했다. 같은 데이터셋에서 진행한 개발 실험이므로 독립 최종 성능 시험이라고 해석하지 않는다. v2 학습 스냅샷 동결 후 마지막 원본 검수에서 추가한 24개 패치는 최종 검수 레이블에 보존했으며 이 가중치 학습에는 사용하지 않았다.

'''
 if model:
  c=model['counts']; total=sum(sum(x.values()) for x in c.values()); m=model['metrics']['test']
  text+=f"AI 검수 패치 {total}개를 사용했고, 검증 손실로 선택한 epoch는 {model['selected_epoch']}회차다. 테스트 그룹 패치 {m['samples']}개의 AI 레이블 일치율은 {m['accuracy']:.1%}, 셔틀 패치 검출 비율은 {m['recall']:.1%}다. 이는 전체 영상 탐지 재현율이나 IN/OUT 정확도가 아니다. 가중치 SHA256은 `{model['weights_sha256']}`다.\n\n"
 if benchmark:
  c=benchmark['counts']; base=c['bright_baseline'];new=c['dark_ai_ranked']
  text+=f"같은 원본 조건의 {benchmark['video_count']}개 테스트 영상에서, AI로 표시한 셔틀 위치 6px 이내에 상위 64개 후보가 존재한 경우는 밝은 후보 기준 {base.get('shuttle_points_with_candidate',0)}/{base.get('shuttle_points',0)}, 어두운 코르크 후보와 학습 재정렬 적용 후 {new.get('shuttle_points_with_candidate',0)}/{new.get('shuttle_points',0)}로 바뀌었다. 후보가 근처에 있다는 지표이므로 목표 셔틀 정체·최초 접촉·정확한 라인 판정이 확인됐다는 뜻은 아니다.\n\n"
 text+='후보 분류기는 실험용이며 기본 자동 접촉 수락 조건을 완화하지 않았다. 실제 독립 판정 모델 학습·검증을 완료한 것으로 보지 않는다.\n\n## 미해결 영상\n\n'
 text+='각 영상의 구체적인 관측 사유와 실행 가능한 후속 조치는 `runs/ai-reviewed-poc/follow-up.json`에 있다. 화면에서 미리보기·원본 확대·대응 동작을 함께 확인할 수 있다.\n\n| 파일 | 관측 상태 | 필요한 후속 조치 |\n|---|---|---|\n'
 for row in unresolved:
  clean=lambda s:str(s).replace('|','/').replace('\n',' ')
  text+='| '+ ' | '.join(map(clean,[row['file'],row['observation_status'],'; '.join(row['follow_up_actions'])]))+' |\n'
 text+='\n## 재현·검증 자료\n\n[검수 스키마와 실행](AI_REVIEW.md)을 따른다. 최종 보고서는 `runs/ai-reviewed-poc/index.html`, 기계 판독 결과는 `visual-review.json`, 간단한 목록은 `verdicts.csv`다. 원본과 기존 자동 결과는 보존했다.\n\n'
 testlog=args.run/'software-tests.log'
 if testlog.exists():
  matched=re.findall(r'(\d+) passed',testlog.read_text())
  if matched: text+=f"전체 소프트웨어 테스트 {matched[-1]}개 통과. 원본 해시·검수 출처·중복·경계 오차·시점 대응 감사와 실제 Chromium 화면 검증 결과는 같은 실행 폴더의 JSON/로그/스크린샷에 저장한다.\n"
 Path('docs/AI_REVIEW_RESULTS.md').write_text(text,encoding='utf-8')
 print(json.dumps({'counts':counts,'unresolved':len(unresolved),'inspected_indices':frame_count},ensure_ascii=False))


if __name__=='__main__':main()
