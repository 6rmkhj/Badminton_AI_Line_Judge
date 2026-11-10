"""Audit the original 15 unresolved pairs against source reinspection and live jobs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.root.resolve();base=root/'runs/review-remediation'
    before=read(base/'baseline/visual-review.json');after=read(root/'runs/ai-reviewed-poc/visual-review.json')
    jobs=read(base/'verification/all-pairs.json');assert jobs['passed']
    original=[p for p in before['filename_pair_audit']['items'] if p['category']=='both_review']
    original.sort(key=lambda p:[(1,int(x)) if x.isdigit() else (0,x) for x in re.split(r'(\d+)',p['pair_key'])])
    assert len(original)==15
    inspections={};sources=[]
    for group in ('boundary','occlusion','target','uhd'):
        path=base/group/'inspection.json';document=read(path)
        sources.append({'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for item in document.get('items',document.get('pairs',[])):
            assert item['pair_key'] not in inspections
            inspections[item['pair_key']]=item
    decisions={p['pair_key']:p for p in jobs['checks']}
    current={v['video_id']:v for v in after['videos']}
    old={v['video_id']:v for v in before['videos']}
    rows=[]
    for pair in original:
        key=pair['pair_key'];inspection=inspections[key];job=decisions[key]
        views=[]
        for v in pair['views']:
            vid=v['video_id'];now=current[vid];a=now['annotation']
            assert a==read(root/f'annotations/ai-visual/{vid}.json')
            assert a['source_sha256']==old[vid]['source_sha256'] and a['human_verified'] is False
            views.append({'video_id':vid,'file':now['file'],'source_sha256':a['source_sha256'],
                          'before_observation':old[vid]['annotation'].get('observation_status'),
                          'after_observation':a.get('observation_status'),'rationale':a.get('rationale'),
                          'direct_decision':now['direct_assisted_decision'],'assisted_decision':now['assisted_decision'],
                          'inspected_frame_indices':a['inspected_frame_indices'],'evidence':now['assets']['evidence'],
                          'follow_up_actions':a.get('follow_up_actions',[])})
        summary=inspection.get('summary_ko') or inspection.get('cause_ko')
        action=inspection.get('next_action_ko') or inspection.get('next_actions') or inspection.get('follow_up_actions')
        if isinstance(action,list): action='; '.join(action)
        if not action: action='; '.join(dict.fromkeys(x for v in views for x in v['follow_up_actions']))
        if job['decision'] in {'IN','OUT'}:
            action='최초 접촉 재주석과 A/B 근거 연결 완료. AI 보조 검수 결과로 표시한다.'
        rows.append({'pair_key':key,'before':'REVIEW','after':job['decision'],
                     'same_event_confirmed':job.get('event_correspondence_source') is not None,
                     'reason_codes':job['reason_codes'],'summary_ko':summary,'next_action_ko':action,
                     'job_id':job['job_id'],'views':views})
    counts={k:sum(row['after']==k for row in rows) for k in ('IN','OUT','REVIEW')}
    assert counts=={'IN':2,'OUT':0,'REVIEW':13},counts
    assert jobs['counts']=={'IN':18,'OUT':5,'REVIEW':14}
    result={'passed':True,'created_at':datetime.now(timezone.utc).isoformat(),
            'scope':'All15 originally both-REVIEW pairs; source visual reinspection and assisted linkage, not independent automatic accuracy',
            'original_both_review_pairs':15,'after_counts':counts,'all37_pair_counts':jobs['counts'],
            'all74_file_counts':after['assisted_counts'],'independent_human_ground_truth':False,
            'automatic_judgment_accuracy':None,'sources':sources,'pairs':rows,
            'final_report_sha256':hashlib.sha256((root/'runs/ai-reviewed-poc/visual-review.json').read_bytes()).hexdigest()}
    (base/'verification/15-pair-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text='''# REVIEW 15쌍 재검수와 개선 결과

기존에 A/B 양쪽 모두 REVIEW였던 15쌍의 원본을 다시 확인했다. **2쌍은 IN으로 해결했고 13쌍은 REVIEW를 유지했다.** 같은 번호가 모두 같은 순간을 담는다고 가정하지 않고, 실제 동작 두 개 이상과 원본 프레임 시각을 대조했다.

전체 통합 결과의 REVIEW는 17쌍에서 14쌍으로 줄었다. 원래 15쌍에 포함되지 않았던 8K/4K 6번의 연결 누락도 해결했기 때문이다. 최종 37쌍은 **IN 18 / OUT 5 / REVIEW 14**, 개별 영상 74개는 **IN 37 / OUT 10 / REVIEW 27**이다. 이 수치는 원본을 다시 본 AI 보조 검수와 시점 연결 결과이며 새 영상 자동 모델 정확도가 아니다.

## 15쌍 전체의 원인과 조치

| 영상 쌍 | 재검수에서 확인한 원인·수정 | 결과 | 남은 조치 |
|---|---|---|---|
'''
    clean=lambda value:str(value or '').replace('|','/').replace('\n',' ')
    for row in rows:
        text+='| '+' | '.join(map(clean,[row['pair_key'].replace('{view}','A/B'),row['summary_ko'],row['after'],row['next_action_ko']]))+' |\n'
    text+='''
## 놓쳤던 착지의 복구

- 일반 A8: 원본 119–121프레임의 첫 접촉은 복식 코트 안쪽이다. 이후 반등해 바깥으로 이동한다. 이전에는 예상 영역 위주로 확대해 오른쪽 첫 접촉을 놓쳤다. 접촉 영역과 외곽선의 오차를 계산한 IN을 같은 사건으로 확인한 B8에 연결했다.
- UHD/HD A3: 원본 137–138프레임에 먼저 바닥에 닿고 139–140프레임에 반등한다. 이후 지나가는 사람의 가림·회수 장면을 접촉 구간으로 잘못 본 기록을 정정했다. 서로 다른 세 동작이 일치하는 B3에 A3의 IN 근거를 연결했다.
- 일반 B3: 첫 접촉이 클립 안의 163–164프레임에 있었다. 기존의 ‘접촉 전 영상 종료’를 정정했다. 접촉과 실제 칠 경계의 오차가 겹쳐 REVIEW는 유지한다. 일반 B4의 마지막 동작도 타격이 아닌 얼굴을 만지는 행동이어서 종료 사유를 바로잡았다.

## 프로그램 개선

1. **사건 연결과 접촉 판정 분리:** 실제 동작 대응이 확인돼도 상대 시점에 IN/OUT이 없으면 사건 연결까지 실패하던 결합을 제거했다. 같은 랠리임을 확인한 것만으로 이진 판정을 만들지는 않는다. A16/B16 및 A18/B18의 기존 불일치는 계속 차단한다.
2. **원본 집중 재분석:** 저장된 원본 해시와 일치하고 실제 검수한 프레임의 AI 후보 구간에 원본 탐색 창과 후보 예산을 우선 배정한다. 밝은 바닥에 어두운 코르크가 합쳐져 제거되는 문제도 해당 모드에서 보완했다. 확인하지 않은 위치·프레임은 힌트로 쓰지 않는다.
3. **후속 단계의 후보 보존:** 집중 구간 후보가 착지 후보 상한에서 다시 밀려나던 문제를 수정했다. 원본 접촉 근거와 미리보기도 해당 후보를 먼저 제공한다. 점수, 대상 식별, 최초 접촉 확정 조건은 올리지 않는다.
4. **REVIEW 진단 화면:** 가림·라인 관계·대상 혼동·시점 불일치를 시점별로 표시하고 검토 구간으로 이동할 수 있다. 유효한 원본과 후보 근거가 있을 때 `접촉 후보 구간 집중 재분석`을 실행한다. 기존 AI 근거를 참고한 분석임을 명시한다.

## 실제 검증과 한계

- 전체 소프트웨어 테스트 **413개 통과**. 모든 37쌍을 실제 API로 재검증했다.
- 같은 원본 7개 영상의 기존 AI 표시점 28개·163프레임에서 후보 64개와 원본 탐색 창 8개를 동일하게 제한했다. 6px 이내 후보 보존은 기본 **5/28 → 집중 탐색 23/28**이었다. 표시점 자체를 검색 영역에 사용한 개발 실험이므로 독립 재현율·자동 정확도가 아니다. 어두운 후보만 켠 경우 3/28로 나빠져 기본 설정을 변경하지 않았다.
- 실제 UHD/4K B3 재분석에서 검토 영역의 착지 후보는 **0→5개**, 해당 원본 연속 근거도 5개로 늘었다. 후보 12개와 원본 근거 6개의 상한은 유지했다. 단일 영상 실행 25.79초, 최종 결과 REVIEW였다. 이는 A/B 전체 작업 시간이나 GPU 성능 측정이 아니다.
- 검수로 찾아낸 IN을 자동 모델의 성공으로 집계하지 않았다. 양 시점에서 접촉이 가려지거나 코르크·칠 경계가 흐리면 추가 시점 또는 더 짧은 노출의 원본이 필요하다. 이를 임의 IN/OUT으로 바꾸지 않았다.

기계 판독 감사는 `runs/review-remediation/verification/15-pair-audit.json`, 쌍별 실제 작업은 `all-pairs.json`, 후보 비교는 `runs/review-remediation/pipeline/candidate-benchmark.json`에 있다. 기존 검수·결과의 변경 전 사본은 `runs/review-remediation/baseline/`에 보관했다. 원본 영상과 과거 자동 분석 결과는 변경하지 않았다.
'''
    (root/'docs/REVIEW_REMEDIATION.md').write_text(text,encoding='utf-8')
    print(json.dumps({'passed':True,'original15':counts,'all37':jobs['counts']},ensure_ascii=False))


if __name__=='__main__':main()
