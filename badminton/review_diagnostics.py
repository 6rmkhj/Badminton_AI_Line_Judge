"""Explain blockers without changing a judgment or manufacturing missing evidence."""
from __future__ import annotations

import math

# Status comes from inspected source evidence, never from an automatic guess.
OBSERVATIONS = {
    'contact_observed': ('최초 접촉 관측', '관측한 접촉점과 유효한 외곽선의 관계를 확인합니다.'),
    'contact_interval': ('접촉 구간 관측', '접촉 전후 연속 프레임에서 가능한 접촉 영역과 칠 경계의 오차를 좁힙니다.'),
    'line_ambiguous': ('접촉과 라인 관계 불명확', '원본 접촉 구간을 확대하고 실제 복식 외곽선의 바깥 칠 경계를 확인합니다. 오차 영역이 선을 가로지르면 더 선명한 시점이 필요합니다.'),
    'contact_occluded': ('선수·기둥에 접촉이 가려짐', '대응 동작으로 반대 시점의 같은 구간을 찾아 확인합니다. 양쪽에서 가려졌다면 추가 시점이 필요합니다.'),
    'contact_out_of_view': ('최초 접촉을 화면에서 확인 못함', '반대 시점의 같은 사건을 확인합니다. 현재 두 시점 모두 접촉 영역을 담지 못했다면 추가 원본이 필요합니다.'),
    'no_contact_in_clip': ('접촉 전 클립 종료', '마지막 타격 이후 최초 바닥 접촉까지 포함한 원본 후속 구간을 입력합니다.'),
    'target_ambiguous': ('경기 셔틀과 다른 물체 구분 불가', '마지막 유효 타격부터 연속으로 추적해 예비 셔틀·회수 동작과 구분하고 반대 시점에서 대조합니다.'),
}

JOINT = {
    'SAME_EVENT_UNCONFIRMED': ('같은 사건 연결 미확인', '파일 번호만으로 같은 착지 사건임을 확인할 수 없습니다.', '두 시점에 공통으로 보이는 서로 다른 동작 두 개를 대조합니다. 이미 같은 랠리임을 확인했다면 입력의 확인 항목을 사용합니다.', 'correspondence', True),
    'KNOWN_DIFFERENT_RALLY': ('서로 다른 랠리', '기존 원본 동작 대조에서 이 조합은 다른 사건으로 확인됐습니다.', '해당 랠리의 올바른 반대 시점으로 교체합니다. 확인 체크만으로 기존 불일치를 무시하지 않습니다.', 'input', False),
    'DUPLICATE_SOURCE': ('동일 영상 중복 입력', 'A와 B 파일 내용의 SHA256이 같습니다.', '서로 다른 카메라에서 촬영한 원본을 입력합니다.', 'input', False),
    'CONFLICTING_VIEW_DECISIONS': ('시점별 직접 판정 충돌', '양쪽에 판정 근거가 있지만 IN/OUT이 서로 다릅니다.', '같은 최초 접촉인지, 코트·경기 규칙과 칠 경계가 같은 의미인지 원본으로 재검수합니다.', 'conflict', None),
    'VIEW_ANALYSIS_FAILED': ('영상 처리 실패', '한쪽 이상의 영상 처리가 완료되지 않았습니다.', '시점별 오류를 확인하고 원본·FFmpeg 설치·입력 제한을 점검한 뒤 다시 실행합니다.', 'processing', None),
    'SOURCE_CHANGED': ('원본 검증 실패', '저장된 검수와 현재 입력의 내용이 일치하는지 확인하지 못했습니다.', '검수 당시 원본을 입력하거나 새 입력을 별도로 분석합니다.', 'input', None),
    'RULES_MISMATCH': ('경기 규칙 불일치', '두 시점의 판정에 적용된 경기 규칙이 다릅니다.', '같은 경기 규칙으로 양쪽 판정을 다시 계산합니다.', 'rules', True),
}


def _strings(value):
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [v for v in value if isinstance(v, str) and v.strip()] if isinstance(value, list) else []


def view_diagnostics(view):
    details = view.get('review_details') or {}
    status = details.get('observation_status')
    label, suggested = OBSERVATIONS.get(status, ('자동 분석 근거 미확정', '원본 접촉 구간과 코트 경계를 함께 확인합니다.'))
    qualified = view.get('qualified') is True and view.get('direct_decision') in {'IN', 'OUT'}
    reasons = set(view.get('reason_codes') or [])
    missing, actions = [], []
    if not qualified:
        if status in OBSERVATIONS:
            missing.append(label)
            actions = _strings(details.get('follow_up_actions')) or [suggested]
        else:
            if reasons & {'CALIBRATION_NOT_ACCEPTED', 'CALIBRATION_SUPPORT_INSUFFICIENT', 'CALIBRATION_NOT_CONFIRMED'}:
                missing.append('목표 코트와 라인 보정 미확정')
                actions.append('경기 코트의 유효 외곽선을 원본에서 지정하고, 자동 보정이 옆 코트나 다른 종목의 선을 선택했는지 확인합니다.')
            if any('IDENTITY' in reason or 'TARGET_' in reason for reason in reasons):
                missing.append('대상 셔틀의 연속 추적 미확정')
                actions.append('마지막 타격부터 후보를 추적하고 정지 물체·예비 셔틀을 제외합니다.')
            if reasons & {'FIRST_CONTACT_NOT_OBSERVED', 'GROUND_CONTACT_NOT_OBSERVED', 'CONTACT_EVIDENCE_UNCERTAIN'}:
                missing.append('최초 바닥 접촉 미확정')
                actions.append('정지 위치 대신 하강·최초 접촉·반동을 포함한 원본 연속 프레임을 확인합니다.')
            if 'MULTIPLE_LANDING_CANDIDATES' in reasons:
                missing.append('착지 후보가 여러 개임')
                actions.append('같은 랠리의 마지막 타격과 이어지는 후보인지 반대 시점에서 대조합니다.')
            if any('GEOMETRY' in reason or 'BOUNDARY' in reason for reason in reasons):
                missing.append('접촉 오차와 칠 경계 분리 불가')
                actions.append('원본 접촉 영역과 실제 바깥 칠 경계를 다시 측정합니다.')
            if not missing:
                missing.append('판정에 필요한 근거 부족' if view.get('status') != 'failed' else '영상 처리 실패')
                actions.append(view.get('error') or suggested)
    focus = details.get('focus_time_seconds')
    if not isinstance(focus, (int, float)) or isinstance(focus, bool) or not math.isfinite(focus) or focus < 0:
        focus = None
    return {'observation_status': status, 'observation_label': label if status else ('직접 판정 근거 확보' if qualified else label),
            'missing_requirements': list(dict.fromkeys(missing)), 'follow_up_actions': list(dict.fromkeys(actions)),
            'rationale': view.get('rationale', ''), 'focus_time_seconds': focus,
            'focus_is_confirmed_contact': bool(qualified and view.get('contact_time_seconds') is not None),
            'evidence_source': view.get('review_source', 'unknown')}


def pair_diagnostics(result):
    views = {camera: view_diagnostics(view) for camera, view in result.get('views', {}).items()}
    issues = []
    if result.get('decision') == 'REVIEW':
        for code in result.get('reason_codes', []):
            if code in JOINT:
                title, detail, action, category, possible = JOINT[code]
                issues.append({'code': code, 'title': title, 'detail': detail, 'camera_ids': ['A','B'],
                               'remediation': action, 'category': category, 'resolvable_with_existing_inputs': possible})
        for camera, detail in views.items():
            if not detail['missing_requirements']:
                continue
            status = detail['observation_status']
            issues.append({'code': 'VIEW_EVIDENCE_INCOMPLETE', 'title': f'{camera} 시점 근거 부족',
                           'detail': ' · '.join(detail['missing_requirements']), 'camera_ids': [camera],
                           'remediation': detail['follow_up_actions'], 'category': 'observation',
                           'resolvable_with_existing_inputs': False if status == 'no_contact_in_clip' else None})
    return {'schema_version': 1, 'summary': ('A/B에서 아래 조건이 충족되지 않아 판정을 보류했습니다.' if issues
                                           else '현재 입력에서 최종 판정에 필요한 근거를 확보했습니다.'),
            'issues': issues, 'views': views,
            'note': '진단과 후속 조치는 관측 근거를 설명하며 판정 기준을 완화하지 않습니다.'}
