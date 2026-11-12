"""Snapshot blinded references, then compact explicitly released final audits.

The snapshot command copies reference bytes without parsing reference contents.
The finalize command requires an explicit holdout-release flag and preserves the
auditor's strict category; a near miss is never converted to a matching case.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil


HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / 'GENERALIZATION_VALIDATION.md'
BASELINE = '7e74b058ae95543bb9c7d121761426462e74f49c'
VIDEO_FIELDS = ('key', 'file', 'group', 'number', 'view', 'split', 'git_blob', 'bytes', 'sha256')
ROW_FIELDS = ('key', 'group', 'view', 'split', 'source_sha256', 'reference_sha256',
              'contact_visibility', 'reference_interval', 'reference_point', 'coordinate_size',
              'tolerance_px', 'frame_tolerance', 'effective_frame_tolerance', 'localization_scorable',
              'status', 'implementation_fingerprint', 'analysis_config_sha256', 'pipeline_version', 'selection_category',
              'automatic_decision', 'decision_abstention', 'candidate_abstention',
              'selected_comparison', 'selected', 'elapsed_seconds',
              'validation_mode', 'validation_run', 'timing_scope')
DIAGNOSIS_FIELDS = ('failure_stage', 'causal_failure_category', 'reference_trajectory_point_count',
                    'detection_artifact_available',
                    'raw_trajectory_hit_count', 'best_track_trajectory_hit_count',
                    'selected_track_trajectory_hit_count', 'selected_track_has_flight_hypothesis')
SELECTED_FIELDS = ('event_id', 'track_id', 'frame_index', 'time', 'candidate_px',
                   'contact_observed', 'method', 'reason_codes')
COMPARISON_FIELDS = ('frame_index', 'reference_coordinate_point', 'distance_px', 'frame_error',
                     'within_spatial_tolerance', 'within_frame_tolerance', 'matches_first_vicinity')
CRITERION_FIELDS = ('contact_visibility', 'reference_interval', 'reference_point', 'coordinate_size',
                    'tolerance_px', 'frame_tolerance', 'effective_frame_tolerance', 'localization_scorable')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_manifest(manifest):
    rows = manifest['videos']
    keys = [row['key'] for row in rows]
    if manifest.get('source_commit') != BASELINE:
        raise ValueError('Unexpected manifest baseline source commit')
    if len(rows) != 24 or len(set(keys)) != 24:
        raise ValueError('Manifest must declare exactly 24 unique input keys')
    if any(not isinstance(key, str) or Path(key).name != key or key in ('.', '..') or '\\' in key for key in keys):
        raise ValueError('Input keys must be safe single path components')
    if Counter(row['split'] for row in rows) != {'development': 16, 'holdout': 8}:
        raise ValueError('Manifest split must contain 16 development and 8 holdout inputs')
    return {row['key']: row for row in rows}


def validate_inventory(document, declared):
    rows = document['references']
    inventory = {row['key']: row for row in rows}
    if len(rows) != len(inventory) or set(inventory) != set(declared):
        raise ValueError('Reference inventory must cover exactly the declared unique input keys')
    for key, row in inventory.items():
        if row['split'] != declared[key]['split'] or row['path'] != f'references/{key}.json':
            raise ValueError('Reference inventory split/path mismatch: ' + key)
    return inventory


def snapshot(args):
    source = load(args.manifest)
    manifest = {'schema_version': 1, 'source_commit': source['source_commit'],
                'selection': source['selection'], 'reference_policy': source['reference_policy'],
                'videos': [{key: row[key] for key in VIDEO_FIELDS} for row in source['videos']]}
    validate_manifest(manifest)
    write(HERE / 'manifest.json', manifest)
    inventory = []
    for row in manifest['videos']:
        key = row['key']
        original = args.references / (key + '.json')
        target = HERE / 'references' / original.name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(original)
        # A later snapshot must not silently replace a frozen reference.
        if target.exists() and sha(target) != digest:
            raise ValueError(f'Frozen reference changed: {key}; resolve the reference history explicitly')
        shutil.copyfile(original, target)
        inventory.append({'key': key, 'split': row['split'], 'path': 'references/' + target.name,
                          'bytes': target.stat().st_size, 'sha256': digest})
    tools = []
    for original in args.tool:
        target = HERE / 'reproduction_tools' / original.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        tools.append({'path': 'reproduction_tools/' + target.name, 'sha256': sha(target)})
    write(HERE / 'inventory.json', {'status': 'frozen_reference_snapshot_metrics_pending',
          'reference_contents_parsed_by_snapshot': False, 'references': inventory, 'tools': tools})
    print('Copied 24 byte-preserved reference files without parsing their contents; metrics remain pending.')


def compact_audits(paths, declared, inventory):
    records = {}
    for path in paths:
        report = load(path)
        if report.get('manifest_source_commit') != BASELINE:
            raise ValueError(f'Unexpected baseline source commit in {path}')
        split = report.get('split')
        if split not in {'development', 'holdout'}:
            raise ValueError(f'Audit must declare its development or holdout split: {path}')
        if split == 'holdout':
            release = report.get('holdout_release') or {}
            if (report.get('holdout_predictions_or_references_opened') is not True
                    or not re.fullmatch(r'[0-9a-f]{40}', str(release.get('run_commit', '')))
                    or not re.fullmatch(r'[0-9a-f]{64}', str(release.get('run_implementation_fingerprint', '')))
                    or not re.fullmatch(r'[0-9a-f]{64}', str(release.get('freeze_manifest_sha256', '')))):
                raise ValueError(f'Holdout audit lacks verified saved-freeze provenance: {path}')
        for original in report['videos']:
            key = original['key']
            if key not in declared or key in records:
                raise ValueError(f'Unexpected or duplicate audit key: {key}')
            if original.get('source_sha256') != declared[key]['sha256']:
                raise ValueError(f'Input SHA mismatch: {key}')
            if original.get('reference_sha256') != inventory[key]['sha256']:
                raise ValueError(f'Frozen reference SHA mismatch: {key}')
            if original.get('split') != split or split != declared[key]['split']:
                raise ValueError(f'Audit split differs from the predeclared split: {key}')
            if split == 'holdout' and original.get('status') == 'audited' and (
                    original.get('implementation_fingerprint') != release['run_implementation_fingerprint']):
                raise ValueError(f'Holdout row differs from its verified code freeze: {key}')
            row = {name: original[name] for name in ROW_FIELDS if name in original}
            for name, fields in [('selected', SELECTED_FIELDS), ('selected_comparison', COMPARISON_FIELDS)]:
                value = original.get(name)
                row[name] = {field: value[field] for field in fields if field in value} if value else None
            row['diagnosis'] = {name: original['diagnosis'][name] for name in DIAGNOSIS_FIELDS
                                if name in (original.get('diagnosis') or {})}
            records[key] = row
    if set(records) != set(declared):
        raise ValueError('Final comparison requires explicit records for all 24 predeclared inputs')
    return records


def tally(rows):
    audited = [row for row in rows if row['status'] == 'audited']
    scorable = [row for row in audited if row.get('localization_scorable')]
    covered = [row for row in audited if (row.get('diagnosis') or {}).get('reference_trajectory_point_count', 0) > 0]
    raw_covered = [row for row in covered if isinstance(row['diagnosis'].get('raw_trajectory_hit_count'), int)]
    return {'declared': len(rows), 'audited': len(audited), 'pending_or_invalid': len(rows) - len(audited),
            'first_vicinity_matches': sum(row.get('selection_category') == 'correct_first_vicinity' for row in scorable),
            'first_vicinity_denominator': len(scorable),
            'automatic_decisions': dict(Counter(row.get('automatic_decision') for row in audited)),
            'reference_trajectory_points': sum(row['diagnosis']['reference_trajectory_point_count'] for row in covered),
            'raw_trajectory_hits': sum(row['diagnosis']['raw_trajectory_hit_count'] for row in raw_covered),
            'raw_reference_trajectory_points': sum(row['diagnosis']['reference_trajectory_point_count'] for row in raw_covered),
            'best_single_track_trajectory_hits': sum(row['diagnosis'].get('best_track_trajectory_hit_count', 0) for row in covered),
            'selected_track_trajectory_hits': sum(row['diagnosis'].get('selected_track_trajectory_hit_count', 0) for row in covered),
            'independent_contact_accuracy': None, 'independent_line_judgment_accuracy': None}


def ratio(count, total):
    return f'{count}/{total}' if total else '측정 대상 없음'


def selection_text(row):
    category = row.get('selection_category', row['status'])
    selected = row.get('selected') or {}
    return category + (f" · f{selected['frame_index']}" if selected.get('frame_index') is not None else '')


def finalize(args):
    if not args.release_holdouts:
        raise ValueError('--release-holdouts is required after the algorithm and references have been frozen')
    if not re.fullmatch(r'[0-9a-f]{40}', args.final_revision):
        raise ValueError('Final revision must be a full saved source commit SHA')
    manifest, inventory_document = load(HERE / 'manifest.json'), load(HERE / 'inventory.json')
    declared = validate_manifest(manifest)
    inventory = validate_inventory(inventory_document, declared)
    if any(key not in declared for key in args.borderline):
        raise ValueError('Unknown borderline key')
    if args.example and (not args.baseline_run or not args.final_run or len(args.example) > 8):
        raise ValueError('Examples require both run directories and at most 8 explicitly selected keys')
    if any(key not in declared for key in args.example):
        raise ValueError('Unknown example key')
    for item in inventory.values():
        if sha(HERE / item['path']) != item['sha256']:
            raise ValueError('Reference snapshot bytes changed: ' + item['key'])
    before = compact_audits(args.baseline_audit, declared, inventory)
    after = compact_audits(args.final_audit, declared, inventory)
    for key in declared:
        for field in CRITERION_FIELDS:
            if before[key].get(field) != after[key].get(field):
                raise ValueError(f'Before/after criterion changed: {key}.{field}')
    fingerprints = {row.get('implementation_fingerprint') for row in after.values() if row['status'] == 'audited'}
    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError('All final audited outputs must share one frozen implementation fingerprint')
    receipt_paths = sorted(HERE.glob('holdout-inspection-release-*.json'))
    receipt = load(receipt_paths[-1]) if receipt_paths else None
    if receipt:
        freeze_path = HERE / 'freeze-release.preinspection.json'
        if (receipt.get('source_commit') != args.final_revision
                or receipt.get('implementation_fingerprint') != next(iter(fingerprints))
                or receipt.get('full_fresh') is not True or receipt.get('holdout_count') != 8
                or sha(freeze_path) != receipt.get('preserved_preinspection_sha256')
                or receipt.get('reference_or_prediction_files_opened_before_receipt') is not False):
            raise ValueError('Holdout release receipt differs from the preserved preinspection freeze')
    controls_path = HERE / 'control-regression.json'
    controls = load(controls_path) if controls_path.is_file() else None
    if controls:
        if (controls.get('implementation_fingerprint') != next(iter(fingerprints))
                or controls.get('source_commit') != args.final_revision):
            raise ValueError('Original-control regression must use the same frozen final source')
        for record in controls['views'].values():
            for artifact in record.get('artifacts', {}).values():
                path = HERE / artifact['path']
                if not path.resolve().is_relative_to(HERE.resolve()) or sha(path) != artifact['sha256']:
                    raise ValueError('Original-control evidence changed or escaped the report directory')
    software_path = HERE / 'software-validation.json'
    software = load(software_path) if software_path.is_file() else None
    if software:
        if software.get('source_commit') != args.final_revision:
            raise ValueError('Software validation must use the same frozen final source')
        browser_path = HERE / software['browser_report']
        if (not browser_path.resolve().is_relative_to(HERE.resolve())
                or sha(browser_path) != software['browser_report_sha256']):
            raise ValueError('Browser validation record changed or escaped the report directory')
        browser = load(browser_path)
        if (browser.get('code_commit') != args.final_revision or browser.get('status') != 'passed'
                or browser.get('javascript_errors') != [] or browser.get('failed_requests') != []
                or {item.get('width') for item in browser.get('mobile', [])
                    if item.get('horizontal_overflow') is False} != {375, 390}
                or software.get('python_tests', {}).get('status') != 'passed'
                or software.get('python_tests', {}).get('process_exit_code') != 0):
            raise ValueError('Software records do not support the passing validation summary')
    run_profiles = {}
    for label, directory in [('baseline', args.baseline_run), ('final', args.final_run)]:
        if directory is None:
            continue
        config_path = directory / 'run-config.json'
        profile = load(config_path)
        expected_fingerprints = {row.get('implementation_fingerprint') for row in
                                 (before if label == 'baseline' else after).values() if row['status'] == 'audited'}
        if profile.get('implementation_fingerprint') not in expected_fingerprints or len(expected_fingerprints) != 1:
            raise ValueError(f'{label} run profile differs from audited implementation fingerprint')
        profile_hash = hashlib.sha256(json.dumps(profile['config'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if any(row.get('analysis_config_sha256') != profile_hash for row in
               (before if label == 'baseline' else after).values() if row['status'] == 'audited'):
            raise ValueError(f'{label} audited outputs do not share the captured analysis configuration')
        run_profiles[label] = {'run_config_sha256': sha(config_path),
                               'analysis_config_sha256': profile_hash,
                               'implementation_fingerprint': profile['implementation_fingerprint'],
                               'config': profile['config']}
    summary = {}
    for split in ('development', 'holdout'):
        keys = [key for key, value in declared.items() if value['split'] == split]
        summary[split] = {'before': tally([before[key] for key in keys]), 'after': tally([after[key] for key in keys])}
    comparison = {'status': 'final_after_explicit_holdout_release', 'baseline_revision': BASELINE,
        'final_revision': args.final_revision, 'implementation_fingerprint': next(iter(fingerprints)),
        'run_profiles': run_profiles,
        'reference_policy': manifest['reference_policy'], 'summary': summary,
        'videos': [{'key': key, 'split': declared[key]['split'], 'before': before[key], 'after': after[key]}
                   for key in declared]}
    execution_modes = {split: dict(Counter(after[key].get('validation_mode', 'full_pipeline')
                         for key, value in declared.items() if value['split'] == split))
                       for split in ('development', 'holdout')}
    if execution_modes['holdout'].get('detector_stage_reuse'):
        raise ValueError('Final holdouts require full fresh detector execution')
    comparison['execution_modes_after'] = execution_modes
    comparison['audit_input_sha256'] = {
        label: [{'file': Path(path).name, 'sha256': sha(path)} for path in paths]
        for label, paths in [('baseline', args.baseline_audit), ('final', args.final_audit)]}
    if receipt:
        comparison['holdout_inspection_release'] = {'receipt': receipt_paths[-1].name,
            'receipt_sha256': sha(receipt_paths[-1]), 'released_at': receipt['released_at'],
            'preinspection_freeze': 'freeze-release.preinspection.json',
            'preinspection_freeze_sha256': receipt['preserved_preinspection_sha256']}
    visual_reviews = []
    for filename in ('holdout-independent-visual-review.json', 'holdout-supplemental-visual-review.json'):
        path = HERE / filename
        if path.is_file():
            review = load(path)
            if (not receipt or review.get('production_commit') != args.final_revision
                    or review.get('release_receipt_sha256') != sha(receipt_paths[-1])):
                raise ValueError('Post-release visual review does not match this frozen run')
            visual_reviews.append({'path': filename, 'sha256': sha(path)})
    comparison['post_release_visual_reviews'] = visual_reviews
    lines = ['# 추가 영상의 추적·첫 착지 부근 검증 — ' + args.version, '',
      '알고리즘과 시각 참조를 동결한 뒤 공개한 추가 영상 비교입니다. 후보의 첫 착지 **부근** 일치와 물리적 바닥 접촉·IN/OUT 정확도는 서로 다른 항목입니다.', '',
      '## 입력과 분리', '',
      '- 사전 선정한 추가 영상 24개: 개발 16개, 홀드아웃 8개. 기존 A1/B1 수정 영상은 제외했습니다.',
      '- 24개 파일은 12쌍의 A/B 촬영 시점입니다. 집계 단위는 영상 파일이며, 같은 사건을 본 두 시점을 독립된 사건 두 개로 해석하지 않습니다.',
      '- 4개 촬영 파일명 그룹에서 각각 개발 2쌍과 홀드아웃 1쌍을 선정했습니다. 같은 프로젝트·체육관 자료이며 독립 체육관 일반화 시험이 아닙니다.',
      '- 소스 영상부터 AI가 시각 참조를 작성했습니다. 독립 사람이 표시한 정답이 아니며 실제 접촉·라인 판정 정확도는 미측정입니다.',
      '- 수정 전 소스: `' + BASELINE + '`. 수정 후 소스: `' + args.final_revision + '`.',
      '- [입력 SHA256·분할](generalization/manifest.json), [동결 참조 해시](generalization/inventory.json), [압축 비교 결과](generalization/comparison.json).', '',
      '## 발견한 실패 원인과 보완', '',
      '- 희미한 셔틀콕이 탐지 후보 제한에서 밀리면 다음 원본 확대 영역도 놓쳤습니다. 실제 관측의 짧은 이동 이력을 이용해 기존 후보·확대 영역 예산 일부를 예약하고, 그 좁은 원본 영역에서만 약한 명암 변화를 추가 검사합니다.',
      '- 누락 구간 뒤의 연결 범위가 고정 거리라 빠른 낙하를 놓쳤습니다. 경과 시간과 관측 속도를 반영하되, 검출되지 않은 좌표를 실제 관측처럼 채우지 않습니다.',
      '- 선수의 몸·신발로 옮겨 붙거나 한 번 튄 뒤 정지한 경로를 반동으로 높게 평가했습니다. 가림 전후 외형 변화와 여러 실제 관측의 반동 방향·일관성을 함께 검사합니다.',
      '- 긴 배경 경로와 늦게 이어진 반동 대안이 짧은 첫 낙하를 밀어냈습니다. 공간별 후보 여유를 확보하고, 낙하 전후 곡선이 연결될 수 있는 이른 반동 대안을 먼저 검토합니다.',
      '- 관측이 끊긴 앞뒤를 하나의 연속 근거처럼 세면 몸·라켓으로 옮겨 붙은 경로를 놓쳤습니다. 신원 변화와 반동 검사는 누락 구간 뒤에서 실제로 이어지는 관측 구간을 기준으로 다시 평가합니다.',
      '- 반동처럼 보이는 갈래 외에 실제 관측으로 계속 진행하는 대안이 있는지 함께 검사하고, 후보 크기가 극단적으로 줄어드는 연결을 재검토해 손·라켓 갈래가 우선되는 경우를 줄였습니다. 이 검사로 모든 오연결이 해결되지는 않았습니다.',
      '- 이 검사는 후보의 근거를 보완합니다. 같은 코트의 셔틀콕이라는 확인이나 실제 코르크·바닥 접촉 관측을 대신하지 않습니다.', '',
      '## 엄격한 첫 착지 부근 비교', '',
      '참조에 처음부터 기록한 프레임 구간과 픽셀 오차를 모두 만족해야 일치로 셉니다. 원본 PTS 시작값이 영상마다 다르므로 이 집계는 정규화된 재생 시간과 원본 시간을 직접 빼지 않고 같은 원본 프레임 번호로 비교합니다. 가림·모호한 참조는 위치 일치 분모에서 제외하고 별도로 남깁니다. 근소한 실패를 나중에 일치로 바꾸지 않습니다.', '',
      '| 분할 | 완료 감사 전→후 | 엄격한 첫 부근 일치 전→후 | 탐지 후보의 참조 경로점 포함 전→후 | 최선 단일 경로의 참조점 포함 전→후 | 선택된 경로의 참조점 포함 전→후 |',
      '|---|---|---|---|---|---|']
    replay_count = execution_modes['development'].get('detector_stage_reuse', 0)
    if receipt:
        position = lines.index('## 발견한 실패 원인과 보완')
        lines[position:position] = [f"[공개 전 동결 기록](generalization/freeze-release.preinspection.json)과 [별도 공개 영수증](generalization/{receipt_paths[-1].name})을 보존했습니다. 새 홀드아웃 8개가 모두 완료되고 코드·설정·입력 해시가 일치한 뒤에만 결과와 참조를 열었습니다.", '']
    if replay_count:
        position = lines.index('## 발견한 실패 원인과 보완')
        lines[position:position] = ['## 실행 범위와 탐지 관측 재사용', '',
          f'- 개발 {replay_count}개는 탐지 코드·설정이 같은 이전 전체 실행의 실제 탐지 관측을 재사용했습니다. 원본 SHA256, 탐지 파일 해시, 프레임 수·정확한 타임스탬프·전처리·설정 및 이전/현재 코드 지문을 확인했습니다.',
          '- 원본 분석 프레임 읽기, 코트 확인, 궤적 연결, 이벤트 선택, 원본 확대 근거 생성은 새 코드로 다시 실행했습니다. 개발 영상 전체에 대해 원본 고해상도 탐지를 새로 실행했다고 주장하지 않습니다.',
          '- 홀드아웃 8개와 기존 A1/B1 회귀 영상은 탐지를 포함한 전체 새 실행입니다. 영역 보조 실험은 이 자동 결과에 포함하지 않습니다.',
          '- 재사용 실행 시간은 전체 분석 속도 개선의 근거로 쓰지 않습니다. 원래 실행에는 라이브러리 버전이 별도로 남아 있지 않아 과거 런타임 환경의 완전한 동일성까지 입증하지는 못합니다.', '']
        lines[position:position] = ['[탐지 단계 재사용 기록](generalization/detector-replay-provenance.json), [개발 엄격 비교·회귀 점검](generalization/development-quality-scoreboard.json), [별도 시각 검토 기록](generalization/development-quality-ledger.json)을 함께 제공합니다. 경계 사례의 시각 해석은 엄격 일치 수를 변경하지 않습니다.', '']
    for split, values in summary.items():
        b, a = values['before'], values['after']
        lines.append(f"| {'개발' if split == 'development' else '홀드아웃'} | {b['audited']}→{a['audited']} | {ratio(b['first_vicinity_matches'], b['first_vicinity_denominator'])}→{ratio(a['first_vicinity_matches'], a['first_vicinity_denominator'])} | {ratio(b['raw_trajectory_hits'], b['raw_reference_trajectory_points'])}→{ratio(a['raw_trajectory_hits'], a['raw_reference_trajectory_points'])} | {ratio(b['best_single_track_trajectory_hits'], b['reference_trajectory_points'])}→{ratio(a['best_single_track_trajectory_hits'], a['reference_trajectory_points'])} | {ratio(b['selected_track_trajectory_hits'], b['reference_trajectory_points'])}→{ratio(a['selected_track_trajectory_hits'], a['reference_trajectory_points'])} |")
    inferred_matches = [key for key, row in after.items() if row.get('selection_category') == 'correct_first_vicinity'
                        and 'occluded' in str((row.get('selected') or {}).get('method', ''))]
    if inferred_matches:
        lines += ['', '엄격한 부근 일치 중 ' + ', '.join('`' + key + '`' for key in inferred_matches)
                  + '는 누락된 관측 사이를 낙하·반동 경로로 근사한 위치입니다. 실제 관측점과 추정 중심을 구분하며, 해당 프레임의 물리적 접촉을 직접 관측했다는 뜻이 아닙니다.']
    lines += ['', '경로점 포함은 접촉 위치를 평가할 수 있는 참조 영상에서 AI가 표기한 일부 시점의 근처 후보가 있는지를 확인한 값입니다. 전체 프레임 추적률, 대상 신원 확인, 실제 착지 정확도를 뜻하지 않습니다. 최선 단일 경로와 실제 화면에 선택된 대표 경로의 포함률을 구분합니다.', '',
      '## 후보 선택과 판정 보류', '', '| 영상 | 분할 | 접촉 참조 가시성 | 수정 전 | 수정 후 | 자동 판정 |', '|---|---|---|---|---|---|']
    for key in declared:
        b, a = before[key], after[key]
        lines.append(f"| {key} | {declared[key]['split']} | {a.get('contact_visibility', '미확정')} | {selection_text(b)} | {selection_text(a)} | {a.get('automatic_decision', a['status'])} |")
    lines += ['', '`REVIEW`는 자동 판정 보류입니다. 근처 후보가 맞아도 물리적 접촉과 코트 보정 조건을 만족하지 않으면 IN/OUT을 확정하지 않습니다.', '']
    if visual_reviews:
        links = ', '.join(f"[검토 기록 {i + 1}](generalization/{item['path']})" for i, item in enumerate(visual_reviews))
        lines += ['홀드아웃 공개 뒤 대표 성공·실패의 원본과 결과를 별도 AI 시각 검토로 확인했습니다. 이 검토는 참조·허용 오차·성공 수를 변경하지 않으며, 독립 사람 정답을 대신하지 않습니다. ' + links + '.', '']
    if controls:
        lines += ['## 기존 A1/B1 회귀 확인 — 추가 24개 집계에서 제외', '',
          '처음 수정에 사용한 두 영상도 같은 동결 소스로 다시 실행했습니다. 원래의 AI 근사 참조 범위를 유지한 개발 회귀 검사이며, 새 영상의 일반화 성능이나 독립 접촉 정확도에 더하지 않습니다.', '',
          '| 원본 시점 | 첫 반동 후보 프레임 | 분석 화면 좌표 | AI 근사 참조와 거리 / 기존 허용 | 기존 범위 안 | 자동 판정 |',
          '|---|---|---|---|---|---|']
        for view, record in controls['views'].items():
            lines.append(f"| {view} | {record['selected_frame']} | {record['selected_px']} | {record['point_discrepancy_px']}px / {record['reference']['radius']}px | {record['within_approximate_reference']} | {record['decision']} |")
        lines += ['', '두 결과 모두 수동 위치 보조 없이 계산했고, 선택 궤적은 실제 탐지 관측으로만 구성했습니다. 누락 프레임은 남아 있으며 실제 코르크·바닥 접촉은 확인되지 않았습니다. 실행 시간은 CPU를 공유한 환경의 측정값이므로 속도 개선 근거로 쓰지 않습니다. [별도 회귀 기록](generalization/control-regression.json).', '',
          '입력 출처도 구분합니다. A1은 첨부 전송 도구의 32MiB 제한 때문에 저장소의 동명 원본을 사용했으며, 첨부 A와 바이트 동일성은 확인하지 못했습니다. B1은 실제 첨부 파일을 사용했고 저장소 원본과 해시가 일치했습니다. [원래 입력 범위와 검증 설명](SHUTTLE_TRACKING_FIX.md).', '']
        for view, record in controls['views'].items():
            for kind, artifact in record.get('artifacts', {}).items():
                lines += [f"![기존 {view} {'전체 장면' if kind == 'overview' else '원본 확대 경로'}](generalization/{artifact['path']})", '']
    missed = [row for row in after.values() if row.get('localization_scorable')
              and row['status'] == 'audited' and row.get('selection_category') != 'correct_first_vicinity']
    if missed:
        lines += ['## 남은 실패와 해석', '',
          '수정 후에도 아래 영상은 원래의 위치·프레임 기준을 충족하지 못했습니다. 실행 완료나 `REVIEW` 표시를 올바른 추적으로 세지 않습니다.', '',
          '| 영상 | 엄격 기준 결과 | 자동 진단상 남은 단계 |', '|---|---|---|']
        for row in missed:
            lines.append(f"| {row['key']} | {row.get('selection_category')} | {(row.get('diagnosis') or {}).get('failure_stage', '근거 부족')} |")
        lines += ['', '단계 진단은 저장된 탐지·궤적·후보와 시각 참조의 근접성을 비교한 결과입니다. 원본에서 계속 가려진 구간, 희미한 바닥 구간, 여러 코트의 동시 움직임은 여전히 대상 식별을 어렵게 합니다. 옆 코트의 실제 셔틀콕도 매끄러운 경로를 만들 수 있으므로 움직임이 자연스럽다는 이유만으로 대상 코트를 확정하지 않습니다.', '']
    if args.borderline:
        lines += ['## 경계 사례 — 엄격 일치와 별도', '', '다음 사례는 원래 오차 기준을 그대로 적용한 비교입니다. 설명을 위해 별도로 표시하며 합계의 성공 수를 늘리지 않습니다.', '']
        for key in args.borderline:
            row, comp = after[key], after[key].get('selected_comparison') or {}
            lines.append(f"- `{key}`: {row.get('selection_category')}, 거리 {comp.get('distance_px', '미측정')}px / 허용 {row.get('tolerance_px', '미정')}px, 구간 밖 프레임 오차 {comp.get('frame_error', '미측정')} / 추가 허용 {row.get('effective_frame_tolerance', 0)}.")
        lines.append('')
    if args.example:
        lines += ['## 같은 원본의 전후 근거', '', '표시는 후보 경로와 추정 위치입니다. 관측 누락과 가림이 남아 있으며 독립적인 바닥 접촉 확정을 뜻하지 않습니다.', '']
        for key in args.example:
            lines += ['### ' + key, '']
            for label, run, suffix in [('수정 전', args.baseline_run, 'before'), ('수정 후', args.final_run, 'after')]:
                for kind, filename, figure_label in [('overview', 'overview.jpg', '전체 장면'),
                                                     ('context', 'evidence/trajectory-context.jpg', '원본 확대 경로')]:
                    original = run / key / filename
                    if not original.is_file():
                        lines.append(label + ': 해당 실행에 ' + figure_label + ' 이미지가 없습니다.\n')
                        continue
                    relative = f'examples/{key}-{suffix}-{kind}.jpg'
                    target = HERE / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, target)
                    lines += [f'![{key} {label} {figure_label}](generalization/{relative})', '']
    if software:
        tests = software['python_tests']
        lines += ['## 소프트웨어·화면 확인', '',
          f"- 동결 소스의 Python 테스트 {tests['count']}개 통과. 테스트 도구가 보고한 실행 시간 {tests['reported_seconds']}초.",
          '- Chromium에서는 완료된 A1/B1 분석 결과를 재생용 fixture로 연결해 원본·근거 동영상 디코딩/재생, 확대 근거, 구간 이동, 새로고침 복원을 확인했습니다. 이번 화면 검사는 브라우저에서 새 파일을 업로드해 분석을 시작한 시험이 아닙니다.',
          '- 모바일 폭 375/390px에서 가로 넘침이 없었고, JavaScript 오류와 실패한 요청은 없었습니다.',
          '- Windows 실행 파일 자체는 Linux 환경에서 직접 실행하지 않았습니다.',
          '- [소프트웨어 검증 기록](generalization/software-validation.json), [브라우저 fixture 검증 기록](generalization/browser-validation.json).', '']
    lines += ['## 재현과 제한', '',
      '- `generalization/manifest.json`의 SHA256과 파일명이 같은 원본을 준비하고, 기록된 수정 전후 소스 및 해당 실행 프로필로 분석합니다. 새 기능 설정이 달라질 수 있으므로 비교 JSON의 `run_profiles`에서 두 설정과 SHA256을 함께 확인합니다.',
      '- 동결된 참조 파일과 감사 도구로 각 분할을 평가한 뒤 `generalization/build_report.py finalize`로 결과를 압축합니다. 정확한 사용법은 [재현 안내](generalization/README.md)에 있습니다.',
      '- 코드 지문, 개별 선택 좌표·프레임과 실패 분류는 비교 JSON에 유지합니다. 영상 전체·탐지 전체·모델 파일은 이 문서 자산에 복제하지 않습니다.',
      '- [전체 소프트웨어 검증](SOFTWARE_VERIFICATION.md), [선택 영역 보조 기능](MANUAL_REVIEW_REGION.md), [실험 모델 검토](EXPERIMENTAL_MODEL_REVIEW.md)를 별도로 기록했습니다. 영역 보조·모델 실험 결과는 이 24개 자동 분석 지표에 합산하지 않습니다.',
      '- Windows 실행 파일 자체의 검증 여부와 소프트웨어 테스트·브라우저 확인 결과는 배포 기록에 별도로 적습니다.', '']
    write(HERE / 'comparison.json', comparison)
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print('Final compact audit and report written without changing strict matching categories.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    snap = commands.add_parser('snapshot')
    snap.add_argument('--manifest', type=Path, required=True)
    snap.add_argument('--references', type=Path, required=True)
    snap.add_argument('--tool', type=Path, action='append', default=[])
    snap.set_defaults(run=snapshot)
    final = commands.add_parser('finalize')
    final.add_argument('--release-holdouts', action='store_true')
    final.add_argument('--baseline-audit', type=Path, action='append', required=True)
    final.add_argument('--final-audit', type=Path, action='append', required=True)
    final.add_argument('--final-revision', required=True)
    final.add_argument('--version', default='0.5.3')
    final.add_argument('--baseline-run', type=Path, required=True)
    final.add_argument('--final-run', type=Path, required=True)
    final.add_argument('--example', action='append', default=[])
    final.add_argument('--borderline', action='append', default=[])
    final.set_defaults(run=finalize)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
