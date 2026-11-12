# 추가 영상의 추적·첫 착지 부근 검증 — 0.5.3

알고리즘과 시각 참조를 동결한 뒤 공개한 추가 영상 비교입니다. 후보의 첫 착지 **부근** 일치와 물리적 바닥 접촉·IN/OUT 정확도는 서로 다른 항목입니다.

## 입력과 분리

- 사전 선정한 추가 영상 24개: 개발 16개, 홀드아웃 8개. 기존 A1/B1 수정 영상은 제외했습니다.
- 24개 파일은 12쌍의 A/B 촬영 시점입니다. 집계 단위는 영상 파일이며, 같은 사건을 본 두 시점을 독립된 사건 두 개로 해석하지 않습니다.
- 4개 촬영 파일명 그룹에서 각각 개발 2쌍과 홀드아웃 1쌍을 선정했습니다. 같은 프로젝트·체육관 자료이며 독립 체육관 일반화 시험이 아닙니다.
- 소스 영상부터 AI가 시각 참조를 작성했습니다. 독립 사람이 표시한 정답이 아니며 실제 접촉·라인 판정 정확도는 미측정입니다.
- 수정 전 소스: `7e74b058ae95543bb9c7d121761426462e74f49c`. 수정 후 소스: `d7f3d3303daa40c699d579c17e40269173220937`.
- [입력 SHA256·분할](generalization/manifest.json), [동결 참조 해시](generalization/inventory.json), [압축 비교 결과](generalization/comparison.json).

[공개 전 동결 기록](generalization/freeze-release.preinspection.json)과 [별도 공개 영수증](generalization/holdout-inspection-release-20261112T080012Z.json)을 보존했습니다. 새 홀드아웃 8개가 모두 완료되고 코드·설정·입력 해시가 일치한 뒤에만 결과와 참조를 열었습니다.

[탐지 단계 재사용 기록](generalization/detector-replay-provenance.json), [개발 엄격 비교·회귀 점검](generalization/development-quality-scoreboard.json), [별도 시각 검토 기록](generalization/development-quality-ledger.json)을 함께 제공합니다. 경계 사례의 시각 해석은 엄격 일치 수를 변경하지 않습니다.

## 실행 범위와 탐지 관측 재사용

- 개발 16개는 탐지 코드·설정이 같은 이전 전체 실행의 실제 탐지 관측을 재사용했습니다. 원본 SHA256, 탐지 파일 해시, 프레임 수·정확한 타임스탬프·전처리·설정 및 이전/현재 코드 지문을 확인했습니다.
- 원본 분석 프레임 읽기, 코트 확인, 궤적 연결, 이벤트 선택, 원본 확대 근거 생성은 새 코드로 다시 실행했습니다. 개발 영상 전체에 대해 원본 고해상도 탐지를 새로 실행했다고 주장하지 않습니다.
- 홀드아웃 8개와 기존 A1/B1 회귀 영상은 탐지를 포함한 전체 새 실행입니다. 영역 보조 실험은 이 자동 결과에 포함하지 않습니다.
- 재사용 실행 시간은 전체 분석 속도 개선의 근거로 쓰지 않습니다. 원래 실행에는 라이브러리 버전이 별도로 남아 있지 않아 과거 런타임 환경의 완전한 동일성까지 입증하지는 못합니다.

## 발견한 실패 원인과 보완

- 희미한 셔틀콕이 탐지 후보 제한에서 밀리면 다음 원본 확대 영역도 놓쳤습니다. 실제 관측의 짧은 이동 이력을 이용해 기존 후보·확대 영역 예산 일부를 예약하고, 그 좁은 원본 영역에서만 약한 명암 변화를 추가 검사합니다.
- 누락 구간 뒤의 연결 범위가 고정 거리라 빠른 낙하를 놓쳤습니다. 경과 시간과 관측 속도를 반영하되, 검출되지 않은 좌표를 실제 관측처럼 채우지 않습니다.
- 선수의 몸·신발로 옮겨 붙거나 한 번 튄 뒤 정지한 경로를 반동으로 높게 평가했습니다. 가림 전후 외형 변화와 여러 실제 관측의 반동 방향·일관성을 함께 검사합니다.
- 긴 배경 경로와 늦게 이어진 반동 대안이 짧은 첫 낙하를 밀어냈습니다. 공간별 후보 여유를 확보하고, 낙하 전후 곡선이 연결될 수 있는 이른 반동 대안을 먼저 검토합니다.
- 관측이 끊긴 앞뒤를 하나의 연속 근거처럼 세면 몸·라켓으로 옮겨 붙은 경로를 놓쳤습니다. 신원 변화와 반동 검사는 누락 구간 뒤에서 실제로 이어지는 관측 구간을 기준으로 다시 평가합니다.
- 반동처럼 보이는 갈래 외에 실제 관측으로 계속 진행하는 대안이 있는지 함께 검사하고, 후보 크기가 극단적으로 줄어드는 연결을 재검토해 손·라켓 갈래가 우선되는 경우를 줄였습니다. 이 검사로 모든 오연결이 해결되지는 않았습니다.
- 이 검사는 후보의 근거를 보완합니다. 같은 코트의 셔틀콕이라는 확인이나 실제 코르크·바닥 접촉 관측을 대신하지 않습니다.

## 엄격한 첫 착지 부근 비교

참조에 처음부터 기록한 프레임 구간과 픽셀 오차를 모두 만족해야 일치로 셉니다. 원본 PTS 시작값이 영상마다 다르므로 이 집계는 정규화된 재생 시간과 원본 시간을 직접 빼지 않고 같은 원본 프레임 번호로 비교합니다. 가림·모호한 참조는 위치 일치 분모에서 제외하고 별도로 남깁니다. 근소한 실패를 나중에 일치로 바꾸지 않습니다.

| 분할 | 완료 감사 전→후 | 엄격한 첫 부근 일치 전→후 | 탐지 후보의 참조 경로점 포함 전→후 | 최선 단일 경로의 참조점 포함 전→후 | 선택된 경로의 참조점 포함 전→후 |
|---|---|---|---|---|---|
| 개발 | 16→16 | 1/9→3/9 | 35/66→44/66 | 23/66→30/66 | 9/66→15/66 |
| 홀드아웃 | 8→8 | 0/4→2/4 | 24/37→28/37 | 14/37→19/37 | 3/37→12/37 |

엄격한 부근 일치 중 `ordinary-12-b`, `8k-7-a`는 누락된 관측 사이를 낙하·반동 경로로 근사한 위치입니다. 실제 관측점과 추정 중심을 구분하며, 해당 프레임의 물리적 접촉을 직접 관측했다는 뜻이 아닙니다.

경로점 포함은 접촉 위치를 평가할 수 있는 참조 영상에서 AI가 표기한 일부 시점의 근처 후보가 있는지를 확인한 값입니다. 전체 프레임 추적률, 대상 신원 확인, 실제 착지 정확도를 뜻하지 않습니다. 최선 단일 경로와 실제 화면에 선택된 대표 경로의 포함률을 구분합니다.

## 후보 선택과 판정 보류

| 영상 | 분할 | 접촉 참조 가시성 | 수정 전 | 수정 후 | 자동 판정 |
|---|---|---|---|---|---|
| 8k-2-a | development | visible | wrong_target_or_spatial_localization · f74 | correct_first_vicinity · f84 | REVIEW |
| 8k-2-b | development | visible | wrong_target_or_spatial_localization · f14 | wrong_target_or_spatial_localization · f12 | REVIEW |
| 8k-5-a | development | occluded | no_observable_ground_truth · f56 | no_observable_ground_truth · f45 | REVIEW |
| 8k-5-b | development | visible | correct_first_vicinity · f39 | correct_first_vicinity · f39 | REVIEW |
| 8k-7-a | holdout | visible | wrong_target_or_spatial_localization · f109 | correct_first_vicinity · f77 | REVIEW |
| 8k-7-b | holdout | ambiguous | no_observable_ground_truth · f27 | no_observable_ground_truth · f73 | REVIEW |
| ordinary-4-a | development | visible | wrong_target_or_spatial_localization · f41 | wrong_target_or_spatial_localization · f42 | REVIEW |
| ordinary-4-b | development | visible | wrong_target_or_spatial_localization · f267 | wrong_target_or_spatial_localization · f142 | REVIEW |
| ordinary-12-a | development | visible | wrong_target_or_spatial_localization · f152 | wrong_target_or_spatial_localization · f150 | REVIEW |
| ordinary-12-b | development | visible | wrong_target_or_spatial_localization · f165 | correct_first_vicinity · f183 | REVIEW |
| ordinary-18-a | holdout | visible | wrong_target_or_spatial_localization · f25 | wrong_target_or_spatial_localization · f67 | REVIEW |
| ordinary-18-b | holdout | visible | wrong_target_or_spatial_localization · f167 | correct_first_vicinity · f186 | REVIEW |
| uhd4k-2-a | development | ambiguous | no_observable_ground_truth · f59 | no_observable_ground_truth · f59 | REVIEW |
| uhd4k-2-b | development | ambiguous | no_observable_ground_truth · f41 | no_observable_ground_truth · f57 | REVIEW |
| uhd4k-4-a | development | ambiguous | no_observable_ground_truth · f45 | no_observable_ground_truth · f45 | REVIEW |
| uhd4k-4-b | development | ambiguous | no_observable_ground_truth · f54 | no_observable_ground_truth · f44 | REVIEW |
| uhd4k-3-a | holdout | ambiguous | no_observable_ground_truth · f23 | no_observable_ground_truth · f76 | REVIEW |
| uhd4k-3-b | holdout | ambiguous | no_observable_ground_truth · f20 | no_observable_ground_truth · f62 | REVIEW |
| uhdhd-2-a | development | ambiguous | no_observable_ground_truth · f89 | no_observable_ground_truth · f88 | REVIEW |
| uhdhd-2-b | development | occluded | no_observable_ground_truth · f20 | no_observable_ground_truth · f60 | REVIEW |
| uhdhd-4-a | development | visible | wrong_target_or_spatial_localization · f138 | wrong_target_or_spatial_localization · f93 | REVIEW |
| uhdhd-4-b | development | visible | wrong_target_or_spatial_localization · f71 | wrong_target_or_spatial_localization · f71 | REVIEW |
| uhdhd-5-a | holdout | visible | wrong_target_or_spatial_localization · f48 | wrong_target_or_spatial_localization · f55 | REVIEW |
| uhdhd-5-b | holdout | occluded | no_observable_ground_truth · f18 | no_observable_ground_truth · f79 | REVIEW |

`REVIEW`는 자동 판정 보류입니다. 근처 후보가 맞아도 물리적 접촉과 코트 보정 조건을 만족하지 않으면 IN/OUT을 확정하지 않습니다.

홀드아웃 공개 뒤 대표 성공·실패의 원본과 결과를 별도 AI 시각 검토로 확인했습니다. 이 검토는 참조·허용 오차·성공 수를 변경하지 않으며, 독립 사람 정답을 대신하지 않습니다. [검토 기록 1](generalization/holdout-independent-visual-review.json), [검토 기록 2](generalization/holdout-supplemental-visual-review.json).

## 기존 A1/B1 회귀 확인 — 추가 24개 집계에서 제외

처음 수정에 사용한 두 영상도 같은 동결 소스로 다시 실행했습니다. 원래의 AI 근사 참조 범위를 유지한 개발 회귀 검사이며, 새 영상의 일반화 성능이나 독립 접촉 정확도에 더하지 않습니다.

| 원본 시점 | 첫 반동 후보 프레임 | 분석 화면 좌표 | AI 근사 참조와 거리 / 기존 허용 | 기존 범위 안 | 자동 판정 |
|---|---|---|---|---|---|
| A | 49 | [360.869, 447.182] | 5.184px / 12px | True | REVIEW |
| B | 42 | [396.667, 200.333] | 1.7px / 15px | True | REVIEW |

두 결과 모두 수동 위치 보조 없이 계산했고, 선택 궤적은 실제 탐지 관측으로만 구성했습니다. 누락 프레임은 남아 있으며 실제 코르크·바닥 접촉은 확인되지 않았습니다. 실행 시간은 CPU를 공유한 환경의 측정값이므로 속도 개선 근거로 쓰지 않습니다. [별도 회귀 기록](generalization/control-regression.json).

입력 출처도 구분합니다. A1은 첨부 전송 도구의 32MiB 제한 때문에 저장소의 동명 원본을 사용했으며, 첨부 A와 바이트 동일성은 확인하지 못했습니다. B1은 실제 첨부 파일을 사용했고 저장소 원본과 해시가 일치했습니다. [원래 입력 범위와 검증 설명](SHUTTLE_TRACKING_FIX.md).

![기존 A 전체 장면](generalization/examples/control-A-overview.jpg)

![기존 A 원본 확대 경로](generalization/examples/control-A-context.jpg)

![기존 B 전체 장면](generalization/examples/control-B-overview.jpg)

![기존 B 원본 확대 경로](generalization/examples/control-B-context.jpg)

## 남은 실패와 해석

수정 후에도 아래 영상은 원래의 위치·프레임 기준을 충족하지 못했습니다. 실행 완료나 `REVIEW` 표시를 올바른 추적으로 세지 않습니다.

| 영상 | 엄격 기준 결과 | 자동 진단상 남은 단계 |
|---|---|---|
| 8k-2-b | wrong_target_or_spatial_localization | event_proposal_missing_or_temporally_mislocalized |
| ordinary-4-a | wrong_target_or_spatial_localization | candidate_rank_wrong |
| ordinary-4-b | wrong_target_or_spatial_localization | detector_contact_candidate_absent |
| ordinary-12-a | wrong_target_or_spatial_localization | candidate_rank_wrong |
| uhdhd-4-a | wrong_target_or_spatial_localization | detector_contact_candidate_absent |
| uhdhd-4-b | wrong_target_or_spatial_localization | event_proposal_missing_or_temporally_mislocalized |
| ordinary-18-a | wrong_target_or_spatial_localization | detector_contact_candidate_absent |
| uhdhd-5-a | wrong_target_or_spatial_localization | event_proposal_missing_or_temporally_mislocalized |

단계 진단은 저장된 탐지·궤적·후보와 시각 참조의 근접성을 비교한 결과입니다. 원본에서 계속 가려진 구간, 희미한 바닥 구간, 여러 코트의 동시 움직임은 여전히 대상 식별을 어렵게 합니다. 옆 코트의 실제 셔틀콕도 매끄러운 경로를 만들 수 있으므로 움직임이 자연스럽다는 이유만으로 대상 코트를 확정하지 않습니다.

## 경계 사례 — 엄격 일치와 별도

다음 사례는 원래 오차 기준을 그대로 적용한 비교입니다. 설명을 위해 별도로 표시하며 합계의 성공 수를 늘리지 않습니다.

- `ordinary-12-a`: wrong_target_or_spatial_localization, 거리 6.6145px / 허용 6px, 구간 밖 프레임 오차 0 / 추가 허용 0.0.

## 같은 원본의 전후 근거

표시는 후보 경로와 추정 위치입니다. 관측 누락과 가림이 남아 있으며 독립적인 바닥 접촉 확정을 뜻하지 않습니다.

### 8k-2-a

![8k-2-a 수정 전 전체 장면](generalization/examples/8k-2-a-before-overview.jpg)

![8k-2-a 수정 전 원본 확대 경로](generalization/examples/8k-2-a-before-context.jpg)

![8k-2-a 수정 후 전체 장면](generalization/examples/8k-2-a-after-overview.jpg)

![8k-2-a 수정 후 원본 확대 경로](generalization/examples/8k-2-a-after-context.jpg)

### 8k-2-b

![8k-2-b 수정 전 전체 장면](generalization/examples/8k-2-b-before-overview.jpg)

![8k-2-b 수정 전 원본 확대 경로](generalization/examples/8k-2-b-before-context.jpg)

![8k-2-b 수정 후 전체 장면](generalization/examples/8k-2-b-after-overview.jpg)

![8k-2-b 수정 후 원본 확대 경로](generalization/examples/8k-2-b-after-context.jpg)

### ordinary-12-b

![ordinary-12-b 수정 전 전체 장면](generalization/examples/ordinary-12-b-before-overview.jpg)

![ordinary-12-b 수정 전 원본 확대 경로](generalization/examples/ordinary-12-b-before-context.jpg)

![ordinary-12-b 수정 후 전체 장면](generalization/examples/ordinary-12-b-after-overview.jpg)

![ordinary-12-b 수정 후 원본 확대 경로](generalization/examples/ordinary-12-b-after-context.jpg)

### ordinary-12-a

![ordinary-12-a 수정 전 전체 장면](generalization/examples/ordinary-12-a-before-overview.jpg)

![ordinary-12-a 수정 전 원본 확대 경로](generalization/examples/ordinary-12-a-before-context.jpg)

![ordinary-12-a 수정 후 전체 장면](generalization/examples/ordinary-12-a-after-overview.jpg)

![ordinary-12-a 수정 후 원본 확대 경로](generalization/examples/ordinary-12-a-after-context.jpg)

### ordinary-18-b

![ordinary-18-b 수정 전 전체 장면](generalization/examples/ordinary-18-b-before-overview.jpg)

![ordinary-18-b 수정 전 원본 확대 경로](generalization/examples/ordinary-18-b-before-context.jpg)

![ordinary-18-b 수정 후 전체 장면](generalization/examples/ordinary-18-b-after-overview.jpg)

![ordinary-18-b 수정 후 원본 확대 경로](generalization/examples/ordinary-18-b-after-context.jpg)

### ordinary-18-a

![ordinary-18-a 수정 전 전체 장면](generalization/examples/ordinary-18-a-before-overview.jpg)

![ordinary-18-a 수정 전 원본 확대 경로](generalization/examples/ordinary-18-a-before-context.jpg)

![ordinary-18-a 수정 후 전체 장면](generalization/examples/ordinary-18-a-after-overview.jpg)

![ordinary-18-a 수정 후 원본 확대 경로](generalization/examples/ordinary-18-a-after-context.jpg)

## 소프트웨어·화면 확인

- 동결 소스의 Python 테스트 556개 통과. 테스트 도구가 보고한 실행 시간 49.4초.
- Chromium에서는 완료된 A1/B1 분석 결과를 재생용 fixture로 연결해 원본·근거 동영상 디코딩/재생, 확대 근거, 구간 이동, 새로고침 복원을 확인했습니다. 이번 화면 검사는 브라우저에서 새 파일을 업로드해 분석을 시작한 시험이 아닙니다.
- 모바일 폭 375/390px에서 가로 넘침이 없었고, JavaScript 오류와 실패한 요청은 없었습니다.
- Windows 실행 파일 자체는 Linux 환경에서 직접 실행하지 않았습니다.
- [소프트웨어 검증 기록](generalization/software-validation.json), [브라우저 fixture 검증 기록](generalization/browser-validation.json).

## 재현과 제한

- `generalization/manifest.json`의 SHA256과 파일명이 같은 원본을 준비하고, 기록된 수정 전후 소스 및 해당 실행 프로필로 분석합니다. 새 기능 설정이 달라질 수 있으므로 비교 JSON의 `run_profiles`에서 두 설정과 SHA256을 함께 확인합니다.
- 동결된 참조 파일과 감사 도구로 각 분할을 평가한 뒤 `generalization/build_report.py finalize`로 결과를 압축합니다. 정확한 사용법은 [재현 안내](generalization/README.md)에 있습니다.
- 코드 지문, 개별 선택 좌표·프레임과 실패 분류는 비교 JSON에 유지합니다. 영상 전체·탐지 전체·모델 파일은 이 문서 자산에 복제하지 않습니다.
- [전체 소프트웨어 검증](SOFTWARE_VERIFICATION.md), [선택 영역 보조 기능](MANUAL_REVIEW_REGION.md), [실험 모델 검토](EXPERIMENTAL_MODEL_REVIEW.md)를 별도로 기록했습니다. 영역 보조·모델 실험 결과는 이 24개 자동 분석 지표에 합산하지 않습니다.
- Windows 실행 파일 자체의 검증 여부와 소프트웨어 테스트·브라우저 확인 결과는 배포 기록에 별도로 적습니다.
