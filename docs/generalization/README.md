# 추가 영상 검증 재현 자산

현재 상태는 **24개 추가 영상의 동결 비교 완료**입니다. 엄격한 첫 착지 부근 일치는 개발 영상에서 **1/9→3/9**, 홀드아웃에서 **0/4→2/4**였습니다. 가림·모호한 참조는 위치 평가 분모에서 제외했으며, 모든24개 결과는 `REVIEW`입니다. 이는 AI 시각 참조와의 근접성 비교이며 독립적인 물리적 접촉·IN/OUT 정확도가 아닙니다. [전체 보고서](../GENERALIZATION_VALIDATION.md)와 [압축 비교 결과](comparison.json)를 함께 확인하세요.

개발16개는 변경되지 않은 탐지 단계의 실제 관측을 재사용하고 나머지 분석·근거 생성을 다시 실행했습니다. 홀드아웃8개와 기존 A1/B1 회귀 영상은 전체 새 실행입니다. `inventory.json`의 스냅샷 단계 상태 표시는 당시의 역사적 기록으로 유지하며, 최종 지표는 `comparison.json`에 있습니다. 동결 참조 바이트와 도구 해시는 그대로 보존합니다.

`manifest.json`은 추가 원본 24개의 파일명, Git blob, SHA256과 사전 개발/홀드아웃 분할을 보존합니다. 원본 파일을 복제하지 않습니다. `references/`는 소스 영상에서 먼저 작성한 AI 시각 참조를 바이트 그대로 보존하고, `inventory.json`은 파일별 SHA256을 기록합니다. AI 참조는 독립 사람 정답이 아닙니다. 참조의 원래 근거 경로는 시각 검사 작업 공간을 가리킬 수 있으며, 최종 문서에 복사한 예시는 `examples/`를 사용합니다.

24개 파일은 12쌍의 A/B 촬영 시점입니다. 집계 단위는 영상 파일이며, 같은 사건의 두 시점을 독립 사건 두 개로 해석하지 않습니다.

## 참조와 입력 목록 동결

다음 명령은 모든 참조 파일을 복사하지만 JSON 참조 내용을 파싱하지 않습니다. 홀드아웃의 위치·결과를 요약하지 않습니다. 이미 복사된 참조와 다른 바이트를 덮어쓰려 하면 중단합니다.

```bash
python docs/generalization/build_report.py snapshot \
  --manifest runs/generalization/manifest.json \
  --references runs/generalization/references \
  --tool runs/generalization/audit.py
```

## 동결된 실행의 최종 비교

원본 SHA256이 일치하는 입력을 수정 전 커밋 `7e74b058ae95543bb9c7d121761426462e74f49c`와 동결된 수정 후 소스로 분석합니다. 각 실행의 설정은 `run-config.json`으로 보존하며, 새 기능 설정이 달라졌다면 최종 `comparison.json`의 `run_profiles`에 두 설정과 파일 SHA256을 함께 기록합니다. 감사 JSON의 개별 행은 `audit.py`의 `videos` 구조를 따릅니다. 개발과 홀드아웃 감사는 별도 파일로 전달할 수 있습니다.

실행 루트에 `manifest.json`을 두고, `input/` 아래에 목록의 파일명으로 원본을 준비하면 저장된 배치 도구로 재실행할 수 있습니다. 설치한 분석 의존성이 있는 Python을 사용하며, 두 소스 체크아웃을 분석 도중 수정하지 않습니다. 캐시는 코드 지문·실행 설정·입력 SHA256이 모두 일치할 때만 재사용합니다.

```bash
python docs/generalization/reproduction_tools/run_batch.py BASELINE_SOURCE baseline all runs/generalization
python docs/generalization/reproduction_tools/run_batch.py FINAL_SOURCE final development runs/generalization
```

개발 결과 검토와 코드 동결이 끝난 후에만 동일한 최종 소스로 `final holdout` 실행을 추가합니다. 홀드아웃 감사에는 `--split holdout --release-holdouts --freeze-manifest freeze.json`을 사용합니다. 동결 JSON은 최종 `commit`, `implementation_fingerprint`, 원본 목록 `manifest_sha256`, 실행별 `runs.<실행 폴더명>.{commit,implementation_fingerprint,run_config_sha256}`를 담습니다. 감사 도구는 실행 설정 파일과 각 결과의 코드·설정 지문이 이 기록과 일치하는지 확인합니다.

홀드아웃은 알고리즘·참조 동결 후에만 공개합니다. `--release-holdouts`는 그 공개 단계에서 명시적으로 사용하며, 아래 `FINAL_SOURCE_COMMIT`은 실제 수정 후 소스 커밋으로 바꿉니다.

```bash
python docs/generalization/build_report.py finalize --release-holdouts \
  --baseline-audit runs/generalization/audit-development-baseline.json \
  --baseline-audit runs/generalization/audit-holdout-baseline.json \
  --final-audit runs/generalization/audit-development-final.json \
  --final-audit runs/generalization/audit-holdout-final.json \
  --final-revision FINAL_SOURCE_COMMIT \
  --baseline-run runs/generalization/baseline \
  --final-run runs/generalization/final \
  --example EXPLICITLY_SELECTED_KEY \
  --borderline EXPLICITLY_SELECTED_BORDERLINE_KEY
```

예시와 경계 사례 인수는 선택 사항입니다. 이미지 예시는 최대 8개 키만 복사하며 대용량 실행 전체는 복제하지 않습니다. 정상·실패·가림 사례를 구분하여 선택하고, 설명하기 쉬운 성공만 전체 결과처럼 제시하지 마세요.

최종 생성기는 24개의 고유 입력과 16/8 분할, 참조 목록의 전체 대응, 모든 감사 행의 입력 SHA256·동결 참조 SHA256, 수정 후 단일 실행 코드 지문을 확인합니다. 홀드아웃 감사에는 저장된 동결 커밋·코드 지문·동결 문서 SHA256의 검증 기록이 필요합니다. 수정 전후 참조 좌표·프레임 구간·허용 오차가 달라져도 중단합니다. 감사 도구가 분류한 `correct_first_vicinity`만 엄격 일치로 세며 경계 사례의 오차 기준을 늘리지 않습니다. `comparison.json`은 선택·비교·실패 분류에서 필요한 필드와 실제 실행 설정을 보존하고 대용량 탐지·궤적 본문을 제외합니다.

경로점 포함률은 접촉 위치를 평가할 수 있는 참조 영상에서 AI가 표시한 일부 프레임의 근접 후보 존재 여부입니다. 탐지 후보, 최선 단일 경로, 실제 선택된 경로의 포함률을 따로 기록합니다. 물리적 접촉·코트 판정 정확도를 이 값으로 대신하지 않습니다.

개발 탐지 단계 재사용은 별도 검증 방식입니다. `reproduction_tools/run_development_detector_replay.py`는 사전 지정된 개발 분할과 저장된 최종 커밋·코드 지문만 허용합니다. 탐지 코드·설정, 파이프라인 입력 계약, 검토된 순차 원본 읽기 동작, 원본 해시, 프레임 수·정확한 타임스탬프·전처리 및 각 탐지 파일 해시를 확인합니다. 원본 분석 프레임 읽기, 궤적 연결, 이벤트 선택과 원본 근거 생성은 새로 실행합니다. 각 결과에 `validation_run`과 재사용 실행 시간 범위를 남기고, `detector-replay-run.json`은 후속 홀드아웃 실행이 `run-config.json`을 다시 기록해도 보존합니다. 홀드아웃 감사는 탐지 재사용 결과를 거부합니다.

이를 개발16개 모두에 대한 새로운 고해상도 탐지 실행이나 전체 파이프라인 속도 측정으로 표현하지 않습니다. 영역 보조·모델 실험도 자동 지표와 합산하지 않습니다. `--preflight-only`는 영상 처리 전 계약을 검사하며, 실제 실행에서는 새로 읽은 프레임과 정확한 타임스탬프를 다시 확인합니다. 탐지·원본 읽기 코드가 바뀌면 검토를 다시 수행하거나 전체 새 분석을 실행해야 합니다.

[공개 전 동결 기록](freeze-release.preinspection.json), [조건부 공개 영수증](holdout-inspection-release-20261112T080012Z.json), [탐지 재사용 기록](detector-replay-provenance.json), [개발 회귀 점검](development-quality-scoreboard.json), [별도 시각 검토](development-quality-ledger.json)를 보존했습니다. 원본 대용량 탐지 배열과 영상 전체는 문서 자산에 포함하지 않습니다.
