# Badminton AI Line Judge

스마트폰 경기 영상을 입력받아 코트·셔틀콕 후보·궤적·착지 후보와 근거를 생성하고, 사람이 검수한 정답으로 평가하는 오프라인 연구용 시스템입니다.

**현재 기본 실행은 OpenCV 후보 생성 방식입니다. 독립 데이터로 성능을 검증한 셔틀콕 모델 가중치는 제공되지 않습니다.** 기하 보정, 셔틀콕 식별 또는 최초 접촉 증거가 부족하면 `REVIEW`를 반환합니다. 실행 성공, 테스트 통과, 합성 영상의 성과를 실제 판정 정확도로 취급하지 않습니다.

## 빠른 실행

Python 3.11 이상과 `ffmpeg`, `ffprobe`가 필요합니다.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,learning]'
python -m badminton analyze video --output runs/latest
python -m badminton serve --run runs/latest --port 8765
```

현재 저장소 영상 실행 결과는 [2026-11-10 검증 기록](docs/validation-2026-11-10.md)에 정리했습니다. 74개 영상 처리는 완료했지만, 자동 판정은 모두 `REVIEW`이고 사람 검수 정답이 없어 실제 정확도는 미측정입니다.

현재 클라우드 환경에는 준비된 Python이 있습니다.

```bash
/workspace/.venvs/badminton/bin/python -m badminton serve --run runs/latest --port 8765
```

브라우저에서 `http://127.0.0.1:8765`를 열면 영상, 후보 궤적, 코트 가설, 착지 근거, 원본 확대 이미지와 검수 양식을 볼 수 있습니다. 한 프레임 이동은 각 영상의 실제 표시 타임스탬프를 사용합니다. 서버는 기본적으로 로컬 컴퓨터에서만 접속할 수 있습니다.

## 포함된 기능

| 모듈 | 동작 및 제한 |
|---|---|
| 입력 | 원본 SHA256, MP4 편집 목록을 반영한 표시 시각, 회전 처리, H.264 검수용 영상, 원본 프레임 확대 근거 |
| 코트 | 배경 합성, 청색·백색 선 분할, Hough 선 정리, 표준 코트 기하 정합, 모호성과 부분 시야 판정 |
| 검출 | 복수 소형 밝은 물체 후보, 배경·시간 차분, 카메라 이동 보정, 선택적 ONNX 모델 및 학습한 패치 분류기 |
| 추적 | Kalman/Hungarian 연결, 관측 누락, 가림 가능성, 모호한 재등장, 전후방 연결 가설 |
| 착지 | 속도 변화·방향 전환·종료 후보, 역방향 관측, 위치 불확실성, 접촉 증거와 후보 좌표 분리 |
| 판정 | 단식·복식·서비스 경계, 라인 접촉, 접촉 영역과 오차 범위, 근거 부족 시 REVIEW |
| 동기화 | 오디오 시간차 탐색, 모호성 검사, 사람이 확인한 두 개 이상 이벤트 기준 시각 정렬 |
| 검수 | A/B 영상, 실제 프레임 탐색, 후보 선택, 접촉 좌표, UNKNOWN 유지, 정답·학습 패치 저장 |
| 평가 | 자동 판정률·REVIEW율, 검수 정답 대비 정확도·신뢰구간, 접촉·탐지·추적·보정 지표 |
| 데이터 | 같은 랠리 후보와 중복 파일을 함께 묶는 train/val/test 분할, 출처 해시, 정답과 초안 분리 |

코트가 `review`인 경우 반환된 homography와 polygon은 **가설**입니다. 정확한 코트 좌표나 성공한 보정으로 사용하지 않습니다. 셔틀콕 중심의 공중 좌표를 착지 위치로 투영하지 않습니다. 기본 착지 후보 생성기는 속도 변화만으로 물리적인 최초 접촉을 확정하지 않습니다.

## 검수와 정답

1. 영상과 확대 근거를 보고 대상 셔틀콕과 최초 접촉이 실제로 보이는지 확인합니다.
2. 불명확하면 `UNKNOWN`을 유지합니다. 라인 근접 사례는 고속 촬영 등 독립 근거 없이 정답으로 확정하지 않습니다.
3. 확인 가능한 경우 판정, 최초 접촉 시각, 접촉 위치, 검수자와 근거를 입력하고 명시적으로 확정합니다.
4. A/B 동기화가 보류되면 두 영상에서 같은 사건을 두 번 이상 지정합니다. 사람이 지정한 동기화는 자동 오디오 결과와 구분해 저장됩니다.
5. 셔틀콕/배경 패치 라벨을 충분히 검수한 뒤 학습 가능 여부를 확인합니다.

```bash
python -m badminton train --manifest runs/latest/manifest.json \
  --annotations runs/latest/annotations --readiness-only
python -m badminton train --manifest runs/latest/manifest.json \
  --annotations runs/latest/annotations --video-dir video \
  --output models/candidate-scorer.pt
python -m badminton evaluate --run runs/latest
```

착지 시계열 모델은 `공중 / 최초 접촉 / 접촉 이후` 라벨과 별도로 검수한 코르크 접촉점을 사용합니다. `train-contact` 명령으로 준비 상태 확인과 학습을 실행합니다. 독립 랠리 단위의 정밀도·접촉점 오차 검증이 충분하지 않은 체크포인트는 접촉 후보만 제공하고 최초 접촉을 확정하지 않습니다.

```bash
python -m badminton train-contact --manifest runs/latest/manifest.json \
  --annotations runs/latest/annotations --readiness-only
python -m badminton train-contact --manifest runs/latest/manifest.json \
  --annotations runs/latest/annotations --output models/contact-model.pt
```

학습한 접촉 모델은 설정의 `events.contact_model_path`로 연결할 수 있습니다. 셔틀콕 후보 분류기는 `detector.scorer_path`, 호환 ONNX 검출기는 `detector.backend="onnx"` 및 `detector.model_path`를 사용합니다. 모델을 연결해도 대상 ID·코트 보정·접촉 불확실성 조건이 충족되지 않으면 REVIEW입니다.

`annotation-drafts/`는 자동 생성한 `UNKNOWN` 초안입니다. `annotations/`의 사람이 명시적으로 검수한 자료만 평가에 사용할 수 있습니다. 초안과 모델 예측을 정답으로 복사해 정확도를 만들지 않습니다. 정답이 없거나 표본이 부족하면 해당 지표는 `null`과 사유를 반환합니다. 사람의 접촉 정보를 사용한 보조 판정은 자동 모델 성과에서 제외합니다. 자세한 스키마는 [평가 문서](docs/evaluation.md)에 있습니다.

## 설정

`--config configuration.json`으로 아래처럼 분석 폭과 규칙을 설정합니다. 기본 규칙은 복식 랠리입니다. 서비스 `half`와 `side`는 **받는 서비스 박스**의 코트 좌표이며, `near`는 y<6.7m, `left`는 x<3.05m입니다.

```json
{
  "analysis_width": 1280,
  "workers": 2,
  "source_crop_count": 1,
  "rules": {"game": "doubles", "phase": "rally"},
  "detector": {"backend": "opencv_candidates"}
}
```

기본 분석은 축소 영상에서 후보를 찾고 원본 해상도의 정확한 프레임을 확대 근거로 보존합니다. 작은 셔틀콕은 축소 과정에서 놓칠 수 있으므로, 후보 수나 후보의 휴리스틱 점수는 재현율 또는 확률이 아닙니다. 후보 예산 소진율과 ID 모호성은 `detection_quality`에 기록됩니다. 실제 사용 가능한 탐지 성능에는 검수 데이터와 학습·독립 평가가 필요합니다.

## 출력과 재현

`runs/latest/`에 다음이 생성됩니다.

- `index.json`: 처리 상태, 영상 목록, A/B 후보와 자동 동기화 결과
- `manifest.json`: 원본 해시, 중복·랠리 그룹, 데이터 분할
- `videos/<id>/result.json`: 코트, 궤적, 이벤트, 판정 사유, 시간·설정·소스 정보
- `videos/<id>/preview.mp4`, `evidence/`: 검수용 영상과 근거 이미지
- `videos/<id>/detections.json`: 프레임별 복수 후보와 후보 예산 진단
- `annotation-drafts/`, `annotations/`: 모델 초안과 사람 검수의 분리된 저장소
- `evaluation.json`: 측정 가능한 지표와 아직 측정할 수 없는 지표의 사유

`python -m badminton.reporting --run runs/latest`로 최종 실행의 한국어 HTML·JSON 보고서를 생성합니다. 보고서는 영상 처리 완료와 자동 판정 성능 입증을 별도로 표시합니다.

기본 재실행은 입력 해시·설정·분석 코드 지문이 같은 결과만 재사용합니다. `--no-resume`으로 다시 실행할 수 있습니다. 실패한 재분석은 이전 성공 결과를 현재 결과로 평가하지 않습니다. 원본 영상은 변경하지 않습니다.

## 테스트

```bash
python -m pytest -q
```

테스트는 규칙 경계·불확실성·가림·ID 연결·시간차 방향·애매한 동기화·데이터 누출·검수 출처·영상 시각·회전·HTTP 접근 검증과 저장을 확인합니다. 합성 테스트는 소프트웨어 동작 확인이며, 실제 대회 정확도의 증거가 아닙니다.

현재 데이터는 독립 체육관·촬영일·기기가 확인된 시험 세트가 아닙니다. 실전 판정 성능의 최종 완료 여부는 검수 정답, 실제 자동 판정률과 오판율, 표본 수와 신뢰구간을 함께 보고 결정해야 합니다.
