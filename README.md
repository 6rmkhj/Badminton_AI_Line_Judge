# Badminton AI Line Judge

스마트폰 경기 영상을 입력받아 코트·셔틀콕 후보·궤적·착지 후보와 근거를 생성하고, 사람이 검수한 정답으로 평가하는 오프라인 통합 PoC입니다. 배치 분석과 카메라별 담당 라인을 지정한 판정 요청을 지원합니다.

**현재 기본 실행은 OpenCV 후보 생성 방식입니다. 독립 데이터로 성능을 검증한 셔틀콕 모델 가중치는 제공되지 않습니다.** 기하 보정, 셔틀콕 식별 또는 최초 접촉 증거가 부족하면 `REVIEW`를 반환합니다. 실행 성공, 테스트 통과, 합성 영상의 성과를 실제 판정 정확도로 취급하지 않습니다.

원본 74개에 대한 [AI 직접 검수와 보조 판정](docs/AI_REVIEW.md)을 추가했습니다. 최초 바운스의 원본 확대 근거, 라인별 불확실성, 실제 동작을 대조한 교차 시점 판정, AI 패치 감독학습을 지원합니다. `annotations/ai-visual/`은 AI 검수 레이블이며 사람이 독립 확인한 정답이 아닙니다. 최종 결과는 [AI 검수 실행 보고서](docs/AI_REVIEW_RESULTS.md)에 기록합니다.

**A/B 영상을 함께 입력하고 하나의 결과 받기:** `start-pair-windows.bat` 또는 `python3 start.py --pair`를 실행합니다. A와 B에 기존 영상을 선택하거나 새 영상 파일을 각각 올리고 `A/B 연결 분석`을 누릅니다. [A/B 통합 판정 안내](docs/PAIR_WORKSPACE.md)에 연결 기준과 실행 범위를 설명합니다. 기존 검수 화면에서도 `A/B 통합 판정`으로 이동할 수 있습니다.

0.5.5에서는 **분석 전 장치 설정 창**과 **입력 영상 정보**를 추가했습니다. `python start.py --pair` 또는 기존 A/B 실행기를 열면 서버 컴퓨터의 OS·CPU·메모리·GPU를 확인하고, 시점당 OpenCV/영상 처리 스레드, A/B 순차·동시 처리, 전체 프레임 버퍼 예산(MiB)을 정합니다. 설정을 확인하기 전에는 분석 API도 실행을 차단하며, 대기·진행 중에는 설정을 바꿀 수 없습니다. 같은 환경의 설정은 실행 폴더에 저장되지만 서버 재시작마다 다시 확인합니다.

0.5.5의 초기 장치 설정은 CPU 실행만 지원했으며, 0.5.6부터 아래 GPU 실행을 지원합니다. 스레드 수는 전체 CPU 사용률 상한이 아니며, 프레임 버퍼 예산은 프로그램 전체 RAM 상한이 아닙니다. 동시 처리 시 예산을 두 시점에 나누고, 초과 영상은 프레임을 생략하지 않고 사유와 함께 중단합니다. 순차 처리로 바꾸거나 예산을 늘려 다시 실행할 수 있습니다.

새 파일을 선택하면 서버가 먼저 원본 정보를 확인해 **평균 FPS·해상도(HD/FHD/QHD/4K)·확장자·코덱·재생 시간·파일 크기·비트레이트·프레임 수**를 표시합니다. 분석은 버튼을 눌러 시작하며, 확인한 파일은 다시 업로드하지 않습니다. 가변 프레임 간격은 별도 안내하고, 읽을 수 없는 값은 확인 불가로 표시합니다. 해상도 등급은 선명도 측정 결과가 아닙니다. 아래 0.5.4 다운로드에는 이 변경이 포함되지 않으며, 이번 변경은 소스 실행 기준입니다.

0.5.4에서는 실제 관측 경로와 원본 장면을 이용해 낙하 후보 순위를 보완했습니다. 기존 평가 가능 영상은 **5/13 → 7/13**으로 개선됐지만, 동결 후 추가한 영상에서는 **0/4**여서 남은 오탐과 일반화 문제는 해결되지 않았습니다. 모든 결과는 계속 `REVIEW`입니다. [수정 내용·원본 근거·실패 분석](docs/FALSE_POSITIVE_REMEDIATION.md)과 [Windows 휴대용 패키지](https://github.com/6rmkhj/Badminton_AI_Line_Judge/archive/refs/heads/downloads/windows-portable-0.5.4.zip)를 제공합니다. 압축을 푼 뒤 `START.bat`를 실행하세요.

0.5.3에서는 관측된 움직임을 따라 원본 탐색 영역을 유지하고, 짧은 낙하 궤적이 후보 제한에서 사라지는 문제를 보완했습니다. 반동 전후 경로의 연결 가능성, 실제 상승 움직임, 가림 뒤 외형 변화를 함께 검사하여 선수 몸·신발로 잘못 이어지는 일부 궤적의 우선순위를 낮췄습니다. [추가 24편의 검증 결과와 남은 한계](docs/GENERALIZATION_VALIDATION.md)를 확인하세요. 대상 코트가 섞이거나 셔틀콕이 희미하게 사라지는 영상에는 오탐과 누락이 남아 있습니다.

여러 코트가 함께 보이면 시점별 미리보기에서 선택 사항인 [검토 영역 지정](docs/MANUAL_REVIEW_REGION.md)을 사용할 수 있습니다. 영역을 지정한 결과는 사용자 보조 분석으로 표시하며, 바닥 접촉이나 대상 셔틀의 확인을 보장하지 않습니다. 반복되는 동일 궤적 계산을 재사용하고, 원본 탐색은 디코더 시간값이 일치할 때만 건너뛰도록 했습니다. 실제 개발 영상의 제한된 원본 읽기 구간에서 시간을 줄이고 픽셀·시각 일치를 확인했으며, 전체 분석 속도 향상률은 별도로 검증해야 합니다.

0.5.2에서는 원본의 밝은 셔틀콕과 어두운 잔상을 분리해 탐지하고, 연속 비행 경로와 첫 반등을 검토합니다. 최초 착지 추정과 이후 정지 위치를 구분하며, 원본 확대 이미지에서 전후 움직임을 확인할 수 있습니다. 새 분석이 기본값이며, 추정 위치는 접촉 확정이나 IN/OUT 판정으로 취급하지 않습니다. [실제 두 영상의 수정 전후와 검증 범위](docs/SHUTTLE_TRACKING_FIX.md)를 확인할 수 있습니다.

0.5.1에서는 A/B 분석의 진행 화면을 추가했습니다. 업로드부터 결과 연결까지의 흐름, 시점별 현재 분석 단계와 처리한 프레임 수, 경과 시간을 확인할 수 있습니다. 새로고침해도 저장된 작업의 진행 상태를 다시 불러옵니다. 단계 진행 정도는 남은 시간의 예측값이 아닙니다.

0.5.0에서는 [REVIEW 15쌍 재검수·개선](docs/REVIEW_REMEDIATION.md), 시점별 실패 사유, 검토 구간 이동과 원본 집중 재분석을 추가했습니다. 기존 검수에서 놓친 두 쌍의 첫 착지를 복구했으며, 검수 보조 결과와 새 자동 분석을 구분합니다.

**새 검수 패키지 보기:** Python 3.11 이상 설치 후 `start-review-windows.bat` 또는 `python3 start.py --review`를 실행합니다. 포함된 영상·근거를 보는 데 추가 패키지 설치가 필요하지 않습니다. 미해결 영상은 사유와 함께 REVIEW로 남습니다.

## GPU 실행 준비 (0.5.6 소스)

장치 설정에서 GPU를 선택하면 기본 후보 생성의 대비·움직임 마스크 연산과 연결된 PyTorch 모델 추론이 선택한 장치에서 실행됩니다. NVIDIA CUDA와 Apple Silicon MPS를 지원하며, YOLO ONNX 모델은 CUDA에서 실행할 수 있습니다. 영상 읽기·연결 성분·코트 보정·추적·판정 등은 CPU에서 계속 실행합니다. 기본 후보 생성은 학습된 셔틀콕 모델이 아니며, GPU 설치가 새 모델 가중치나 판정 정확도를 제공하지는 않습니다.

서버를 종료한 뒤 프로젝트 폴더에서 준비합니다. Python 3.11 이상과 GPU를 지원하는 OS 드라이버가 필요합니다. 설치기는 프로젝트의 독립 `.venv`만 사용하고, CUDA용 PyTorch 2.10.0은 공식 `cu128` 저장소에서 설치합니다. CUDA Toolkit 전체를 별도로 설치하는 대신 PyTorch 휠에 포함된 실행 라이브러리를 사용합니다.

```bash
# NVIDIA GPU · Windows/Linux
python scripts/setup_gpu.py --backend cuda
# ONNX 모델도 사용하는 경우
python scripts/setup_gpu.py --backend cuda --onnx

# Apple Silicon macOS (ONNX CUDA 모델 제외)
python3 scripts/setup_gpu.py --backend mps
```

Windows에서는 `setup-gpu-windows.bat`를 실행해도 됩니다. ONNX용 설치는 `setup-gpu-windows.bat --onnx`입니다. `--dry-run`을 붙이면 실제 명령만 확인하고, `--check-only`를 붙이면 설치 없이 기존 환경의 GPU 연산을 확인합니다. 설치기는 모델 파일을 내려받지 않으며, `onnxruntime`과 `onnxruntime-gpu`의 충돌은 프로젝트 가상환경 안에서 정리합니다. 다른 프로그램의 Python 환경은 변경하지 않습니다.

설치가 완료되면 기존 `start-pair-windows.bat` 또는 `python start.py --pair`를 실행합니다. 실행기는 준비된 `.venv`를 우선 사용합니다. 설정 창에는 실제 연산 확인을 통과한 GPU만 선택 가능하게 표시합니다. GPU 메모리 부족·드라이버 오류 시 CPU로 자동 대체하지 않고 실패 원인을 표시하므로, 문제를 해결하거나 장치 설정을 CPU로 바꿔 다시 실행하세요. AMD/Intel GPU용 설치기와 Windows 휴대용 ZIP 재패키징은 이번 소스 변경에 포함되지 않습니다.

기존 모델을 사용하려면 `configs/poc.json`의 `detector.scorer_path`, `events.contact_model_path` 또는 `events.source_contact_model_path`에 호환되는 PyTorch 체크포인트를 지정합니다. ONNX는 `detector.backend: "onnx"`, `detector.model_path`와 해당 모델의 `onnx_format`, `shuttle_class_id`를 명시합니다. 모델의 학습 전처리·입력 크기·검증 기준은 그대로 적용되며, GPU 선택만으로 미검증 모델이 판정 근거로 승격되지 않습니다.

## 빠른 실행

Python 3.11 이상과 `ffmpeg`, `ffprobe`가 필요합니다.

기존 자동 분석 폴더 `runs/integrated-poc`가 있는 환경에서는 Windows의 `start-windows.bat`, macOS/Linux의 `python3 start.py`로 자동 분석 검수 화면을 엽니다. 처음 실행 시 기본 Python 의존성이 없으면 프로젝트 안의 가상환경에 설치합니다. 새 AI 검수 전용 패키지는 위의 `start-review-windows.bat` 또는 `python3 start.py --review`를 사용합니다. 이미 생성된 결과 열람에는 FFmpeg나 학습용 PyTorch가 필요하지 않습니다. 새 영상 분석에는 아래 도구와 원본 영상이 필요합니다.

Windows 명령 프롬프트에서 직접 분석 명령을 실행할 때는 `python -X utf8 -m badminton ...` 형식으로 실행해 한국어 파일명·JSON 인코딩을 통일하세요. 제공한 실행 파일은 UTF-8 모드를 자동 적용합니다.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,learning]'
python -m badminton analyze video --config configs/poc.json --output runs/integrated-poc
python -m badminton serve --run runs/integrated-poc --port 8765
```

통합 범위와 설정은 [계획서 대응표](docs/INTEGRATED_POC.md), 프로젝트 영상 검증은 [통합 PoC 검증 기록](docs/integrated-poc-validation.md)에 정리합니다. [이전 버전 실행 기록](docs/validation-2026-11-10.md)은 별도로 보존했습니다. 처리 완료와 실제 자동 판정 정확도는 구분합니다.

카메라별 사건 구간·담당 경계를 지정한 요청은 다음과 같이 실행합니다. 예제에는 임의의 타격·접촉 정답을 넣지 않았으므로, 실제 동일 사건 여부를 확인하기 전에는 `REVIEW`를 유지합니다.

```bash
python -m badminton judge examples/request.json --video-dir video \
  --config configs/poc.json --output runs/integrated-request
```

`runs/integrated-request/request-summary.html`을 열면 요청 결과와 각 카메라의 원본 근거·궤적 영상이 표시됩니다. 전체 배치 분석은 `summary.html`을 자동 생성합니다.

현재 클라우드 환경에는 준비된 Python이 있습니다.

```bash
/workspace/.venvs/badminton/bin/python -m badminton serve --run runs/integrated-poc --port 8765
```

브라우저에서 `http://127.0.0.1:8765`를 열면 영상, 후보 궤적, 코트 가설, 착지 근거, 원본 확대 이미지와 검수 양식을 볼 수 있습니다. 한 프레임 이동은 각 영상의 실제 표시 타임스탬프를 사용합니다. 서버는 기본적으로 로컬 컴퓨터에서만 접속할 수 있습니다.

다른 컴퓨터로 결과 폴더를 옮긴 경우 `serve --video-dir <원본 영상 폴더>`를 지정할 수 있습니다. 재분석할 때 파일 이름과 원본 SHA256을 확인하며, 다른 내용의 영상을 원본으로 사용하지 않습니다. 저장된 미리보기와 근거를 열람하는 데에는 원본 영상 연결이 필요하지 않습니다.

## 포함된 기능

| 모듈 | 동작 및 제한 |
|---|---|
| 입력 | 원본 SHA256, MP4 편집 목록을 반영한 표시 시각, 회전 처리, H.264 검수용 영상, 원본 프레임 확대 근거 |
| 코트 | 배경 합성, 색상/선택적 ONNX 선 분할, 표준 기하 정합, 부분 라인 관측 영역, 알려진 렌즈 왜곡 보정, 모든 프레임의 카메라 안정성 검사 |
| 검출 | 전역 후보와 원본 해상도 ROI 재탐지 결합, 중복 제거, 선택적 ONNX 모델·패치 분류기, HOG 사람 후보와 가림 추정 |
| 추적 | Kalman/Hungarian 연결, 관측 누락, 가림 가능성, 모호한 재등장, 전후방 연결 가설 |
| 식별 | 요청 구간과 수동 타격 앵커를 이용한 역방향 tracklet 연결, 경쟁 경로·누락 관측 보존 |
| 착지 | 운동 변화 후보, 원본 5프레임 정밀 패치, 선택적 시계열 접촉 모델, 접촉 증거·오차와 후보 좌표 분리 |
| 판정 | 단식·복식·서비스 경계, 라인 접촉, 접촉 영역과 오차 범위, 근거 부족 시 REVIEW |
| 요청 | 한두 카메라의 담당 경계, 사건 구간, 대응 확인, 미해결·상충 시 REVIEW, 독립 HTML 결과 |
| 근거 | 원본 표시 시각을 유지한 궤적 영상·프레임 대응표, 원본 접촉 연속 이미지, JSON |
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

기본 분석은 축소 전역 스캔과 원본 해상도 ROI 재탐지를 결합합니다. `configs/poc.json`은 전역 폭 960px, 원본 192px ROI 최대 16개/프레임, 원본 접촉 96px 패치 5프레임을 사용합니다. 전역 스캔의 폭을 줄여도 원본 ROI는 축소하지 않습니다. ROI와 후보 예산이 유한하므로 작은 셔틀콕을 모두 찾아낸다는 보장은 없습니다. 후보 수나 휴리스틱 점수는 재현율 또는 확률이 아니며, 예산 소진율·원본 재탐지량·사람 후보·ID 모호성을 `detection_quality`에 기록합니다.

사람 검출은 일반 보행자 HOG 모델의 추정이며 실제 선수 정답이 아닙니다. 일부 라인만 보이거나 렌즈 정보·선 대응이 불충분하면 코트 좌표를 확정하지 않습니다. 카메라 안정성 검사를 일부 프레임으로 제한하면 자동 판정을 보류합니다. 선택적 모델과 렌즈 설정, 원본 접촉 모델 학습은 [통합 문서](docs/INTEGRATED_POC.md)와 [접촉 모델 문서](docs/native-contact.md)를 참고하세요.

## 출력과 재현

`runs/latest/`에 다음이 생성됩니다.

- `index.json`: 처리 상태, 영상 목록, A/B 후보와 자동 동기화 결과
- `manifest.json`: 원본 해시, 중복·랠리 그룹, 데이터 분할
- `videos/<id>/result.json`: 코트, 궤적, 이벤트, 판정 사유, 시간·설정·소스 정보
- `videos/<id>/preview.mp4`, `evidence/`: 검수용 영상과 근거 이미지
- `videos/<id>/evidence/annotated.mp4`, `annotated-timing.json`: 궤적·원본 시각과 프레임 대응표
- `videos/<id>/evidence/source-contact-*.jpg`: 원본 해상도 전후 5프레임
- `videos/<id>/detections.json`: 프레임별 복수 후보와 후보 예산 진단
- `annotation-drafts/`, `annotations/`: 모델 초안과 사람 검수의 분리된 저장소
- `evaluation.json`: 측정 가능한 지표와 아직 측정할 수 없는 지표의 사유

`python -m badminton.reporting --run runs/latest`로 최종 실행의 한국어 HTML·JSON 보고서를 생성합니다. 보고서는 영상 처리 완료와 자동 판정 성능 입증을 별도로 표시합니다.

기본 재실행은 입력 해시·설정·분석 코드 지문·학습 가중치 내용이 같은 결과만 재사용합니다. 같은 경로의 모델을 다시 학습해 교체해도 캐시가 무효화됩니다. `--no-resume`으로 다시 실행할 수 있습니다. 실패한 재분석은 이전 성공 결과를 현재 결과로 평가하지 않습니다. 원본 영상은 변경하지 않습니다.

## 테스트

```bash
python -m pytest -q
```

테스트는 규칙 경계·불확실성·가림·ID 연결·시간차 방향·애매한 동기화·데이터 누출·검수 출처·영상 시각·회전·HTTP 접근 검증과 저장을 확인합니다. 합성 테스트는 소프트웨어 동작 확인이며, 실제 대회 정확도의 증거가 아닙니다.

0.5.4 전체 Python 테스트는 684개가 53.10초에 통과했습니다. 실제 최초 A/B와 개선된 두 영상의 완료 결과 재생 검증은 [0.5.4 소프트웨어·브라우저 기록](docs/software-verification/verification.json)에, 이전 버전의 합성 검토 영역 UI와 원본 읽기 검증은 [과거 검증 기록](docs/SOFTWARE_VERIFICATION.md)에 정리했습니다.

현재 데이터는 독립 체육관·촬영일·기기가 확인된 시험 세트가 아닙니다. 실전 판정 성능의 최종 완료 여부는 검수 정답, 실제 자동 판정률과 오판율, 표본 수와 신뢰구간을 함께 보고 결정해야 합니다.
