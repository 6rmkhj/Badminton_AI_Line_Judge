# 계획서의 통합 PoC 구현 범위

이 문서는 개발계획서의 9~10주차 산출물인 **독립 실행 통합 PoC**와 구현을 대응시킨다. 실제 대회 조건의 정확도 달성·최종 실증 보고서는 정답 라벨 및 독립 시험이 필요한 후속 검증이다. 스마트폰 녹화 앱, 태블릿 관제, 운영 서버, 정밀 3D 복원은 이번 범위에 포함하지 않는다.

## 처리 흐름과 산출물

| 계획서 항목 | 구현 | 출력·수락 조건 |
|---|---|---|
| 01 영상 입력·프레임 | `media.py` | 원본 SHA256, 회전, MP4 edit list, 실제 표시 시각, 동일 프레임의 H.264 미리보기 |
| 02 자동 보정 | `calibration.py` | 배경 합성, 색상 또는 ONNX 선 분할, Hough·기하 정합, homography·품질·경쟁 가설 |
| 부분 라인·렌즈 | `calibration.py` | 관측 라인 영역과 미확인 역할 분리. 알려진 내부 파라미터로 왜곡 보정. 단일/모호한 선으로 전체 코트 좌표를 만들지 않음 |
| 고정 카메라 확인 | `assess_camera_stability` | 모든 분석 프레임의 배경 optical flow/RANSAC. 이동·관측 불가·일부 검사만 수행하면 자동 보정 수락을 무효화 |
| 03 복수 후보 | `detection.py` | 전역 저해상도 스캔 + 원본 ROI 재탐지 + NMS. ONNX 타일 추론·학습 패치 분류기 교체 가능 |
| 04 ID·가림 | `tracking.py` | Kalman/Hungarian tracklet, 전후방 연결 가설, 경쟁 ID 보류. HOG 사람 후보와 큰 이동 영역을 가림 가능성으로 사용 |
| 05~06 사건·대상 | `events.py`, `identity.py` | 요청 구간 내 후보부터 역추적. 수동 타격 앵커와 경로·시간 일관성 비교. 복수 경로·공백이 남으면 REVIEW |
| 07 최초 접촉 | `events.py`, `contact_model.py` | 원본 5프레임 ROI, 원본/분석 좌표 변환, 접촉 모델의 국소 위치·반경·오차. 공중 중심을 코트 평면에 투영하지 않음 |
| 08 규칙·근거 | `decision.py`, `evidence.py` | 단식·복식·서브, 라인 포함, 접촉 오차와 보정 오차 전파, IN/OUT/REVIEW·사유·궤적 영상 |
| 요청·카메라 책임 | `requests.py` | 요청 ID, 각 카메라의 로컬 사건 구간과 담당 경계. 동일 사건 미확인·해당 영상 미해결·상충·위치 불일치는 REVIEW |
| 실험·검수·평가 | `dataset.py`, `learning.py`, `evaluation.py`, `web.py` | 그룹 분할, UNKNOWN 초안과 사람 정답 분리, 탐지/접촉 모델 학습·평가, 시각·좌표 검수 UI |

색상 분할과 OpenCV 소형 후보 생성은 실행 가능한 기본 방식이다. ONNX 선/셔틀콕 모델, 패치 분류기, 접촉 모델의 **학습된 실전 가중치는 제공되지 않는다**. 기본 원본 패치는 관측 근거를 보존할 뿐 `contact_observed=true`를 만들지 않는다. 시계열 모델 또한 검수 라벨, 독립 그룹 평가, 촬영 영역·전처리 일치 조건을 충족해야 접촉 관측으로 수락된다.

## 실행

```bash
python -m pip install -e '.[test,learning]'
python -m badminton analyze video --config configs/poc.json --output runs/integrated-poc
python -m badminton serve --run runs/integrated-poc --port 8765
python -m badminton judge examples/request.json --video-dir video \
  --config configs/poc.json --output runs/integrated-request
```

`summary.html`은 배치 결과, `request-summary.html`은 요청별 결과다. 판정 근거 영상의 시각은 원본 시각이며, `annotated-timing.json`으로 모든 재생 프레임을 원본 프레임과 대조할 수 있다. 파일을 직접 열거나 로컬 검수 서버를 사용한다.

`configs/poc.json`은 전역 폭 960px/동시 작업 2개, 원본 ROI 최대 8개/프레임, 960px HOG 16프레임 간격, 최대 6개 착지 후보의 원본 5프레임 근거를 사용한다. 모든 원본 프레임에 대해 전역·ROI 분석을 수행한다. 큰 단일 확대 이미지는 중복 디코딩 비용을 줄이려고 이 설정에서 끄고, 원본 연속 패치를 기본 근거로 사용한다. 단일 384px 크롭도 원하면 `source_crop_count`를 1 이상으로 설정한다. 처리 속도는 CPU와 영상 해상도에 따라 달라진다.

## 판정 요청과 타격 앵커

각 영상의 `event_window_seconds`는 **그 영상의 첫 표시 프레임부터 잰 초**다. 요청 구간은 보존되고 실제 관측 프레임으로 맞춘 구간이 분석기에 전달된다. A/B 파일명이 같다고 동일 사건이나 동기화가 확정되지는 않는다. 둘을 같은 사건으로 확인한 경우에만 `same_event_confirmed`를 지정한다.

반대편 카메라의 코트 좌표 방향도 자동으로 같다고 가정하지 않는다. `same_court_coordinate_system_confirmed`는 같은 사건 확인과 별도로, 양쪽의 x/y축과 담당 라인이 공통 코트 좌표로 정렬됐음을 확인하는 값이다. 미확인인 두 시점은 좌표를 합쳐 판정하지 않고 REVIEW로 보낸다. 이 확인값 자체가 좌표 변환을 수행하지 않는다.

담당 경계 `left/right/near/far`는 코트 좌표의 x최소/x최대/y최소/y최대다. 서브이면 지정된 받는 서비스 박스의 경계를 뜻한다. `requested_boundary=all`은 네 경계 모두의 담당 시야를 요구한다. 수락 가능한 담당 영상이 없거나 관련 영상이 서로 모순되면 REVIEW다.

타격 위치를 실제로 확인했다면 해당 view의 `hit_anchors`에 다음 형태로 입력할 수 있다. 이는 계획서가 초기 실험에 허용한 대상 식별 보조 입력이며 접촉 정답은 아니다.

```json
{"time": 0.4, "point_px": [320, 210], "radius_px": 24,
 "time_tolerance_seconds": 0.06, "coordinate_system": "analysis_pixels"}
```

좌표는 설정된 분석 영상 크기 기준이다. 필요하면 앵커마다 `event_window_seconds`를 넣어 해당 착지 사건에만 적용한다. 앵커로 목표 궤적이 선택돼도 접촉 근거가 없으면 REVIEW다. 같은 대상에서 여러 착지 후보가 남아도 하나를 임의 확정하지 않는다. 사람의 요청 대응·타격·직접 선택을 사용한 결과는 자동 모델 성능 집계에서 제외한다.

## 선택적 모델과 렌즈

- 셔틀콕 ONNX: `detector.backend="onnx"`, `detector.model_path`, `onnx_format`과 `shuttle_class_id`를 모델에 맞춘다. YOLOv5/v8의 후처리 전 출력을 지원한다.
- 패치 분류기: `detector.scorer_path`로 검수 패치 기반 체크포인트를 연결한다.
- 라인 분할: `calibration.line_model_path`와 관련 `line_model_*` 설정으로 ONNX 마스크 출력을 연결한다. 마스크가 있어도 기하 검증이 실패하면 보류한다.
- 렌즈: `calibration.lens`에 `camera_matrix`, `dist_coeffs`, `image_size:[width,height]`를 지정한다. 알려진 OpenCV 방사/접선 왜곡 계수만 사용하며 자동으로 미지의 계수를 추정하지 않는다. 검출·검수 좌표는 원본 분석 좌표를 유지하고, 판정 직전에 homography 좌표계로 변환한다.
- 접촉: `train-contact --source-resolution`으로 학습한 체크포인트를 `events.source_contact_model_path`에 연결한다. 기존 축소 모델의 `events.contact_model_path`와 동시에 설정할 수 없다. 상세 계약은 [원본 접촉 분석](native-contact.md)에 있다.

모델 경로가 같아도 파일 내용 SHA256이 달라지면 기존 분석 캐시를 사용하지 않는다. 영상마다 코드 지문·설정·가중치 지문을 남겨 서로 다른 실험 결과가 섞이는 것을 확인할 수 있다.

## 완료 판정의 범위

실제 프로젝트 영상 전체를 넣어 파이프라인의 실행·중간 결과·근거 영상·보류 동작을 검증한다. 상세 수치는 [실행 검증 기록](integrated-poc-validation.md)에 남긴다. 사람이 확인한 정답이 없는 영상에서 정확도, mm 위치 오차, 기존 TrackNetV3 대비 향상, 99% 목표 달성을 주장하지 않는다. 데이터 수집·라벨링·가중치 학습과 독립 환경 성능 입증은 소프트웨어 통합 완료와 별개다.
