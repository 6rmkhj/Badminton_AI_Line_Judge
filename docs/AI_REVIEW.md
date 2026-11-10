# 원본 영상 AI 검수와 보조 판정

프로젝트의 74개 영상을 AI가 직접 시각 검수하고, 원본 확대 연속 프레임에서 목표 셔틀의 최초 접촉을 확인한 결과다. `annotations/ai-visual/`에 영상별 기록을 저장한다. 사람이 독립 검수한 정답을 만든 것으로 표시하지 않으며, 기존 자동 분석 결과도 덮어쓰지 않는다. 최종 수치와 미해결 목록은 `runs/ai-reviewed-poc/visual-review.json` 및 `follow-up.json`에 있다.

## 판정 근거

- 최초 접촉은 하강·첫 바운스·반등의 연속 장면으로 확인한다. 최종 정지점은 첫 접촉점과 다를 수 있다. 실제 영상에서 안쪽 접촉 후 밖으로 튀어나가는 사례를 확인했다.
- 한 프레임을 정확한 접촉 순간으로 확정할 수 없으면 원본 프레임 구간과 가능한 지면 접촉 영역을 기록한다. 공중 셔틀 중심이나 깃털 끝을 지면 접촉점으로 바꾸지 않는다.
- 코트 전체의 homography가 없어도, 역할이 확인된 실제 바깥 칠 경계와 접촉 영역이 충분히 떨어져 있으면 OUT을 계산할 수 있다. IN은 유효 코트 영역 또는 유효 코트 안쪽임을 확인한 보수적인 내부 영역 전체를 사용한다.
- `court_interior`는 코트 내부의 안전한 부분 영역이다. 그 내부의 접촉에만 IN을 줄 수 있으며, 영역 밖이라는 이유로 OUT을 만들 수 없다. 반코트의 네트 경계 역시 OUT 경계가 아니다.
- 접촉 위치와 경계 추정 오차를 합산한다. 가까운 실제 칠 경계와 멀리 있는 근사 경계의 오차가 다를 때는 경계별 오차를 기록한다. 가능한 접촉 영역이 경계 불확실성과 겹치면 REVIEW다.
- 다른 시점의 결론을 연결하려면 실제로 확인한 서로 다른 두 동작의 프레임 대응, 두 원본의 해시, 대응 근거 이미지가 필요하다. 파일 이름만으로 같은 랠리라고 정하지 않는다. 직접 관측한 판정만 전달하며, 다른 영상에서 전달받은 판정을 다시 전달하지 않는다.

교차 시점 판정은 대상 영상의 접촉 좌표를 새로 만들어내지 않는다. `originating_video_id`와 `donor_contact`에 실제 접촉이 보인 원본 영상·좌표·시각을 보존한다. 실제 상충하는 관측은 보류하고, 정보가 없는 보조 시점이 직접 관측한 결론을 지우지는 않는다.

보고서의 `filename_pair_audit`는 같은 번호의 A/B 후보를 대조해 양쪽 판정 일치, 양쪽 REVIEW, 한쪽만 REVIEW, 판정 충돌을 집계한다. 특히 상대 시점에 판정이 있는데도 REVIEW인 항목은 화면 상단에 대응 검수 누락·실패 사유와 함께 표시한다. 다른 번호 영상과 원본 내용이 중복된 경우도 표시한다. 후보쌍 목록은 누락을 찾는 진단이며, 실제 동작 대응 검증 없이 판정을 변경하지 않는다.

## 레이블 출처

`source=ai_visual_review`, `status=reviewed_by_ai`, `human_verified=false`를 사용한다. AI가 보조한 판정은 `evaluation_eligible=false`, `ground_truth_eligible=false`다. `UNKNOWN`은 미해결 레이블이며 IN/OUT 정답으로 학습·집계하지 않는다. 탐지용 셔틀/배경 패치는 별도의 `detection_labels_verified_by_ai` 표시와 원본 해시·프레임·좌표를 요구한다. 판정이 UNKNOWN이어도 실제 보이는 셔틀 패치는 탐지 학습에 사용할 수 있다.

74개 모두 검수 기록이 있다는 것과 74개 모두 판정할 근거가 있다는 것은 다르다. 화면 밖 접촉, 가림, 접촉 전 클립 종료, 근접 라인 모호성은 영상별 사유와 근거를 남긴다. 나중에 추가 관측이 생기면 같은 검증기를 통해 보완할 수 있다.

## 후보 탐지와 학습

원본에서 코르크가 목재 바닥보다 어둡게 보이는 문제가 확인되어 `detector.include_dark_components=true` 선택지를 추가했다. 밝은 물체만 찾던 후보 생성에 어두운 소형 이동 성분을 포함한다. 이 자체는 접촉 확인이나 셔틀 정체 확인이 아니다. 후보가 늘어날 수 있으므로 기본값은 유지하고 실험 설정으로 제공한다.

학습 분류기는 최종 64개 제한 **전**의 최대 512개 후보를 점수화할 수 있게 변경했다. 기존에는 상위 후보에서 이미 탈락한 셔틀을 학습 점수로 복구할 수 없었다. 실제로 확인한 AI 패치만 명시적으로 선택하여 분류기를 학습하며, 동일 내용의 중복 영상 패치는 이중 학습하지 않는다.

`annotations/ai-training-v1/`과 `annotations/ai-training-v2/`는 각 학습 시점의 고정 스냅샷이다. v2는 원본 재검수에서 정정한 패치 좌표와 추가 검수 패치를 반영한다. 이후 교차 시점 검수나 라벨 보완과 분리하여 학습을 재현한다. 모델에는 레이블 출처, 레이블 지문, 원본 데이터셋 지문, 그룹별 분할, 가중치 해시, 검증 선택 epoch를 기록한다. 테스트 그룹은 checkpoint 선택에 사용하지 않는다. 소규모 AI 패치에 대한 일치율은 독립된 사람 정답 정확도나 전체 영상 탐지 재현율, 자동 IN/OUT 성능이 아니다. 촬영 환경도 독립 검증되지 않았다.

## 실행

패키지에서 보고서만 볼 때는 Python 3.11 이상으로 `start-review-windows.bat` 또는 `python3 start.py --review`를 실행한다. 보고서 보기에는 외부 Python 패키지나 원본 영상이 필요하지 않다. 포함된 미리보기와 검수 확대 이미지를 사용한다.

내장 서버는 영상의 바이트 범위 요청을 지원하므로 접촉 시각 이동 버튼과 영상 탐색을 사용할 수 있다. Python의 기본 `http.server`로 대체하면 브라우저에서 영상 탐색이 제한될 수 있으므로 제공된 런처를 사용한다.

원본이 있는 개발 환경에서 보고서를 다시 생성하려면:

```bash
python -m badminton review-labels --run runs/integrated-poc \
  --annotations annotations/ai-visual --output runs/ai-reviewed-poc
```

AI 패치 감독학습은 명시적인 옵션으로만 가능하다. 기본 학습 경로는 기존 사람 검수 레이블만 받는다.

```bash
python -m badminton train --manifest runs/inspection/input-manifest-before.json \
  --annotations annotations/ai-training-v2 --label-source ai_visual_review \
  --output models/ai-candidate-v2.pt --epochs 40
python scripts/evaluate_review_detector.py \
  --annotations annotations/ai-training-v2 --model models/ai-candidate-v2.pt --output runs/ai-reviewed-poc/detector-validation-v2.json
```

학습·재분석에는 기존 설치 절차와 원본 영상이 필요하다. 제공된 실행 패키지는 원본 녹화 파일을 중복 포함하지 않는다. `evaluate_review_detector.py`는 원본을 다시 읽어 같은 조건의 후보 생성과 AI 점수 재정렬을 비교하며, 사람이 완전하게 표시한 탐지 정답에 대한 성능이라고 주장하지 않는다.
