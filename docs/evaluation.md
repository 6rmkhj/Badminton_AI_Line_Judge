# 데이터 분할과 실제 성능 평가

이 시스템의 자동 후보 점수는 판정 정확도가 아니다. 합성 테스트 통과, 영상 처리 성공, 사람이 확인한 정답에 대한 실제 성능을 구분한다. 정답이 없을 때 정확도·오판율·접촉 오차는 JSON `null`이며, 자동 판정 비율과 REVIEW 비율만 관측값으로 보고한다.

## 실행

```bash
python -m badminton analyze video --output runs/latest
python -m badminton serve --run runs/latest --port 8765
python -m badminton evaluate --run runs/latest
```

분석은 `manifest.json`, `annotation-drafts/`, `evaluation.json`을 생성한다. 검수 화면에서 사람이 확인한 내용은 별도 `annotations/`에 저장된다. 초안 생성은 기존 파일을 덮어쓰지 않는다. `evaluate`는 예측이나 정답을 수정하지 않고 평가 결과만 다시 쓴다.

## 분할 단위

`dataset.build_manifest(video_dir, seed=42)`는 각 파일의 SHA256과 파일명 A/B 후보 쌍을 이용해 연결된 그룹을 만든다. 같은 내용의 중복 파일이 서로 다른 A/B 쌍에 나타나면 두 쌍 전체가 같은 그룹에 들어간다. 예를 들어 `b시점 17`과 `b시점 18`이 동일하면 A17·B17·A18·B18을 서로 다른 학습/평가 집합에 나누지 않는다.

그룹 단위로 약 70% train, 15% val, 15% test를 결정적으로 배정한다. 세 그룹 미만이면 검증/시험 집합을 만들 수 없다고 명시한다. 파일명 쌍은 동일 랠리나 동기화가 확인됐다는 뜻이 아니다. 체육관·촬영일·기기가 독립적이라는 확인 자료가 없으므로, 이 분할만으로 새로운 경기장에 대한 일반화 성능을 주장할 수 없다.

각 `videos` 항목은 `id`, `file`, 절대 `path`, `sha256`, `bytes`, `pair_key`, `view`, `group_id`, `split`을 제공한다. `id`는 상대 경로에서 결정되며, 동일 내용의 다른 파일도 별도 기록을 갖는다.

## 검수 라벨

```json
{
  "schema_version": 1,
  "video_id": "v_...",
  "source": "human_review",
  "status": "verified",
  "reviewer": "검수자 이름",
  "label": "IN",
  "contact_observed": true,
  "time": 1.25,
  "contact_px": [480, 620],
  "court_id": "court-1",
  "near_line_distance_m": 0.02,
  "independent_evidence": "독립 고속 카메라 영상과 확인 위치",
  "notes": "최초 접촉을 확인한 근거"
}
```

픽셀 좌표는 결과의 `analysis.width/height` 좌표계이다. 시간은 영상의 재생 타임스탬프 초 단위이다. `machine_draft`, `needs_review`, UNKNOWN, 검수자 없음, 관측한 접촉 근거 없음은 인·아웃 정확도에서 제외한다. UNKNOWN은 정답을 확인할 수 없다는 유효한 검수 결과이며 IN/OUT으로 강제 변환하지 않는다. 경계선 5cm 이내 사례는 독립 관측 근거가 없으면 정확도 평가에서 제외한다.

자동 판정 정확도는 **자동 IN/OUT을 출력한 적격 정답 사례 중 정답 비율**이다. REVIEW를 정확한 자동 판정으로 세지 않는다. 자동 판정 비율, REVIEW 비율, 오류 비율, IN/OUT/REVIEW 혼동표와 Wilson 95% 신뢰구간을 함께 제공한다. 작은 표본이나 모든 사례를 REVIEW로 보내는 시스템을 99% 정확도로 표현하지 않는다.

## 선택적 상세 정답

- 탐지: `detection_frames: [{frame_index, complete: true, objects: [{x,y,track_id}], matching_radius_px: 8}]`. 해당 프레임의 모든 셔틀콕을 표시한 경우에만 precision/recall을 계산한다. 학습용 일부 `detection_labels` 패치는 완전한 탐지 정답이 아니다.
- 추적: 위 객체의 `track_id`와 예측 궤적을 공간적으로 대응시켜 ID 변경을 센다. `reconnection_episodes: [{track_id,before_frame,after_frame}]`가 있으면 전후 대응이 확인된 가림 구간의 ID 유지율을 계산한다.
- 코트 보정: `calibration_reference: {valid: true, independent_reference: true, reference_points: [{image_px:[x,y],court_m:[x,y]}]}`. 사람이 검수한 보정 성공/실패와 독립 기준점을 이용하며 수동 보정은 자동 성공으로 세지 않는다.
- 접촉 위치: `court_position_m`, `coordinate_system: "badminton_court_m"`, `position_reference_verified: true`와 예측·정답의 일치하는 `court_id`가 필요하다. 관측하지 않은 공중 중심점이나 서로 다른 좌표계로 밀리미터 오차를 계산하지 않는다.
- 대상 셔틀콕: `target_track_id`와 사람이 확인한 `predicted_track_identity` 매핑이 있을 때만 다른 코트 선택 오류율을 계산한다.

부분 라벨, 기준 좌표 부재, 독립 자료 부재는 각각 계산 불가 또는 제외 사유로 남는다. 평가 결과 파일은 기술 검증의 측정 기록이며 대회 적용 승인이나 목표 성능 달성을 자동 보증하지 않는다.
