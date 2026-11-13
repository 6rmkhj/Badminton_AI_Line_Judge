"""Render original frames for the frozen ordinary-1-a sequence comparison.

Requires OpenCV and Pillow. Run after baseline/causal-v4 and
summarize_sequence_experiments.py exist.
Markers reproduce saved candidate coordinates. The first-vicinity marker is an
inferred centre, not a source detection or a certified physical contact. Crops
are display-only and never feed inference or change the frozen references.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/landing-engine/examples/ordinary-1-a-sequence.jpg'
REPORT = ROOT / 'docs/landing-engine/causal-family-experiments.json'
CONTEXTS = [
    {'title': '첫 후보 주변 · 원본 확대 (위치 표시 없음)',
     'frames': [156, 158, 159, 162], 'crop_analysis_xyxy': [171, 292, 249, 350], 'top': 708},
    {'title': '후속 반동 주변 · 원본 확대 (위치 표시 없음)',
     'frames': [185, 187, 189, 192], 'crop_analysis_xyxy': [143, 308, 221, 366], 'top': 1100},
]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def render():
    baseline_path = ROOT / 'runs/landing-engine/baseline/ordinary-1-a/result.json'
    improved_path = ROOT / 'runs/landing-engine/causal-v4/ordinary-1-a/result.json'
    baseline, improved = [json.loads(path.read_text()) for path in (baseline_path, improved_path)]
    reference_path = ROOT / 'docs/remediation/new8_references/ordinary-1-a.json'
    reference = json.loads(reference_path.read_text())
    source = ROOT / 'video' / baseline['file']
    source_sha = sha(source)
    assert source_sha == baseline['source_sha256'] == improved['source_sha256'] == reference['source_sha256']
    before, after = baseline['selected_landing'], improved['selected_landing']
    assert before['frame_index'] == 187 and after['frame_index'] == 158
    assert before['image_motion_event']['point_kind'] == 'observed_object_center'
    assert after['image_motion_event']['point_kind'] == 'inferred_object_center'
    assert before['contact_observed'] is False and after['contact_observed'] is False
    indices = {187, 158, *(frame for context in CONTEXTS for frame in context['frames'])}
    frames, frame_hashes = {}, {}
    capture = cv2.VideoCapture()
    capture.set(cv2.CAP_PROP_N_THREADS, 1)
    if not capture.open(str(source)):
        raise RuntimeError('Cannot open original source video')
    try:
        for index in range(max(indices) + 1):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f'Cannot decode actual source frame {index}')
            if index in indices:
                frame_hashes[str(index)] = hashlib.sha256(frame.tobytes()).hexdigest()
                frames[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    width, height = frames[158].size
    aw, ah = improved['analysis']['width'], improved['analysis']['height']
    assert (baseline['analysis']['width'], baseline['analysis']['height']) == (aw, ah)
    assert (width, height) == (baseline['metadata']['width'], baseline['metadata']['height'])

    def distance(candidate):
        point = candidate['candidate_px']
        return math.dist([point[0] * reference['coordinate_size'][0] / aw,
                          point[1] * reference['coordinate_size'][1] / ah], reference['landing_point_px'])

    canvas = Image.new('RGB', (1600, 1580), '#f4f6f8')
    draw = ImageDraw.Draw(canvas)
    regular = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    bold = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

    def label(x, y, text, size=22, color='#273444', strong=False):
        draw.text((x, y), text, font=ImageFont.truetype(bold if strong else regular, size), fill=color)

    label(32, 18, '첫 낙하 후보 순서 개선: 원본 영상 비교', 34, strong=True)
    label(32, 72, '관측 중심과 추정 중심을 구분합니다. 실제 코르크 접촉과 IN/OUT은 미확정입니다.', 23)
    panels = []
    for x, candidate, title, subtitle, color, inferred in [
            (32, before, '기존 선택 · 후속 반동 부근 f187', '관측 물체 중심 후보 · 접촉 미확정', '#ffae36', False),
            (816, after, '개선 선택 · 첫 낙하 부근 추정 f158', '추적 공백의 추정 중심 · 실측점 아님', '#55b9ff', True)]:
        label(x, 114, title, 25, strong=True)
        label(x, 151, subtitle, 20)
        image = frames[candidate['frame_index']].resize((752, 423), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, 185))
        point = candidate['candidate_px']
        px, py = x + point[0] * 752 / aw, 185 + point[1] * 423 / ah
        bbox = (px - 15, py - 15, px + 15, py + 15)
        if inferred:
            for angle in range(0, 360, 45):
                draw.arc(bbox, angle, angle + 27, fill=color, width=4)
        else:
            draw.ellipse(bbox, outline=color, width=4)
        panels.append({'source_frame_index': candidate['frame_index'], 'timestamp_seconds': candidate['time'],
            'candidate_analysis_px': point, 'candidate_source_px': [point[0] * width / aw, point[1] * height / ah],
            'point_kind': candidate['image_motion_event']['point_kind'],
            'display_rect_px': [x, 185, 752, 423], 'marker_display_px': [px, py],
            'marker_radius_display_px': 15, 'marker_style': 'dashed_outline' if inferred else 'solid_outline',
            'marker_semantics': 'saved candidate location, not an uncertainty radius or fabricated observation',
            'contact_observed': False})
    label(32, 625, '주황 실선: 관측 중심 후보  |  파란 점선: 추정 중심 (f158의 실제 검출점이 아닙니다)', 22)
    label(32, 661, '추적 자료의 관측 공백 f146–158 · 원본 영상은 존재하며, 아래에 실제 프레임을 그대로 확대했습니다.', 21)
    crop_panels = []
    for context in CONTEXTS:
        top = context['top']
        label(32, top, context['title'], 25, strong=True)
        crop = context['crop_analysis_xyxy']
        crop_source = [round(crop[0] * width / aw), round(crop[1] * height / ah),
                       round(crop[2] * width / aw), round(crop[3] * height / ah)]
        for column, index in enumerate(context['frames']):
            x = 32 + column * 388
            label(x, top + 42, f'f{index} · {baseline["timestamps"][index]:.3f}s', 21, strong=True)
            tile = frames[index].crop(crop_source).resize((372, 277), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x, top + 80))
            crop_panels.append({'source_frame_index': index, 'timestamp_seconds': baseline['timestamps'][index],
                'analysis_crop_xyxy': crop, 'source_crop_xyxy': crop_source,
                'display_rect_px': [x, top + 80, 372, 277], 'prediction_overlay': False})
    label(32, 1480, f'고정 AI 근사 참조: 중심 오차 {distance(before):.2f}px → {distance(after):.2f}px '
          f'· 허용 {reference["tolerance_px"]}px / f158–160 (분석 영상 {aw}×{ah})', 21)
    label(32, 1522, '첫 낙하 부근의 근사 중심 비교이며, 독립적인 사람 정답에 대한 접촉·IN/OUT 정확도가 아닙니다.', 19)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, quality=94, subsampling=0)
    metadata = {'path': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT),
        'renderer': str(Path(__file__).relative_to(ROOT)), 'renderer_sha256': sha(__file__),
        'source_path': str(source.relative_to(ROOT)), 'source_sha256': source_sha,
        'source_size': [width, height], 'analysis_size': [aw, ah], 'figure_size': list(canvas.size),
        'timestamp_origin_seconds': baseline['metadata'].get('time_origin_seconds'),
        'timestamp_semantics': 'Saved normalized source timestamps; original PTS origin is recorded separately.',
        'decoded_bgr_frame_sha256': frame_hashes, 'decoder': {'library': 'opencv', 'version': cv2.__version__,
            'frame_selection': 'sequential decode from frame zero, no seeking or frame interpolation', 'threads': 1},
        'renderer_environment': {'pillow_version': PILLOW_VERSION,
            'fonts': [{'path': font, 'sha256': sha(font)} for font in (regular, bold)]},
        'reference_path': str(reference_path.relative_to(ROOT)), 'reference_sha256': sha(reference_path),
        'reference_coordinate_size': reference['coordinate_size'], 'reference_point_px': reference['landing_point_px'],
        'reference_frame_interval': reference['landing_frame_interval'], 'reference_tolerance_px': reference['tolerance_px'],
        'reference_distance_before_px': distance(before), 'reference_distance_after_px': distance(after),
        'baseline_result_path': str(baseline_path.relative_to(ROOT)), 'baseline_result_sha256': sha(baseline_path),
        'improved_result_path': str(improved_path.relative_to(ROOT)), 'improved_result_sha256': sha(improved_path),
        'baseline_implementation_fingerprint': baseline['implementation_fingerprint'],
        'improved_implementation_fingerprint': improved['implementation_fingerprint'],
        'improved_execution_mode': improved['benchmark_execution']['mode'],
        'estimated_event_time_seconds': after['image_motion_event']['estimated_time'],
        'missing_track_observation_interval': [146, 158],
        'overview_panels': panels, 'adjacent_context_panels': crop_panels,
        'crop_selection_policy': 'Display-only neighborhoods chosen after frozen experiments; these rectangles are never inference inputs.',
        'resampling': 'Pillow LANCZOS for display-only resizing; all crops are actual source pixels without prediction overlays',
        'scope': 'Actual source frames and unchanged saved candidate markers. The f158 coordinate is inferred, not a measured source observation. No physical contact or target identity is certified.'}
    if not REPORT.is_file():
        raise RuntimeError('Create causal-family-experiments.json before attaching figure provenance')
    report = json.loads(REPORT.read_text())
    report['source_evidence_figure'] = metadata
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'figure': str(OUT), 'source_sha256': source_sha, 'renderer_sha256': metadata['renderer_sha256']}))


if __name__ == '__main__':
    render()
