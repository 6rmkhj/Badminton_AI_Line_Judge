"""Render actual source frames and preserve coordinate/provenance metadata.

Run from any directory after the declared baseline/full-v5 runs exist.
No prediction is generated, corrected, interpolated, or moved by this renderer.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/landing-engine/examples/8k-2-b-recovery.jpg'
REPORT = ROOT / 'docs/landing-engine/recovery-experiments.json'
CROP_ANALYSIS = [399, 182, 442, 220]  # x0,y0,x1,y1: display-only context


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def render():
    baseline_path = ROOT / 'runs/landing-engine/baseline/8k-2-b/result.json'
    recovered_path = ROOT / 'runs/landing-engine/full-v5/8k-2-b/result.json'
    baseline = json.loads(baseline_path.read_text())
    recovered = json.loads(recovered_path.read_text())
    reference_path = ROOT / 'docs/generalization/references/8k-2-b.json'
    reference = json.loads(reference_path.read_text())
    source = ROOT / 'video' / baseline['file']
    if not source.is_file():
        source = Path(baseline['source_path'])
    source_sha = sha(source)
    assert source_sha == baseline['source_sha256'] == recovered['source_sha256']
    assert source_sha == reference['source_sha256']
    before, after = baseline['selected_landing'], recovered['selected_landing']
    assert before['frame_index'] == 12 and after['frame_index'] == 45
    assert before['contact_observed'] is False and after['contact_observed'] is False
    indices = {before['frame_index'], after['frame_index'], 44, 45, 46, 47}
    frames = {}
    capture = cv2.VideoCapture()
    capture.set(cv2.CAP_PROP_N_THREADS, 1)
    assert capture.open(str(source))
    try:
        for index in range(max(indices) + 1):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f'Missing actual source frame {index}')
            if index in indices:
                frames[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    width, height = frames[45].size
    aw, ah = recovered['analysis']['width'], recovered['analysis']['height']
    assert (baseline['analysis']['width'], baseline['analysis']['height']) == (aw, ah)
    def reference_distance(candidate):
        point = candidate['candidate_px']
        scaled = [point[0] * reference['coordinate_size'][0] / aw,
                  point[1] * reference['coordinate_size'][1] / ah]
        return math.dist(scaled, reference['landing_point_px'])
    before_distance, after_distance = reference_distance(before), reference_distance(after)
    crop_source = [round(CROP_ANALYSIS[0] * width / aw), round(CROP_ANALYSIS[1] * height / ah),
                   round(CROP_ANALYSIS[2] * width / aw), round(CROP_ANALYSIS[3] * height / ah)]
    canvas = Image.new('RGB', (1600, 1190), '#f4f6f8')
    draw = ImageDraw.Draw(canvas)
    regular = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    bold = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    def label(x, y, text, size=22, color='#273444', strong=False):
        draw.text((x, y), text, font=ImageFont.truetype(bold if strong else regular, size), fill=color)
    label(32, 20, '낙하 후보 복구: 원본 영상 비교', 34, strong=True)
    label(32, 73, '후보 위치만 표시 · 실제 첫 접촉과 IN/OUT은 미확정', 23)
    panels = []
    for x, result, candidate, title in [(32, baseline, before, '기존 선택 · f12'),
                                       (816, recovered, after, '원본 재검사 · f45')]:
        label(x, 113, title, 25, strong=True)
        point = candidate['candidate_px']
        label(x, 147, f'분석 좌표 ({point[0]:.2f}, {point[1]:.2f}) · 접촉 미확정', 19)
        image = frames[candidate['frame_index']].resize((752, 423), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, 181))
        px, py = x + point[0] * 752 / aw, 181 + point[1] * 423 / ah
        draw.ellipse((px - 13, py - 13, px + 13, py + 13), outline='#ffae36', width=4)
        panels.append({'source_frame_index': candidate['frame_index'],
            'candidate_analysis_px': point, 'candidate_source_px': [point[0] * width / aw, point[1] * height / ah],
            'display_rect_px': [x, 181, 752, 423], 'marker_display_px': [px, py],
            'marker_radius_display_px': 13, 'contact_observed': False})
    label(32, 622, '주황색 원: 알고리즘 후보 위치  |  아래 확대 영역에는 위치 표시를 덧그리지 않았습니다.', 21)
    label(32, 670, '원본 인접 프레임 · 동일 영역 확대', 26, strong=True)
    crop_panels = []
    for column, index in enumerate([44, 45, 46, 47]):
        x = 32 + column * 388
        label(x, 711, f'f{index}', 23, strong=True)
        crop = frames[index].crop(crop_source).resize((372, 329), Image.Resampling.LANCZOS)
        canvas.paste(crop, (x, 749))
        crop_panels.append({'source_frame_index': index, 'source_crop_xyxy': crop_source,
                            'display_rect_px': [x, 749, 372, 329], 'prediction_overlay': False})
    label(32, 1100, f'고정 AI 근사 참조와의 거리: {before_distance:.2f}px → {after_distance:.2f}px '
          f'(허용 {reference["tolerance_px"]}px, 분석 영상 {aw}×{ah} 기준)', 21)
    label(32, 1142, '원본 프레임을 확대·축소한 진단 자료입니다. 독립 사람 정답에 대한 접촉 또는 IN/OUT 정확도가 아닙니다.', 20)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, quality=94, subsampling=0)
    metadata = {'path': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT),
        'renderer': str(Path(__file__).relative_to(ROOT)), 'renderer_sha256': sha(__file__),
        'source_path': str(source.relative_to(ROOT)), 'source_sha256': source_sha,
        'source_size': [width, height], 'analysis_size': [aw, ah], 'figure_size': list(canvas.size),
        'reference_path': str(reference_path.relative_to(ROOT)), 'reference_sha256': sha(reference_path),
        'reference_coordinate_size': reference['coordinate_size'],
        'reference_point_px': reference['landing_point_px'],
        'reference_frame_interval': reference['landing_frame_interval'],
        'reference_tolerance_px': reference['tolerance_px'],
        'reference_distance_before_px': before_distance, 'reference_distance_after_px': after_distance,
        'baseline_result_path': str(baseline_path.relative_to(ROOT)), 'baseline_result_sha256': sha(baseline_path),
        'recovered_result_path': str(recovered_path.relative_to(ROOT)), 'recovered_result_sha256': sha(recovered_path),
        'baseline_implementation_fingerprint': baseline['implementation_fingerprint'],
        'recovered_implementation_fingerprint': recovered['implementation_fingerprint'],
        'recovered_execution_mode': recovered['benchmark_execution']['mode'],
        'overview_panels': panels, 'adjacent_context_panels': crop_panels,
        'resampling': 'Pillow LANCZOS for display only; no generated or interpolated trajectory observations',
        'scope': 'Actual source frames, selected candidate markers and source-only adjacent crops; no physical-contact or target-identity certification.'}
    report = json.loads(REPORT.read_text())
    report['source_evidence_figure'] = metadata
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'figure': str(OUT), 'source_sha256': source_sha, 'renderer_sha256': metadata['renderer_sha256']}))


if __name__ == '__main__':
    render()
