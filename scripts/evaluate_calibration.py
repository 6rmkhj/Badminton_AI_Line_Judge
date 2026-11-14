#!/usr/bin/env python3
"""Replay court detection on a fixed, hash-checked set of video frame samples.

The optional visible-line references are approximate AI image annotations. Their
coverage measures image-line recovery, not court identities, metric precision,
contact accuracy, or IN/OUT accuracy. References never enter calibration.

Example:
  python scripts/evaluate_calibration.py --manifest docs/court-calibration/manifest.json \
      --cache-root runs/court-calibration/cache --output runs/court-calibration/current

If a frame cache is absent, the original video must exist under --video-root
(default: repository root), or at CACHE_ROOT/CASE_KEY/source.mp4. FFmpeg extracts
only the frozen frame indices, so decoding has the same orientation semantics
as the production media pipeline. --module can load an immutable baseline copy
of badminton/calibration.py for a controlled before/after comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_frames(case, cache_root, video_root):
    folder = cache_root / case['key']
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / 'frames.npz'
    if not cache.exists():
        video = video_root / case['source_path']
        if not video.is_file():
            video = folder / 'source.mp4'
        if not video.is_file() or sha(video) != case['source_sha256']:
            raise ValueError(f"Original video missing or SHA256 mismatch: {case['source_path']}")
        width, height = case['analysis_size']
        select = '+'.join('eq(n\\,%d)' % n for n in case['frame_indices'])
        command = ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(video),
                   '-vf', f'select={select},scale={width}:{height}', '-vsync', '0',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
        raw = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(len(case['frame_indices']), height, width, 3)
        np.savez_compressed(cache, frames=frames)
    with np.load(cache, allow_pickle=False) as data:
        frames = data['frames']
    width, height = case['analysis_size']
    if frames.shape != (len(case['frame_indices']), height, width, 3) or frames.dtype != np.uint8:
        raise ValueError(f"Unexpected cached frame format: {case['key']}")
    if hashlib.sha256(frames.tobytes()).hexdigest() != case['frames_sha256']:
        raise ValueError(f"Cached frame SHA256 mismatch: {case['key']}")
    return list(frames)


def visible_line_coverage(segments, references, tolerance=5.0):
    """Sample each reference against finite detected segments, in image pixels."""
    if not references:
        return None
    detected = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    rows = []
    for reference in references:
        points = np.linspace(reference[0], reference[1], 64)
        if not len(detected):
            errors = np.full(len(points), np.inf)
        else:
            start = detected[:, 0]
            vector = detected[:, 1] - start
            denominator = np.maximum((vector * vector).sum(axis=1), 1e-9)
            fraction = np.clip(((points[:, None] - start) * vector).sum(axis=2) / denominator, 0, 1)
            nearest = start + fraction[:, :, None] * vector
            errors = np.linalg.norm(points[:, None] - nearest, axis=2).min(axis=1)
        rows.append({'reference_segment_px': reference,
                     'sample_coverage': float(np.mean(errors <= tolerance)),
                     'median_distance_px': float(np.median(errors)) if np.isfinite(errors).all() else None})
    return {'tolerance_px': tolerance, 'samples_per_segment': 64,
            'mean_sample_coverage': float(np.mean([r['sample_coverage'] for r in rows])),
            'segments_at_least_80_percent_covered': sum(r['sample_coverage'] >= .8 for r in rows),
            'reference_segment_count': len(rows), 'segments': rows}


def annotate(background, result, references):
    image = background.copy()
    for reference in references:
        a, b = np.rint(reference).astype(int)
        cv2.line(image, tuple(a), tuple(b), (255, 0, 255), 1, cv2.LINE_AA)
    for segment in result.get('line_segments_px', []):
        a, b = np.rint(np.asarray(segment).reshape(2, 2)).astype(int)
        cv2.line(image, tuple(a), tuple(b), (0, 230, 80), 2, cv2.LINE_AA)
    if result.get('polygon_px') is not None:
        polygon = np.rint(result['polygon_px']).astype(np.int32)
        cv2.polylines(image, [polygon], True, (0, 150, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (20, 20, 20), -1)
    cv2.putText(image, f"{result['status']} | detected segments: {len(result.get('line_segments_px', []))} | green: detected; orange: fit hypothesis", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def non_court_clutter(segments, regions):
    """Measure detected length in explicit negative regions, not full precision."""
    if not regions:
        return None
    contours = [np.asarray(region, np.float32) for region in regions]
    total_length = negative_length = 0.0
    wholly_inside = 0
    for segment in segments:
        start, end = np.asarray(segment, float).reshape(2, 2)
        length = float(np.linalg.norm(end-start))
        points = np.linspace(start, end, 64)
        inside = [any(cv2.pointPolygonTest(contour, tuple(point), False) >= 0
                      for contour in contours) for point in points]
        total_length += length
        negative_length += length * float(np.mean(inside))
        wholly_inside += all(inside)
    return {'wholly_in_non_court_region_segment_count': wholly_inside,
            'non_court_detected_length_px': round(negative_length, 4),
            'total_detected_length_px': round(total_length, 4),
            'non_court_length_fraction': negative_length / total_length if total_length else 0.0}


def run(args):
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if args.module:
        spec = importlib.util.spec_from_file_location('badminton.calibration_benchmark', args.module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from badminton import calibration as module
    module_hash = sha(module.__file__)
    cv2.setNumThreads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in manifest['cases']:
        frames = load_frames(case, args.cache_root, args.video_root)
        started = time.perf_counter()
        result = module.calibrate(frames)
        elapsed = time.perf_counter() - started
        references = case.get('visible_line_references_px', [])
        coverage = visible_line_coverage(result.get('line_segments_px', []), references)
        clutter = non_court_clutter(result.get('line_segments_px', []), case.get('non_court_regions_px', []))
        background_count = result.get('background_frame_count', min(5, len(frames)))
        background_indices = result.get('background_frame_indices')
        if background_indices is None:
            background_indices = np.linspace(0, len(frames)-1, background_count).astype(int)
        background = np.median(np.stack([frames[int(i)] for i in background_indices]), axis=0).astype(np.uint8)
        overlay = annotate(background, result, references)
        cv2.imwrite(str(args.output / f"{case['key']}-overlay.jpg"), overlay)
        cv2.imwrite(str(args.output / f"{case['key']}-background.jpg"), background)
        if hasattr(module, 'segment_lines'):
            mask, _ = module.segment_lines(background)
            cv2.imwrite(str(args.output / f"{case['key']}-mask.png"), mask)
        record = {'key': case['key'], 'source_path': case['source_path'],
                  'source_sha256': case['source_sha256'], 'frames_sha256': case['frames_sha256'],
                  'elapsed_seconds': round(elapsed, 4), 'line_coverage': coverage,
                  'non_court_clutter': clutter, 'calibration': result}
        records.append(record)
        (args.output / f"{case['key']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(case['key'], result['status'], len(result.get('line_segments_px', [])), result.get('reason_codes'), flush=True)
    if sha(module.__file__) != module_hash:
        raise ValueError('Calibration source changed during replay; rerun from an immutable source snapshot.')
    report = {'schema_version': 1,
              'scope': manifest.get('scope'), 'reference_limitations': manifest.get('reference_limitations'),
              'module_sha256': module_hash, 'module_path': str(module.__file__),
              'mask_image_scope': 'Diagnostic single-background candidate mask; calibration may select a colour and aggregate temporal support separately.',
              'manifest_sha256': sha(args.manifest), 'opencv_version': cv2.__version__,
              'summary': {'video_count': len(records),
                          'full_court_fits': sum(r['calibration']['status'] == 'ok' for r in records),
                          'videos_with_detected_lines': sum(bool(r['calibration'].get('line_segments_px')) for r in records),
                          'wholly_non_court_segments': sum((r['non_court_clutter'] or {}).get('wholly_in_non_court_region_segment_count', 0) for r in records),
                          'non_court_detected_length_px': round(sum((r['non_court_clutter'] or {}).get('non_court_detected_length_px', 0) for r in records), 3),
                          'total_detected_length_px': round(sum((r['non_court_clutter'] or {}).get('total_detected_length_px', 0) for r in records), 3),
                          'visible_line_sample_coverage': float(np.mean([r['line_coverage']['mean_sample_coverage'] for r in records if r['line_coverage']])) if any(r['line_coverage'] for r in records) else None},
              'cases': records}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, default=ROOT / 'runs/court-calibration/cache')
    parser.add_argument('--video-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', type=Path)
    run(parser.parse_args())
