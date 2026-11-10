"""Compare equal-budget proposals on real, AI-reviewed difficult source windows.

Regions are derived from the reviewed points themselves. This measures whether
an explicitly assisted search can retain visible proposals, not generalization,
whole-video recall, independent human accuracy, or IN/OUT correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from badminton.detection import detect_sequence
from badminton.media import SourceFrameSequence, file_hash, probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('runs/ai-reviewed-poc/visual-review.json'))
    parser.add_argument('--output', type=Path, default=Path('runs/review-remediation/pipeline/candidate-benchmark.json'))
    parser.add_argument('--limit-videos', type=int)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    report_bytes = args.report.read_bytes()
    report = json.loads(report_bytes)
    detector_path = Path(__file__).resolve().parents[1] / 'badminton/detection.py'
    detector_digest = hashlib.sha256(detector_path.read_bytes()).hexdigest()
    selected = {view['video_id'] for pair in report['filename_pair_audit']['items']
                if pair['category'] == 'both_review' for view in pair['views']}
    totals = {mode: Counter() for mode in ('baseline', 'dark', 'focused', 'focused_spatial')}
    records = []
    for video in report['videos']:
        if video['video_id'] not in selected:
            continue
        annotation = video['annotation']
        points = [point for point in annotation.get('detection_labels', []) if point['label'] == 'shuttle']
        if not points:
            continue
        if args.limit_videos is not None and len(records) >= args.limit_videos:
            break
        source = Path(video['source_integrity']['path'])
        digest = file_hash(source)
        if digest != video['source_sha256']:
            raise ValueError('Original source hash differs: '+str(source))
        metadata = probe(source)
        first = max(0, min(point['frame_index'] for point in points)-5)
        last = min(len(metadata['timestamps'])-1, max(point['frame_index'] for point in points)+5)
        size = annotation['analysis_size']
        width, height = size['width'], size['height']
        raw = []
        with SourceFrameSequence(source, len(metadata['timestamps'])) as sequence:
            for index in range(first, last+1):
                raw.append(sequence[index])
        frames = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in raw]
        timestamps = metadata['timestamps'][first:last+1]
        x0 = max(0, min(point['x'] for point in points)-24)
        y0 = max(0, min(point['y'] for point in points)-24)
        x1 = min(width, max(point['x'] for point in points)+24)
        y1 = min(height, max(point['y'] for point in points)+24)
        region = {'frame_start': 0, 'frame_end': len(frames)-1, 'bbox': [x0, y0, x1-x0, y1-y0]}
        common = {'person_backend': 'none', 'source_frames': raw, 'max_candidates': 64,
                  'highres_max_rois': 8, 'highres_roi_size': 192, 'highres_exploration_rois': 2}
        modes = {'baseline': {}, 'dark': {'include_dark_components': True},
                 'focused': {'include_dark_components': True, 'refinement_regions': [region]},
                 'focused_spatial': {'include_dark_components': True, 'refinement_regions': [region],
                                     'candidate_selection': 'spatial'}}
        result = {'video_id': video['video_id'], 'file': video['file'], 'source_sha256': digest,
                  'source_size': [raw[0].shape[1], raw[0].shape[0]], 'analysis_size': [width, height],
                  'source_frame_interval': [first, last], 'assisted_region': region,
                  'ai_positive_point_count': len(points), 'modes': {}}
        for name, mode in modes.items():
            started = time.perf_counter()
            detections = detect_sequence(frames, timestamps, {**common, **mode})
            elapsed = time.perf_counter()-started
            matches = []
            for point in points:
                row = detections[point['frame_index']-first]
                proposals = row['detections']
                nearest = min(proposals, key=lambda item: math.hypot(item['x']-point['x'], item['y']-point['y']), default=None)
                distance = math.hypot(nearest['x']-point['x'], nearest['y']-point['y']) if nearest else None
                hit = distance is not None and distance <= 6.0
                totals[name]['points'] += 1
                totals[name]['points_with_proposal'] += int(hit)
                matches.append({'source_frame_index': point['frame_index'], 'ai_point_px': [point['x'], point['y']],
                                'nearest_distance_px': distance, 'within_6px': hit,
                                'nearest_method': nearest.get('method') if nearest else None,
                                'nearest_native': bool(nearest and nearest.get('native_resolution_observation')),
                                'nearest_roi_origin': nearest.get('roi_origin') if nearest else None})
            counts = [len(row['detections']) for row in detections]
            assert max(counts, default=0) <= 64
            assert all(row['high_resolution']['roi_count'] <= 8 for row in detections)
            totals[name]['frames'] += len(frames)
            totals[name]['proposals'] += sum(counts)
            result['modes'][name] = {'detector_seconds': elapsed, 'proposals': sum(counts),
                                     'frames': len(frames), 'matches': matches}
            print(json.dumps({'file': video['file'], 'mode': name, 'seconds': round(elapsed, 3),
                              'point_hits': sum(item['within_6px'] for item in matches),
                              'points': len(points)}, ensure_ascii=False), flush=True)
        records.append(result)
        del frames, raw, detections
    output = {'created_at': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
              'candidate_limit': 64, 'native_roi_limit': 8, 'native_roi_size': 192,
              'person_backend': 'none', 'match_radius_px': 6.0, 'preprocessing': 'opencv_source_bgr_area_v1',
              'evaluation_points_used_to_define_search_region': True,
              'independent_human_ground_truth': False, 'automatic_judgment_accuracy': None,
              'default_detector_configuration_changed': False,
              'detector_sha256': detector_digest,
              'report_sha256': hashlib.sha256(report_bytes).hexdigest(),
              'video_count': len(records), 'counts': {name: dict(counts) for name, counts in totals.items()},
              'videos': records}
    if hashlib.sha256(detector_path.read_bytes()).hexdigest() != detector_digest:
        raise ValueError('Detector source changed during benchmark; rerun on frozen code')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(output['counts'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
