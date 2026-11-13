"""Annotated, timestamp-preserving replay of a bounded candidate interval."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from .media import probe, repair_preview_tail, _FrameProgress, _run_ffmpeg_progress


def write_annotated_clip(frames, timestamps, calibration, tracks, landing, decision,
                         output_dir, config=None, *, progress_callback=None):
    cfg = dict(config or {})
    if not frames or len(frames) != len(timestamps):
        raise ValueError('Evidence requires one timestamp per frame')
    before = max(0., min(5., float(cfg.get('seconds_before', .7))))
    after = max(0., min(5., float(cfg.get('seconds_after', .7))))
    centre = float(landing['time']) if landing else float(timestamps[len(frames) // 2])
    indices = [i for i, stamp in enumerate(timestamps) if centre - before <= stamp <= centre + after]
    if not indices:
        indices = [min(range(len(frames)), key=lambda i: abs(timestamps[i] - centre))]
    # No temporal subsampling: each replay frame refers to an actual source frame.
    output = Path(output_dir) / 'evidence'
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'annotated.mp4'
    selected_id = str(landing.get('track_id')) if landing else None
    ordered = sorted(tracks, key=lambda track: (str(track['track_id']) != selected_id,
                                                -len(track.get('observations', []))))
    shown = ordered[:max(1, min(40, int(cfg.get('max_overlay_tracks', 12))))]
    indexed = defaultdict(list)
    for track in shown:
        for observation in track.get('observations', []):
            indexed[int(observation['frame_index'])].append((str(track['track_id']), observation))
    trails = defaultdict(list)
    period = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1 / 30
    origin = timestamps[indices[0]]
    timeline = []
    progress = _FrameProgress(progress_callback, len(indices), '궤적·착지 근거 프레임 생성')
    progress.report(0, force=True)
    with tempfile.TemporaryDirectory(prefix='badminton-evidence-') as temporary:
        temporary = Path(temporary)
        concat = ['ffconcat version 1.0']
        for ordinal, index in enumerate(indices):
            canvas = frames[index].copy()
            height, width = canvas.shape[:2]
            polygon = calibration.get('polygon_px')
            if polygon:
                cv2.polylines(canvas, [np.rint(polygon).astype(np.int32)], True, (0, 190, 255), 1)
            for identifier, observation in indexed[index]:
                trails[identifier].append((index, (round(observation['x']), round(observation['y']))))
            for identifier in list(trails):
                trails[identifier] = [(frame, xy) for frame, xy in trails[identifier] if index - frame <= 12]
                points = np.asarray([xy for _, xy in trails[identifier]], dtype=np.int32)
                if not len(points):
                    continue
                colour = (70, 150, 255) if identifier == selected_id else (150, 230, 90)
                if len(points) > 1:
                    cv2.polylines(canvas, [points], False, colour, 1)
                if trails[identifier][-1][0] == index:
                    cv2.circle(canvas, tuple(points[-1]), 4, colour, 1)
            if landing and abs(timestamps[index] - landing['time']) <= max(period, .08):
                point = landing.get('contact_px') or landing.get('candidate_px')
                if point:
                    cv2.circle(canvas, tuple(np.rint(point).astype(int)), 12, (40, 80, 255), 2)
            scale = max(.3, min(.6, width / 1600))
            cv2.rectangle(canvas, (0, 0), (width, min(height, 52)), (22, 26, 32), -1)
            cv2.putText(canvas, f"{decision.get('decision', 'REVIEW')} | SOURCE {timestamps[index]:.6f}s | FRAME {index}",
                        (8, 19), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(canvas, 'CANDIDATES / CONTACT REQUIRES OBSERVATION', (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, scale * .8, (80, 200, 255), 1, cv2.LINE_AA)
            name = f'frame-{ordinal:04d}.jpg'
            if not cv2.imwrite(str(temporary / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError('Could not write replay evidence frame')
            duration = timestamps[indices[ordinal + 1]] - timestamps[index] if ordinal + 1 < len(indices) else period
            concat.extend([f"file '{name}'", 'option framerate 1000000', f'duration {duration:.9f}'])
            timeline.append({'replay_frame_index': ordinal, 'source_frame_index': index,
                             'source_time': timestamps[index], 'replay_time': timestamps[index] - origin})
            progress.report(ordinal + 1)
        progress.report(len(indices), force=True)
        manifest = temporary / 'frames.ffconcat'
        manifest.write_text('\n'.join(concat) + '\n', encoding='utf-8')
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-y', '-v', 'error', '-safe', '0',
                   '-f', 'concat', '-i', str(manifest), '-an', '-c:v', 'libx264',
                   '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p', '-threads', '1',
                   '-fps_mode', 'passthrough', '-enc_time_base', '1:1000000',
                   '-bsf:v', f'setts=pts=PTS:dts=DTS:duration=1/({1 / period:.9f}*TB)',
                   '-movflags', '+faststart', str(destination)]
        if progress_callback is None:
            subprocess.run(command, check=True, capture_output=True, timeout=180)
        else:
            encoded = _run_ffmpeg_progress(command, progress_callback, len(indices), '근거 영상 인코딩', 180)
            encoded.check_returncode()
    if progress_callback is not None:
        progress_callback({'detail': '근거 영상 프레임·시각 검증'})
    repair_preview_tail(destination, 1 / period)
    encoded = probe(destination)
    if encoded['frame_count'] != len(timeline) or any(
            abs(actual - expected['replay_time']) > .0001
            for actual, expected in zip(encoded['timestamps'], timeline)):
        raise ValueError('Annotated evidence lost source frame/timestamp alignment')
    timing = output / 'annotated-timing.json'
    timing.write_text(json.dumps({'source_time_origin': origin, 'frames': timeline}, indent=2) + '\n')
    return {'path': 'evidence/annotated.mp4', 'timing_path': 'evidence/annotated-timing.json',
            'frame_count': len(timeline), 'source_start_frame': indices[0], 'source_end_frame': indices[-1],
            'source_time_origin': origin, 'timestamp_alignment_max_error_seconds': max(
                abs(actual - expected['replay_time']) for actual, expected in zip(encoded['timestamps'], timeline)),
            'overlay_track_count': len(shown), 'total_track_count': len(tracks),
            'note': 'Candidate overlays and native timestamps are evidence for review, not ground-truth contacts.'}
