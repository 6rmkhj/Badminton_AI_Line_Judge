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


def draw_observed_trail(canvas, observations, colour, thickness=1):
    """Connect consecutive observations only; leave missing frames visible."""
    segment = []
    previous = None
    for frame, point in observations:
        if previous is not None and frame != previous + 1:
            if len(segment) > 1:
                cv2.polylines(canvas, [np.asarray(segment, dtype=np.int32)], False, colour, thickness)
            segment = []
        segment.append(point)
        previous = frame
    if len(segment) > 1:
        cv2.polylines(canvas, [np.asarray(segment, dtype=np.int32)], False, colour, thickness)


def write_trajectory_context(source_frames, timestamps, analysis_size, landing, output_dir):
    """Show the approach and rebound in a fixed, trajectory-sized source crop.

    These presentation crops are separate from the fixed native contact-model
    patches. Markers indicate observed candidate centroids, never inferred
    missing observations or a confirmed cork/floor contact.
    """
    width, height = analysis_size
    event = landing.get('image_motion_event') or {}
    observations = event.get('trajectory') or []
    if not observations:
        return None
    observations = sorted(observations, key=lambda item: item['frame_index'])
    by_frame = {int(item['frame_index']): item for item in observations}
    centre = int(landing['frame_index'])
    period = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1 / 30
    first = max(0, centre - max(2, round(.1 / period)))
    last = min(len(timestamps) - 1, centre + max(3, round(.133333 / period)))
    interval = event.get('frame_interval') or [centre, centre]
    first, last = min(first, interval[0]), max(last, interval[1])
    # A high-frame-rate occlusion can span more than twelve frames. In that
    # case show measured context on BOTH sides, with honest source labels,
    # rather than filling the strip with only the unobserved gap.
    if last - first < 12:
        indices = list(range(first, last + 1))
    else:
        before = [i for i in by_frame if i < centre][-5:]
        after = [i for i in by_frame if i > centre][:5]
        indices = sorted(set(before + [centre] + after))
    consecutive = all(b == a + 1 for a, b in zip(indices, indices[1:]))
    context = [o for o in observations if o['frame_index'] in indices]
    # A missed observation at the edge must not crop the incoming object out.
    earlier = [o for o in observations if o['frame_index'] < indices[0]]
    later = [o for o in observations if o['frame_index'] > indices[-1]]
    context = earlier[-1:] + context + later[:1]
    points = np.asarray([[o['x'], o['y']] for o in context] + [landing['candidate_px']], dtype=float)
    margin = max(12., float(landing.get('uncertainty_px', 2)) * 2)
    low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
    midpoint = (low + high) / 2
    extent = np.maximum(high - low, 48)
    low = np.maximum(0, midpoint - extent / 2)
    high = np.minimum([width, height], midpoint + extent / 2)
    tiles, source_roi, source_size = [], None, None
    tile_width, tile_height = 320, 320
    for index in indices:
        source = source_frames[index]
        sh, sw = source.shape[:2]
        scale = np.array([sw / width, sh / height])
        left, top = np.floor(low * scale).astype(int)
        right, bottom = np.ceil(high * scale).astype(int)
        right, bottom = min(sw, right), min(sh, bottom)
        crop = source[top:bottom, left:right]
        ratio = min(tile_width / crop.shape[1], (tile_height - 48) / crop.shape[0])
        resized = cv2.resize(crop, (max(1, round(crop.shape[1] * ratio)),
                                     max(1, round(crop.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
        canvas = np.full((tile_height, tile_width, 3), (22, 26, 32), dtype=np.uint8)
        xoff, yoff = (tile_width - resized.shape[1]) // 2, 48
        canvas[yoff:yoff + resized.shape[0], xoff:xoff + resized.shape[1]] = resized

        def display_point(point):
            return tuple(np.rint((np.asarray(point) * scale - [left, top]) * ratio
                                 + [xoff, yoff]).astype(int))

        if index in by_frame:
            observation = by_frame[index]
            point = display_point([observation['x'], observation['y']])
            cv2.circle(canvas, point, 8, (255, 220, 50), 1, cv2.LINE_AA)
        if index == centre:
            cv2.drawMarker(canvas, display_point(landing['candidate_px']), (40, 100, 255),
                           cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FRAME {index}  |  {timestamps[index]:.3f}s", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'MOTION ESTIMATE - CONTACT UNCONFIRMED' if consecutive else
                    'SAMPLED CONTEXT - CONTACT UNCONFIRMED', (10, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, .34, (80, 200, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
        source_roi, source_size = [int(left), int(top), int(right), int(bottom)], [sw, sh]
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], (22, 26, 32)))
    strip = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    relative = 'evidence/trajectory-context.jpg'
    destination = Path(output_dir) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), strip, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError('Could not write trajectory context evidence')
    return {'path': relative, 'frame_indices': indices, 'times': [timestamps[i] for i in indices],
            'analysis_size': {'width': width, 'height': height}, 'source_size': source_size,
            'analysis_roi': [*low.tolist(), *high.tolist()], 'source_roi': source_roi,
            'sampling': 'consecutive' if consecutive else 'observed_context_with_inferred_frame',
            'contact_observed': False,
            'note': 'Fixed source crop spanning the observed approach and rebound; image motion estimates contact only.'}


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
    shown = ordered[:max(1, min(40, int(cfg.get('max_overlay_tracks', 1 if landing else 12))))]
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
            if polygon and calibration.get('status') == 'ok':
                cv2.polylines(canvas, [np.rint(polygon).astype(np.int32)], True, (0, 190, 255), 1)
            for identifier, observation in indexed[index]:
                trails[identifier].append((index, (round(observation['x']), round(observation['y']))))
            for identifier in list(trails):
                trails[identifier] = [(frame, xy) for frame, xy in trails[identifier] if index - frame <= 12]
                points = np.asarray([xy for _, xy in trails[identifier]], dtype=np.int32)
                if not len(points):
                    continue
                colour = (70, 150, 255) if identifier == selected_id else (150, 230, 90)
                draw_observed_trail(canvas, trails[identifier], colour)
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
