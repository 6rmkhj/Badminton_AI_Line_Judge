"""Compose the complete offline analysis, evidence, and review workflow."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock
import cv2
import numpy as np
from . import __version__
from .media import prepare_preview, load_frames, source_crop, SourceFrameSequence


def implementation_fingerprint():
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ('media.py', 'pipeline.py', 'calibration.py', 'detection.py', 'tracking.py',
                 'events.py', 'decision.py', 'learning.py', 'contact_model.py', 'sync.py'):
        digest.update(name.encode())
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def aggregate_decision(decisions):
    if len(decisions) == 1:
        return decisions[0]
    reasons = sorted({reason for d in decisions for reason in d.get('reason_codes', [])})
    reasons.append('MULTIPLE_LANDING_CANDIDATES' if decisions else 'NO_LANDING_CANDIDATE')
    return {'decision': 'REVIEW', 'reason_codes': sorted(set(reasons)), 'confidence': None,
            'note': 'A unique, observed target landing is required for automatic IN/OUT.'}


def _draw_frame(frame, calibration, tracks, frame_index, landing=None):
    canvas = frame.copy()
    polygon = calibration.get('polygon_px')
    if polygon:
        cv2.polylines(canvas, [np.asarray(polygon, dtype=np.int32)], True, (0, 210, 240), 2)
    for track in tracks:
        points = [o for o in track.get('observations', []) if 0 <= frame_index - o['frame_index'] <= 12]
        if not points:
            continue
        pts = np.array([[o['x'], o['y']] for o in points], dtype=np.int32)
        color = ((int(track['track_id']) * 53) % 200 + 55, 220, 100) if str(track['track_id']).isdigit() else (200, 220, 100)
        if len(pts) > 1:
            cv2.polylines(canvas, [pts], False, color, 1)
        x, y = pts[-1]
        cv2.circle(canvas, (int(x), int(y)), 5, color, 1)
        cv2.putText(canvas, str(track['track_id']), (int(x)+6, int(y)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    if landing and landing.get('candidate_px'):
        x, y = map(int, landing['candidate_px'])
        cv2.circle(canvas, (x, y), 20, (20, 80, 255), 2)
    cv2.putText(canvas, 'RESEARCH CANDIDATES - NOT VERIFIED CONTACT', (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 200, 255), 2)
    return canvas


def analyze_video(video_path, output_dir, config=None):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.analysis.lock'), timeout=1800):
        return _analyze_video(video_path, output, config)


def _analyze_video(video_path, output_dir, config=None):
    from .calibration import calibrate
    from .detection import detect_sequence, summarize_detection_quality
    from .tracking import track_detections
    from .events import estimate_landings
    from .decision import decide
    config = dict(config or {})
    source, output = Path(video_path).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    video_id = config.get('video_id') or 'v_' + hashlib.sha256(source.name.encode()).hexdigest()[:12]
    start = time.perf_counter()
    timings = {}
    metadata, presentation, source_sha = prepare_preview(source, output / 'preview.mp4', config.get('analysis_width', 1280))
    detector_config = dict(config.get('detector') or {})
    event_config = dict(config.get('events') or {})
    learned_input = bool(detector_config.get('model_path') or detector_config.get('scorer_path') or event_config.get('contact_model_path'))
    if learned_input:
        frames = load_frames(source, presentation['frame_count'], config.get('max_frames', 3000),
                             resize_to=(presentation['width'], presentation['height']))
        preprocessing = 'opencv_source_bgr_area_v1'
    else:
        frames = load_frames(output / 'preview.mp4', presentation['frame_count'], config.get('max_frames', 3000))
        preprocessing = 'h264_preview_crf21_bgr_v1'
    timestamps = presentation.pop('timestamps')
    metadata.pop('timestamps', None)
    timings['decode_and_preview_seconds'] = time.perf_counter() - start
    clock = time.perf_counter()
    calibration_frames = [frames[int(i)] for i in np.linspace(0, len(frames)-1, min(9, len(frames)))]
    calibration = calibrate(calibration_frames, config.get('court_hint'))
    timings['calibration_seconds'] = time.perf_counter() - clock
    clock = time.perf_counter()
    detector_config['source_path'] = str(source)
    detector_config['frame_preprocessing'] = preprocessing
    source_frames = None
    if detector_config.get('backend') == 'onnx':
        source_frames = SourceFrameSequence(source, len(frames))
        detector_config['source_frames'] = source_frames
    try:
        detections = detect_sequence(frames, timestamps, detector_config)
    finally:
        if source_frames is not None:
            source_frames.close()
    timings['detection_seconds'] = time.perf_counter() - clock
    clock = time.perf_counter()
    tracks = track_detections(detections, config.get('tracker'))
    timings['tracking_seconds'] = time.perf_counter() - clock
    clock = time.perf_counter()
    event_config['frame_preprocessing'] = preprocessing
    landings = estimate_landings(tracks, frames, timestamps, calibration, event_config)
    for i, landing in enumerate(landings):
        landing['event_id'] = f'event_{i:03d}'
        landing['decision'] = decide(landing, calibration, config.get('rules'))
    decision = aggregate_decision([landing['decision'] for landing in landings])
    timings['events_and_decision_seconds'] = time.perf_counter() - clock
    selected = max(landings, key=lambda x: x.get('confidence', 0), default=None)
    artifacts = {'preview': 'preview.mp4', 'detections': 'detections.json', 'evidence': []}
    evidence_dir = output / 'evidence'
    evidence_dir.mkdir(exist_ok=True)
    # Preserve several candidate contexts, including original-resolution crops.
    for landing in sorted(landings, key=lambda x: x.get('confidence', 0), reverse=True)[:6]:
        index = max(0, min(len(frames)-1, int(landing.get('frame_index', 0))))
        filename = f"{landing['event_id']}.jpg"
        cv2.imwrite(str(evidence_dir / filename), _draw_frame(frames[index], calibration, tracks, index, landing))
        evidence = {'event_id': landing['event_id'], 'frame_index': index, 'time': timestamps[index],
                    'image': 'evidence/' + filename}
        center = landing.get('candidate_px')
        if center and len(artifacts['evidence']) < int(config.get('source_crop_count', 1)):
            crop_name = f"{landing['event_id']}-source.jpg"
            actual_center = [center[0] * metadata['width'] / frames[0].shape[1],
                             center[1] * metadata['height'] / frames[0].shape[0]]
            evidence['source_crop_info'] = source_crop(source, timestamps[index], actual_center, evidence_dir / crop_name, frame_index=index)
            if evidence['source_crop_info']['status'] == 'saved':
                evidence['source_crop'] = 'evidence/' + crop_name
        artifacts['evidence'].append(evidence)
    representative = int(len(frames) / 2)
    cv2.imwrite(str(output / 'overview.jpg'), _draw_frame(frames[representative], calibration, tracks, representative))
    artifacts['overview'] = 'overview.jpg'
    timings['total_seconds'] = time.perf_counter() - start
    result = {'schema_version': 1, 'pipeline_version': __version__, 'video_id': video_id,
              'file': source.name, 'source_path': str(source), 'source_sha256': source_sha,
              'metadata': metadata, 'analysis': {'width': frames[0].shape[1], 'height': frames[0].shape[0],
                'frame_count': len(frames), 'preprocessing': preprocessing,
                'source_scaling': 'Original-resolution crops retained for review'},
              'timestamps': timestamps, 'calibration': calibration, 'tracks': tracks,
              'detection_quality': summarize_detection_quality(detections, tracks),
              'landings': landings, 'selected_landing': selected, 'decision': decision,
              'timings': timings, 'artifacts': artifacts, 'config': config,
              'detector': {'method': detector_config.get('backend', 'classical_motion_candidates'),
                           'trained_model_provided': bool(detector_config.get('model_path') or detector_config.get('scorer_path')),
                           'calibrated_accuracy': None},
              'validation_status': 'unvalidated_on_independent_human_ground_truth',
              'warnings': sorted(set(metadata.get('warnings', []) + presentation.get('warnings', []))),
              'scope': 'Offline research pipeline. Candidate confidence is not measured judgment accuracy.'}
    write_json(output / 'detections.json', detections)
    write_json(output / 'result.json', result)
    del frames
    return result


def analyze_dataset(video_dir, run_dir, config=None, limit=None, resume=True):
    from .dataset import build_manifest, draft_annotations
    from .sync import synchronize
    from .evaluation import evaluate_run
    config = dict(config or {})
    run = Path(run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(video_dir)
    write_json(run / 'manifest.json', manifest)
    entries = manifest['videos'][:limit] if limit else manifest['videos']
    index = {'run_id': run.name, 'schema_version': 1, 'videos': [], 'pairs': [],
             'pipeline_version': __version__, 'status': 'running',
             'scope': 'Research analysis and human review; no independent accuracy claim.'}
    results = []
    implementation = implementation_fingerprint()
    config_digest = hashlib.sha256((json.dumps(config, sort_keys=True) + implementation).encode()).hexdigest()
    write_json(run / 'index.json', index)

    def process_entry(entry):
        folder = run / 'videos' / entry['id']
        row = {'id': entry['id'], 'file': entry['file'], 'source_path': entry['path'],
               'sha256': entry['sha256'], 'group_id': entry.get('group_id'), 'split': entry.get('split'),
               'result_path': f"videos/{entry['id']}/result.json",
               'preview_path': f"videos/{entry['id']}/preview.mp4"}
        try:
            existing = folder / 'result.json'
            result = json.loads(existing.read_text()) if resume and existing.exists() else None
            if not result or result.get('source_sha256') != entry['sha256'] or result.get('run_config_digest') != config_digest:
                result = analyze_video(entry['path'], folder, {**config, 'video_id': entry['id']})
                result['run_config_digest'] = config_digest
                result['implementation_fingerprint'] = implementation
                write_json(existing, result)
            row.update(status='completed', decision=result['decision']['decision'],
                       calibration_status=result['calibration']['status'], track_count=len(result['tracks']),
                       landing_count=len(result['landings']))
        except Exception as error:
            result = None
            row.update(status='failed', decision='REVIEW', error=f'{type(error).__name__}: {error}')
            if (folder / 'result.json').exists():
                (folder / 'result.json').replace(folder / 'previous-result.json')
            write_json(folder / 'error.json', row)
        return row, result

    workers = max(1, min(2, int(config.get('workers', 2))))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(process_entry, entry) for entry in entries]
        for number, future in enumerate(as_completed(pending), 1):
            row, result = future.result()
            if result is not None:
                results.append(result)
                draft_annotations([result], run / 'annotation-drafts')
            index['videos'].append(row)
            index['videos'].sort(key=lambda item: item['file'])
            write_json(run / 'index.json', index)
            print(json.dumps({'event': 'video_completed', 'completed': number, 'total': len(entries), **row}, ensure_ascii=False), flush=True)
    by_pair = {}
    for entry in entries:
        if entry.get('pair_key') and entry.get('view'):
            by_pair.setdefault(entry['pair_key'], {})[entry['view'].lower()] = entry
    for pair_key, pair in sorted(by_pair.items()):
        if set(pair) != {'a', 'b'}:
            continue
        a, b = pair['a'], pair['b']
        try:
            sync = synchronize(a['path'], b['path'], {'cache_dir': str(run / 'audio-cache')})
        except Exception as error:
            sync = {'status': 'failed', 'offset_seconds': None, 'reason_codes': ['SYNC_ERROR'], 'error': str(error)}
        index['pairs'].append({'id': f"pair_{len(index['pairs']):03d}", 'pair_key': pair_key, 'a_id': a['id'], 'b_id': b['id'], 'sync': sync,
                               'decision': 'REVIEW', 'reason_codes': ['HUMAN_EVENT_CORRESPONDENCE_REQUIRED']})
        write_json(run / 'index.json', index)
    draft_annotations(results, run / 'annotation-drafts')
    index['status'] = 'completed' if all(r['status'] == 'completed' for r in index['videos']) else 'completed_with_failures'
    write_json(run / 'index.json', index)
    evaluation = evaluate_run(run)
    write_json(run / 'evaluation.json', evaluation)
    return index
