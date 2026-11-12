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
from .media import prepare_preview, load_frames, source_crop, SourceFrameSequence, file_hash


def implementation_fingerprint():
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ('media.py', 'pipeline.py', 'calibration.py', 'detection.py', 'tracking.py',
                 'events.py', 'identity.py', 'identity_continuity.py', 'candidate_identity.py',
                 'evidence.py', 'decision.py', 'learning.py',
                 'contact_model.py', 'sync.py'):
        digest.update(name.encode())
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def model_fingerprints(config):
    """Changing a checkpoint at the same path must invalidate cached analysis."""
    signatures = {}
    for section in ('detector', 'events', 'calibration'):
        for name, value in (config.get(section) or {}).items():
            if name.endswith('_path') and ('model' in name or 'scorer' in name):
                path = Path(value).expanduser().resolve()
                signatures[f'{section}.{name}'] = {'path': str(path),
                    'sha256': file_hash(path) if path.is_file() else None}
    return signatures


def select_requested_landings(landings, identity_config):
    """Anchor matching selects identity, never physical contact or a best score."""
    if identity_config.get('hit_anchors'):
        selected = [item for item in landings if item.get('request_match') == 'selected']
        # A conflicting candidate keeps the request unresolved.
        if any(item.get('request_match') == 'ambiguous' for item in landings):
            return landings
        return selected
    return landings


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
    from .evidence import draw_observed_trail
    canvas = frame.copy()
    polygon = calibration.get('polygon_px')
    if polygon and calibration.get('status') == 'ok':
        cv2.polylines(canvas, [np.asarray(polygon, dtype=np.int32)], True, (0, 210, 240), 2)
    for track in tracks:
        if landing and str(track['track_id']) != str(landing.get('track_id')):
            continue
        points = [o for o in track.get('observations', []) if 0 <= frame_index - o['frame_index'] <= 12]
        if not points:
            continue
        pts = np.array([[o['x'], o['y']] for o in points], dtype=np.int32)
        color = ((int(track['track_id']) * 53) % 200 + 55, 220, 100) if str(track['track_id']).isdigit() else (200, 220, 100)
        draw_observed_trail(canvas, [(o['frame_index'], point) for o, point in zip(points, pts)], color)
        x, y = pts[-1]
        cv2.circle(canvas, (int(x), int(y)), 5, color, 1)
        cv2.putText(canvas, str(track['track_id']), (int(x)+6, int(y)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    if landing and landing.get('candidate_px'):
        x, y = map(int, landing['candidate_px'])
        cv2.circle(canvas, (x, y), 20, (20, 80, 255), 2)
    cv2.putText(canvas, 'SHUTTLE PATH ESTIMATE - CONTACT UNCONFIRMED' if landing else
                'RESEARCH CANDIDATES - NOT VERIFIED CONTACT', (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 200, 255), 2)
    return canvas


def analyze_video(video_path, output_dir, config=None, *, progress_callback=None):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if progress_callback is not None:
        progress_callback({'stage': 'prepare', 'stage_label': '영상 준비',
                           'stage_index': 1, 'stage_count': 7, 'detail': '영상 분석 준비·실행 잠금 확인'})
    with FileLock(str(output / '.analysis.lock'), timeout=1800):
        if progress_callback is None:
            return _analyze_video(video_path, output, config)
        return _analyze_video(video_path, output, config, progress_callback=progress_callback)


def _analyze_video(video_path, output_dir, config=None, *, progress_callback=None):
    from .calibration import calibrate, assess_camera_stability, guard_calibration_stability
    from .detection import detect_sequence, summarize_detection_quality
    from .tracking import track_detections
    from .events import (estimate_landings, refine_source_contacts, validate_contact_modes,
                         prioritize_review_candidates, review_refinement_regions, review_region_indices)
    from .identity import apply_identity
    from .decision import decide
    config = dict(config or {})
    source, output = Path(video_path).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    video_id = config.get('video_id') or 'v_' + hashlib.sha256(source.name.encode()).hexdigest()[:12]
    start = time.perf_counter()
    timings = {}
    stages = {'prepare': (1, '영상 준비'), 'calibration': (2, '코트 확인'),
              'detection': (3, '셔틀콕 후보 탐지'), 'tracking': (4, '궤적 연결'),
              'contact': (5, '착지·라인 확인'), 'evidence': (6, '근거 영상 생성'), 'save': (7, '결과 저장')}

    def report(stage, **fields):
        if progress_callback is not None:
            index, label = stages[stage]
            progress_callback({'stage': stage, 'stage_label': label, 'stage_index': index,
                               'stage_count': len(stages), **fields})

    def observe(stage):
        return {'progress_callback': lambda fields: report(stage, **fields)} if progress_callback is not None else {}

    report('prepare', detail='원본 확인·미리보기 준비')
    metadata, presentation, source_sha = prepare_preview(source, output / 'preview.mp4', config.get('analysis_width', 1280),
                                                       **observe('prepare'))
    detector_config = dict(config.get('detector') or {})
    event_config = dict(config.get('events') or {})
    validate_contact_modes(event_config)
    learned_input = bool(detector_config.get('model_path') or detector_config.get('scorer_path')
                         or event_config.get('contact_model_path'))
    if learned_input or detector_config.get('refinement_regions') or detector_config.get('source_analysis_enabled', False):
        frames = load_frames(source, presentation['frame_count'], config.get('max_frames', 3000),
                             resize_to=(presentation['width'], presentation['height']), **observe('prepare'))
        preprocessing = 'opencv_source_bgr_area_v1'
    else:
        frames = load_frames(output / 'preview.mp4', presentation['frame_count'], config.get('max_frames', 3000),
                             **observe('prepare'))
        preprocessing = 'h264_preview_crf21_bgr_v1'
    timestamps = presentation.pop('timestamps')
    metadata.pop('timestamps', None)
    timings['decode_and_preview_seconds'] = time.perf_counter() - start
    clock = time.perf_counter()
    report('calibration', detail='코트 선과 카메라 안정성 확인')
    calibration_frames = [frames[int(i)] for i in np.linspace(0, len(frames)-1, min(9, len(frames)))]
    calibration = calibrate(calibration_frames, config.get('court_hint'), config.get('calibration'))
    stability = assess_camera_stability(frames, config.get('camera_stability'))
    calibration = guard_calibration_stability(calibration, stability)
    timings['calibration_seconds'] = time.perf_counter() - clock
    clock = time.perf_counter()
    report('detection', detail='셔틀콕 후보 탐지 준비')
    detector_config['source_path'] = str(source)
    detector_config['frame_preprocessing'] = preprocessing
    source_frames = None
    if detector_config.get('backend') == 'onnx' or detector_config.get('highres_enabled', True):
        source_frames = SourceFrameSequence(source, len(frames))
        detector_config['source_frames'] = source_frames
    try:
        detections = detect_sequence(frames, timestamps, detector_config, **observe('detection'))
    finally:
        if source_frames is not None:
            source_frames.close()
    timings['detection_seconds'] = time.perf_counter() - clock
    clock = time.perf_counter()
    report('tracking', detail='프레임 사이의 후보 궤적 연결')
    tracks = track_detections(detections, config.get('tracker'), **observe('tracking'))
    timings['tracking_seconds'] = time.perf_counter() - clock
    clock = time.perf_counter()
    report('contact', detail='최초 접촉 후보·라인 관계 검토')
    event_config['frame_preprocessing'] = preprocessing
    event_config['review_refinement_regions'] = detector_config.get('refinement_regions',
        event_config.get('review_refinement_regions', []))
    review_regions = review_refinement_regions(event_config,
        (frames[0].shape[1], frames[0].shape[0]), len(timestamps))
    identity_config = dict(config.get('identity') or {})
    if identity_config.get('event_window_seconds') is not None:
        event_config['event_window_seconds'] = identity_config['event_window_seconds']
    landings = estimate_landings(tracks, frames, timestamps, calibration, event_config)
    landings = apply_identity(landings, tracks, timestamps, calibration, identity_config)
    if event_config.get('source_contact_enabled', True):
        with SourceFrameSequence(source, len(frames)) as originals:
            landings = refine_source_contacts(landings, originals, timestamps,
                analysis_size=(frames[0].shape[1], frames[0].shape[0]),
                config=event_config, output_dir=output)
    for i, landing in enumerate(landings):
        landing['event_id'] = f'event_{i:03d}'
        landing['decision'] = decide(landing, calibration, config.get('rules'))
        if landing.get('assistance_used'):
            landing['decision'].update(assistance_used=True, evaluation_eligible=False)
    requested = select_requested_landings(landings, identity_config)
    decision = aggregate_decision([landing['decision'] for landing in requested])
    assistance_used = bool(identity_config.get('hit_anchors') or event_config.get('target_track_id') is not None
                           or config.get('request_context') or review_regions)
    if assistance_used:
        decision.update(assistance_used=True, evaluation_eligible=False)
    timings['events_and_decision_seconds'] = time.perf_counter() - clock
    display_candidates = prioritize_review_candidates(requested, review_regions)
    selected = display_candidates[0] if display_candidates else None
    report('evidence', detail='원본 확대 근거·검수 영상 생성')
    artifacts = {'preview': 'preview.mp4', 'detections': 'detections.json', 'evidence': []}
    evidence_dir = output / 'evidence'
    evidence_dir.mkdir(exist_ok=True)
    trajectory_context = None
    if selected and selected.get('image_motion_event'):
        from .evidence import write_trajectory_context
        with SourceFrameSequence(source, len(frames)) as originals:
            trajectory_context = write_trajectory_context(originals, timestamps,
                (frames[0].shape[1], frames[0].shape[0]), selected, output)
    # Preserve several candidate contexts, including original-resolution crops.
    for landing in prioritize_review_candidates(landings, review_regions)[:6]:
        index = max(0, min(len(frames)-1, int(landing.get('frame_index', 0))))
        filename = f"{landing['event_id']}.jpg"
        cv2.imwrite(str(evidence_dir / filename), _draw_frame(frames[index], calibration, tracks, index, landing))
        evidence = {'event_id': landing['event_id'], 'frame_index': index, 'time': timestamps[index],
                    'image': 'evidence/' + filename}
        if landing is selected and trajectory_context:
            evidence['trajectory_context'] = trajectory_context['path']
            evidence['trajectory_context_info'] = trajectory_context
        native = landing.get('source_contact_evidence', {})
        if native.get('image'):
            evidence['source_sequence'] = native['image']
            evidence['source_sequence_info'] = native
        center = landing.get('candidate_px')
        if center and len(artifacts['evidence']) < int(config.get('source_crop_count', 1)):
            crop_name = f"{landing['event_id']}-source.jpg"
            actual_center = [center[0] * metadata['width'] / frames[0].shape[1],
                             center[1] * metadata['height'] / frames[0].shape[0]]
            evidence['source_crop_info'] = source_crop(source, timestamps[index], actual_center, evidence_dir / crop_name, frame_index=index)
            if evidence['source_crop_info']['status'] == 'saved':
                evidence['source_crop'] = 'evidence/' + crop_name
        artifacts['evidence'].append(evidence)
    review_refinement = {
        'status': ('matched_candidates' if any(review_region_indices(item, review_regions) for item in landings)
                   else 'no_matching_candidate') if review_regions else 'not_requested',
        'regions': review_regions,
        'matching_event_ids': [item['event_id'] for item in landings if review_region_indices(item, review_regions)],
        'selected_event_id': selected['event_id'] if selected else None,
        'selected_in_requested_region': bool(selected and review_region_indices(selected, review_regions)),
        'scope': 'Evidence prioritization only; all landing decisions remain in the request aggregate.',
        'assistance_used': bool(review_regions),
    }
    representative = int(selected['frame_index']) if selected else int(len(frames) / 2)
    cv2.imwrite(str(output / 'overview.jpg'), _draw_frame(frames[representative], calibration, tracks, representative, selected))
    artifacts['overview'] = 'overview.jpg'
    if config.get('evidence', {}).get('annotated_clip', True):
        from .evidence import write_annotated_clip
        artifacts['annotated_clip'] = write_annotated_clip(frames, timestamps, calibration, tracks,
            selected, decision, output, config.get('evidence'), **observe('evidence'))
    report('save', detail='분석 결과·탐지 근거 저장')
    timings['total_seconds'] = time.perf_counter() - start
    result = {'schema_version': 1, 'pipeline_version': __version__, 'video_id': video_id,
              'implementation_fingerprint': implementation_fingerprint(),
              'file': source.name, 'source_path': str(source), 'source_sha256': source_sha,
              'metadata': metadata, 'analysis': {'width': frames[0].shape[1], 'height': frames[0].shape[0],
                'frame_count': len(frames), 'preprocessing': preprocessing,
                'source_scaling': 'Global source-frame scan with native ROI refinement and temporal contact evidence'},
              'timestamps': timestamps, 'calibration': calibration, 'camera_stability': stability, 'tracks': tracks,
              'detection_quality': summarize_detection_quality(detections, tracks),
              'landings': landings, 'selected_landing': selected, 'decision': decision,
              'review_refinement': review_refinement,
              'timings': timings, 'artifacts': artifacts, 'config': config,
              'assistance_used': assistance_used, 'automatic_evaluation_eligible': not assistance_used,
              'identity': {'requested_event_count': len(requested),
                           'hit_anchor_count': len(identity_config.get('hit_anchors', [])),
                           'event_window_seconds': identity_config.get('event_window_seconds')},
              'model_fingerprints': model_fingerprints(config),
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
    cv2.setNumThreads(max(1, min(4, int(config.get('opencv_threads', 2)))))
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
    config_digest = hashlib.sha256((json.dumps(config, sort_keys=True) + implementation
                                   + json.dumps(model_fingerprints(config), sort_keys=True)).encode()).hexdigest()
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
    from .reporting import write_run_report
    write_run_report(run)
    return index
