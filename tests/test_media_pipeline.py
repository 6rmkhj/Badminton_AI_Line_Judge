import json
from pathlib import Path
import subprocess
import cv2
import pytest
from badminton.media import probe, prepare_preview, load_frames, source_crop, SourceFrameSequence, repair_preview_tail
from badminton.pipeline import analyze_video, aggregate_decision


def clip(tmp_path, gap=False):
    path = tmp_path / 'input.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
               '-i', 'testsrc2=size=240x160:rate=30', '-frames:v', '12']
    if gap:
        command += ['-vf', 'setpts=if(gte(N\\,6)\\,PTS+0.1/TB\\,PTS)']
    command += ['-c:v', 'libx264', '-threads', '1', '-fps_mode', 'passthrough',
                '-enc_time_base', '1/90000', '-pix_fmt', 'yuv420p', str(path)]
    subprocess.run(command, check=True)
    return path


def test_variable_timestamps_survive_preview(tmp_path):
    source = clip(tmp_path, gap=True)
    original, preview, _ = prepare_preview(source, tmp_path / 'preview.mp4', 160)
    assert original['frame_count'] == preview['frame_count'] == 12
    assert 'PRESENTATION_GAP' in original['warnings']
    assert 'PRESENTATION_GAP' in preview['warnings']
    assert max(abs(a-b) for a,b in zip(original['timestamps'], preview['timestamps'])) < .0001
    assert len(load_frames(tmp_path / 'preview.mp4', 12)) == 12
    assert preview['duration'] > preview['timestamps'][-1] + .01


def test_rotation_preserves_display_coordinate_mapping(tmp_path):
    source = clip(tmp_path)
    rotated = tmp_path / 'rotated.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-display_rotation', '90', '-i', str(source),
                    '-c', 'copy', str(rotated)], check=True)
    original, preview, _ = prepare_preview(rotated, tmp_path / 'preview.mp4', 160)
    assert (original['width'], original['height']) == (160, 240)
    assert (preview['width'], preview['height']) == (160, 240)
    evidence = source_crop(rotated, 0.1, [80, 120], tmp_path / 'crop.jpg', radius=30, frame_index=3)
    assert evidence['status'] == 'saved'
    assert abs(evidence['alignment_error_seconds']) < .001
    assert cv2.imread(str(tmp_path / 'crop.jpg')).shape[:2] == (60, 60)


def test_actual_pipeline_stays_review_without_ground_contact(tmp_path):
    source = clip(tmp_path)
    progress = []
    result = analyze_video(source, tmp_path / 'run', {'analysis_width': 160, 'source_crop_count': 0},
                           progress_callback=progress.append)
    assert result['decision']['decision'] == 'REVIEW'
    assert result['assessment']['line_verdict'] == result['decision']['decision']
    assert result['assessment']['changes_decision'] is False
    assert json.loads((tmp_path / 'run/result.json').read_text())['assessment'] == result['assessment']
    assert result['validation_status'] == 'unvalidated_on_independent_human_ground_truth'
    assert result['analysis']['frame_count'] == 12
    assert len(result['timestamps']) == 12
    assert json.loads((tmp_path / 'run/result.json').read_text())['source_sha256'] == result['source_sha256']
    assert (tmp_path / 'run/preview.mp4').exists()
    assert cv2.imread(str(tmp_path / 'run' / result['artifacts']['calibration_overlay'])) is not None
    assert all(event['contact_observed'] is False for event in result['landings'])
    assert list(dict.fromkeys(event['stage'] for event in progress)) == [
        'prepare', 'calibration', 'detection', 'tracking', 'contact', 'evidence', 'save']
    assert [event['stage_index'] for event in progress] == sorted(event['stage_index'] for event in progress)
    assert all(event['stage_count'] == 7 for event in progress)
    counts = [event for event in progress if event.get('unit') == 'frames']
    assert all(0 <= event['completed_units'] <= event['total_units'] for event in counts)
    for detail in ('미리보기 변환', '분석 프레임 읽기', '프레임별 셔틀콕 후보 탐지', '근거 영상 인코딩'):
        observed = [event for event in counts if event.get('detail') == detail]
        assert observed[0]['completed_units'] == 0
        assert observed[-1]['completed_units'] == observed[-1]['total_units'] == 12
        assert [event['completed_units'] for event in observed] == sorted(event['completed_units'] for event in observed)
    assert 'progress_callback' not in result['config']


def test_frame_progress_throttles_intermediate_counts_and_retains_final(monkeypatch):
    from badminton.media import _FrameProgress
    clock = [0.]
    monkeypatch.setattr('badminton.media.time.monotonic', lambda: clock[0])
    events = []
    progress = _FrameProgress(events.append, 12, '프레임 검사')
    progress.report(0, force=True)
    for count in range(1, 12):
        clock[0] = count * .1
        progress.report(count)
    progress.report(12, force=True)
    assert [event['completed_units'] for event in events] == [0, 5, 10, 12]
    assert all(event['total_units'] == 12 for event in events)


def test_multiple_unselected_landings_cannot_confirm_judgment():
    assert aggregate_decision([{'decision':'IN'}, {'decision':'IN'}])['decision'] == 'REVIEW'
    assert aggregate_decision([])['decision'] == 'REVIEW'


def test_original_resolution_sequence_keeps_frame_order_and_one_cache(tmp_path):
    source = clip(tmp_path)
    frames = load_frames(source)
    sequence = SourceFrameSequence(source, len(frames))
    try:
        for index in [0, 5, 9, 2, 11]:
            assert (sequence[index] == frames[index]).all()
            assert sequence.frame.shape[:2] == (160, 240)
    finally:
        sequence.close()
    assert sequence.frame is None


def test_legacy_preview_tail_repair_preserves_frames_and_makes_last_frame_seekable(tmp_path):
    """Reproduce the old fine-time-base encode's missing final display interval.

    The legacy preview flags produce a duration ending exactly at the last PTS
    with the supported FFmpeg version. This fixture does not fabricate packet
    flags or assume corruption in the source recording.
    """
    source = clip(tmp_path)
    preview = tmp_path / 'legacy-preview.mp4'
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(source),
        '-vf', 'scale=160:106:flags=area,setpts=PTS-STARTPTS',
        '-c:v', 'libx264', '-threads', '1', '-crf', '21',
        '-fps_mode', 'passthrough', '-enc_time_base', 'demux', str(preview),
    ], check=True)
    before = probe(preview)
    assert before['stored_frame_count'] == before['frame_count'] == 12
    assert abs(before['duration'] - before['timestamps'][-1]) < .0001
    pixels_before = load_frames(preview, 12)

    def encoded_packets():
        data = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
            '-show_data_hash', 'sha256', '-show_entries',
            'packet=pts_time,dts_time,data_hash', '-of', 'json', str(preview),
        ]))
        return data['packets']

    packets_before = encoded_packets()
    assert repair_preview_tail(preview, fps=30) is True
    after = probe(preview)
    assert after['frame_count'] == after['stored_frame_count'] == 12
    assert len(after['timestamps']) == len(before['timestamps'])
    assert max(abs(a-b) for a, b in zip(after['timestamps'], before['timestamps'])) < .0001
    assert after['duration'] - after['timestamps'][-1] >= .5 / 30
    # Remuxing must retain encoded image packets, as well as decoded pixels;
    # frame count alone would miss leading-frame substitutions or reencoding.
    assert encoded_packets() == packets_before
    pixels_after = load_frames(preview, 12)
    assert all((a == b).all() for a, b in zip(pixels_before, pixels_after))
    capture = cv2.VideoCapture(str(preview), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        assert capture.set(cv2.CAP_PROP_POS_MSEC, after['timestamps'][-1] * 1000)
        ok, last_frame = capture.read()
        assert ok and (last_frame == pixels_before[-1]).all()
    finally:
        capture.release()
    assert repair_preview_tail(preview, fps=30) is False


def test_tail_repair_rejects_leading_editlist_without_changing_video(tmp_path):
    """A real leading-discard edit list must not be mistaken for a lost tail."""
    source = clip(tmp_path)
    edited = tmp_path / 'leading-edit.mp4'
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(source),
        '-c:v', 'libx264', '-threads', '1', '-fps_mode', 'passthrough',
        '-enc_time_base', 'demux', '-output_ts_offset', '-0.01', str(edited),
    ], check=True)
    before = probe(edited)
    assert before['stored_frame_count'] == 12 and before['frame_count'] == 11
    packets = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=pts_time,flags', '-of', 'json', str(edited),
    ]))['packets']
    displayed = [float(packet['pts_time']) for packet in packets if 'D' not in packet['flags']]
    assert any('D' in packet['flags'] and float(packet['pts_time']) < min(displayed) for packet in packets)
    original_bytes = edited.read_bytes()
    with pytest.raises(ValueError, match='non-tail'):
        repair_preview_tail(edited, fps=30)
    assert edited.read_bytes() == original_bytes
