"""Exercise the actual codec/filesystem boundary used by Windows evidence."""
import errno
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from badminton.evidence import write_annotated_clip, write_calibration_overlay, write_trajectory_context
from badminton.image_io import read_image, write_image
from badminton.media import probe
from badminton.video_estimate import write_estimate_overlay


def test_unicode_image_roundtrip_does_not_use_opencv_filename_io(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Filename-based OpenCV I/O reintroduces the Windows Unicode bug')
    monkeypatch.setattr(cv2, 'imread', forbidden)
    monkeypatch.setattr(cv2, 'imwrite', forbidden)
    folder = tmp_path / '한글 사용자' / '배드민턴 결과 A'
    folder.mkdir(parents=True)
    pixels = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    path = folder / '코트 사진.png'
    write_image(path, pixels)
    assert np.array_equal(read_image(path), pixels)
    assert read_image(path, cv2.IMREAD_GRAYSCALE).shape == pixels.shape[:2]
    assert list(folder.iterdir()) == [path]


def test_image_encoding_failure_preserves_existing_file(tmp_path):
    destination = tmp_path / 'existing.not-an-image-format'
    destination.write_bytes(b'previous evidence')
    with pytest.raises(OSError, match='Could not encode image') as captured:
        write_image(destination, np.zeros((2, 2, 3), np.uint8))
    assert captured.value.filename == str(destination)
    assert destination.read_bytes() == b'previous evidence'
    assert list(tmp_path.iterdir()) == [destination]


def test_image_write_failure_removes_partial_file_and_preserves_errno(tmp_path, monkeypatch):
    destination = tmp_path / '근거 사진.jpg'
    destination.write_bytes(b'previous evidence')
    real_write = Path.write_bytes
    def disk_full(path, content):
        real_write(path, content[:5])
        raise OSError(errno.ENOSPC, 'No space left on device', str(path))
    monkeypatch.setattr(Path, 'write_bytes', disk_full)
    with pytest.raises(OSError) as captured:
        write_image(destination, np.zeros((8, 8, 3), np.uint8))
    assert captured.value.errno == errno.ENOSPC
    assert captured.value.filename == str(destination)
    assert destination.read_bytes() == b'previous evidence'
    assert list(tmp_path.iterdir()) == [destination]


def test_image_read_distinguishes_unreadable_file_from_invalid_bytes(tmp_path):
    path = tmp_path / '손상.jpg'
    with pytest.raises(FileNotFoundError):
        read_image(path)
    for content in (b'', b'not an image'):
        path.write_bytes(content)
        assert read_image(path) is None


def test_source_contact_image_error_keeps_observed_source_window(tmp_path, monkeypatch):
    from badminton import events
    frames = [np.zeros((120, 160, 3), np.uint8) for _ in range(8)]
    candidate = {'frame_index': 3, 'time': 3 / 60, 'candidate_px': [20, 15],
                 'contact_observed': False, 'contact_px': None, 'reason_codes': []}
    times = [i / 60 for i in range(8)]
    baseline = events.refine_source_contacts([candidate], frames, times, (40, 30),
                                             {'source_contact_patch_size': 24})
    def failed_image(*args, **kwargs):
        raise PermissionError('Evidence folder is read-only')
    monkeypatch.setattr(events, 'write_image', failed_image)
    result = events.refine_source_contacts([candidate], frames, times, (40, 30),
                                           {'source_contact_patch_size': 24}, tmp_path)
    evidence = result[0]['source_contact_evidence']
    original = baseline[0]['source_contact_evidence']
    assert evidence['status'] == original['status'] == 'complete'
    for key in ('frames', 'frame_indices', 'times', 'scale_xy', 'candidate_source_px'):
        assert evidence[key] == original[key]
    assert result[0]['contact_observed'] is False and 'image' not in evidence
    assert evidence['image_error'] == 'cannot_write_source_temporal_strip'
    assert evidence['artifact_warnings'] == [{'artifact': 'source_contact',
        'message': 'Evidence folder is read-only', 'error_type': 'PermissionError'}]


def test_all_evidence_writers_accept_unicode_output_and_temp_paths(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Evidence used filename-based OpenCV image I/O')
    monkeypatch.setattr(cv2, 'imread', forbidden)
    monkeypatch.setattr(cv2, 'imwrite', forbidden)
    temporary = tmp_path / '한글 임시 폴더'
    temporary.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(temporary))
    output = tmp_path / '배드민턴 결과 A'
    frames = [np.full((160, 240, 3), value, np.uint8) for value in (0, 40, 80, 120, 160, 200)]
    times = [0., .041708, .083416, .208541, .250249, .291957]
    landing = {'time': times[2], 'frame_index': 2, 'candidate_px': [120, 100],
               'image_motion_event': {'trajectory': [
                   {'frame_index': i, 'x': 120, 'y': 80 + i} for i in range(len(frames))]}}
    calibration = write_calibration_overlay(frames, {}, output)
    context = write_trajectory_context(frames, times, (240, 160), landing, output)
    estimate = {'eligible': True, 'decision': 'IN', 'frame_interval': [1, 3],
                'possible_contact_polygon_px': [[110, 90], [130, 90], [130, 110], [110, 110]],
                'native_evidence': {'artifact': context['path']}}
    overlay = write_estimate_overlay(frames, {}, estimate, output)
    assert read_image(output / calibration).shape == frames[0].shape
    assert read_image(output / context['path']).shape[1] == 960
    # A decoded native panel must be included below the scene, not silently lost.
    assert read_image(output / overlay).shape[0] > 640
    progress = []
    replay = write_annotated_clip(frames, times, {}, [], landing, {'decision': 'REVIEW'},
                                  output, progress_callback=progress.append)
    encoded = probe(output / replay['path'])
    assert encoded['frame_count'] == len(frames)
    assert max(abs(a - b) for a, b in zip(encoded['timestamps'], times)) < .0001
    mapping = json.loads((output / replay['timing_path']).read_text())
    assert [row['source_frame_index'] for row in mapping['frames']] == list(range(len(frames)))
    assert progress[-1]['detail'] == '근거 영상 프레임·시각 검증'
    assert not list(temporary.iterdir())
