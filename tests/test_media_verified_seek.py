"""Fast source seeks must preserve displayed pixels and reject wrong PTS."""
import subprocess

import cv2
import numpy as np
import pytest

from badminton import media


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'variable.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
        '-i', 'testsrc2=size=240x160:rate=30', '-frames:v', '60',
        '-vf', 'setpts=if(gte(N\\,30)\\,PTS+0.1/TB\\,PTS)',
        '-c:v', 'libx264', '-g', '12', '-threads', '1', '-fps_mode', 'passthrough',
        '-enc_time_base', '1/90000', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


@pytest.mark.parametrize('variant', ['variable', 'rotated', 'edited', 'offset'])
def test_guarded_random_access_matches_sequential_pixels_and_pts(source, tmp_path, variant):
    path = source
    if variant == 'rotated':
        path = tmp_path / 'rotated.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-display_rotation', '90', '-i',
                        str(source), '-c', 'copy', str(path)], check=True)
    elif variant == 'edited':
        path = tmp_path / 'edited.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-c:v', 'libx264',
                        '-threads', '1', '-g', '12', '-fps_mode', 'passthrough',
                        '-enc_time_base', 'demux', '-output_ts_offset', '-0.01', str(path)], check=True)
    elif variant == 'offset':
        path = tmp_path / 'offset.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-c', 'copy',
                        '-output_ts_offset', '3.25', str(path)], check=True)
    info = media.probe(path)
    reference = media.load_frames(path, info['frame_count'])
    with media.SourceFrameSequence(path, len(reference)) as sequence:
        for index in [37, len(reference) - 1, 3, 38, 39, len(reference) - 2]:
            frame = sequence[index]
            assert np.array_equal(frame, reference[index])
            decoder_pts = sequence.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            assert abs(decoder_pts - sequence._decoder_origin - info['timestamps'][index]) < .0001
        assert sequence.seek_statistics['attempts'] > 0
    assert sequence.frame is None


def test_bad_decoder_timestamp_reopens_and_falls_back_to_exact_sequence(source, monkeypatch):
    real_capture = cv2.VideoCapture
    reference = media.load_frames(source)
    captures = []

    class BadSeekTimestamp:
        def __init__(self, *args):
            self.capture, self.sought = real_capture(*args), False
            captures.append(self)
        def set(self, prop, value):
            self.sought = True
            return self.capture.set(prop, value)
        def get(self, prop):
            result = self.capture.get(prop)
            return result + 400 if self.sought and prop == cv2.CAP_PROP_POS_MSEC else result
        def __getattr__(self, name):
            return getattr(self.capture, name)

    monkeypatch.setattr(media.cv2, 'VideoCapture', BadSeekTimestamp)
    with media.SourceFrameSequence(source, len(reference)) as sequence:
        assert np.array_equal(sequence[37], reference[37])
        assert np.array_equal(sequence[38], reference[38])
        assert sequence.seek_statistics == {'attempts': 1, 'accepted': 0, 'fallbacks': 1}
        assert len(captures) == 2


def test_sequential_or_disabled_access_does_not_probe(source, monkeypatch):
    reference = media.load_frames(source)
    def forbidden(*args):
        raise AssertionError('Sequential decoding must not require ffprobe')
    monkeypatch.setattr(media, 'probe', forbidden)
    with media.SourceFrameSequence(source, len(reference)) as sequence:
        for i in range(10):
            assert np.array_equal(sequence[i], reference[i])
        assert sequence.seek_statistics['attempts'] == 0
    with media.SourceFrameSequence(source, len(reference), verified_seek=False) as sequence:
        assert np.array_equal(sequence[50], reference[50])
        assert sequence.seek_statistics['attempts'] == 0


@pytest.mark.parametrize('unusable', ['unavailable', 'duplicate', 'mismatched_count'])
def test_unavailable_or_ambiguous_probe_keeps_exact_sequential_behavior(source, monkeypatch, unusable):
    reference = media.load_frames(source)
    def unusable_probe(*args):
        if unusable == 'unavailable':
            raise FileNotFoundError('ffprobe unavailable')
        return {'timestamps': [0.] * len(reference) if unusable == 'duplicate'
                else [i / 30 for i in range(len(reference) + 1)]}
    monkeypatch.setattr(media, 'probe', unusable_probe)
    with media.SourceFrameSequence(source, len(reference)) as sequence:
        assert np.array_equal(sequence[50], reference[50])
        assert sequence.seek_statistics['attempts'] == 0
