"""Source facts and frame budgets, using real media rather than browser guesses."""
import json
import subprocess

import pytest

from badminton.media import load_frames, probe


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'source.MP4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=320x180:rate=30000/1001', '-frames:v', '12',
                    '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


def test_probe_reports_original_container_and_video_facts(video):
    metadata = probe(video)
    assert (metadata['width'], metadata['height']) == (320, 180)
    assert metadata['average_fps'] == pytest.approx(30000 / 1001)
    assert metadata['extension'] == '.mp4'
    assert metadata['codec'] == 'h264'
    assert 'mp4' in metadata['container']
    assert metadata['size_bytes'] == video.stat().st_size
    assert metadata['bitrate_bps'] > 0
    assert metadata['container_bitrate_bps'] >= metadata['bitrate_bps']
    assert metadata['variable_frame_rate'] is False
    assert metadata['frame_count'] == 12
    assert metadata['duration'] == pytest.approx(12 * 1001 / 30000, abs=.001)


def test_missing_or_nonfinite_optional_metadata_remains_unknown(monkeypatch, tmp_path):
    path = tmp_path / 'sample.mkv'
    path.write_bytes(b'stub')
    raw = {'streams': [{'width': 180, 'height': 320, 'avg_frame_rate': '0/0',
                        'bit_rate': 'N/A', 'duration': '.1'}],
           'format': {'bit_rate': 'NaN'}, 'packets': [{'pts_time': '0'}]}
    monkeypatch.setattr('badminton.media.subprocess.check_output', lambda *a, **kw: json.dumps(raw))
    metadata = probe(path)
    assert metadata['average_fps'] is None
    assert metadata['bitrate_bps'] is None
    assert metadata['container_bitrate_bps'] is None
    assert metadata['container'] is None
    assert metadata['variable_frame_rate'] is None
    assert metadata['extension'] == '.mkv'
    json.dumps(metadata, allow_nan=False)


def test_frame_budget_fails_instead_of_returning_truncated_clip(video):
    exact_budget = 12 * 320 * 180 * 3
    assert len(load_frames(video, expected_count=12, max_memory_bytes=exact_budget)) == 12
    with pytest.raises(ValueError, match='프레임 메모리 한도'):
        load_frames(video, max_memory_bytes=exact_budget - 1)
    # The failed decoder is released; a new bounded read still succeeds.
    assert len(load_frames(video, expected_count=12, resize_to=(160, 90),
                           max_memory_bytes=exact_budget // 4)) == 12


@pytest.mark.parametrize('budget', [0, -1, True, 1.5, '1000'])
def test_invalid_frame_budgets_are_rejected_before_open(video, budget):
    with pytest.raises(ValueError, match='1바이트'):
        load_frames(video, max_memory_bytes=budget)
