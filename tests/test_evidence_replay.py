import json

import numpy as np

from badminton.evidence import write_annotated_clip
from badminton.media import probe, load_frames


def test_annotated_replay_keeps_variable_source_times_and_all_frames(tmp_path):
    frames = [np.full((160, 240, 3), value, np.uint8) for value in [0, 40, 80, 120, 160, 200]]
    times = [0., .041708, .083416, .208541, .250249, .291957]
    result = write_annotated_clip(frames, times, {}, [], None, {'decision': 'REVIEW'}, tmp_path)
    metadata = probe(tmp_path / result['path'])
    assert metadata['frame_count'] == len(frames)
    assert len(load_frames(tmp_path / result['path'])) == len(frames)
    assert max(abs(a-b) for a, b in zip(times, metadata['timestamps'])) < .0001
    assert metadata['duration'] > metadata['timestamps'][-1]
    mapping = json.loads((tmp_path / result['timing_path']).read_text())
    assert [row['source_frame_index'] for row in mapping['frames']] == list(range(len(frames)))
    assert result['timestamp_alignment_max_error_seconds'] < .0001


def test_replay_window_records_nonzero_source_time_origin(tmp_path):
    frames = [np.zeros((160, 240, 3), np.uint8) for _ in range(15)]
    times = [i / 10 for i in range(15)]
    landing = {'time': 1., 'frame_index': 10, 'candidate_px': [100, 100], 'track_id': 't'}
    result = write_annotated_clip(frames, times, {}, [], landing, {'decision': 'REVIEW'}, tmp_path,
                                  {'seconds_before': .2, 'seconds_after': .2})
    assert result['source_start_frame'] == 8
    assert result['source_end_frame'] == 12
    assert result['source_time_origin'] == .8
