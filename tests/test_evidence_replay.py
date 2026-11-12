import json

import numpy as np

from badminton.evidence import write_annotated_clip, write_trajectory_context, draw_observed_trail
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


def test_source_context_keeps_incoming_path_and_bounce_in_same_crop(tmp_path):
    import cv2
    # At 8x source scale, the old 96px crop lost this 80-analysis-pixel descent.
    times = [i / 30 for i in range(9)]
    points = [(100, y) for y in [30, 50, 70, 90, 110, 100, 95, 98, 103]]
    frames = [np.zeros((1280, 1920, 3), np.uint8) for _ in times]
    for frame, (x, y) in zip(frames, points):
        cv2.circle(frame, (x * 8, y * 8), 8, (255, 255, 255), -1)
    requested = []

    class Source:
        def __getitem__(self, index):
            requested.append(index)
            return frames[index]

    landing = {'frame_index': 4, 'candidate_px': [100, 110], 'uncertainty_px': 2,
               'image_motion_event': {'trajectory': [
                   {'frame_index': i, 'time': times[i], 'x': x, 'y': y}
                   for i, (x, y) in enumerate(points)]}}
    result = write_trajectory_context(Source(), times, (240, 160), landing, tmp_path)
    assert requested == list(range(1, 9))
    left, top, right, bottom = result['source_roi']
    assert all(left < x * 8 < right and top < y * 8 < bottom for x, y in points)
    assert result['times'] == times[1:]
    assert result['contact_observed'] is False
    assert cv2.imread(str(tmp_path / result['path'])).shape == (960, 960, 3)


def test_missing_observation_is_not_drawn_as_solid_measured_motion():
    canvas = np.zeros((40, 80, 3), np.uint8)
    draw_observed_trail(canvas, [(1, (10, 20)), (2, (20, 20)), (5, (50, 20)), (6, (60, 20))],
                        (255, 255, 255))
    assert canvas[20, 15].all() and canvas[20, 55].all()
    assert not canvas[20, 35].any()


def test_context_includes_both_sides_of_high_frame_rate_occlusion(tmp_path):
    times = [i / 120 for i in range(70)]
    observed = [25, 26, 27, 28, 29, 60, 61, 62, 63]
    landing = {'frame_index': 45, 'candidate_px': [100, 110],
               'image_motion_event': {'trajectory': [
                   {'frame_index': i, 'time': times[i], 'x': 100, 'y': 90}
                   for i in observed]}}
    result = write_trajectory_context([np.zeros((160, 240, 3), np.uint8)] * 70,
                                     times, (240, 160), landing, tmp_path)
    assert result['frame_indices'] == sorted(observed + [45])
    assert result['sampling'] == 'observed_context_with_inferred_frame'
    assert result['contact_observed'] is False
