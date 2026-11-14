"""Native-pixel evidence tests, with no learned model or annotation injection."""
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest

from badminton.native_landing import estimate_native_landing


def scene(*, x=360, scale=1, mode='shuttle', second=False):
    times = [index/30 for index in range(30)]
    frames, observations, other = [], [], []
    for index, time in enumerate(times):
        offset = index-14
        y = 100+10*index if index <= 14 else 240-10*offset+offset*offset
        frame = np.full((540*scale, 960*scale, 3), 150, np.uint8)
        # Fixed court markings disappear from the temporal foreground.
        cv2.line(frame, (100*scale, 280*scale), (800*scale, 280*scale), (235, 235, 235), 3*scale)
        centers = [x, x+25] if second else [x]
        for center in centers:
            if mode == 'shuttle':
                cv2.ellipse(frame, (center*scale, (y-5)*scale), (5*scale, 8*scale), 0, 0, 360, (225, 225, 225), -1)
                cv2.circle(frame, (center*scale, (y+4)*scale), 3*scale, (65, 65, 65), -1)
            elif mode == 'shadow':
                cv2.ellipse(frame, (center*scale, y*scale), (6*scale, 10*scale), 0, 0, 360, (65, 65, 65), -1)
            elif mode == 'person':
                cv2.rectangle(frame, ((center-20)*scale, (y-45)*scale), ((center+20)*scale, (y+45)*scale), (230, 230, 230), -1)
            elif mode == 'static':
                cv2.circle(frame, (center*scale, 240*scale), 8*scale, (65, 65, 65), -1)
        frames.append(frame)
        observations.append({'frame_index': index, 'time': time, 'x': float(x), 'y': float(y), 'observed': True})
        other.append({**observations[-1], 'x': float(x+25)})
    family = {'family_id': 'synthetic', 'track_ids': [1], 'members': [{'frame_index': 14, 'track_id': 1}]}
    tracks = [{'track_id': 1, 'observations': observations}]
    if second:
        tracks.append({'track_id': 2, 'observations': other})
        family['track_ids'].append(2)
        family['members'].append({'frame_index': 14, 'track_id': 2})
    calibration = {'camera_stability': {'status': 'ok', 'all_frames_sampled': True, 'max_displacement_px': 0.}}
    return family, tracks, frames, times, [960, 540], calibration


def estimate(args, **kwargs):
    return estimate_native_landing(*args, **kwargs)


def test_actual_native_components_support_conditional_envelope_without_certifying_contact(tmp_path):
    args = scene()
    original = deepcopy((args[0], args[1]))
    result = estimate(args, output_dir=tmp_path)
    assert result['status'] == 'supported'
    assert result['contact_observed'] is False
    assert result['identity_confirmed'] is False
    assert result['physical_ground_contact_verified'] is False
    assert result['evaluation_eligible'] is False
    assert result['confidence'] is None
    assert (args[0], args[1]) == original
    assert result['observed_frame_count'] == 10
    assert result['decoded_frame_count'] == 19
    assert result['frame_interval'] == [13, 15]
    polygon = np.array(result['possible_contact_polygon_px'])
    assert polygon[:, 0].min() <= 355 and polygon[:, 0].max() >= 365
    assert polygon[:, 1].min() <= 227 and polygon[:, 1].max() >= 247
    assert result['temporal_model']['between_frame_overshoot_px'] > 0
    assert result['uncertainty_px'] >= 1
    row = result['support_frames'][0]
    x0, y0, x1, y1 = result['source_roi']
    crop = args[2][row['frame_index']][y0:y1, x0:x1]
    assert row['source_crop_sha256'] == hashlib.sha256(crop.tobytes()).hexdigest()
    assert cv2.imread(str(tmp_path/result['artifact'])) is not None
    json.dumps(result, allow_nan=False)


def test_native_image_write_failure_retains_measured_envelope(tmp_path, monkeypatch):
    from badminton import native_landing
    args = scene()
    baseline = estimate(args)
    def failed_image(*args, **kwargs):
        raise PermissionError('Evidence folder is read-only')
    monkeypatch.setattr(native_landing, 'write_image', failed_image)
    result = estimate(args, output_dir=tmp_path)
    assert result['status'] == baseline['status'] == 'supported'
    for name in ('possible_contact_polygon_px', 'uncertainty_px', 'frame_interval', 'time_interval',
                 'support_frames', 'contact_observed', 'identity_confirmed'):
        assert result[name] == baseline[name]
    assert 'artifact' not in result
    assert result['artifact_warnings'] == [{'artifact': 'native_landing',
        'message': 'Evidence folder is read-only', 'error_type': 'PermissionError'}]


@pytest.mark.parametrize('mode', ['blank', 'static', 'shadow', 'person'])
def test_motion_track_does_not_replace_missing_shuttle_appearance(mode):
    result = estimate(scene(mode=mode))
    assert result['status'] == 'review'
    assert result['possible_contact_polygon_px'] is None
    assert result['contact_observed'] is False


def test_two_native_objects_cannot_be_resolved_by_picking_larger_component():
    result = estimate(scene(second=True))
    assert result['status'] == 'review'
    assert result['contact_observed'] is False


@pytest.mark.parametrize('x', [220, 710])
def test_no_source_position_is_hardcoded(x):
    result = estimate(scene(x=x))
    assert result['status'] == 'supported'
    polygon = np.array(result['possible_contact_polygon_px'])
    assert x-6 <= polygon[:, 0].min() <= x
    assert x <= polygon[:, 0].max() <= x+7


def test_source_resolution_is_used_and_reported():
    result = estimate(scene(scale=2))
    assert result['status'] == 'supported'
    assert result['source_scale'] == 2
    assert result['source_size'] == {'width': 1920, 'height': 1080}
    assert np.array(result['possible_contact_polygon_px'])[:, 0].min() == 355


def test_interpolated_detections_do_not_count_as_observed_flight():
    args = scene()
    for row in args[1][0]['observations']:
        row['interpolated'] = True
    result = estimate(args)
    assert result['status'] == 'review'
    assert 'NATIVE_ESTIMATE_FLIGHT_OBSERVATIONS_INSUFFICIENT' in result['reason_codes']


def test_occluder_report_is_a_rejection_not_a_contact_candidate():
    args = scene()
    args[1][0]['observations'][13]['possible_occluder_overlap'] = True
    result = estimate(args)
    assert 'NATIVE_ESTIMATE_OCCLUDER_OVERLAP' in result['reason_codes']


@pytest.mark.parametrize('error', [float('nan'), float('inf'), -1])
def test_invalid_camera_error_cannot_produce_finite_looking_certainty(error):
    args = scene()
    args[5]['camera_stability']['max_displacement_px'] = error
    result = estimate(args)
    assert result['status'] == 'review'


def test_camera_motion_error_is_added_to_measurement_uncertainty():
    args = scene()
    baseline = estimate(args)
    args[5]['camera_stability']['max_displacement_px'] = 2.5
    result = estimate(args)
    assert result['status'] == 'supported'
    assert result['uncertainty_px'] == pytest.approx(baseline['uncertainty_px']+2.5)


def test_clipped_source_roi_is_rejected():
    result = estimate(scene(x=10))
    assert result['status'] == 'review'
    assert 'NATIVE_ESTIMATE_SOURCE_ROI_CLIPPED' in result['reason_codes']


def test_time_or_source_corruption_returns_review():
    args = list(scene())
    args[3][15] = args[3][14]
    assert estimate(args)['status'] == 'review'
    args = list(scene())
    args[2][13] = np.zeros((100, 100, 3), np.uint8)
    assert estimate(args)['status'] == 'review'


def test_disabled_estimation_does_not_read_source():
    class Unreadable:
        def __len__(self):
            raise AssertionError('disabled feature read source')
    result = estimate_native_landing({}, [], Unreadable(), [], [960, 540], {}, config={'enabled': False})
    assert result['reason_codes'] == ['NATIVE_ESTIMATE_DISABLED']


def test_source_frames_are_read_once_in_increasing_order():
    args = list(scene())
    frames = args[2]
    class RecordingSource:
        indices = []
        def __len__(self):
            return len(frames)
        def __getitem__(self, index):
            self.indices.append(index)
            return frames[index]
    source = RecordingSource()
    args[2] = source
    assert estimate(args)['status'] == 'supported'
    assert source.indices == list(range(5, 24))


def test_small_secondary_shuttle_support_is_not_discarded_by_area_rank():
    args = list(scene())
    secondary = []
    for index in (16, 18):
        original = args[1][0]['observations'][index]
        x, y = int(original['x'])+25, int(original['y'])
        cv2.ellipse(args[2][index], (x, y-3), (2, 4), 0, 0, 360, (230, 230, 230), -1)
        cv2.circle(args[2][index], (x, y+2), 2, (50, 50, 50), -1)
        secondary.append({**original, 'x': float(x)})
    args[0]['track_ids'].append(2)
    args[1].append({'track_id': 2, 'observations': secondary})
    result = estimate(args)
    assert result['status'] == 'review'
    assert 'NATIVE_ESTIMATE_SECOND_OBJECT_SUPPORT' in result['reason_codes']
    assert len(result['alternative_component_audit']) == 2


def test_dark_foreground_alternatives_are_audited_against_independent_frames():
    args = list(scene())
    shadow_track = []
    for index in (16, 18):
        original = args[1][0]['observations'][index]
        x, y = int(original['x'])+17, int(original['y'])+20
        cv2.circle(args[2][index], (x, y), 8, (60, 60, 60), -1)
        shadow_track.append({**original, 'x': float(x), 'y': float(y)})
    args[0]['track_ids'].append(2)
    args[1].append({'track_id': 2, 'observations': shadow_track})
    result = estimate(args)
    assert result['status'] == 'supported'
    audit = result['alternative_component_audit']
    assert [row['frame_index'] for row in audit] == [16, 18]
    assert all(row['matching_component_count'] == 1 for row in audit)
    assert result['contact_observed'] is False
    assert result['observed_frame_count'] == 10


def test_native_recovery_counts_a_real_neighbor_only_after_independent_disambiguation():
    args = list(scene())
    args[0]['requires_contiguous_native_peak'] = True
    original = args[1][0]['observations'][16]
    x, y = int(original['x'])+17, int(original['y'])+20
    cv2.circle(args[2][16], (x, y), 8, (60, 60, 60), -1)
    args[0]['track_ids'].append(2)
    args[1].append({'track_id': 2, 'observations': [{**original, 'x': float(x), 'y': float(y)}]})
    result = estimate(args)
    assert result['status'] == 'supported'
    assert result['recovery_peak_coverage']['observed_frame_indices'] == list(range(11, 18))
    assert result['recovery_peak_coverage']['contiguous'] is True
    audit = result['alternative_component_audit']
    assert [row['frame_index'] for row in audit] == [16]
    assert audit[0]['matching_component_count'] == 1 and len(audit[0]['components']) == 2
    assert result['contact_observed'] is False


def _add_small_alternatives(args, positions):
    secondary = []
    for index, x_offset in positions:
        original = args[1][0]['observations'][index]
        x, y = int(original['x'])+x_offset, int(original['y'])
        cv2.ellipse(args[2][index], (x, y-3), (2, 4), 0, 0, 360, (230, 230, 230), -1)
        cv2.circle(args[2][index], (x, y+2), 2, (50, 50, 50), -1)
        secondary.append({**original, 'x': float(x)})
    args[0]['track_ids'].append(2)
    args[1].append({'track_id': 2, 'observations': secondary})


def test_isolated_fragments_are_not_counted_as_one_continuous_second_object():
    args = list(scene())
    _add_small_alternatives(args, [(11, 25), (15, 25)])
    result = estimate(args)
    assert result['status'] == 'supported'
    assert len(result['alternative_component_audit']) == 2
    assert result['secondary_component_links'] == []
    assert result['contact_observed'] is False


def test_spatially_disconnected_fragments_do_not_make_a_second_path():
    args = list(scene())
    _add_small_alternatives(args, [(16, -25), (18, 25)])
    result = estimate(args)
    assert result['status'] == 'supported'
    assert len(result['alternative_component_audit']) == 2
    assert result['secondary_component_links'] == []
    assert result['identity_confirmed'] is False


def test_temporally_and_spatially_linked_small_object_still_requires_review():
    args = list(scene())
    _add_small_alternatives(args, [(16, 25), (18, 25)])
    result = estimate(args)
    assert result['status'] == 'review'
    assert result['reason_codes'] == ['NATIVE_ESTIMATE_SECOND_OBJECT_SUPPORT']
    assert result['secondary_component_links'][0]['frame_interval'] == [16, 18]
