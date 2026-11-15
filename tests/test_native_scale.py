"""Axis-scale regressions: source resize cannot change a line verdict."""
import cv2
import numpy as np
import pytest

from badminton.native_landing import estimate_native_landing
from badminton.video_estimate import evaluate_possible_region
from test_native_landing import scene


def boundary_court():
    return {'image_width': 960, 'image_height': 540,
        'camera_stability': {'status': 'ok', 'all_frames_sampled': True, 'reason_codes': []},
        'primary_court': {'status': 'selected', 'scope': 'near_half', 'confidence': .9,
            'region_supported': True, 'game': 'doubles', 'rules': {'game': 'doubles', 'phase': 'rally'},
            'polygon_px': [[710, 300], [900, 300], [900, 180], [710, 180]],
            'judgment_scope_polygon_px': [[680, 330], [930, 330], [930, 150], [680, 150]],
            'boundary_segments_px': [
                {'role': role, 'segment_px': segment, 'observed': True,
                 'inside_point_px': [800, 240], 'uncertainty_px': .2}
                for role, segment in [('near', [710, 300, 900, 300]),
                                      ('left', [710, 300, 710, 180]),
                                      ('right', [900, 300, 900, 180])]]}}


@pytest.mark.parametrize('source_size', [(978, 540), (960, 550)])
def test_nonuniform_source_resize_preserves_analysis_region_and_boundary_review(source_size, tmp_path):
    reference_args = scene(x=710)
    reference = estimate_native_landing(*reference_args)
    scaled_args = list(scene(x=710))
    scaled_args[2] = [cv2.resize(frame, source_size, interpolation=cv2.INTER_NEAREST)
                      for frame in scaled_args[2]]
    result = estimate_native_landing(*scaled_args, output_dir=tmp_path)
    assert reference['status'] == result['status'] == 'supported'
    # Source pixel rounding is bounded by one original analysis pixel. The old
    # averaged transform displaced x by 7.32 px for the 978x540 source.
    assert np.asarray(result['possible_contact_polygon_px']) == pytest.approx(
        np.asarray(reference['possible_contact_polygon_px']), abs=1.)
    assert result['source_scale_xy'] == pytest.approx([source_size[0]/960, source_size[1]/540])
    assert result['uncertainty_components']['foreground_quantization_px'] >= max(960/source_size[0], 540/source_size[1])
    for native in (reference, result):
        geometry = evaluate_possible_region(native['possible_contact_polygon_px'], native['uncertainty_px'], boundary_court())
        assert geometry['decision'] == 'REVIEW'
        assert geometry['reason_codes'] == ['VIDEO_ESTIMATE_BOUNDARY_AMBIGUOUS']
    # Source evidence drawing must use the same axis-wise transform.
    assert cv2.imread(str(tmp_path / result['artifact'])) is not None
