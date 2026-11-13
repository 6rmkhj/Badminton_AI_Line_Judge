"""Job-level GPU routing; mock devices are not hardware performance validation."""
from copy import deepcopy
import subprocess

import pytest

from badminton import detection, events, inference, pipeline


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / 'clip.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=160x160:rate=10:duration=0.6', '-c:v', 'libx264',
                    '-threads', '1', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


@pytest.mark.parametrize('selected', ['cpu', 'cuda:1', 'mps'])
def test_job_selection_overrides_stale_stage_devices_and_is_recorded(clip, tmp_path, monkeypatch, selected):
    checks, calls = [], {}

    def resolve(device='cpu', *, verify=False):
        checks.append((device, verify))
        return device

    monkeypatch.setattr(inference, 'resolve_device', resolve)
    cpu_detection, cpu_events = detection.detect_sequence, events.estimate_landings

    def detector(frames, timestamps, config=None, **kwargs):
        calls['detector'] = {key: config[key] for key in ('inference_device', 'gpu_memory_percent', 'gpu_parallel_views')}
        records = cpu_detection(frames, timestamps, {**config, 'inference_device': 'cpu'}, **kwargs)
        for row in records:
            row['detector'].update(inference_device=selected,
                                   gpu_stages=[] if selected == 'cpu' else ['candidate_masks'])
        return records

    def contact(*args, **kwargs):
        config = kwargs.get('config') if 'config' in kwargs else args[4]
        calls['events'] = deepcopy(config)
        return cpu_events(*args, **kwargs)

    monkeypatch.setattr(detection, 'detect_sequence', detector)
    monkeypatch.setattr(events, 'estimate_landings', contact)
    config = {'analysis_width': 160, 'evidence': {'annotated_clip': False},
              'detector': {'inference_device': 'cuda:9', 'person_backend': 'none', 'highres_enabled': False},
              'events': {'inference_device': 'cuda:9'},
              'resource_settings': {'device': selected, 'cpu_threads': 1, 'parallel_views': 2,
                                    'frame_memory_mb': 64, 'gpu_memory_percent': 30}}
    result = pipeline.analyze_video(clip, tmp_path / 'out', config)
    assert (selected, True) in checks
    assert calls['detector'] == {'inference_device': selected, 'gpu_memory_percent': 30, 'gpu_parallel_views': 2}
    assert calls['events']['inference_device'] == selected
    assert result['execution']['requested_device'] == selected
    assert result['execution']['detector_device'] == selected
    assert result['execution']['gpu_stages'] == ([] if selected == 'cpu' else ['candidate_masks'])
    assert result['execution']['automatic_cpu_fallback'] is False


def test_unavailable_selected_gpu_fails_before_video_conversion(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise inference.InferenceDeviceError('GPU가 연결되지 않았습니다.')
    monkeypatch.setattr(inference, 'resolve_device', unavailable)
    monkeypatch.setattr(pipeline, 'prepare_preview', lambda *args, **kwargs: pytest.fail('Preview must not run'))
    with pytest.raises(inference.InferenceDeviceError, match='연결되지'):
        pipeline.analyze_video(tmp_path / 'source.mp4', tmp_path / 'out',
                               {'resource_settings': {'device': 'cuda:0'}})
