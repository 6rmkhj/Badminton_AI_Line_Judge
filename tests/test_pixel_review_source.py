from pathlib import Path

import pytest

from badminton import pipeline


@pytest.mark.parametrize('enabled', [False, True])
def test_pixel_review_requires_source_pixels_even_without_detector_source_mode(tmp_path, monkeypatch, enabled):
    source = tmp_path / 'input.mp4'
    source.write_bytes(b'pipeline-input-route-test')
    output = tmp_path / 'result'
    monkeypatch.setattr(pipeline, 'prepare_preview', lambda *args, **kwargs: (
        {}, {'frame_count': 7, 'width': 96, 'height': 54, 'timestamps': [i / 30 for i in range(7)]}, 'source-hash'))
    loaded = []

    def stop_after_input_route(path, *args, **kwargs):
        loaded.append((Path(path), kwargs.get('resize_to')))
        raise RuntimeError('stop after input routing')

    monkeypatch.setattr(pipeline, 'load_frames', stop_after_input_route)
    with pytest.raises(RuntimeError, match='stop after input routing'):
        pipeline._analyze_video(source, output, {
            'detector': {'source_analysis_enabled': False}, 'events': {'scene_context_ranking': enabled}})
    assert loaded == [(source if enabled else output / 'preview.mp4', (96, 54) if enabled else None)]
