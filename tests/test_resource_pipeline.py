"""Resource choices affect allocations rather than only UI labels."""
from copy import deepcopy

import pytest

from badminton import media, pipeline


POLICY = {"device": "cpu", "cpu_threads": 1, "parallel_views": 2, "frame_memory_mb": 64}


class DecodeReached(Exception):
    pass


def patch_probe(monkeypatch, metadata):
    monkeypatch.setattr(media, "probe", lambda path: deepcopy(metadata))
    if hasattr(pipeline, "probe"):
        monkeypatch.setattr(pipeline, "probe", lambda path: deepcopy(metadata))


def test_oversized_video_is_rejected_before_preview_or_decoding(tmp_path, monkeypatch):
    from badminton.resources import ResourceSettingsError
    source = tmp_path / "large.mp4"
    source.write_bytes(b"source")
    patch_probe(monkeypatch, {"width": 3840, "height": 2160, "frame_count": 3000})

    def unexpected_work(*args, **kwargs):
        pytest.fail("Oversized analysis must fail before preview encoding/frame allocation")

    monkeypatch.setattr(pipeline, "prepare_preview", unexpected_work)
    monkeypatch.setattr(pipeline, "load_frames", unexpected_work)
    with pytest.raises(ResourceSettingsError, match="예산|버퍼"):
        pipeline.analyze_video(source, tmp_path / "out", {
            "analysis_width": 960, "resource_settings": POLICY,
        })


@pytest.mark.parametrize("source_decode", [False, True])
@pytest.mark.parametrize("parallel_views", [1, 2])
def test_each_decode_path_receives_its_share_of_total_frame_budget(
        tmp_path, monkeypatch, source_decode, parallel_views):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    metadata = {"width": 320, "height": 180, "frame_count": 2, "timestamps": [0., .04]}
    patch_probe(monkeypatch, metadata)
    monkeypatch.setattr(pipeline, "prepare_preview", lambda *args, **kwargs: (
        deepcopy(metadata), deepcopy(metadata), "a" * 64))
    calls = []

    def capture_decode(path, expected_count=None, max_frames=3000, resize_to=None, **kwargs):
        calls.append((path, expected_count, resize_to, kwargs))
        raise DecodeReached

    monkeypatch.setattr(pipeline, "load_frames", capture_decode)
    policy = {**POLICY, "parallel_views": parallel_views}
    with pytest.raises(DecodeReached):
        pipeline.analyze_video(source, tmp_path / "out", {
            "analysis_width": 320, "resource_settings": policy,
            "detector": {"source_analysis_enabled": source_decode},
        })
    assert len(calls) == 1
    path, count, size, options = calls[0]
    assert options["max_memory_bytes"] == 64 * 1024 * 1024 // parallel_views
    assert count == 2
    expected_path = source.resolve() if source_decode else (tmp_path / "out" / "preview.mp4").resolve()
    assert path == expected_path
    assert size == ((320, 180) if source_decode else None)


def test_legacy_cli_without_resource_policy_keeps_existing_decode_contract(tmp_path, monkeypatch):
    metadata = {"width": 320, "height": 180, "frame_count": 1, "timestamps": [0.]}
    monkeypatch.setattr(pipeline, "prepare_preview", lambda *args, **kwargs: (
        deepcopy(metadata), deepcopy(metadata), "b" * 64))

    def original_signature(path, expected_count, max_frames, resize_to=None):
        raise DecodeReached

    monkeypatch.setattr(pipeline, "load_frames", original_signature)
    with pytest.raises(DecodeReached):
        pipeline.analyze_video(tmp_path / "clip.mp4", tmp_path / "out", {"analysis_width": 320})
