from pathlib import Path
import numpy as np
import pytest

from badminton.sync import synchronize, synchronize_audio_arrays, synchronize_event_anchors


def audio(seed=4, rate=8000, seconds=5):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.0001, rate * seconds)
    for i, position in enumerate([0.32, 0.71, 1.29, 1.85, 2.04, 2.79, 3.21, 3.77, 4.38]):
        n = int(rate * (0.022 + i * 0.002))
        start = int(position * rate)
        values[start:start + n] += rng.normal(0, 0.4 + i * 0.04, n) * np.hanning(n)
    return values


@pytest.mark.parametrize("offset", [0.235, -0.31])
def test_offset_sign_and_precision(offset):
    rate = 8000
    a = audio()
    shift = round(abs(offset) * rate)
    b = np.r_[np.zeros(shift), a] if offset > 0 else a[shift:]
    result = synchronize_audio_arrays(a, b, rate)
    assert result["status"] == "ok", result
    assert result["offset_seconds"] == pytest.approx(offset, abs=0.01)
    assert result["supporting_windows"] >= 2


def test_repeated_periodic_audio_is_ambiguous():
    rate = 8000
    rng = np.random.default_rng(12)
    one = np.zeros(rate // 2)
    one[400:560] = rng.normal(0, 1, 160) * np.hanning(160)
    a = np.tile(one, 10)
    result = synchronize_audio_arrays(a, a.copy(), rate)
    assert result["status"] == "review"
    assert result["offset_seconds"] is None
    assert "AUDIO_ALIGNMENT_AMBIGUOUS" in result["reason_codes"]


def test_unrelated_and_silent_audio_are_not_aligned():
    result = synchronize_audio_arrays(np.zeros(16000), np.zeros(16000), 8000)
    assert result["offset_seconds"] is None and result["status"] == "review"
    a = audio()
    b = np.random.default_rng(9).normal(0, 0.2, len(a))
    result = synchronize_audio_arrays(a, b, 8000)
    assert result["status"] == "review", result
    assert result["offset_seconds"] is None


def test_duplicate_files_do_not_prove_two_camera_sync(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"same video")
    b.write_bytes(a.read_bytes())
    result = synchronize(a, b)
    assert result["status"] == "review"
    assert "DUPLICATE_VIDEO_CONTENT_NOT_INDEPENDENT_VIEWS" in result["reason_codes"]


def test_missing_video_is_explicit(tmp_path):
    assert synchronize(tmp_path / "missing-a", tmp_path / "missing-b")["status"] == "failed"


def test_reviewed_anchors_are_explicit_manual_fallback():
    result = synchronize_event_anchors([
        {"time_a": 0.5, "time_b": 0.8, "verified": True},
        {"time_a": 2, "time_b": 2.3, "verified": True},
    ], duration_a=4, duration_b=4)
    assert result["status"] == "ok"
    assert result["method"] == "operator_verified_event_anchors"
    assert result["source"] == "manual_review"
    assert result["offset_seconds"] == pytest.approx(0.3)
    assert result["overlap_seconds"] == pytest.approx(3.7)


def test_unreviewed_or_inconsistent_anchors_cannot_prove_sync():
    assert synchronize_event_anchors([
        {"time_a": 0.5, "time_b": 0.8, "verified": False},
        {"time_a": 2, "time_b": 2.3, "verified": False},
    ])["status"] == "review"
    result = synchronize_event_anchors([
        {"time_a": 0.5, "time_b": 0.8, "verified": True},
        {"time_a": 2, "time_b": 3, "verified": True},
    ])
    assert result["status"] == "review"
    assert result["offset_seconds"] is None


def test_no_audio_and_reviewed_fallback(tmp_path):
    import subprocess
    paths = []
    for color in ("red", "blue"):
        path = tmp_path / f"{color}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-nostdin", "-f", "lavfi",
                        "-i", f"color=c={color}:s=32x32:r=10:d=2", "-c:v", "libx264", "-threads", "1", str(path)], check=True)
        paths.append(path)
    result = synchronize(*paths)
    assert result["status"] == "review"
    assert "AUDIO_STREAM_UNAVAILABLE" in result["reason_codes"]
    result = synchronize(*paths, {"event_correspondences": [
        {"time_a": 0.5, "time_b": 0.7, "verified": True},
        {"time_a": 1.5, "time_b": 1.7, "verified": True},
    ]})
    assert result["status"] == "ok"
    assert result["method"] == "operator_verified_event_anchors"
