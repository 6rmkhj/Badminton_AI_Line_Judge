"""Timestamp-aware, conservative two-view synchronization using recorded audio."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter


def _base(status="review", reasons=None):
    return {"status": status, "offset_seconds": None, "proposed_offset_seconds": None,
            "offset_convention": "t_b = t_a + offset_seconds",
            "confidence": 0.0, "confidence_kind": "uncalibrated_alignment_score",
            "method": "audio_onset_envelope_cross_correlation", "reason_codes": reasons or [],
            "overlap_seconds": 0.0, "uncertainty_seconds": None,
            "same_rally_verified": False,
            "precision_note": "Audio alignment does not establish the same rally, physical camera capture-time precision, or rolling shutter."}


def _envelope(audio, sample_rate, hop_seconds):
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values -= values.mean() if len(values) else 0.0
    hop = max(1, int(round(sample_rate * hop_seconds)))
    count = len(values) // hop
    if count < 3:
        return np.empty(0)
    # RMS envelopes tolerate the different waveforms/reflections seen by two mics.
    rms = np.sqrt(np.mean(values[:count * hop].reshape(count, hop) ** 2, axis=1))
    transformed = np.log1p(rms / max(float(np.median(rms)), 1e-6))
    smooth_size = max(3, int(round(0.25 / hop_seconds)) | 1)
    background = median_filter(transformed, size=min(smooth_size, count | 1), mode="nearest")
    return np.maximum(transformed - background, 0.0)


def _correlations(a, b, minimum_bins, maximum_lag=None):
    na, nb = len(a), len(b)
    lags = signal.correlation_lags(nb, na, mode="full")
    dots = signal.correlate(b, a, mode="full", method="fft")
    start_a, start_b = np.maximum(0, -lags), np.maximum(0, lags)
    lengths = np.minimum(na - start_a, nb - start_b)
    cumulative_a = np.r_[0.0, np.cumsum(a)]
    cumulative_b = np.r_[0.0, np.cumsum(b)]
    squares_a = np.r_[0.0, np.cumsum(a * a)]
    squares_b = np.r_[0.0, np.cumsum(b * b)]
    sums_a = cumulative_a[start_a + lengths] - cumulative_a[start_a]
    sums_b = cumulative_b[start_b + lengths] - cumulative_b[start_b]
    variance_a = squares_a[start_a + lengths] - squares_a[start_a] - sums_a ** 2 / lengths
    variance_b = squares_b[start_b + lengths] - squares_b[start_b] - sums_b ** 2 / lengths
    denominator = np.sqrt(np.maximum(variance_a, 0) * np.maximum(variance_b, 0))
    valid = (lengths >= minimum_bins) & (denominator > 1e-8)
    if maximum_lag is not None:
        valid &= np.abs(lags) <= maximum_lag
    score = np.full(len(lags), -np.inf)
    score[valid] = (dots[valid] - sums_a[valid] * sums_b[valid] / lengths[valid]) / denominator[valid]
    return lags, np.clip(score, -1, 1), valid, lengths


def _local_support(a, b, lag, hop_seconds):
    sa, sb = max(0, -lag), max(0, lag)
    count = min(len(a) - sa, len(b) - sb)
    minimum = max(20, round(0.35 / hop_seconds))
    if count < minimum:
        return []
    windows = []
    parts = min(3, count // minimum)
    for indices in np.array_split(np.arange(count), parts):
        aa, bb = a[sa + indices], b[sb + indices]
        if np.std(aa) < 1e-5 or np.std(bb) < 1e-5:
            continue
        lags, scores, valid, _ = _correlations(aa, bb, max(12, len(indices) // 2), round(0.12 / hop_seconds))
        if not valid.any():
            continue
        peak = int(np.argmax(np.where(valid, scores, -np.inf)))
        windows.append({"start_a_seconds": round((sa + int(indices[0])) * hop_seconds, 6),
                        "local_offset_residual_seconds": round(float(lags[peak] * hop_seconds), 6),
                        "correlation": round(float(scores[peak]), 5)})
    return windows


def synchronize_audio_arrays(audio_a, audio_b, sample_rate, config=None):
    """Align decoded mono audio; positive offset means event appears later in B."""
    cfg = dict(config or {})
    result = _base()
    if not isinstance(sample_rate, (int, float)) or not math.isfinite(sample_rate) or sample_rate <= 0:
        return _base("failed", ["INVALID_AUDIO_SAMPLE_RATE"])
    a, b = np.asarray(audio_a, dtype=float).reshape(-1), np.asarray(audio_b, dtype=float).reshape(-1)
    if not len(a) or not len(b):
        return _base("review", ["AUDIO_MISSING_OR_EMPTY"])
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return _base("failed", ["NONFINITE_AUDIO_SAMPLES"])
    rms_a, rms_b = float(np.std(a)), float(np.std(b))
    if min(rms_a, rms_b) < float(cfg.get("silence_rms", 1e-5)):
        return _base("review", ["AUDIO_SILENT_OR_TOO_QUIET"])
    hop_seconds = max(0.001, float(cfg.get("envelope_hop_seconds", 0.005)))
    hop_seconds = max(1, round(sample_rate * hop_seconds)) / sample_rate
    a_feature, b_feature = _envelope(a, sample_rate, hop_seconds), _envelope(b, sample_rate, hop_seconds)
    min_overlap = float(cfg.get("minimum_overlap_seconds", 1.0))
    minimum_bins = max(10, int(math.ceil(min_overlap / hop_seconds)))
    if min(len(a_feature), len(b_feature)) < minimum_bins:
        return _base("review", ["INSUFFICIENT_AUDIO_OVERLAP"])
    max_offset = cfg.get("maximum_offset_seconds")
    max_bins = int(abs(float(max_offset)) / hop_seconds) if max_offset is not None else None
    lags, correlations, valid, overlaps = _correlations(a_feature, b_feature, minimum_bins, max_bins)
    if not valid.any():
        return _base("review", ["AUDIO_HAS_NO_DISTINCT_TRANSIENTS"])
    peak = int(np.argmax(np.where(valid, correlations, -np.inf)))
    lag, correlation = int(lags[peak]), float(correlations[peak])
    exclusion = max(1, int(round(float(cfg.get("peak_exclusion_seconds", 0.12)) / hop_seconds)))
    competing = valid & (np.abs(lags - lag) > exclusion)
    second = float(np.max(correlations[competing])) if competing.any() else -1.0
    prominence = correlation - second
    offset = lag * hop_seconds
    support = _local_support(a_feature, b_feature, lag, hop_seconds)
    consistent = [window for window in support
                  if abs(window["local_offset_residual_seconds"]) <= 0.025 and window["correlation"] >= 0.25]
    # Width is measured locally, not across unrelated secondary peaks.
    near_peak = valid & (np.abs(lags - lag) <= exclusion) & (correlations >= correlation - 0.05)
    width = (int(lags[near_peak].max()) - int(lags[near_peak].min())) * hop_seconds if near_peak.any() else hop_seconds
    overlap = int(overlaps[peak]) * hop_seconds
    reasons = []
    if correlation < float(cfg.get("minimum_correlation", 0.4)):
        reasons.append("AUDIO_CORRELATION_WEAK")
    if prominence < float(cfg.get("minimum_peak_prominence", 0.08)):
        reasons.append("AUDIO_ALIGNMENT_AMBIGUOUS")
    if len(consistent) < int(cfg.get("minimum_supporting_windows", 2)):
        reasons.append("AUDIO_WINDOW_AGREEMENT_INSUFFICIENT")
    if width > float(cfg.get("maximum_peak_width_seconds", 0.12)):
        reasons.append("AUDIO_ALIGNMENT_PEAK_TOO_BROAD")
    confidence = np.clip(max(0, correlation) * min(1.0, max(0, prominence) / 0.3) * min(1.0, len(consistent) / 2), 0, 1)
    result.update({"status": "review" if reasons else "ok", "reason_codes": reasons,
                   "proposed_offset_seconds": round(offset, 6),
                   "offset_seconds": None if reasons else round(offset, 6),
                   "confidence": round(float(confidence), 5),
                   "overlap_seconds": round(overlap, 6),
                   "uncertainty_seconds": None if reasons else round(max(hop_seconds, width / 2), 6),
                   "proposal_peak_halfwidth_seconds": round(max(hop_seconds, width / 2), 6),
                   "correlation": round(correlation, 5), "second_peak_correlation": round(second, 5),
                   "peak_prominence": round(prominence, 5), "peak_width_seconds": round(width, 6),
                   "supporting_windows": len(consistent), "window_diagnostics": support,
                   "audio_duration_a_seconds": round(len(a) / sample_rate, 6),
                   "audio_duration_b_seconds": round(len(b) / sample_rate, 6),
                   "sample_rate": sample_rate, "envelope_hop_seconds": hop_seconds})
    return result


def synchronize_event_anchors(correspondences, duration_a=None, duration_b=None, config=None):
    """Use independently reviewed event timestamps, explicitly as manual sync.

    At least two distinct events must agree on a constant offset. This does not
    establish that a landing is visible or label a line decision as correct.
    Unreviewed machine suggestions are intentionally excluded.
    """
    cfg = dict(config or {})
    pairs = []
    for entry in correspondences or []:
        if entry.get("verified") is not True:
            continue
        try:
            time_a, time_b = float(entry["time_a"]), float(entry["time_b"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(time_a) or not math.isfinite(time_b) or min(time_a, time_b) < 0:
            continue
        if duration_a is not None and time_a > duration_a:
            continue
        if duration_b is not None and time_b > duration_b:
            continue
        pairs.append((time_a, time_b))
    pairs = sorted(set(pairs))
    result = _base("review", ["VERIFIED_EVENT_ANCHORS_INSUFFICIENT"])
    result["method"] = "operator_verified_event_anchors"
    result["source"] = "manual_review"
    result["confidence_kind"] = "not_a_probability_manual_anchor_consistency"
    result["verified_event_count"] = len(pairs)
    if len(pairs) < 2:
        return result
    minimum_span = max(0.0, float(cfg.get("minimum_anchor_span_seconds", 0.3)))
    if pairs[-1][0] - pairs[0][0] < minimum_span or max(b for _, b in pairs) - min(b for _, b in pairs) < minimum_span:
        result["reason_codes"] = ["VERIFIED_EVENT_ANCHORS_TOO_CLOSE"]
        return result
    offsets = np.array([b - a for a, b in pairs])
    offset = float(np.median(offsets))
    residuals = offsets - offset
    maximum_error = float(cfg.get("maximum_event_residual_seconds", 0.05))
    result["proposed_offset_seconds"] = round(offset, 6)
    result["verified_event_residuals_seconds"] = residuals.tolist()
    if np.max(np.abs(residuals)) > maximum_error:
        result["reason_codes"] = ["VERIFIED_EVENT_ANCHORS_INCONSISTENT"]
        return result
    result.update({"status": "ok", "reason_codes": [], "offset_seconds": round(offset, 6),
                   "confidence": 1.0,
                   "uncertainty_seconds": max(float(cfg.get("manual_timestamp_uncertainty_seconds", 1 / 30)),
                                              float(np.max(np.abs(residuals))))})
    if duration_a is not None and duration_b is not None:
        result["overlap_seconds"] = round(max(0.0, min(duration_a, duration_b - offset) - max(0.0, -offset)), 6)
    return result


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_audio(path, sample_rate, cache_dir=None):
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ], text=True, timeout=30))
    streams = [stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "audio"]
    if not streams:
        return None, metadata, 0.0
    stream = streams[0]
    start = float(stream.get("start_time", 0) or 0)
    cached_path = None
    if cache_dir:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        state = path.stat()
        key = hashlib.sha256(f"{path.resolve()}:{state.st_size}:{state.st_mtime_ns}:{sample_rate}".encode()).hexdigest()
        cached_path = cache / f"{key}.f32"
        if cached_path.is_file():
            return np.fromfile(cached_path, dtype="<f4"), metadata, start
    process = subprocess.run([
        "ffmpeg", "-hide_banner", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"
    ], capture_output=True, timeout=120, check=True)
    values = np.frombuffer(process.stdout, dtype="<f4").copy()
    if cached_path:
        with tempfile.NamedTemporaryFile(dir=cached_path.parent, prefix=cached_path.stem, suffix=".tmp", delete=False) as temporary:
            temporary.write(process.stdout)
            temporary_path = Path(temporary.name)
        temporary_path.replace(cached_path)
    return values, metadata, start


def synchronize(video_a, video_b, config=None):
    """Read two local videos and return an evidence-qualified synchronization."""
    cfg = dict(config or {})
    path_a, path_b = Path(video_a), Path(video_b)
    if not path_a.is_file() or not path_b.is_file():
        return _base("failed", ["VIDEO_INPUT_MISSING"])
    if path_a.resolve() == path_b.resolve() or (path_a.stat().st_size == path_b.stat().st_size and _file_hash(path_a) == _file_hash(path_b)):
        return _base("review", ["DUPLICATE_VIDEO_CONTENT_NOT_INDEPENDENT_VIEWS"])
    rate = max(1000, min(48000, int(cfg.get("sample_rate", 8000))))
    try:
        audio_a, metadata_a, start_a = _read_audio(path_a, rate, cfg.get("cache_dir"))
        audio_b, metadata_b, start_b = _read_audio(path_b, rate, cfg.get("cache_dir"))
    except (subprocess.SubprocessError, ValueError, OSError) as error:
        result = _base("failed", ["AUDIO_EXTRACTION_FAILED"])
        result["error"] = str(error)[:1000]
        return result
    duration_a = float(metadata_a.get("format", {}).get("duration", len(audio_a) / rate if audio_a is not None else 0))
    duration_b = float(metadata_b.get("format", {}).get("duration", len(audio_b) / rate if audio_b is not None else 0))
    manual = synchronize_event_anchors(cfg.get("event_correspondences", []), duration_a, duration_b, cfg)
    if audio_a is None or audio_b is None:
        if manual["status"] == "ok":
            manual["audio_alignment_status"] = "unavailable"
            return manual
        result = _base("review", ["AUDIO_STREAM_UNAVAILABLE"])
        result["manual_anchor_result"] = manual
        return result
    result = synchronize_audio_arrays(audio_a, audio_b, rate, cfg)
    result["audio_start_a_seconds"], result["audio_start_b_seconds"] = start_a, start_b
    origin_adjustment = start_b - start_a
    for key in ("offset_seconds", "proposed_offset_seconds"):
        if result.get(key) is not None:
            result[key] = round(result[key] + origin_adjustment, 6)
    proposed = result.get("proposed_offset_seconds")
    if proposed is not None:
        result["overlap_seconds"] = round(max(0.0, min(duration_a, duration_b - proposed) - max(0.0, -proposed)), 6)
    # Optional independent event correspondences check an audio proposal. They
    # cannot convert a weak audio result into a confident alignment silently.
    correspondences = cfg.get("event_correspondences", [])
    residuals = []
    for pair in correspondences:
        if not pair.get("verified") or proposed is None:
            continue
        try:
            residual = float(pair["time_b"]) - float(pair["time_a"]) - proposed
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(residual):
            residuals.append(residual)
    if residuals:
        result["verified_event_residuals_seconds"] = residuals
        threshold = float(cfg.get("maximum_event_residual_seconds", 0.05))
        if max(abs(value) for value in residuals) > threshold:
            result["status"] = "review"
            result["offset_seconds"] = None
            result["reason_codes"].append("AUDIO_OFFSET_CONTRADICTS_VERIFIED_EVENTS")
    result["verified_event_count"] = len(residuals)
    if manual["status"] == "ok":
        # A conscious manual alignment is a separate result, never relabeled as
        # an automatic audio success. Preserve the audio diagnostics for review.
        manual["audio_alignment"] = result
        manual["audio_alignment_status"] = result["status"]
        return manual
    return result
