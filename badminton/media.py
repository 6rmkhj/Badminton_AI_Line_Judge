"""Timestamp-aware decoding and browser-compatible evidence videos."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import cv2
from .resources import decoder_threads, ResourceSettingsError


def _positive_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _frame_rate(value):
    try:
        numerator, denominator = str(value).split('/')
        return _positive_number(float(numerator) / float(denominator))
    except (ValueError, ZeroDivisionError, OverflowError):
        return _positive_number(value)


class _FrameProgress:
    """Report observed work at most twice a second, retaining the final count."""
    def __init__(self, callback, total, detail):
        self.callback, self.total, self.detail = callback, total, detail
        self.last_time, self.last_count = float('-inf'), None

    def report(self, count, *, force=False):
        if self.callback is None or count == self.last_count:
            return
        now = time.monotonic()
        if force or now - self.last_time >= .5:
            self.callback({'completed_units': int(count), 'total_units': self.total,
                           'unit': 'frames', 'detail': self.detail})
            self.last_time, self.last_count = now, count


def _run_ffmpeg_progress(command, callback, total, detail, timeout):
    """Read FFmpeg's actual encoded-frame counter without blocking its stderr.

    The output file is polled while communicate drains stderr. A process that
    stops reporting progress still has the same wall-clock timeout as before.
    """
    reporter = _FrameProgress(callback, total, detail)
    reporter.report(0, force=True)
    with tempfile.TemporaryDirectory(prefix='badminton-progress-') as folder:
        progress = Path(folder) / 'ffmpeg.txt'
        observed = [command[0], '-progress', str(progress), '-stats_period', '0.5', '-nostats', *command[1:]]
        process = subprocess.Popen(observed, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   text=True, encoding='utf-8', errors='replace')
        deadline = time.monotonic() + timeout

        def observed_count():
            if not progress.exists():
                return None
            values = [line.partition('=')[2].strip() for line in progress.read_text(encoding='utf-8').splitlines()
                      if line.startswith('frame=')]
            counts = [int(value) for value in values if value.isdigit()]
            return min(counts[-1], total) if counts else None

        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    _, stderr = process.communicate(timeout=min(.5, remaining))
                    break
                except subprocess.TimeoutExpired:
                    count = observed_count()
                    if count is not None:
                        reporter.report(count)
        except BaseException:
            process.kill()
            process.communicate()
            raise
        if process.returncode == 0:
            count = observed_count()
            if count is not None:
                reporter.report(count, force=True)
        return subprocess.CompletedProcess(command, process.returncode, stdout='', stderr=stderr)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path):
    raw = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
        '-show_packets', '-show_entries',
        'stream=index,codec_name,width,height,duration,avg_frame_rate,nb_frames,bit_rate,pix_fmt:format=format_name,bit_rate:stream_side_data=rotation:stream_tags=rotate:packet=pts_time,flags',
        '-of', 'json', str(path)], timeout=120)
    data = json.loads(raw)
    if not data.get('streams'):
        raise ValueError('No video stream')
    stream = data['streams'][0]
    timestamps = sorted(float(packet['pts_time']) for packet in data.get('packets', [])
                        if 'pts_time' in packet and 'D' not in packet.get('flags', ''))
    if not timestamps:
        raise ValueError('Video has no presentation timestamps')
    start = timestamps[0]
    timestamps = [round(t - start, 9) for t in timestamps]
    periods = sorted(b - a for a, b in zip(timestamps, timestamps[1:]) if b > a)
    period = periods[len(periods) // 2] if periods else 1 / 30
    warnings = []
    if len(set(timestamps)) != len(timestamps):
        warnings.append('DUPLICATE_PRESENTATION_TIMESTAMP')
    if any(b - a > period * 1.5 for a, b in zip(timestamps, timestamps[1:])):
        warnings.append('PRESENTATION_GAP')
    rotation = float(stream.get('tags', {}).get('rotate', 0))
    for side_data in stream.get('side_data_list', []):
        if 'rotation' in side_data:
            rotation = float(side_data['rotation'])
    coded_width, coded_height = int(stream['width']), int(stream['height'])
    width, height = (coded_height, coded_width) if round(abs(rotation)) % 180 == 90 else (coded_width, coded_height)
    stored = stream.get('nb_frames', '')
    format_info = data.get('format') or {}
    try:
        size_bytes = Path(path).stat().st_size
    except OSError:
        size_bytes = None
    return {'width': width, 'height': height, 'coded_width': coded_width, 'coded_height': coded_height,
            'rotation_degrees': rotation,
            'codec': stream.get('codec_name'), 'duration': float(stream.get('duration') or timestamps[-1] + period),
            'fps': 1 / period, 'frame_count': len(timestamps),
            'average_fps': _frame_rate(stream.get('avg_frame_rate')),
            'variable_frame_rate': any(abs(p - period) > period * .02 for p in periods) if periods else None,
            'extension': Path(path).suffix.lower() or None, 'size_bytes': size_bytes,
            'container': format_info.get('format_name'), 'pixel_format': stream.get('pix_fmt'),
            'bitrate_bps': _positive_number(stream.get('bit_rate')),
            'container_bitrate_bps': _positive_number(format_info.get('bit_rate')),
            'stored_frame_count': int(stored) if str(stored).isdigit() else None,
            'timestamps': timestamps, 'time_origin_seconds': start, 'warnings': warnings}


def prepare_preview(source, destination, width=1280, *, progress_callback=None):
    source, destination = Path(source).resolve(), Path(destination)
    metadata = probe(source)
    width = min(int(width), metadata['width'])
    if width < 160:
        raise ValueError('Analysis width must be at least 160')
    width -= width % 2
    height = max(2, round(metadata['height'] * width / metadata['width'] / 2) * 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    signature = {'source_sha256': file_hash(source), 'width': width, 'height': height, 'version': 1}
    sidecar = destination.with_suffix('.source.json')
    if not destination.exists() or not sidecar.exists() or json.loads(sidecar.read_text(encoding='utf-8')) != signature:
        temporary = destination.with_name(destination.stem + f'.partial-{uuid.uuid4().hex}.mp4')
        command = ['ffmpeg', '-hide_banner', '-nostdin', '-y', '-loglevel', 'error',
                   '-xerror', '-err_detect', 'explode', '-threads', str(decoder_threads()), '-filter_threads', '1',
                   '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
                   '-vf', f'scale={width}:{height}:flags=area,setpts=PTS-STARTPTS',
                   '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21', '-pix_fmt', 'yuv420p',
                   '-threads', str(decoder_threads()), '-fps_mode', 'passthrough', '-enc_time_base', 'demux',
                   '-bsf:v', f'setts=pts=PTS:dts=DTS:duration=1/({metadata["fps"]:.9f}*TB)',
                   '-movflags', '+faststart', str(temporary)]
        if progress_callback is None:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       encoding='utf-8', errors='replace', timeout=1800)
        else:
            completed = _run_ffmpeg_progress(command, progress_callback, metadata['frame_count'],
                                             '미리보기 변환', 1800)
        (destination.parent / 'preview-encode.log').write_text(completed.stderr, encoding='utf-8')
        if completed.returncode or completed.stderr.strip():
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f'Preview encoding failed: {completed.stderr[-2000:]}')
        temporary.replace(destination)
        sidecar.write_text(json.dumps(signature, indent=2) + '\n', encoding='utf-8')
    if progress_callback is not None:
        progress_callback({'detail': '미리보기 프레임·시각 검증'})
    repair_preview_tail(destination, metadata['fps'])
    presentation = probe(destination)
    if (presentation['frame_count'] != metadata['frame_count'] or
            (presentation['width'], presentation['height']) != (width, height) or
            any(abs(a-b) > .0001 for a,b in zip(metadata['timestamps'], presentation['timestamps']))):
        raise ValueError('Preview no longer matches source frames, coordinates or timestamps; remove it to rebuild')
    return metadata, presentation, signature['source_sha256']


def repair_preview_tail(path, fps=None):
    """Give the final frame display time without changing pixels, PTS or DTS.

    Some variable-rate MP4 encodes omit the final packet duration. HTML video
    then ends at that frame's PTS, so the last frame cannot be selected.
    """
    path = Path(path)
    before = probe(path)
    fps = float(fps or before['fps'])
    expected_count = before['stored_frame_count'] or before['frame_count']
    if before['frame_count'] == expected_count and before['duration'] - before['timestamps'][-1] >= .5 / fps:
        return False
    if before['frame_count'] != expected_count:
        packets = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
            '-show_entries', 'packet=pts_time,flags', '-of', 'json', str(path)], timeout=120))['packets']
        last_displayed = before['time_origin_seconds'] + before['timestamps'][-1]
        if any('D' in packet.get('flags', '') and
               float(packet.get('pts_time', '-inf')) <= last_displayed + .000001 for packet in packets):
            raise ValueError('Preview has non-tail discarded frames; refusing to alter its edit list')
    temporary = path.with_name(path.stem + f'.duration-{uuid.uuid4().hex}.mp4')
    command = ['ffmpeg', '-hide_banner', '-nostdin', '-y', '-v', 'error', '-ignore_editlist', '1', '-i', str(path),
               '-map', '0:v:0', '-c:v', 'copy', '-bsf:v',
               f'setts=pts=PTS-STARTPTS:dts=DTS-STARTPTS:duration=1/({fps:.9f}*TB)',
               '-movflags', '+faststart', str(temporary)]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    after = probe(temporary)
    if (len(after['timestamps']) != expected_count or
            any(abs(a-b) > 0.0001 for a,b in zip(before['timestamps'], after['timestamps']))):
        temporary.unlink(missing_ok=True)
        raise ValueError('Preview duration repair changed presentation timestamps')
    temporary.replace(path)
    return True


def load_frames(path, expected_count=None, max_frames=3000, resize_to=None, *, progress_callback=None,
                max_memory_bytes=None):
    if max_memory_bytes is not None and (isinstance(max_memory_bytes, bool)
            or not isinstance(max_memory_bytes, int) or max_memory_bytes < 1):
        raise ValueError('프레임 메모리 한도는 1바이트 이상의 정수여야 합니다.')
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, decoder_threads()])
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    frames = []
    memory_bytes = 0
    reporter = _FrameProgress(progress_callback, expected_count, '분석 프레임 읽기')
    reporter.report(0, force=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if len(frames) >= max_frames:
                raise ValueError(f'Clip exceeds {max_frames} frames; split it before analysis')
            if resize_to is not None and (frame.shape[1], frame.shape[0]) != tuple(resize_to):
                frame = cv2.resize(frame, tuple(resize_to), interpolation=cv2.INTER_AREA)
            if max_memory_bytes is not None and memory_bytes + frame.nbytes > max_memory_bytes:
                raise ResourceSettingsError('분석 프레임 메모리 한도를 초과했습니다. 장치 설정에서 프레임 버퍼 예산을 늘리거나 영상 구간을 줄여 주세요.')
            frames.append(frame)
            memory_bytes += frame.nbytes
            reporter.report(len(frames))
    finally:
        capture.release()
    if not frames:
        raise ValueError('Decoded zero frames')
    if expected_count is not None and len(frames) != expected_count:
        raise ValueError(f'Timestamp/frame count mismatch: {len(frames)} != {expected_count}')
    reporter.report(len(frames), force=True)
    return frames


class SourceFrameSequence:
    """Read displayed original frames with a one-frame memory cache.

    OpenCV applies the file's rotation and edit list, matching ``load_frames``.
    Large skips may seek, but only a matching decoded presentation timestamp
    can accept the result. Unsupported or inaccurate seeking falls back to
    sequential decoding from the beginning, preserving exact frame indexing.
    """
    def __init__(self, path, count, *, verified_seek=True):
        self.path, self.count = str(path), int(count)
        self.capture = None
        self.index, self.frame = -1, None
        self._seek_enabled = bool(verified_seek)
        self._timestamps, self._decoder_origin = None, None
        self.seek_statistics = {'attempts': 0, 'accepted': 0, 'fallbacks': 0}

    def __len__(self):
        return self.count

    def _open(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, decoder_threads()])
        self.index, self.frame, self._decoder_origin = -1, None, None

    def _grab(self, requested):
        if not self.capture.grab():
            raise ValueError(f'Original-resolution frame {requested} is unavailable')
        self.index += 1
        if self.index == 0:
            # Backends can report source timestamps or stream-relative ones.
            # Calibrate from an actually decoded first frame, not a seek index.
            self._decoder_origin = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000

    def _verified_seek(self, index):
        if not self._seek_enabled:
            return False
        if self._timestamps is None:
            try:
                timestamps = probe(self.path)['timestamps']
                if (len(timestamps) != self.count or not all(math.isfinite(t) for t in timestamps)
                        or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
                    self._seek_enabled = False
                    return False
                self._timestamps = timestamps
            except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                # Random access is an optimization; it must not add a probe
                # dependency to callers that could already decode the source.
                self._seek_enabled = False
                return False
        if self.index < 0:
            self._grab(index)
        expected = self._decoder_origin + self._timestamps[index]
        if not math.isfinite(expected):
            self._seek_enabled = False
            return False
        adjacent_steps = ([self._timestamps[index] - self._timestamps[index - 1]] if index else [])
        if index + 1 < self.count:
            adjacent_steps.append(self._timestamps[index + 1] - self._timestamps[index])
        tolerance = min(.0001, min(adjacent_steps, default=.001) * .25)
        self.seek_statistics['attempts'] += 1
        try:
            moved = self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            grabbed = moved and self.capture.grab()
            actual = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 if grabbed else math.nan
            # ffprobe prints microsecond precision. A tolerance below one
            # tenth of a millisecond covers rounding, never a normal frame.
            valid = grabbed and math.isfinite(actual) and abs(actual - expected) <= tolerance
        except cv2.error:
            valid = False
        if not valid:
            self.seek_statistics['fallbacks'] += 1
            self._open()
            return False
        self.index, self.frame = index, None
        self.seek_statistics['accepted'] += 1
        return True

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < self.count:
            raise IndexError(index)
        if index == self.index and self.frame is not None:
            return self.frame
        if self.capture is None or index < self.index:
            self._open()
        # Sequential detector reads do not probe or seek. Evidence windows
        # starting late in a clip can avoid decoding every preceding GOP.
        if index - self.index >= 32:
            self._verified_seek(index)
        while self.index < index:
            self._grab(index)
        ok, self.frame = self.capture.retrieve()
        if not ok or self.frame is None:
            raise ValueError(f'Original-resolution frame {index} is unavailable')
        return self.frame

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.frame = None
        self.index = -1
        self._decoder_origin = None
        self._timestamps = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def source_crop(source, time_seconds, center_px, output_path, radius=192, frame_index=None):
    """Save original-resolution context; this does not certify physical contact."""
    metadata = probe(source)
    if frame_index is None:
        frame_index = min(range(len(metadata['timestamps'])), key=lambda i: abs(metadata['timestamps'][i] - time_seconds))
    if not 0 <= frame_index < len(metadata['timestamps']):
        return {'status': 'frame_out_of_range'}
    x, y = map(int, center_px)
    h, w = metadata['height'], metadata['width']
    x0, x1 = max(0, x-radius), min(w, x+radius)
    y0, y1 = max(0, y-radius), min(h, y+radius)
    if x1 <= x0 or y1 <= y0:
        return {'status': 'outside_frame'}
    command = ['ffmpeg', '-hide_banner', '-nostdin', '-y', '-loglevel', 'error',
               '-threads', str(decoder_threads()), '-filter_threads', '1', '-i', str(source), '-map', '0:v:0',
               '-vf', f'select=eq(n\\,{frame_index}),crop={x1-x0}:{y1-y0}:{x0}:{y0}',
               '-frames:v', '1', '-fps_mode', 'passthrough', '-q:v', '2', '-threads', '1', str(output_path)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=180)
    if completed.returncode or not Path(output_path).exists():
        return {'status': 'unavailable', 'error': completed.stderr[-1000:]}
    return {'status': 'saved', 'requested_time': time_seconds,
            'reported_time': metadata['timestamps'][frame_index], 'decoded_frame_index': frame_index,
            'alignment_error_seconds': metadata['timestamps'][frame_index] - time_seconds,
            'source_box': [x0, y0, x1, y1],
            'note': 'Original-resolution decoded frame crop; frame alignment does not certify physical contact.'}
