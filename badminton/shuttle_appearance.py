"""Does a landing candidate look like a shuttle or part of a larger moving body?

A shuttle touching the floor is a tiny object (a few analysis pixels) on an
otherwise unchanged floor. A knee bending and straightening, a foot planting,
a racket swing or shorts moving past produce the same "descent then rebound"
image motion as a landing, but they sit inside a large region that differs
from the clip background. This module measures that region.

``foreground_fraction`` is the share of a disk around the candidate (radius
:data:`RADIUS_PX` at a 960-px-wide frame) whose largest per-channel
difference from the clip's temporal-median background exceeds
:data:`DIFFERENCE_THRESHOLD`. Candidates above :data:`MAX_FOREGROUND_FRACTION`
are treated as "on a larger moving object".

Frames are always the analysis frames decoded from ``preview.mp4`` (or the
same frames already in memory), so a saved analysis re-evaluated later gets
identical measurements. The check is skipped when the camera is not fixed,
because a moving camera breaks the background model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np


logger = logging.getLogger(__name__)
METHOD = 'clip_median_background_v1'
DIFFERENCE_THRESHOLD = 30      # max-channel |frame - background| counted as foreground
RADIUS_PX = 20.0               # disk radius at a 960-px-wide frame
BACKGROUND_SAMPLES = 25        # evenly spaced frames in the median background
# A 2-3 px shuttle covers well under 3% of the disk even with motion blur; a
# body part covers a large share. 0.10 leaves room for a nearby shoe edge.
MAX_FOREGROUND_FRACTION = 0.10


def _sample_indices(count):
    return sorted({int(i) for i in np.linspace(0, count - 1, min(BACKGROUND_SAMPLES, count)).round()})


def _fraction(frame, background, point, width):
    radius = max(3, int(round(RADIUS_PX * width / 960.0)))
    height, frame_width = frame.shape[:2]
    x, y = float(point[0]), float(point[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - radius), min(frame_width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = np.abs(frame[y0:y1, x0:x1].astype(np.int16) - background[y0:y1, x0:x1]).max(axis=2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    if not disk.any():
        return None
    return float((patch[disk] > DIFFERENCE_THRESHOLD).mean())


class ForegroundProbe:
    """Batch foreground-fraction measurements over one clip's analysis frames.

    Build it from frames already in memory, or from ``preview.mp4`` with
    :meth:`from_preview`; the preview is decoded twice in a streaming fashion
    (background samples, then the requested frames) so memory stays small.
    """

    def __init__(self, frames=None, *, preview_path=None, frame_count=None):
        self._frames = frames
        self._preview = Path(preview_path) if preview_path is not None else None
        self._count = len(frames) if frames is not None else (int(frame_count) if frame_count else None)
        self._background = None
        self._failed = False

    @classmethod
    def from_preview(cls, preview_path, frame_count):
        path = Path(preview_path)
        if not path.is_file() or not frame_count:
            return None
        return cls(preview_path=path, frame_count=frame_count)

    # -- decoding ---------------------------------------------------------
    def _stream(self, wanted):
        """Yield (index, frame) for the wanted indices, in order."""
        if self._frames is not None:
            for index in sorted(wanted):
                if 0 <= index < len(self._frames):
                    yield index, self._frames[index]
            return
        capture = cv2.VideoCapture(str(self._preview))
        if not capture.isOpened():
            raise ValueError('Cannot open preview for appearance check')
        try:
            last = max(wanted) if wanted else -1
            index = 0
            while index <= last:
                ok, frame = capture.read()
                if not ok:
                    break
                if index in wanted:
                    yield index, frame
                index += 1
        finally:
            capture.release()

    def _ensure_background(self):
        if self._background is not None or self._failed:
            return self._background
        try:
            if not self._count:
                raise ValueError('Unknown frame count')
            indices = _sample_indices(self._count)
            samples = [frame for _, frame in self._stream(set(indices))]
            if len(samples) != len(indices):
                raise ValueError('Preview frame count differs from the analysis')
            self._background = np.median(np.stack(samples), axis=0).astype(np.int16)
        except Exception:  # an optional check must never break the estimate
            logger.exception('Could not build the appearance background')
            self._failed = True
        return self._background

    # -- public -----------------------------------------------------------
    def frame(self, index):
        """One analysis frame (for evidence images), or None."""
        found = list(self._stream({int(index)}))
        return found[0][1] if found else None

    def fractions(self, requests):
        """Foreground fraction for each (frame_index, point); None when unmeasured."""
        results = [None] * len(requests)
        background = self._ensure_background()
        if background is None:
            return results
        width = background.shape[1]
        by_frame = {}
        for position, (index, point) in enumerate(requests):
            try:
                key = int(index)
            except (TypeError, ValueError):
                continue
            by_frame.setdefault(key, []).append((position, point))
        try:
            for index, frame in self._stream(set(by_frame)):
                for position, point in by_frame[index]:
                    results[position] = _fraction(frame, background, point, width)
        except Exception:
            logger.exception('Could not measure candidate appearance')
            return [None] * len(requests)
        return results


def is_isolated(fraction):
    """True/False when measured, None when the check could not run."""
    if fraction is None:
        return None
    return fraction <= MAX_FOREGROUND_FRACTION
