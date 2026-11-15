"""Image codecs with Python file I/O for Unicode paths on Windows.

OpenCV's filename-based image I/O does not reliably accept non-ASCII Windows
paths. Keep filenames out of the codec API, including temporary replay frames.
"""
from __future__ import annotations

import errno
from pathlib import Path
import uuid

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    """Decode a file; return None for invalid image bytes, propagate I/O errors."""
    data = Path(path).read_bytes()
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)


def write_image(path, image, params=None):
    """Encode and atomically replace an image, or raise a useful OSError.

The temporary file is closed before replacement, as required on Windows. An
encoding or write failure cannot advertise a partially written final image.
"""
    destination = Path(path)
    try:
        encoded, data = cv2.imencode(destination.suffix, image, params or [])
    except cv2.error as error:
        raise OSError(errno.EIO, f'Could not encode image: {str(error)[:500]}', str(destination)) from error
    if not encoded:
        raise OSError(errno.EIO, 'Could not encode image', str(destination))
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_bytes(data.tobytes())
        temporary.replace(destination)
    except OSError as error:
        raise OSError(error.errno, f'Could not write image: {error.strerror or error}', str(destination)) from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # Preserve the original storage error if cleanup is also denied.
            pass
