"""Frame reading helpers. Frames are returned upright: the rotation metadata of phone videos is applied."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    # Apply the rotation tag (display matrix) the way players do. This is OpenCV's
    # default with the FFmpeg backend; set it explicitly so it never silently changes.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


_ROTATE_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Turn a frame clockwise by 0, 90, 180 or 270 degrees."""
    degrees %= 360
    return frame if degrees == 0 else cv2.rotate(frame, _ROTATE_CODES[degrees])


def iter_frames(
    path: Path,
    fps_hint: float,
    start: float = 0.0,
    end: float | None = None,
    decode_every: float | None = None,
    rotate: int = 0,
) -> Iterator[tuple[float, np.ndarray | None]]:
    """Yield ``(timestamp_s, frame)`` for frames with ``start <= t < end``.

    If ``decode_every`` is given, only frames at least that many seconds apart
    are decoded; the others are yielded with ``frame=None`` (cheap ``grab``).
    Seeking is done to ~2 s before ``start`` and verified via timestamps.
    ``rotate`` turns every frame clockwise on top of the file's rotation metadata.
    """
    cap = _open(path)
    try:
        index = 0
        if start > 2.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, (start - 2.0) * 1000.0)
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos > start or pos <= 0:  # unreliable seek: start over from the beginning
                cap.release()
                cap = _open(path)
            else:
                index = int(round(pos * fps_hint))
        last_t = -1.0
        next_decode = start
        while True:
            if not cap.grab():
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= last_t or (t == 0.0 and index > 0):  # backend without usable timestamps
                t = index / fps_hint
            last_t = t
            index += 1
            if t < start:
                continue
            if end is not None and t >= end:
                break
            if decode_every is None or t + 1e-6 >= next_decode:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if decode_every is not None:
                    next_decode = t + decode_every
                yield t, rotate_frame(frame, rotate)
            else:
                yield t, None
    finally:
        cap.release()
