"""Frame reading helpers (OpenCV applies the rotation metadata of phone videos)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def iter_frames(
    path: Path,
    fps_hint: float,
    start: float = 0.0,
    end: float | None = None,
    decode_every: float | None = None,
) -> Iterator[tuple[float, np.ndarray | None]]:
    """Yield ``(timestamp_s, frame)`` for frames with ``start <= t < end``.

    If ``decode_every`` is given, only frames at least that many seconds apart
    are decoded; the others are yielded with ``frame=None`` (cheap ``grab``).
    Seeking is done to ~2 s before ``start`` and verified via timestamps.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    try:
        index = 0
        if start > 2.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, (start - 2.0) * 1000.0)
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos > start or pos <= 0:  # unreliable seek: start over from the beginning
                cap.release()
                cap = cv2.VideoCapture(str(path))
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
                yield t, frame
            else:
                yield t, None
    finally:
        cap.release()
