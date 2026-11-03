"""Pick the best 10 s window of a video: face visible, confident, steady."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .faces import Face, FaceDetector
from .metadata import VideoInfo
from .video import iter_frames


@dataclass
class Sample:
    t: float
    face: Face | None


@dataclass
class Window:
    start: float
    end: float
    score: float
    coverage: float  # fraction of samples with a detected face


def analyze(info: VideoInfo, detector: FaceDetector, sample_fps: float = 2.0) -> tuple[list[Sample], tuple[int, int]]:
    """Run face detection on frames sampled at ``sample_fps``."""
    samples: list[Sample] = []
    shape = (info.height, info.width)
    prev: Face | None = None
    for t, frame in iter_frames(info.path, info.fps, decode_every=1.0 / sample_fps, rotate=info.extra_rotation):
        if frame is None:
            continue
        shape = frame.shape[:2]
        face = detector.detect(frame, prev)
        samples.append(Sample(t, face))
        prev = face if face is not None else prev
    return samples, shape


def choose_window(
    samples: list[Sample],
    duration: float,
    frame_shape: tuple[int, int],
    length: float = 10.0,
    step: float = 0.5,
) -> Window:
    """Score every candidate window and return the best one.

    score = coverage * mean confidence
            - 0.5 * face motion (eye-centre std / eye distance)
            - 0.5 * relative face-size variation
            + 0.25 * face size relative to the frame (bigger faces align better)
    """
    if duration <= length or not samples:
        cov = float(np.mean([s.face is not None for s in samples])) if samples else 0.0
        return Window(0.0, min(duration, length) if duration > 0 else length, cov, cov)

    min_side = float(min(frame_shape))
    ts = np.array([s.t for s in samples])
    best: Window | None = None
    for start in np.arange(0.0, duration - length + 1e-9, step):
        sel = [s for s, t in zip(samples, ts) if start <= t < start + length]
        if not sel:
            continue
        faces = [s.face for s in sel if s.face is not None]
        coverage = len(faces) / len(sel)
        if faces:
            conf = float(np.mean([f.score for f in faces]))
            eye_d = np.array([f.eye_distance for f in faces])
            med_d = max(float(np.median(eye_d)), 1.0)
            centers = np.array([f.eye_center for f in faces])
            motion = float(np.linalg.norm(centers.std(axis=0))) / med_d if len(faces) > 1 else 0.0
            size_var = float(eye_d.std()) / med_d
            size = min(1.0, med_d / (0.25 * min_side))
            score = coverage * conf - 0.5 * min(motion, 2.0) - 0.5 * min(size_var, 1.0) + 0.25 * size
        else:
            score = -1.0
        if best is None or score > best.score + 1e-9:
            best = Window(float(start), float(start) + length, score, coverage)
    assert best is not None
    return best
