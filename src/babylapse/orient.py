"""Guess the upright orientation of videos that carry no rotation metadata.

Some phone videos are saved without a rotation tag (typically when the phone
points straight down while recording starts), so they play sideways. For such
videos a handful of frames is turned by 0, 90, 180 and 270 degrees and the face
detector is run on each. YuNet is clearly more confident on upright faces, so
the turn with the highest mean face score wins, provided it beats the stored
orientation by a clear margin.
"""

from __future__ import annotations

from dataclasses import dataclass

from .faces import FaceDetector
from .metadata import VideoInfo
from .video import iter_frames, rotate_frame

ANGLES = (0, 90, 180, 270)


@dataclass
class OrientationGuess:
    rotate: int  # clockwise correction in degrees (0 = keep as is)
    scores: dict[int, float]  # mean face score per correction (0 for frames without a face)
    frames: int  # number of frames examined


def choose_rotation(scores: dict[int, float], min_score: float = 0.75, margin: float = 0.15) -> int:
    """Leave the stored orientation only when another correction finds faces
    confidently (mean score >= ``min_score``) and beats 0 degrees by ``margin``.

    On upright, correctly tagged phone videos the stored orientation scores
    0.8-0.95 and wrong turns stay at least ~0.05 below it, so the margin keeps
    babies who lie sideways in an upright video from being turned.
    """
    best = max(ANGLES, key=lambda a: scores.get(a, 0.0))
    if best != 0 and scores[best] >= min_score and scores[best] - scores.get(0, 0.0) >= margin:
        return best
    return 0


def guess_rotation(info: VideoInfo, detector: FaceDetector, n_frames: int = 12) -> OrientationGuess:
    """Examine ``n_frames`` frames spread over the video and guess the clockwise correction."""
    totals = {a: 0.0 for a in ANGLES}
    step = max(info.duration / n_frames, 0.1)
    frames = 0
    for _, frame in iter_frames(info.path, info.fps, start=step / 2, decode_every=step):
        if frame is None:
            continue
        frames += 1
        for a in ANGLES:
            face = detector.detect(rotate_frame(frame, a))
            if face is not None:
                totals[a] += face.score
    scores = {a: (t / frames if frames else 0.0) for a, t in totals.items()}
    return OrientationGuess(choose_rotation(scores), scores, frames)
