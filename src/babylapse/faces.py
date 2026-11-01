"""Face detection with OpenCV's YuNet (5 landmarks: eyes, nose, mouth corners)."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:  # OpenCV 5 warns about DNN targets on every detector creation; keep the console clean.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def default_model_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "babylapse"
    return cache / "face_detection_yunet_2023mar.onnx"


def ensure_model(path: Path | None = None) -> Path:
    path = Path(path) if path else default_model_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading YuNet face model to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(YUNET_URL, tmp)
        tmp.replace(path)
    return path


@dataclass
class Face:
    box: np.ndarray  # x, y, w, h
    landmarks: np.ndarray  # (5, 2): right eye, left eye, nose, right mouth, left mouth (image coords)
    score: float

    @property
    def eye_center(self) -> np.ndarray:
        return self.landmarks[:2].mean(axis=0)

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.landmarks[0] - self.landmarks[1]))


class FaceDetector:
    """Detects the most prominent face; runs on a downscaled copy for speed."""

    def __init__(self, model: Path | None = None, score_threshold: float = 0.6, max_side: int = 640):
        self.model = str(ensure_model(model))
        self.score_threshold = score_threshold
        self.max_side = max_side
        self._det = None
        self._size = None

    def _detector(self, size: tuple[int, int]):
        if self._det is None:
            self._det = cv2.FaceDetectorYN.create(self.model, "", size, self.score_threshold, 0.3, 5000)
            self._size = size
        elif self._size != size:
            self._det.setInputSize(size)
            self._size = size
        return self._det

    def detect(self, frame: np.ndarray, prev: Face | None = None) -> Face | None:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = frame if scale == 1.0 else cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        _, faces = self._detector((sw, sh)).detect(small)
        if faces is None or len(faces) == 0:
            return None
        cands = [
            Face(box=f[0:4] / scale, landmarks=f[4:14].reshape(5, 2) / scale, score=float(f[14]))
            for f in faces
        ]
        if prev is not None:
            # Track: prefer the face closest to the previous one (avoid jumping to a parent's face).
            ref = prev.eye_center
            return min(cands, key=lambda c: np.linalg.norm(c.eye_center - ref) / max(prev.eye_distance, 1.0) - c.score)
        # Otherwise: biggest confident face (the baby is usually the subject filling the frame).
        return max(cands, key=lambda c: c.score * c.box[2] * c.box[3])
