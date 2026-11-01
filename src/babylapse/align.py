"""Map each source frame so the face lands on a fixed canonical position.

A 5-point similarity transform (rotation + uniform scale + translation) is
fitted per frame against a canonical face template.  The transform is then
re-parametrised as (source anchor x/y, log-scale, angle), gaps are
interpolated, and the parameters are smoothed over time to avoid jitter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ArcFace 112x112 reference landmarks: right eye, left eye, nose, right mouth, left mouth.
_ARCFACE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]
)
_EYE_MID = _ARCFACE[:2].mean(axis=0)
_EYE_DIST = float(np.linalg.norm(_ARCFACE[0] - _ARCFACE[1]))
# Template normalised to eye midpoint = origin, inter-eye distance = 1.
TEMPLATE = (_ARCFACE - _EYE_MID) / _EYE_DIST
# Weight eyes higher: they are the most reliable landmarks and the visual anchor.
WEIGHTS = np.array([2.0, 2.0, 1.0, 0.75, 0.75])


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    eye_y: float = 0.42  # eye midpoint height as fraction of output height
    face_scale: float = 0.12  # inter-eye distance as fraction of output height

    @property
    def anchor(self) -> np.ndarray:
        return np.array([self.width / 2.0, self.eye_y * self.height])

    def target_points(self) -> np.ndarray:
        return self.anchor + TEMPLATE * (self.face_scale * self.height)


def fit_similarity(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted least-squares similarity transform (Umeyama, no reflection). Returns 2x3 matrix."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (w[:, None, None] * d0[:, :, None] * s0[:, None, :]).sum(0)
    u, sig, vt = np.linalg.svd(cov)
    d = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[1, 1] = -1
    r = u @ d @ vt
    var_s = (w * (s0**2).sum(1)).sum()
    scale = float(np.trace(np.diag(sig) @ d) / var_s)
    m = np.zeros((2, 3))
    m[:, :2] = scale * r
    m[:, 2] = mu_d - scale * r @ mu_s
    return m


def matrix_to_params(m: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """(cx, cy, log_scale, angle): source point mapped onto ``anchor``, scale, rotation."""
    a, b = m[0, 0], m[1, 0]
    scale = float(np.hypot(a, b))
    angle = float(np.arctan2(b, a))
    c = np.linalg.solve(m[:, :2], anchor - m[:, 2])
    return np.array([c[0], c[1], np.log(scale), angle])


def params_to_matrix(p: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    cx, cy, log_s, angle = p
    s = np.exp(log_s)
    r = s * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    m = np.zeros((2, 3))
    m[:, :2] = r
    m[:, 2] = anchor - r @ np.array([cx, cy])
    return m


def cover_params(frame_shape: tuple[int, int], layout: Layout, fit: bool = False) -> np.ndarray:
    """Scale the frame to cover the output, centred (fallback when no face is found).

    With ``fit=True`` the whole frame is shown instead (no cropping); the rest of
    the output is filled according to the fill option.
    """
    h, w = frame_shape
    s = (min if fit else max)(layout.width / w, layout.height / h)
    # The source point that must land on the anchor so that the frame centre lands on the output centre.
    out_c = np.array([layout.width / 2.0, layout.height / 2.0])
    c = np.array([w / 2.0, h / 2.0]) + (layout.anchor - out_c) / s
    return np.array([c[0], c[1], np.log(s), 0.0])


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(x) < 2:
        return x
    radius = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, j], k, mode="valid") for j in range(x.shape[1])], axis=1)


def build_track(
    times: np.ndarray,
    landmarks: list[np.ndarray | None],
    layout: Layout,
    frame_shape: tuple[int, int],
    mode: str = "smooth",
    smooth_seconds: float = 0.4,
) -> np.ndarray:
    """Per-frame transform parameters, shape (N, 4).

    mode="smooth": follow the face, low-pass filtered (sigma = ``smooth_seconds``).
    mode="fixed":  one transform per segment (median); no motion, face may drift.
    mode="none":   no face alignment; the whole (upright) frame is fitted into the output.
    """
    times = np.asarray(times, float)
    n = len(times)
    if mode == "none":
        return np.tile(cover_params(frame_shape, layout, fit=True), (n, 1))
    target = layout.target_points()
    params = np.full((n, 4), np.nan)
    for i, lm in enumerate(landmarks):
        if lm is not None:
            params[i] = matrix_to_params(fit_similarity(lm, target, WEIGHTS), layout.anchor)
    valid = ~np.isnan(params[:, 0])
    if not valid.any():
        return np.tile(cover_params(frame_shape, layout), (n, 1))

    params[valid, 3] = np.unwrap(params[valid, 3])
    for j in range(4):
        params[:, j] = np.interp(times, times[valid], params[valid, j])

    if mode == "fixed":
        return np.tile(np.median(params, axis=0), (n, 1))
    if mode != "smooth":
        raise ValueError(f"unknown alignment mode {mode!r}")
    dt = float(np.median(np.diff(times))) if n > 1 else 1.0
    # Median filter first to reject single-frame detection glitches, then Gaussian low-pass.
    return _gaussian_smooth(_median_filter(params, 2), smooth_seconds / max(dt, 1e-3))


def _median_filter(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(x) < 3:
        return x
    padded = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1, axis=0)
    return np.median(windows, axis=-1)
