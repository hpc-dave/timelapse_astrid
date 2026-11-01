"""Render one aligned segment (video via OpenCV, encoding + audio via ffmpeg) and concatenate."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .align import Layout, build_track, params_to_matrix
from .faces import Face, FaceDetector
from .metadata import VideoInfo
from .video import iter_frames


@dataclass
class RenderOptions:
    layout: Layout
    fps: float = 30.0
    crf: int = 20
    preset: str = "medium"
    align_mode: str = "smooth"
    smooth_seconds: float = 0.4
    fill: str = "blur"  # "blur" or "black"
    audio: bool = True
    date_format: str = "%d/%m/%Y"
    font_scale: float = 1.0
    speed: float = 1.0  # playback speed of the final video (2.0 = twice as fast)


def track_segment(info: VideoInfo, start: float, end: float, detector: FaceDetector, opts: RenderOptions):
    """Pass 1: detect the face in every frame of the segment; return timestamps and transform params."""
    times: list[float] = []
    landmarks: list[np.ndarray | None] = []
    shape = (info.height, info.width)
    prev: Face | None = None
    for t, frame in iter_frames(info.path, info.fps, start, end):
        shape = frame.shape[:2]
        face = detector.detect(frame, prev)
        if face is not None:
            prev = face
        times.append(t)
        landmarks.append(None if face is None else face.landmarks)
    if not times:
        raise RuntimeError(f"{info.path}: no frames decoded in [{start:.2f}, {end:.2f})")
    params = build_track(np.array(times), landmarks, opts.layout, shape, opts.align_mode, opts.smooth_seconds)
    coverage = sum(lm is not None for lm in landmarks) / len(landmarks)
    return np.array(times), params, coverage


def _background(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Blurred, darkened 'cover' version of the frame used to fill areas outside the warped image."""
    fh, fw = frame.shape[:2]
    s = max(w / fw, h / fh) / 8.0  # work at 1/8 resolution: blur is cheap there
    sw, sh = max(1, round(w / 8)), max(1, round(h / 8))
    m = np.array([[s, 0, sw / 2 - s * fw / 2], [0, s, sh / 2 - s * fh / 2]])
    small = cv2.warpAffine(frame, m, (sw, sh), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    small = cv2.GaussianBlur(small, (0, 0), 4)
    return (cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) * 0.6).astype(np.uint8)


def draw_date(img: np.ndarray, text: str, font_scale: float = 1.0) -> None:
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = font_scale * h / 900.0
    thick = max(1, round(2 * scale))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    margin = round(0.03 * h)
    org = (w - tw - margin, h - margin - base)
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thick + max(2, round(3 * scale)), cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _ffmpeg_segment_cmd(info: VideoInfo, start: float, dur: float, out: Path, opts: RenderOptions) -> list[str]:
    w, h = opts.layout.width, opts.layout.height
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{opts.fps}", "-i", "-",
    ]
    if opts.audio:
        if info.has_audio:
            cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(info.path)]
        else:
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        fade_out = max(0.0, dur - 0.15)
        cmd += [
            "-map", "0:v", "-map", "1:a:0",
            "-af", f"afade=t=in:d=0.15,afade=t=out:st={fade_out:.3f}:d=0.15,apad",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        ]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf), "-pix_fmt", "yuv420p",
        "-t", f"{dur:.3f}", str(out),
    ]
    return cmd


def render_segment(
    info: VideoInfo,
    start: float,
    end: float,
    recorded: datetime,
    out: Path,
    detector: FaceDetector,
    opts: RenderOptions,
) -> float:
    """Render ``[start, end)`` of ``info`` aligned to the canonical face layout. Returns face coverage."""
    times, params, coverage = track_segment(info, start, end, detector, opts)
    w, h = opts.layout.width, opts.layout.height
    anchor = opts.layout.anchor
    label = recorded.strftime(opts.date_format)
    dur = end - start
    n_out = max(1, round(dur * opts.fps))

    proc = subprocess.Popen(_ffmpeg_segment_cmd(info, start, dur, out, opts), stdin=subprocess.PIPE)
    try:
        k = 0
        last: np.ndarray | None = None

        def emit(frame: np.ndarray, t: float) -> None:
            p = np.array([np.interp(t, times, params[:, j]) for j in range(4)])
            m = params_to_matrix(p, anchor)
            if opts.fill == "blur":
                out_img = _background(frame, w, h)
                warped = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR)
                mask = cv2.warpAffine(np.full(frame.shape[:2], 255, np.uint8), m, (w, h), flags=cv2.INTER_LINEAR)
                a = (mask.astype(np.float32) / 255.0)[..., None]
                out_img = (warped * a + out_img * (1.0 - a)).astype(np.uint8)
            else:
                out_img = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            draw_date(out_img, label, opts.font_scale)
            proc.stdin.write(out_img.tobytes())

        # Constant-frame-rate resampling: output frame k shows the latest source frame with ts <= start + k/fps.
        for t, frame in iter_frames(info.path, info.fps, start, end):
            while k < n_out and start + k / opts.fps < t and last is not None:
                emit(last[1], last[0])
                k += 1
            last = (t, frame)
        while k < n_out and last is not None:
            emit(last[1], last[0])
            k += 1
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed while encoding segment of {info.path}")
    return coverage


def concat(segments: list[Path], out: Path) -> None:
    lst = out.with_suffix(".concat.txt")
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in segments))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
             "-c", "copy", "-movflags", "+faststart", str(out)],
            check=True,
        )
    finally:
        lst.unlink(missing_ok=True)


def atempo_chain(factor: float) -> str:
    """ffmpeg audio filter that changes tempo by ``factor`` without changing pitch.

    Older ffmpeg builds only accept 0.5 <= atempo <= 2, so larger or smaller
    factors are split into a chain of steps within that range.
    """
    if factor <= 0:
        raise ValueError("speed factor must be positive")
    steps = []
    while factor > 2.0:
        steps.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        steps.append(0.5)
        factor /= 0.5
    steps.append(factor)
    return ",".join(f"atempo={s:.6g}" for s in steps)


def speed_up(src: Path, out: Path, factor: float, opts: RenderOptions) -> None:
    """Re-encode ``src`` played ``factor`` times faster (``factor < 1`` slows it down).

    The frame rate stays at ``opts.fps``: when speeding up, frames are dropped
    rather than shown faster. Audio keeps its pitch.
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-filter:v", f"setpts=PTS/{factor:.6g}", "-r", f"{opts.fps:g}",
           "-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf), "-pix_fmt", "yuv420p"]
    if opts.audio:
        cmd += ["-filter:a", atempo_chain(factor), "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    subprocess.run(cmd + ["-movflags", "+faststart", str(out)], check=True)
