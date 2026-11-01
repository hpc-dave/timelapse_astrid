"""Self-contained babylapse demo: no baby videos needed.

It makes a few synthetic "baby videos" from one face photo, then runs babylapse on them.
The face drifts, tilts and zooms, and each video has a different size, orientation,
length and recording date, so you can see the date handling, window choice and
face alignment at work.

    .venv/bin/python examples/demo.py                 # uses a public-domain NASA photo
    .venv/bin/python examples/demo.py --face me.jpg   # or any photo with one face

Output goes to examples/output/: the input videos in videos/, plus plan.json and
timelapse.mp4.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from babylapse.cli import main as babylapse
from babylapse.faces import FaceDetector

HERE = Path(__file__).resolve().parent
# Astronaut Eileen Collins (NASA, public domain), as shipped with scikit-image.
FACE_URL = "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.24.0/skimage/data/astronaut.png"

# file name, creation_time tag (None: date comes from the file name), width, height,
# duration (s), face size (fraction of the shorter side), base tilt (deg), face position
CLIPS = [
    ("first_bath.mov", "2025-01-02T23:30:00Z", 720, 1280, 20, 0.35, -20, (0.40, 0.40)),
    ("park.mp4", "2025-03-15T10:00:00Z", 1280, 720, 15, 0.20, 15, (0.65, 0.50)),
    ("VID_20250601_120000.mp4", None, 960, 540, 6, 0.15, 0, (0.50, 0.55)),
]


def load_face(path: Path | None, cache: Path) -> np.ndarray:
    if path is None:
        path = cache / "astronaut.png"
        if not path.exists():
            print(f"Downloading sample face to {path}")
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(FACE_URL, path)
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read image {path}")
    return img


def face_center_and_size(img: np.ndarray) -> tuple[np.ndarray, float]:
    face = FaceDetector(None, 0.6).detect(img)
    if face is None:
        raise SystemExit("no face found in the image; try another photo")
    x, y, w, h = face.box
    return np.array([x + w / 2, y + h / 2]), float(max(w, h))


def make_video(img, center, face_px, out: Path, created, W, H, dur, size, tilt, pos, fps=30):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
        "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac",
    ]
    if created:
        cmd += ["-metadata", f"creation_time={created}"]
    proc = subprocess.Popen(cmd + [str(out)], stdin=subprocess.PIPE)
    for i in range(dur * fps):
        t = i / fps
        frame = np.full((H, W, 3), (60 + i % 80, 110, 150), np.uint8)
        scale = size * min(W, H) / face_px * (1 + 0.15 * np.sin(0.6 * t))
        angle = tilt + 12 * np.sin(0.7 * t)
        m = cv2.getRotationMatrix2D(tuple(map(float, center)), angle, scale)
        target = np.array([pos[0] * W + 0.12 * W * np.sin(0.5 * t), pos[1] * H + 0.05 * H * np.cos(0.4 * t)])
        m[:, 2] += target - center
        cv2.warpAffine(img, m, (W, H), dst=frame, borderMode=cv2.BORDER_TRANSPARENT)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait():
        raise SystemExit(f"ffmpeg failed for {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--face", type=Path, help="photo with one face (default: download a sample)")
    p.add_argument("--out", type=Path, default=HERE / "output", help="output directory (default: %(default)s)")
    args = p.parse_args()

    videos = args.out / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    img = load_face(args.face, args.out)
    center, face_px = face_center_and_size(img)

    for name, created, W, H, dur, size, tilt, pos in CLIPS:
        path = videos / name
        print(f"Making {path.name}: {W}x{H}, {dur} s, recorded {created or 'per file name'}")
        make_video(img, center, face_px, path, created, W, H, dur, size, tilt, pos)
    # The last clip has no metadata date, so give it a misleading mtime to show the file name wins.
    os.utime(videos / CLIPS[-1][0], (1735689600, 1735689600))  # 2025-01-01

    plan = args.out / "plan.json"
    plan.unlink(missing_ok=True)  # always re-analyse in the demo
    output = args.out / "timelapse.mp4"
    babylapse([str(videos), "-o", str(output), "--plan", str(plan),
               "--timezone", "Europe/Amsterdam", "--size", "720x720", "--preset", "veryfast"])
    print(f"\nOpen {output}. Expected order and dates: 03/01/2025 (UTC 23:30 on 02/01 is "
          "already 03/01 in Amsterdam), 15/03/2025, 01/06/2025.")


if __name__ == "__main__":
    main()
