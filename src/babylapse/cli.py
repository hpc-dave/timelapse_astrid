"""Command-line interface: ``babylapse VIDEOS... -o out.mp4``."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .align import Layout
from .pipeline import run
from .render import RenderOptions


def _size(s: str) -> tuple[int, int]:
    w, _, h = s.lower().partition("x")
    w_i, h_i = int(w), int(h)
    if w_i % 2 or h_i % 2:
        raise argparse.ArgumentTypeError("width and height must be even (yuv420p)")
    return w_i, h_i


def _speed(s: str) -> float:
    v = float(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("speed must be positive")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="babylapse",
        description="Take the best 10 s of each baby video, align the face across clips, "
                    "and merge them chronologically with the recording date in the lower right.",
    )
    p.add_argument("inputs", nargs="+", type=Path, help="video files and/or directories (searched recursively)")
    p.add_argument("-o", "--output", type=Path, default=Path("timelapse.mp4"), help="output file (default: %(default)s)")
    p.add_argument("--length", type=float, default=10.0, help="seconds taken from each video (default: %(default)s)")
    p.add_argument("--size", type=_size, default=(1080, 1080), help="output WIDTHxHEIGHT (default: 1080x1080)")
    p.add_argument("--fps", type=float, default=30.0, help="output frame rate (default: %(default)s)")
    p.add_argument("--face-scale", type=float, default=0.12,
                   help="inter-eye distance as a fraction of output height (default: %(default)s)")
    p.add_argument("--eye-y", type=float, default=0.42,
                   help="vertical eye position as a fraction of output height (default: %(default)s)")
    p.add_argument("--align", choices=["smooth", "fixed", "none"], default="smooth",
                   help="smooth: follow the face every frame (low-pass filtered); "
                        "fixed: one transform per clip; none: no face alignment, the whole upright frame "
                        "is fitted into the output (default: %(default)s)")
    p.add_argument("--smooth", type=float, default=0.4, help="smoothing time constant in seconds (default: %(default)s)")
    p.add_argument("--fill", choices=["blur", "black"], default="blur",
                   help="how to fill areas outside the source frame (default: %(default)s)")
    p.add_argument("--no-audio", action="store_true", help="drop audio")
    p.add_argument("--speed", type=_speed, default=1.0,
                   help="playback speed of the final video, e.g. 2 for twice as fast or 0.5 for half speed; "
                        "audio keeps its pitch (default: %(default)s)")
    p.add_argument("--crf", type=int, default=20, help="x264 quality, lower is better (default: %(default)s)")
    p.add_argument("--preset", default="medium", help="x264 preset (default: %(default)s)")
    p.add_argument("--font-scale", type=float, default=1.0, help="date label size multiplier")
    p.add_argument("--date-format", default="%d/%m/%Y", help="strftime format of the label (default: DD/MM/YYYY)")
    p.add_argument("--timezone", default=None,
                   help="IANA zone for dates, e.g. Europe/Amsterdam (default: system local time)")
    p.add_argument("--sample-fps", type=float, default=2.0,
                   help="frames per second analysed when choosing the window (default: %(default)s)")
    p.add_argument("--min-confidence", type=float, default=0.6, help="face detector score threshold (default: %(default)s)")
    p.add_argument("--plan", type=Path, default=None,
                   help="JSON file with the chosen windows. Written after analysis; if it already exists it is "
                        "used instead of analysing (edit start/end/recorded by hand to override).")
    p.add_argument("--analyze-only", action="store_true", help="only choose the windows (use with --plan)")
    p.add_argument("--model", type=Path, default=None, help="path to YuNet .onnx (downloaded automatically if omitted)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tz = ZoneInfo(args.timezone) if args.timezone else datetime.now().astimezone().tzinfo
    w, h = args.size
    opts = RenderOptions(
        layout=Layout(w, h, eye_y=args.eye_y, face_scale=args.face_scale),
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        align_mode=args.align,
        smooth_seconds=args.smooth,
        fill=args.fill,
        audio=not args.no_audio,
        date_format=args.date_format,
        font_scale=args.font_scale,
        speed=args.speed,
    )
    run(
        args.inputs,
        args.output,
        opts,
        tz,
        length=args.length,
        sample_fps=args.sample_fps,
        plan_path=args.plan,
        analyze_only=args.analyze_only,
        model=args.model,
        score_threshold=args.min_confidence,
    )


if __name__ == "__main__":
    main()
