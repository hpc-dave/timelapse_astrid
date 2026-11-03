"""Example: run babylapse on one directory of videos, with every option given as a CLI argument.

    .venv/bin/python examples/run.py /path/to/videos              # writes to ./output/
    .venv/bin/python examples/run.py /path/to/videos my_output    # writes to my_output/
    .venv/bin/python examples/run.py /path/to/videos --length 4 --speed 2   # 4 s per video, twice as fast
    .venv/bin/python examples/run.py /path/to/videos --align none # no face alignment, whole videos upright

The options below are passed to babylapse exactly as on the command line, so this is
the same as running `babylapse /path/to/videos --output output/timelapse.mp4 ...`.
--length, --speed and --align can be set when calling this script; change the other
values below to taste. `babylapse --help` lists all options.

Note: plan.json stores the chosen segments, so delete it (or use another output
folder) after changing --length; --speed and --align apply without re-analysis.
Running again on the same output folder only analyses and renders videos that are
new in video_dir. The other clips are reused from output/timelapse_segments/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from babylapse.cli import main as babylapse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video_dir", type=Path, help="directory with the videos")
    p.add_argument("output_dir", type=Path, nargs="?", default=Path("output"),
                   help="directory for timelapse.mp4 and plan.json (default: %(default)s)")
    p.add_argument("--length", type=float, default=10.0,
                   help="seconds taken from each video; smaller gives a shorter result (default: %(default)s)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed of the result, e.g. 2 for twice as fast (default: %(default)s)")
    p.add_argument("--align", choices=["smooth", "fixed", "none"], default="smooth",
                   help="face alignment; none shows each whole video upright (default: %(default)s)")
    args = p.parse_args()

    if not args.video_dir.is_dir():
        raise SystemExit(f"not a directory: {args.video_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "timelapse.mp4"

    babylapse([
        str(args.video_dir),
        "--output", str(output),                       # final video
        "--plan", str(args.output_dir / "plan.json"),  # chosen windows; reused next run (delete to re-analyse)

        # What is taken from each video
        "--length", str(args.length),  # seconds per video
        "--sample-fps", "2",         # frames per second analysed when choosing the window
        "--min-confidence", "0.6",   # face detector score threshold

        # Output format
        "--size", "1080x1080",       # WIDTHxHEIGHT
        "--fps", "30",               # frame rate
        "--speed", str(args.speed),  # 2 = twice as fast, 0.5 = half speed (sound keeps its pitch)
        "--crf", "20",               # x264 quality, lower is better
        "--preset", "medium",        # x264 preset: veryfast ... slow

        # Face alignment (--face-scale, --eye-y and --smooth have no effect with --align none)
        "--align", args.align,       # smooth: follow the face; fixed: one transform per clip; none: off
        "--face-scale", "0.12",      # distance between the eyes / output height (bigger = more zoom)
        "--eye-y", "0.42",           # vertical eye position / output height
        "--smooth", "0.4",           # smoothing time constant in seconds
        "--fill", "blur",            # fill outside the source frame: blur or black

        # Date label
        "--timezone", "Europe/Amsterdam",  # zone for the displayed date
        "--date-format", "%d/%m/%Y",       # strftime format
        "--font-scale", "1.0",             # label size multiplier

        # Optional extras:
        # "--no-audio",              # drop the sound
        # "--analyze-only",          # only write plan.json, so you can edit it before rendering
    ])
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
