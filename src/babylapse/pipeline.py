"""End-to-end: probe -> pick windows -> render aligned segments -> concatenate."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, tzinfo
from pathlib import Path

from .faces import FaceDetector
from .metadata import VideoInfo, find_videos, probe_video
from .render import RenderOptions, concat, render_segment
from .select import analyze, choose_window


@dataclass
class PlanEntry:
    file: str
    recorded: str  # ISO 8601
    recorded_source: str
    start: float
    end: float
    score: float | None = None
    coverage: float | None = None


def make_plan(
    infos: list[VideoInfo],
    detector: FaceDetector,
    length: float,
    sample_fps: float,
    log=print,
) -> list[PlanEntry]:
    plan = []
    for i, info in enumerate(infos, 1):
        log(f"[{i}/{len(infos)}] analysing {info.path.name} ({info.duration:.1f} s, recorded {info.recorded:%d/%m/%Y %H:%M} via {info.recorded_source})")
        samples, shape = analyze(info, detector, sample_fps)
        win = choose_window(samples, info.duration, shape, length)
        if info.duration < length:
            log(f"    note: video is shorter than {length:g} s; using all {info.duration:.1f} s")
            win.end = info.duration
        if win.coverage == 0:
            log("    warning: no face found; segment will be centre-cropped")
        log(f"    -> {win.start:.1f}-{win.end:.1f} s, face in {win.coverage:.0%} of samples, score {win.score:.2f}")
        plan.append(PlanEntry(str(info.path), info.recorded.isoformat(), info.recorded_source,
                              win.start, win.end, round(win.score, 4), round(win.coverage, 4)))
    return plan


def save_plan(plan: list[PlanEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in plan], indent=2) + "\n")


def load_plan(path: Path) -> list[PlanEntry]:
    return [PlanEntry(**d) for d in json.loads(path.read_text())]


def run(
    inputs: list[Path],
    output: Path,
    opts: RenderOptions,
    local_tz: tzinfo,
    length: float = 10.0,
    sample_fps: float = 2.0,
    plan_path: Path | None = None,
    analyze_only: bool = False,
    model: Path | None = None,
    score_threshold: float = 0.6,
    log=print,
) -> None:
    detector = FaceDetector(model, score_threshold)

    if plan_path is not None and plan_path.exists():
        log(f"Using existing plan {plan_path} (delete it to re-analyse)")
        plan = load_plan(plan_path)
        infos = {e.file: probe_video(Path(e.file), local_tz) for e in plan}
    else:
        files = find_videos(inputs)
        if not files:
            raise SystemExit("no video files found")
        infos_list = []
        for f in files:
            try:
                infos_list.append(probe_video(f, local_tz))
            except Exception as exc:  # unreadable / not a video
                log(f"skipping {f}: {exc}")
        infos_list.sort(key=lambda v: v.recorded)
        plan = make_plan(infos_list, detector, length, sample_fps, log)
        infos = {str(v.path): v for v in infos_list}
        if plan_path is not None:
            save_plan(plan, plan_path)
            log(f"Plan written to {plan_path}")

    plan.sort(key=lambda e: datetime.fromisoformat(e.recorded))
    if analyze_only:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".babylapse_", dir=output.parent) as tmp:
        segments = []
        for i, e in enumerate(plan, 1):
            info = infos[e.file]
            seg = Path(tmp) / f"seg_{i:04d}.mp4"
            log(f"[{i}/{len(plan)}] rendering {Path(e.file).name} {e.start:.1f}-{e.end:.1f} s")
            cov = render_segment(info, e.start, e.end, datetime.fromisoformat(e.recorded), seg, detector, opts)
            log(f"    face tracked in {cov:.0%} of frames")
            segments.append(seg)
        log(f"Concatenating {len(segments)} segments -> {output}")
        concat(segments, output)
    log("Done.")
