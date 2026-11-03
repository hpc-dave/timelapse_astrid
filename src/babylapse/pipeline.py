"""End-to-end: probe -> pick windows -> render aligned segments -> concatenate."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, tzinfo
from pathlib import Path

from .faces import FaceDetector
from .metadata import VideoInfo, find_videos, probe_video, with_extra_rotation
from .orient import guess_rotation
from .render import RenderOptions, concat, render_segment, speed_up
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
    rotate: int = 0  # clockwise correction on top of the rotation metadata (auto-detected, or edit by hand)
    skip: bool = False  # leave this video out (kept in the plan so a rescan does not add it again)


def fix_orientation(infos: list[VideoInfo], detector: FaceDetector, log=print) -> list[VideoInfo]:
    """Turn videos without rotation metadata upright when the face detector clearly prefers another orientation."""
    untagged = [v for v in infos if v.rotation == 0]
    if untagged:
        log(f"Checking orientation of {len(untagged)} video(s) without rotation metadata")
    out = []
    for info in infos:
        if info.rotation == 0:
            guess = guess_rotation(info, detector)
            if guess.rotate:
                scores = ", ".join(f"{a}\N{DEGREE SIGN}: {s:.2f}" for a, s in guess.scores.items())
                log(f"    {info.path.name}: no rotation tag but plays sideways; turning {guess.rotate}\N{DEGREE SIGN} "
                    f"clockwise (mean face score {scores})")
                info = with_extra_rotation(info, guess.rotate)
        out.append(info)
    return out


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
                              win.start, win.end, round(win.score, 4), round(win.coverage, 4),
                              info.extra_rotation))
    return plan


def save_plan(plan: list[PlanEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in plan], indent=2) + "\n")


def load_plan(path: Path) -> list[PlanEntry]:
    return [PlanEntry(**d) for d in json.loads(path.read_text())]


def probe_all(files: list[Path], local_tz: tzinfo, log=print) -> list[VideoInfo]:
    infos = []
    for f in files:
        try:
            infos.append(probe_video(f, local_tz))
        except Exception as exc:  # unreadable / not a video
            log(f"skipping {f}: {exc}")
    infos.sort(key=lambda v: v.recorded)
    return infos


def _same_file(path: str) -> Path:
    return Path(path).expanduser().resolve()


def update_plan(
    plan: list[PlanEntry],
    inputs: list[Path],
    detector: FaceDetector,
    local_tz: tzinfo,
    length: float,
    sample_fps: float,
    auto_rotate: bool,
    log=print,
) -> list[PlanEntry]:
    """Analyse the videos in ``inputs`` that are not in ``plan`` yet; return only the new entries."""
    known = {_same_file(e.file) for e in plan}
    new_files = [f for f in find_videos(inputs) if f.resolve() not in known]
    if not new_files:
        return []
    log(f"Found {len(new_files)} new video(s) not in the plan")
    infos = probe_all(new_files, local_tz, log)
    if auto_rotate:
        infos = fix_orientation(infos, detector, log)
    return make_plan(infos, detector, length, sample_fps, log)


def segment_key(entry: PlanEntry, opts: RenderOptions, detector_cfg: dict) -> str:
    """Hash of everything that changes the rendered segment of ``entry``.

    The source file's size and mtime are included so a replaced video is rendered again.
    ``speed`` is left out because it is applied to the merged video.
    """
    st = _same_file(entry.file).stat()
    render_opts = asdict(opts)
    render_opts.pop("speed")
    data = {
        "file": str(_same_file(entry.file)), "size": st.st_size, "mtime": st.st_mtime_ns,
        "start": entry.start, "end": entry.end, "recorded": entry.recorded, "rotate": entry.rotate,
        "opts": render_opts, "detector": detector_cfg,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]


def segment_dir(output: Path) -> Path:
    """Where rendered segments are kept between runs: ``<output stem>_segments/`` next to the output."""
    return output.with_name(f"{output.stem}_segments")


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
    auto_rotate: bool = True,
    rescan: bool = True,
    cache: bool = True,
    log=print,
) -> None:
    detector = FaceDetector(model, score_threshold)

    if plan_path is not None and plan_path.exists():
        plan = load_plan(plan_path)
        log(f"Using existing plan {plan_path} with {len(plan)} video(s) (delete it to re-analyse everything)")
        if rescan:
            added = update_plan(plan, inputs, detector, local_tz, length, sample_fps, auto_rotate, log)
            if added:
                plan += added
                plan.sort(key=lambda e: datetime.fromisoformat(e.recorded))
                save_plan(plan, plan_path)
                log(f"Added {len(added)} video(s) to {plan_path}")
            else:
                log("No new videos found")
    else:
        files = find_videos(inputs)
        if not files:
            raise SystemExit("no video files found")
        infos_list = probe_all(files, local_tz, log)
        if auto_rotate:
            infos_list = fix_orientation(infos_list, detector, log)
        plan = make_plan(infos_list, detector, length, sample_fps, log)
        if plan_path is not None:
            save_plan(plan, plan_path)
            log(f"Plan written to {plan_path}")

    plan.sort(key=lambda e: datetime.fromisoformat(e.recorded))
    if analyze_only:
        return

    entries = []
    for e in plan:
        if e.skip:
            continue
        if not _same_file(e.file).is_file():
            log(f"warning: {e.file} is in the plan but no longer exists; leaving it out")
            continue
        entries.append(e)
    if not entries:
        raise SystemExit("nothing to render")

    output.parent.mkdir(parents=True, exist_ok=True)
    detector_cfg = {"model": str(model) if model else None, "score_threshold": score_threshold}
    with tempfile.TemporaryDirectory(prefix=".babylapse_", dir=output.parent) as tmp:
        seg_dir = segment_dir(output) if cache else Path(tmp)
        seg_dir.mkdir(parents=True, exist_ok=True)
        segments, rendered = [], 0
        for i, e in enumerate(entries, 1):
            seg = seg_dir / f"seg_{segment_key(e, opts, detector_cfg)}.mp4"
            if seg.exists():
                segments.append(seg)
                continue
            log(f"[{i}/{len(entries)}] rendering {Path(e.file).name} {e.start:.1f}-{e.end:.1f} s")
            info = with_extra_rotation(probe_video(Path(e.file), local_tz), e.rotate)
            part = Path(tmp) / f"part_{i:04d}.mp4"  # moved into place only when complete
            cov = render_segment(info, e.start, e.end, datetime.fromisoformat(e.recorded), part, detector, opts)
            if cov is not None:
                log(f"    face tracked in {cov:.0%} of frames")
            part.replace(seg)
            segments.append(seg)
            rendered += 1
        if cache:
            log(f"Rendered {rendered} new segment(s), reused {len(segments) - rendered} from {seg_dir}")
            used = set(segments)
            for old in seg_dir.glob("seg_*.mp4"):
                if old not in used:  # from a video that was removed, re-timed or rendered with other options
                    old.unlink()

        log(f"Concatenating {len(segments)} segments -> {output}")
        if opts.speed == 1.0:
            concat(segments, output)
        else:
            merged = Path(tmp) / "merged.mp4"
            concat(segments, merged)
            log(f"Changing speed by {opts.speed:g}x")
            speed_up(merged, output, opts.speed, opts)
    log("Done.")
