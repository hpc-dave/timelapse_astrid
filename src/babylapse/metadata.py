"""Video metadata: recording time, duration, fps, audio presence (via ffprobe)."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp", ".mts", ".m2ts", ".webm"}


@dataclass
class VideoInfo:
    path: Path
    recorded: datetime  # timezone-aware
    recorded_source: str  # where the timestamp came from
    duration: float  # seconds
    fps: float
    width: int
    height: int
    has_audio: bool


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def _parse_iso(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    # Normalise "Z" and "+0100" style offsets to something fromisoformat accepts.
    v = value.replace("Z", "+00:00")
    v = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", v)
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.year < 1990:  # bogus "1904/1970" epoch dates written by some devices
        return None
    return dt


# Patterns such as VID_20240131_142233.mp4, PXL_20240131_142233123.mp4,
# 2024-01-31 14.22.33.mov, IMG-20240131-WA0001.mp4, signal-2024-01-31-142233.mp4
_FILENAME_PATTERNS = [
    re.compile(r"(?P<y>20\d{2})[-_.]?(?P<m>\d{2})[-_.]?(?P<d>\d{2})[ _T-]?(?P<H>\d{2})[-_.:]?(?P<M>\d{2})[-_.:]?(?P<S>\d{2})"),
    re.compile(r"(?P<y>20\d{2})[-_.]?(?P<m>\d{2})[-_.]?(?P<d>\d{2})"),
]


def parse_filename_datetime(name: str) -> datetime | None:
    for pat in _FILENAME_PATTERNS:
        for m in pat.finditer(name):
            g = m.groupdict()
            try:
                return datetime(
                    int(g["y"]), int(g["m"]), int(g["d"]),
                    int(g.get("H") or 0), int(g.get("M") or 0), int(g.get("S") or 0),
                )
            except ValueError:
                continue
    return None


def recording_time(path: Path, probe: dict, local_tz: tzinfo) -> tuple[datetime, str]:
    """Best-effort recording timestamp, returned as an aware datetime in ``local_tz``.

    Priority: QuickTime creationdate (has local offset) > container/stream
    creation_time (UTC) > date in filename (assumed local) > file mtime.
    """
    fmt_tags = {k.lower(): v for k, v in probe.get("format", {}).get("tags", {}).items()}
    stream_tags = [
        {k.lower(): v for k, v in s.get("tags", {}).items()} for s in probe.get("streams", [])
    ]

    candidates: list[tuple[str, str]] = []
    if "com.apple.quicktime.creationdate" in fmt_tags:
        candidates.append(("quicktime:creationdate", fmt_tags["com.apple.quicktime.creationdate"]))
    if "creation_time" in fmt_tags:
        candidates.append(("format:creation_time", fmt_tags["creation_time"]))
    for i, tags in enumerate(stream_tags):
        if "creation_time" in tags:
            candidates.append((f"stream{i}:creation_time", tags["creation_time"]))

    for source, value in candidates:
        dt = _parse_iso(value)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(local_tz), source

    dt = parse_filename_datetime(path.name)
    if dt is not None:
        return dt.replace(tzinfo=local_tz), "filename"

    return datetime.fromtimestamp(path.stat().st_mtime, tz=local_tz), "file-mtime"


def _fps(stream: dict) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = stream.get(key, "0/0").partition("/")
        try:
            val = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            continue
        if 1 <= val <= 300:
            return val
    return 30.0


def probe_video(path: Path, local_tz: tzinfo) -> VideoInfo:
    probe = ffprobe(path)
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path}: no video stream")
    duration = float(probe.get("format", {}).get("duration") or video.get("duration") or 0.0)
    recorded, source = recording_time(path, probe, local_tz)
    return VideoInfo(
        path=path,
        recorded=recorded,
        recorded_source=source,
        duration=duration,
        fps=_fps(video),
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def find_videos(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files
