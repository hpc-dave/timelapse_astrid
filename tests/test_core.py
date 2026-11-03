from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import shutil
import subprocess

import numpy as np
import pytest

from babylapse.align import Layout, build_track, fit_similarity, matrix_to_params, params_to_matrix
from babylapse.faces import Face
from babylapse.render import atempo_chain
from babylapse.metadata import parse_filename_datetime, probe_video, recording_time, rotation
from babylapse.select import Sample, choose_window

AMS = ZoneInfo("Europe/Amsterdam")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("VID_20240131_142233.mp4", datetime(2024, 1, 31, 14, 22, 33)),
        ("PXL_20240131_142233123.mp4", datetime(2024, 1, 31, 14, 22, 33)),
        ("2024-01-31 14.22.33.mov", datetime(2024, 1, 31, 14, 22, 33)),
        ("IMG-20240131-WA0001.mp4", datetime(2024, 1, 31)),
        ("holiday.mp4", None),
    ],
)
def test_filename_dates(name, expected):
    assert parse_filename_datetime(name) == expected


def test_recording_time_priority(tmp_path):
    f = tmp_path / "VID_20200101_000000.mp4"
    f.write_bytes(b"")
    probe = {
        "format": {"tags": {"creation_time": "2024-03-31T23:30:00.000000Z",
                            "com.apple.quicktime.creationdate": "2024-04-01T01:30:00+0200"}},
        "streams": [],
    }
    dt, src = recording_time(f, probe, AMS)
    assert src == "quicktime:creationdate" and dt.strftime("%d/%m/%Y") == "01/04/2024"

    del probe["format"]["tags"]["com.apple.quicktime.creationdate"]
    dt, src = recording_time(f, probe, AMS)  # UTC 23:30 on 31/03 is already 01/04 in Amsterdam
    assert src == "format:creation_time" and dt.strftime("%d/%m/%Y") == "01/04/2024"

    probe["format"]["tags"]["creation_time"] = "1970-01-01T00:00:00Z"  # bogus -> fall back to filename
    dt, src = recording_time(f, probe, AMS)
    assert src == "filename" and dt.year == 2020


def test_similarity_roundtrip():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (5, 2))
    ang, s, t = 0.3, 1.7, np.array([12.0, -5.0])
    r = s * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = src @ r.T + t
    m = fit_similarity(src, dst)
    np.testing.assert_allclose(m[:, :2], r, atol=1e-9)
    np.testing.assert_allclose(m[:, 2], t, atol=1e-9)
    anchor = np.array([50.0, 40.0])
    np.testing.assert_allclose(params_to_matrix(matrix_to_params(m, anchor), anchor), m, atol=1e-9)


def test_track_maps_landmarks_to_template_and_interpolates():
    layout = Layout(400, 400)
    target = layout.target_points()
    # Source landmarks: the template shrunk, rotated and moved.
    ang = -0.2
    r = 0.5 * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    lms = target @ r.T + np.array([30.0, 60.0])
    times = np.arange(10) / 10
    track = build_track(times, [lms, None, None, lms] + [lms] * 6, layout, (300, 300), smooth_seconds=0)
    for p in track:
        m = params_to_matrix(p, layout.anchor)
        mapped = lms @ m[:, :2].T + m[:, 2]
        np.testing.assert_allclose(mapped, target, atol=1e-6)


def test_track_without_faces_covers_frame():
    layout = Layout(200, 100)
    track = build_track(np.arange(3.0), [None] * 3, layout, (100, 100))
    m = params_to_matrix(track[0], layout.anchor)
    # Frame centre maps to output centre and the frame covers the output width.
    np.testing.assert_allclose(m @ [50, 50, 1], [100, 50])
    assert m[0, 0] == pytest.approx(2.0)


def _face(x, y, d=40.0, score=0.9):
    lm = np.array([[x - d / 2, y], [x + d / 2, y], [x, y + 20], [x - 15, y + 35], [x + 15, y + 35]])
    return Face(box=np.array([x - 50, y - 50, 100, 100]), landmarks=lm, score=score)


def test_choose_window_prefers_steady_visible_face():
    rng = np.random.default_rng(1)
    samples = []
    for i in range(60):  # 30 s at 2 fps
        t = i / 2
        if t < 10:
            samples.append(Sample(t, None))  # no face
        elif t < 20:
            samples.append(Sample(t, _face(*rng.uniform(50, 500, 2))))  # jumping around
        else:
            samples.append(Sample(t, _face(300, 300)))  # steady
    win = choose_window(samples, 30.0, (720, 1280), 10.0)
    assert win.start == pytest.approx(20.0)
    assert win.coverage == 1.0


def test_choose_window_short_video():
    win = choose_window([Sample(0.0, None)], 4.0, (100, 100), 10.0)
    assert (win.start, win.end) == (0.0, 4.0)


@pytest.mark.parametrize("factor, expected", [
    (1.5, "atempo=1.5"),
    (8, "atempo=2,atempo=2,atempo=2"),
    (5, "atempo=2,atempo=2,atempo=1.25"),
    (0.25, "atempo=0.5,atempo=0.5"),
])
def test_atempo_chain(factor, expected):
    assert atempo_chain(factor) == expected
    with pytest.raises(ValueError):
        atempo_chain(0)


@pytest.mark.parametrize("stream, expected", [
    ({}, 0),
    ({"side_data_list": [{"rotation": -90}]}, 90),
    ({"side_data_list": [{"rotation": 90}]}, 270),
    ({"side_data_list": [{"rotation": 180}]}, 180),
    ({"tags": {"rotate": "90"}}, 90),
])
def test_rotation(stream, expected):
    assert rotation(stream) == expected


def test_align_none_fits_whole_frame():
    layout = Layout(1080, 1080)
    params = build_track(np.array([0.0, 1.0]), [None, None], layout, (1920, 1080), "none")
    m = params_to_matrix(params[0], layout.anchor)
    corners = np.array([[0, 0, 1], [1080, 1920, 1]], float) @ m.T
    # Portrait 1080x1920 into a square: full height, centred horizontally, not rotated.
    np.testing.assert_allclose(corners, [[236.25, 0], [843.75, 1080]], atol=1e-6)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_rotated_video_is_read_upright(tmp_path):
    from babylapse.video import iter_frames

    plain, rotated = tmp_path / "plain.mp4", tmp_path / "rotated.mp4"
    # Landscape 320x180 frame with a white box in the top-left corner.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=black:s=320x180:d=1,drawbox=x=0:y=0:w=80:h=40:color=white:t=fill",
                    "-c:v", "libx264", str(plain)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(plain), "-c", "copy",
                    "-metadata:s:v:0", "rotate=90", str(rotated)], check=True)
    # Reference: the first frame as ffmpeg itself displays it (autorotated).
    ref = subprocess.run(["ffmpeg", "-v", "error", "-i", str(rotated), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], check=True, capture_output=True).stdout

    info = probe_video(rotated, AMS)
    assert (info.width, info.height) == (180, 320)
    _, frame = next(iter_frames(rotated, info.fps))
    assert frame.shape[:2] == (320, 180)
    ref = np.frombuffer(ref, np.uint8).reshape(320, 180)
    assert np.abs(frame[..., 1].astype(int) - ref).mean() < 5


@pytest.mark.parametrize("scores, expected", [
    ({0: 0.92, 90: 0.30, 180: 0.80, 270: 0.60}, 0),   # upright video: keep
    ({0: 0.56, 90: 0.92, 180: 0.00, 270: 0.75}, 90),  # untagged sideways video: turn
    ({0: 0.80, 90: 0.59, 180: 0.49, 270: 0.85}, 0),   # slightly better elsewhere: not enough margin
    ({0: 0.10, 90: 0.50, 180: 0.00, 270: 0.00}, 0),   # no confident face anywhere: keep
    ({0: 0.00, 90: 0.00, 180: 0.90, 270: 0.00}, 180),
])
def test_choose_rotation(scores, expected):
    from babylapse.orient import choose_rotation

    assert choose_rotation(scores) == expected


def test_with_extra_rotation_swaps_size():
    from babylapse.metadata import VideoInfo, with_extra_rotation

    info = VideoInfo(Path("x.mp4"), datetime(2026, 1, 1, tzinfo=AMS), "test", 1.0, 30.0, 1920, 1080, False)
    turned = with_extra_rotation(info, 90)
    assert (turned.width, turned.height, turned.extra_rotation) == (1080, 1920, 90)
    assert (with_extra_rotation(turned, 180).width, with_extra_rotation(turned, 0).width) == (1920, 1920)
    with pytest.raises(ValueError):
        with_extra_rotation(info, 45)


def test_old_plan_without_rotate_loads(tmp_path):
    from babylapse.pipeline import load_plan

    f = tmp_path / "plan.json"
    f.write_text('[{"file": "a.mp4", "recorded": "2026-01-01T00:00:00+01:00", "recorded_source": "x", '
                 '"start": 0.0, "end": 5.0}]')
    assert load_plan(f)[0].rotate == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_iter_frames_extra_rotation(tmp_path):
    from babylapse.video import iter_frames

    src = tmp_path / "plain.mp4"
    # Landscape 320x180 frame with a white box in the top-left corner.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=black:s=320x180:d=1,drawbox=x=0:y=0:w=80:h=40:color=white:t=fill",
                    "-c:v", "libx264", str(src)], check=True)
    _, frame = next(iter_frames(src, 25.0, rotate=90))
    assert frame.shape[:2] == (320, 180)
    # Turned clockwise, the top-left box ends up in the top-right corner (40 wide, 80 tall).
    assert frame[:80, -40:].mean() > 200 and frame[:80, :40].mean() < 50


def test_update_plan_only_analyses_new_videos(tmp_path, monkeypatch):
    from babylapse import pipeline
    from babylapse.pipeline import PlanEntry, update_plan

    for name in ("a.mp4", "b.mp4", "c.mov"):
        (tmp_path / name).write_bytes(b"x")
    plan = [PlanEntry(str(tmp_path / "a.mp4"), "2026-01-01T00:00:00+01:00", "x", 0, 5),
            PlanEntry(str(tmp_path / "b.mp4"), "2026-01-02T00:00:00+01:00", "x", 0, 5, skip=True)]
    seen = []
    monkeypatch.setattr(pipeline, "probe_all", lambda files, tz, log: seen.extend(files) or [])
    monkeypatch.setattr(pipeline, "make_plan", lambda infos, *a, **k: ["new"])
    assert update_plan(plan, [tmp_path], None, timezone.utc, 10, 2, False, log=lambda *_: None) == ["new"]
    assert seen == [tmp_path / "c.mov"]  # a is known, b is known (skipped)


def test_segment_key_tracks_entry_options_and_source(tmp_path):
    import os
    from dataclasses import replace
    from babylapse.pipeline import PlanEntry, segment_key
    from babylapse.render import RenderOptions

    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    e = PlanEntry(str(src), "2026-01-01T00:00:00+01:00", "x", 0.0, 5.0)
    opts = RenderOptions(Layout(720, 720))
    det = {"model": None, "score_threshold": 0.6}
    k = segment_key(e, opts, det)
    assert k == segment_key(e, replace(opts, speed=2.0), det)  # speed is applied after joining
    assert k != segment_key(replace(e, start=1.0), opts, det)
    assert k != segment_key(e, replace(opts, align_mode="none"), det)
    os.utime(src, ns=(0, 0))
    assert k != segment_key(e, opts, det)
