from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from babylapse.align import Layout, build_track, fit_similarity, matrix_to_params, params_to_matrix
from babylapse.faces import Face
from babylapse.metadata import parse_filename_datetime, recording_time
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
