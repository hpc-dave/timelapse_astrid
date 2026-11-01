# babylapse

Turn a folder of baby videos into one chronological "growing up" video:

1. **Read the recording time** of every video (QuickTime `creationdate` → container `creation_time` → date in the file name → file mtime).
2. **Pick the best 10 s of each video.** It samples frames at 2 fps, detects the face (OpenCV YuNet), and scores each window on how often the face is visible, how confident the detection is, how steady it is and how large it is.
3. **Align the face.** For every frame of the chosen window it fits a 5-point similarity transform (eyes, nose, mouth corners → a canonical face template). The transform is smoothed over time, so the face sits in the same place, at the same size and the same angle, in every clip.
4. **Render and merge.** It writes the clips in recording order, with the date (`DD/MM/YYYY`) in the lower right, and joins them into one H.264/AAC MP4.

## Install

Needs Python ≥ 3.10 and `ffmpeg`/`ffprobe` on `PATH`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The first run downloads the 230 kB YuNet face model to `~/.cache/babylapse/`.

## Usage

```bash
babylapse /path/to/videos -o timelapse.mp4 --timezone Europe/Amsterdam --plan plan.json
```

Useful options (see `babylapse --help`):

| option | default | meaning |
|---|---|---|
| `--length` | 10 | seconds per video |
| `--size` | 1080x1080 | output resolution, e.g. `1920x1080` |
| `--face-scale` | 0.12 | distance between the eyes as a fraction of output height (bigger means more zoom) |
| `--eye-y` | 0.42 | vertical position of the eyes |
| `--align` | smooth | `smooth` follows the face every frame; `fixed` uses one transform per clip; `none` turns face alignment off and shows each whole video, upright, fitted into the output |
| `--smooth` | 0.4 | smoothing time constant (s); larger is calmer but lags fast movement |
| `--fill` | blur | fill areas outside the source with a blurred copy (`blur`) or `black` |
| `--speed` | 1 | playback speed of the final video (`2` = twice as fast, `0.5` = half speed); the sound keeps its pitch |
| `--no-audio` | | drop the sound |
| `--timezone` | system | zone used to turn UTC timestamps into the displayed date |

`examples/run.py` is a starting point with every option written out and commented. It reads all videos from one directory and writes `timelapse.mp4` and `plan.json` into an output folder:

```bash
.venv/bin/python examples/run.py /path/to/videos          # writes to ./output/
.venv/bin/python examples/run.py /path/to/videos my_out   # writes to my_out/
.venv/bin/python examples/run.py /path/to/videos --length 4 --speed 2   # 4 s per video, twice as fast
.venv/bin/python examples/run.py /path/to/videos --align none           # no face alignment
```

### Example without your own videos

```bash
.venv/bin/python examples/demo.py                # sample face: a public-domain NASA photo
.venv/bin/python examples/demo.py --face me.jpg  # or any photo with one face
```

This makes three synthetic videos in `examples/output/videos/`. Each has a moving, tilting and zooming face, a different size and orientation, and a different recording date: one from metadata just before midnight UTC, one from metadata, and one from the file name only. It then runs babylapse on them and writes `examples/output/timelapse.mp4` and `plan.json`.

### Overriding the automatic choices

With `--plan plan.json`, the chosen windows are saved after the analysis step. On later runs, if that file already exists, it is used and the analysis is skipped. Edit `start`/`end` to pick a different moment, or `recorded` to fix a wrong date. Delete an entry to drop a video. Use `--analyze-only` to only produce the plan.

## Notes

- Videos shorter than the segment length are used whole.
- If no face is found in a window, the clip is centre-cropped and a warning is printed.
- When several faces are visible, the largest one is chosen first and then tracked, so a parent's face won't steal the alignment.
- Phone rotation metadata is applied automatically by OpenCV.

## Tests

```bash
.venv/bin/pytest
```
