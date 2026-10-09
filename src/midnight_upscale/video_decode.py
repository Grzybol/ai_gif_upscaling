"""Plan which source frames to keep, then decode them to lossless RGBA PNG files.

The frame plan is explicit. Changing the output FPS always reports how many
frames were dropped or repeated. Frames are chosen by index, never by seeking,
so the count shown before processing is the count that gets decoded.
"""

from __future__ import annotations

import bisect
import io
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from midnight_upscale.progress import checkpoint, report
from midnight_upscale.utils import PipelineError, frame_path, require_binary, tail_text
from midnight_upscale.video_inspect import VideoInfo, decoder_args

_EPS = 1e-6
POLL_SECONDS = 0.15


@dataclass
class FramePlan:
    """Which source frame feeds each output frame, and how long each one shows."""

    source_indices: list[int]
    durations_ms: list[int]
    fps: float | None
    window_frames: int
    dropped: int
    duplicated: int

    @property
    def count(self) -> int:
        return len(self.source_indices)

    @property
    def total_ms(self) -> int:
        return sum(self.durations_ms)

    def describe(self) -> str:
        if self.fps is None:
            return f"{self.count} frames (source timing kept)"
        parts = [f"{self.count} frames at {self.fps:g} fps"]
        if self.dropped:
            parts.append(f"{self.dropped} source frames dropped")
        if self.duplicated:
            parts.append(f"{self.duplicated} frames repeated")
        return ", ".join(parts)


def plan_frames(
    info: VideoInfo,
    *,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    fps: float | None = None,
) -> FramePlan:
    """Choose output frames. ``fps=None`` keeps every source frame and its own duration."""

    span = info.duration_sec
    start = max(0.0, float(start_sec or 0.0))
    end = span if end_sec in (None, 0) or float(end_sec) > span else float(end_sec)
    if start >= end:
        raise PipelineError(f"Start time {start:g} s must be before end time {end:g} s")
    if fps is not None and fps <= 0:
        raise PipelineError(f"Output FPS must be above zero, got {fps}")

    times = info.frame_times
    window = [i for i, t in enumerate(times) if start - _EPS <= t < end - _EPS]
    if not window:
        raise PipelineError(f"No frames between {start:g} s and {end:g} s")

    if fps is None:
        return FramePlan(
            source_indices=window,
            durations_ms=[info.durations_ms[i] for i in window],
            fps=None,
            window_frames=len(window),
            dropped=0,
            duplicated=0,
        )

    count = max(1, math.floor((end - start) * fps + _EPS))
    indices: list[int] = []
    for k in range(count):
        when = start + k / fps
        pick = bisect.bisect_right(times, when + _EPS) - 1
        indices.append(max(pick, 0))
    # Whole-millisecond durations, each boundary rounded once.
    durations = [round((k + 1) * 1000.0 / fps) - round(k * 1000.0 / fps) for k in range(count)]
    used = set(indices)
    return FramePlan(
        source_indices=indices,
        durations_ms=durations,
        fps=float(fps),
        window_frames=len(window),
        dropped=len([i for i in window if i not in used]),
        duplicated=count - len(used),
    )


def select_filter(low: int, high: int) -> str:
    return f"select=between(n\\,{low}\\,{high})"


def decode_command(
    ffmpeg: str, info: VideoInfo, low: int, high: int, destination: Path
) -> list[str]:
    return [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-y",
        *decoder_args(info),
        "-i",
        str(info.path),
        "-map",
        "0:v:0",
        "-vf",
        select_filter(low, high),
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(high - low + 1),
        "-pix_fmt",
        "rgba",
        "-compression_level",
        "1",
        "-start_number",
        "0",
        str(destination / "%06d.png"),
    ]


def decode_plan(
    info: VideoInfo, plan: FramePlan, destination: Path, *, scratch: Path
) -> list[Path]:
    """Write ``destination/000000.png`` ... for every planned output frame.

    The needed source range is decoded once into ``scratch``. Frames the plan does not
    use are discarded, and a frame the plan repeats is copied.
    """

    ffmpeg = require_binary("ffmpeg")
    low, high = min(plan.source_indices), max(plan.source_indices)
    expected = high - low + 1
    for folder in (destination, scratch):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    command = decode_command(ffmpeg, info, low, high, scratch)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=errors, stdin=subprocess.DEVNULL
        )
        try:
            while process.poll() is None:
                checkpoint()
                done = sum(1 for _ in scratch.glob("*.png"))
                report(
                    "Decode video",
                    frames_done=min(done, expected),
                    frames_total=expected,
                    frames_kind="completed",
                    message=f"Decoding frame {done} / {expected}",
                )
                time.sleep(POLL_SECONDS)
        except BaseException:
            process.kill()
            process.wait()
            raise
        errors.seek(0)
        detail = errors.read().decode("utf-8", errors="replace")
    if process.returncode != 0:
        raise PipelineError(f"ffmpeg could not decode {info.filename}.\n{tail_text(detail)}")

    decoded = sorted(scratch.glob("*.png"))
    if len(decoded) < expected:
        raise PipelineError(
            f"ffmpeg decoded {len(decoded)} of {expected} frames from {info.filename}. "
            "The file may be truncated."
        )

    outputs: list[Path] = []
    uses = Counter(plan.source_indices)
    for out_index, source_index in enumerate(plan.source_indices):
        checkpoint()
        source = decoded[source_index - low]
        target = frame_path(destination, out_index)
        uses[source_index] -= 1
        if uses[source_index] > 0:
            shutil.copyfile(source, target)
        else:
            os.replace(source, target)
        outputs.append(target)
    report(
        "Decode video",
        frames_done=expected,
        frames_total=expected,
        frames_kind="completed",
        message=f"Decoded {plan.count} frames",
    )
    shutil.rmtree(scratch, ignore_errors=True)
    return outputs


def decode_single_frame(info: VideoInfo, index: int) -> Image.Image:
    """Decode one source frame (by index) to an RGBA image."""

    if not 0 <= index < info.frame_count:
        raise PipelineError(f"Frame {index} is outside 0..{info.frame_count - 1}")
    command = [
        require_binary("ffmpeg"),
        "-v",
        "error",
        "-nostdin",
        *decoder_args(info),
        "-i",
        str(info.path),
        "-map",
        "0:v:0",
        "-vf",
        f"select=eq(n\\,{index})",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgba",
        "-f",
        "image2pipe",
        "-c:v",
        "png",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL, check=False)
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace")
        raise PipelineError(f"ffmpeg could not read frame {index}.\n{tail_text(detail)}")
    image = Image.open(io.BytesIO(completed.stdout))
    image.load()
    return image.convert("RGBA")


def sample_has_alpha(info: VideoInfo) -> bool:
    """True when the first, middle, or last frame has a pixel that is not fully opaque."""

    last = info.frame_count - 1
    for index in sorted({0, last // 2, last}):
        try:
            frame = decode_single_frame(info, index)
        except PipelineError:
            continue
        if np.any(np.asarray(frame)[:, :, 3] < 255):
            return True
    return False
