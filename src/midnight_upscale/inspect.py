"""Read GIF timing, size, loop, and transparency without writing frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from midnight_upscale.decode import iter_composited_frames
from midnight_upscale.models import GifInspection
from midnight_upscale.utils import PipelineError


def inspect_gif(path: Path) -> GifInspection:
    durations: list[int] = []
    width = 0
    height = 0
    loop: int | None = None
    has_transparency = False
    has_semitransparency = False

    for frame in iter_composited_frames(path):
        if not durations:
            width = frame.width
            height = frame.height
            loop = frame.gif_loop_count
        durations.append(frame.duration_ms)
        if not (has_transparency and has_semitransparency):
            alpha = np.asarray(frame.image)[:, :, 3]
            if np.any(alpha < 255):
                has_transparency = True
            if np.any((alpha > 0) & (alpha < 255)):
                has_semitransparency = True
        frame.image.close()

    if not durations:
        raise PipelineError(f"{path} contains no frames")

    total = sum(durations)
    return GifInspection(
        source=str(path),
        width=width,
        height=height,
        frame_count=len(durations),
        frame_durations_ms=durations,
        total_duration_ms=total,
        gif_loop_count=loop,
        durations_constant=len(set(durations)) <= 1,
        estimated_fps=None if total <= 0 else len(durations) / (total / 1000.0),
        has_transparency=has_transparency,
        has_semitransparency=has_semitransparency,
    )
