"""Synthetic green-screen clips: a red square moving across a green background."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from midnight_upscale.utils import require_binary

GREEN = (0, 177, 64)


def green_screen_frame(
    index: int,
    *,
    size: tuple[int, int] = (64, 48),
    square: int = 12,
    step: int = 4,
    top: int = 16,
    background: tuple[int, int, int] = GREEN,
) -> np.ndarray:
    """RGB frame with a red square whose left edge is at ``2 + index * step``."""

    width, height = size
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = background
    left = 2 + index * step
    frame[top : top + square, left : left + square] = (220, 20, 20)
    return frame


def write_png_frames(directory: Path, frames: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(directory / f"{i:06d}.png")


def make_video(
    path: Path,
    frames: list[np.ndarray],
    *,
    fps: int = 10,
    codec_args: list[str] | None = None,
) -> Path:
    """Encode PNG frames to ``path`` with ffmpeg. The container follows the suffix."""

    staging = path.parent / f"{path.stem}_src"
    write_png_frames(staging, frames)
    args = codec_args
    if args is None:
        if path.suffix.lower() == ".gif":
            args = []
        else:
            args = ["-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "12"]
    command = [
        require_binary("ffmpeg"), "-v", "error", "-y",
        "-framerate", str(fps), "-i", str(staging / "%06d.png"), *args, str(path),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return path
