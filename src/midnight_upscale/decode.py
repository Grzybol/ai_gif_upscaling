"""Decode animated GIFs to full composited RGBA frames.

Pillow 10.4+ applies GIF89a disposal and partial frame updates inside
``GifImagePlugin.load_end`` before a frame is displayed. This module seeks
each frame, copies that displayed canvas, and converts the copy to RGBA.

It does not treat a raw subframe tile as a full picture, and it does not
composite the frames a second time. Transparent pixels stay transparent:
nothing is flattened onto black, white, or any other background.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from midnight_upscale.utils import PipelineError


@dataclass
class CompositedFrame:
    index: int
    duration_ms: int
    image: Image.Image
    width: int
    height: int
    frame_count: int
    gif_loop_count: int | None


def iter_composited_frames(path: Path) -> Iterator[CompositedFrame]:
    """Yield every displayed RGBA frame in order, including its delay in milliseconds."""

    if not path.is_file():
        raise PipelineError(f"Input file not found: {path}")

    image = Image.open(path)
    try:
        if (image.format or "").upper() != "GIF":
            raise PipelineError(
                f"{path} is {image.format or 'not a GIF'}, expected an animated GIF. "
                "The source file was not modified."
            )
        frame_count = int(image.n_frames)
        if frame_count < 1:
            raise PipelineError(f"{path} contains no frames")
        loop = image.info["loop"] if "loop" in image.info else None
        if loop is not None:
            loop = int(loop)
        width, height = image.size

        for index in range(frame_count):
            image.seek(index)
            duration = int(image.info.get("duration") or 0)
            # Convert a copy so palette-alpha setup cannot mutate the decoder buffer.
            frame = image.copy().convert("RGBA")
            if frame.size != (width, height):
                raise PipelineError(
                    f"Composited frame {index} is {frame.size[0]}x{frame.size[1]}, "
                    f"expected the GIF canvas {width}x{height}. "
                    "Refusing to pad or crop the frame."
                )
            yield CompositedFrame(
                index=index,
                duration_ms=duration,
                image=frame,
                width=width,
                height=height,
                frame_count=frame_count,
                gif_loop_count=loop,
            )
    finally:
        image.close()
