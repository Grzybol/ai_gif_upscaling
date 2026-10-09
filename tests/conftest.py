"""Tiny GIF fixtures. Frames differ so Pillow does not merge their delays."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def save_rgba_gif(
    path: Path,
    frames: list[Image.Image],
    durations: list[int],
    *,
    disposal: int | list[int] = 1,
    loop: int | None = 0,
) -> None:
    first, *rest = frames
    options: dict[str, object] = {
        "format": "GIF",
        "save_all": True,
        "append_images": rest,
        "duration": durations,
        "disposal": disposal,
        "optimize": True,
    }
    if loop is not None:
        options["loop"] = loop
    first.save(path, **options)


def solid_frame(size: tuple[int, int], color: tuple[int, int, int, int]) -> Image.Image:
    return Image.new("RGBA", size, color)


def character_frame(
    index: int,
    *,
    size: tuple[int, int] = (8, 8),
    mark: tuple[int, int] = (1, 1),
) -> Image.Image:
    """Transparent canvas, an opaque red body pixel, and a unique mark."""

    frame = Image.new("RGBA", size, (0, 0, 0, 0))
    frame.putpixel((0, 0), (255, 0, 0, 255))
    frame.putpixel(mark, (index * 20, 255, 0, 255))
    return frame
