"""Whole-clip geometry: one crop box for every frame, padding, and resizing.

Cropping each frame on its own would make the character jump, so the box is the
union of the visible area across all frames and is applied identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from midnight_upscale.utils import PipelineError

# Alpha below this does not count as content (keeps matting noise from growing the box).
VISIBLE_ALPHA = 8
# Alpha at or above this counts as solid foreground when borrowing edge colors.
SOLID_ALPHA = 250

Box = tuple[int, int, int, int]  # left, top, right, bottom (right and bottom exclusive)


def alpha_bbox(rgba: np.ndarray, threshold: int = VISIBLE_ALPHA) -> Box | None:
    """Bounding box of pixels with alpha >= threshold, or ``None`` when there are none."""

    visible = rgba[..., 3] >= threshold
    rows = np.flatnonzero(visible.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(visible.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def union_box(boxes: list[Box | None]) -> Box | None:
    found = [box for box in boxes if box is not None]
    if not found:
        return None
    return (
        min(b[0] for b in found),
        min(b[1] for b in found),
        max(b[2] for b in found),
        max(b[3] for b in found),
    )


@dataclass(frozen=True)
class CropPlan:
    """Copy ``source_box`` from each frame to ``offset`` on a transparent canvas."""

    source_box: Box
    offset: tuple[int, int]
    canvas: tuple[int, int]

    @property
    def width(self) -> int:
        return self.canvas[0]

    @property
    def height(self) -> int:
        return self.canvas[1]


def plan_crop(box: Box, frame_size: tuple[int, int], padding: int, center: bool) -> CropPlan:
    """Plan one crop for the whole clip.

    ``center`` puts the content box in the middle of a canvas ``padding`` larger on every
    side, even where that reaches past the source frame. Without it, the padded box is
    simply clamped to the source frame.
    """

    padding = max(0, int(padding))
    left, top, right, bottom = box
    if center:
        return CropPlan(
            source_box=box,
            offset=(padding, padding),
            canvas=(right - left + 2 * padding, bottom - top + 2 * padding),
        )
    width, height = frame_size
    clamped = (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )
    return CropPlan(
        source_box=clamped,
        offset=(0, 0),
        canvas=(clamped[2] - clamped[0], clamped[3] - clamped[1]),
    )


def apply_crop(rgba: np.ndarray, plan: CropPlan) -> np.ndarray:
    left, top, right, bottom = plan.source_box
    canvas = np.zeros((plan.height, plan.width, 4), dtype=np.uint8)
    dx, dy = plan.offset
    canvas[dy : dy + (bottom - top), dx : dx + (right - left)] = rgba[top:bottom, left:right]
    return canvas


@dataclass(frozen=True)
class ResizeSettings:
    mode: str = "source"  # source | scale | custom
    scale: float = 1.0
    width: int = 0  # custom; 0 means "work it out from the other side"
    height: int = 0
    keep_aspect: bool = True

    @property
    def active(self) -> bool:
        return self.mode != "source" and not (self.mode == "scale" and self.scale == 1.0)


def target_size(width: int, height: int, settings: ResizeSettings) -> tuple[int, int]:
    if settings.mode == "source":
        return width, height
    if settings.mode == "scale":
        if settings.scale <= 0:
            raise PipelineError(f"Scale must be above zero, got {settings.scale}")
        return max(1, round(width * settings.scale)), max(1, round(height * settings.scale))
    if settings.mode != "custom":
        raise PipelineError(f"Unknown resize mode {settings.mode!r}")
    target_w, target_h = int(settings.width or 0), int(settings.height or 0)
    if target_w < 0 or target_h < 0:
        raise PipelineError("Custom width and height must not be negative")
    if target_w == 0 and target_h == 0:
        return width, height
    if not settings.keep_aspect:
        return target_w or width, target_h or height
    if target_w and target_h:
        ratio = min(target_w / width, target_h / height)
        return max(1, round(width * ratio)), max(1, round(height * ratio))
    if target_w:
        return target_w, max(1, round(height * target_w / width))
    return max(1, round(width * target_h / height)), target_h


def resize_rgba(rgba: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize RGBA with premultiplied alpha, so transparent pixels cannot tint the edges."""

    height, width = rgba.shape[:2]
    if (width, height) == size:
        return rgba
    image = Image.fromarray(rgba).convert("RGBa")
    resized = image.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
    return np.asarray(resized, dtype=np.uint8).copy()


def fill_transparent_rgb(
    rgba: np.ndarray, radius: float = 3.0, solid: int = SOLID_ALPHA
) -> np.ndarray:
    """Replace the color of every non-solid pixel with color from nearby solid pixels.

    Alpha is untouched. A segmentation mask keeps the original RGB, so a partly
    transparent edge pixel still carries the old (white) background: a light halo
    on a dark page. Taking its color from the solid foreground next to it removes
    the halo. It also stops chroma subsampling and bilinear filtering from pulling
    background color into the edge.
    """

    alpha = rgba[..., 3]
    loose = alpha < solid
    if not loose.any():
        return rgba
    core = (alpha >= solid).astype(np.uint8) * 255
    if not core.any():
        return rgba
    color = Image.fromarray(np.ascontiguousarray(rgba[..., :3] * (core[..., None] > 0)))
    weight = Image.fromarray(core)
    filled = np.zeros(rgba.shape[:2] + (3,), dtype=np.float32)
    have = np.zeros(rgba.shape[:2], dtype=bool)
    # Near first; wider blurs only reach pixels the narrow one could not.
    for scale in (1.0, 3.0, 9.0):
        blur = ImageFilter.GaussianBlur(radius * scale)
        c = np.asarray(color.filter(blur), dtype=np.float32)
        w = np.asarray(weight.filter(blur), dtype=np.float32) / 255.0
        usable = loose & ~have & (w > 0.02)
        filled[usable] = np.clip(c[usable] / w[usable][:, None], 0, 255)
        have |= usable
    out = rgba.copy()
    out[..., :3][loose] = 0
    out[..., :3][have] = (filled[have] + 0.5).astype(np.uint8)
    return out
