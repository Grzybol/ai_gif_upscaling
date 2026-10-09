"""Split, clean, and deterministically resize alpha. Alpha never goes to SeedVR2."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from midnight_upscale.utils import PipelineError

logger = logging.getLogger(__name__)

ALPHA_MODES = ("lanczos", "bicubic", "nearest")
EDGE_CLEANUP_CHOICES = ("auto", "off", "simple")

# Recolor only pixels that are effectively invisible. Visible hair and
# antialiased edges (alpha above this) keep both their RGB and their alpha.
EDGE_LOW_ALPHA = 16
EDGE_OPAQUE_ALPHA = 200
# A few source pixels, enough to cover a Lanczos kernel without flooding the canvas.
EDGE_RADIUS = 3

_RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
}


def split_rgba(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Split one composited frame into straight RGB and an 8-bit alpha mask.

    RGB is not premultiplied and is not composited onto a background.
    Alpha is grayscale, not a binary mask.
    """

    rgba = np.asarray(image.convert("RGBA"))
    rgb = Image.fromarray(np.array(rgba[:, :, :3], copy=True), mode="RGB")
    alpha = Image.fromarray(np.array(rgba[:, :, 3], copy=True), mode="L")
    return rgb, alpha


def recombine_rgba(rgb: Image.Image, alpha: Image.Image) -> Image.Image:
    """Join straight RGB and a grayscale alpha mask. Sizes must already match."""

    rgb_image = rgb.convert("RGB")
    alpha_image = alpha.convert("L")
    if rgb_image.size != alpha_image.size:
        raise PipelineError(
            f"Cannot recombine RGB {rgb_image.size[0]}x{rgb_image.size[1]} "
            f"with alpha {alpha_image.size[0]}x{alpha_image.size[1]}"
        )
    merged = np.dstack([np.asarray(rgb_image), np.asarray(alpha_image)])
    return Image.fromarray(merged, mode="RGBA")


def upscale_alpha(image: Image.Image, scale: int, mode: str) -> Image.Image:
    """Resize an alpha mask. Values are resampled, never thresholded."""

    if scale < 1:
        raise PipelineError(f"Scale must be >= 1, got {scale}")
    if mode not in _RESAMPLE:
        raise PipelineError(f"Unknown alpha mode {mode!r}. Choose one of: {', '.join(ALPHA_MODES)}")
    gray = image.convert("L")
    width, height = gray.size
    return gray.resize((width * scale, height * scale), resample=_RESAMPLE[mode])


def resolve_edge_cleanup(mode: str, *, has_transparency: bool) -> str:
    """Turn ``auto`` into ``simple`` only when the source has any transparency."""

    if mode not in EDGE_CLEANUP_CHOICES:
        raise PipelineError("--edge-cleanup must be auto, off, or simple")
    if mode != "auto":
        return mode
    resolved = "simple" if has_transparency else "off"
    logger.info(
        "Edge cleanup auto: source %s transparency, using %s",
        "has" if has_transparency else "has no",
        resolved,
    )
    return resolved


def edge_cleanup_simple(image: Image.Image) -> Image.Image:
    """Copy nearby visible RGB a few pixels outward into transparent pixels.

    GIF transparency is 1-bit, so the RGB stored under a transparent index is
    often wrong (commonly black). Lanczos on the alpha mask later blends that
    hidden color into the soft edge. This recolors only those hidden pixels.

    What it changes:
    - RGB of pixels whose alpha is <= 16, and only when an opaque pixel
      (alpha >= 200) lies inside a radius of 3 source pixels.
    - Those RGB values become the average of the opaque neighbors.

    What it does not change:
    - The alpha mask, at any pixel. Transparent pixels stay transparent.
    - RGB of every pixel with alpha above 16, so visible geometry stays put.
    - Pixels farther than the radius from opaque content.

    The mask is not eroded, dilated, or thresholded.
    """

    rgba = np.asarray(image.convert("RGBA"))
    cleaned = _edge_cleanup_array(rgba)
    return Image.fromarray(cleaned, mode="RGBA")


def _box_sum(channel: np.ndarray, radius: int) -> np.ndarray:
    """Sum a square window. Outside the image counts as zero."""

    padded = np.pad(channel.astype(np.float64, copy=False), radius, mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    span = 2 * radius + 1
    return (
        integral[span:, span:]
        - integral[:-span, span:]
        - integral[span:, :-span]
        + integral[:-span, :-span]
    )


def _edge_cleanup_array(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise PipelineError("Edge cleanup expected an RGBA image")
    out = np.array(rgba, copy=True)
    alpha = out[:, :, 3]
    target = alpha <= EDGE_LOW_ALPHA
    opaque = alpha >= EDGE_OPAQUE_ALPHA
    if not np.any(target) or not np.any(opaque):
        return out

    weight = _box_sum(opaque.astype(np.float64), EDGE_RADIUS)
    near = target & (weight > 0)
    if not np.any(near):
        return out

    rgb = out[:, :, :3].astype(np.float64)
    opaque_weight = opaque.astype(np.float64)
    for channel_index in range(3):
        summed = _box_sum(rgb[:, :, channel_index] * opaque_weight, EDGE_RADIUS)
        average = np.zeros_like(summed)
        np.divide(summed, weight, out=average, where=weight > 0)
        updated = np.clip(np.rint(average[near]), 0, 255).astype(np.uint8)
        out[:, :, channel_index][near] = updated
    return out
