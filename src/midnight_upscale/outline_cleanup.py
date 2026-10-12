"""Conservative, alpha-boundary-only cleanup for illustrated RGBA frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class OutlineCleanupSettings:
    enabled: bool = False
    strength: str = "Medium"
    outline_width: float = 1.0
    alpha_smoothing: float = 0.2
    remove_speckles: bool = True
    normalize: bool = True
    protect_dark: bool = True

    def __post_init__(self) -> None:
        if self.strength not in ("Low", "Medium", "High"):
            raise ValueError("Strength must be Low, Medium or High")
        if self.outline_width not in (1.0, 1.5, 2.0):
            raise ValueError("Outline width must be 1, 1.5 or 2 pixels")
        if not 0 <= self.alpha_smoothing <= 1:
            raise ValueError("Alpha smoothing must be between 0 and 1")


def _neighbors(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1)
    height, width = mask.shape
    count = np.zeros((height, width), dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                count += padded[dy:dy + height, dx:dx + width]
    return count


def clean_outline(frame: np.ndarray, settings: OutlineCleanupSettings) -> np.ndarray:
    """Clean isolated edge defects while retaining connected dark contour pixels.

    The algorithm never changes interior pixels. It is deliberately conservative:
    long contour runs, including uneven hand-drawn strokes, are not regularized.
    """
    if not settings.enabled:
        return frame.copy()
    if frame.ndim != 3 or frame.shape[2] != 4 or frame.dtype != np.uint8:
        raise ValueError("Outline cleanup requires a uint8 RGBA frame")
    result = frame.copy()
    alpha = frame[:, :, 3]
    opaque = alpha >= 96
    if not opaque.any() or opaque.all():
        return result

    radius = max(1, round(settings.outline_width + 1))
    alpha_image = Image.fromarray(alpha, "L")
    inward = np.asarray(alpha_image.filter(ImageFilter.MinFilter(2 * radius + 1)))
    outward = np.asarray(alpha_image.filter(ImageFilter.MaxFilter(2 * radius + 1)))
    edge = (inward < 160) & (outward > 32)
    edge_inside = edge & (alpha > 0)

    rgb = frame[:, :, :3].astype(np.float32)
    luma = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    dark = edge_inside & (luma < 82) & opaque
    dark_neighbors = _neighbors(dark)
    # Two nearby dark pixels indicate a line; a lone pixel is likely residue.
    protected = dark & (dark_neighbors >= 2) if settings.protect_dark else np.zeros_like(dark)
    dark_specks = dark & (dark_neighbors <= 1) & ~protected

    visibility = _neighbors(opaque)
    strength = {"Low": 3, "Medium": 4, "High": 5}[settings.strength]
    alpha_spikes = edge_inside & opaque & (visibility <= strength) & ~protected
    if settings.remove_speckles:
        # Only detach one-pixel protrusions or sparse dark dots, not whole lines.
        removal = alpha_spikes & (visibility <= 2)
        result[:, :, 3][removal] = 0
        # Replace RGB of isolated dark contamination with the closest visible
        # local color, but keep alpha to avoid punching holes in the contour.
        candidates = dark_specks & ~removal
        if candidates.any():
            padded_rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
            padded_good = np.pad(opaque & ~dark, 1)
            total = np.zeros_like(rgb)
            count = np.zeros(alpha.shape, dtype=np.float32)
            h, w = alpha.shape
            for dy in range(3):
                for dx in range(3):
                    good = padded_good[dy:dy + h, dx:dx + w]
                    total += padded_rgb[dy:dy + h, dx:dx + w] * good[:, :, None]
                    count += good
            replace = candidates & (count >= 2)
            average = (total / np.maximum(count[:, :, None], 1)).astype(np.uint8)
            result[:, :, :3][replace] = average[replace]

    if settings.normalize:
        # Taper only isolated excess dark pixels at the outermost alpha edge.
        # Connected dark contour runs remain untouched.
        fringe = dark_specks & (visibility <= strength) & ~protected
        softened = np.minimum(result[:, :, 3][fringe],
                              np.uint8({"Low": 208, "Medium": 176, "High": 144}[settings.strength]))
        result[:, :, 3][fringe] = softened

    if settings.alpha_smoothing:
        smoothed = np.asarray(alpha_image.filter(ImageFilter.GaussianBlur(radius=0.65))).astype(np.float32)
        blend = settings.alpha_smoothing * {"Low": 0.35, "Medium": 0.5, "High": 0.65}[settings.strength]
        target = result[:, :, 3].astype(np.float32)
        # Preserve solid interior and protected strokes, even when the user
        # requests a stronger smoothing value.
        smoothing_zone = edge & ~protected & (inward < 96)
        target[smoothing_zone] = (1 - blend) * target[smoothing_zone] + blend * smoothed[smoothing_zone]
        result[:, :, 3][smoothing_zone] = np.rint(target[smoothing_zone]).astype(np.uint8)
    return result
