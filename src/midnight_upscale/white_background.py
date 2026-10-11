"""Remove a white video background while retaining fine foreground edges."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def remove_white_background(rgba: np.ndarray) -> np.ndarray:
    """Key near-white pixels, including gaps enclosed by the subject."""

    result = np.ascontiguousarray(rgba).copy()
    rgb = result[..., :3].astype(np.float32)
    distance = 255.0 - rgb.min(axis=2)
    color_range = np.ptp(rgb, axis=2)
    alpha = np.maximum(
        np.clip((distance - 12.0) / 38.0, 0, 1),
        np.clip((color_range - 8.0) / 16.0, 0, 1),
    )
    rgb, alpha = _unmix_white_edges(rgb, alpha)
    result[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    result[..., 3] = np.rint(result[..., 3] * alpha).astype(np.uint8)
    result[result[..., 3] == 0, :3] = 0
    return result


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").astype(np.float32)
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
    height, width = values.shape
    window = 2 * radius + 1
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]
    total = (
        integral[y + window, x + window]
        - integral[y, x + window]
        - integral[y + window, x]
        + integral[y, x]
    )
    return total / (window * window)


def _unmix_white_edges(rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate foreground coverage and undo compositing against white."""

    opaque = Image.fromarray((alpha >= 0.99).astype(np.uint8) * 255)
    core = np.asarray(opaque.filter(ImageFilter.MinFilter(9))) > 0
    weight = _box_blur(core.astype(np.float32), 7)
    reference = np.stack(
        [_box_blur(rgb[..., channel] * core, 7) / np.maximum(weight, 1e-6) for channel in range(3)],
        axis=2,
    )
    # Thin leaves disappear under a 9px erosion. Borrow color from nearby
    # dark, opaque pixels when the broad interior has no reference.
    thin_core = (alpha >= 0.99) & (rgb.min(axis=2) < 150)
    thin_weight = _box_blur(thin_core.astype(np.float32), 4)
    thin_reference = np.stack(
        [
            _box_blur(rgb[..., channel] * thin_core, 4) / np.maximum(thin_weight, 1e-6)
            for channel in range(3)
        ],
        axis=2,
    )
    fallback = (weight <= 0.01) & (thin_weight > 0.01)
    reference = np.where(fallback[..., None], thin_reference, reference)
    weight = np.where(fallback, thin_weight, weight)
    direction = 255.0 - reference
    denominator = np.sum(direction * direction, axis=2)
    coverage = np.clip(np.sum((255.0 - rgb) * direction, axis=2) / np.maximum(denominator, 1), 0, 1)
    predicted = 255.0 - coverage[..., None] * direction
    residual = np.max(np.abs(rgb - predicted), axis=2)
    near_background = (
        np.asarray(
            Image.fromarray((alpha < 0.1).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))
        )
        > 0
    )
    pale_fringe = near_background & (rgb.min(axis=2) > 100) & (np.ptp(rgb, axis=2) < 70)
    edge = (
        (~core)
        & (weight > 0.01)
        & (alpha > 0)
        & (denominator > 400)
        & ((residual < 18) | (rgb.min(axis=2) > 150) | pale_fringe)
    )
    coverage = np.where(coverage >= 0.9, 1.0, coverage)
    refined = np.where(edge, coverage, alpha)
    cleaned = 255.0 - (255.0 - rgb) / np.maximum(refined[..., None], 1 / 255)
    cleaned = np.where((residual >= 18)[..., None], reference, cleaned)
    return np.where(edge[..., None], cleaned, rgb), refined
