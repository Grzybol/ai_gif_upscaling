"""Remove a white video background while retaining fine foreground edges."""

from __future__ import annotations

from collections import deque

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


def remove_exact_white_background(rgba: np.ndarray) -> np.ndarray:
    """Key only RGB #FFFFFF, preserving every other pixel and source alpha."""

    result = np.ascontiguousarray(rgba).copy()
    pure_white = np.all(result[..., :3] == 255, axis=2)
    result[pure_white] = 0
    return result


def add_comics_outline(
    rgba: np.ndarray, *, fade_strength: float = 0.0, fade_depth: int = 1,
    darken_edge: bool = True, discard_faint_alpha: bool = True,
) -> np.ndarray:
    """Darken just the inner, one-pixel silhouette edge after final resizing.

    Keeping the original alpha avoids an opaque halo around antialiased details.
    Very faint keyed pixels can be removed to suppress isolated white specks;
    exact-white keying preserves pre-existing source alpha instead.
    """

    result = np.ascontiguousarray(rgba).copy()
    alpha = result[..., 3]
    if discard_faint_alpha:
        alpha[alpha < 16] = 0
    silhouette = alpha > 0
    if not np.any(silhouette):
        return result
    eroded = np.asarray(
        Image.fromarray(silhouette.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))
    ) > 0
    edge = silhouette & ~eroded
    if darken_edge:
        result[edge, :3] = np.rint(result[edge, :3].astype(np.float32) * 0.5).astype(np.uint8)
    strength = float(np.clip(fade_strength, 0.0, 1.0))
    depth = max(0, min(int(fade_depth), 16))
    if strength > 0 and depth > 0:
        current = silhouette
        alpha_float = alpha.astype(np.float32)
        for ring_index in range(depth):
            inner = np.asarray(
                Image.fromarray(current.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))
            ) > 0
            ring = current & ~inner
            if not np.any(ring):
                break
            factor = 1.0 - strength * (1.0 - ring_index / depth)
            alpha_float[ring] *= factor
            current = inner
        result[..., 3] = np.rint(alpha_float).astype(np.uint8)
    result[~silhouette, :3] = 0
    return result


def tighten_drawn_matte(rgba: np.ndarray) -> np.ndarray:
    """Discard the contaminated outer pixel and rebuild the new edge from inside.

    A color threshold cannot recognize white mixed with gold, skin, or red by video
    compression. This opt-in matte trims one output pixel regardless of its color,
    then borrows RGB from the neighboring interior before the comic outline.
    """

    result = np.ascontiguousarray(rgba).copy()
    mask = result[..., 3] >= 64
    if not np.any(mask):
        result[...] = 0
        return result
    inset = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))
    ) > 0
    interior = np.asarray(
        Image.fromarray(inset.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))
    ) > 0
    edge = inset & ~interior
    if np.any(edge) and np.any(interior):
        source = result[..., :3] * interior[..., None]
        blurred_rgb = np.asarray(
            Image.fromarray(source).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32
        )
        blurred_weight = np.asarray(
            Image.fromarray(interior.astype(np.uint8) * 255).filter(
                ImageFilter.GaussianBlur(2)
            ),
            dtype=np.float32,
        )
        usable = edge & (blurred_weight > 2)
        recovered = blurred_rgb * (255.0 / np.maximum(blurred_weight[..., None], 1))
        result[..., :3][usable] = np.clip(np.rint(recovered[usable]), 0, 255).astype(np.uint8)
    result[..., 3][~inset] = 0
    result[..., 3][edge] = np.minimum(result[..., 3][edge], 192)
    result[..., :3][~inset] = 0
    return result


def draw_outward_border(
    rgba: np.ndarray, *, width: int, color: tuple[int, int, int], smooth_depth: int = 0
) -> np.ndarray:
    """Draw a supersampled outward contour with optional subtle inward color blend."""

    source = np.ascontiguousarray(rgba)
    width = max(0, min(int(width), 16))
    if width == 0 or not np.any(source[..., 3]):
        return source.copy()

    # A fixed transparent gutter prevents clipping when a character touches
    # the canvas edge. It is the same size on every animation frame.
    pad = width + 2
    result = np.pad(source, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    alpha = result[..., 3]
    height, frame_width = alpha.shape
    factor = 4
    geometry_alpha = np.where(alpha >= 16, alpha, 0).astype(np.uint8)
    mask = Image.fromarray(geometry_alpha, "L").resize(
        (frame_width * factor, height * factor), Image.Resampling.BILINEAR
    )
    expanded = mask.filter(ImageFilter.MaxFilter(2 * width * factor + 1))
    expanded_alpha = np.asarray(
        expanded.resize((frame_width, height), Image.Resampling.BOX), dtype=np.int16
    )
    border_alpha = np.maximum(expanded_alpha - alpha.astype(np.int16), 0).astype(np.float32) / 255
    source_alpha = alpha.astype(np.float32) / 255
    behind = border_alpha * (1 - source_alpha)
    combined = source_alpha + behind
    # Straight-alpha compositing keeps opaque artwork unchanged when the
    # optional inward transition is disabled.
    affected = behind > 0
    if np.any(affected):
        original_rgb = result[..., :3][affected].astype(np.float32)
        color_rgb = np.asarray(color, dtype=np.float32)
        mixed = (original_rgb * source_alpha[affected, None]
                 + color_rgb * behind[affected, None]) / combined[affected, None]
        result[..., :3][affected] = np.rint(np.clip(mixed, 0, 255)).astype(np.uint8)
        result[..., 3][affected] = np.rint(combined[affected] * 255).astype(np.uint8)

    depth = max(0, min(int(smooth_depth), 16))
    if depth:
        # Only color a narrow band *inside* the original silhouette. Eroded
        # one-pixel rings establish distance from its edge; blurring this
        # scalar weight (never the artwork RGB) softens stair-step boundaries.
        current = geometry_alpha >= 16
        weight = np.zeros(alpha.shape, dtype=np.float32)
        for ring_index in range(depth):
            eroded = np.asarray(
                Image.fromarray(current.astype(np.uint8) * 255, "L").filter(
                    ImageFilter.MinFilter(3)
                )
            ) > 0
            weight[current & ~eroded] = 0.35 * (1 - ring_index / depth) ** 2
            current = eroded
            if not np.any(current):
                break
        weight = np.asarray(
            Image.fromarray(np.rint(weight * 255).astype(np.uint8), "L").filter(
                ImageFilter.GaussianBlur(0.45)
            ), dtype=np.float32,
        ) / 255
        weight *= (geometry_alpha >= 16) * source_alpha
        color_rgb = np.asarray(color, dtype=np.float32)
        rgb = result[..., :3].astype(np.float32)
        # Existing lineart close to the selected border color needs no second
        # darkening. The transition remains gentle even on bright artwork.
        difference = np.max(np.abs(rgb - color_rgb), axis=2)
        weight *= np.minimum(difference / 96, 1)
        changed = weight > 0
        result[..., :3][changed] = np.rint(
            np.clip(rgb[changed] * (1 - weight[changed, None])
                    + color_rgb * weight[changed, None], 0, 255)
        ).astype(np.uint8)
    return result


def force_remove_white_edges(
    rgba: np.ndarray, *, depth: int, darkest_shade: int
) -> np.ndarray:
    """Make near-white edge pixels transparent without touching interior whites.

    An opaque source may still have a white canvas (the None mode). In that case,
    peel only border-connected white first to establish an asset silhouette.
    """

    result = np.ascontiguousarray(rgba).copy()
    depth = max(0, min(int(depth), 16))
    if depth == 0:
        return result
    cutoff = max(0, min(int(darkest_shade), 255))
    candidate = np.all(result[..., :3] >= cutoff, axis=2)
    alpha = result[..., 3]
    if np.all(alpha == 255) and np.any(candidate):
        border_white = _border_connected(candidate)
        alpha[border_white] = 0
    current = alpha > 0
    for _ in range(depth):
        if not np.any(current):
            break
        inner = np.asarray(
            Image.fromarray(current.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))
        ) > 0
        alpha[(current & ~inner) & candidate] = 0
        current = inner
    result[alpha == 0, :3] = 0
    return result


def _border_connected(candidate: np.ndarray) -> np.ndarray:
    """Four-connected white canvas, seeded only at the image border."""

    height, width = candidate.shape
    visited = np.zeros((height, width), dtype=np.uint8)
    queue: deque[tuple[int, int]] = deque()

    def enqueue(y: int, x: int) -> None:
        if not visited[y, x] and candidate[y, x]:
            visited[y, x] = 1
            queue.append((y, x))

    for x in range(width):
        enqueue(0, x)
        enqueue(height - 1, x)
    for y in range(height):
        enqueue(y, 0)
        enqueue(y, width - 1)
    while queue:
        y, x = queue.popleft()
        if y > 0:
            enqueue(y - 1, x)
        if y + 1 < height:
            enqueue(y + 1, x)
        if x > 0:
            enqueue(y, x - 1)
        if x + 1 < width:
            enqueue(y, x + 1)
    return visited.astype(bool)


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
