"""Chroma key with soft alpha, color decontamination, and spill suppression.

Keying works on the chroma plane (Cb/Cr), not on RGB distance, so a green
screen that is lit unevenly or falls into shadow still keys cleanly. Alpha is
a smooth ramp between a hard-transparent distance and a hard-opaque distance,
so antialiased edges and hair stay semi-transparent. A binary mask is produced
only when ``hard_mask`` is requested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

# Maps the 0..1 sliders onto chroma-plane distances (Cb/Cr span about 0..1).
TOLERANCE_SCALE = 0.5
SOFTNESS_SCALE = 0.4

EDGE_CLEANUP_LEVELS = ("Off", "Light", "Medium", "Strong")
# level -> (noise gate, erosion passes, blur radius in px)
_EDGE_PARAMS = {
    "off": (0.0, 0, 0.0),
    "light": (0.06, 0, 0.0),
    "medium": (0.06, 1, 0.5),
    "strong": (0.06, 2, 0.8),
}

_SPILL_MIN_DOMINANCE = 40.0


@dataclass(frozen=True)
class ChromaSettings:
    key_color: tuple[int, int, int] = (0, 255, 0)
    tolerance: float = 0.30
    softness: float = 0.20
    spill: float = 0.60
    edge_cleanup: str = "light"
    hard_mask: bool = False


def parse_hex_color(text: str) -> tuple[int, int, int]:
    """``#00ff00``, ``00ff00``, or ``rgb(0, 255, 0)`` to an RGB tuple."""

    value = (text or "").strip()
    if value.lower().startswith("rgb"):
        numbers = value[value.index("(") + 1 : value.rindex(")")].split(",")
        channels = [int(round(float(part))) for part in numbers[:3]]
    else:
        value = value.lstrip("#")
        if len(value) == 3:
            value = "".join(ch * 2 for ch in value)
        if len(value) != 6:
            raise ValueError(f"Not a color: {text!r}")
        channels = [int(value[i : i + 2], 16) for i in (0, 2, 4)]
    if len(channels) != 3:
        raise ValueError(f"Not a color: {text!r}")
    return tuple(max(0, min(255, c)) for c in channels)  # type: ignore[return-value]


def to_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def chroma_plane(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BT.601 Cb and Cr of an RGB array in 0..255, each in about -0.5..0.5."""

    data = rgb.astype(np.float32) / 255.0
    red, green, blue = data[..., 0], data[..., 1], data[..., 2]
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue
    return cb, cr


def chroma_distance(rgb: np.ndarray, key: tuple[int, int, int]) -> np.ndarray:
    cb, cr = chroma_plane(rgb)
    key_cb, key_cr = chroma_plane(np.array(key, dtype=np.uint8).reshape(1, 1, 3))
    return np.hypot(cb - key_cb[0, 0], cr - key_cr[0, 0])


def smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def chroma_alpha(rgb: np.ndarray, settings: ChromaSettings) -> np.ndarray:
    """Soft alpha in 0..1. 0 is background, 1 is foreground."""

    distance = chroma_distance(rgb, settings.key_color)
    inner = max(0.0, settings.tolerance) * TOLERANCE_SCALE
    width = max(settings.softness, 0.0) * SOFTNESS_SCALE
    if width < 1e-6:
        return (distance > inner).astype(np.float32)
    return smoothstep((distance - inner) / width).astype(np.float32)


def clean_alpha(alpha: np.ndarray, level: str) -> np.ndarray:
    """Drop faint noise, then optionally pull the edge in by a pixel or two."""

    gate, passes, blur = _EDGE_PARAMS.get(level.lower(), _EDGE_PARAMS["off"])
    if gate <= 0.0 and passes == 0:
        return alpha
    cleaned = np.where(alpha < gate, 0.0, alpha)
    cleaned = np.where(cleaned > 1.0 - gate, 1.0, cleaned).astype(np.float32)
    if passes == 0:
        return cleaned
    image = Image.fromarray((cleaned * 255.0 + 0.5).astype(np.uint8))
    for _ in range(passes):
        image = image.filter(ImageFilter.MinFilter(3))
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(image, dtype=np.float32) / 255.0


def spill_channels(key: tuple[int, int, int]) -> tuple[list[int], list[int]] | None:
    """Channels that carry the key color, and the channels that cap them.

    A green key lets green exceed max(red, blue). A cyan or yellow key has two
    carrying channels, capped by the third. A gray key has no spill to remove.
    """

    order = sorted(range(3), key=lambda i: key[i])
    low, mid, top = order
    if key[top] - key[mid] >= _SPILL_MIN_DOMINANCE:
        return [top], [low, mid]
    if key[mid] - key[low] >= _SPILL_MIN_DOMINANCE:
        return [top, mid], [low]
    return None


def despill(rgb: np.ndarray, key: tuple[int, int, int], strength: float) -> np.ndarray:
    """Pull key-colored contamination out of the foreground. ``rgb`` is float 0..255."""

    layout = spill_channels(key)
    if layout is None or strength <= 0:
        return rgb
    carriers, caps = layout
    limit = rgb[..., caps[0]]
    for cap in caps[1:]:
        limit = np.maximum(limit, rgb[..., cap])
    out = rgb.copy()
    for channel in carriers:
        excess = np.maximum(rgb[..., channel] - limit, 0.0)
        out[..., channel] = rgb[..., channel] - min(strength, 1.0) * excess
    return out


def decontaminate(
    rgb: np.ndarray, alpha: np.ndarray, key: tuple[int, int, int], strength: float
) -> np.ndarray:
    """Remove the background's share from semi-transparent pixels.

    A pixel with alpha a was blended as ``a * fg + (1 - a) * key``. Solving for fg
    takes the key color back out of antialiased edges and hair.
    """

    if strength <= 0:
        return rgb
    partial = (alpha > 0.02) & (alpha < 0.999)
    if not partial.any():
        return rgb
    safe = np.maximum(alpha, 0.05)[..., None]
    key_arr = np.array(key, dtype=np.float32).reshape(1, 1, 3)
    solved = np.clip((rgb - (1.0 - safe) * key_arr) / safe, 0.0, 255.0)
    weight = min(strength, 1.0) * partial[..., None]
    return rgb + weight * (solved - rgb)


def chroma_key(rgb: np.ndarray, settings: ChromaSettings) -> np.ndarray:
    """Key an ``HxWx3`` uint8 image. Returns ``HxWx4`` uint8 RGBA."""

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("chroma_key needs an HxWx3 RGB array")
    rgb = np.ascontiguousarray(rgb[..., :3])
    alpha = clean_alpha(chroma_alpha(rgb, settings), settings.edge_cleanup)
    if settings.hard_mask:
        alpha = (alpha >= 0.5).astype(np.float32)
    color = rgb.astype(np.float32)
    color = decontaminate(color, alpha, settings.key_color, settings.spill)
    color = despill(color, settings.key_color, settings.spill)
    out = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = np.clip(color + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def estimate_key_color(
    frames: list[np.ndarray], *, ring_fraction: float = 0.03
) -> tuple[tuple[int, int, int], float]:
    """Median border color of the sample frames, and how much of the border matches it."""

    samples: list[np.ndarray] = []
    for frame in frames:
        height, width = frame.shape[:2]
        ring = max(2, int(round(min(height, width) * ring_fraction)))
        rgb = frame[..., :3]
        samples.append(rgb[:ring].reshape(-1, 3))
        samples.append(rgb[-ring:].reshape(-1, 3))
        samples.append(rgb[ring:-ring, :ring].reshape(-1, 3))
        samples.append(rgb[ring:-ring, -ring:].reshape(-1, 3))
    pixels = np.concatenate(samples, axis=0)
    median = np.median(pixels, axis=0)
    color = (int(median[0]), int(median[1]), int(median[2]))
    distance = chroma_distance(pixels.reshape(1, -1, 3), color)
    coverage = float(np.mean(distance < 0.08))
    return color, coverage


def looks_chroma_green(color: tuple[int, int, int], coverage: float) -> bool:
    """A strongly green, uniform border: the signature of a green-screen clip."""

    red, green, blue = color
    return coverage >= 0.55 and green - max(red, blue) >= 40
