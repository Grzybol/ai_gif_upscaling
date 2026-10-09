"""Conservative temporal smoothing of per-frame masks.

Per-frame AI segmentation flickers: an edge or a small region flips between
frames. This module averages each frame's alpha with its neighbors. Only alpha
is touched, so frame geometry and RGB are unchanged, and no frame is blended
into another or interpolated.

A neighbor only contributes where it differs from the current frame by a
flicker-sized amount. A bigger difference is real motion (the character moved),
so it is ignored and no ghost is smeared along the movement.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from midnight_upscale.utils import PipelineError


@dataclass(frozen=True)
class TemporalLevel:
    name: str
    # Weights for frames -radius..+radius around the current one.
    weights: tuple[float, ...]
    # Largest alpha difference (0..1) a neighbor may have and still count as flicker.
    max_correction: float

    @property
    def radius(self) -> int:
        return len(self.weights) // 2


LEVELS: dict[str, TemporalLevel] = {
    "off": TemporalLevel("off", (1.0,), 0.0),
    "low": TemporalLevel("low", (1.0, 2.0, 1.0), 0.35),
    "medium": TemporalLevel("medium", (1.0, 2.0, 3.0, 2.0, 1.0), 0.6),
}
TEMPORAL_LABELS = {"Off": "off", "Low": "low", "Medium": "medium"}


def get_level(name: str) -> TemporalLevel:
    try:
        return LEVELS[name.lower()]
    except KeyError as exc:
        raise PipelineError(
            f"Unknown temporal smoothing {name!r}. Use off, low, or medium."
        ) from exc


def smooth_frame(window: list[np.ndarray], center: int, level: TemporalLevel) -> np.ndarray:
    """Smooth ``window[center]`` using the neighbors that exist in ``window``.

    ``window`` holds consecutive ``HxW`` uint8 alpha masks. At the ends of a clip
    the missing neighbors are dropped and the remaining weights are renormalized.
    """

    current = window[center].astype(np.float32)
    if level.max_correction <= 0.0 or len(window) == 1:
        return window[center].copy()
    radius = level.radius
    gate = level.max_correction * 255.0
    total = current * level.weights[radius]
    weight_sum = np.full(current.shape, level.weights[radius], dtype=np.float32)
    for offset in range(-radius, radius + 1):
        position = center + offset
        if offset == 0 or not 0 <= position < len(window):
            continue
        neighbor = window[position].astype(np.float32)
        # A neighbor that differs by more than flicker-size is motion, not noise. Skip it.
        weight = level.weights[offset + radius] * (np.abs(neighbor - current) <= gate)
        total += weight * neighbor
        weight_sum += weight
    return np.clip(total / weight_sum + 0.5, 0, 255).astype(np.uint8)


def smooth_masks(
    load: Callable[[int], np.ndarray], count: int, level: TemporalLevel
) -> Iterator[np.ndarray]:
    """Yield the smoothed mask for frames ``0..count-1``, holding only a sliding window."""

    if count <= 0:
        return
    radius = level.radius
    window: deque[np.ndarray] = deque()
    first = 0  # index of window[0]
    next_load = 0
    for index in range(count):
        while next_load <= min(index + radius, count - 1):
            window.append(load(next_load))
            next_load += 1
        while first < index - radius:
            window.popleft()
            first += 1
        yield smooth_frame(list(window), index - first, level)
