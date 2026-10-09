"""Interpolation hook.

RIFE is intentionally not part of the default pipeline. Calling anything other
than ``none`` is an error so a run cannot accidentally retiming the loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from midnight_upscale.utils import PipelineError


class FrameInterpolator(Protocol):
    """Future RIFE (or other) interpolator.

    Implementations must return a new frame list and a new per-frame duration
    list whose sum is the timing the encoder should use. Nothing in the
    default pipeline calls this.
    """

    def interpolate(
        self, frames: list[Path], durations_ms: list[int]
    ) -> tuple[list[Path], list[int]]:
        """Return interpolated frames and their durations."""


def resolve_interpolation(mode: str) -> str:
    """Accept only ``none``.

    A later interpolator can be selected here without becoming the default.
    """

    if mode != "none":
        raise PipelineError(
            f"Interpolation mode {mode!r} is not implemented. "
            "The only supported value is --interpolate none, which leaves every "
            "frame and its original duration unchanged. RIFE is not part of this pipeline."
        )
    return mode
