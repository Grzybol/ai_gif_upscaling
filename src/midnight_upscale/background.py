"""Choose and apply a background-removal mode for each frame.

Modes: Auto, None, Chroma Key, White background, AI Segmentation. Auto reports what it
resolved to, so the choice is never silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from midnight_upscale.chroma import (
    ChromaSettings,
    chroma_key,
    estimate_key_color,
    looks_chroma_green,
    to_hex,
)
from midnight_upscale.segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    INSTALL_HELP,
    BackgroundRemover,
    BackgroundRemoverUnavailable,
    ai_available,
    create_remover,
)
from midnight_upscale.utils import PipelineError
from midnight_upscale.white_background import remove_white_background

MODES = ("auto", "none", "chroma", "white", "ai")
MODE_LABELS = {
    "Auto": "auto",
    "None": "none",
    "Chroma Key": "chroma",
    "White background": "white",
    "AI Segmentation": "ai",
}

# Resolved modes: what a frame is actually run through.
RESOLVED_ALPHA = "alpha"
RESOLVED_NONE = "none"
RESOLVED_CHROMA = "chroma"
RESOLVED_WHITE = "white"
RESOLVED_AI = "ai"


@dataclass(frozen=True)
class BackgroundSettings:
    mode: str = "auto"
    chroma: ChromaSettings = field(default_factory=ChromaSettings)
    # Manual Chroma Key: read the key color from the frame border instead of the picker.
    sample_key_color: bool = False
    ai_backend: str = DEFAULT_BACKEND
    ai_model: str = DEFAULT_MODEL


@dataclass(frozen=True)
class ResolvedBackground:
    mode: str
    label: str
    reason: str
    chroma: ChromaSettings | None = None
    ai_backend: str = DEFAULT_BACKEND
    ai_model: str = DEFAULT_MODEL

    @property
    def needs_ai(self) -> bool:
        return self.mode == RESOLVED_AI


def resolve_background(
    settings: BackgroundSettings,
    *,
    has_alpha: bool,
    sample_frames: list[np.ndarray],
) -> ResolvedBackground:
    """Decide the mode. ``sample_frames`` are RGB(A) arrays used to look for a green screen."""

    mode = settings.mode
    if mode not in MODES:
        raise PipelineError(f"Unknown background mode {mode!r}")

    if mode == "none":
        return ResolvedBackground(RESOLVED_NONE, "NONE", "Frames are used as decoded.")

    if mode == "chroma":
        chroma = settings.chroma
        reason = f"Key color {to_hex(chroma.key_color)} from the color picker."
        if settings.sample_key_color and sample_frames:
            color, coverage = estimate_key_color(sample_frames)
            chroma = _with_key(chroma, color)
            reason = f"Key color {to_hex(color)} sampled from the frame border ({coverage:.0%})."
        return ResolvedBackground(RESOLVED_CHROMA, "CHROMA KEY", reason, chroma=chroma)

    if mode == "white":
        return ResolvedBackground(
            RESOLVED_WHITE,
            "WHITE BACKGROUND",
            "Near-white pixels are keyed throughout the frame, including enclosed gaps.",
        )

    if mode == "ai":
        return _ai(settings, "AI SEGMENTATION", "Selected manually.")

    # Auto
    if has_alpha:
        return ResolvedBackground(
            RESOLVED_ALPHA,
            "AUTO -> PRESERVE ALPHA",
            "The input already contains transparency. It is kept as is.",
        )
    if sample_frames:
        color, coverage = estimate_key_color(sample_frames)
        if looks_chroma_green(color, coverage):
            chroma = _with_key(settings.chroma, color)
            return ResolvedBackground(
                RESOLVED_CHROMA,
                "AUTO -> CHROMA KEY",
                f"The border is chroma green {to_hex(color)} ({coverage:.0%} of it matches).",
                chroma=chroma,
            )
    return _ai(settings, "AUTO -> AI SEGMENTATION", "No alpha and no uniform green border.")


def _ai(settings: BackgroundSettings, label: str, reason: str) -> ResolvedBackground:
    if not ai_available(settings.ai_backend):
        # Fail before any frame work, with the install instructions.
        raise BackgroundRemoverUnavailable(f"{label}: {INSTALL_HELP}")
    return ResolvedBackground(
        RESOLVED_AI,
        label,
        reason,
        ai_backend=settings.ai_backend,
        ai_model=settings.ai_model,
    )


def _with_key(chroma: ChromaSettings, color: tuple[int, int, int]) -> ChromaSettings:
    return ChromaSettings(
        key_color=color,
        tolerance=chroma.tolerance,
        softness=chroma.softness,
        spill=chroma.spill,
        edge_cleanup=chroma.edge_cleanup,
        hard_mask=chroma.hard_mask,
    )


def open_remover(resolved: ResolvedBackground) -> BackgroundRemover | None:
    """Create the AI model for a resolved AI mode. Other modes need none."""

    if not resolved.needs_ai:
        return None
    return create_remover(resolved.ai_backend, resolved.ai_model)


def remove_background(
    resolved: ResolvedBackground,
    rgba: np.ndarray,
    remover: BackgroundRemover | None = None,
) -> np.ndarray:
    """Return a new ``HxWx4`` uint8 RGBA frame for one decoded frame.

    Existing transparency in the source is multiplied in, never replaced.
    """

    if resolved.mode in {RESOLVED_ALPHA, RESOLVED_NONE}:
        return np.ascontiguousarray(rgba)
    source_alpha = rgba[..., 3]
    if resolved.mode == RESOLVED_CHROMA:
        assert resolved.chroma is not None
        keyed = chroma_key(rgba[..., :3], resolved.chroma)
    elif resolved.mode == RESOLVED_WHITE:
        # This function multiplies the source alpha itself.
        return remove_white_background(rgba)
    elif resolved.mode == RESOLVED_AI:
        if remover is None:
            raise PipelineError("AI segmentation needs a loaded model")
        mask = remover.alpha(rgba[..., :3])
        if mask.shape != rgba.shape[:2]:
            raise PipelineError(
                f"Segmentation mask is {mask.shape[1]}x{mask.shape[0]}, "
                f"expected {rgba.shape[1]}x{rgba.shape[0]}"
            )
        keyed = np.empty_like(rgba)
        keyed[..., :3] = rgba[..., :3]
        keyed[..., 3] = mask
    else:  # pragma: no cover - guarded by resolve_background
        raise PipelineError(f"Unknown resolved mode {resolved.mode!r}")
    if np.any(source_alpha < 255):
        scaled = keyed[..., 3].astype(np.uint16) * source_alpha.astype(np.uint16)
        keyed[..., 3] = ((scaled + 127) // 255).astype(np.uint8)
    return keyed
