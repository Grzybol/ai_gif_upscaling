"""User-selected colors and painted protection for the drawn/comics converter."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from midnight_upscale.utils import PipelineError

RGB = tuple[int, int, int]
ColorRange = tuple[RGB, RGB]

_HEX = r"#[0-9a-fA-F]{6}"
_EXACT = re.compile(rf"^({_HEX})$")
_RANGE = re.compile(rf"^({_HEX})\s*\.\.\s*({_HEX})$")
_TOLERANCE = re.compile(rf"^({_HEX})\s*\+/-\s*(\d{{1,3}})$")


def _rgb(value: str) -> RGB:
    return tuple(bytes.fromhex(value[1:]))  # type: ignore[return-value]


def parse_color_ranges(text: str | None) -> tuple[ColorRange, ...]:
    """One exact color, inclusive RGB range, or per-channel tolerance per line."""

    rules: list[ColorRange] = []
    for number, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        exact = _EXACT.fullmatch(line)
        span = _RANGE.fullmatch(line)
        tolerance = _TOLERANCE.fullmatch(line)
        if exact:
            color = _rgb(exact[1])
            rules.append((color, color))
        elif span:
            low, high = _rgb(span[1]), _rgb(span[2])
            if any(a > b for a, b in zip(low, high, strict=True)):
                raise PipelineError(f"Color range on line {number} must run from low to high RGB.")
            rules.append((low, high))
        elif tolerance:
            center, radius = _rgb(tolerance[1]), int(tolerance[2])
            if radius > 255:
                raise PipelineError(f"Color tolerance on line {number} must be 0–255.")
            rules.append((
                tuple(max(0, channel - radius) for channel in center),
                tuple(min(255, channel + radius) for channel in center),
            ))
        else:
            raise PipelineError(
                f"Invalid color rule on line {number}: {line!r}. Use #RRGGBB, "
                "#LOW..#HIGH, or #RRGGBB +/- N."
            )
        if len(rules) > 64:
            raise PipelineError("Use at most 64 color rules per conversion.")
    return tuple(rules)


def selected_color_mask(rgb: np.ndarray, rules: tuple[ColorRange, ...]) -> np.ndarray:
    """Match the *decoded source* RGB, before edge decontamination changes it."""

    selected = np.zeros(rgb.shape[:2], dtype=bool)
    for low, high in rules:
        selected |= np.all((rgb >= low) & (rgb <= high), axis=2)
    return selected


def remove_selected_colors(
    keyed: np.ndarray, original: np.ndarray, rules: tuple[ColorRange, ...]
) -> np.ndarray:
    """Make selected source RGB colors transparent in an already-keyed frame."""

    if not rules:
        return keyed
    result = np.ascontiguousarray(keyed).copy()
    selected = selected_color_mask(original[..., :3], rules)
    result[selected] = 0
    return result


def save_painted_mask(editor: object, directory: Path) -> str:
    """Persist the editor's paint layers as one binary mask; empty paint means off."""

    if not isinstance(editor, dict) or not editor.get("layers"):
        return ""
    background = editor.get("background")
    if background is None:
        return ""
    height, width = np.asarray(background).shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for layer in editor["layers"]:
        image = np.asarray(layer)
        if image.shape[:2] != (height, width):
            raise PipelineError("Protection brush layer does not match the source frame size.")
        if image.ndim != 3 or image.shape[2] != 4:
            raise PipelineError("Protection brush layer must contain transparency.")
        mask |= image[..., 3] > 0
    if not np.any(mask):
        return ""
    payload = mask.astype(np.uint8).tobytes()
    digest = hashlib.sha256(width.to_bytes(4, "big") + height.to_bytes(4, "big") + payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.png"
    if not path.exists():
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path)
    return str(path.resolve())


@lru_cache(maxsize=12)
def load_painted_mask(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def restore_protected_pixels(
    keyed: np.ndarray, original: np.ndarray, mask: np.ndarray | None
) -> np.ndarray:
    if mask is None:
        return keyed
    if mask.shape != keyed.shape[:2]:
        raise PipelineError("Protection mask size differs from the decoded video frame.")
    keyed[mask] = original[mask]
    return keyed
