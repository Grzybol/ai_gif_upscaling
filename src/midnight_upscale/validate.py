"""Fail the job when counts, sizes, alpha, or duration do not match."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from midnight_upscale.models import JobMetadata
from midnight_upscale.utils import ValidationError, list_indexed_frames


def assert_frame_count(directory: Path, expected: int, label: str) -> list[Path]:
    frames = list_indexed_frames(directory)
    if len(frames) != expected:
        raise ValidationError(
            f"{label} has {len(frames)} frames in {directory}, expected {expected}. "
            "Missing frames are not replaced."
        )
    return frames


def assert_same_size(paths: list[Path], size: tuple[int, int], label: str) -> None:
    width, height = size
    for path in paths:
        with Image.open(path) as image:
            if image.size != size:
                raise ValidationError(
                    f"{label} frame {path.name} is {image.size[0]}x{image.size[1]}, "
                    f"expected {width}x{height}"
                )


def assert_png_has_alpha(path: Path) -> None:
    with Image.open(path) as image:
        if "A" not in image.getbands():
            raise ValidationError(
                f"{path.name} has no alpha channel (mode {image.mode}). "
                "Transparency was lost before encoding."
            )


def sample_indexes(count: int) -> list[int]:
    if count <= 0:
        return []
    if count <= 8:
        return list(range(count))
    picked = {0, count // 4, count // 2, (3 * count) // 4, count - 1}
    return sorted(picked)


def assert_alpha_preserved(source: Image.Image, final: Image.Image, frame_name: str) -> None:
    """Catch a mask that was flattened or forced to binary.

    Lanczos can move an exact 0 or 255 by a few levels, so the final image is
    checked for surviving transparent, opaque, and semitransparent pixels
    rather than for identical values.
    """

    source_alpha = np.asarray(source.convert("RGBA"))[:, :, 3]
    final_alpha = np.asarray(final.convert("RGBA"))[:, :, 3]
    if np.any(source_alpha == 0) and not np.any(final_alpha < 32):
        raise ValidationError(
            f"{frame_name}: the source has fully transparent pixels but the output alpha "
            "is fully opaque. Transparency was destroyed."
        )
    if np.any(source_alpha == 255) and not np.any(final_alpha > 224):
        raise ValidationError(
            f"{frame_name}: the source has opaque pixels but the output alpha lost them."
        )
    source_semi = np.any((source_alpha > 0) & (source_alpha < 255))
    final_semi = np.any((final_alpha > 0) & (final_alpha < 255))
    if source_semi and not final_semi:
        raise ValidationError(
            f"{frame_name}: the source has semitransparent pixels but the output alpha is "
            "only 0 or 255. The alpha mask looks thresholded."
        )


def assert_sampled_alpha(source_dir: Path, final_dir: Path) -> None:
    source_frames = list_indexed_frames(source_dir)
    final_frames = list_indexed_frames(final_dir)
    if len(source_frames) != len(final_frames):
        raise ValidationError(
            f"Cannot compare alpha: source has {len(source_frames)} frames and "
            f"final RGBA has {len(final_frames)}"
        )
    for index in sample_indexes(len(source_frames)):
        with Image.open(source_frames[index]) as source, Image.open(final_frames[index]) as final:
            assert_png_has_alpha(final_frames[index])
            assert_alpha_preserved(source, final, f"frame {index:06d}")


def assert_duration_close(actual_sec: float, expected_sec: float, tolerance_sec: float) -> None:
    if abs(actual_sec - expected_sec) > tolerance_sec:
        raise ValidationError(
            f"Output duration {actual_sec:.6f}s does not match the encoded timeline "
            f"{expected_sec:.6f}s (tolerance {tolerance_sec:.6f}s). "
            "Animation speed was not preserved."
        )


def format_result_report(
    metadata: JobMetadata,
    output: Path,
    probed_duration_sec: float,
    *,
    notes: list[str] | None = None,
) -> str:
    lines = [
        "SOURCE",
        f"{metadata.original_width}x{metadata.original_height}",
        f"{metadata.frame_count} frames",
        f"{metadata.total_duration_ms / 1000:.3f} s",
        "",
        "TARGET",
        f"{metadata.target_width}x{metadata.target_height}",
        f"{metadata.frame_count} frames",
        f"{probed_duration_sec:.3f} s",
        "",
        "Output:",
        str(output),
    ]
    for note in notes or []:
        lines.extend(["", note])
    return "\n".join(lines) + "\n"
