"""Join upscaled RGB frames with the separately upscaled alpha masks."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from midnight_upscale.alpha import recombine_rgba
from midnight_upscale.progress import checkpoint, report
from midnight_upscale.utils import ValidationError, frame_path, list_indexed_frames


def recombine_directories(rgb_dir: Path, alpha_dir: Path, output_dir: Path) -> list[Path]:
    rgb_frames = list_indexed_frames(rgb_dir)
    alpha_frames = list_indexed_frames(alpha_dir)
    if len(rgb_frames) != len(alpha_frames):
        raise ValidationError(
            f"Upscaled RGB has {len(rgb_frames)} frames in {rgb_dir}, "
            f"but upscaled alpha has {len(alpha_frames)} frames in {alpha_dir}. "
            "Refusing to drop, repeat, or invent frames."
        )
    if not rgb_frames:
        raise ValidationError(f"No upscaled RGB frames in {rgb_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    total = len(rgb_frames)
    report(
        "Recombine RGBA",
        message=f"Recombining {total} frames",
        frames_total=total,
        frames_kind="completed",
    )
    for index, (rgb_path, alpha_path) in enumerate(zip(rgb_frames, alpha_frames, strict=True)):
        checkpoint()
        with Image.open(rgb_path) as rgb, Image.open(alpha_path) as alpha:
            if rgb.size != alpha.size:
                raise ValidationError(
                    f"Frame {index:06d}: RGB {rgb_path.name} is {rgb.size[0]}x{rgb.size[1]} "
                    f"but alpha {alpha_path.name} is {alpha.size[0]}x{alpha.size[1]}. "
                    "Alpha was not resized to hide the mismatch."
                )
            merged = recombine_rgba(rgb, alpha)
        destination = frame_path(output_dir, index)
        merged.save(destination, format="PNG")
        written.append(destination)
        report(
            "Recombine RGBA",
            message=f"Recombined frame {index + 1}/{total}",
            frames_done=index + 1,
            frames_total=total,
            frames_kind="completed",
            preview_path=str(destination),
        )
    return written
