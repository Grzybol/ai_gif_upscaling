"""Write processed RGBA frames as WebM alpha, APNG, GIF, PNG frames, or a spritesheet.

Encoding and probing reuse ``encode.py``. Frames are lossless PNGs until the final
write, so no lossy video format is ever transcoded into another.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from midnight_upscale.encode import (
    _concat_prefix,
    _drop_concat_sentinel,
    encode_sequence,
    write_concat_file,
)
from midnight_upscale.progress import checkpoint, report
from midnight_upscale.spritesheet import (
    SpritesheetPlan,
    SpritesheetSettings,
    plan_spritesheet,
    sheet_names,
    write_spritesheets,
)
from midnight_upscale.utils import (
    PipelineError,
    ValidationError,
    duration_tolerance_sec,
    expected_encoded_duration_ms,
    frame_path,
    require_binary,
    tail_text,
)
from midnight_upscale.validate import assert_duration_close
from midnight_upscale.video_decode import sample_has_alpha
from midnight_upscale.video_inspect import inspect_video

WEBM = "webm"
APNG = "apng"
GIF = "gif"
PNG_SEQUENCE = "png_sequence"
SPRITESHEET = "spritesheet"

FORMAT_CHOICES = {
    "Transparent WebM": WEBM,
    "APNG": APNG,
    "GIF Preview": GIF,
    "PNG Frame Sequence": PNG_SEQUENCE,
    "PNG Spritesheet": SPRITESHEET,
}
FORMAT_NAMES = {value: key for key, value in FORMAT_CHOICES.items()}

FORMAT_HELP = (
    "**WebM Alpha** — recommended for runtime video animation.  \n"
    "**Spritesheet** — recommended for engine sprite animation.  \n"
    "**APNG** — lossless transparent animation.  \n"
    "**GIF** — preview / compatibility only; transparency quality is limited.  \n"
    "**PNG frames** — one lossless RGBA file per frame."
)
GIF_WARNING = (
    "GIF does not preserve high-quality semi-transparent alpha. Each pixel is either "
    "opaque or transparent. Use WebM Alpha, APNG, or a spritesheet for production."
)


def output_paths(
    stem: str, formats: tuple[str, ...], directory: Path, sheet_count: int = 1
) -> dict[str, list[Path]]:
    """Deterministic output paths per format. The spritesheet list ends with JSON, then the zip."""

    paths: dict[str, list[Path]] = {}
    for fmt in formats:
        if fmt == WEBM:
            paths[fmt] = [directory / f"{stem}_transparent.webm"]
        elif fmt == APNG:
            paths[fmt] = [directory / f"{stem}_transparent.apng"]
        elif fmt == GIF:
            paths[fmt] = [directory / f"{stem}_preview.gif"]
        elif fmt == PNG_SEQUENCE:
            paths[fmt] = [directory / f"{stem}_frames"]
        elif fmt == SPRITESHEET:
            pngs, json_name = sheet_names(stem, sheet_count)
            zip_name = f"{stem}_spritesheet.zip"
            paths[fmt] = [directory / n for n in pngs] + [
                directory / json_name,
                directory / zip_name,
            ]
        else:
            raise PipelineError(f"Unknown output format {fmt!r}")
    return paths


def resolve_output_stem(
    stem: str,
    formats: tuple[str, ...],
    directory: Path,
    sheet_count: int,
    *,
    overwrite: bool,
) -> str:
    """``stem``, or ``stem_v1``, ``stem_v2`` ... when any target exists and overwrite is off."""

    def taken(name: str) -> bool:
        for group in output_paths(name, formats, directory, sheet_count).values():
            if any(path.exists() for path in group):
                return True
        return False

    if overwrite or not taken(stem):
        return stem
    number = 1
    while taken(f"{stem}_v{number}"):
        number += 1
    return f"{stem}_v{number}"


def remove_outputs(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def zip_spritesheet(files: list[Path], destination: Path) -> Path:
    """Bundle the sheet PNGs and the JSON atlas into one download."""

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    return destination


def export_png_sequence(frame_paths: list[Path], directory: Path, stem: str) -> list[Path]:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    written: list[Path] = []
    for index, source in enumerate(frame_paths):
        checkpoint()
        target = directory / f"{stem}_{index:06d}.png"
        shutil.copyfile(source, target)
        written.append(target)
        report(
            "Export",
            frames_done=index + 1,
            frames_total=len(frame_paths),
            frames_kind="completed",
            message=f"Writing PNG frame {index + 1}",
        )
    return written


def export_spritesheet(
    frame_paths: list[Path],
    durations_ms: list[int],
    directory: Path,
    stem: str,
    settings: SpritesheetSettings,
) -> tuple[list[Path], SpritesheetPlan]:
    """Write the sheet PNGs and JSON. Returns every file written, JSON last."""

    with Image.open(frame_paths[0]) as probe:
        width, height = probe.size
    plan = plan_spritesheet(width, height, len(frame_paths), settings)
    names, json_name = sheet_names(stem, len(plan.sheets))
    pngs = [directory / name for name in names]
    json_path = directory / json_name

    def on_frame(done: int) -> None:
        checkpoint()
        report(
            "Export",
            frames_done=done,
            frames_total=len(frame_paths),
            frames_kind="completed",
            message=f"Packing spritesheet frame {done}",
        )

    write_spritesheets(frame_paths, durations_ms, pngs, json_path, settings, on_frame=on_frame)
    return [*pngs, json_path], plan


def export_video(
    frame_paths: list[Path], durations_ms: list[int], destination: Path, fmt: str, crf: int
) -> Path:
    """WebM VP9 with alpha, APNG, or GIF from the PNG frames and their real durations."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    encode_sequence(
        frame_paths,
        durations_ms,
        destination,
        fmt=fmt,
        loop=0,
        crf=crf,
        webm_pix_fmt="yuva420p",
    )
    for helper in ("concat.txt", "palette.png"):
        leftover = frame_paths[0].parent / helper
        if leftover.exists():
            leftover.unlink()
    return destination


@dataclass
class ValidationReport:
    path: Path
    lines: list[str]
    warnings: list[str]


def validate_video_output(
    path: Path,
    fmt: str,
    *,
    frame_count: int,
    size: tuple[int, int],
    durations_ms: list[int],
    expect_transparency: bool,
) -> ValidationReport:
    """Check an encoded file with ffprobe. Raises ``ValidationError`` when it is wrong.

    Frame count and timing come from packet timestamps, which every container here
    reports (APNG has no stream duration).
    """

    info = inspect_video(path, check_alpha=False, any_extension=True)
    lines = [
        f"{path.name}: {info.width}x{info.height}, {info.frame_count} frames, "
        f"{info.duration_sec:.3f} s, pix_fmt {info.pix_fmt or 'unknown'}"
    ]
    warnings: list[str] = []
    if (info.width, info.height) != size:
        raise ValidationError(
            f"{path.name} is {info.width}x{info.height}, expected {size[0]}x{size[1]}"
        )
    assert_duration_close(
        info.duration_sec,
        expected_encoded_duration_ms(durations_ms) / 1000.0,
        duration_tolerance_sec(durations_ms),
    )
    if fmt == GIF:
        if info.frame_count != frame_count:
            warnings.append(
                f"{path.name} has {info.frame_count} frames; the source had {frame_count}. "
                "GIF may merge identical frames."
            )
        warnings.append(GIF_WARNING)
        return ValidationReport(path, lines, warnings)

    if info.frame_count != frame_count:
        raise ValidationError(f"{path.name} has {info.frame_count} frames, expected {frame_count}")
    if not info.alpha_declared:
        raise ValidationError(
            f"{path.name} pixel format is {info.pix_fmt or 'unknown'}: the alpha plane was dropped."
        )
    if expect_transparency:
        if not sample_has_alpha(info):
            raise ValidationError(
                f"{path.name} declares alpha but every sampled pixel is opaque. "
                "The transparency was flattened."
            )
        lines.append(f"{path.name}: decoded alpha contains transparent pixels")
    return ValidationReport(path, lines, warnings)


def validate_spritesheet_output(
    pngs: list[Path], json_path: Path, plan: SpritesheetPlan, max_size: int
) -> list[str]:
    lines: list[str] = []
    for sheet, path in zip(plan.sheets, pngs, strict=True):
        with Image.open(path) as image:
            if image.mode != "RGBA":
                raise ValidationError(f"{path.name} is {image.mode}, expected RGBA")
            if image.size != (sheet.width, sheet.height):
                raise ValidationError(f"{path.name} is {image.size}, expected a planned size")
            if max(image.size) > max_size:
                raise ValidationError(f"{path.name} exceeds the {max_size}px texture limit")
        lines.append(f"{path.name}: {sheet.width}x{sheet.height}, {sheet.frame_count} frames")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data["frame_count"] != len(data["frames"]) or data["frame_count"] != plan.frame_count:
        raise ValidationError(f"{json_path.name} frame list does not match the frame count")
    lines.append(f"{json_path.name}: {data['frame_count']} frames, {len(data['sheets'])} sheet(s)")
    return lines


def checkerboard(width: int, height: int, cell: int = 16) -> np.ndarray:
    ys, xs = np.indices((height, width))
    light = ((xs // cell) + (ys // cell)) % 2 == 0
    board = np.empty((height, width, 4), dtype=np.uint8)
    board[light] = (232, 232, 232, 255)
    board[~light] = (188, 188, 188, 255)
    return board


def composite_on_background(rgba: np.ndarray, name: str) -> Image.Image:
    """RGBA array to an RGB image on a checkerboard, black, or white background."""

    image = Image.fromarray(np.ascontiguousarray(rgba))
    if name == "black":
        base = Image.new("RGBA", image.size, (0, 0, 0, 255))
    elif name == "white":
        base = Image.new("RGBA", image.size, (255, 255, 255, 255))
    else:
        base = Image.fromarray(checkerboard(image.width, image.height))
    base.alpha_composite(image)
    return base.convert("RGB")


def make_browser_preview(
    frame_paths: list[Path],
    durations_ms: list[int],
    destination: Path,
    scratch: Path,
    background: str,
    on_frame: Callable[[int], None] | None = None,
) -> Path | None:
    """H.264 clip on a flat background so a browser can play it. Not a production file."""

    scratch.mkdir(parents=True, exist_ok=True)
    flat: list[Path] = []
    try:
        for index, source in enumerate(frame_paths):
            with Image.open(source) as image:
                rgba = np.asarray(image.convert("RGBA"))
            target = frame_path(scratch, index)
            composite_on_background(rgba, background).save(target, compress_level=1)
            flat.append(target)
            if on_frame is not None:
                on_frame(index + 1)
        concat = scratch / "concat.txt"
        write_concat_file(flat, durations_ms, concat)
        ffmpeg = require_binary("ffmpeg")
        command = _concat_prefix(ffmpeg, concat)
        sentinel = _drop_concat_sentinel(len(flat))
        command.extend(
            [
                "-vf",
                f"{sentinel},pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-fps_mode",
                "vfr",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-t",
                f"{expected_encoded_duration_ms(durations_ms) / 1000.0:.6f}",
                str(destination),
            ]
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise PipelineError(f"Browser preview failed: {tail_text(completed.stderr or '')}")
    except (PipelineError, OSError):
        return None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return destination if destination.is_file() else None
