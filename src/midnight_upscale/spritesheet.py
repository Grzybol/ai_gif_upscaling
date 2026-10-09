"""Spritesheet export: a fixed cell size, a JSON atlas, and automatic multi-sheet splitting.

Every cell has the same size, taken from the common canvas. Frames are never
resized or cropped individually. The sheet is only padded, never the frames.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from midnight_upscale.utils import PipelineError
from midnight_upscale.video_inspect import TIMING_JITTER_MS

PADDING_CHOICES = (0, 1, 2, 4)
MAX_SIZE_CHOICES = (2048, 4096, 8192)


@dataclass(frozen=True)
class SpritesheetSettings:
    columns: int | None = None  # None = automatic
    padding: int = 2
    max_size: int = 4096
    power_of_two: bool = False


@dataclass(frozen=True)
class SheetLayout:
    index: int
    first_frame: int
    frame_count: int
    columns: int
    rows: int
    width: int
    height: int
    cells: tuple[tuple[int, int], ...]  # top-left corner of each frame, in order


@dataclass(frozen=True)
class SpritesheetPlan:
    frame_width: int
    frame_height: int
    padding: int
    sheets: tuple[SheetLayout, ...] = field(default_factory=tuple)

    @property
    def frame_count(self) -> int:
        return sum(sheet.frame_count for sheet in self.sheets)


def next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def largest_power_of_two_at_most(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def sheet_names(stem: str, sheet_count: int) -> tuple[list[str], str]:
    """PNG names and the JSON name. One sheet has no number; several are ``_00``, ``_01``..."""

    base = f"{stem}_spritesheet"
    if sheet_count <= 1:
        return [f"{base}.png"], f"{base}.json"
    return [f"{base}_{i:02d}.png" for i in range(sheet_count)], f"{base}.json"


def plan_spritesheet(
    frame_width: int, frame_height: int, frame_count: int, settings: SpritesheetSettings
) -> SpritesheetPlan:
    """Lay out ``frame_count`` equal cells without exceeding ``settings.max_size``."""

    if frame_width < 1 or frame_height < 1:
        raise PipelineError(f"Frame size {frame_width}x{frame_height} is not valid")
    if frame_count < 1:
        raise PipelineError("A spritesheet needs at least one frame")
    pad = max(0, int(settings.padding))
    limit = int(settings.max_size)
    if settings.power_of_two:
        # The padded sheet must still fit, so only a power of two at or below the limit is usable.
        limit = largest_power_of_two_at_most(limit)
    stride_x = frame_width + pad
    stride_y = frame_height + pad
    max_columns = (limit - pad) // stride_x
    max_rows = (limit - pad) // stride_y
    if max_columns < 1 or max_rows < 1:
        raise PipelineError(
            f"A {frame_width}x{frame_height} frame with {pad}px padding does not fit in a "
            f"{limit}px texture. Choose a larger maximum texture size or reduce the frame size."
        )

    if settings.columns:
        columns = int(settings.columns)
        if columns < 1:
            raise PipelineError("Columns must be at least 1")
        if columns > max_columns:
            raise PipelineError(
                f"{columns} columns of {frame_width}px frames need more than {limit}px. "
                f"At most {max_columns} columns fit."
            )
    else:
        columns = max(1, math.ceil(math.sqrt(frame_count * stride_y / stride_x)))
        columns = min(columns, max_columns)
        if math.ceil(frame_count / columns) > max_rows:
            columns = max_columns  # several full sheets

    per_sheet = columns * max_rows
    sheets: list[SheetLayout] = []
    first = 0
    while first < frame_count:
        count = min(per_sheet, frame_count - first)
        rows = math.ceil(count / columns)
        used_columns = min(columns, count)
        width = pad + used_columns * stride_x
        height = pad + rows * stride_y
        if settings.power_of_two:
            width, height = next_power_of_two(width), next_power_of_two(height)
        cells = tuple(
            (pad + (i % columns) * stride_x, pad + (i // columns) * stride_y) for i in range(count)
        )
        sheets.append(
            SheetLayout(
                index=len(sheets),
                first_frame=first,
                frame_count=count,
                columns=columns,
                rows=rows,
                width=width,
                height=height,
                cells=cells,
            )
        )
        first += count
    return SpritesheetPlan(frame_width, frame_height, pad, tuple(sheets))


def build_metadata(
    plan: SpritesheetPlan,
    durations_ms: list[int],
    sheet_files: list[str],
    *,
    loop: bool = True,
    power_of_two: bool = False,
) -> dict[str, object]:
    if len(durations_ms) != plan.frame_count:
        raise PipelineError(
            f"{plan.frame_count} frames but {len(durations_ms)} durations for the spritesheet"
        )
    if len(sheet_files) != len(plan.sheets):
        raise PipelineError("Sheet file names do not match the sheet plan")
    total_ms = sum(durations_ms)
    fps = plan.frame_count / (total_ms / 1000.0) if total_ms > 0 else 0.0
    first = plan.sheets[0]
    frames: list[dict[str, object]] = []
    for sheet in plan.sheets:
        for offset, (x, y) in enumerate(sheet.cells):
            index = sheet.first_frame + offset
            frames.append(
                {
                    "index": index,
                    "sheet": sheet.index,
                    "x": x,
                    "y": y,
                    "w": plan.frame_width,
                    "h": plan.frame_height,
                    "duration_ms": durations_ms[index],
                }
            )
    return {
        "frame_width": plan.frame_width,
        "frame_height": plan.frame_height,
        "frame_count": plan.frame_count,
        "fps": round(fps, 4),
        "duration_ms": total_ms,
        "columns": first.columns,
        "rows": first.rows,
        "loop": loop,
        "padding": plan.padding,
        "power_of_two": power_of_two,
        # Whole-millisecond rounding of a constant rate (83/84 ms at 12 fps) is not variable timing.
        "variable_timing": max(durations_ms) - min(durations_ms) > TIMING_JITTER_MS,
        "sheets": [
            {
                "file": sheet_files[sheet.index],
                "width": sheet.width,
                "height": sheet.height,
                "columns": sheet.columns,
                "rows": sheet.rows,
                "first_frame": sheet.first_frame,
                "frame_count": sheet.frame_count,
            }
            for sheet in plan.sheets
        ],
        "frames": frames,
    }


def write_spritesheets(
    frame_paths: list[Path],
    durations_ms: list[int],
    png_paths: list[Path],
    json_path: Path,
    settings: SpritesheetSettings,
    *,
    loop: bool = True,
    on_frame: Callable[[int], None] | None = None,
) -> SpritesheetPlan:
    """Pack ``frame_paths`` (identical size RGBA PNGs) into sheets and write the JSON atlas."""

    if not frame_paths:
        raise PipelineError("No frames to pack")
    with Image.open(frame_paths[0]) as probe:
        frame_w, frame_h = probe.size
    plan = plan_spritesheet(frame_w, frame_h, len(frame_paths), settings)
    if len(png_paths) != len(plan.sheets):
        raise PipelineError(
            f"The plan needs {len(plan.sheets)} sheet file(s), got {len(png_paths)} names"
        )
    for sheet, destination in zip(plan.sheets, png_paths, strict=True):
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGBA", (sheet.width, sheet.height), (0, 0, 0, 0))
        for offset, (x, y) in enumerate(sheet.cells):
            index = sheet.first_frame + offset
            with Image.open(frame_paths[index]) as frame:
                if frame.size != (frame_w, frame_h):
                    raise PipelineError(
                        f"Frame {index} is {frame.size[0]}x{frame.size[1]}, expected "
                        f"{frame_w}x{frame_h}. Every spritesheet cell must be the same size."
                    )
                # paste without a mask copies RGBA as is; nothing is composited.
                canvas.paste(frame.convert("RGBA"), (x, y))
            if on_frame is not None:
                on_frame(index + 1)
        canvas.save(destination, format="PNG")
    metadata = build_metadata(
        plan,
        durations_ms,
        [path.name for path in png_paths],
        loop=loop,
        power_of_two=settings.power_of_two,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return plan
