"""Job and inspection records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from midnight_upscale.utils import PipelineError


def format_loop(loop: int | None) -> str:
    if loop is None:
        return "absent (play once)"
    if loop == 0:
        return "0 (infinite)"
    return str(loop)


@dataclass
class GifInspection:
    source: str
    width: int
    height: int
    frame_count: int
    frame_durations_ms: list[int]
    total_duration_ms: int
    gif_loop_count: int | None
    durations_constant: bool
    estimated_fps: float | None
    has_transparency: bool
    has_semitransparency: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_report(self) -> str:
        fps = "n/a" if self.estimated_fps is None else f"{self.estimated_fps:.3f}"
        timing = "constant" if self.durations_constant else "variable"
        durations = ", ".join(str(duration) for duration in self.frame_durations_ms)
        return (
            "SOURCE\n"
            f"{self.source}\n"
            f"{self.width}x{self.height}\n"
            f"{self.frame_count} frames\n"
            f"{self.total_duration_ms / 1000:.3f} s\n"
            f"durations: {timing}\n"
            f"frame durations (ms): {durations}\n"
            f"estimated fps: {fps}\n"
            f"loop count: {format_loop(self.gif_loop_count)}\n"
            f"transparency: {'yes' if self.has_transparency else 'no'}\n"
            f"semitransparency: {'yes' if self.has_semitransparency else 'no'}\n"
        )


@dataclass
class JobMetadata:
    source: str
    original_width: int
    original_height: int
    frame_count: int
    frame_durations_ms: list[int]
    total_duration_ms: int
    gif_loop_count: int | None
    durations_constant: bool
    estimated_fps: float | None
    has_transparency: bool
    has_semitransparency: bool
    requested_scale: int
    target_width: int
    target_height: int
    alpha_mode: str
    edge_cleanup: str
    interpolate: str
    backend: str | None = None
    workdir: str = ""
    duration_warnings: list[str] | None = None
    seedvr2_resolution: int | None = None
    chunking: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> JobMetadata:
        if not path.is_file():
            raise PipelineError(f"Job metadata not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise PipelineError(f"Job metadata is not an object: {path}")
        known = {item.name for item in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in known}
        return cls(**filtered)
