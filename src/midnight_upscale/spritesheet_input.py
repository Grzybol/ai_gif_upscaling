"""Read spritesheet ZIPs exported by Upscale or Video Converter as RGBA frames."""

from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError

from midnight_upscale.utils import PipelineError


class SpritesheetArchive:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        try:
            self.archive = ZipFile(path)
            names = self.archive.namelist()
            manifests = [name for name in names if name.lower().endswith(".json")]
            if len(manifests) != 1:
                raise PipelineError("Spritesheet ZIP must contain exactly one JSON metadata file.")
            manifest = manifests[0]
            if self.archive.getinfo(manifest).file_size > 16_000_000:
                raise PipelineError("Spritesheet metadata is too large.")
            self.metadata = json.loads(self.archive.read(manifest))
            if not isinstance(self.metadata, dict):
                raise PipelineError("Spritesheet metadata must be a JSON object.")
            records = self.metadata.get("frames")
            if not isinstance(records, list) or not records:
                raise PipelineError("Spritesheet metadata contains no frames.")
            if not all(isinstance(item, dict) for item in records):
                raise PipelineError("Spritesheet frame entries must be objects.")
            self.frames = sorted(records, key=lambda item: int(item["index"]))
            if [int(item["index"]) for item in self.frames] != list(range(len(self.frames))):
                raise PipelineError("Spritesheet frame indices must be consecutive from zero.")
            self.durations_ms = [int(item["duration_ms"]) for item in self.frames]
            if any(duration <= 0 for duration in self.durations_ms):
                raise PipelineError("Spritesheet frame durations must be positive.")
            self.width, self.height = int(self.frames[0]["w"]), int(self.frames[0]["h"])
            if self.width <= 0 or self.height <= 0:
                raise PipelineError("Spritesheet frames must have positive dimensions.")
            for item in self.frames:
                if int(item["w"]) != self.width or int(item["h"]) != self.height:
                    raise PipelineError("Spritesheet frames must share one size.")
                name = self._sheet_name(item)
                if name not in names or not name.lower().endswith(".png"):
                    raise PipelineError(f"Spritesheet image missing from ZIP: {name}")
            self._cached_name: str | None = None
            self._cached_image: Image.Image | None = None
        except (BadZipFile, OSError, KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            if hasattr(self, "archive"):
                self.archive.close()
            raise PipelineError(f"Invalid spritesheet ZIP: {exc}") from exc
        except BaseException:
            if hasattr(self, "archive"):
                self.archive.close()
            raise

    def _sheet_name(self, item: dict) -> str:
        sheet = item["sheet"]
        if isinstance(sheet, int):
            descriptor = self.metadata["sheets"][sheet]
            return str(descriptor["file"] if isinstance(descriptor, dict) else descriptor)
        return str(sheet)

    def frame(self, index: int) -> Image.Image:
        if not 0 <= index < len(self.frames):
            raise PipelineError(f"Spritesheet frame {index} is out of range.")
        item = self.frames[index]
        name = self._sheet_name(item)
        if name != self._cached_name:
            if self._cached_image is not None:
                self._cached_image.close()
            member = self.archive.getinfo(name)
            if member.file_size > 512_000_000:
                raise PipelineError(f"Spritesheet image is too large: {name}")
            try:
                image = Image.open(io.BytesIO(self.archive.read(name)))
                image.load()
                self._cached_image = image.convert("RGBA")
                image.close()
            except (OSError, BadZipFile, UnidentifiedImageError) as exc:
                raise PipelineError(f"Could not read spritesheet image {name}: {exc}") from exc
            self._cached_name = name
        x, y = int(item["x"]), int(item["y"])
        if x < 0 or y < 0 or x + self.width > self._cached_image.width or y + self.height > self._cached_image.height:
            raise PipelineError(f"Frame {index} lies outside spritesheet {name}.")
        return self._cached_image.crop((x, y, x + self.width, y + self.height))

    def has_alpha(self) -> bool:
        for index in sorted({0, len(self.frames) // 2, len(self.frames) - 1}):
            if np.any(np.asarray(self.frame(index))[:, :, 3] < 255):
                return True
        return False

    def close(self) -> None:
        if getattr(self, "_cached_image", None) is not None:
            self._cached_image.close()
            self._cached_image = None
        self.archive.close()

    def __enter__(self) -> SpritesheetArchive:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def inspect_spritesheet(path: Path):
    """Produce the same timing/alpha description used for decoded videos."""
    from midnight_upscale.video_inspect import TIMING_JITTER_MS, VideoInfo

    with SpritesheetArchive(path) as sheet:
        duration_ms = sum(sheet.durations_ms)
        elapsed = 0
        times = []
        for delay in sheet.durations_ms:
            times.append(elapsed / 1000)
            elapsed += delay
        return VideoInfo(
            path=Path(path), width=sheet.width, height=sheet.height,
            codec="png", pix_fmt="rgba", container="spritesheet_zip",
            duration_sec=duration_ms / 1000,
            fps=len(sheet.frames) * 1000 / duration_ms,
            frame_count=len(sheet.frames), frame_times=times,
            durations_ms=sheet.durations_ms,
            variable_timing=max(sheet.durations_ms) - min(sheet.durations_ms) > TIMING_JITTER_MS,
            alpha_declared=True, has_alpha=sheet.has_alpha(),
        )
