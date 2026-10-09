"""Shared errors, paths, frame lists, and configuration."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

JOB_SUBDIRS = (
    "source_rgba",
    "rgb",
    "alpha",
    "upscaled_rgb",
    "upscaled_alpha",
    "final_rgba",
    "output",
    "batches",
)

# GIF delay 0 means "as fast as possible". Encoders need a real duration.
# 10 ms matches the usual browser floor for a zero GIF delay.
ZERO_DELAY_ENCODED_MS = 10

_FRAME_NAME = re.compile(r"^(\d+)\.png$", re.IGNORECASE)
_NATURAL_SPLIT = re.compile(r"(\d+)")


class PipelineError(Exception):
    """A failure the user can act on. The message is printed as-is."""


class ValidationError(PipelineError):
    """Frame counts, dimensions, alpha, or timing failed a check."""


class WorkflowConfigError(PipelineError):
    """The ComfyUI workflow or its config mapping is not usable."""


class ComfyError(PipelineError):
    """ComfyUI could not be reached or rejected the prompt."""


DEFAULT_CONFIG: dict[str, Any] = {
    "comfyui": {
        "url": "http://127.0.0.1:8188",
        "timeout_sec": 14400,
        "poll_interval_sec": 2.0,
    },
    "seedvr2": {
        "workflow": "workflows/seedvr2.example.json",
        "chunking": "windows",
        "input": {"node_id": "1", "field": "directory", "mode": "directory"},
        "model": {"node_id": "", "field": "model", "value": ""},
        "scale": {"node_id": "4", "field": "resolution", "mode": "shortest_edge"},
        "batch_size": {"node_id": "4", "field": "batch_size"},
        "temporal_overlap": {"node_id": "4", "field": "temporal_overlap"},
        "output": {"node_id": "5", "field": "output_path", "mode": "directory"},
    },
    "frame_upscale": {
        "workflow": "",
        "input": {"node_id": "", "field": "image", "mode": "upload_image"},
        "output": {"node_id": "", "field": "filename_prefix", "mode": "history"},
    },
    "encode": {
        "crf": 18,
        "webm_pix_fmt": "yuva420p",
    },
}


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def require_binary(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise PipelineError(
            f"Required program {name!r} was not found on PATH. "
            "Install FFmpeg, including ffprobe, and open a new terminal so PATH updates. "
            "On Windows: winget install Gyan.FFmpeg"
        )
    return found


def natural_sort_key(path: Path) -> list[tuple[int, int | str]]:
    key: list[tuple[int, int | str]] = []
    for part in _NATURAL_SPLIT.split(path.name):
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def list_indexed_frames(directory: Path) -> list[Path]:
    """Return ``000000.png``, ``000001.png``, ... or raise if the sequence has gaps."""

    if not directory.is_dir():
        raise ValidationError(f"Missing frame directory: {directory}")

    found: dict[int, Path] = {}
    unexpected: list[str] = []
    for path in directory.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        match = _FRAME_NAME.match(path.name)
        if match is None:
            if path.suffix.lower() == ".png":
                unexpected.append(path.name)
            continue
        index = int(match.group(1))
        if index in found:
            raise ValidationError(f"Duplicate frame index {index} in {directory}")
        found[index] = path

    if unexpected:
        names = ", ".join(unexpected[:8])
        raise ValidationError(
            f"Unexpected PNG names in {directory}: {names}. "
            "Expected 000000.png, 000001.png, and so on."
        )
    if not found:
        return []

    indexes = sorted(found)
    if indexes != list(range(len(indexes))):
        preview = ", ".join(str(i) for i in indexes[:12])
        raise ValidationError(
            f"Frame sequence in {directory} is not contiguous from 000000.png "
            f"(found indexes {preview}). Missing frames are not replaced or reordered."
        )
    return [found[index] for index in indexes]


def frame_path(directory: Path, index: int) -> Path:
    return directory / f"{index:06d}.png"


def ensure_job_dirs(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    for name in JOB_SUBDIRS:
        (workdir / name).mkdir(exist_ok=True)


def directory_occupied(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def next_available_directory(path: Path) -> Path:
    """Return ``path``, or ``path_v1``, ``path_v2``, ... when it already has files."""

    if not directory_occupied(path):
        return path
    number = 1
    while directory_occupied(path.parent / f"{path.name}_v{number}"):
        number += 1
    return path.parent / f"{path.name}_v{number}"


def next_available_file(path: Path) -> Path:
    """Return ``path``, or ``stem_v1.suffix``, ``stem_v2.suffix``, ... when it exists."""

    if not path.exists():
        return path
    number = 1
    while True:
        candidate = path.with_name(f"{path.stem}_v{number}{path.suffix}")
        if not candidate.exists():
            return candidate
        number += 1


def reset_workdir(workdir: Path, *, overwrite: bool, source: Path | None = None) -> Path:
    workdir = workdir.resolve()
    if workdir == Path(workdir.anchor) or workdir.parent == workdir:
        raise PipelineError(f"Refusing to use {workdir} as a work directory")
    if source is not None:
        source = source.resolve()
        if workdir == source or workdir in source.parents or source in workdir.parents:
            raise PipelineError(
                "Work directory must not contain the source file, and the source must not "
                f"live inside the work directory. source={source} workdir={workdir}"
            )
    if directory_occupied(workdir):
        if overwrite:
            shutil.rmtree(workdir)
        else:
            versioned = next_available_directory(workdir)
            logger.info("Work directory %s already exists. Using %s.", workdir, versioned)
            workdir = versioned
    ensure_job_dirs(workdir)
    return workdir


def clear_pngs(directory: Path, *, overwrite: bool) -> None:
    pngs = [path for path in directory.glob("*.png") if path.is_file()]
    if pngs and not overwrite:
        raise PipelineError(
            f"{directory} already contains {len(pngs)} PNG file(s). "
            "Pass --overwrite to replace them."
        )
    for path in pngs:
        path.unlink()


def encoded_duration_ms(duration_ms: int) -> int:
    if duration_ms < 0:
        raise ValidationError(f"Negative frame duration {duration_ms} ms")
    if duration_ms == 0:
        return ZERO_DELAY_ENCODED_MS
    return duration_ms


def expected_encoded_duration_ms(durations_ms: list[int]) -> int:
    return sum(encoded_duration_ms(duration) for duration in durations_ms)


def duration_tolerance_sec(durations_ms: list[int]) -> float:
    """Allow a small encoder rounding error, not a timing rewrite."""

    encoded = [encoded_duration_ms(duration) for duration in durations_ms] or [
        ZERO_DELAY_ENCODED_MS
    ]
    ordered = sorted(encoded)
    median_sec = ordered[len(ordered) // 2] / 1000.0
    total_sec = sum(encoded) / 1000.0
    return max(0.02, min(0.5, max(2 * median_sec, 0.02 * total_sec)))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None) -> tuple[dict[str, Any], Path | None]:
    if path is None:
        default = Path("config.yaml")
        if default.is_file():
            path = default
        else:
            return deep_merge(DEFAULT_CONFIG, {}), None
    if not path.is_file():
        raise PipelineError(f"Config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise PipelineError(f"Config {path} must be a YAML mapping")
    return deep_merge(DEFAULT_CONFIG, loaded), path.resolve()


def resolve_existing_file(value: str, config_file: Path | None) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    options: list[Path] = []
    if config_file is not None and not candidate.is_absolute():
        options.append(config_file.parent / candidate)
    if not candidate.is_absolute():
        options.append(Path.cwd() / candidate)
    for option in options:
        if option.is_file():
            return option.resolve()
    looked = ", ".join(str(option) for option in options) or str(candidate)
    raise PipelineError(f"File not found: {value} (looked in {looked})")


def as_node_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def tail_text(text: str, limit: int = 2000) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]


def find_pngs(directory: Path) -> list[Path]:
    """PNG files in ``directory``, or in its single PNG-bearing subdirectory."""

    if not directory.is_dir():
        raise ValidationError(f"Output directory does not exist: {directory}")
    direct = sorted(
        (path for path in directory.glob("*.png") if path.is_file()),
        key=natural_sort_key,
    )
    if direct:
        return direct
    nested: list[tuple[Path, list[Path]]] = []
    for folder in sorted(
        (path for path in directory.iterdir() if path.is_dir()), key=lambda p: p.name
    ):
        found = sorted(
            (path for path in folder.glob("*.png") if path.is_file()), key=natural_sort_key
        )
        if found:
            nested.append((folder, found))
    if len(nested) == 1:
        logger.info("Reading frames from %s", nested[0][0])
        return nested[0][1]
    if len(nested) > 1:
        names = ", ".join(str(folder) for folder, _ in nested)
        raise ValidationError(
            f"Multiple subfolders in {directory} contain PNGs ({names}). "
            "The output node must write one ordered sequence."
        )
    return []
