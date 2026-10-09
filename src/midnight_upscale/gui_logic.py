"""Job settings and orchestration for the local GUI.

The Gradio page calls this module. This module calls the same prepare, upscale,
and finalize functions as the CLI. It does not decode GIFs or talk to SeedVR2
on its own.
"""

from __future__ import annotations

import copy
import logging
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from PIL import Image

from midnight_upscale.alpha import ALPHA_MODES, EDGE_CLEANUP_CHOICES
from midnight_upscale.comfy import (
    ComfyClient,
    assert_nodes_available,
    assert_workflow_configured,
    known_seedvr2_status,
    load_workflow,
    schema_has_input,
)
from midnight_upscale.inspect import inspect_gif
from midnight_upscale.interpolate import resolve_interpolation
from midnight_upscale.models import GifInspection, JobMetadata, format_loop
from midnight_upscale.pipeline import finalize_job, prepare_asset, upscale_job
from midnight_upscale.progress import (
    JobCancelled,
    ProgressBus,
    bind_bus,
    format_status,
    reset_bus,
    stamp_log,
)
from midnight_upscale.seedvr2 import validate_seedvr2_batch_size, validate_temporal_overlap
from midnight_upscale.seedvr2_native import assess_installation, format_gui_status
from midnight_upscale.utils import (
    ComfyError,
    PipelineError,
    WorkflowConfigError,
    expected_encoded_duration_ms,
    frame_path,
    load_config,
    require_binary,
    resolve_existing_file,
)

logger = logging.getLogger(__name__)

GUI_BATCH_SIZES = (5, 9, 13)
GUI_SCALES = (2, 3, 4)
GUI_OVERLAPS = (0, 1, 2, 3)
OUTPUT_FORMATS = ("webm", "apng", "gif")

BACKEND_CHOICES = {
    "Native ComfyUI": "seedvr2-native",
    "Numz custom node": "seedvr2",
    "frame-upscale fallback": "frame-upscale",
}
SCALE_CHOICES = {"2x": 2, "3x": 3, "4x": 4}
ALPHA_CHOICES = {"Lanczos": "lanczos", "Bicubic": "bicubic", "Nearest": "nearest"}
EDGE_CHOICES = {"Auto": "auto", "Off": "off", "Simple": "simple"}
FORMAT_CHOICES = {"WebM Alpha": "webm", "APNG": "apng", "GIF Preview": "gif"}
RESULT_LABELS = {"webm": "WebM VP9 Alpha", "apng": "APNG", "gif": "GIF Preview"}

STAGE_ORDER = (
    "Inspecting input",
    "Decoding GIF",
    "Preparing RGB / alpha",
    "Alpha upscaling",
    "SeedVR2 upscaling",
    "Recombining RGBA",
    "Encoding output",
    "Validating output",
    "Finished",
)

OVERLAP_UNSUPPORTED = "Temporal overlap not supported by installed SeedVR2 node"

_PLAN_LINE = re.compile(r"SeedVR2: (\d+) RGB frames, (\d+) ComfyUI prompt")
_BATCH_LINE = re.compile(r"ComfyUI batch (\d+): frames (\d+)-(\d+)")

JOB_LOCK = threading.Lock()
ACTIVE_CANCEL = threading.Event()
ACTIVE_BUS: ProgressBus | None = None
PREVIEW_BACKGROUND = "checkerboard"

EventCallback = Callable[[str, str, str], None]


class ComfyConnection(Protocol):
    def object_info(self) -> dict[str, Any]:
        """Return ComfyUI ``/object_info``."""

    def close(self) -> None:
        """Close the client."""


@dataclass
class GuiJobConfig:
    source: Path
    backend: str
    scale: int
    batch_size: int
    temporal_overlap: int
    alpha_mode: str
    edge_cleanup: str
    output_format: str
    keep_workdir: bool
    comfy_url: str
    workflow_path: str
    output_dir: Path
    overwrite: bool
    verbose: bool
    interpolate: str = "none"
    temporal_mode: str = "auto"


@dataclass
class PreflightResult:
    config_path: Path
    temporal_overlap_supported: bool | None


@dataclass
class ComfyStatus:
    online: bool
    summary: str
    seedvr2_available: bool
    node_lines: list[str]
    overlap_supported: bool | None
    overlap_message: str
    native_available: bool = False
    selected_label: str = ""
    ready: bool = False


@dataclass
class JobResult:
    source_name: str
    output_path: Path
    preview_path: Path | None
    workdir: Path | None
    comparison_source: Path | None
    comparison_upscaled: Path | None
    frame_count: int
    middle_frame: int
    result_text: str
    log_lines: list[str] = field(default_factory=list)


@dataclass
class JobEvent:
    stage: str = ""
    detail: str = ""
    log_line: str = ""
    error: str = ""
    finished: bool = False
    result: JobResult | None = None
    queue: list[tuple[str, str]] | None = None
    panel: str = ""
    preview_path: str = ""
    caption: str = ""


def gui_config_from_form(
    source: Path,
    *,
    backend_label: str,
    scale_label: str,
    batch_size: int,
    temporal_overlap: int,
    alpha_label: str,
    edge_label: str,
    format_label: str,
    keep_workdir: bool,
    comfy_url: str,
    workflow_path: str,
    output_dir: str,
    overwrite: bool,
    verbose: bool,
    interpolate_label: str,
    temporal_mode_label: str = "Auto",
) -> GuiJobConfig:
    """Map the controls on the page onto the values the pipeline already accepts."""

    if backend_label not in BACKEND_CHOICES:
        raise PipelineError(f"Unknown backend {backend_label!r}.")
    if scale_label not in SCALE_CHOICES:
        raise PipelineError("Scale must be 2x, 3x, or 4x.")
    if alpha_label not in ALPHA_CHOICES:
        raise PipelineError("Alpha resize must be Lanczos, Bicubic, or Nearest.")
    if edge_label not in EDGE_CHOICES:
        raise PipelineError("Edge cleanup must be Auto, Off, or Simple.")
    if format_label not in FORMAT_CHOICES:
        raise PipelineError("Output format must be WebM Alpha, APNG, or GIF Preview.")
    temporal_modes = {"Auto": "auto", "Unchunked": "unchunked", "Chunked": "chunked"}
    if temporal_mode_label not in temporal_modes:
        raise PipelineError("Temporal processing must be Auto, Unchunked, or Chunked.")
    interpolate = "none" if interpolate_label.strip().lower() == "none" else interpolate_label
    directory = Path(output_dir.strip() or "output")
    return GuiJobConfig(
        source=Path(source),
        backend=BACKEND_CHOICES[backend_label],
        scale=SCALE_CHOICES[scale_label],
        batch_size=int(batch_size),
        temporal_overlap=int(temporal_overlap),
        alpha_mode=ALPHA_CHOICES[alpha_label],
        edge_cleanup=EDGE_CHOICES[edge_label],
        output_format=FORMAT_CHOICES[format_label],
        keep_workdir=bool(keep_workdir),
        comfy_url=(comfy_url or "http://127.0.0.1:8188").strip(),
        workflow_path=(workflow_path or "").strip(),
        output_dir=directory,
        overwrite=bool(overwrite),
        verbose=bool(verbose),
        interpolate=interpolate,
        temporal_mode=temporal_modes[temporal_mode_label],
    )


def validate_gui_config(config: GuiJobConfig) -> None:
    """Reject settings the page should never send into the pipeline."""

    resolve_interpolation(config.interpolate)
    if config.backend not in {"seedvr2", "seedvr2-native", "frame-upscale"}:
        raise PipelineError("Backend must be Native ComfyUI, Numz custom node, or frame-upscale.")
    if config.scale not in GUI_SCALES:
        raise PipelineError("Scale must be 2x, 3x, or 4x.")
    if config.alpha_mode not in ALPHA_MODES:
        raise PipelineError("Alpha resize must be Lanczos, Bicubic, or Nearest.")
    if config.edge_cleanup not in EDGE_CLEANUP_CHOICES:
        raise PipelineError("Edge cleanup must be Auto, Off, or Simple.")
    if config.output_format not in OUTPUT_FORMATS:
        raise PipelineError("Output format must be WebM Alpha, APNG, or GIF Preview.")
    if config.temporal_overlap not in GUI_OVERLAPS:
        raise PipelineError("Temporal overlap must be 0, 1, 2, or 3.")
    if config.temporal_mode not in {"auto", "unchunked", "chunked"}:
        raise PipelineError("Temporal processing must be Auto, Unchunked, or Chunked.")
    if config.backend == "seedvr2":
        validate_seedvr2_batch_size(config.batch_size)
        if config.batch_size not in GUI_BATCH_SIZES:
            raise PipelineError("Choose a batch size of 5, 9, or 13.")
        validate_temporal_overlap(config.temporal_overlap, config.batch_size)
    if not config.comfy_url.startswith(("http://", "https://")):
        raise PipelineError(
            f"ComfyUI URL {config.comfy_url!r} is not an http(s) URL. "
            "Example: http://127.0.0.1:8188"
        )


def output_filename(source: Path, scale: int, backend: str, output_format: str) -> str:
    suffix = {"webm": ".webm", "apng": ".apng", "gif": ".gif"}[output_format]
    tags = {
        "seedvr2": "seedvr2",
        "seedvr2-native": "seedvr2-native",
        "frame-upscale": "frame-upscale",
    }
    tag = tags.get(backend, backend)
    return f"{source.stem}_{scale}x_{tag}{suffix}"


def output_file(config: GuiJobConfig) -> Path:
    return config.output_dir / output_filename(
        config.source, config.scale, config.backend, config.output_format
    )


def workdir_for(config: GuiJobConfig) -> Path:
    return Path("work") / config.source.stem


def pipeline_arguments(config: GuiJobConfig, config_path: Path) -> dict[str, Any]:
    """The exact arguments passed through to prepare, upscale, and finalize."""

    return {
        "prepare": {
            "scale": config.scale,
            "workdir": workdir_for(config),
            "alpha_mode": config.alpha_mode,
            "edge_cleanup": config.edge_cleanup,
            "interpolate": config.interpolate,
            "overwrite": config.overwrite,
        },
        "upscale": {
            "backend": config.backend,
            "comfy_url": config.comfy_url,
            "batch_size": config.batch_size,
            "config_path": config_path,
            "timeout_sec": None,
            "overwrite": config.overwrite,
            "temporal_overlap": config.temporal_overlap,
            "temporal_mode": config.temporal_mode,
        },
        "finalize": {
            "fmt": config.output_format,
            "output": output_file(config),
            "overwrite": config.overwrite,
            "crf": None,
            "webm_pix_fmt": None,
            "config_path": config_path,
        },
        "keep_workdir": config.keep_workdir,
    }


def format_source_card(info: GifInspection) -> str:
    """Short source panel. Inspection itself stays in ``inspect_gif``."""

    fps = _format_fps(info.estimated_fps)
    timing = "NO" if info.durations_constant else "YES"
    transparency = "yes" if info.has_transparency else "no"
    return "\n".join(
        [
            "Source",
            "------",
            Path(info.source).name,
            f"{info.width} × {info.height}",
            f"{info.frame_count} frames",
            f"{info.total_duration_ms / 1000:.2f} sec",
            fps,
            f"Loop: {format_loop(info.gif_loop_count)}",
            f"Transparency: {transparency}",
            f"Variable timing: {timing}",
        ]
    )


def format_result_card(metadata: JobMetadata, destination: Path, output_format: str) -> str:
    seconds = expected_encoded_duration_ms(metadata.frame_durations_ms) / 1000.0
    return "\n".join(
        [
            "RESULT",
            "------",
            f"{metadata.target_width} × {metadata.target_height}",
            f"{metadata.frame_count} frames",
            f"{seconds:.3f} sec",
            RESULT_LABELS[output_format],
            format_size(destination.stat().st_size),
            destination.name,
        ]
    )


def format_size(size: int) -> str:
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _is_seedvr(backend: str) -> bool:
    return backend in {"seedvr2", "seedvr2-native"}


def format_stages(current: str, detail: str, *, backend: str) -> str:
    names = list(STAGE_ORDER)
    if not _is_seedvr(backend):
        names[names.index("SeedVR2 upscaling")] = "Frame upscaling"
    current_name = current
    if current == "SeedVR2 upscaling" and not _is_seedvr(backend):
        current_name = "Frame upscaling"
    if current_name not in names:
        current_name = names[0]
    current_index = names.index(current_name)
    lines = ["Progress", ""]
    for index, name in enumerate(names):
        if index < current_index:
            mark = "done"
        elif index == current_index:
            mark = "now"
        else:
            mark = "pending"
        lines.append(f"[{mark}] {index + 1}. {name}")
    if detail:
        lines.extend(["", detail])
    return "\n".join(lines)


def format_queue(items: list[tuple[str, str]]) -> str:
    if not items:
        return "Queue\n\nNo files yet."
    width = max(len(name) for name, _state in items)
    lines = ["Queue", ""]
    for name, state in items:
        lines.append(f"{name.ljust(width)}  {state}")
    return "\n".join(lines)


def format_gui_error(exc: BaseException, *, comfy_url: str = "") -> str:
    """User-facing text. Tracebacks stay in the log, not in this string."""

    if isinstance(exc, JobCancelled):
        return "CANCELLED\n\n" + str(exc)
    if isinstance(exc, ComfyError) and "could not reach comfyui" in str(exc).lower():
        target = comfy_url or "the configured URL"
        return f"ComfyUI is not reachable at {target}."
    if isinstance(exc, PipelineError):
        text = str(exc)
        lowered = text.lower()
        if "ffmpeg" in lowered and "not found" in lowered:
            return "FFmpeg was not found. Install FFmpeg and restart the terminal."
        if "ffprobe" in lowered and "not found" in lowered:
            return "FFmpeg was not found. Install FFmpeg and restart the terminal."
        return text
    return "The upscale failed before it could finish. The traceback is in the terminal."


def note_from_log(
    message: str,
    prompts: int | None,
    frames: int | None,
) -> tuple[int | None, int | None, str]:
    """Turn an existing pipeline log line into a batch detail, when it has one."""

    plan = _PLAN_LINE.search(message)
    if plan:
        return int(plan.group(2)), int(plan.group(1)), ""
    batch = _BATCH_LINE.search(message)
    if batch is None:
        return prompts, frames, ""
    index = int(batch.group(1)) + 1
    last_frame = int(batch.group(3)) + 1
    total = prompts if prompts else "?"
    frame_total = frames if frames else "?"
    detail = f"Processing SeedVR2 batch {index} / {total} — frames {last_frame} / {frame_total}"
    return prompts, frames, detail


def accept_log_line(message: str) -> bool:
    text = message.strip()
    if not text or len(text) > 400:
        return False
    if text[0] in "[{":
        return False
    return True


def check_comfy_connection(
    url: str,
    *,
    connect: Callable[[str], ComfyConnection] | None = None,
) -> ComfyStatus:
    """Read live ``/object_info``. Node names are whatever that response contains."""

    opener = connect or _open_client
    client = opener(url)
    try:
        object_info = client.object_info()
        stats: dict[str, Any] = {}
        getter = getattr(client, "system_stats", None)
        if callable(getter):
            try:
                loaded = getter()
            except ComfyError:
                loaded = {}
            if isinstance(loaded, dict):
                stats = loaded
    except ComfyError as exc:
        return ComfyStatus(
            online=False,
            summary="ComfyUI: OFFLINE",
            seedvr2_available=False,
            node_lines=[format_gui_error(exc, comfy_url=url)],
            overlap_supported=None,
            overlap_message="",
        )
    finally:
        client.close()

    report = assess_installation(object_info, stats)
    detected = [name for name in object_info if "seedvr" in str(name).lower()]
    known_lines = known_seedvr2_status(object_info)
    overlap = _schema_has_temporal_overlap(object_info, detected)
    message = "" if overlap or not detected else OVERLAP_UNSUPPORTED
    lines = format_gui_status(report)
    if detected:
        lines.append("Detected: " + ", ".join(detected))
    lines.extend(known_lines)
    return ComfyStatus(
        online=True,
        summary="ComfyUI: CONNECTED",
        seedvr2_available=report.numz_installed,
        node_lines=lines,
        overlap_supported=overlap,
        overlap_message=message,
        native_available=len(report.native_nodes) == 3,
        selected_label=report.selected_label,
        ready=report.ready,
    )


def preflight(
    config: GuiJobConfig,
    *,
    connect: Callable[[str], ComfyConnection] | None = None,
) -> PreflightResult:
    """Fail before a GPU job starts. Raises the pipeline's own errors."""

    validate_gui_config(config)
    if not config.source.is_file():
        raise PipelineError(f"Input file not found: {config.source}")
    if config.source.suffix.lower() != ".gif":
        raise PipelineError(
            f"{config.source.name} is not a GIF. The current decoder only accepts GIF. "
            "PNG and animated WebP are not supported."
        )
    require_binary("ffmpeg")
    require_binary("ffprobe")
    config_path = write_runtime_config(config)
    overlap_supported: bool | None = None
    if config.backend == "seedvr2":
        opener = connect or _open_client
        client = opener(config.comfy_url)
        try:
            object_info = client.object_info()
        finally:
            client.close()
        loaded, _from = load_config(config_path)
        settings = loaded["seedvr2"]
        workflow_path = resolve_existing_file(str(settings["workflow"]), config_path)
        workflow = load_workflow(workflow_path)
        assert_workflow_configured(workflow, workflow_path)
        assert_nodes_available(workflow, object_info)
        detected = [name for name in object_info if "seedvr" in str(name).lower()]
        overlap_supported = _schema_has_temporal_overlap(object_info, detected)
        if detected and not overlap_supported:
            logger.warning("%s", OVERLAP_UNSUPPORTED)
    elif config.backend == "seedvr2-native":
        opener = connect or _open_client
        client = opener(config.comfy_url)
        try:
            object_info = client.object_info()
        finally:
            client.close()
        report = assess_installation(object_info)
        detected = [name for name in object_info if "seedvr" in str(name).lower()]
        overlap_supported = _schema_has_temporal_overlap(object_info, detected)
        if not report.ready:
            detail = "\n".join(report.missing) or report.workflow_note
            raise WorkflowConfigError(detail or "Native SeedVR2 is not ready.")
    return PreflightResult(config_path=config_path, temporal_overlap_supported=overlap_supported)


def write_runtime_config(config: GuiJobConfig) -> Path:
    """Merge the page's URL and workflow onto config.yaml without editing that file."""

    merged = copy.deepcopy(load_config(None)[0])
    merged.setdefault("comfyui", {})["url"] = config.comfy_url
    if config.workflow_path:
        workflow = Path(config.workflow_path)
        if not workflow.is_file():
            raise PipelineError(f"Workflow file not found: {workflow}")
        merged.setdefault("seedvr2", {})["workflow"] = str(workflow)
    destination = config.output_dir / ".gui" / f"{config.source.stem}.runtime-config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return destination


def request_cancel() -> None:
    """Ask the running job to interrupt its own ComfyUI prompt. Does not submit a new one."""

    ACTIVE_CANCEL.set()


def set_preview_background(name: str) -> None:
    global PREVIEW_BACKGROUND
    if name in {"checkerboard", "black", "white"}:
        PREVIEW_BACKGROUND = name


def render_live_frame(source: Path, destination: Path) -> Path:
    """Copy a completed frame for the page. RGBA is composited. The source file is unchanged."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        if image.mode == "RGBA":
            if PREVIEW_BACKGROUND == "black":
                base = Image.new("RGB", image.size, (0, 0, 0))
                base.paste(image, mask=image.getchannel("A"))
                base.save(destination, format="PNG")
            elif PREVIEW_BACKGROUND == "white":
                base = Image.new("RGB", image.size, (255, 255, 255))
                base.paste(image, mask=image.getchannel("A"))
                base.save(destination, format="PNG")
            else:
                composite_rgba_on_checkerboard(image).save(destination, format="PNG")
        else:
            image.convert("RGB").save(destination, format="PNG")
    return destination


def run_prepared_job(
    config: GuiJobConfig,
    checked: PreflightResult,
    *,
    on_event: EventCallback | None = None,
    inspect: Callable[..., GifInspection] = inspect_gif,
    prepare: Callable[..., Path] = prepare_asset,
    upscale: Callable[..., None] = upscale_job,
    finalize: Callable[..., Path] = finalize_job,
) -> JobResult:
    """Run one file through the existing pipeline and report stage changes."""

    if config.verbose:
        logging.getLogger("midnight_upscale").setLevel(logging.DEBUG)
    arguments = pipeline_arguments(config, checked.config_path)
    progress = {"stage": "Inspecting input", "prompts": None, "frames": None, "lines": []}

    def emit(stage: str = "", detail: str = "", log_line: str = "") -> None:
        if stage:
            progress["stage"] = _advance(str(progress["stage"]), stage, config.backend)
        if log_line and accept_log_line(log_line):
            lines: list[str] = progress["lines"]
            lines.append(log_line)
            del lines[:-200]
        if on_event is not None:
            on_event(str(progress["stage"]), detail, log_line if accept_log_line(log_line) else "")

    def on_log(message: str) -> None:
        prompts, frames, detail = note_from_log(
            message,
            progress["prompts"],
            progress["frames"],
        )
        progress["prompts"] = prompts
        progress["frames"] = frames
        stage = ""
        if message.startswith("Decoding "):
            stage = "Decoding GIF"
        elif message.startswith("Edge cleanup"):
            stage = "Preparing RGB / alpha"
        elif message.startswith("Prepared "):
            stage = "Alpha upscaling"
        elif message.startswith("Encoding "):
            stage = "Encoding output"
        elif detail:
            stage = "SeedVR2 upscaling"
        emit(stage=stage, detail=detail, log_line=message)

    handler = _CallbackHandler(on_log)
    pipeline_logger = logging.getLogger("midnight_upscale")
    pipeline_logger.addHandler(handler)
    ACTIVE_CANCEL.clear()
    bus = ProgressBus(cancel_event=ACTIVE_CANCEL)
    global ACTIVE_BUS
    ACTIVE_BUS = bus
    token = bind_bus(bus)
    live_dir = config.output_dir / ".gui" / "live"
    seen = {"at": 0.0, "key": ""}

    def on_progress(event: object) -> None:
        from midnight_upscale.progress import PipelineProgressEvent

        if not isinstance(event, PipelineProgressEvent) or on_event is None:
            return
        key = (
            event.status,
            event.stage,
            event.message,
            event.comfy_node_id,
            event.node_step,
            event.frames_done,
            event.queue_state,
        )
        now = time.monotonic()
        if key == seen["key"] and now - float(seen["at"]) < 0.25 and not event.preview_bytes:
            return
        seen["key"] = key
        seen["at"] = now
        preview = ""
        caption = ""
        if event.preview_bytes and event.preview_kind == "comfy":
            blob = live_dir / "comfy-preview.jpg"
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(event.preview_bytes)
            preview = str(blob)
            caption = "ComfyUI AI preview — not final output"
        elif event.preview_path and event.preview_kind == "completed":
            source = Path(event.preview_path)
            if source.is_file():
                preview = str(render_live_frame(source, live_dir / "latest-frame.png"))
                caption = "Latest completed frame"
        line = stamp_log(event.message) if event.message else ""
        if line:
            log_path = live_dir / "job.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        on_event(event.stage, "", line, format_status(event), preview, caption)

    bus.add_listener(on_progress)
    job_dir: Path | None = None
    try:
        emit(stage="Inspecting input")
        info = inspect(config.source)
        emit(log_line=f"Source: {info.width}x{info.height}")
        emit(stage="Decoding GIF")
        job_dir = prepare(config.source, **arguments["prepare"])
        metadata = JobMetadata.load(job_dir / "metadata.json")
        emit(stage="Alpha upscaling", log_line=f"Decoded {metadata.frame_count} frames")
        emit(log_line=f"Target: {metadata.target_width}x{metadata.target_height}")
        emit(log_line=f"Edge cleanup: {config.edge_cleanup} -> {metadata.edge_cleanup}")
        emit(
            log_line=(
                f"Alpha resized with {metadata.alpha_mode} "
                f"to {metadata.target_width}x{metadata.target_height}"
            )
        )
        if checked.temporal_overlap_supported is False:
            emit(log_line=OVERLAP_UNSUPPORTED)
        upscale_stage = "SeedVR2 upscaling" if _is_seedvr(config.backend) else "Frame upscaling"
        emit(stage=upscale_stage)
        upscale(job_dir, **arguments["upscale"])
        emit(log_line="Upscale step finished")
        emit(stage="Recombining RGBA")
        destination = finalize(job_dir, **arguments["finalize"])
        emit(log_line=f"Recombined {metadata.frame_count} RGBA frames")
        emit(stage="Validating output")
        seconds = expected_encoded_duration_ms(metadata.frame_durations_ms) / 1000.0
        emit(log_line=f"Output duration verified: {seconds:.3f} sec")
        comparison_dir = config.output_dir / ".gui" / "compare" / config.source.stem
        middle = metadata.frame_count // 2
        try:
            source_preview, upscaled_preview = _export_comparison(job_dir, middle, comparison_dir)
        except OSError as exc:
            logger.warning("Comparison frames were not written: %s", exc)
            source_preview, upscaled_preview = None, None
        preview = _try_browser_preview(config, metadata, destination)
        try:
            frame_root = _archive_compare_frames(
                job_dir, config.output_dir / ".gui" / "frames" / config.source.stem
            )
        except OSError as exc:
            logger.warning("Comparison frames were not copied: %s", exc)
            frame_root = None
        if not config.keep_workdir:
            _remove_workdir(job_dir, destination)
        if frame_root is not None:
            kept_workdir = frame_root
        elif job_dir.exists():
            kept_workdir = job_dir
        else:
            kept_workdir = None
        emit(stage="Finished", log_line=f"Output: {destination}")
        return JobResult(
            source_name=config.source.name,
            output_path=destination,
            preview_path=preview,
            workdir=kept_workdir,
            comparison_source=source_preview,
            comparison_upscaled=upscaled_preview,
            frame_count=metadata.frame_count,
            middle_frame=middle,
            result_text=format_result_card(metadata, destination, config.output_format),
            log_lines=list(progress["lines"]),
        )
    except JobCancelled:
        if job_dir is not None and job_dir.exists() and not config.keep_workdir:
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        ACTIVE_BUS = None
        reset_bus(token)
        pipeline_logger.removeHandler(handler)


def iter_queue(configs: list[GuiJobConfig]) -> Iterator[JobEvent]:
    """Run files one after another. A second call does not start another GPU job."""

    if not configs:
        yield JobEvent(error="Choose a GIF first.")
        return
    stems = [config.source.stem for config in configs]
    if len(set(stems)) != len(stems):
        yield JobEvent(
            error=(
                "Two GIFs would use the same work directory. Rename one so their file names differ."
            )
        )
        return
    if not JOB_LOCK.acquire(blocking=False):
        yield JobEvent(
            error=("An upscale is already running. Wait for it to finish before starting another.")
        )
        return

    events: queue.Queue[JobEvent | None] = queue.Queue()

    def worker() -> None:
        try:
            _run_queue(configs, events)
        except Exception as exc:
            logger.exception("GUI queue failed")
            url = configs[0].comfy_url if configs else ""
            events.put(JobEvent(error=format_gui_error(exc, comfy_url=url)))
        finally:
            events.put(None)
            JOB_LOCK.release()

    thread = threading.Thread(target=worker, name="midnight-upscale-gui", daemon=True)
    started = False
    try:
        thread.start()
        started = True
        while True:
            try:
                item = events.get(timeout=1.0)
            except queue.Empty:
                if not thread.is_alive():
                    break
                bus = ACTIVE_BUS
                if bus is None:
                    yield JobEvent()
                else:
                    snap = bus.snapshot()
                    yield JobEvent(stage=snap.stage, panel=format_status(snap))
                continue
            if item is None:
                break
            yield item
    finally:
        if not started:
            JOB_LOCK.release()


def comparison_frame(workdir: Path, index: int, destination_dir: Path) -> tuple[Path, Path]:
    """Render one source frame and one upscaled frame on a checkerboard."""

    return _export_comparison(workdir, index, destination_dir)


def composite_rgba_on_checkerboard(image: Image.Image, *, cell: int = 16) -> Image.Image:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    ys, xs = np.indices((height, width))
    light = ((xs // cell) + (ys // cell)) % 2 == 0
    board = np.empty((height, width, 4), dtype=np.uint8)
    board[light] = (232, 232, 232, 255)
    board[~light] = (188, 188, 188, 255)
    base = Image.fromarray(board, mode="RGBA")
    base.alpha_composite(rgba)
    return base.convert("RGB")


def browser_preview_command(
    ffmpeg: str,
    source: Path,
    board: Path,
    destination: Path,
) -> list[str]:
    """H.264 preview with transparency flattened onto a checkerboard.

    The production WebM is not an input that this command overwrites.
    """

    return [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-loop",
        "1",
        "-i",
        str(board),
        "-filter_complex",
        "[1:v][0:v]scale2ref[bg][fg];[bg][fg]overlay=shortest=1",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def _run_queue(configs: list[GuiJobConfig], events: queue.Queue[JobEvent | None]) -> None:
    prepared: list[tuple[GuiJobConfig, PreflightResult]] = []
    for config in configs:
        prepared.append((config, preflight(config)))
    items = [(config.source.name, "WAITING") for config, _checked in prepared]
    events.put(JobEvent(queue=list(items), stage="Inspecting input"))
    for index, (config, checked) in enumerate(prepared):
        items[index] = (config.source.name, "PROCESSING")
        events.put(
            JobEvent(
                queue=list(items),
                stage="Inspecting input",
                log_line=f"Starting {config.source.name}",
            )
        )
        outcome = _iter_one(config, checked, events, items)
        if outcome is None:
            items[index] = (config.source.name, "ERROR")
            events.put(JobEvent(queue=list(items)))
            return
        items[index] = (config.source.name, "DONE")
        events.put(JobEvent(queue=list(items), finished=True, result=outcome, stage="Finished"))


def _iter_one(
    config: GuiJobConfig,
    checked: PreflightResult,
    events: queue.Queue[JobEvent | None],
    items: list[tuple[str, str]],
) -> JobResult | None:
    inbox: queue.Queue[JobEvent | None] = queue.Queue()

    def on_event(
        stage: str,
        detail: str = "",
        log_line: str = "",
        panel: str = "",
        preview: str = "",
        caption: str = "",
    ) -> None:
        inbox.put(
            JobEvent(
                stage=stage,
                detail=detail,
                log_line=log_line,
                queue=list(items),
                panel=panel,
                preview_path=preview,
                caption=caption,
            )
        )

    def worker() -> None:
        try:
            result = run_prepared_job(config, checked, on_event=on_event)
            inbox.put(JobEvent(finished=True, result=result, stage="Finished", queue=list(items)))
        except Exception as exc:
            if config.verbose:
                logger.exception("Upscale failed")
            else:
                logger.error("Upscale failed: %s", exc)
            inbox.put(
                JobEvent(
                    error=format_gui_error(exc, comfy_url=config.comfy_url),
                    queue=list(items),
                )
            )
        finally:
            inbox.put(None)

    thread = threading.Thread(target=worker, name="midnight-upscale-job", daemon=True)
    thread.start()
    failure = False
    result: JobResult | None = None
    while True:
        item = inbox.get()
        if item is None:
            break
        if item.error:
            failure = True
        if item.result is not None:
            result = item.result
        events.put(item)
    thread.join()
    if failure or result is None:
        return None
    return result


def _archive_compare_frames(job_dir: Path, destination: Path) -> Path | None:
    """Copy source and final RGBA so the frame slider still works after work/ is removed."""

    source = job_dir / "source_rgba"
    final = job_dir / "final_rgba"
    if not source.is_dir() or not final.is_dir():
        return None
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination / "source_rgba")
    shutil.copytree(final, destination / "final_rgba")
    return destination


def _export_comparison(
    job_dir: Path,
    index: int,
    destination_dir: Path,
) -> tuple[Path | None, Path | None]:
    source = frame_path(job_dir / "source_rgba", index)
    upscaled = frame_path(job_dir / "final_rgba", index)
    if not source.is_file() or not upscaled.is_file():
        logger.warning("Comparison frames are missing in %s", job_dir)
        return None, None
    destination_dir.mkdir(parents=True, exist_ok=True)
    source_out = destination_dir / f"source-{index:06d}.png"
    upscaled_out = destination_dir / f"upscaled-{index:06d}.png"
    with Image.open(source) as image:
        composite_rgba_on_checkerboard(image).save(source_out, format="PNG")
    with Image.open(upscaled) as image:
        composite_rgba_on_checkerboard(image).save(upscaled_out, format="PNG")
    return source_out, upscaled_out


def _try_browser_preview(
    config: GuiJobConfig,
    metadata: JobMetadata,
    destination: Path,
) -> Path | None:
    if config.output_format != "webm":
        return None
    preview = destination.with_name(f"{destination.stem}.browser-preview.mp4")
    if preview.resolve() == destination.resolve():
        return None
    board = destination.with_name(f"{destination.stem}.checkerboard.png")
    try:
        ffmpeg = require_binary("ffmpeg")
        _write_checkerboard(board, metadata.target_width, metadata.target_height)
        command = browser_preview_command(ffmpeg, destination, board, preview)
        if Path(command[-1]).resolve() == destination.resolve():
            raise PipelineError(
                "Refusing to overwrite the production WebM with the browser preview."
            )
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            tail = (completed.stderr or "")[-600:]
            raise PipelineError(f"Browser preview failed: {tail}")
    except (PipelineError, OSError) as exc:
        logger.warning("Browser preview was not created: %s", exc)
        return None
    finally:
        if board.exists():
            board.unlink()
    return preview if preview.is_file() else None


def _write_checkerboard(path: Path, width: int, height: int) -> None:
    blank = Image.new("RGBA", (max(width, 1), max(height, 1)), (0, 0, 0, 0))
    image = composite_rgba_on_checkerboard(blank)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _remove_workdir(job_dir: Path, destination: Path) -> None:
    if not job_dir.exists():
        return
    output_path = destination.resolve()
    job_path = job_dir.resolve()
    if output_path == job_path or job_path in output_path.parents:
        logger.warning("Keeping %s because the encoded file is inside it.", job_dir)
        return
    shutil.rmtree(job_dir)
    logger.info("Removed work directory %s", job_dir)


def _advance(current: str, new: str, backend: str) -> str:
    names = list(STAGE_ORDER)
    if not _is_seedvr(backend):
        names[names.index("SeedVR2 upscaling")] = "Frame upscaling"
    current_name = current
    new_name = new
    if not _is_seedvr(backend):
        if current_name == "SeedVR2 upscaling":
            current_name = "Frame upscaling"
        if new_name == "SeedVR2 upscaling":
            new_name = "Frame upscaling"
    if new_name not in names:
        return current
    if current_name not in names or names.index(new_name) >= names.index(current_name):
        return new
    return current


def _schema_has_temporal_overlap(object_info: dict[str, Any], detected: list[str]) -> bool:
    return any(schema_has_input(object_info, name, "temporal_overlap") for name in detected)


def _open_client(url: str) -> ComfyClient:
    loaded, _from = load_config(None)
    comfy_cfg = loaded["comfyui"]
    return ComfyClient(
        url,
        timeout_sec=float(comfy_cfg["timeout_sec"]),
        poll_interval_sec=float(comfy_cfg["poll_interval_sec"]),
    )


def _format_fps(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value - round(value)) < 0.05:
        return f"{round(value):.0f} FPS"
    return f"{value:.2f} FPS"


class _CallbackHandler(logging.Handler):
    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__(level=logging.INFO)
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if accept_log_line(message):
            self.callback(message)


def coerce_upload_paths(uploaded: Any) -> list[Path]:
    """Normalize Gradio's file payload to paths. The files are not modified."""

    if uploaded is None:
        return []
    items = uploaded if isinstance(uploaded, (list, tuple)) else [uploaded]
    paths: list[Path] = []
    for item in items:
        if item is None or item == "":
            continue
        if isinstance(item, (str, Path)):
            paths.append(Path(item))
            continue
        name = getattr(item, "name", None)
        if name:
            paths.append(Path(str(name)))
    return paths
