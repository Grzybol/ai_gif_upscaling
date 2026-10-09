"""Form handling, job queue, and status text for the VIDEO CONVERTER tab.

The Gradio page calls this module. It does not need ComfyUI. The only shared
resource with the Upscale tab is the GPU: AI segmentation takes the same job
lock as a SeedVR2 upscale, so the two never run together.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from midnight_upscale.background import MODE_LABELS, BackgroundSettings
from midnight_upscale.chroma import EDGE_CLEANUP_LEVELS, ChromaSettings, parse_hex_color
from midnight_upscale.mask_temporal import TEMPORAL_LABELS
from midnight_upscale.progress import (
    JobCancelled,
    PipelineProgressEvent,
    ProgressBus,
    bind_bus,
    reset_bus,
    stamp_log,
)
from midnight_upscale.progress import _bar as progress_bar
from midnight_upscale.progress import _clock as clock
from midnight_upscale.segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    INSTALL_HELP,
    MODEL_CHOICES,
    ai_available,
    backend_names,
)
from midnight_upscale.spritesheet import SpritesheetSettings
from midnight_upscale.utils import PipelineError
from midnight_upscale.video_convert import (
    CONVERTER_STAGES,
    STAGE_FINISHED,
    ConvertResult,
    ConvertSettings,
    ReviewSet,
    RunObserver,
    convert_video,
    estimate_output,
    preview_one_frame,
)
from midnight_upscale.video_decode import FramePlan
from midnight_upscale.video_export import (
    FORMAT_CHOICES,
    composite_on_background,
)
from midnight_upscale.video_inspect import VideoInfo, inspect_video
from midnight_upscale.video_transform import ResizeSettings

logger = logging.getLogger(__name__)

FPS_CHOICES = ["Source", "12", "15", "20", "24", "25", "30"]
RESIZE_CHOICES = {"Keep source": 1.0, "0.5x": 0.5, "1x": 1.0, "2x": 2.0}
RESIZE_LABELS = [*RESIZE_CHOICES, "Custom"]
PADDING_CHOICES = ["0", "2", "4", "8", "16"]
SHEET_PADDING_CHOICES = ["0", "1", "2", "4"]
MAX_TEXTURE_CHOICES = ["2048", "4096", "8192"]
PREVIEW_BACKGROUNDS = ["checkerboard", "black", "white"]
DEFAULT_FORMATS = ["Transparent WebM"]

STAGE_TITLES = {
    "Inspect input": "Inspecting input",
    "Decode video": "Decoding video",
    "Analyze background": "Analyzing background",
    "Remove background": "Removing background",
    "Stabilize mask": "Stabilizing mask",
    "Crop / pad": "Cropping / padding",
    "Resize": "Resizing",
    "Export": "Exporting",
    "Validate": "Validating",
    "Finished": "Finished",
}

CONVERTER_LOCK = threading.Lock()
PREVIEW_BACKGROUND = "checkerboard"
_ACTIVE_BUS: ProgressBus | None = None
_CANCEL_ALL = threading.Event()
_INFO_CACHE: dict[tuple[str, float, int], VideoInfo] = {}


@dataclass
class ConverterEvent:
    log_line: str = ""
    panel: str = ""
    preview_path: str = ""
    caption: str = ""
    error: str = ""
    queue: list[tuple[str, str]] | None = None
    result: ConvertResult | None = None
    finished: bool = False


@dataclass
class _RunState:
    observer: RunObserver
    started: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------- form parsing


def set_preview_background(name: str) -> None:
    global PREVIEW_BACKGROUND
    if name in PREVIEW_BACKGROUNDS:
        PREVIEW_BACKGROUND = name


def parse_number(value: object, name: str, *, minimum: float = 0.0) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        raise PipelineError(f"{name} must be a number, got {value!r}") from None
    if number < minimum:
        raise PipelineError(f"{name} must be at least {minimum:g}, got {number:g}")
    return number


def parse_fps(value: object) -> float | None:
    text = str(value or "Source").strip()
    if text.lower() in {"", "source"}:
        return None
    fps = parse_number(text, "Output FPS")
    if fps <= 0:
        raise PipelineError("Output FPS must be above zero")
    return fps


def settings_from_form(
    *,
    start: object,
    end: object,
    fps: object,
    background_mode: str,
    key_color: str,
    tolerance: float,
    softness: float,
    spill: float,
    edge_cleanup: str,
    hard_mask: bool,
    sample_key: bool,
    ai_backend: str,
    ai_model: str,
    temporal: str,
    crop: bool,
    padding: object,
    center: bool,
    resize: str,
    resize_width: object,
    resize_height: object,
    keep_aspect: bool,
    formats: list[str],
    sheet_columns: object,
    sheet_padding: object,
    max_texture: object,
    power_of_two: bool,
    output_dir: str,
    overwrite: bool,
    keep_workdir: bool,
) -> ConvertSettings:
    selected = tuple(FORMAT_CHOICES[label] for label in (formats or []) if label in FORMAT_CHOICES)
    if not selected:
        raise PipelineError("Choose at least one output format.")
    start_sec = parse_number(start or 0, "Start time")
    end_value = parse_number(end or 0, "End time")
    end_sec = end_value if end_value > 0 else None
    if end_sec is not None and end_sec <= start_sec:
        raise PipelineError("End time must be after the start time.")

    chroma = ChromaSettings(
        key_color=parse_hex_color(key_color or "#00ff00"),
        tolerance=float(tolerance) / 100.0,
        softness=float(softness) / 100.0,
        spill=float(spill) / 100.0,
        edge_cleanup=edge_cleanup if edge_cleanup in EDGE_CLEANUP_LEVELS else "Light",
        hard_mask=bool(hard_mask),
    )
    background = BackgroundSettings(
        mode=MODE_LABELS.get(background_mode, "auto"),
        chroma=chroma,
        sample_key_color=bool(sample_key),
        ai_backend=ai_backend or DEFAULT_BACKEND,
        ai_model=(ai_model or DEFAULT_MODEL).strip(),
    )

    if resize == "Custom":
        resize_settings = ResizeSettings(
            mode="custom",
            width=int(parse_number(resize_width or 0, "Width")),
            height=int(parse_number(resize_height or 0, "Height")),
            keep_aspect=bool(keep_aspect),
        )
    else:
        scale = RESIZE_CHOICES.get(resize, 1.0)
        resize_settings = ResizeSettings(
            mode="source" if resize == "Keep source" else "scale", scale=scale
        )

    columns_text = str(sheet_columns or "Auto").strip()
    columns = (
        None
        if columns_text.lower() in {"", "auto", "0"}
        else int(parse_number(columns_text, "Columns", minimum=1))
    )
    sheet = SpritesheetSettings(
        columns=columns,
        padding=int(parse_number(sheet_padding if sheet_padding != "" else 2, "Frame padding")),
        max_size=int(parse_number(max_texture or 4096, "Max texture size", minimum=16)),
        power_of_two=bool(power_of_two),
    )
    return ConvertSettings(
        start_sec=start_sec,
        end_sec=end_sec,
        fps=parse_fps(fps),
        background=background,
        temporal=TEMPORAL_LABELS.get(temporal, "low"),
        crop=bool(crop),
        padding=int(parse_number(padding if padding != "" else 4, "Padding")),
        center=bool(center),
        resize=resize_settings,
        formats=selected,
        sheet=sheet,
        output_dir=Path(output_dir or "output"),
        overwrite=bool(overwrite),
        keep_workdir=bool(keep_workdir),
        browser_preview=True,
    )


# ------------------------------------------------------------------- inspection


def cached_info(path: Path) -> VideoInfo:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime, stat.st_size)
    if key not in _INFO_CACHE:
        if len(_INFO_CACHE) > 16:
            _INFO_CACHE.clear()
        _INFO_CACHE[key] = inspect_video(path)
    return _INFO_CACHE[key]


def format_estimate(info: VideoInfo, plan: FramePlan) -> str:
    lines = ["Estimated output", "----------------"]
    lines.append(f"Frames:   {plan.count}")
    if plan.fps is None:
        lines.append("FPS:      source (every frame kept, original timing)")
        if info.variable_timing:
            lines.append("Timing:   variable. Each frame keeps its own duration.")
    else:
        lines.append(f"FPS:      {plan.fps:g} (explicit change)")
        lines.append(f"Resampled from {plan.window_frames} source frames in the trimmed range.")
        if plan.dropped:
            lines.append(f"{plan.dropped} source frames will be dropped.")
        if plan.duplicated:
            lines.append(f"{plan.duplicated} frames will be repeated.")
    lines.append(f"Duration: {plan.total_ms / 1000.0:.3f} s")
    return "\n".join(lines)


def format_auto_note(label: str, reason: str) -> str:
    return f"Background mode:\n{label}\n{reason}"


def mid_index(plan: FramePlan) -> int:
    return plan.count // 2


# -------------------------------------------------------------------- previews


def preview_dir() -> Path:
    directory = Path("work") / "_preview"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def render_triplet(
    original: np.ndarray, mask: np.ndarray | None, result: np.ndarray | None, background: str
) -> tuple[str, str | None, str | None]:
    """Save Original / Mask / Transparent result images for the page."""

    directory = preview_dir()
    token = f"{time.time_ns()}"
    for old in directory.glob("view_*.png"):
        if old.stat().st_mtime < time.time() - 120:
            old.unlink(missing_ok=True)
    original_path = directory / f"view_{token}_original.png"
    Image.fromarray(np.ascontiguousarray(original[..., :3])).save(original_path)
    mask_path = result_path = None
    if mask is not None:
        mask_path = directory / f"view_{token}_mask.png"
        Image.fromarray(np.ascontiguousarray(mask)).save(mask_path)
    if result is not None:
        result_path = directory / f"view_{token}_result.png"
        composite_on_background(result, background).save(result_path)
    return (
        str(original_path),
        str(mask_path) if mask_path else None,
        (str(result_path) if result_path else None),
    )


@contextlib.contextmanager
def gpu_guard() -> Iterator[None]:
    """Hold the shared GPU lock for AI work. Fails at once if an upscale is running."""

    from midnight_upscale.gui_logic import JOB_LOCK

    if not JOB_LOCK.acquire(blocking=False):
        raise PipelineError(
            "A SeedVR2 upscale or another AI job is using the GPU. "
            "Wait for it to finish, or use Chroma Key which does not need the GPU."
        )
    try:
        yield
    finally:
        JOB_LOCK.release()


def preview_frame(
    path: Path, settings: ConvertSettings, plan_index: int, *, run_ai: bool
) -> tuple[tuple[str, str | None, str | None], str]:
    """Original / mask / result for one planned output frame. Returns images and a note."""

    info = cached_info(path)
    plan = estimate_output(info, settings)
    plan_index = max(0, min(plan_index, plan.count - 1))
    source_index = plan.source_indices[plan_index]
    needs_ai = _would_use_ai(info, settings)
    if needs_ai and not run_ai:
        from midnight_upscale.video_decode import decode_single_frame

        original = np.asarray(decode_single_frame(info, source_index))
        images = render_triplet(original, None, None, PREVIEW_BACKGROUND)
        return images, "AI segmentation runs when you press Preview frame."
    if needs_ai:
        with gpu_guard():
            original, mask, result, resolved = preview_one_frame(info, settings, source_index)
    else:
        original, mask, result, resolved = preview_one_frame(info, settings, source_index)
    images = render_triplet(original, mask, result, PREVIEW_BACKGROUND)
    return images, f"Background mode: {resolved.label}. {resolved.reason}"


def _would_use_ai(info: VideoInfo, settings: ConvertSettings) -> bool:
    mode = settings.background.mode
    if mode == "ai":
        return True
    if mode != "auto" or info.has_alpha:
        return False
    # Auto: green borders mean chroma. Look at the middle frame only.
    from midnight_upscale.chroma import estimate_key_color, looks_chroma_green
    from midnight_upscale.video_decode import decode_single_frame

    frame = np.asarray(decode_single_frame(info, info.frame_count // 2))
    color, coverage = estimate_key_color([frame])
    return not looks_chroma_green(color, coverage)


def review_frame(
    review: ReviewSet, position: int, background: str
) -> tuple[tuple[str, str | None, str | None], int]:
    """Images for the ``position``-th saved review frame, and its frame number."""

    position = max(0, min(position, len(review.frame_numbers) - 1))
    number = review.frame_numbers[position]
    stem = review.directory / f"{number:06d}"
    original = np.asarray(Image.open(stem.with_name(stem.name + "_original.png")).convert("RGBA"))
    mask = np.asarray(Image.open(stem.with_name(stem.name + "_mask.png")).convert("L"))
    result = np.asarray(Image.open(stem.with_name(stem.name + "_rgba.png")).convert("RGBA"))
    return render_triplet(original, mask, result, background), number


# ----------------------------------------------------------------- availability


def ai_status_text(backend: str) -> str:
    if ai_available(backend):
        return f"AI background removal is available ({backend})."
    return INSTALL_HELP


def ai_backend_choices() -> list[str]:
    return list(backend_names())


def ai_model_choices() -> list[str]:
    return list(MODEL_CHOICES)


# ----------------------------------------------------------------- status text


def format_converter_status(event: PipelineProgressEvent, observer: RunObserver) -> str:
    stage = event.stage if event.stage in CONVERTER_STAGES else CONVERTER_STAGES[0]
    current = CONVERTER_STAGES.index(stage)
    lines = ["STATUS", event.status, "", "Stage:", STAGE_TITLES.get(stage, stage)]
    total = event.frames_total
    if total and event.frames_done is not None and stage in observer_frame_stages():
        done = event.frames_done
        lines.extend(["", "Frames:", f"{done} / {total}"])
        lines.extend(["", "Current frame:", str(done)])
        lines.extend(["", "Progress:", progress_bar(100.0 * done / total)])
    elif event.message:
        lines.extend(["", event.message])
    lines.extend(["", "Elapsed:", clock(event.elapsed_seconds)])
    if observer.background_label:
        lines.extend(["", "Background mode:", observer.background_label])
    lines.extend(["", "Pipeline:"])
    for index, name in enumerate(CONVERTER_STAGES):
        if name in observer.skipped and index <= current:
            mark = "-"
            suffix = " (skipped)"
        elif index < current or stage == STAGE_FINISHED and event.status == "FINISHED":
            mark, suffix = "x", ""
        elif index == current:
            mark, suffix = ">", ""
        else:
            mark, suffix = " ", ""
        lines.append(f" [{mark}] {index + 1}. {name}{suffix}")
    return "\n".join(lines)


def observer_frame_stages() -> tuple[str, ...]:
    return (
        "Decode video",
        "Remove background",
        "Stabilize mask",
        "Crop / pad",
        "Resize",
        "Export",
    )


def format_result(result: ConvertResult) -> str:
    lines = [
        "RESULT",
        "------",
        f"Source:           {result.source.name}",
        f"Background mode:  {result.background_label}",
        f"Frames:           {result.frame_count}",
        f"Canvas:           {result.size[0]} x {result.size[1]}",
        f"Duration:         {sum(result.durations_ms) / 1000.0:.3f} s",
        f"Elapsed:          {clock(result.elapsed_sec)}",
        "",
        "Files:",
    ]
    for path in result.files:
        lines.append(f"  {path.name}  ({_size_text(path)})")
    if result.validation:
        lines.extend(["", "Checked:", *[f"  {line}" for line in result.validation]])
    for warning in result.warnings:
        lines.extend(["", f"WARNING: {warning}"])
    lines.extend(["", f"Log: {result.log_path}"])
    return "\n".join(lines)


def _size_text(path: Path) -> str:
    if path.is_dir():
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        count = sum(1 for p in path.glob("*.png"))
        return f"{count} files, {total / 1024:.0f} KB"
    size = path.stat().st_size
    return f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.0f} KB"


def format_queue(items: list[tuple[str, str]]) -> str:
    if not items:
        return "Queue\n\nNo files yet."
    return "Queue\n\n" + "\n".join(f"{state:<11} {name}" for name, state in items)


# ------------------------------------------------------------------------ queue


def request_cancel() -> None:
    """Stop the current video at the next frame and skip the rest of the queue."""

    _CANCEL_ALL.set()
    bus = _ACTIVE_BUS
    if bus is not None:
        bus.cancel_event.set()


def iter_converter_queue(paths: list[Path], settings: ConvertSettings) -> Iterator[ConverterEvent]:
    """Convert files one after another. Only one converter job runs at a time."""

    if not paths:
        yield ConverterEvent(error="Choose a video first.")
        return
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        yield ConverterEvent(error="Two files share a name. Rename one so outputs differ.")
        return
    if not CONVERTER_LOCK.acquire(blocking=False):
        yield ConverterEvent(error="A conversion is already running.")
        return

    events: queue.Queue[ConverterEvent | None] = queue.Queue()
    observer_box: dict[str, RunObserver] = {}
    _CANCEL_ALL.clear()

    def worker() -> None:
        try:
            _run_queue(paths, settings, events, observer_box)
        except Exception as exc:
            logger.exception("Converter queue failed")
            events.put(ConverterEvent(error=str(exc)))
        finally:
            events.put(None)
            CONVERTER_LOCK.release()

    thread = threading.Thread(target=worker, name="midnight-converter", daemon=True)
    started = False
    try:
        thread.start()
        started = True
        while True:
            try:
                item = events.get(timeout=0.5)
            except queue.Empty:
                if not thread.is_alive():
                    break
                bus = _ACTIVE_BUS
                observer = observer_box.get("current")
                if bus is not None and observer is not None:
                    yield ConverterEvent(panel=format_converter_status(bus.snapshot(), observer))
                continue
            if item is None:
                break
            yield item
    finally:
        if not started:
            CONVERTER_LOCK.release()


def _run_queue(
    paths: list[Path],
    settings: ConvertSettings,
    events: queue.Queue[ConverterEvent | None],
    observer_box: dict[str, RunObserver],
) -> None:
    global _ACTIVE_BUS
    items = [(path.name, "WAITING") for path in paths]
    events.put(ConverterEvent(queue=list(items)))
    last_preview = {"at": 0.0}

    for index, path in enumerate(paths):
        if _CANCEL_ALL.is_set():
            items[index] = (path.name, "SKIPPED")
            continue
        items[index] = (path.name, "PROCESSING")
        events.put(ConverterEvent(queue=list(items), log_line=f"Starting {path.name}"))

        def on_log(message: str) -> None:
            events.put(ConverterEvent(log_line=message))

        def on_preview(image: Path, caption: str) -> None:
            last_preview["at"] = time.monotonic()
            events.put(ConverterEvent(preview_path=str(image), caption=caption))

        observer = RunObserver(
            log=on_log, preview=on_preview, preview_background=lambda: PREVIEW_BACKGROUND
        )
        observer.gpu_guard = gpu_guard
        observer_box["current"] = observer
        bus = ProgressBus()
        if _CANCEL_ALL.is_set():
            bus.cancel_event.set()
        _ACTIVE_BUS = bus
        token = bind_bus(bus)
        try:
            result = convert_video(path, settings, observer)
        except JobCancelled:
            items[index] = (path.name, "CANCELLED")
            events.put(
                ConverterEvent(
                    queue=list(items),
                    error="Cancelled. Incomplete output was removed. The log was kept.",
                    panel=format_converter_status(bus.snapshot(), observer),
                )
            )
        except Exception as exc:
            logger.exception("Conversion failed")
            items[index] = (path.name, "FAILED")
            events.put(ConverterEvent(queue=list(items), error=str(exc)))
        else:
            items[index] = (path.name, "DONE")
            bus.mark_finished("Finished")
            events.put(
                ConverterEvent(
                    queue=list(items),
                    result=result,
                    panel=format_converter_status(bus.snapshot(), observer),
                )
            )
        finally:
            _ACTIVE_BUS = None
            reset_bus(token)
    events.put(ConverterEvent(finished=True, queue=list(items)))


def log_stamp(message: str) -> str:
    return stamp_log(message)
