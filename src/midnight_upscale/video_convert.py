"""The frame-based converter pipeline.

    inspect -> decode to PNG -> analyze -> remove background -> stabilize mask
            -> crop / pad -> resize -> export -> validate

Every stage between decode and export reads and writes lossless RGBA PNG files,
so nothing is transcoded through a lossy format. Progress is real: each loop
reports the frame it just finished. Cancellation is checked between frames.
This module never talks to ComfyUI.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from midnight_upscale.background import (
    BackgroundSettings,
    ResolvedBackground,
    open_remover,
    remove_background,
    resolve_background,
)
from midnight_upscale.browser_player import make_browser_player
from midnight_upscale.mask_temporal import get_level, smooth_masks
from midnight_upscale.progress import JobCancelled, checkpoint, report, stamp_log
from midnight_upscale.spritesheet import SpritesheetSettings, plan_spritesheet
from midnight_upscale.utils import DEFAULT_CONFIG, PipelineError, frame_path
from midnight_upscale.video_decode import FramePlan, decode_plan, plan_frames
from midnight_upscale.video_export import (
    APNG,
    GIF,
    PNG_SEQUENCE,
    SPRITESHEET,
    WEBM,
    ValidationReport,
    composite_on_background,
    export_png_sequence,
    export_spritesheet,
    export_video,
    make_browser_preview,
    output_paths,
    remove_outputs,
    resolve_output_stem,
    validate_spritesheet_output,
    validate_video_output,
    zip_spritesheet,
)
from midnight_upscale.video_inspect import VideoInfo, inspect_video
from midnight_upscale.video_transform import (
    ResizeSettings,
    alpha_bbox,
    apply_crop,
    fill_transparent_rgb,
    plan_crop,
    resize_rgba,
    target_size,
    union_box,
)

STAGE_INSPECT = "Inspect input"
STAGE_DECODE = "Decode video"
STAGE_ANALYZE = "Analyze background"
STAGE_REMOVE = "Remove background"
STAGE_STABILIZE = "Stabilize mask"
STAGE_CROP = "Crop / pad"
STAGE_RESIZE = "Resize"
STAGE_EXPORT = "Export"
STAGE_VALIDATE = "Validate"
STAGE_FINISHED = "Finished"

CONVERTER_STAGES: tuple[str, ...] = (
    STAGE_INSPECT,
    STAGE_DECODE,
    STAGE_ANALYZE,
    STAGE_REMOVE,
    STAGE_STABILIZE,
    STAGE_CROP,
    STAGE_RESIZE,
    STAGE_EXPORT,
    STAGE_VALIDATE,
    STAGE_FINISHED,
)

CACHE_VERSION = 2  # white-key processing and its edge cleanup changed
REVIEW_SAMPLES = 12
PREVIEW_MIN_EVERY = 3
PREVIEW_MAX_EVERY = 10


@dataclass(frozen=True)
class ConvertSettings:
    start_sec: float = 0.0
    end_sec: float | None = None
    fps: float | None = None  # None keeps the source frames and their timing
    background: BackgroundSettings = field(default_factory=BackgroundSettings)
    temporal: str = "low"
    crop: bool = False
    padding: int = 4
    center: bool = True
    resize: ResizeSettings = field(default_factory=ResizeSettings)
    formats: tuple[str, ...] = (WEBM,)
    sheet: SpritesheetSettings = field(default_factory=SpritesheetSettings)
    webm_crf: int = int(DEFAULT_CONFIG["encode"]["crf"])
    output_dir: Path = Path("output")
    work_dir: Path = Path("work")
    overwrite: bool = False
    keep_workdir: bool = False
    browser_preview: bool = False
    # Reuse the processed frames when only the output settings changed.
    use_cache: bool = True


@dataclass
class ReviewSet:
    """A few full-size frames saved for the Original / Mask / Result viewer."""

    directory: Path
    frame_numbers: list[int]
    total_frames: int


@dataclass
class ConvertResult:
    source: Path
    outputs: dict[str, list[Path]]
    frame_count: int
    size: tuple[int, int]
    durations_ms: list[int]
    background_label: str
    background_reason: str
    elapsed_sec: float
    log_path: Path
    review: ReviewSet | None
    browser_preview: Path | None = None
    browser_player: Path | None = None
    validation: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    workdir: Path | None = None

    @property
    def files(self) -> list[Path]:
        found: list[Path] = []
        for group in self.outputs.values():
            found.extend(group)
        return found

    @property
    def downloads(self) -> list[Path]:
        """What to offer for download: a spritesheet is one zip, not its loose parts."""

        found: list[Path] = []
        for fmt, group in self.outputs.items():
            found.extend(group[-1:] if fmt == SPRITESHEET else group)
        return found


@dataclass
class Processed:
    """Frames after background removal, crop and resize, ready to export."""

    final: list[Path]
    transparent: bool
    size: tuple[int, int]
    label: str
    reason: str
    mode: str
    review: ReviewSet | None


def processing_fingerprint(source: Path, info: VideoInfo, settings: ConvertSettings) -> str:
    """Hash of everything that changes the processed frames. Output formats are not in it."""

    stat = source.stat()
    parts = {
        "version": CACHE_VERSION,
        "source": [str(source.resolve()), stat.st_size, stat.st_mtime_ns],
        "alpha": info.has_alpha,
        "range": [settings.start_sec, settings.end_sec, settings.fps],
        "background": asdict(settings.background),
        "temporal": settings.temporal,
        "crop": [settings.crop, settings.padding, settings.center],
        "resize": asdict(settings.resize),
    }
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


class FrameCache:
    """The last processed frames of one source, so a new output needs no reprocessing."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @property
    def frames_dir(self) -> Path:
        return self.directory / "final"

    def load(self, fingerprint: str, frame_count: int) -> Processed | None:
        meta_path = self.directory / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if meta.get("fingerprint") != fingerprint or meta.get("frame_count") != frame_count:
            return None
        frames = [frame_path(self.frames_dir, i) for i in range(frame_count)]
        if not all(path.is_file() for path in frames):
            return None
        review = None
        review_dir = Path(meta.get("review_dir") or "")
        if meta.get("review_numbers") and review_dir.is_dir():
            review = ReviewSet(review_dir, list(meta["review_numbers"]), frame_count)
        return Processed(
            final=frames,
            transparent=bool(meta["transparent"]),
            size=(int(meta["size"][0]), int(meta["size"][1])),
            label=str(meta["label"]),
            reason=str(meta["reason"]),
            mode=str(meta["mode"]),
            review=review,
        )

    def store(self, fingerprint: str, done: Processed) -> None:
        """Move the final frames here and point ``done`` at them."""

        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True)
        source_dir = done.final[0].parent
        shutil.move(str(source_dir), str(self.frames_dir))
        done.final = [frame_path(self.frames_dir, i) for i in range(len(done.final))]
        meta = {
            "fingerprint": fingerprint,
            "frame_count": len(done.final),
            "transparent": done.transparent,
            "size": list(done.size),
            "label": done.label,
            "reason": done.reason,
            "mode": done.mode,
            "review_dir": str(done.review.directory) if done.review else "",
            "review_numbers": done.review.frame_numbers if done.review else [],
        }
        (self.directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


class RunObserver:
    """Receives log lines and live preview frames. All callbacks are optional."""

    def __init__(
        self,
        log: Callable[[str], None] | None = None,
        preview: Callable[[Path, str], None] | None = None,
        preview_background: Callable[[], str] | None = None,
    ) -> None:
        self._log = log
        self._preview = preview
        self._background = preview_background or (lambda: "checkerboard")
        # Context manager factory held around GPU work (AI segmentation only).
        self.gpu_guard: Callable[[], contextlib.AbstractContextManager[None]] = (
            contextlib.nullcontext
        )
        self.skipped: set[str] = set()
        self.background_label = ""
        self.background_reason = ""
        self.frame_count = 0
        self._file: Path | None = None
        self._preview_dir: Path | None = None
        self._preview_serial = 0

    def attach_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self._file = path

    def attach_preview_dir(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._preview_dir = directory

    def say(self, message: str) -> None:
        line = stamp_log(message)
        if self._file is not None:
            with self._file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self._log is not None:
            self._log(message)

    def show(self, rgba: np.ndarray, frame_number: int, total: int) -> None:
        """Publish a finished frame, composited on the chosen preview background."""

        if self._preview is None or self._preview_dir is None:
            return
        self._preview_serial += 1
        path = self._preview_dir / f"latest_{self._preview_serial:05d}.png"
        composite_on_background(rgba, self._background()).save(path, compress_level=1)
        for old in self._preview_dir.glob("latest_*.png"):
            if old != path and old.stat().st_mtime < time.time() - 30:
                old.unlink(missing_ok=True)
        self._preview(path, f"Latest processed frame — {frame_number} / {total}")


def read_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA")).copy()


def write_rgba(array: np.ndarray, path: Path, *, compress: int = 1) -> None:
    Image.fromarray(np.ascontiguousarray(array)).save(path, format="PNG", compress_level=compress)


def preview_interval(total: int) -> int:
    return max(PREVIEW_MIN_EVERY, min(PREVIEW_MAX_EVERY, total // 25))


def estimate_output(info: VideoInfo, settings: ConvertSettings) -> FramePlan:
    """The frame plan for these settings. Its ``count`` is the exact output frame count."""

    return plan_frames(
        info, start_sec=settings.start_sec, end_sec=settings.end_sec, fps=settings.fps
    )


def convert_video(
    source: Path,
    settings: ConvertSettings,
    observer: RunObserver | None = None,
    *,
    info: VideoInfo | None = None,
) -> ConvertResult:
    """Run one video through the whole pipeline. Raises ``JobCancelled`` when cancelled.

    On cancel or failure no output file is left behind, the log is kept, and the
    work directory is removed unless ``keep_workdir`` is set.
    """

    obs = observer or RunObserver()
    source = Path(source)
    stem = source.stem
    started = time.monotonic()
    workdir = (settings.work_dir / f"{stem}_converter").resolve()
    log_path = settings.output_dir / "logs" / f"{stem}_converter.log"
    obs.attach_file(log_path)
    created: list[Path] = []
    try:
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        # Outside the work directory, so the last frame stays viewable after cleanup.
        obs.attach_preview_dir(workdir.parent / "_preview")

        result = _run(source, stem, settings, obs, info, workdir, created, log_path, started)
        obs.say(f"Finished in {time.monotonic() - started:.1f} s")
        report(STAGE_FINISHED, message="Finished")
        return result
    except JobCancelled:
        remove_outputs(created)
        obs.say("CANCELLED. Incomplete output was removed.")
        raise
    except BaseException as exc:
        remove_outputs(created)
        obs.say(f"FAILED: {exc}")
        raise
    finally:
        if not settings.keep_workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


def _run(
    source: Path,
    stem: str,
    settings: ConvertSettings,
    obs: RunObserver,
    info: VideoInfo | None,
    workdir: Path,
    created: list[Path],
    log_path: Path,
    started: float,
) -> ConvertResult:
    # 1. Inspect
    checkpoint()
    report(STAGE_INSPECT, message="Inspecting input")
    obs.say(f"Inspecting {source.name}")
    info = info or inspect_video(source)
    plan = estimate_output(info, settings)
    total = plan.count
    obs.frame_count = total
    obs.say(
        f"Source: {info.width}x{info.height}, {info.frame_count} frames, "
        f"{info.duration_sec:.3f} s, alpha {'yes' if info.has_alpha else 'no'}"
    )
    obs.say(f"Plan: {plan.describe()}")

    # 2-7. Decode, remove background, stabilize, crop, resize. Skipped on a cache hit.
    cache = FrameCache(settings.work_dir / "_cache" / stem)
    fingerprint = processing_fingerprint(source, info, settings)
    hit = cache.load(fingerprint, total) if settings.use_cache else None
    if hit is not None:
        done = hit
        obs.say("Reusing cached processed frames. Only the export is repeated.")
        for name in (
            STAGE_DECODE,
            STAGE_ANALYZE,
            STAGE_REMOVE,
            STAGE_STABILIZE,
            STAGE_CROP,
            STAGE_RESIZE,
        ):
            obs.skipped.add(name)
            report(
                name,
                message="Cached",
                frames_done=total,
                frames_total=total,
                frames_kind="completed",
            )
        obs.background_label = done.label
        obs.background_reason = done.reason
        obs.say(f"Background mode: {done.label}. {done.reason}")
    else:
        done = _process_frames(info, plan, settings, obs, workdir, source)
        cache.store(fingerprint, done)
    final, transparent, size = done.final, done.transparent, done.size
    resolved_mode = done.mode
    review = done.review
    obs.say(f"Common canvas: {size[0]}x{size[1]}")

    # 8. Export
    sheet_count = 1
    if SPRITESHEET in settings.formats:
        sheet_count = len(plan_spritesheet(size[0], size[1], total, settings.sheet).sheets)
    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = resolve_output_stem(
        stem, settings.formats, out_dir, sheet_count, overwrite=settings.overwrite
    )
    if out_stem != stem:
        obs.say(f"Outputs exist. Writing as {out_stem}")
    targets = output_paths(out_stem, settings.formats, out_dir, sheet_count)
    outputs: dict[str, list[Path]] = {}
    sheet_plan = None
    for number, fmt in enumerate(settings.formats, start=1):
        checkpoint()
        paths = targets[fmt]
        created.extend(paths)
        report(
            STAGE_EXPORT,
            message=f"Exporting {fmt} ({number} / {len(settings.formats)})",
            frames_done=0,
            frames_total=total,
            frames_kind="completed",
        )
        obs.say(f"Exporting {fmt}")
        if fmt in (WEBM, APNG, GIF):
            export_video(final, plan.durations_ms, paths[0], fmt, settings.webm_crf)
        elif fmt == PNG_SEQUENCE:
            export_png_sequence(final, paths[0], out_stem)
        elif fmt == SPRITESHEET:
            written, sheet_plan = export_spritesheet(
                final, plan.durations_ms, out_dir, out_stem, settings.sheet
            )
            paths[:] = [*written, zip_spritesheet(written, paths[-1])]
        outputs[fmt] = list(paths)

    browser_preview = None
    if settings.browser_preview and WEBM in outputs:
        checkpoint()
        report(STAGE_EXPORT, message="Browser preview")
        browser_preview = make_browser_preview(
            final,
            plan.durations_ms,
            workdir.parent / "_review" / f"{stem}_preview.mp4",
            workdir / "preview_frames",
            "checkerboard",
        )

    browser_player = None
    if settings.browser_preview:
        checkpoint()
        browser_player = make_browser_player(
            outputs, plan.durations_ms, out_dir, out_stem, video_preview=browser_preview
        )
        if browser_player.suffix == ".html":
            created.append(browser_player)
            if SPRITESHEET in outputs:
                with ZipFile(outputs[SPRITESHEET][-1], "a") as bundle:
                    bundle.write(browser_player, arcname=browser_player.name)

    # 9. Validate
    checkpoint()
    report(STAGE_VALIDATE, message="Validating output")
    validation: list[str] = []
    warnings: list[str] = []
    for fmt, paths in outputs.items():
        if fmt in (WEBM, APNG, GIF):
            check: ValidationReport = validate_video_output(
                paths[0],
                fmt,
                frame_count=total,
                size=size,
                durations_ms=plan.durations_ms,
                expect_transparency=transparent,
            )
            validation.extend(check.lines)
            warnings.extend(check.warnings)
        elif fmt == PNG_SEQUENCE:
            count = len(list(paths[0].glob("*.png")))
            if count != total:
                raise PipelineError(f"PNG sequence has {count} files, expected {total}")
            validation.append(f"{paths[0].name}: {count} PNG frames")
        elif fmt == SPRITESHEET and sheet_plan is not None:
            validation.extend(
                validate_spritesheet_output(
                    paths[:-2], paths[-2], sheet_plan, settings.sheet.max_size
                )
            )
    if not transparent and resolved_mode != "none":
        warnings.append("The result has no transparent pixels. Check the background settings.")
    for line in validation:
        obs.say(line)
    for line in warnings:
        obs.say(f"Warning: {line}")

    return ConvertResult(
        source=source,
        outputs=outputs,
        frame_count=total,
        size=size,
        durations_ms=plan.durations_ms,
        background_label=done.label,
        background_reason=done.reason,
        elapsed_sec=time.monotonic() - started,
        log_path=log_path,
        review=review,
        browser_preview=browser_preview,
        browser_player=browser_player,
        validation=validation,
        warnings=warnings,
        workdir=workdir if settings.keep_workdir else None,
    )


def _process_frames(
    info: VideoInfo,
    plan: FramePlan,
    settings: ConvertSettings,
    obs: RunObserver,
    workdir: Path,
    source: Path,
) -> Processed:
    total = plan.count
    # 2. Decode
    report(STAGE_DECODE, message="Decoding video", frames_done=0, frames_total=total)
    decoded_dir = workdir / "1_decoded"
    decode_plan(info, plan, decoded_dir, scratch=workdir / "decode_scratch")
    decoded = [frame_path(decoded_dir, i) for i in range(total)]

    # 3. Analyze
    checkpoint()
    report(STAGE_ANALYZE, message="Analyzing background")
    sample_ids = sorted({0, total // 2, total - 1})
    samples = [read_rgba(decoded[i]) for i in sample_ids]
    resolved = resolve_background(
        settings.background, has_alpha=info.has_alpha, sample_frames=samples
    )
    obs.background_label = resolved.label
    obs.background_reason = resolved.reason
    obs.say(f"Background mode: {resolved.label}. {resolved.reason}")

    # 4. Remove background
    keyed_dir = workdir / "2_keyed"
    keyed_dir.mkdir()
    _remove_backgrounds(resolved, decoded, keyed_dir, obs)
    keyed = [frame_path(keyed_dir, i) for i in range(total)]

    # 5. Stabilize
    _stabilize(resolved, settings.temporal, keyed, obs)
    review = _save_review(decoded, keyed, workdir.parent / "_review" / source.stem, obs)
    if not settings.keep_workdir:
        shutil.rmtree(decoded_dir, ignore_errors=True)

    # 6. Crop / pad
    current = keyed
    current = _crop(current, workdir / "3_cropped", settings, obs)

    # 7. Resize (also cleans the color under fully transparent pixels)
    final, transparent = _finalize(current, workdir / "4_final", settings.resize, obs)
    with Image.open(final[0]) as probe:
        size = probe.size
    return Processed(
        final=final,
        transparent=transparent,
        size=size,
        label=resolved.label,
        reason=resolved.reason,
        mode=resolved.mode,
        review=review,
    )


def _remove_backgrounds(
    resolved: ResolvedBackground, frames: list[Path], destination: Path, obs: RunObserver
) -> None:
    total = len(frames)
    every = preview_interval(total)
    report(
        STAGE_REMOVE,
        message="Removing background",
        frames_done=0,
        frames_total=total,
        frames_kind="completed",
    )
    guard = obs.gpu_guard() if resolved.needs_ai else contextlib.nullcontext()
    with guard:
        if resolved.needs_ai:
            obs.say(f"Loading AI model {resolved.ai_model} (first run downloads it)")
        remover = open_remover(resolved)
        if resolved.needs_ai:
            obs.say(f"Removing background from {total} frames")
        log_every = max(1, total // 10)
        try:
            for index, path in enumerate(frames):
                checkpoint()
                rgba = remove_background(resolved, read_rgba(path), remover)
                write_rgba(rgba, frame_path(destination, index))
                done = index + 1
                report(
                    STAGE_REMOVE,
                    message=f"Frame {done} / {total}",
                    frames_done=done,
                    frames_total=total,
                    frames_kind="completed",
                )
                if done % every == 0 or done == total:
                    obs.show(rgba, done, total)
                if done % log_every == 0 and done < total:
                    obs.say(f"Frame {done} / {total}")
        finally:
            if remover is not None:
                remover.close()
    obs.say(f"Removed background from {total} frames")


def _stabilize(
    resolved: ResolvedBackground, name: str, keyed: list[Path], obs: RunObserver
) -> None:
    level = get_level(name)
    total = len(keyed)
    if not resolved.needs_ai or level.max_correction <= 0 or total < 2:
        obs.skipped.add(STAGE_STABILIZE)
        why = "not needed for this mode" if not resolved.needs_ai else "off or a single frame"
        obs.say(f"Mask stabilization skipped ({why})")
        report(
            STAGE_STABILIZE,
            message="Skipped",
            frames_done=total,
            frames_total=total,
            frames_kind="completed",
        )
        return
    report(
        STAGE_STABILIZE,
        message="Stabilizing mask",
        frames_done=0,
        frames_total=total,
        frames_kind="completed",
    )

    def load(index: int) -> np.ndarray:
        return read_rgba(keyed[index])[..., 3]

    for index, mask in enumerate(smooth_masks(load, total, level)):
        checkpoint()
        rgba = read_rgba(keyed[index])
        rgba[..., 3] = mask
        write_rgba(rgba, keyed[index])
        report(
            STAGE_STABILIZE,
            message=f"Frame {index + 1} / {total}",
            frames_done=index + 1,
            frames_total=total,
            frames_kind="completed",
        )
    obs.say(f"Stabilized masks ({level.name}); RGB and frame geometry unchanged")


def _crop(
    frames: list[Path], destination: Path, settings: ConvertSettings, obs: RunObserver
) -> list[Path]:
    total = len(frames)
    if not settings.crop:
        obs.skipped.add(STAGE_CROP)
        obs.say("Crop skipped")
        report(
            STAGE_CROP,
            message="Skipped",
            frames_done=total,
            frames_total=total,
            frames_kind="completed",
        )
        return frames
    destination.mkdir()
    report(
        STAGE_CROP,
        message="Finding the common bounding box",
        frames_done=0,
        frames_total=2 * total,
        frames_kind="completed",
    )
    boxes = []
    frame_size = (0, 0)
    for index, path in enumerate(frames):
        checkpoint()
        rgba = read_rgba(path)
        frame_size = (rgba.shape[1], rgba.shape[0])
        boxes.append(alpha_bbox(rgba))
        report(
            STAGE_CROP,
            message=f"Measuring frame {index + 1} / {total}",
            frames_done=index + 1,
            frames_total=2 * total,
            frames_kind="completed",
        )
    box = union_box(boxes)
    if box is None:
        raise PipelineError(
            "Every frame is fully transparent, so there is nothing to crop to. "
            "Check the background removal settings."
        )
    plan = plan_crop(box, frame_size, settings.padding, settings.center)
    obs.say(
        f"One crop box for all {total} frames: {box}, padding {settings.padding}px, "
        f"canvas {plan.width}x{plan.height}"
    )
    out: list[Path] = []
    for index, path in enumerate(frames):
        checkpoint()
        target = frame_path(destination, index)
        write_rgba(apply_crop(read_rgba(path), plan), target)
        out.append(target)
        report(
            STAGE_CROP,
            message=f"Cropping frame {index + 1} / {total}",
            frames_done=total + index + 1,
            frames_total=2 * total,
            frames_kind="completed",
        )
    return out


def _finalize(
    frames: list[Path], destination: Path, resize: ResizeSettings, obs: RunObserver
) -> tuple[list[Path], bool]:
    """Resize (when asked), clean transparent-pixel color, and write the final frames."""

    destination.mkdir()
    total = len(frames)
    with Image.open(frames[0]) as probe:
        width, height = probe.size
    size = target_size(width, height, resize)
    if not resize.active:
        obs.skipped.add(STAGE_RESIZE)
        obs.say("Resize skipped (source size kept)")
    else:
        obs.say(f"Resizing {width}x{height} -> {size[0]}x{size[1]} (premultiplied alpha)")
    report(
        STAGE_RESIZE,
        message="Resizing" if resize.active else "Finalizing frames",
        frames_done=0,
        frames_total=total,
        frames_kind="completed",
    )
    out: list[Path] = []
    transparent = False
    for index, path in enumerate(frames):
        checkpoint()
        rgba = read_rgba(path)
        if resize.active:
            rgba = resize_rgba(rgba, size)
        rgba = fill_transparent_rgb(rgba)
        if np.any(rgba[..., 3] < 255):
            transparent = True
        target = frame_path(destination, index)
        write_rgba(rgba, target, compress=3)
        out.append(target)
        report(
            STAGE_RESIZE,
            message=f"Frame {index + 1} / {total}",
            frames_done=index + 1,
            frames_total=total,
            frames_kind="completed",
        )
    return out, transparent


def _save_review(
    decoded: list[Path], keyed: list[Path], directory: Path, obs: RunObserver
) -> ReviewSet:
    """Keep a handful of Original / Mask / Result frames for quick checking after the job."""

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    total = len(keyed)
    picks = sorted(
        {round(i * (total - 1) / max(REVIEW_SAMPLES - 1, 1)) for i in range(REVIEW_SAMPLES)}
    )
    picks = [p for p in picks if 0 <= p < total]
    for number in picks:
        stem = directory / f"{number:06d}"
        shutil.copyfile(decoded[number], stem.with_name(stem.name + "_original.png"))
        result = read_rgba(keyed[number])
        Image.fromarray(np.ascontiguousarray(result[..., 3])).save(
            stem.with_name(stem.name + "_mask.png")
        )
        write_rgba(result, stem.with_name(stem.name + "_rgba.png"))
    obs.say(f"Saved {len(picks)} review frames")
    return ReviewSet(directory=directory, frame_numbers=picks, total_frames=total)


def preview_one_frame(
    info: VideoInfo, settings: ConvertSettings, index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ResolvedBackground]:
    """Run a single source frame through background removal.

    Returns (original RGBA, mask HxW, result RGBA, resolved mode). No files are written.
    Mask smoothing needs neighbors, so it is not applied to a single frame.
    """

    from midnight_upscale.video_decode import decode_single_frame

    count = info.frame_count
    first = np.asarray(decode_single_frame(info, 0))
    middle = np.asarray(decode_single_frame(info, count // 2)) if count > 1 else first
    original = np.asarray(decode_single_frame(info, index)).copy()
    resolved = resolve_background(
        settings.background, has_alpha=info.has_alpha, sample_frames=[first, middle]
    )
    remover = open_remover(resolved)
    try:
        result = remove_background(resolved, original, remover)
    finally:
        if remover is not None:
            remover.close()
    return original, result[..., 3].copy(), result, resolved


__all__ = [
    "CONVERTER_STAGES",
    "ConvertResult",
    "ConvertSettings",
    "ReviewSet",
    "RunObserver",
    "convert_video",
    "estimate_output",
    "preview_one_frame",
]
