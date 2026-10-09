"""Prepare, upscale, and finalize a character animation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from midnight_upscale.alpha import (
    ALPHA_MODES,
    EDGE_CLEANUP_CHOICES,
    edge_cleanup_simple,
    resolve_edge_cleanup,
    split_rgba,
    upscale_alpha,
)
from midnight_upscale.decode import iter_composited_frames
from midnight_upscale.encode import assert_encoded_stream, encode_sequence, probe_video
from midnight_upscale.interpolate import resolve_interpolation
from midnight_upscale.models import JobMetadata
from midnight_upscale.progress import (
    JobCancelled,
    ProgressBus,
    bind_bus,
    checkpoint,
    current_bus,
    report,
    reset_bus,
)
from midnight_upscale.recombine import recombine_directories
from midnight_upscale.seedvr2 import (
    WorkflowRunner,
    run_frame_upscale,
    run_seedvr2,
    validate_seedvr2_batch_size,
    validate_temporal_overlap,
)
from midnight_upscale.utils import (
    ZERO_DELAY_ENCODED_MS,
    PipelineError,
    clear_pngs,
    expected_encoded_duration_ms,
    frame_path,
    load_config,
    next_available_file,
    reset_workdir,
)
from midnight_upscale.validate import (
    assert_frame_count,
    assert_same_size,
    assert_sampled_alpha,
    format_result_report,
)

logger = logging.getLogger(__name__)

FORMATS = ("webm", "apng", "gif")
FORMAT_SUFFIXES = {
    "webm": {".webm"},
    "apng": {".apng", ".png"},
    "gif": {".gif"},
}


def prepare_asset(
    source: Path,
    *,
    scale: int,
    workdir: Path | None,
    alpha_mode: str,
    edge_cleanup: str,
    interpolate: str,
    overwrite: bool,
) -> Path:
    if scale < 1:
        raise PipelineError(f"--scale must be >= 1, got {scale}")
    if alpha_mode not in ALPHA_MODES:
        raise PipelineError(f"--alpha-mode must be one of {', '.join(ALPHA_MODES)}")
    if edge_cleanup not in EDGE_CLEANUP_CHOICES:
        raise PipelineError("--edge-cleanup must be auto, off, or simple")
    resolve_interpolation(interpolate)

    source = source.resolve()
    job_dir = reset_workdir(
        workdir or Path("work") / source.stem,
        overwrite=overwrite,
        source=source,
    )
    source_before = source.stat().st_mtime_ns

    durations: list[int] = []
    width = 0
    height = 0
    loop: int | None = None
    expected = 0
    has_transparency = False
    has_semitransparency = False

    logger.info("Decoding %s", source)
    report("Inspect input", message=f"Inspecting {source.name}")
    report("Decode GIF", message=f"Decoding {source.name}")
    for frame in iter_composited_frames(source):
        checkpoint()
        if not durations:
            width = frame.width
            height = frame.height
            loop = frame.gif_loop_count
            expected = frame.frame_count
        rgba = frame.image
        alpha_channel = np.asarray(rgba)[:, :, 3]
        if np.any(alpha_channel < 255):
            has_transparency = True
        if np.any((alpha_channel > 0) & (alpha_channel < 255)):
            has_semitransparency = True
        rgba.save(frame_path(job_dir / "source_rgba", frame.index), format="PNG")
        durations.append(frame.duration_ms)
        frame.image.close()
        report(
            "Decode GIF",
            message=f"Decoded frame {len(durations)}/{expected or len(durations)}",
            frames_done=len(durations),
            frames_total=expected or len(durations),
            frames_kind="completed",
        )

    resolved_cleanup = resolve_edge_cleanup(edge_cleanup, has_transparency=has_transparency)
    report(
        "Prepare RGB / alpha",
        message=f"Edge cleanup: {resolved_cleanup}",
        frames_total=len(durations),
        frames_kind="completed",
    )
    for index in range(len(durations)):
        checkpoint()
        with Image.open(frame_path(job_dir / "source_rgba", index)) as loaded:
            rgba = loaded.convert("RGBA").copy()
        if resolved_cleanup == "simple":
            rgba = edge_cleanup_simple(rgba)
        rgb, alpha = split_rgba(rgba)
        rgb.save(frame_path(job_dir / "rgb", index), format="PNG")
        alpha.save(frame_path(job_dir / "alpha", index), format="PNG")
        report(
            "Prepare RGB / alpha",
            message=f"Prepared RGB {index + 1}/{len(durations)}",
            frames_done=index + 1,
            frames_total=len(durations),
            frames_kind="completed",
        )
        upscaled = upscale_alpha(alpha, scale, alpha_mode)
        upscaled.save(frame_path(job_dir / "upscaled_alpha", index), format="PNG")
        report(
            "Upscale alpha",
            message=f"Alpha resize {index + 1}/{len(durations)}",
            frames_done=index + 1,
            frames_total=len(durations),
            frames_kind="completed",
        )
        rgba.close()

    if len(durations) != expected:
        raise PipelineError(
            f"Decoded {len(durations)} frames from {source}, expected {expected}. "
            "No frame was skipped."
        )
    if source.stat().st_mtime_ns != source_before:
        raise PipelineError(f"Source file was modified during prepare: {source}")

    warnings: list[str] = []
    if any(duration == 0 for duration in durations):
        warnings.append(
            "One or more GIF frame delays are 0 ms (as fast as possible). "
            f"Those frames are encoded at {ZERO_DELAY_ENCODED_MS} ms. "
            "Non-zero delays are unchanged."
        )
    total = sum(durations)
    metadata = JobMetadata(
        source=str(source),
        original_width=width,
        original_height=height,
        frame_count=len(durations),
        frame_durations_ms=durations,
        total_duration_ms=total,
        gif_loop_count=loop,
        durations_constant=len(set(durations)) <= 1,
        estimated_fps=None if total <= 0 else len(durations) / (total / 1000.0),
        has_transparency=has_transparency,
        has_semitransparency=has_semitransparency,
        requested_scale=scale,
        target_width=width * scale,
        target_height=height * scale,
        alpha_mode=alpha_mode,
        edge_cleanup=resolved_cleanup,
        interpolate=interpolate,
        workdir=str(job_dir),
        duration_warnings=warnings,
    )
    metadata.save(job_dir / "metadata.json")
    assert_frame_count(job_dir / "source_rgba", metadata.frame_count, "decoded RGBA")
    assert_frame_count(job_dir / "rgb", metadata.frame_count, "RGB")
    assert_frame_count(job_dir / "alpha", metadata.frame_count, "alpha")
    assert_frame_count(job_dir / "upscaled_alpha", metadata.frame_count, "upscaled alpha")
    assert_same_size(
        assert_frame_count(job_dir / "upscaled_alpha", metadata.frame_count, "upscaled alpha"),
        (metadata.target_width, metadata.target_height),
        "upscaled alpha",
    )
    logger.info(
        "Prepared %s frames at %sx%s, alpha %s, scale %s -> %sx%s",
        metadata.frame_count,
        width,
        height,
        alpha_mode,
        scale,
        metadata.target_width,
        metadata.target_height,
    )
    for warning in warnings:
        logger.warning("%s", warning)
    return job_dir


def upscale_job(
    workdir: Path,
    *,
    backend: str,
    comfy_url: str | None,
    batch_size: int,
    config_path: Path | None,
    timeout_sec: float | None,
    overwrite: bool,
    runner: WorkflowRunner | None = None,
    temporal_overlap: int = 1,
    temporal_mode: str = "auto",
) -> None:
    metadata = JobMetadata.load(workdir / "metadata.json")
    resolve_interpolation(metadata.interpolate)
    if backend == "auto":
        from midnight_upscale.seedvr2_native import resolve_requested_backend

        backend = resolve_requested_backend(comfy_url, config_path)
    if backend == "seedvr2":
        validate_seedvr2_batch_size(batch_size)
        validate_temporal_overlap(temporal_overlap, batch_size)
        run_seedvr2(
            workdir,
            metadata,
            batch_size=batch_size,
            config_path=config_path,
            comfy_url=comfy_url,
            timeout_sec=timeout_sec,
            overwrite=overwrite,
            runner=runner,
            temporal_overlap=temporal_overlap,
        )
    elif backend == "seedvr2-native":
        from midnight_upscale.seedvr2_native import run_seedvr2_native

        run_seedvr2_native(
            workdir,
            metadata,
            comfy_url=comfy_url,
            config_path=config_path,
            timeout_sec=timeout_sec,
            overwrite=overwrite,
            temporal_overlap=temporal_overlap,
            temporal_mode=temporal_mode,
        )
    elif backend == "frame-upscale":
        run_frame_upscale(
            workdir,
            metadata,
            config_path=config_path,
            comfy_url=comfy_url,
            timeout_sec=timeout_sec,
            overwrite=overwrite,
            runner=runner,
        )
    else:
        raise PipelineError("backend must be auto, seedvr2-native, seedvr2, or frame-upscale")


def finalize_job(
    workdir: Path,
    *,
    fmt: str,
    output: Path | None,
    overwrite: bool,
    crf: int | None,
    webm_pix_fmt: str | None,
    config_path: Path | None,
) -> Path:
    if fmt not in FORMATS:
        raise PipelineError(f"--format must be one of {', '.join(FORMATS)}")
    metadata = JobMetadata.load(workdir / "metadata.json")
    resolve_interpolation(metadata.interpolate)
    config, _loaded = load_config(config_path)
    encode_cfg = config.get("encode") or {}
    crf_value = int(encode_cfg.get("crf", 18) if crf is None else crf)
    pix_fmt = str(
        encode_cfg.get("webm_pix_fmt", "yuva420p") if webm_pix_fmt is None else webm_pix_fmt
    )

    rgb_frames = assert_frame_count(workdir / "upscaled_rgb", metadata.frame_count, "upscaled RGB")
    alpha_frames = assert_frame_count(
        workdir / "upscaled_alpha", metadata.frame_count, "upscaled alpha"
    )
    with Image.open(rgb_frames[0]) as first_rgb, Image.open(alpha_frames[0]) as first_alpha:
        if first_rgb.size != first_alpha.size:
            raise PipelineError(
                f"Upscaled RGB is {first_rgb.size[0]}x{first_rgb.size[1]} but upscaled alpha is "
                f"{first_alpha.size[0]}x{first_alpha.size[1]}. "
                "Alpha was not stretched to match. Check the SeedVR2 resolution mapping."
            )
        target_size = first_rgb.size
    if target_size != (metadata.target_width, metadata.target_height):
        raise PipelineError(
            f"Upscaled frames are {target_size[0]}x{target_size[1]}, "
            f"expected {metadata.target_width}x{metadata.target_height} "
            f"from scale {metadata.requested_scale}. "
            "The SeedVR2 resolution input is the shortest edge on the numz node, "
            "not a multiplier. Set scale.mode in the config, or keep the model "
            "from rounding the size."
        )
    assert_same_size(rgb_frames, target_size, "upscaled RGB")
    assert_same_size(alpha_frames, target_size, "upscaled alpha")

    final_dir = workdir / "final_rgba"
    final_dir.mkdir(parents=True, exist_ok=True)
    clear_pngs(final_dir, overwrite=True)
    recombine_directories(workdir / "upscaled_rgb", workdir / "upscaled_alpha", final_dir)
    final_frames = assert_frame_count(final_dir, metadata.frame_count, "final RGBA")
    assert_same_size(final_frames, target_size, "final RGBA")
    assert_sampled_alpha(workdir / "source_rgba", final_dir)

    destination = _check_output_path(
        output or _default_output(metadata.source, fmt),
        fmt,
        metadata.source,
        overwrite=overwrite,
    )
    report("Encode output", message=f"Encoding {destination.name}")
    logger.info("Encoding %s", destination)
    encode_sequence(
        final_frames,
        metadata.frame_durations_ms,
        destination,
        fmt=fmt,
        loop=metadata.gif_loop_count,
        crf=crf_value,
        webm_pix_fmt=pix_fmt,
    )
    report("Validate output", message=f"Validating {destination.name}")
    probed = probe_video(destination)
    assert_encoded_stream(probed, metadata, fmt=fmt, webm_pix_fmt=pix_fmt)
    notes = list(metadata.duration_warnings or [])
    encoded_ms = expected_encoded_duration_ms(metadata.frame_durations_ms)
    if encoded_ms != metadata.total_duration_ms:
        notes.append(
            f"Encoded timeline is {encoded_ms / 1000:.3f}s because zero GIF delays "
            f"were written as {ZERO_DELAY_ENCODED_MS} ms."
        )
    print(
        format_result_report(
            metadata,
            destination,
            float(probed["duration_sec"]),
            notes=notes,
        ),
        end="",
    )
    return destination


def process_asset(
    source: Path,
    *,
    scale: int,
    backend: str,
    comfy_url: str | None,
    batch_size: int,
    fmt: str,
    output: Path | None,
    alpha_mode: str,
    edge_cleanup: str,
    interpolate: str,
    workdir: Path | None,
    config_path: Path | None,
    timeout_sec: float | None,
    overwrite: bool,
    keep_workdir: bool,
    crf: int | None,
    webm_pix_fmt: str | None,
    runner: WorkflowRunner | None = None,
    temporal_overlap: int = 1,
    temporal_mode: str = "auto",
) -> Path:
    if backend == "auto":
        from midnight_upscale.seedvr2_native import resolve_requested_backend

        backend = resolve_requested_backend(comfy_url, config_path)
    if backend == "seedvr2":
        validate_seedvr2_batch_size(batch_size)
        validate_temporal_overlap(temporal_overlap, batch_size)
    owned_bus = current_bus() is None
    token = bind_bus(ProgressBus()) if owned_bus else None
    job_dir: Path | None = None
    try:
        job_dir = prepare_asset(
            source,
            scale=scale,
            workdir=workdir,
            alpha_mode=alpha_mode,
            edge_cleanup=edge_cleanup,
            interpolate=interpolate,
            overwrite=overwrite,
        )
        succeeded = False
        destination: Path | None = None
        try:
            upscale_job(
                job_dir,
                backend=backend,
                comfy_url=comfy_url,
                batch_size=batch_size,
                config_path=config_path,
                timeout_sec=timeout_sec,
                overwrite=overwrite,
                runner=runner,
                temporal_overlap=temporal_overlap,
                temporal_mode=temporal_mode,
            )
            destination = finalize_job(
                job_dir,
                fmt=fmt,
                output=output,
                overwrite=overwrite,
                crf=crf,
                webm_pix_fmt=webm_pix_fmt,
                config_path=config_path,
            )
            succeeded = True
            bus = current_bus()
            if bus is not None:
                bus.mark_finished(f"Output: {destination}")
            return destination
        finally:
            if (
                succeeded
                and not keep_workdir
                and job_dir is not None
                and job_dir.exists()
                and destination is not None
            ):
                output_path = destination.resolve()
                job_path = job_dir.resolve()
                if output_path == job_path or job_path in output_path.parents:
                    logger.warning("Keeping %s because the encoded file is inside it.", job_dir)
                else:
                    shutil.rmtree(job_dir)
                    logger.info("Removed work directory %s", job_dir)
    except JobCancelled:
        if job_dir is not None and not keep_workdir and job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        if token is not None:
            reset_bus(token)


def _default_output(source: str, fmt: str) -> Path:
    suffix = {"webm": ".webm", "apng": ".apng", "gif": ".gif"}[fmt]
    return Path("output") / f"{Path(source).stem}_upscaled{suffix}"


def _check_output_path(output: Path, fmt: str, source: str, *, overwrite: bool) -> Path:
    if output.suffix.lower() not in FORMAT_SUFFIXES[fmt]:
        allowed = ", ".join(sorted(FORMAT_SUFFIXES[fmt]))
        raise PipelineError(
            f"Output {output} does not match --format {fmt}. Expected a suffix of {allowed}."
        )
    if output.resolve() == Path(source).resolve():
        raise PipelineError("Refusing to overwrite the source file")
    if output.exists() and not overwrite:
        versioned = next_available_file(output)
        logger.info("Output %s already exists. Using %s.", output, versioned)
        if versioned.resolve() == Path(source).resolve():
            raise PipelineError("Refusing to overwrite the source file")
        return versioned
    return output
