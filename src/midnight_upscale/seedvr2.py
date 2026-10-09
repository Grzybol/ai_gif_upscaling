"""SeedVR2 batching, and the explicit non-SeedVR2 frame-upscale fallback."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from midnight_upscale.comfy import (
    apply_seedvr2_overrides,
    assert_nodes_available,
    assert_workflow_configured,
    images_from_history,
    load_workflow,
    resolution_for_scale,
    resolve_temporal_overlap_target,
)
from midnight_upscale.models import JobMetadata
from midnight_upscale.utils import (
    PipelineError,
    ValidationError,
    as_node_id,
    clear_pngs,
    find_pngs,
    frame_path,
    list_indexed_frames,
    load_config,
    resolve_existing_file,
)

logger = logging.getLogger(__name__)

# SeedVR2VideoUpscaler batch_size must be 4n+1. Video batches start at 5.
SEEDVR2_MIN_BATCH_SIZE = 5
MAX_TEMPORAL_OVERLAP = 4


class WorkflowRunner(Protocol):
    def object_info(self) -> dict[str, Any]:
        """Return ComfyUI ``/object_info``."""

    def run_workflow(self, workflow: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Submit one prompt and return its history entry."""

    def upload_image(self, path: Path, *, subfolder: str = "") -> str:
        """Upload one PNG and return the name ComfyUI stored."""

    def download(self, image_info: dict[str, Any], dest: Path) -> None:
        """Download one history image to ``dest``."""


def validate_seedvr2_batch_size(batch_size: int) -> None:
    if batch_size < SEEDVR2_MIN_BATCH_SIZE or (batch_size - 1) % 4 != 0:
        raise PipelineError(
            f"SeedVR2 batch size must follow 4n+1. Use 5, 9, 13, ... Got {batch_size}."
        )


def validate_temporal_overlap(overlap: int, batch_size: int) -> None:
    """Reject overlaps that would swallow the batch or go past a small blend."""

    if overlap < 0 or overlap > MAX_TEMPORAL_OVERLAP or overlap >= batch_size:
        limit = min(MAX_TEMPORAL_OVERLAP, max(batch_size - 1, 0))
        raise PipelineError(
            f"SeedVR2 temporal overlap must be from 0 to {limit}. "
            "Suggested: batch 5 -> overlap 1; batch 9 -> overlap 1 or 2. "
            f"Got overlap {overlap} with batch size {batch_size}."
        )


def iter_windows(count: int, batch_size: int) -> Iterator[tuple[int, int]]:
    if batch_size < 1:
        raise PipelineError("batch size must be >= 1")
    start = 0
    while start < count:
        end = min(count, start + batch_size)
        yield start, end
        start = end


def run_seedvr2(
    workdir: Path,
    metadata: JobMetadata,
    *,
    batch_size: int,
    config_path: Path | None,
    comfy_url: str | None,
    timeout_sec: float | None,
    overwrite: bool,
    runner: WorkflowRunner | None = None,
    temporal_overlap: int = 1,
) -> None:
    validate_seedvr2_batch_size(batch_size)
    validate_temporal_overlap(temporal_overlap, batch_size)
    config, loaded_from = load_config(config_path)
    settings = config["seedvr2"]
    chunking = str(settings.get("chunking") or "windows")
    if chunking not in {"windows", "all"}:
        raise PipelineError("seedvr2.chunking must be 'windows' or 'all'")

    rgb_frames = list_indexed_frames(workdir / "rgb")
    if len(rgb_frames) != metadata.frame_count:
        raise ValidationError(
            f"RGB frame count {len(rgb_frames)} does not match metadata {metadata.frame_count}"
        )
    output_dir = workdir / "upscaled_rgb"
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_pngs(output_dir, overwrite=overwrite or not any(output_dir.glob("*.png")))

    workflow_path = resolve_existing_file(str(settings["workflow"]), loaded_from)
    template = load_workflow(workflow_path)
    assert_workflow_configured(template, workflow_path)

    owns_runner = runner is None
    if runner is None:
        from midnight_upscale.comfy import ComfyClient

        comfy_cfg = config["comfyui"]
        runner = ComfyClient(
            comfy_url or str(comfy_cfg["url"]),
            timeout_sec=float(timeout_sec if timeout_sec is not None else comfy_cfg["timeout_sec"]),
            poll_interval_sec=float(comfy_cfg["poll_interval_sec"]),
        )
    try:
        object_info = runner.object_info()
        assert_nodes_available(template, object_info)
        resolution, rounded = resolution_for_scale(
            str((settings.get("scale") or {}).get("mode") or "shortest_edge"),
            metadata.requested_scale,
            metadata.original_width,
            metadata.original_height,
        )
        if rounded:
            logger.warning(
                "SeedVR2 shortest-edge resolution was rounded from %s to %s so it is even. "
                "Alpha is still upscaled to %sx%s. Finalize aborts if the RGB size differs.",
                resolution - 1,
                resolution,
                metadata.target_width,
                metadata.target_height,
            )
        windows = list(_windows_for_chunking(metadata.frame_count, batch_size, chunking))
        overlap_target = resolve_temporal_overlap_target(template, settings, object_info)
        if overlap_target is None:
            logger.warning(
                "SeedVR2 temporal_overlap is not an input on the configured upscaler node. "
                "Continuing without it (the node keeps its own default, usually 0). "
                "Requested overlap was %s.",
                temporal_overlap,
            )
            overlap_node, overlap_field = "", ""
            overlap_value = None
        else:
            overlap_node, overlap_field = overlap_target
            overlap_value = temporal_overlap
            logger.info(
                "SeedVR2 temporal overlap %s -> node %s input %s",
                temporal_overlap,
                overlap_node,
                overlap_field,
            )
        logger.info(
            "SeedVR2: %s RGB frames, %s ComfyUI prompt(s), node batch_size %s",
            metadata.frame_count,
            len(windows),
            batch_size,
        )
        for batch_index, (start, end) in enumerate(windows):
            from midnight_upscale.progress import report

            report(
                "SeedVR2 processing",
                message=(
                    f"Chunk {batch_index + 1}/{len(windows)} "
                    f"frames {start + 1}-{end}/{metadata.frame_count}"
                ),
                batch_index=batch_index + 1,
                batch_total=len(windows),
                frames_total=metadata.frame_count,
                frames_kind="submitted",
            )
            _run_window(
                runner,
                template,
                settings,
                rgb_frames[start:end],
                workdir=workdir,
                batch_index=batch_index,
                start_index=start,
                output_dir=output_dir,
                scale=metadata.requested_scale,
                width=metadata.original_width,
                height=metadata.original_height,
                batch_size=batch_size,
                temporal_overlap=overlap_value,
                temporal_overlap_node=overlap_node,
                temporal_overlap_field=overlap_field,
            )
    finally:
        if owns_runner:
            runner.close()

    produced = list_indexed_frames(output_dir)
    if len(produced) != metadata.frame_count:
        raise ValidationError(
            f"SeedVR2 wrote {len(produced)} RGB frames, expected {metadata.frame_count}. "
            "No missing frame was replaced."
        )
    metadata.backend = "seedvr2"
    metadata.seedvr2_resolution = resolution
    metadata.chunking = chunking
    metadata.save(workdir / "metadata.json")


def run_frame_upscale(
    workdir: Path,
    metadata: JobMetadata,
    *,
    config_path: Path | None,
    comfy_url: str | None,
    timeout_sec: float | None,
    overwrite: bool,
    runner: WorkflowRunner | None = None,
) -> None:
    """Export frames for a separate image upscaler. Real-ESRGAN is not included."""

    rgb_frames = list_indexed_frames(workdir / "rgb")
    if len(rgb_frames) != metadata.frame_count:
        raise ValidationError(
            f"RGB frame count {len(rgb_frames)} does not match metadata {metadata.frame_count}"
        )
    manifest = {
        "input_rgb_dir": str((workdir / "rgb").resolve()),
        "alpha_dir": str((workdir / "alpha").resolve()),
        "expected_output_dir": str((workdir / "upscaled_rgb").resolve()),
        "frame_count": metadata.frame_count,
        "target_width": metadata.target_width,
        "target_height": metadata.target_height,
        "note": (
            "Upscale only the RGB PNGs. Do not send the alpha directory to an image model. "
            "Write 000000.png onward into expected_output_dir, one file per source frame, "
            "in the same order. Real-ESRGAN is not bundled."
        ),
    }
    manifest_path = workdir / "frame_upscale_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    config, loaded_from = load_config(config_path)
    workflow_value = str((config.get("frame_upscale") or {}).get("workflow") or "").strip()
    if not workflow_value:
        raise PipelineError(
            "frame-upscale prepared the RGB sequence and wrote "
            f"{manifest_path}. No image-upscaler workflow is configured, so no frames were "
            "upscaled and none were copied into upscaled_rgb. Set frame_upscale.workflow in "
            "the config, or place the upscaled RGB PNGs in "
            f"{workdir / 'upscaled_rgb'} and run finalize. Real-ESRGAN is not bundled."
        )

    settings = config["frame_upscale"]
    workflow_path = resolve_existing_file(workflow_value, loaded_from)
    template = load_workflow(workflow_path)
    assert_workflow_configured(template, workflow_path)
    output_dir = workdir / "upscaled_rgb"
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_pngs(output_dir, overwrite=overwrite or not any(output_dir.glob("*.png")))

    owns_runner = runner is None
    if runner is None:
        from midnight_upscale.comfy import ComfyClient

        comfy_cfg = config["comfyui"]
        runner = ComfyClient(
            comfy_url or str(comfy_cfg["url"]),
            timeout_sec=float(timeout_sec if timeout_sec is not None else comfy_cfg["timeout_sec"]),
            poll_interval_sec=float(comfy_cfg["poll_interval_sec"]),
        )
    try:
        assert_nodes_available(template, runner.object_info())
        input_mode = str((settings.get("input") or {}).get("mode") or "upload_image")
        if input_mode == "directory":
            _run_window(
                runner,
                template,
                settings,
                rgb_frames,
                workdir=workdir,
                batch_index=0,
                start_index=0,
                output_dir=output_dir,
                scale=metadata.requested_scale,
                width=metadata.original_width,
                height=metadata.original_height,
                batch_size=1,
                require_controls=False,
            )
        elif input_mode == "upload_image":
            _run_frames_individually(
                runner,
                template,
                settings,
                rgb_frames,
                workdir=workdir,
                output_dir=output_dir,
                scale=metadata.requested_scale,
                width=metadata.original_width,
                height=metadata.original_height,
            )
        else:
            raise PipelineError(
                f"frame_upscale.input.mode must be directory or upload_image, got {input_mode!r}"
            )
    finally:
        if owns_runner:
            runner.close()

    produced = list_indexed_frames(output_dir)
    if len(produced) != metadata.frame_count:
        raise ValidationError(
            f"frame-upscale wrote {len(produced)} RGB frames, expected {metadata.frame_count}"
        )
    metadata.backend = "frame-upscale"
    metadata.save(workdir / "metadata.json")


def _windows_for_chunking(count: int, batch_size: int, chunking: str) -> Iterator[tuple[int, int]]:
    if chunking == "all":
        yield 0, count
        return
    yield from iter_windows(count, batch_size)


def _run_window(
    runner: WorkflowRunner,
    template: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    frames: list[Path],
    *,
    workdir: Path,
    batch_index: int,
    start_index: int,
    output_dir: Path,
    scale: int,
    width: int,
    height: int,
    batch_size: int,
    require_controls: bool = True,
    temporal_overlap: int | None = None,
    temporal_overlap_node: str = "",
    temporal_overlap_field: str = "",
) -> None:
    batch_dir = workdir / "batches" / f"{batch_index:04d}"
    stage_dir = batch_dir / "input"
    raw_output = batch_dir / "output"
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    stage_dir.mkdir(parents=True)
    raw_output.mkdir(parents=True)
    for index, source in enumerate(frames):
        shutil.copy2(source, frame_path(stage_dir, index))

    workflow = json.loads(json.dumps(template))
    apply_seedvr2_overrides(
        workflow,
        settings,
        input_path=str(stage_dir.resolve()),
        output_path=str(raw_output.resolve()),
        uploaded_name=None,
        scale=scale,
        width=width,
        height=height,
        batch_size=batch_size,
        require_controls=require_controls,
        temporal_overlap=temporal_overlap,
        temporal_overlap_node=temporal_overlap_node,
        temporal_overlap_field=temporal_overlap_field,
    )
    _reject_alpha_input(workflow, workdir)
    if len(frames) < batch_size:
        logger.info(
            "Batch %s has %s frame(s), fewer than batch size %s. "
            "Frames are not duplicated to fill the window.",
            batch_index,
            len(frames),
            batch_size,
        )
    logger.info(
        "ComfyUI batch %s: frames %s-%s (%s files)",
        batch_index,
        start_index,
        start_index + len(frames) - 1,
        len(frames),
    )
    history = runner.run_workflow(workflow)
    output_mode = str((settings.get("output") or {}).get("mode") or "directory")
    if output_mode == "history":
        _download_history_images(
            runner,
            history,
            as_node_id((settings.get("output") or {}).get("node_id")),
            raw_output,
            expected=len(frames),
        )
    produced = find_pngs(raw_output)
    if len(produced) != len(frames):
        raise ValidationError(
            f"Batch {batch_index} produced {len(produced)} PNG(s) in {raw_output}, "
            f"expected {len(frames)}. Refusing to skip, replace, or reorder frames."
        )
    for offset, source in enumerate(produced):
        destination = frame_path(output_dir, start_index + offset)
        if destination.exists():
            raise ValidationError(
                f"Refusing to overwrite existing upscaled frame {destination.name}"
            )
        shutil.copy2(source, destination)


def _run_frames_individually(
    runner: WorkflowRunner,
    template: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    frames: list[Path],
    *,
    workdir: Path,
    output_dir: Path,
    scale: int,
    width: int,
    height: int,
) -> None:
    for index, source in enumerate(frames):
        raw_output = workdir / "batches" / "frames" / f"{index:06d}"
        if raw_output.exists():
            shutil.rmtree(raw_output)
        raw_output.mkdir(parents=True)
        uploaded = runner.upload_image(source, subfolder="midnight_upscale")
        workflow = json.loads(json.dumps(template))
        apply_seedvr2_overrides(
            workflow,
            settings,
            input_path=None,
            output_path=str(raw_output.resolve()),
            uploaded_name=uploaded,
            scale=scale,
            width=width,
            height=height,
            batch_size=1,
            require_controls=False,
        )
        _reject_alpha_input(workflow, workdir)
        history = runner.run_workflow(workflow)
        output_mode = str((settings.get("output") or {}).get("mode") or "history")
        if output_mode == "history":
            _download_history_images(
                runner,
                history,
                as_node_id((settings.get("output") or {}).get("node_id")),
                raw_output,
                expected=1,
            )
        produced = find_pngs(raw_output)
        if len(produced) != 1:
            raise ValidationError(
                f"Frame {index:06d} produced {len(produced)} PNG(s), expected 1. "
                "The frame was not replaced."
            )
        shutil.copy2(produced[0], frame_path(output_dir, index))


def _download_history_images(
    runner: WorkflowRunner,
    history: dict[str, Any],
    node_id: str,
    destination: Path,
    *,
    expected: int,
) -> None:
    images = images_from_history(history, node_id)
    images = sorted(images, key=lambda item: str(item.get("filename") or ""))
    if len(images) != expected:
        raise ValidationError(
            f"ComfyUI history node {node_id} returned {len(images)} images, expected {expected}"
        )
    for index, image_info in enumerate(images):
        runner.download(image_info, frame_path(destination, index))


def _reject_alpha_input(workflow: dict[str, dict[str, Any]], workdir: Path) -> None:
    """Stop if a mapped path points at the alpha or RGBA folders."""

    alpha = (workdir / "alpha").resolve()
    source_rgba = (workdir / "source_rgba").resolve()
    final_rgba = (workdir / "final_rgba").resolve()
    forbidden = {alpha, source_rgba, final_rgba}
    for node_id, node in workflow.items():
        for field, value in node.get("inputs", {}).items():
            if not isinstance(value, str) or not value:
                continue
            try:
                candidate = Path(value).resolve()
            except OSError:
                continue
            if candidate in forbidden or any(parent in forbidden for parent in candidate.parents):
                raise PipelineError(
                    f"Node {node_id} input {field} points at {candidate}. "
                    "SeedVR2 and the frame upscaler may only read the RGB frames. "
                    "Alpha is resized separately and is not sent to ComfyUI."
                )
