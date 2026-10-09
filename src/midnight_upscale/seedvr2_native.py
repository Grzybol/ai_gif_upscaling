"""Native ComfyUI SeedVR2 backend.

The graph follows the installed core nodes, not the numz SeedVR2VideoUpscaler
pack. RGB frames are carried in a temporary video because LoadVideo is the
native input. That file uses a constant 24 fps only so the container has a
clock. Final animation timing still comes from the source GIF durations in
metadata.json.

SeedVR2Preprocess repeats the last frame until the count is 4n+1. Those
repeated tail frames are removed after the run so the saved sequence matches
the source. Any other count aborts the job.

SaveVideo on this install only offers H.264. The upscaled frames are saved
with SaveImage so the RGB result is not passed through that lossy codec.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from midnight_upscale.comfy import ComfyClient, schema_has_input
from midnight_upscale.models import JobMetadata
from midnight_upscale.progress import checkpoint, current_bus, report
from midnight_upscale.utils import (
    ComfyError,
    PipelineError,
    ValidationError,
    WorkflowConfigError,
    clear_pngs,
    frame_path,
    list_indexed_frames,
    load_config,
    require_binary,
)

logger = logging.getLogger(__name__)

TRANSPORT_FPS = 24
PREFERRED_UNET = "seedvr2_3b_int8_convrot.safetensors"
PREFERRED_VAE = "seedvr2_ema_vae_fp16.safetensors"
FRAMES_PER_CHUNK = 5
# Official native SeedVR2 template: one-step restoration.
OFFICIAL_STEPS = 1
OFFICIAL_CFG = 1.0
OFFICIAL_SAMPLER = "euler"
OFFICIAL_SCHEDULER = "simple"
OFFICIAL_DENOISE = 1.0
# Schema default is lab. The 4060 Ti starting point skips that extra pass.
COLOR_CORRECTION = "none"
# Full-frame VAE.decode OOMs on a long 2x clip and then stalls while releasing
# that allocation. VAEDecodeTiled skips that path. SeedVR2 keeps its own
# causal temporal cache and only tiles in space. 512 gives the 4060 Ti a
# larger conv than 256, so the card spends more of each step computing.
DECODE_TILE = 512
DECODE_OVERLAP = 64
DECODE_TEMPORAL_SIZE = 64
DECODE_TEMPORAL_OVERLAP = 8

REQUIRED_NODES = (
    "SeedVR2Preprocess",
    "SeedVR2Conditioning",
    "SeedVR2PostProcessing",
)
CHUNK_NODES = ("SeedVR2TemporalChunk", "SeedVR2TemporalMerge")
SUPPORT_NODES = (
    "LoadVideo",
    "GetVideoComponents",
    "ImageScaleBy",
    "UNETLoader",
    "VAELoader",
    "VAEEncodeTiled",
    "VAEDecodeTiled",
    "KSampler",
    "SaveImage",
)
TEMPORAL_MODES = ("auto", "unchunked", "chunked")
BACKEND_NATIVE = "seedvr2-native"
BACKEND_NUMZ = "seedvr2"


@dataclass
class NativeReport:
    reachable: bool
    gpu: str = "unknown"
    native_nodes: list[str] = field(default_factory=list)
    chunk_nodes: list[str] = field(default_factory=list)
    numz_installed: bool = False
    unet: str | None = None
    vae: str | None = None
    ffmpeg_ok: bool = False
    workflow_ok: bool = False
    workflow_note: str = ""
    ready: bool = False
    selected_backend: str = ""
    missing: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def selected_label(self) -> str:
        if self.selected_backend == BACKEND_NATIVE:
            return "Native ComfyUI"
        if self.selected_backend == BACKEND_NUMZ:
            return "Numz custom node"
        return "none"

    @property
    def model_label(self) -> str:
        if self.unet and "3b" in self.unet.lower() and "int8" in self.unet.lower():
            return "3B INT8"
        return self.unet or "NOT FOUND"


def padded_frame_count(count: int) -> int:
    """Frame count after SeedVR2Preprocess repeats the last frame to 4n+1."""

    if count < 1:
        raise ValidationError(f"Frame count must be at least 1, got {count}")
    remainder = (count - 1) % 4
    if remainder == 0:
        return count
    return count + (4 - remainder)


def choose_temporal_mode(
    requested: str,
    *,
    frame_count: int,
    width: int,
    height: int,
    scale: int,
    chunk_nodes: bool,
) -> str:
    """Pick unchunked or chunked. Auto prefers unchunked for short clips.

    A short Midnight Lounge loop stays one sequence when the long edge at the
    requested scale is at most 2048 and the clip has at most 25 frames. That
    keeps the whole loop temporally connected on a 16 GB RTX 4060 Ti. Longer
    clips use the native chunk and merge nodes when they are installed.
    """

    if requested not in TEMPORAL_MODES:
        raise PipelineError("Temporal processing must be Auto, Unchunked, or Chunked.")
    if requested == "unchunked":
        return "unchunked"
    if requested == "chunked":
        if not chunk_nodes:
            raise WorkflowConfigError(
                "Chunked mode needs SeedVR2TemporalChunk and SeedVR2TemporalMerge. "
                "Those nodes are not in /object_info."
            )
        return "chunked"
    long_edge = max(width, height) * scale
    if frame_count <= 25 and long_edge <= 2048:
        return "unchunked"
    if not chunk_nodes:
        return "unchunked"
    return "chunked"


def native_nodes_present(object_info: dict[str, Any]) -> bool:
    return all(name in object_info for name in REQUIRED_NODES)


def resolve_backend(object_info: dict[str, Any]) -> str:
    """Prefer native SeedVR2 when its nodes are installed."""

    if native_nodes_present(object_info):
        return BACKEND_NATIVE
    if "SeedVR2VideoUpscaler" in object_info:
        return BACKEND_NUMZ
    raise WorkflowConfigError(
        "No SeedVR2 implementation was found. Native nodes SeedVR2Preprocess, "
        "SeedVR2Conditioning, and SeedVR2PostProcessing are missing, and "
        "SeedVR2VideoUpscaler is not installed."
    )


def resolve_requested_backend(comfy_url: str | None, config_path: Path | None) -> str:
    config, _loaded = load_config(config_path)
    url = comfy_url or str(config["comfyui"]["url"])
    comfy_cfg = config["comfyui"]
    client = ComfyClient(
        url,
        timeout_sec=float(comfy_cfg["timeout_sec"]),
        poll_interval_sec=float(comfy_cfg["poll_interval_sec"]),
    )
    try:
        return resolve_backend(client.object_info())
    finally:
        client.close()


def select_unet(names: list[str]) -> str | None:
    """Prefer 3B INT8. Never substitute a 7B file."""

    seed = [name for name in names if "seedvr2" in name.lower() and "vae" not in name.lower()]
    preferred = [name for name in seed if "3b" in name.lower() and "int8" in name.lower()]
    if preferred:
        return sorted(preferred)[0]
    three_b = [name for name in seed if "3b" in name.lower()]
    if three_b:
        return sorted(three_b)[0]
    return None


def select_vae(names: list[str]) -> str | None:
    seed = [name for name in names if "seedvr2" in name.lower()]
    ema = [name for name in seed if "ema" in name.lower() or "vae" in name.lower()]
    if ema:
        return sorted(ema)[0]
    return None


def missing_model_lines(unet_names: list[str], vae_names: list[str]) -> list[str]:
    lines: list[str] = []
    if select_unet(unet_names) is None:
        lines.append(f"Need diffusion model: {PREFERRED_UNET}")
        lines.append("A 7B file is not used in its place.")
        if unet_names:
            lines.append("UNET loader currently lists: " + ", ".join(unet_names[:8]))
    if select_vae(vae_names) is None:
        lines.append(f"Need VAE: {PREFERRED_VAE}")
        if vae_names:
            lines.append("VAE loader currently lists: " + ", ".join(vae_names[:8]))
    return lines


def assess_installation(
    object_info: dict[str, Any] | None,
    stats: dict[str, Any] | None = None,
    *,
    ffmpeg_ok: bool | None = None,
    error: str = "",
) -> NativeReport:
    if object_info is None:
        return NativeReport(reachable=False, error=error or "ComfyUI is not reachable")
    ffmpeg = shutil.which("ffmpeg") is not None if ffmpeg_ok is None else ffmpeg_ok
    unet_names = combo_choices(input_spec(object_info, "UNETLoader", "unet_name"))
    vae_names = combo_choices(input_spec(object_info, "VAELoader", "vae_name"))
    unet = select_unet(unet_names)
    vae = select_vae(vae_names)
    present = [name for name in REQUIRED_NODES if name in object_info]
    chunks = [name for name in CHUNK_NODES if name in object_info]
    numz = "SeedVR2VideoUpscaler" in object_info
    try:
        selected = resolve_backend(object_info)
    except WorkflowConfigError:
        selected = ""
    missing = missing_model_lines(unet_names, vae_names) if present else []
    workflow_ok = False
    workflow_note = ""
    if len(present) == len(REQUIRED_NODES) and unet and vae:
        try:
            build_native_workflow(
                object_info,
                video_name="transport.mp4",
                unet_name=unet,
                vae_name=vae,
                scale=2,
                temporal_mode="unchunked",
                temporal_overlap=0,
                seed=0,
            )
            if len(chunks) == len(CHUNK_NODES):
                build_native_workflow(
                    object_info,
                    video_name="transport.mp4",
                    unet_name=unet,
                    vae_name=vae,
                    scale=2,
                    temporal_mode="chunked",
                    temporal_overlap=0,
                    seed=0,
                )
            workflow_ok = True
            workflow_note = "OK"
        except WorkflowConfigError as exc:
            workflow_note = str(exc)
    elif len(present) == len(REQUIRED_NODES):
        workflow_note = "INCOMPLETE"
    report = NativeReport(
        reachable=True,
        gpu=gpu_name(stats or {}),
        native_nodes=present,
        chunk_nodes=chunks,
        numz_installed=numz,
        unet=unet,
        vae=vae,
        ffmpeg_ok=ffmpeg,
        workflow_ok=workflow_ok,
        workflow_note=workflow_note,
        selected_backend=selected,
        missing=missing,
    )
    report.ready = bool(
        report.selected_backend == BACKEND_NATIVE and unet and vae and ffmpeg and workflow_ok
    )
    return report


def format_comfy_check(report: NativeReport) -> str:
    if not report.reachable:
        detail = report.error or "Could not reach ComfyUI"
        return f"ComfyUI: NOT REACHABLE\n{detail}\nReady: NO"
    native = "OK" if len(report.native_nodes) == len(REQUIRED_NODES) else "NOT FOUND"
    lines = [
        "ComfyUI: OK",
        f"GPU: {report.gpu}",
        f"SeedVR2 native: {native}",
        f"Model: {report.unet or 'NOT FOUND'}",
        f"VAE: {report.vae or 'NOT FOUND'}",
        f"FFmpeg: {'OK' if report.ffmpeg_ok else 'NOT FOUND'}",
        f"Backend: {report.selected_backend or 'none'}",
        f"Workflow: {report.workflow_note or 'INCOMPLETE'}",
        f"Ready: {'YES' if report.ready else 'NO'}",
    ]
    lines.extend(report.missing)
    if report.native_nodes:
        lines.append("Nodes: " + ", ".join(report.native_nodes + report.chunk_nodes))
    return "\n".join(lines)


def format_gui_status(report: NativeReport) -> list[str]:
    if not report.reachable:
        return [report.error or "ComfyUI is not reachable"]
    native = "AVAILABLE" if len(report.native_nodes) == len(REQUIRED_NODES) else "NOT INSTALLED"
    numz = "AVAILABLE" if report.numz_installed else "NOT INSTALLED"
    lines = [
        "SeedVR2 implementations:",
        f"Native ComfyUI: {native}",
        f"Numz custom node: {numz}",
        f"Selected: {report.selected_label}",
        f"Model: {report.model_label}",
    ]
    if report.unet and report.model_label != report.unet:
        lines.append(f"Model file: {report.unet}")
    lines.append(f"VAE: {report.vae or 'NOT FOUND'}")
    lines.append(f"Status: {'READY' if report.ready else 'NOT READY'}")
    lines.extend(report.missing)
    return lines


def run_comfy_check(comfy_url: str | None, config_path: Path | None) -> tuple[str, bool]:
    config, _loaded = load_config(config_path)
    url = comfy_url or str(config["comfyui"]["url"])
    comfy_cfg = config["comfyui"]
    client = ComfyClient(
        url,
        timeout_sec=float(comfy_cfg["timeout_sec"]),
        poll_interval_sec=float(comfy_cfg["poll_interval_sec"]),
    )
    try:
        info = client.object_info()
        try:
            stats = client.system_stats()
        except ComfyError:
            stats = {}
    except ComfyError as exc:
        report = assess_installation(None, error=str(exc))
        return format_comfy_check(report), False
    finally:
        client.close()
    report = assess_installation(info, stats)
    return format_comfy_check(report), report.ready


def build_native_workflow(
    object_info: dict[str, Any],
    *,
    video_name: str,
    unet_name: str,
    vae_name: str,
    scale: int,
    temporal_mode: str,
    temporal_overlap: int,
    seed: int,
    frames_per_chunk: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Build an API prompt from the live schema. Node ids are local to this graph."""

    for name in (*REQUIRED_NODES, *SUPPORT_NODES):
        if name not in object_info:
            raise WorkflowConfigError(
                f"ComfyUI has no {name} node. The native SeedVR2 workflow was not queued."
            )
    chunked = temporal_mode == "chunked"
    if chunked and any(name not in object_info for name in CHUNK_NODES):
        raise WorkflowConfigError(
            "Chunked mode needs SeedVR2TemporalChunk and SeedVR2TemporalMerge."
        )
    _require_choice(object_info, "UNETLoader", "unet_name", unet_name)
    _require_choice(object_info, "VAELoader", "vae_name", vae_name)
    _require_choice(object_info, "ImageScaleBy", "upscale_method", "lanczos")
    _require_choice(object_info, "KSampler", "sampler_name", OFFICIAL_SAMPLER)
    _require_choice(object_info, "KSampler", "scheduler", OFFICIAL_SCHEDULER)
    _require_choice(
        object_info, "SeedVR2PostProcessing", "color_correction_method", COLOR_CORRECTION
    )
    weight = "default"
    _require_choice(object_info, "UNETLoader", "weight_dtype", weight)

    workflow: dict[str, dict[str, Any]] = {
        "1": _node(
            object_info,
            "LoadVideo",
            {"file": video_name},
        ),
        "2": _node(object_info, "GetVideoComponents", {"video": ["1", 0]}),
        "3": _node(
            object_info,
            "ImageScaleBy",
            {"image": ["2", 0], "upscale_method": "lanczos", "scale_by": float(scale)},
        ),
        "4": _node(object_info, "SeedVR2Preprocess", {"resized_images": ["3", 0]}),
        "5": _node(
            object_info,
            "UNETLoader",
            {"unet_name": unet_name, "weight_dtype": weight},
        ),
        "6": _node(object_info, "VAELoader", {"vae_name": vae_name}),
        "7": _node(
            object_info,
            "VAEEncodeTiled",
            _with_vae_tiles(object_info, "VAEEncodeTiled", {"pixels": ["4", 0], "vae": ["6", 0]}),
        ),
    }
    latent_from: list[object] = ["7", 0]
    if chunked:
        keys = dynamic_option_keys(input_spec(object_info, "SeedVR2TemporalChunk", "chunking_mode"))
        # The dynamic combo's nested widget is a dotted prompt key, not a sibling.
        # "auto" lets the installed node pick a 4n+1 chunk that fits free VRAM.
        # A fixed size is sent only when the caller asks for one.
        manual_key = "chunking_mode.frames_per_chunk"
        chunk_inputs: dict[str, object] = {
            "latent": ["7", 0],
            "chunking_mode": "auto" if frames_per_chunk is None else "manual",
        }
        extra_ok: set[str] = set()
        if frames_per_chunk is None:
            if keys and "auto" not in keys:
                raise WorkflowConfigError(
                    "SeedVR2TemporalChunk chunking_mode has no auto option. "
                    f"Choices: {', '.join(keys)}"
                )
        else:
            if "frames_per_chunk" not in dynamic_manual_fields(object_info, "SeedVR2TemporalChunk"):
                raise WorkflowConfigError(
                    "SeedVR2TemporalChunk does not expose frames_per_chunk. "
                    "The chunked workflow was not queued."
                )
            if keys and "manual" not in keys:
                raise WorkflowConfigError(
                    "SeedVR2TemporalChunk chunking_mode has no manual option. "
                    f"Choices: {', '.join(keys)}"
                )
            if frames_per_chunk < 1 or (frames_per_chunk - 1) % 4 != 0:
                raise WorkflowConfigError(
                    f"frames_per_chunk must be 4n+1 (1, 5, 9, 13, ...), got {frames_per_chunk}."
                )
            chunk_inputs[manual_key] = frames_per_chunk
            extra_ok.add(manual_key)
        if schema_has_input(object_info, "SeedVR2TemporalChunk", "temporal_overlap"):
            chunk_inputs["temporal_overlap"] = temporal_overlap
        workflow["8"] = _node(
            object_info,
            "SeedVR2TemporalChunk",
            chunk_inputs,
            extra_ok=extra_ok,
        )
        latent_from = ["8", 0]
    workflow["9"] = _node(
        object_info,
        "SeedVR2Conditioning",
        {"model": ["5", 0], "vae_conditioning": latent_from},
    )
    workflow["10"] = _node(
        object_info,
        "KSampler",
        {
            "model": ["5", 0],
            "seed": seed,
            "steps": OFFICIAL_STEPS,
            "cfg": OFFICIAL_CFG,
            "sampler_name": OFFICIAL_SAMPLER,
            "scheduler": OFFICIAL_SCHEDULER,
            "positive": ["9", 0],
            "negative": ["9", 1],
            "latent_image": latent_from,
            "denoise": OFFICIAL_DENOISE,
        },
    )
    decode_from: list[object] = ["10", 0]
    if chunked:
        merge_inputs: dict[str, object] = {"latents": ["10", 0]}
        if schema_has_input(object_info, "SeedVR2TemporalMerge", "temporal_overlap"):
            merge_inputs["temporal_overlap"] = ["8", 1]
        workflow["11"] = _node(object_info, "SeedVR2TemporalMerge", merge_inputs)
        decode_from = ["11", 0]
    workflow["12"] = _node(
        object_info,
        "VAEDecodeTiled",
        _with_vae_tiles(object_info, "VAEDecodeTiled", {"samples": decode_from, "vae": ["6", 0]}),
    )
    workflow["13"] = _node(
        object_info,
        "SeedVR2PostProcessing",
        {
            "images": ["12", 0],
            "original_resized_images": ["3", 0],
            "color_correction_method": COLOR_CORRECTION,
        },
    )
    workflow["14"] = _node(
        object_info,
        "SaveImage",
        {"images": ["13", 0], "filename_prefix": "midnight_seedvr2"},
    )
    return workflow


def run_seedvr2_native(
    workdir: Path,
    metadata: JobMetadata,
    *,
    comfy_url: str | None,
    config_path: Path | None,
    timeout_sec: float | None,
    overwrite: bool,
    temporal_overlap: int,
    temporal_mode: str = "auto",
    client: ComfyClient | None = None,
) -> None:
    rgb_frames = list_indexed_frames(workdir / "rgb")
    if len(rgb_frames) != metadata.frame_count:
        raise ValidationError(
            f"RGB directory has {len(rgb_frames)} frames, metadata says {metadata.frame_count}"
        )
    _assert_rgb_only(rgb_frames)
    output_dir = workdir / "upscaled_rgb"
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_pngs(output_dir, overwrite=overwrite or not any(output_dir.glob("*.png")))

    config, _loaded = load_config(config_path)
    comfy_cfg = config["comfyui"]
    owns_client = client is None
    if client is None:
        client = ComfyClient(
            comfy_url or str(comfy_cfg["url"]),
            timeout_sec=float(timeout_sec or comfy_cfg["timeout_sec"]),
            poll_interval_sec=float(comfy_cfg["poll_interval_sec"]),
        )
    try:
        object_info = client.object_info()
        if not native_nodes_present(object_info):
            raise WorkflowConfigError(
                "Native SeedVR2 nodes are not installed. "
                "SeedVR2Preprocess, SeedVR2Conditioning, and SeedVR2PostProcessing "
                "were not in /object_info."
            )
        unet_names = combo_choices(input_spec(object_info, "UNETLoader", "unet_name"))
        vae_names = combo_choices(input_spec(object_info, "VAELoader", "vae_name"))
        unet = select_unet(unet_names)
        vae = select_vae(vae_names)
        if unet is None or vae is None:
            raise WorkflowConfigError(
                "\n".join(missing_model_lines(unet_names, vae_names) + ["The job was not queued."])
            )
        mode = choose_temporal_mode(
            temporal_mode,
            frame_count=metadata.frame_count,
            width=metadata.original_width,
            height=metadata.original_height,
            scale=metadata.requested_scale,
            chunk_nodes=all(name in object_info for name in CHUNK_NODES),
        )
        logger.info("Temporal mode: %s", mode.upper())
        overlap = temporal_overlap
        chunk_has_overlap = schema_has_input(
            object_info, "SeedVR2TemporalChunk", "temporal_overlap"
        )
        if mode == "chunked":
            logger.info("Chunk sizing: auto")
        if mode == "chunked" and chunk_has_overlap:
            logger.info("Temporal overlap: %s", overlap)
        elif mode == "chunked":
            overlap = 0
        transport = workdir / "transport" / "rgb_transport.mp4"
        report(
            "Build SeedVR2 input",
            message=f"Building transport for {metadata.frame_count} RGB frames",
            frames_total=metadata.frame_count,
            frames_kind="submitted",
        )
        encode_rgb_transport(rgb_frames, transport)
        uploaded = client.upload_input(transport, content_type="video/mp4")
        workflow = build_native_workflow(
            object_info,
            video_name=uploaded,
            unet_name=unet,
            vae_name=vae,
            scale=metadata.requested_scale,
            temporal_mode=mode,
            temporal_overlap=overlap,
            seed=0,
        )
        logger.info("SeedVR2 native model: %s", unet)
        logger.info("SeedVR2 native VAE: %s", vae)
        logger.info(
            "Transport video: %s fps, lossless H.264 (crf 0). "
            "Final timing still comes from the GIF metadata.",
            TRANSPORT_FPS,
        )
        logger.info(
            "VAE encode and decode: tiled, tile %s, overlap %s.",
            DECODE_TILE,
            DECODE_OVERLAP,
        )
        report(
            "Build SeedVR2 input",
            message=(
                f"Transport video ready. {metadata.frame_count} frames submitted. "
                f"Temporal mode: {mode}."
            ),
            frames_total=metadata.frame_count,
            frames_kind="submitted",
        )
        bus = current_bus()
        if bus is not None:
            bus.set_nodes(
                {
                    node_id: str(node.get("class_type") or "")
                    for node_id, node in workflow.items()
                    if isinstance(node, dict)
                }
            )
        report(
            "Queue ComfyUI job",
            message="Queueing the SeedVR2 prompt",
            frames_total=metadata.frame_count,
            frames_kind="submitted",
        )
        history = client.run_workflow(workflow)
        report("Retrieve SeedVR2 result", message="SeedVR2 output received")
        saved = _images_for_node(history, "14")
        if len(saved) < 1:
            raise ComfyError("SeedVR2 native finished without SaveImage frames.")
        staging = workdir / "transport" / "decoded"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        downloaded: list[Path] = []
        for index, image_info in enumerate(saved):
            checkpoint()
            dest = staging / f"{index:06d}.png"
            client.download(image_info, dest)
            downloaded.append(dest)
            report(
                "Retrieve SeedVR2 result",
                message=f"Retrieved frame {index + 1}/{len(saved)}",
                frames_done=index + 1,
                frames_total=len(saved),
                frames_kind="completed",
            )
        kept = take_source_frames(downloaded, metadata.frame_count)
        report(
            "Decode upscaled RGB",
            message=f"Keeping {len(kept)} RGB frames",
            frames_total=len(kept),
            frames_kind="completed",
        )
        for index, path in enumerate(kept):
            destination = frame_path(output_dir, index)
            _copy_rgb(path, destination)
            report(
                "Decode upscaled RGB",
                message=f"RGB frame {index + 1}/{len(kept)}",
                frames_done=index + 1,
                frames_total=len(kept),
                frames_kind="completed",
                preview_path=str(destination),
            )
    finally:
        if owns_client:
            client.close()
    metadata.backend = BACKEND_NATIVE
    metadata.chunking = mode
    metadata.seedvr2_resolution = metadata.requested_scale
    metadata.save(workdir / "metadata.json")


def encode_rgb_transport(frames: list[Path], dest: Path) -> None:
    """Write a temporary RGB video. FPS is transport only and is not the GIF timing."""

    ffmpeg = require_binary("ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pattern = frames[0].parent / "%06d.png"
    last_error = ""
    for pix_fmt in ("yuv444p", "yuv420p"):
        command = transport_ffmpeg_command(ffmpeg, pattern, len(frames), dest, pix_fmt=pix_fmt)
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            if pix_fmt != "yuv444p":
                logger.info("Transport video used yuv420p because the yuv444p encode was rejected.")
            return
        last_error = (completed.stderr or "")[-2000:]
    raise PipelineError(f"ffmpeg could not encode the SeedVR2 transport video.\n{last_error}")


def transport_ffmpeg_command(
    ffmpeg: str,
    pattern: Path,
    count: int,
    dest: Path,
    *,
    pix_fmt: str,
) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-framerate",
        str(TRANSPORT_FPS),
        "-start_number",
        "0",
        "-i",
        str(pattern),
        "-frames:v",
        str(count),
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        pix_fmt,
        str(dest),
    ]


def take_source_frames(paths: list[Path], source_count: int) -> list[Path]:
    padded = padded_frame_count(source_count)
    if len(paths) == source_count:
        return list(paths)
    if len(paths) == padded:
        logger.info(
            "SeedVR2Preprocess padded %s frames to %s by repeating the last frame. "
            "Those extra frames are dropped. No frames were interpolated.",
            source_count,
            padded,
        )
        return list(paths[:source_count])
    raise ValidationError(
        f"SeedVR2 returned {len(paths)} frames for {source_count} source frames. "
        "The job was aborted. No frames were interpolated."
    )


def input_spec(object_info: dict[str, Any], class_type: str, field: str) -> Any:
    node = object_info.get(class_type)
    if not isinstance(node, dict):
        return None
    groups = node.get("input")
    if not isinstance(groups, dict):
        return None
    for section in ("required", "optional"):
        inputs = groups.get(section)
        if isinstance(inputs, dict) and field in inputs:
            return inputs[field]
    return None


def combo_choices(spec: Any) -> list[str]:
    if not isinstance(spec, list) or not spec:
        return []
    if isinstance(spec[0], list) and all(isinstance(item, str) for item in spec[0]):
        return list(spec[0])
    if len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options")
        if isinstance(options, list) and all(isinstance(item, str) for item in options):
            return list(options)
    return []


def dynamic_option_keys(spec: Any) -> list[str]:
    if not isinstance(spec, list) or len(spec) < 2 or not isinstance(spec[1], dict):
        return []
    options = spec[1].get("options") or []
    return [str(item["key"]) for item in options if isinstance(item, dict) and "key" in item]


def dynamic_manual_fields(object_info: dict[str, Any], class_type: str) -> set[str]:
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("key") == "manual":
                inputs = value.get("inputs") or {}
                if isinstance(inputs, dict):
                    for section in ("required", "optional"):
                        block = inputs.get(section)
                        if isinstance(block, dict):
                            found.update(str(name) for name in block)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(object_info.get(class_type))
    return found


def gpu_name(stats: dict[str, Any]) -> str:
    devices = stats.get("devices")
    if not isinstance(devices, list) or not devices or not isinstance(devices[0], dict):
        return "unknown"
    raw = str(devices[0].get("name") or "unknown")
    text = re.sub(r"^cuda:\d+\s*", "", raw, flags=re.IGNORECASE)
    text = text.split(":")[0].strip()
    return text or raw


def _with_vae_tiles(
    object_info: dict[str, Any],
    class_type: str,
    inputs: dict[str, object],
) -> dict[str, object]:
    tiled = dict(inputs)
    for input_name, value in (
        ("tile_size", DECODE_TILE),
        ("overlap", DECODE_OVERLAP),
        ("temporal_size", DECODE_TEMPORAL_SIZE),
        ("temporal_overlap", DECODE_TEMPORAL_OVERLAP),
    ):
        if schema_has_input(object_info, class_type, input_name):
            tiled[input_name] = value
    return tiled


def _node(
    object_info: dict[str, Any],
    class_type: str,
    inputs: dict[str, object],
    *,
    extra_ok: set[str] | None = None,
) -> dict[str, Any]:
    allowed = extra_ok or set()
    missing = [
        key
        for key in inputs
        if key not in allowed and not schema_has_input(object_info, class_type, key)
    ]
    if missing:
        raise WorkflowConfigError(
            f"{class_type} has no input {', '.join(missing)}. The workflow was not queued."
        )
    return {"class_type": class_type, "inputs": inputs}


def _require_choice(object_info: dict[str, Any], class_type: str, field: str, value: str) -> None:
    if not schema_has_input(object_info, class_type, field):
        raise WorkflowConfigError(f"{class_type} has no input {field}.")
    choices = combo_choices(input_spec(object_info, class_type, field))
    if not choices:
        raise WorkflowConfigError(f"Could not read {class_type}.{field} choices from /object_info.")
    if value not in choices:
        shown = ", ".join(choices[:12])
        raise WorkflowConfigError(
            f"{class_type}.{field} does not offer {value!r}. Installed choices: {shown}"
        )


def _assert_rgb_only(frames: list[Path]) -> None:
    for path in frames:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ValidationError(
                    f"{path.name} is {image.mode}. SeedVR2 native only receives RGB. "
                    "Alpha stays on the local resize path."
                )


def _copy_rgb(source: Path, dest: Path) -> None:
    with Image.open(source) as image:
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        if image.mode != "RGB":
            logger.info("Dropped an extra channel from %s before saving RGB.", source.name)
        rgb.save(dest, format="PNG")


def _images_for_node(history: dict[str, Any], node_id: str) -> list[dict[str, Any]]:
    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        return []
    node_outputs = outputs.get(node_id)
    if not isinstance(node_outputs, dict):
        return []
    images = node_outputs.get("images")
    if not isinstance(images, list):
        return []
    ordered = [item for item in images if isinstance(item, dict)]
    return sorted(ordered, key=lambda item: str(item.get("filename") or ""))
