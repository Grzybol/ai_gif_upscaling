"""Bounded, frame-oriented upscaling backends for the Upscale tab."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from midnight_upscale.comfy import ComfyClient, images_from_history
from midnight_upscale.encode import write_concat_file
from midnight_upscale.models import JobMetadata
from midnight_upscale.seedvr2_native import run_seedvr2_native
from midnight_upscale.utils import PipelineError, frame_path, require_binary
from midnight_upscale.video_decode import decode_plan, decode_single_frame, plan_frames
from midnight_upscale.video_export import export_png_sequence, export_spritesheet, export_video, zip_spritesheet
from midnight_upscale.video_inspect import inspect_video
from midnight_upscale.spritesheet import SpritesheetSettings
from midnight_upscale.outline_cleanup import OutlineCleanupSettings, clean_outline

log = logging.getLogger(__name__)
MODES = ("FAST — Anime4K", "BALANCED — Real-ESRGAN", "QUALITY — SeedVR2")
FAST_MODELS = ("acnet-f8b8-box", "acnet-f8b8", "acnet-legacy-gan")
ESRGAN_MODEL = "RealESRGAN_x4plus_anime_6B.pth"
ANIME4K = Path(os.environ.get("MIDNIGHT_ANIME4K_CLI", str(Path(__file__).resolve().parents[2] / "tools" / "anime4k" / "ac_cli.exe")))
QUALITY_TIMEOUT = 180


@dataclass
class Clip:
    frames: list[np.ndarray]
    durations_ms: list[int]
    fps: float | None


class DiskFrames(Sequence[np.ndarray]):
    """Load only the frame a PNG/sheet/video encoder currently needs."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        with Image.open(self.paths[index]) as image:
            return np.asarray(image.convert("RGBA")).copy()


def source_summary(source: Path) -> tuple[int, int, int, float]:
    if source.suffix.lower() == ".png":
        with Image.open(source) as image:
            return 1, image.width, image.height, 0.0
    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            manifest = next((name for name in archive.namelist() if name.endswith(".json")), None)
            if manifest is None:
                raise PipelineError("Spritesheet ZIP has no JSON metadata.")
            data = json.loads(archive.read(manifest))
        return (int(data["frame_count"]), int(data["frame_width"]),
                int(data["frame_height"]), int(data["duration_ms"]) / 1000)
    info = inspect_video(source)
    return (info.frame_count or max(1, round((info.fps or 1) * info.duration_sec)),
            info.width, info.height, info.duration_sec)


def read_source_frame(source: Path, index: int, workdir: Path) -> np.ndarray:
    suffix = source.suffix.lower()
    if suffix in {".png", ".zip"}:
        clip = read_clip(source, workdir)
        return clip.frames[min(max(index, 0), len(clip.frames) - 1)]
    if suffix not in {".gif", ".mp4", ".webm"}:
        raise PipelineError(f"Unsupported input: {source.suffix}")
    info = inspect_video(source)
    return np.asarray(decode_single_frame(info, index).convert("RGBA")).copy()


def read_clip(source: Path, workdir: Path) -> Clip:
    suffix = source.suffix.lower()
    if suffix == ".png":
        with Image.open(source) as image:
            return Clip([np.asarray(image.convert("RGBA")).copy()], [1000], None)
    if suffix == ".zip":
        return _read_sheet(source)
    if suffix not in {".gif", ".mp4", ".webm"}:
        raise PipelineError("Choose GIF, MP4, WebM, PNG, or a spritesheet ZIP with JSON metadata.")
    info = inspect_video(source)
    plan = plan_frames(info)
    paths = decode_plan(info, plan, workdir / "decoded", scratch=workdir / "scratch_decode")
    with_frames = []
    for path in paths:
        with Image.open(path) as image:
            with_frames.append(np.asarray(image.convert("RGBA")).copy())
    return Clip(with_frames, plan.durations_ms, info.fps)


def _read_sheet(source: Path) -> Clip:
    with ZipFile(source) as archive:
        names = archive.namelist()
        manifests = [name for name in names if name.lower().endswith(".json")]
        if len(manifests) != 1:
            raise PipelineError("Spritesheet ZIP must contain one JSON metadata file.")
        meta = json.loads(archive.read(manifests[0]))
        frames_meta = meta.get("frames") or []
        if not frames_meta:
            raise PipelineError("Spritesheet metadata has no frames.")
        sheets: dict[str, Image.Image] = {}
        frames: list[np.ndarray] = []
        durations: list[int] = []
        try:
            from io import BytesIO

            for item in sorted(frames_meta, key=lambda value: value["index"]):
                sheet = item["sheet"]
                if isinstance(sheet, int):
                    descriptor = meta["sheets"][sheet]
                    name = descriptor["file"] if isinstance(descriptor, dict) else descriptor
                else:
                    name = sheet
                if name not in sheets:
                    sheets[name] = Image.open(BytesIO(archive.read(name))).convert("RGBA")
                x, y = int(item["x"]), int(item["y"])
                w, h = int(item["w"]), int(item["h"])
                frames.append(np.asarray(sheets[name].crop((x, y, x + w, y + h))).copy())
                durations.append(int(item["duration_ms"]))
        finally:
            for image in sheets.values():
                image.close()
    indices = [item["index"] for item in sorted(frames_meta, key=lambda value: value["index"])]
    if indices != list(range(len(frames))):
        raise PipelineError("Spritesheet frame indices have gaps or duplicates.")
    return Clip(frames, durations, meta.get("fps"))


def _rgb_input(frame: np.ndarray) -> Image.Image:
    """Inpaint invisible RGB before upscaling; alpha is resized independently."""
    array = frame[:, :, :3].copy()
    visible = frame[:, :, 3] > 0
    if not np.all(visible):
        # Extend silhouette colors a few pixels under invisible alpha, avoiding
        # both white GIF matte fringes and a synthetic black halo.
        array[~visible] = 0
        for _ in range(4):
            if np.all(visible):
                break
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                src_y = slice(max(0, -dy), array.shape[0] - max(0, dy))
                src_x = slice(max(0, -dx), array.shape[1] - max(0, dx))
                dst_y = slice(max(0, dy), array.shape[0] - max(0, -dy))
                dst_x = slice(max(0, dx), array.shape[1] - max(0, -dx))
                fill = ~visible[dst_y, dst_x] & visible[src_y, src_x]
                array[dst_y, dst_x][fill] = array[src_y, src_x][fill]
                visible[dst_y, dst_x][fill] = True
    return Image.fromarray(array, "RGB")


def _combine(rgb: Image.Image, original: np.ndarray, scale: int) -> np.ndarray:
    height, width = original.shape[:2]
    expected = (width * scale, height * scale)
    if rgb.size != expected:
        rgb = rgb.resize(expected, Image.Resampling.LANCZOS)
    alpha = Image.fromarray(original[:, :, 3], "L").resize(
        expected, Image.Resampling.LANCZOS
    )
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    return np.asarray(result).copy()


def _anime4k(rgb: Image.Image, scale: int, model: str, workdir: Path,
             cancel: threading.Event) -> Image.Image:
    if model not in FAST_MODELS:
        raise PipelineError(f"Unknown Anime4K model: {model}")
    if not ANIME4K.is_file():
        raise PipelineError(f"Anime4K CLI missing: {ANIME4K}")
    source, target = workdir / "fast_in.png", workdir / "fast_out.png"
    rgb.save(source)
    command = [str(ANIME4K), "-i", str(source), "-o", str(target), "-p", "cuda",
               "-d", "0", "-m", model, "-f", str(scale)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    deadline = time.monotonic() + 90
    while process.poll() is None:
        if cancel.is_set() or time.monotonic() > deadline:
            process.kill()
            process.communicate()
            raise PipelineError("Anime4K cancelled or exceeded its 90 s per-frame limit.")
        time.sleep(0.1)
    out, err = process.communicate()
    if process.returncode or not target.is_file():
        raise PipelineError(f"Anime4K GPU failed: {(err or out)[-1000:]}")
    with Image.open(target) as loaded:
        return loaded.convert("RGB").copy()


def _esrgan(rgb: Image.Image, scale: int, workdir: Path, comfy_url: str,
            timeout: int, cancel: threading.Event) -> Image.Image:
    if cancel.is_set():
        raise PipelineError("Cancelled.")
    source, target = workdir / "esrgan_in.png", workdir / "esrgan_out.png"
    rgb.save(source)
    client = ComfyClient(comfy_url, timeout_sec=timeout, poll_interval_sec=0.5)
    try:
        info = client.object_info()
        options = (info.get("UpscaleModelLoader", {}).get("input", {})
                   .get("required", {}).get("model_name", []))
        names = options[1].get("options", []) if len(options) > 1 else []
        if ESRGAN_MODEL not in names:
            raise PipelineError(f"ComfyUI cannot see {ESRGAN_MODEL} in upscale_models.")
        name = client.upload_image(source, subfolder="midnight_upscale")
        workflow = {
            "1": {"class_type": "LoadImage", "inputs": {"image": f"midnight_upscale/{name}"}},
            "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": ESRGAN_MODEL}},
            "3": {"class_type": "ImageUpscaleWithModel", "inputs": {
                "upscale_model": ["2", 0], "image": ["1", 0]}},
            "4": {"class_type": "SaveImage", "inputs": {
                "images": ["3", 0], "filename_prefix": "midnight_esrgan"}},
        }
        prompt = client.submit(workflow)
        deadline = time.monotonic() + timeout
        while True:
            if cancel.is_set():
                client.cancel_tracked(prompt)
                raise PipelineError("Cancelled Real-ESRGAN prompt.")
            if time.monotonic() >= deadline:
                client.cancel_tracked(prompt)
                raise PipelineError(f"Real-ESRGAN frame exceeded {timeout}s timeout.")
            payload = client._json(client._request("GET", f"/history/{prompt}"))
            entry = payload.get(prompt) if isinstance(payload, dict) else None
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise PipelineError(f"Real-ESRGAN ComfyUI error: {status.get('messages')}")
                if status.get("completed"):
                    images = images_from_history(entry, "4")
                    if len(images) != 1:
                        raise PipelineError("Real-ESRGAN returned an unexpected number of frames.")
                    client.download(images[0], target)
                    with Image.open(target) as loaded:
                        return loaded.convert("RGB").copy()
            time.sleep(0.4)
    finally:
        client.close()


def _esrgan_batch(frames: list[np.ndarray], *, scale: int, workdir: Path,
                  comfy_url: str, timeout: int, cancel: threading.Event) -> list[np.ndarray]:
    """Submit a bounded IMAGE batch in one ComfyUI prompt/model load."""
    if len(frames) == 1:
        return [upscale_frame(frames[0], mode=MODES[1], scale=scale,
                              fast_model=FAST_MODELS[0], workdir=workdir,
                              comfy_url=comfy_url, timeout=timeout, cancel=cancel)]
    workdir.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(comfy_url, timeout_sec=timeout, poll_interval_sec=0.5)
    try:
        model_info = client.object_info().get("UpscaleModelLoader", {})
        spec = model_info.get("input", {}).get("required", {}).get("model_name", [])
        names = spec[1].get("options", []) if len(spec) > 1 else []
        if ESRGAN_MODEL not in names:
            raise PipelineError(f"ComfyUI cannot see {ESRGAN_MODEL}.")
        graph: dict[str, dict] = {
            "100": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": ESRGAN_MODEL}}
        }
        previous = None
        for index, frame in enumerate(frames):
            path = workdir / f"batch_{index:02d}.png"
            _rgb_input(frame).save(path)
            name = client.upload_image(path, subfolder="midnight_upscale")
            node = str(index + 1)
            graph[node] = {"class_type": "LoadImage", "inputs": {
                "image": f"midnight_upscale/{name}"}}
            if previous is None:
                previous = node
            else:
                merged = str(50 + index)
                graph[merged] = {"class_type": "ImageBatch", "inputs": {
                    "image1": [previous, 0], "image2": [node, 0]}}
                previous = merged
        graph["101"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
            "upscale_model": ["100", 0], "image": [previous, 0]}}
        graph["102"] = {"class_type": "SaveImage", "inputs": {
            "images": ["101", 0], "filename_prefix": "midnight_esrgan_batch"}}
        prompt = client.submit(graph)
        deadline = time.monotonic() + timeout
        while True:
            if cancel.is_set() or time.monotonic() >= deadline:
                client.cancel_tracked(prompt)
                raise PipelineError("Real-ESRGAN batch cancelled or timed out.")
            payload = client._json(client._request("GET", f"/history/{prompt}"))
            entry = payload.get(prompt) if isinstance(payload, dict) else None
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise PipelineError(f"Real-ESRGAN batch failed: {status.get('messages')}")
                if status.get("completed"):
                    images = images_from_history(entry, "102")
                    if len(images) != len(frames):
                        raise PipelineError("Real-ESRGAN batch returned wrong frame count.")
                    results = []
                    for index, (image_info, frame) in enumerate(zip(images, frames, strict=True)):
                        target = workdir / f"result_{index:02d}.png"
                        client.download(image_info, target)
                        with Image.open(target) as loaded:
                            results.append(_combine(loaded.convert("RGB"), frame, scale))
                    return results
            time.sleep(0.4)
    finally:
        client.close()


def upscale_frame(frame: np.ndarray, *, mode: str, scale: int, fast_model: str,
                  workdir: Path, comfy_url: str, timeout: int,
                  cancel: threading.Event) -> np.ndarray:
    if scale not in (2, 4):
        raise PipelineError("Upscale supports 2× or 4×.")
    workdir.mkdir(parents=True, exist_ok=True)
    rgb = _rgb_input(frame)
    if mode == MODES[0]:
        result = _anime4k(rgb, scale, fast_model, workdir, cancel)
    elif mode == MODES[1]:
        result = _esrgan(rgb, scale, workdir, comfy_url, timeout, cancel)
    else:
        raise PipelineError(f"Unknown single-frame upscale mode: {mode}")
    return _combine(result, frame, scale)


def _quality(frames: list[np.ndarray], *, scale: int, workdir: Path,
             comfy_url: str, timeout: int, cancel: threading.Event,
             on_frame: Callable[[int, Path], None] | None = None) -> list[Path]:
    """Run the installed temporal model on bounded 9-frame overlapping windows."""
    if cancel.is_set():
        raise PipelineError("Cancelled.")
    workdir.mkdir(parents=True, exist_ok=True)
    rgb_dir = workdir / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    for index, frame in enumerate(frames):
        _rgb_input(frame).save(frame_path(rgb_dir, index))
    height, width = frames[0].shape[:2]
    target_pixels = width * height * scale * scale
    window_frames = 9 if target_pixels <= 512 * 512 else 5 if target_pixels <= 2_000_000 else 1
    metadata = JobMetadata(
        source="quality_clip", original_width=width, original_height=height,
        frame_count=len(frames), frame_durations_ms=[1000] * len(frames),
        total_duration_ms=len(frames) * 1000, gif_loop_count=None,
        durations_constant=True, estimated_fps=None,
        has_transparency=any(np.any(frame[:, :, 3] < 255) for frame in frames),
        has_semitransparency=False, requested_scale=scale,
        target_width=width * scale, target_height=height * scale,
        alpha_mode="lanczos", edge_cleanup="off", interpolate="none",
        workdir=str(workdir),
    )
    metadata.save(workdir / "metadata.json")
    client = ComfyClient(comfy_url, timeout_sec=timeout, poll_interval_sec=0.5)
    client.external_cancel = cancel
    rgba_dir = workdir / "final_rgba"
    rgba_dir.mkdir(exist_ok=True)
    def completed(index: int, path: Path) -> None:
        with Image.open(path) as image:
            rgba = _combine(image.convert("RGB"), frames[index], scale)
        destination = frame_path(rgba_dir, index)
        Image.fromarray(rgba, "RGBA").save(destination)
        if on_frame is not None:
            on_frame(index, destination)
    try:
        run_seedvr2_native(
            workdir, metadata, comfy_url=comfy_url, config_path=None,
            timeout_sec=timeout, overwrite=True, temporal_overlap=0,
            temporal_mode="windowed", client=client, on_frame=completed,
            window_frames=window_frames,
        )
    finally:
        client.close()
    results: list[Path] = []
    for index in range(len(frames)):
        path = frame_path(rgba_dir, index)
        if not path.is_file():
            raise PipelineError(f"SeedVR2 did not produce frame {index + 1}/{len(frames)}.")
        results.append(path)
    return results


def preview_frame(frame: np.ndarray, *, mode: str, scale: int, fast_model: str,
                  workdir: Path, comfy_url: str, timeout: int,
                  cancel: threading.Event) -> np.ndarray:
    if mode == MODES[2]:
        path = _quality([frame], scale=scale, workdir=workdir / "quality",
                        comfy_url=comfy_url, timeout=timeout, cancel=cancel)[0]
        with Image.open(path) as image:
            return np.asarray(image.convert("RGBA")).copy()
    return upscale_frame(frame, mode=mode, scale=scale, fast_model=fast_model,
                         workdir=workdir, comfy_url=comfy_url, timeout=timeout,
                         cancel=cancel)


def _export_mp4(frames: list[np.ndarray], durations: list[int], destination: Path,
                 workdir: Path) -> Path:
    rgba = workdir / "mp4_rgb"
    rgba.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        # MP4/H.264 cannot carry alpha. Composite explicitly against black.
        image = Image.new("RGB", (frame.shape[1], frame.shape[0]), "black")
        layer = Image.fromarray(frame, "RGBA")
        image.paste(layer, mask=layer.getchannel("A"))
        path = frame_path(rgba, index)
        image.save(path)
        paths.append(path)
    concat = write_concat_file(paths, durations, rgba / "concat.txt")
    duration = sum(max(1, d) for d in durations) / 1000
    last_pts = sum(durations[:-1])
    last_duration = max(1, durations[-1])
    command = [require_binary("ffmpeg"), "-y", "-f", "concat", "-safe", "0",
               "-i", str(concat), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
               "-vf", f"select=lt(n\\,{len(frames)}),setpts=PTS-STARTPTS",
               "-fps_mode", "vfr", "-c:v", "libx264", "-bf", "0", "-enc_time_base", "1/1000", "-crf", "18",
               "-pix_fmt", "yuv420p", "-video_track_timescale", "1000",
               "-bsf:v", f"setts=duration=if(eq(PTS\\,{last_pts})\\,{last_duration}\\,DURATION)",
               "-c:a", "aac", "-b:a", "16k", "-t", f"{duration:.3f}", str(destination)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if done.returncode:
        raise PipelineError(f"MP4 export failed: {done.stderr[-1000:]}")
    return destination


def upscale_clip(source: Path, *, mode: str, scale: int, fast_model: str,
                 fmt: str, output_dir: Path, workdir: Path, comfy_url: str,
                 timeout: int, fallback: bool, cancel: threading.Event,
                 progress=None, status=None,
                 cleanup: OutlineCleanupSettings = OutlineCleanupSettings()) -> list[Path]:
    clip = read_clip(source, workdir)
    if not clip.frames:
        raise PipelineError("Input has no frames.")
    if scale not in (2, 4):
        raise PipelineError("Only 2× and 4× are supported.")
    if cleanup.enabled:
        clip.frames = [clean_outline(frame, cleanup) for frame in clip.frames]
    output_paths: list[Path] = []
    start = time.monotonic()
    total = len(clip.frames)
    if mode == MODES[2]:
        try:
            if status:
                status("SeedVR2: processing VRAM-aware temporal windows")
            def temporal_progress(index: int, preview: Path) -> None:
                elapsed = time.monotonic() - start
                if progress:
                    progress(index + 1, total, elapsed,
                             (elapsed / (index + 1)) * (total - index - 1), preview)
            output_paths = _quality(clip.frames, scale=scale, workdir=workdir / "quality",
                                    comfy_url=comfy_url, timeout=timeout, cancel=cancel,
                                    on_frame=temporal_progress)
        except Exception as exc:
            if not fallback or cancel.is_set():
                raise
            log.exception("SeedVR2 failed; falling back to Real-ESRGAN for this clip")
            if status:
                status(f"SeedVR2 failed ({exc}); restarting clip with Real-ESRGAN")
            output_paths.clear()
    if len(output_paths) != total:
        index = 0
        selected = MODES[1] if mode == MODES[2] else mode
        while index < total:
            if cancel.is_set():
                raise PipelineError("Upscale cancelled.")
            frame = clip.frames[index]
            # Four small frames share one GPU prompt. Large frames run singly to
            # bound activation memory on the 16 GB card.
            pixels = frame.shape[0] * frame.shape[1] * scale * scale
            batch_size = min(4, max(1, 4_000_000 // max(1, pixels))) if selected == MODES[1] else 1
            batch = clip.frames[index:index + batch_size]
            if selected == MODES[1]:
                try:
                    results = _esrgan_batch(batch, scale=scale,
                                            workdir=workdir / "scratch",
                                            comfy_url=comfy_url, timeout=timeout, cancel=cancel)
                except Exception as exc:
                    if len(batch) == 1 or cancel.is_set():
                        raise
                    log.warning("Real-ESRGAN batch failed (%s); retrying one frame at a time", exc)
                    if status:
                        status(f"Real-ESRGAN batch failed ({exc}); retrying single frames")
                    results = []
                    for single in batch:
                        results.append(upscale_frame(
                            single, mode=MODES[1], scale=scale,
                            fast_model=fast_model, workdir=workdir / "scratch",
                            comfy_url=comfy_url, timeout=timeout, cancel=cancel,
                        ))
            else:
                results = [upscale_frame(frame, mode=selected, scale=scale,
                                         fast_model=fast_model, workdir=workdir / "scratch",
                                         comfy_url=comfy_url, timeout=timeout, cancel=cancel)]
            for result in results:
                preview = frame_path(workdir / "preview", index)
                preview.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(result, "RGBA").save(preview)
                output_paths.append(preview)
                elapsed = time.monotonic() - start
                if progress:
                    progress(index + 1, total, elapsed,
                             (elapsed / (index + 1)) * (total - index - 1), preview)
                index += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = DiskFrames(output_paths)
    if fmt == "mp4":
        destination = output_dir / f"{source.stem}_upscaled.mp4"
        number = 1
        while destination.exists():
            destination = output_dir / f"{source.stem}_upscaled_v{number}.mp4"
            number += 1
        return [_export_mp4(outputs, clip.durations_ms, destination, workdir)]
    stem = source.stem + "_upscaled"
    version = 1
    while (output_dir / f"{stem}.webm").exists() or (output_dir / f"{stem}.zip").exists() or (output_dir / stem).exists():
        stem = f"{source.stem}_upscaled_v{version}"
        version += 1
    if fmt == "webm":
        return [export_video(output_paths, clip.durations_ms, output_dir / f"{stem}.webm", "webm", 18)]
    if fmt == "png":
        folder = output_dir / stem
        files = export_png_sequence(output_paths, folder, stem)
        metadata = folder / "animation.json"
        metadata.write_text(json.dumps({"frame_count": len(files),
                                        "frame_durations_ms": clip.durations_ms,
                                        "duration_ms": sum(clip.durations_ms),
                                        "fps": clip.fps,
                                        "frames": [path.name for path in files]}, indent=2), encoding="utf-8")
        return [*files, metadata]
    if fmt == "spritesheet":
        files, _ = export_spritesheet(output_paths, clip.durations_ms, output_dir,
                                      stem, SpritesheetSettings(columns=None, padding=0,
                                                               max_size=8192, power_of_two=False))
        return [zip_spritesheet(files, output_dir / f"{stem}.zip")]
    raise PipelineError(f"Unsupported upscale export format: {fmt}")
