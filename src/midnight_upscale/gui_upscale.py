"""Three-mode upscale UI with a trial-frame comparison before queueing jobs."""

from __future__ import annotations

import queue
import tempfile
import threading
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageFilter
from midnight_upscale.outline_cleanup import OutlineCleanupSettings, clean_outline

from midnight_upscale.upscale_modes import (
    FAST_MODELS,
    MODES,
    preview_frame,
    read_source_frame,
    source_summary,
    upscale_clip,
)
from midnight_upscale.utils import PipelineError

_cancel = threading.Event()
_job_lock = threading.Lock()


def _paths(files: object) -> list[Path]:
    if not files:
        return []
    if not isinstance(files, list):
        files = [files]
    return [Path(getattr(item, "name", item)) for item in files]


def _shown(frame: np.ndarray, background: str) -> Image.Image:
    rgba = Image.fromarray(frame, "RGBA")
    if background == "Transparent":
        return rgba
    color = {"Dark": "#202124", "White": "white", "Green": "#188b56"}[background]
    canvas = Image.new("RGBA", rgba.size, color)
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def _cleanup_settings(enabled, strength, width, smoothing, speckles, normalize, protect):
    return OutlineCleanupSettings(bool(enabled), strength, float(width), float(smoothing),
                                  bool(speckles), bool(normalize), bool(protect))


def _cleanup_preview(files, index, background, enabled, strength, width, smoothing,
                     speckles, normalize, protect):
    paths = _paths(files)
    if not paths:
        raise gr.Error("Choose an input first.")
    settings = _cleanup_settings(enabled, strength, width, smoothing, speckles, normalize, protect)
    with tempfile.TemporaryDirectory() as temp:
        count, _, _, _ = source_summary(paths[0])
        position = min(max(int(index), 0), count - 1)
        original = read_source_frame(paths[0], position, Path(temp))
    cleaned = clean_outline(original, settings)
    # Choose an edge patch near the center instead of sending a huge 4× image
    # for every full-resolution animation frame.
    mask = Image.fromarray(cleaned[:, :, 3], "L")
    edge = np.asarray(mask.filter(ImageFilter.MaxFilter(3))) > np.asarray(
        mask.filter(ImageFilter.MinFilter(3)))
    points = np.argwhere(edge)
    if len(points):
        center = np.asarray(cleaned.shape[:2]) / 2
        y, x = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
    else:
        y, x = cleaned.shape[0] // 2, cleaned.shape[1] // 2
    radius = 64
    box = (max(0, int(x) - radius), max(0, int(y) - radius),
           min(cleaned.shape[1], int(x) + radius),
           min(cleaned.shape[0], int(y) + radius))
    zoom_source = _shown(cleaned, background).crop(box)
    zoom = zoom_source.resize((zoom_source.width * 4, zoom_source.height * 4),
                              Image.Resampling.NEAREST)
    alpha = Image.fromarray(cleaned[:, :, 3], "L")
    return (_shown(original, background), _shown(cleaned, background), zoom, alpha,
            f"Outline cleanup preview · frame {position + 1}/{count}")


def _source(files: object, index: int, background: str):
    paths = _paths(files)
    if not paths:
        return None, "No input selected.", gr.update(maximum=1, value=0)
    try:
        with tempfile.TemporaryDirectory() as temp:
            count, width, height, duration = source_summary(paths[0])
            position = min(max(int(index), 0), count - 1)
            frame = read_source_frame(paths[0], position, Path(temp))
        text = (f"{paths[0].name} · {count} frames · "
                f"{width}×{height} · {duration:.3f}s")
        return _shown(frame, background), text, gr.update(
            maximum=max(1, count - 1), value=position, interactive=True
        )
    except Exception as exc:
        return None, str(exc), gr.update(maximum=1, value=0)


def _trial(files: object, index: int, mode: str, scale: int, model: str,
           background: str, comfy_url: str, timeout: int,
           enabled, strength, width, smoothing, speckles, normalize, protect):
    paths = _paths(files)
    if not paths:
        raise gr.Error("Choose an input first.")
    _cancel.clear()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        count, _, _, _ = source_summary(paths[0])
        original_frame = read_source_frame(paths[0], min(max(int(index), 0), count - 1), root)
        frame = clean_outline(original_frame, _cleanup_settings(
            enabled, strength, width, smoothing, speckles, normalize, protect))
        upscaled = preview_frame(frame, mode=mode, scale=int(scale), fast_model=model,
                                 workdir=root / "trial", comfy_url=comfy_url,
                                 timeout=int(timeout), cancel=_cancel)
        return (_shown(original_frame, background), _shown(upscaled, background),
                f"Trial frame {int(index) + 1}/{count} · "
                f"{upscaled.shape[1]}×{upscaled.shape[0]} · {mode}")


def _run(files: object, mode: str, scale: int, model: str, fmt: str,
         background: str, comfy_url: str, timeout: int, fallback: bool,
         output_dir: str, enabled, strength, width, smoothing, speckles,
         normalize, protect):
    paths = _paths(files)
    if not paths:
        yield "Choose an input first.", None, None
        return
    if not _job_lock.acquire(blocking=False):
        yield "An upscale job is already running.", None, None
        return
    _cancel.clear()
    messages: queue.Queue[tuple] = queue.Queue()
    finished: list[Path] = []

    def worker():
        try:
            for job_index, path in enumerate(paths):
                if _cancel.is_set():
                    raise PipelineError("Cancelled.")
                messages.put(("status", f"Queued {job_index + 1}/{len(paths)}: {path.name}"))
                workdir = Path("work") / "upscale_modes" / f"{path.stem}_{job_index}"
                workdir.mkdir(parents=True, exist_ok=True)
                def progress(done, total, elapsed, eta, preview):
                    percent = 100 * done / total
                    eta_text = "—" if eta is None else f"{eta:.1f}s"
                    messages.put(("progress", f"{path.name}: frame {done}/{total} "
                                  f"({percent:.1f}%) · elapsed {elapsed:.1f}s · "
                                  f"ETA {eta_text}", preview))
                outputs = upscale_clip(path, mode=mode, scale=int(scale),
                                       fast_model=model, fmt=fmt,
                                       output_dir=Path(output_dir), workdir=workdir,
                                       comfy_url=comfy_url, timeout=int(timeout),
                                       fallback=fallback, cancel=_cancel,
                                       cleanup=_cleanup_settings(enabled, strength, width,
                                                                 smoothing, speckles, normalize,
                                                                 protect),
                                       progress=progress,
                                       status=lambda message: messages.put(("status", message)))
                finished.extend(outputs)
                messages.put(("status", f"Completed {job_index + 1}/{len(paths)}: {path.name}"))
            messages.put(("done", finished))
        except Exception as exc:
            messages.put(("error", str(exc)))
        finally:
            _job_lock.release()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    label = "Queued"
    live = None
    while True:
        try:
            event = messages.get(timeout=0.5)
        except queue.Empty:
            yield label, live, [str(path) for path in finished] or None
            continue
        if event[0] == "progress":
            label, live = event[1], str(event[2])
            yield label, live, [str(path) for path in finished] or None
        elif event[0] == "status":
            label = event[1]
            yield label, live, [str(path) for path in finished] or None
        elif event[0] == "done":
            yield "All jobs completed.", live, [str(path) for path in finished]
            return
        else:
            yield f"Error: {event[1]}", live, [str(path) for path in finished] or None
            return


def _stop():
    _cancel.set()
    return "Cancellation requested. The active GPU operation will stop."


def mount_upscale_tab() -> None:
    gr.Markdown("## Upscale\nFAST uses Anime4K on CUDA; BALANCED uses Real-ESRGAN; "
                "QUALITY uses bounded SeedVR2 windows. Preview a frame before exporting.")
    with gr.Row():
        with gr.Column():
            files = gr.File(label="Input GIF / MP4 / WebM / PNG / spritesheet ZIP",
                            file_count="multiple", type="filepath",
                            file_types=[".gif", ".mp4", ".webm", ".png", ".zip"])
            info = gr.Markdown("No input selected.")
            frame = gr.Slider(label="Preview frame", minimum=0, maximum=1,
                              step=1, value=0, interactive=False)
            mode = gr.Radio(label="Mode", choices=list(MODES), value=MODES[1])
            model = gr.Dropdown(label="Anime4K model (FAST)", choices=list(FAST_MODELS),
                                value=FAST_MODELS[0])
            scale = gr.Radio(label="Scale", choices=[2, 4], value=2)
            background = gr.Radio(label="Preview background",
                                  choices=["Transparent", "Dark", "White", "Green"],
                                  value="Dark")
            with gr.Accordion("Outline Cleanup (transparent cartoon assets)", open=False):
                cleanup_enabled = gr.Checkbox(label="Enable outline-preserving cleanup", value=False)
                cleanup_strength = gr.Radio(label="Edge cleanup strength", choices=["Low", "Medium", "High"], value="Medium")
                cleanup_width = gr.Radio(label="Expected outline thickness (px)", choices=[1.0, 1.5, 2.0], value=1.0)
                cleanup_smoothing = gr.Slider(label="Alpha smoothing", minimum=0, maximum=1, step=0.05, value=0.2)
                cleanup_speckles = gr.Checkbox(label="Remove edge speckles", value=True)
                cleanup_normalize = gr.Checkbox(label="Normalize outline thickness", value=True)
                cleanup_protect = gr.Checkbox(label="Protect dark intentional outlines", value=True)
                cleanup_preview = gr.Button("Preview outline cleanup on selected frame")
            comfy_url = gr.Textbox(label="ComfyUI URL", value="http://127.0.0.1:8188")
            timeout = gr.Slider(label="GPU timeout per frame/window (seconds)",
                                minimum=30, maximum=600, step=10, value=180)
            fallback = gr.Checkbox(label="QUALITY: fall back to Real-ESRGAN on failure",
                                   value=True)
            fmt = gr.Radio(label="Export", choices=["png", "spritesheet", "mp4", "webm"],
                           value="webm")
            output_dir = gr.Textbox(label="Output directory", value="output")
            trial = gr.Button("Preview this frame", variant="secondary")
            run = gr.Button("Upscale queue", variant="primary")
            cancel = gr.Button("Cancel")
        with gr.Column():
            trial_status = gr.Markdown("Select a frame, mode and scale, then preview it.")
            with gr.Row():
                original = gr.Image(label="Original frame", interactive=False)
                enhanced = gr.Image(label="Upscaled trial frame", interactive=False)
            with gr.Row():
                cleaned = gr.Image(label="Cleaned frame", interactive=False)
                zoomed = gr.Image(label="Cleaned frame (4× pixel zoom)", interactive=False)
                alpha = gr.Image(label="Cleaned alpha", interactive=False)
            status = gr.Markdown("Idle")
            live = gr.Image(label="Live upscaled frame", interactive=False)
            outputs = gr.File(label="Exports", file_count="multiple", interactive=False)
    files.change(_source, [files, frame, background], [original, info, frame])
    frame.change(_source, [files, frame, background], [original, info, frame])
    background.change(_source, [files, frame, background], [original, info, frame])
    cleanup_inputs = [cleanup_enabled, cleanup_strength, cleanup_width, cleanup_smoothing,
                      cleanup_speckles, cleanup_normalize, cleanup_protect]
    cleanup_preview.click(_cleanup_preview, [files, frame, background, *cleanup_inputs],
                          [original, cleaned, zoomed, alpha, trial_status], concurrency_limit=1)
    trial.click(_trial, [files, frame, mode, scale, model, background, comfy_url, timeout,
                         *cleanup_inputs],
                [original, enhanced, trial_status], concurrency_limit=1)
    run.click(_run, [files, mode, scale, model, fmt, background, comfy_url, timeout,
                     fallback, output_dir, *cleanup_inputs], [status, live, outputs],
              concurrency_limit=1)
    cancel.click(_stop, outputs=[status])
