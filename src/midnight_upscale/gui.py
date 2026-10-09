"""Local Gradio page for the Midnight Lounge upscaler.

Launch with ``midnight-upscale-gui`` or ``python -m midnight_upscale.gui``.
The server binds to 127.0.0.1 and does not create a public share link.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

try:
    import gradio as gr
except ImportError:  # pragma: no cover - exercised when the extra is not installed
    gr = None

from midnight_upscale.gui_logic import (
    BACKEND_CHOICES,
    JobEvent,
    check_comfy_connection,
    coerce_upload_paths,
    comparison_frame,
    format_gui_error,
    format_queue,
    format_source_card,
    format_stages,
    gui_config_from_form,
    iter_queue,
    request_cancel,
    set_preview_background,
)
from midnight_upscale.inspect import inspect_gif
from midnight_upscale.progress import stamp_log
from midnight_upscale.utils import PipelineError

logger = logging.getLogger(__name__)

CSS = """
.gradio-container {
  background: #14161c !important;
  color: #eceae6 !important;
  max-width: 1180px !important;
}
.gradio-container h1 {
  letter-spacing: 0.01em;
  margin-bottom: 0.15rem;
}
.upscale-button button {
  min-height: 3.25rem;
  font-weight: 700;
  letter-spacing: 0.12em;
}
footer { display: none !important; }
"""


def _theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.zinc,
    ).set(
        body_background_fill="#14161c",
        body_background_fill_dark="#14161c",
        block_background_fill="#1c1f27",
        block_background_fill_dark="#1c1f27",
        body_text_color="#eceae6",
        body_text_color_dark="#eceae6",
        block_label_text_color="#d7d3cc",
        block_label_text_color_dark="#d7d3cc",
        input_background_fill="#12141a",
        input_background_fill_dark="#12141a",
        button_primary_background_fill="#d9782d",
        button_primary_background_fill_dark="#d9782d",
        button_primary_text_color="#1a120c",
        button_primary_text_color_dark="#1a120c",
        border_color_accent="#d9782d",
        border_color_accent_dark="#d9782d",
    )


def build_upscale_demo() -> gr.Blocks:
    if gr is None:
        raise PipelineError('Gradio is not installed. Install it with: pip install -e ".[gui]"')
    with gr.Blocks(title="Midnight Upscale") as demo:
        gr.Markdown("# Midnight Upscale")
        gr.Markdown("Transparent GIF / animation upscaler using SeedVR2 + ComfyUI")
        gr.Markdown("Local only. This page is bound to 127.0.0.1 and is not shared.")

        with gr.Row():
            comfy_url = gr.Textbox(
                label="ComfyUI URL",
                value="http://127.0.0.1:8188",
                scale=4,
            )
            check_button = gr.Button("Check connection", scale=1)
        comfy_status = gr.Markdown(_panel("ComfyUI status\n------\nOFFLINE\nNot checked yet."))
        overlap_note = gr.Markdown("")

        with gr.Row():
            with gr.Column():
                uploads = gr.File(
                    label="Input GIF",
                    file_count="multiple",
                    file_types=[".gif"],
                    type="filepath",
                )
                source_info = gr.Markdown(_panel("Source\n------\nNo file selected."))
                backend = gr.Dropdown(
                    label="Backend",
                    choices=list(_BACKENDS),
                    value="Native ComfyUI",
                )
                scale = gr.Dropdown(label="Scale", choices=["2x", "3x", "4x"], value="2x")
                batch_size = gr.Dropdown(
                    label="Batch size",
                    choices=[5, 9, 13],
                    value=5,
                    info="Numz custom node only. Must be 4n+1, starting at 5.",
                )
                temporal_mode = gr.Dropdown(
                    label="Temporal processing",
                    choices=["Auto", "Unchunked", "Chunked"],
                    value="Auto",
                    info="Native SeedVR2. Auto keeps short clips unchunked.",
                )
                temporal_overlap = gr.Dropdown(
                    label="Temporal overlap",
                    choices=[0, 1, 2, 3],
                    value=1,
                    info="Used when the installed node exposes temporal_overlap.",
                )
                alpha_mode = gr.Dropdown(
                    label="Alpha resize",
                    choices=["Lanczos", "Bicubic", "Nearest"],
                    value="Lanczos",
                )
                edge_cleanup = gr.Dropdown(
                    label="Edge cleanup",
                    choices=["Auto", "Off", "Simple"],
                    value="Auto",
                )
                output_format = gr.Dropdown(
                    label="Output format",
                    choices=["WebM Alpha", "APNG", "GIF Preview"],
                    value="WebM Alpha",
                )
                keep_workdir = gr.Checkbox(
                    label="Keep work directory",
                    value=False,
                    info="Off deletes work/<name> after a successful encode.",
                )
                with gr.Accordion("Advanced", open=False):
                    workflow_path = gr.Textbox(
                        label="SeedVR2 workflow JSON path",
                        value="",
                        info=(
                            "Only the Numz custom node uses a workflow file. "
                            "Native ComfyUI builds its own graph."
                        ),
                    )
                    output_dir = gr.Textbox(label="Output directory", value="output")
                    overwrite = gr.Checkbox(
                        label="Overwrite existing output",
                        value=False,
                        info="Off keeps the previous run and writes the next _v1, _v2, _v3 folder.",
                    )
                    verbose = gr.Checkbox(
                        label="Verbose",
                        value=False,
                        info="Tracebacks go to the terminal, not this page.",
                    )
                    interpolate = gr.Dropdown(
                        label="Interpolation",
                        choices=["None"],
                        value="None",
                        info="RIFE is not implemented. Frames and delays stay as they are.",
                    )
                with gr.Row():
                    upscale_button = gr.Button(
                        "UPSCALE",
                        variant="primary",
                        elem_classes=["upscale-button"],
                        scale=3,
                    )
                    cancel_button = gr.Button("CANCEL JOB", scale=1)

            with gr.Column():
                source_preview = gr.Image(
                    label="Source preview",
                    type="filepath",
                    interactive=False,
                )
                progress = gr.Markdown(_panel("STATUS\n\nIdle."))
                live_caption = gr.Markdown("")
                live_preview = gr.Image(
                    label="Live preview",
                    type="filepath",
                    interactive=False,
                )
                preview_background = gr.Dropdown(
                    label="Preview background",
                    choices=["checkerboard", "black", "white"],
                    value="checkerboard",
                )
                queue_box = gr.Markdown(_panel("Queue\n\nNo files yet."))
                with gr.Accordion("Job log", open=True):
                    job_log = gr.Textbox(label="Log", lines=14, max_lines=18, interactive=False)
                result_box = gr.Markdown(_panel("RESULT\n------\nNo output yet."))
                output_file = gr.File(label="Production output", interactive=False)
                preview_note = gr.Markdown("")
                browser_preview = gr.Video(
                    label="Browser preview — transparency composited for display",
                    interactive=False,
                )
                gr.Markdown(
                    "Before / after. Transparent pixels are drawn on a checkerboard. "
                    "This does not change the production file."
                )
                frame_slider = gr.Slider(
                    label="Frame",
                    minimum=0,
                    maximum=1,
                    step=1,
                    value=0,
                    interactive=False,
                )
                with gr.Row():
                    source_frame = gr.Image(label="SOURCE FRAME", interactive=False)
                    upscaled_frame = gr.Image(label="UPSCALED FRAME", interactive=False)

        compare_state = gr.State({})

        backend.change(
            _on_backend,
            inputs=[backend],
            outputs=[batch_size, workflow_path],
        )
        cancel_button.click(_on_cancel, outputs=[live_caption])
        preview_background.change(
            _on_preview_background,
            inputs=[preview_background],
            outputs=[],
        )
        check_button.click(
            _on_check,
            inputs=[comfy_url],
            outputs=[comfy_status, overlap_note, temporal_overlap, backend],
        )
        uploads.change(
            _on_files,
            inputs=[uploads],
            outputs=[source_info, source_preview, queue_box],
        )
        upscale_button.click(
            _on_upscale,
            inputs=[
                uploads,
                backend,
                scale,
                batch_size,
                temporal_mode,
                temporal_overlap,
                alpha_mode,
                edge_cleanup,
                output_format,
                keep_workdir,
                comfy_url,
                workflow_path,
                output_dir,
                overwrite,
                verbose,
                interpolate,
            ],
            outputs=[
                upscale_button,
                progress,
                live_caption,
                live_preview,
                job_log,
                result_box,
                output_file,
                preview_note,
                browser_preview,
                source_frame,
                upscaled_frame,
                frame_slider,
                queue_box,
                compare_state,
            ],
            concurrency_limit=1,
        )
        frame_slider.change(
            _on_frame,
            inputs=[frame_slider, compare_state],
            outputs=[source_frame, upscaled_frame],
        )
    return demo


def build_demo() -> gr.Blocks:
    """Two tabs: the SeedVR2 upscaler (needs ComfyUI) and the video converter (does not)."""

    if gr is None:
        raise PipelineError('Gradio is not installed. Install it with: pip install -e ".[gui]"')
    from midnight_upscale.gui_converter import build_converter_demo

    return gr.TabbedInterface(
        [build_upscale_demo(), build_converter_demo()],
        ["UPSCALE", "VIDEO CONVERTER"],
        title="Midnight Upscale",
    )


_BACKENDS = tuple(BACKEND_CHOICES)


class _View:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.stage = "Inspecting input"
        self.detail = ""
        self.lines: list[str] = []
        self.result = _panel("RESULT\n------\nRunning.")
        self.output: str | None = None
        self.note = ""
        self.video: str | None = None
        self.source_frame: str | None = None
        self.upscaled_frame: str | None = None
        self.slider: dict[str, object] | gr.Slider = gr.update(interactive=False)
        self.queue = _panel(format_queue([]))
        self.state: dict[str, object] = {}
        self.panel = ""
        self.live_preview: str | None = None
        self.caption = ""

    def apply(self, event: JobEvent) -> None:
        if event.queue is not None:
            self.queue = _panel(format_queue(event.queue))
        if event.stage:
            self.stage = event.stage
        if event.detail:
            self.detail = event.detail
        if event.panel:
            self.panel = event.panel
        if event.preview_path:
            self.live_preview = event.preview_path
        if event.caption:
            self.caption = event.caption
        if event.log_line:
            line = event.log_line if event.log_line.startswith("[") else stamp_log(event.log_line)
            self.lines.append(line)
            del self.lines[:-200]
        if event.error:
            self.result = _panel("Error\n------\n" + event.error)
        if event.result is not None:
            result = event.result
            self.result = _panel(result.result_text)
            self.output = str(result.output_path)
            self.source_frame = str(result.comparison_source) if result.comparison_source else None
            self.upscaled_frame = (
                str(result.comparison_upscaled) if result.comparison_upscaled else None
            )
            self.video = str(result.preview_path) if result.preview_path else None
            if result.preview_path is not None:
                self.note = (
                    "**Browser preview — transparency composited for display.** "
                    "The MP4 is only for this page. The production asset is the "
                    "transparent file in Production output."
                )
            else:
                self.note = (
                    "No separate browser preview. Production output is the file to keep. "
                    "A flattened preview is created only for WebM, when FFmpeg can write it."
                )
            interactive = result.workdir is not None and result.frame_count > 1
            last_frame = max(result.frame_count - 1, 0)
            self.slider = gr.update(
                value=min(result.middle_frame, last_frame),
                maximum=max(last_frame, 1),
                interactive=interactive,
            )
            compare_dir = ""
            if result.comparison_source is not None:
                compare_dir = str(result.comparison_source.parent)
            self.state = {
                "workdir": str(result.workdir) if result.workdir else "",
                "compare_dir": compare_dir,
            }

    def packet(self, *, interactive: bool) -> tuple[object, ...]:
        status = self.panel or format_stages(self.stage, self.detail, backend=self.backend)
        return (
            gr.update(interactive=interactive),
            _panel(status),
            self.caption,
            self.live_preview,
            "\n".join(self.lines),
            self.result,
            self.output,
            self.note,
            self.video,
            self.source_frame,
            self.upscaled_frame,
            self.slider,
            self.queue,
            self.state,
        )


def _on_backend(label: str) -> tuple[dict[str, object], dict[str, object]]:
    numz = label == "Numz custom node"
    return gr.update(visible=numz), gr.update(visible=numz)


def _on_cancel() -> str:
    request_cancel()
    return "Cancel requested. This prompt will stop. It will not be submitted again."


def _on_preview_background(name: str) -> None:
    set_preview_background(name or "checkerboard")


def _on_check(url: str) -> tuple[str, str, dict[str, object], dict[str, object]]:
    target = (url or "").strip() or "http://127.0.0.1:8188"
    try:
        status = check_comfy_connection(target)
    except Exception as exc:
        logger.exception("ComfyUI check failed")
        status_text = "ComfyUI status\n------\nOFFLINE\n" + format_gui_error(exc, comfy_url=target)
        return _panel(status_text), "", gr.update(), gr.update()
    body = "\n".join(["ComfyUI status", "------", status.summary, *status.node_lines])
    note = status.overlap_message
    dropdown = gr.update(interactive=status.overlap_supported is not False)
    if status.selected_label in {"", "none"}:
        backend = gr.update()
    else:
        backend = gr.update(value=status.selected_label)
    return _panel(body), note, dropdown, backend


def _on_files(uploaded: object) -> tuple[str, str | None, str]:
    paths = coerce_upload_paths(uploaded)
    if not paths:
        return _panel("Source\n------\nNo file selected."), None, _panel(format_queue([]))
    queue = _panel(format_queue([(path.name, "WAITING") for path in paths]))
    first = paths[0]
    extra = ""
    if len(paths) > 1:
        extra = f"\n{len(paths)} files selected. They run one at a time."
    try:
        card = format_source_card(inspect_gif(first)) + extra
    except PipelineError as exc:
        card = "Source\n------\n" + str(exc)
    preview = str(first) if first.suffix.lower() == ".gif" and first.is_file() else None
    return _panel(card), preview, queue


def _on_upscale(
    uploaded: object,
    backend: str,
    scale: str,
    batch_size: float | int,
    temporal_mode: str,
    temporal_overlap: float | int,
    alpha_mode: str,
    edge_cleanup: str,
    output_format: str,
    keep_workdir: bool,
    comfy_url: str,
    workflow_path: str,
    output_dir: str,
    overwrite: bool,
    verbose: bool,
    interpolate: str,
) -> object:
    view = _View(BACKEND_CHOICES.get(backend, "frame-upscale"))
    try:
        paths = coerce_upload_paths(uploaded)
        if not paths:
            raise PipelineError("Choose a GIF first.")
        configs = [
            gui_config_from_form(
                path,
                backend_label=backend,
                scale_label=scale,
                batch_size=int(batch_size),
                temporal_overlap=int(temporal_overlap),
                temporal_mode_label=temporal_mode,
                alpha_label=alpha_mode,
                edge_label=edge_cleanup,
                format_label=output_format,
                keep_workdir=bool(keep_workdir),
                comfy_url=comfy_url,
                workflow_path=workflow_path or "",
                output_dir=output_dir or "output",
                overwrite=bool(overwrite),
                verbose=bool(verbose),
                interpolate_label=interpolate or "None",
            )
            for path in paths
        ]
    except (PipelineError, TypeError, ValueError) as exc:
        view.result = _panel("Error\n------\n" + format_gui_error(exc, comfy_url=comfy_url or ""))
        yield view.packet(interactive=True)
        return

    yield view.packet(interactive=False)
    try:
        for event in iter_queue(configs):
            if _keepalive(event):
                yield view.packet(interactive=False)
                continue
            view.apply(event)
            yield view.packet(interactive=False)
    except Exception as exc:
        logger.exception("GUI upscale failed")
        view.result = _panel("Error\n------\n" + format_gui_error(exc, comfy_url=comfy_url or ""))
    yield view.packet(interactive=True)


def _on_frame(index: float, state: object) -> tuple[object, object]:
    if not isinstance(state, dict):
        return gr.update(), gr.update()
    workdir = str(state.get("workdir") or "")
    compare_dir = str(state.get("compare_dir") or "")
    if not workdir or not compare_dir:
        return gr.update(), gr.update()
    try:
        source, upscaled = comparison_frame(Path(workdir), int(index), Path(compare_dir))
    except PipelineError as exc:
        logger.error("Could not render frame %s: %s", index, exc)
        return gr.update(), gr.update()
    if source is None or upscaled is None:
        return gr.update(), gr.update()
    return str(source), str(upscaled)


def _keepalive(event: JobEvent) -> bool:
    return not any(
        (
            event.stage,
            event.detail,
            event.log_line,
            event.error,
            event.finished,
            event.result is not None,
            event.queue is not None,
            event.panel,
            event.preview_path,
            event.caption,
        )
    )


def _panel(text: str) -> str:
    body = text.replace("```", "'''")
    return f"```\n{body}\n```"


def _allowed_paths() -> list[str]:
    root = Path.cwd()
    return [str(root), str(root / "output"), str(root / "work")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="midnight-upscale-gui",
        description="Open the local Midnight Upscale page on 127.0.0.1.",
    )
    parser.add_argument("--port", type=int, default=7860, help="Default 7860")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        print("Port must be between 1 and 65535.", file=sys.stderr)
        return 2
    if gr is None:
        print(
            'Gradio is not installed. Install the GUI extra:\n  pip install -e ".[gui]"',
            file=sys.stderr,
        )
        return 1
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=not args.no_browser,
        allowed_paths=_allowed_paths(),
        theme=_theme(),
        css=CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
