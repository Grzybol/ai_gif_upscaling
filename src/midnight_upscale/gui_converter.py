"""The VIDEO CONVERTER / BACKGROUND REMOVAL tab.

Separate from the SeedVR2 upscale page. It runs without ComfyUI.
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import gradio as gr
except ImportError:  # pragma: no cover - exercised when the extra is not installed
    gr = None

from midnight_upscale import converter_logic as logic
from midnight_upscale.background import MODE_LABELS
from midnight_upscale.chroma import EDGE_CLEANUP_LEVELS
from midnight_upscale.gui_logic import coerce_upload_paths
from midnight_upscale.mask_temporal import TEMPORAL_LABELS
from midnight_upscale.progress import stamp_log
from midnight_upscale.segmentation import DEFAULT_BACKEND, DEFAULT_MODEL
from midnight_upscale.utils import PipelineError
from midnight_upscale.video_convert import ConvertSettings, estimate_output
from midnight_upscale.video_export import FORMAT_CHOICES, FORMAT_HELP, GIF_WARNING
from midnight_upscale.video_inspect import VIDEO_EXTENSIONS, format_video_card

logger = logging.getLogger(__name__)

# Order of the settings inputs. ``_settings`` zips these with the values Gradio passes.
SETTING_KEYS = (
    "start", "end", "fps", "background_mode", "key_color", "tolerance", "softness", "spill",
    "edge_cleanup", "hard_mask", "sample_key", "ai_backend", "ai_model", "temporal", "crop",
    "padding", "center", "resize", "resize_width", "resize_height", "keep_aspect", "formats",
    "sheet_columns", "sheet_padding", "max_texture", "power_of_two", "output_dir", "overwrite",
    "keep_workdir",
)  # fmt: skip


def _panel(text: str) -> str:
    body = text.replace("```", "'''")
    return f"```\n{body}\n```"


def _settings(values: tuple[object, ...]) -> ConvertSettings:
    return logic.settings_from_form(**dict(zip(SETTING_KEYS, values, strict=True)))  # type: ignore[arg-type]


def build_converter_tab() -> None:
    """Add the converter components to the current Blocks / Tab context."""

    if gr is None:
        raise PipelineError('Gradio is not installed. Install it with: pip install -e ".[gui]"')

    gr.Markdown("## Video Converter / Background Removal")
    gr.Markdown(
        "Turn an MP4, MOV, WebM or GIF into transparent runtime assets: WebM alpha, "
        "APNG, GIF preview, PNG frames or a spritesheet. Works without ComfyUI."
    )
    with gr.Row():
        with gr.Column():
            uploads = gr.File(
                label="Input video",
                file_count="multiple",
                file_types=list(VIDEO_EXTENSIONS),
                type="filepath",
            )
            source_card = gr.Markdown(_panel("Source\n------\nNo file selected."))

            with gr.Accordion("Trim / frame rate", open=True):
                with gr.Row():
                    start = gr.Number(label="Start time (s)", value=0, minimum=0)
                    end = gr.Number(
                        label="End time (s)", value=0, minimum=0, info="0 = until the end"
                    )
                fps = gr.Dropdown(
                    label="Output FPS",
                    choices=logic.FPS_CHOICES,
                    value="Source",
                    allow_custom_value=True,
                    info=(
                        "Source keeps every frame and its own timing. Type any number for a "
                        "custom rate. Changing the rate drops or repeats frames, and the "
                        "estimate below says how many."
                    ),
                )
                estimate = gr.Markdown(_panel("Estimated output\n----------------\nNo file yet."))

            with gr.Accordion("Background removal", open=True):
                background_mode = gr.Dropdown(
                    label="Background removal", choices=list(MODE_LABELS), value="Auto"
                )
                mode_note = gr.Markdown("")
                with gr.Group(visible=True) as chroma_group:
                    key_color = gr.ColorPicker(label="Key color", value="#00ff00")
                    sample_key = gr.Checkbox(
                        label="Sample key color from the frame border",
                        value=False,
                        info="Auto mode always samples the border.",
                    )
                    tolerance = gr.Slider(
                        label="Tolerance", minimum=0, maximum=100, step=1, value=30,
                        info="How far from the key color still counts as background.",
                    )  # fmt: skip
                    softness = gr.Slider(
                        label="Softness / feather", minimum=0, maximum=100, step=1, value=20,
                        info="Width of the semi-transparent edge.",
                    )  # fmt: skip
                    spill = gr.Slider(
                        label="Spill suppression", minimum=0, maximum=100, step=1, value=60,
                        info="Removes green contamination from hair, skin and edges.",
                    )  # fmt: skip
                    edge_cleanup = gr.Dropdown(
                        label="Edge cleanup", choices=list(EDGE_CLEANUP_LEVELS), value="Light"
                    )
                    hard_mask = gr.Checkbox(
                        label="Hard (binary) mask",
                        value=False,
                        info="Off keeps soft antialiased edges. On gives 0/1 alpha only.",
                    )
                with gr.Group(visible=True) as ai_group:
                    ai_status = gr.Markdown(_panel(logic.ai_status_text(DEFAULT_BACKEND)))
                    ai_backend = gr.Dropdown(
                        label="AI backend",
                        choices=logic.ai_backend_choices(),
                        value=DEFAULT_BACKEND,
                    )
                    ai_model = gr.Dropdown(
                        label="AI model",
                        choices=logic.ai_model_choices(),
                        value=DEFAULT_MODEL,
                        allow_custom_value=True,
                        info="Any model name the backend accepts.",
                    )
                temporal = gr.Radio(
                    label="Temporal smoothing",
                    choices=list(TEMPORAL_LABELS),
                    value="Low",
                    info=(
                        "Smooths flicker in AI masks across neighboring frames. Geometry is "
                        "untouched and no frames are interpolated. Chroma Key skips it."
                    ),
                )

            with gr.Accordion("Crop / padding / resize", open=False):
                crop = gr.Checkbox(
                    label="Crop transparent borders",
                    value=False,
                    info="One box for ALL frames, so the character never jumps.",
                )
                padding = gr.Dropdown(
                    label="Padding (px)",
                    choices=logic.PADDING_CHOICES,
                    value="4",
                    allow_custom_value=True,
                )
                center = gr.Checkbox(label="Center content in fixed canvas", value=True)
                resize = gr.Dropdown(
                    label="Resize", choices=logic.RESIZE_LABELS, value="Keep source"
                )
                with gr.Row():
                    resize_width = gr.Number(
                        label="Width (px)", value=0, precision=0, visible=False,
                        info="0 = work it out from the height",
                    )  # fmt: skip
                    resize_height = gr.Number(
                        label="Height (px)", value=0, precision=0, visible=False,
                        info="0 = work it out from the width",
                    )  # fmt: skip
                keep_aspect = gr.Checkbox(label="Keep aspect ratio", value=True, visible=False)

            with gr.Accordion("Export", open=True):
                formats = gr.CheckboxGroup(
                    label="Output formats",
                    choices=list(FORMAT_CHOICES),
                    value=logic.DEFAULT_FORMATS,
                )
                gr.Markdown(FORMAT_HELP)
                gif_warning = gr.Markdown(f"**GIF:** {GIF_WARNING}", visible=False)
                with gr.Accordion("Spritesheet", open=False):
                    sheet_columns = gr.Dropdown(
                        label="Columns",
                        choices=["Auto"],
                        value="Auto",
                        allow_custom_value=True,
                        info="Rows are calculated. Type a number for a fixed column count.",
                    )
                    sheet_padding = gr.Dropdown(
                        label="Frame padding (px)",
                        choices=logic.SHEET_PADDING_CHOICES,
                        value="2",
                        allow_custom_value=True,
                    )
                    max_texture = gr.Dropdown(
                        label="Max texture size (px)",
                        choices=logic.MAX_TEXTURE_CHOICES,
                        value="4096",
                        allow_custom_value=True,
                        info="More sheets (_00, _01, ...) are written if frames do not fit.",
                    )
                    power_of_two = gr.Checkbox(
                        label="Power-of-two canvas",
                        value=False,
                        info="Pads the sheet only. Frames are never rescaled.",
                    )
                with gr.Accordion("Advanced", open=False):
                    output_dir = gr.Textbox(label="Output directory", value="output")
                    overwrite = gr.Checkbox(
                        label="Overwrite existing output",
                        value=False,
                        info="Off writes name_v1, name_v2, ... instead.",
                    )
                    keep_workdir = gr.Checkbox(
                        label="Keep work directory", value=False,
                        info="Keeps the lossless PNG frames of every stage.",
                    )  # fmt: skip

            with gr.Row():
                convert_button = gr.Button(
                    "CONVERT", variant="primary", elem_classes=["upscale-button"], scale=3
                )
                cancel_button = gr.Button("CANCEL JOB", scale=1)

        with gr.Column():
            source_video = gr.Video(label="Source preview", interactive=False)
            gr.Markdown("### Inspect a frame")
            frame_caption = gr.Markdown("")
            frame_slider = gr.Slider(
                label="Frame", minimum=0, maximum=1, step=1, value=0, interactive=False
            )
            with gr.Row():
                preview_background = gr.Dropdown(
                    label="Preview background", choices=logic.PREVIEW_BACKGROUNDS,
                    value="checkerboard", scale=2,
                )  # fmt: skip
                preview_button = gr.Button("Preview frame", scale=1)
            preview_note = gr.Markdown("")
            with gr.Row():
                original_image = gr.Image(label="ORIGINAL", interactive=False)
                mask_image = gr.Image(label="MASK", interactive=False)
                result_image = gr.Image(label="TRANSPARENT RESULT", interactive=False)

            progress = gr.Markdown(_panel("STATUS\n\nIdle."))
            live_caption = gr.Markdown("")
            live_preview = gr.Image(label="Latest processed frame", interactive=False)
            queue_box = gr.Markdown(_panel(logic.format_queue([])))
            with gr.Accordion("Job log", open=True):
                job_log = gr.Textbox(label="Log", lines=12, max_lines=16, interactive=False)
            result_box = gr.Markdown(_panel("RESULT\n------\nNo output yet."))
            output_files = gr.File(label="Output files", file_count="multiple", interactive=False)
            browser_note = gr.Markdown("")
            browser_video = gr.Video(
                label="Browser preview — transparency shown on a checkerboard", interactive=False
            )

    view_state = gr.State({})
    setting_inputs = [
        start, end, fps, background_mode, key_color, tolerance, softness, spill, edge_cleanup,
        hard_mask, sample_key, ai_backend, ai_model, temporal, crop, padding, center, resize,
        resize_width, resize_height, keep_aspect, formats, sheet_columns, sheet_padding,
        max_texture, power_of_two, output_dir, overwrite, keep_workdir,
    ]  # fmt: skip
    assert len(setting_inputs) == len(SETTING_KEYS)
    image_outputs = [original_image, mask_image, result_image, preview_note]

    # ---------------------------------------------------------------- handlers

    def on_mode(mode: str, backend: str) -> tuple[object, object, str]:
        chroma_on = mode in {"Auto", "Chroma Key"}
        ai_on = mode in {"Auto", "AI Segmentation"}
        note = {
            "Auto": "Auto: keeps existing alpha, else Chroma Key for a green border, else AI.",
            "None": "No background removal. Frames are used as decoded.",
        }.get(mode, "")
        return gr.update(visible=chroma_on), gr.update(visible=ai_on), note

    def on_backend(backend: str) -> str:
        return _panel(logic.ai_status_text(backend))

    def on_resize(choice: str) -> tuple[object, object, object]:
        custom = choice == "Custom"
        return (
            gr.update(visible=custom),
            gr.update(visible=custom),
            gr.update(visible=custom),
        )

    def on_formats(selected: list[str]) -> object:
        return gr.update(visible="GIF Preview" in (selected or []))

    def on_files(uploaded: object, start_v: object, end_v: object, fps_v: object):
        paths = coerce_upload_paths(uploaded)
        empty = (
            _panel("Source\n------\nNo file selected."), None,
            _panel("Estimated output\n----------------\nNo file yet."),
            gr.update(interactive=False, value=0, maximum=1), "", None, None, None, "",
            _panel(logic.format_queue([])), {},
        )  # fmt: skip
        if not paths:
            return empty
        first = paths[0]
        queue_text = _panel(logic.format_queue([(p.name, "WAITING") for p in paths]))
        try:
            info = logic.cached_info(first)
            plan = _plan(info, start_v, end_v, fps_v)
        except PipelineError as exc:
            return (
                _panel("Source\n------\n" + str(exc)), None,
                _panel("Estimated output\n----------------\nUnavailable."),
                gr.update(interactive=False, value=0, maximum=1), "", None, None, None, "",
                queue_text, {},
            )  # fmt: skip
        extra = (
            f"\n\n{len(paths)} files selected. They run one at a time." if len(paths) > 1 else ""
        )
        middle = plan.count // 2
        slider = gr.update(
            interactive=plan.count > 1, minimum=0, maximum=max(plan.count - 1, 1), value=middle
        )
        shown = (None, None, None, "")
        try:
            settings = ConvertSettings(fps=plan.fps)
            images, note = logic.preview_frame(first, settings, middle, run_ai=False)
            shown = (*images, note)
        except PipelineError as exc:
            shown = (None, None, None, str(exc))
        video = str(first) if first.suffix.lower() != ".gif" else None
        return (
            _panel(format_video_card(info) + extra),
            video,
            _panel(logic.format_estimate(info, plan)),
            slider,
            f"Frame {middle + 1} / {plan.count}",
            *shown,
            queue_text,
            {"mode": "preview"},
        )

    def _plan(info, start_v, end_v, fps_v):
        end_value = float(end_v or 0)
        return estimate_output(
            info,
            ConvertSettings(
                start_sec=float(start_v or 0),
                end_sec=end_value if end_value > 0 else None,
                fps=logic.parse_fps(fps_v),
            ),
        )

    def on_estimate(uploaded: object, start_v: object, end_v: object, fps_v: object):
        paths = coerce_upload_paths(uploaded)
        if not paths:
            return gr.update(), gr.update(), gr.update()
        try:
            info = logic.cached_info(paths[0])
            plan = _plan(info, start_v, end_v, fps_v)
        except (PipelineError, ValueError) as exc:
            return (
                _panel("Estimated output\n----------------\n" + str(exc)),
                gr.update(),
                gr.update(),
            )
        slider = gr.update(
            interactive=plan.count > 1, minimum=0, maximum=max(plan.count - 1, 1),
            value=plan.count // 2,
        )  # fmt: skip
        return (
            _panel(logic.format_estimate(info, plan)),
            slider,
            f"Frame {plan.count // 2 + 1} / {plan.count}",
        )

    def show_frame(position: float, uploaded: object, state: object, *values: object):
        background = values[-1]
        form = values[:-1]
        logic.set_preview_background(str(background))
        if isinstance(state, dict) and state.get("mode") == "review":
            review = state["review"]
            images, number = logic.review_frame(review, int(position), str(background))
            caption = f"Frame {number + 1} / {review.total_frames}"
            return caption, images[0], images[1], images[2], "Processed frame (saved sample)."
        return _preview(position, uploaded, form, run_ai=False)

    def run_preview(position: float, uploaded: object, state: object, *values: object):
        form = values[:-1]
        logic.set_preview_background(str(values[-1]))
        if isinstance(state, dict) and state.get("mode") == "review":
            return show_frame(position, uploaded, state, *values)
        return _preview(position, uploaded, form, run_ai=True)

    def _preview(position: float, uploaded: object, form: tuple[object, ...], *, run_ai: bool):
        paths = coerce_upload_paths(uploaded)
        if not paths:
            return "", None, None, None, "Choose a video first."
        try:
            settings = _settings(form)
            info = logic.cached_info(paths[0])
            plan = estimate_output(info, settings)
            index = max(0, min(int(position), plan.count - 1))
            images, note = logic.preview_frame(paths[0], settings, index, run_ai=run_ai)
        except (PipelineError, ValueError) as exc:
            return "", None, None, None, str(exc)
        return f"Frame {index + 1} / {plan.count}", images[0], images[1], images[2], note

    def on_cancel() -> str:
        logic.request_cancel()
        return "Cancel requested. Processing stops before the next frame."

    class View:
        def __init__(self) -> None:
            self.panel = _panel("STATUS\n\nStarting.")
            self.caption = ""
            self.preview: str | None = None
            self.lines: list[str] = []
            self.result = _panel("RESULT\n------\nRunning.")
            self.files: list[str] | None = None
            self.video: str | None = None
            self.queue = _panel(logic.format_queue([]))
            self.slider: object = gr.update()
            self.slider_caption = gr.update()
            self.images: tuple[object, object, object, object] = (
                gr.update(), gr.update(), gr.update(), gr.update(),
            )  # fmt: skip
            self.state: object = gr.update()
            self.browser_note = ""

        def apply(self, event: logic.ConverterEvent) -> None:
            if event.panel:
                self.panel = _panel(event.panel)
            if event.preview_path:
                self.preview = event.preview_path
            if event.caption:
                self.caption = event.caption
            if event.queue is not None:
                self.queue = _panel(logic.format_queue(event.queue))
            if event.log_line:
                self.lines.append(stamp_log(event.log_line))
                del self.lines[:-300]
            if event.error:
                self.result = _panel("Error\n------\n" + event.error)
            if event.result is not None:
                result = event.result
                self.result = _panel(logic.format_result(result))
                self.files = [str(p) for p in result.files if p.is_file()]
                self.video = str(result.browser_preview) if result.browser_preview else None
                self.browser_note = (
                    "Browser preview only. The production files are in Output files."
                    if self.video
                    else ""
                )
                review = result.review
                if review is not None and review.frame_numbers:
                    middle = len(review.frame_numbers) // 2
                    images, number = logic.review_frame(review, middle, logic.PREVIEW_BACKGROUND)
                    self.slider = gr.update(
                        interactive=len(review.frame_numbers) > 1,
                        minimum=0,
                        maximum=max(len(review.frame_numbers) - 1, 1),
                        value=middle,
                    )
                    self.slider_caption = f"Frame {number + 1} / {review.total_frames}"
                    self.images = (*images, "Processed frames: scrub the slider to compare.")
                    self.state = {"mode": "review", "review": review}

        def packet(self, *, running: bool) -> tuple[object, ...]:
            return (
                gr.update(interactive=not running),
                self.panel,
                self.caption,
                self.preview,
                "\n".join(self.lines),
                self.result,
                self.files,
                self.video,
                self.browser_note,
                self.queue,
                self.slider,
                self.slider_caption,
                *self.images,
                self.state,
            )

    def on_convert(uploaded: object, *values: object):
        view = View()
        try:
            paths = coerce_upload_paths(uploaded)
            if not paths:
                raise PipelineError("Choose a video first.")
            settings = _settings(values)
        except (PipelineError, ValueError) as exc:
            view.result = _panel("Error\n------\n" + str(exc))
            yield view.packet(running=False)
            return
        yield view.packet(running=True)
        try:
            for event in logic.iter_converter_queue(paths, settings):
                view.apply(event)
                yield view.packet(running=True)
        except Exception as exc:
            logger.exception("Converter failed")
            view.result = _panel("Error\n------\n" + str(exc))
        yield view.packet(running=False)

    # ------------------------------------------------------------------ wiring

    background_mode.change(
        on_mode, inputs=[background_mode, ai_backend], outputs=[chroma_group, ai_group, mode_note]
    )
    ai_backend.change(on_backend, inputs=[ai_backend], outputs=[ai_status])
    resize.change(on_resize, inputs=[resize], outputs=[resize_width, resize_height, keep_aspect])
    formats.change(on_formats, inputs=[formats], outputs=[gif_warning])
    uploads.change(
        on_files,
        inputs=[uploads, start, end, fps],
        outputs=[
            source_card, source_video, estimate, frame_slider, frame_caption,
            original_image, mask_image, result_image, preview_note, queue_box, view_state,
        ],
    )  # fmt: skip
    for control in (start, end, fps):
        control.change(
            on_estimate,
            inputs=[uploads, start, end, fps],
            outputs=[estimate, frame_slider, frame_caption],
        )
    preview_inputs = [frame_slider, uploads, view_state, *setting_inputs, preview_background]
    preview_outputs = [frame_caption, original_image, mask_image, result_image, preview_note]
    frame_slider.release(show_frame, inputs=preview_inputs, outputs=preview_outputs)
    preview_background.change(show_frame, inputs=preview_inputs, outputs=preview_outputs)
    preview_button.click(run_preview, inputs=preview_inputs, outputs=preview_outputs)
    cancel_button.click(on_cancel, outputs=[live_caption])
    convert_button.click(
        on_convert,
        inputs=[uploads, *setting_inputs],
        outputs=[
            convert_button, progress, live_caption, live_preview, job_log, result_box,
            output_files, browser_video, browser_note, queue_box, frame_slider, frame_caption,
            *image_outputs, view_state,
        ],
        concurrency_limit=1,
    )  # fmt: skip


def build_converter_demo() -> gr.Blocks:
    if gr is None:
        raise PipelineError('Gradio is not installed. Install it with: pip install -e ".[gui]"')
    with gr.Blocks(title="Video Converter") as demo:
        build_converter_tab()
    return demo


def converter_output_dirs() -> list[str]:
    return [str(Path.cwd() / "output"), str(Path.cwd() / "work")]
