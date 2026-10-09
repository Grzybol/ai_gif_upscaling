import shutil
from pathlib import Path

import pytest
from PIL import Image

from midnight_upscale.gui_logic import (
    GUI_BATCH_SIZES,
    JOB_LOCK,
    OVERLAP_UNSUPPORTED,
    GuiJobConfig,
    PreflightResult,
    browser_preview_command,
    check_comfy_connection,
    composite_rgba_on_checkerboard,
    format_gui_error,
    format_source_card,
    gui_config_from_form,
    iter_queue,
    note_from_log,
    output_filename,
    pipeline_arguments,
    preflight,
    run_prepared_job,
    validate_gui_config,
)
from midnight_upscale.inspect import inspect_gif
from midnight_upscale.models import GifInspection, JobMetadata
from midnight_upscale.utils import ComfyError, PipelineError, WorkflowConfigError
from tests.conftest import character_frame, save_rgba_gif


def _config(tmp_path: Path, **overrides: object) -> GuiJobConfig:
    source = tmp_path / "wow.gif"
    if not source.exists():
        source.write_bytes(b"gif")
    values: dict[str, object] = {
        "source": source,
        "backend": "seedvr2",
        "scale": 2,
        "batch_size": 5,
        "temporal_overlap": 1,
        "alpha_mode": "lanczos",
        "edge_cleanup": "auto",
        "output_format": "webm",
        "keep_workdir": False,
        "comfy_url": "http://127.0.0.1:8188",
        "workflow_path": "",
        "output_dir": tmp_path / "output",
        "overwrite": False,
        "verbose": False,
        "interpolate": "none",
    }
    values.update(overrides)
    return GuiJobConfig(**values)  # type: ignore[arg-type]


def test_output_filename_matches_the_source_scale_and_backend() -> None:
    assert output_filename(Path("wow.gif"), 2, "seedvr2", "webm") == "wow_2x_seedvr2.webm"
    assert output_filename(Path("wow.gif"), 4, "frame-upscale", "gif") == "wow_4x_frame-upscale.gif"
    assert 7 not in GUI_BATCH_SIZES


def test_form_settings_map_onto_pipeline_arguments(tmp_path: Path) -> None:
    source = tmp_path / "wow.gif"
    source.write_bytes(b"gif")
    config = gui_config_from_form(
        source,
        backend_label="Numz custom node",
        scale_label="2x",
        batch_size=5,
        temporal_overlap=1,
        alpha_label="Lanczos",
        edge_label="Auto",
        format_label="WebM Alpha",
        keep_workdir=False,
        comfy_url="http://127.0.0.1:8188",
        workflow_path="",
        output_dir=str(tmp_path / "output"),
        overwrite=False,
        verbose=False,
        interpolate_label="None",
    )
    arguments = pipeline_arguments(config, tmp_path / "runtime.yaml")
    assert arguments["prepare"] == {
        "scale": 2,
        "workdir": Path("work") / "wow",
        "alpha_mode": "lanczos",
        "edge_cleanup": "auto",
        "interpolate": "none",
        "overwrite": False,
    }
    assert arguments["upscale"]["backend"] == "seedvr2"
    assert arguments["upscale"]["batch_size"] == 5
    assert arguments["upscale"]["temporal_overlap"] == 1
    assert arguments["upscale"]["temporal_mode"] == "auto"
    assert arguments["upscale"]["comfy_url"] == "http://127.0.0.1:8188"
    assert arguments["finalize"]["fmt"] == "webm"
    assert arguments["finalize"]["output"].name == "wow_2x_seedvr2.webm"
    assert arguments["keep_workdir"] is False


def test_batch_size_7_is_rejected() -> None:
    config = GuiJobConfig(
        source=Path("wow.gif"),
        backend="seedvr2",
        scale=2,
        batch_size=7,
        temporal_overlap=1,
        alpha_mode="lanczos",
        edge_cleanup="auto",
        output_format="webm",
        keep_workdir=False,
        comfy_url="http://127.0.0.1:8188",
        workflow_path="",
        output_dir=Path("output"),
        overwrite=False,
        verbose=False,
    )
    with pytest.raises(PipelineError, match=r"4n\+1"):
        validate_gui_config(config)


def test_source_card_uses_existing_inspection(tmp_path: Path) -> None:
    source = tmp_path / "wow.gif"
    save_rgba_gif(source, [character_frame(0), character_frame(1, mark=(2, 2))], [40, 40])
    card = format_source_card(inspect_gif(source))
    assert "wow.gif" in card
    assert "8 × 8" in card
    assert "2 frames" in card
    assert "Transparency: yes" in card
    assert "Variable timing: NO" in card


def test_log_line_becomes_batch_progress() -> None:
    prompts, frames, detail = note_from_log(
        "SeedVR2: 125 RGB frames, 25 ComfyUI prompt(s), node batch_size 5",
        None,
        None,
    )
    assert (prompts, frames, detail) == (25, 125, "")
    _prompts, _frames, detail = note_from_log(
        "ComfyUI batch 2: frames 10-14 (5 files)",
        prompts,
        frames,
    )
    assert detail == "Processing SeedVR2 batch 3 / 25 — frames 15 / 125"


def test_checkerboard_keeps_opaque_color_and_hides_nothing_with_black() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((8, 8), (255, 0, 0, 255))
    composed = composite_rgba_on_checkerboard(image)
    assert composed.getpixel((8, 8)) == (255, 0, 0)
    corner = composed.getpixel((0, 0))
    assert corner != (0, 0, 0)
    assert corner[0] == corner[1] == corner[2]


def test_browser_preview_command_does_not_replace_the_webm() -> None:
    source = Path("output/wow_2x_seedvr2.webm")
    preview = Path("output/wow_2x_seedvr2.browser-preview.mp4")
    command = browser_preview_command("ffmpeg", source, Path("board.png"), preview)
    assert command[-1] == str(preview)
    assert str(source) in command
    assert command[-1] != str(source)


def test_preflight_rejects_missing_file_and_non_gif(tmp_path: Path) -> None:
    missing = _config(tmp_path, source=tmp_path / "gone.gif")
    with pytest.raises(PipelineError, match="not found"):
        preflight(missing)
    png = tmp_path / "still.png"
    png.write_bytes(b"png")
    with pytest.raises(PipelineError, match="only accepts GIF"):
        preflight(_config(tmp_path, source=png))


def test_preflight_reports_missing_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(PipelineError, match="not found") as caught:
        preflight(_config(tmp_path))
    assert format_gui_error(caught.value) == (
        "FFmpeg was not found. Install FFmpeg and restart the terminal."
    )


def test_preflight_reports_missing_comfyui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "ffmpeg")

    class Offline:
        def object_info(self) -> dict[str, object]:
            raise ComfyError("Could not reach ComfyUI at http://127.0.0.1:8188: refused")

        def close(self) -> None:
            return None

    with pytest.raises(ComfyError) as caught:
        preflight(_config(tmp_path), connect=lambda _url: Offline())
    assert (
        format_gui_error(caught.value, comfy_url="http://127.0.0.1:8188")
        == "ComfyUI is not reachable at http://127.0.0.1:8188."
    )


def test_preflight_reports_a_missing_workflow_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "ffmpeg")
    workflow = tmp_path / "flow.json"
    workflow.write_text(
        """
        {
          "4": {"class_type": "SeedVR2VideoUpscaler", "inputs": {"image": ["1", 0]}},
          "24": {"class_type": "NotInstalled", "inputs": {}}
        }
        """,
        encoding="utf-8",
    )

    class Client:
        def object_info(self) -> dict[str, object]:
            return {"SeedVR2VideoUpscaler": {}}

        def close(self) -> None:
            return None

    config = _config(tmp_path, workflow_path=str(workflow))
    with pytest.raises(WorkflowConfigError, match="node 24"):
        preflight(config, connect=lambda _url: Client())


def test_connection_check_uses_live_object_info() -> None:
    class Client:
        def object_info(self) -> dict[str, object]:
            return {
                "SeedVR2VideoUpscaler": {
                    "input": {"optional": {"temporal_overlap": ["INT", {"default": 0}]}}
                }
            }

        def close(self) -> None:
            return None

    status = check_comfy_connection("http://127.0.0.1:8188", connect=lambda _url: Client())
    assert status.summary == "ComfyUI: CONNECTED"
    assert status.seedvr2_available is True
    assert status.overlap_supported is True
    assert "SeedVR2VideoUpscaler" in "\n".join(status.node_lines)
    assert status.overlap_message == ""

    class NoOverlap:
        def object_info(self) -> dict[str, object]:
            return {"SeedVR2VideoUpscaler": {"input": {"required": {"image": ["IMAGE"]}}}}

        def close(self) -> None:
            return None

    bare = check_comfy_connection("http://127.0.0.1:8188", connect=lambda _url: NoOverlap())
    assert bare.overlap_message == OVERLAP_UNSUPPORTED
    assert bare.seedvr2_available is True


def test_run_prepared_job_calls_the_pipeline_with_those_settings(tmp_path: Path) -> None:
    source = tmp_path / "wow.gif"
    source.write_bytes(b"gif")
    config = _config(tmp_path, source=source, output_format="gif", keep_workdir=True)
    seen: dict[str, object] = {}

    def inspect(path: Path) -> GifInspection:
        seen["inspect"] = path
        return GifInspection(
            source=str(path),
            width=8,
            height=8,
            frame_count=2,
            frame_durations_ms=[40, 40],
            total_duration_ms=80,
            gif_loop_count=0,
            durations_constant=True,
            estimated_fps=25.0,
            has_transparency=True,
            has_semitransparency=False,
        )

    def prepare(path: Path, **kwargs: object) -> Path:
        seen["prepare"] = kwargs
        job = tmp_path / "job"
        job.mkdir()
        JobMetadata(
            source=str(path),
            original_width=8,
            original_height=8,
            frame_count=2,
            frame_durations_ms=[40, 40],
            total_duration_ms=80,
            gif_loop_count=0,
            durations_constant=True,
            estimated_fps=25.0,
            has_transparency=True,
            has_semitransparency=False,
            requested_scale=2,
            target_width=16,
            target_height=16,
            alpha_mode="lanczos",
            edge_cleanup="simple",
            interpolate="none",
        ).save(job / "metadata.json")
        return job

    def upscale(job_dir: Path, **kwargs: object) -> None:
        seen["upscale"] = kwargs

    def finalize(job_dir: Path, **kwargs: object) -> Path:
        seen["finalize"] = kwargs
        destination = Path(kwargs["output"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gif")
        return destination

    result = run_prepared_job(
        config,
        PreflightResult(config_path=tmp_path / "runtime.yaml", temporal_overlap_supported=True),
        inspect=inspect,
        prepare=prepare,
        upscale=upscale,
        finalize=finalize,
    )
    assert seen["prepare"]["scale"] == 2  # type: ignore[index]
    assert seen["prepare"]["edge_cleanup"] == "auto"  # type: ignore[index]
    assert seen["upscale"]["batch_size"] == 5  # type: ignore[index]
    assert seen["upscale"]["temporal_overlap"] == 1  # type: ignore[index]
    assert seen["finalize"]["fmt"] == "gif"  # type: ignore[index]
    assert result.output_path.name == "wow_2x_seedvr2.gif"
    assert "Edge cleanup: auto -> simple" in result.log_lines
    assert "2 frames" in result.result_text


def test_second_queue_is_rejected_while_one_is_active(tmp_path: Path) -> None:
    assert JOB_LOCK.acquire(blocking=False)
    try:
        events = list(iter_queue([_config(tmp_path)]))
    finally:
        JOB_LOCK.release()
    assert events
    assert "already running" in events[0].error


def test_queue_rejects_duplicate_work_directories(tmp_path: Path) -> None:
    first = tmp_path / "one" / "wow.gif"
    second = tmp_path / "two" / "wow.gif"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"gif")
    second.write_bytes(b"gif")
    events = list(iter_queue([_config(tmp_path, source=first), _config(tmp_path, source=second)]))
    assert "same work directory" in events[0].error


def test_demo_builds() -> None:
    pytest.importorskip("gradio")
    from midnight_upscale.gui import build_demo

    demo = build_demo()
    assert demo is not None
