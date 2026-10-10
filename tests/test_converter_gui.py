import shutil
from pathlib import Path

import pytest

from midnight_upscale import converter_logic as logic
from midnight_upscale.progress import PipelineProgressEvent
from midnight_upscale.utils import PipelineError
from midnight_upscale.video_convert import RunObserver
from tests.video_fixtures import green_screen_frame, make_video

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def _form(**overrides) -> dict:
    values = {
        "start": 0,
        "end": 0,
        "fps": "Source",
        "background_mode": "Auto",
        "key_color": "#00ff00",
        "tolerance": 30,
        "softness": 20,
        "spill": 60,
        "edge_cleanup": "Light",
        "hard_mask": False,
        "sample_key": False,
        "ai_backend": "rembg",
        "ai_model": "isnet-general-use",
        "temporal": "Low",
        "crop": False,
        "padding": "4",
        "center": True,
        "resize": "Keep source",
        "resize_width": 0,
        "resize_height": 0,
        "keep_aspect": True,
        "formats": ["Transparent WebM"],
        "sheet_columns": "Auto",
        "sheet_padding": "2",
        "max_texture": "4096",
        "power_of_two": False,
        "output_dir": "output",
        "overwrite": False,
        "keep_workdir": False,
        "use_cache": True,
    }
    values.update(overrides)
    return values


def test_defaults_match_the_spec() -> None:
    settings = logic.settings_from_form(**_form())
    assert settings.formats == ("webm",)
    assert settings.fps is None
    assert settings.background.mode == "auto"
    assert settings.temporal == "low"
    assert settings.crop is False and settings.padding == 4 and settings.center is True
    assert settings.sheet.padding == 2 and settings.sheet.max_size == 4096
    assert settings.sheet.columns is None and settings.sheet.power_of_two is False
    assert settings.resize.mode == "source"


def test_form_values_are_translated() -> None:
    settings = logic.settings_from_form(
        **_form(
            fps="24",
            background_mode="Chroma Key",
            key_color="#00b140",
            hard_mask=True,
            resize="Custom",
            resize_width=256,
            formats=["PNG Spritesheet", "GIF Preview"],
            sheet_columns="8",
            max_texture="2048",
            power_of_two=True,
            end=2.5,
        )
    )
    assert settings.fps == 24.0
    assert settings.background.mode == "chroma"
    assert settings.background.chroma.key_color == (0, 177, 64)
    assert settings.background.chroma.hard_mask is True
    assert settings.resize.mode == "custom" and settings.resize.width == 256
    assert settings.formats == ("spritesheet", "gif")
    assert settings.sheet.columns == 8 and settings.sheet.max_size == 2048
    assert settings.end_sec == 2.5


@pytest.mark.parametrize(
    "overrides",
    [
        {"formats": []},
        {"fps": "fast"},
        {"fps": "-3"},
        {"start": 3, "end": 2},
        {"max_texture": "tiny"},
    ],
)
def test_bad_form_values_are_reported(overrides: dict) -> None:
    with pytest.raises(PipelineError):
        logic.settings_from_form(**_form(**overrides))


def test_ai_status_tells_how_to_install(monkeypatch) -> None:
    from midnight_upscale import segmentation

    monkeypatch.setitem(segmentation._AVAILABILITY, "rembg", lambda: False)
    text = logic.ai_status_text("rembg")
    assert "AI background removal is not installed." in text
    assert 'pip install -e ".[bgremove]"' in text


def test_status_panel_shows_real_frame_progress() -> None:
    observer = RunObserver()
    observer.background_label = "AUTO -> CHROMA KEY"
    observer.skipped.add("Stabilize mask")
    event = PipelineProgressEvent(
        stage="Remove background",
        frames_done=84,
        frames_total=125,
        frames_kind="completed",
        elapsed_seconds=27.0,
        status="ACTIVE",
    )
    text = logic.format_converter_status(event, observer)
    assert "Removing background" in text
    assert "84 / 125" in text
    assert "00:00:27" in text
    assert "AUTO -> CHROMA KEY" in text
    assert "[>] 4. Remove background" in text
    assert "[x] 1. Inspect input" in text
    assert "[-] 5. Stabilize mask (skipped)" not in text  # not reached yet
    event.stage = "Crop / pad"
    later = logic.format_converter_status(event, observer)
    assert "[-] 5. Stabilize mask (skipped)" in later


@needs_ffmpeg
def test_estimate_text_states_dropped_and_repeated_frames(tmp_path: Path) -> None:
    from midnight_upscale.video_convert import ConvertSettings, estimate_output

    source = make_video(tmp_path / "c.mp4", [green_screen_frame(i) for i in range(10)], fps=10)
    info = logic.cached_info(source)
    plan = estimate_output(info, ConvertSettings(fps=5))
    text = logic.format_estimate(info, plan)
    assert "Frames:   5" in text
    assert "5 source frames will be dropped" in text
    same = logic.format_estimate(info, estimate_output(info, ConvertSettings()))
    assert "every frame kept" in same


@needs_ffmpeg
def test_queue_runs_a_file_and_reports_result(tmp_path: Path) -> None:
    source = make_video(tmp_path / "a.mp4", [green_screen_frame(i) for i in range(6)], fps=10)
    settings = logic.settings_from_form(
        **_form(formats=["PNG Spritesheet"], output_dir=str(tmp_path / "out"), overwrite=True)
    )
    events = list(logic.iter_converter_queue([source], settings))
    results = [e.result for e in events if e.result is not None]
    assert len(results) == 1 and results[0].frame_count == 6
    assert not [e.error for e in events if e.error]
    assert events[-1].finished
    assert any(e.log_line for e in events)
    states = [dict(e.queue)["a.mp4"] for e in events if e.queue]
    assert states[0] == "WAITING" and "PROCESSING" in states and states[-1] == "DONE"
    assert "RESULT" in logic.format_result(results[0])


def test_queue_requires_a_file_and_unique_names(tmp_path: Path) -> None:
    settings = logic.settings_from_form(**_form())
    assert "Choose a video" in list(logic.iter_converter_queue([], settings))[0].error
    clash = list(
        logic.iter_converter_queue([tmp_path / "x" / "a.mp4", tmp_path / "y" / "a.mp4"], settings)
    )
    assert "share a name" in clash[0].error


def test_only_one_converter_job_runs_at_a_time(tmp_path: Path) -> None:
    settings = logic.settings_from_form(**_form())
    assert logic.CONVERTER_LOCK.acquire(blocking=False)
    try:
        events = list(logic.iter_converter_queue([tmp_path / "a.mp4"], settings))
    finally:
        logic.CONVERTER_LOCK.release()
    assert "already running" in events[0].error


def test_ai_work_is_refused_while_an_upscale_holds_the_gpu() -> None:
    from midnight_upscale.gui_logic import JOB_LOCK

    assert JOB_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(PipelineError, match="using the GPU"):
            with logic.gpu_guard():
                pass
    finally:
        JOB_LOCK.release()
    with logic.gpu_guard():  # free again
        pass


@needs_ffmpeg
def test_cancel_through_the_queue(tmp_path: Path) -> None:
    source = make_video(tmp_path / "b.mp4", [green_screen_frame(i) for i in range(10)], fps=10)
    settings = logic.settings_from_form(**_form(output_dir=str(tmp_path / "out"), overwrite=True))
    events = []
    for event in logic.iter_converter_queue([source], settings):
        events.append(event)
        if event.preview_path:
            logic.request_cancel()
    assert [e.error for e in events if "Cancelled" in e.error]
    assert dict(events[-1].queue)["b.mp4"] == "CANCELLED"
    assert not list((tmp_path / "out").glob("*.webm"))


def test_demo_builds_with_both_tabs() -> None:
    pytest.importorskip("gradio")
    from midnight_upscale.gui import build_demo

    demo = build_demo()
    assert demo is not None
