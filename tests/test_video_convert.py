import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from midnight_upscale.background import BackgroundSettings
from midnight_upscale.encode import encode_sequence
from midnight_upscale.progress import (
    JobCancelled,
    PipelineProgressEvent,
    ProgressBus,
    bind_bus,
    reset_bus,
)
from midnight_upscale.segmentation import register_remover
from midnight_upscale.spritesheet import SpritesheetSettings
from midnight_upscale.video_convert import (
    STAGE_REMOVE,
    ConvertSettings,
    RunObserver,
    convert_video,
)
from midnight_upscale.video_decode import decode_single_frame
from midnight_upscale.video_inspect import inspect_video
from midnight_upscale.video_transform import ResizeSettings
from tests.test_chroma_background import FakeRemover
from tests.video_fixtures import GREEN, green_screen_frame, make_video, write_png_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")

FRAMES = 10


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    frames = [green_screen_frame(i) for i in range(FRAMES)]
    return make_video(tmp_path / "red_square.mp4", frames, fps=10)


def _settings(tmp_path: Path, **overrides) -> ConvertSettings:
    values = {
        "output_dir": tmp_path / "out",
        "work_dir": tmp_path / "work",
        "overwrite": True,
    }
    values.update(overrides)
    return ConvertSettings(**values)


def _run_with_bus(source: Path, settings: ConvertSettings, on_event=None, observer=None):
    bus = ProgressBus()
    if on_event is not None:
        bus.add_listener(on_event)
    token = bind_bus(bus)
    try:
        return convert_video(source, settings, observer)
    finally:
        reset_bus(token)


def _webm_frames(path: Path, count: int) -> list[np.ndarray]:
    info = inspect_video(path)
    return [np.asarray(decode_single_frame(info, i)) for i in range(count)]


def test_mp4_to_transparent_webm(clip: Path, tmp_path: Path) -> None:
    result = convert_video(clip, _settings(tmp_path, formats=("webm",)))
    webm = result.outputs["webm"][0]
    assert webm.name == "red_square_transparent.webm"
    assert result.frame_count == FRAMES
    assert result.background_label == "AUTO -> CHROMA KEY"
    info = inspect_video(webm)
    assert info.has_alpha
    assert info.frame_count == FRAMES
    assert (info.width, info.height) == (64, 48)
    frames = _webm_frames(webm, FRAMES)
    for index, frame in enumerate(frames):
        assert frame[0, 0, 3] < 10  # background is transparent
        left = 2 + index * 4
        assert frame[22, left + 6, 3] > 245  # the square is opaque and in the right place
        visible = frame[frame[..., 3] > 200]
        assert not ((visible[:, 1].astype(int) - visible[:, 0].astype(int)) > 40).any()  # no green


def test_mp4_to_spritesheet_matches_frames(clip: Path, tmp_path: Path) -> None:
    settings = _settings(tmp_path, formats=("spritesheet",), sheet=SpritesheetSettings(padding=2))
    result = convert_video(clip, settings)
    png, json_path, bundle = result.outputs["spritesheet"]
    assert (png.name, json_path.name) == (
        "red_square_spritesheet.png",
        "red_square_spritesheet.json",
    )
    assert bundle.name == "red_square_spritesheet.zip"
    with zipfile.ZipFile(bundle) as archive:
        assert sorted(archive.namelist()) == [json_path.name, png.name]
    assert result.downloads == [bundle]  # offered as one zip, not loose parts
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["frame_count"] == FRAMES
    assert data["frame_width"] == 64 and data["frame_height"] == 48
    assert data["duration_ms"] == 1000
    sheet = np.asarray(Image.open(png))
    assert sheet.shape[2] == 4
    for entry in data["frames"]:
        x, y = entry["x"], entry["y"]
        cell = sheet[y : y + 48, x : x + 64]
        assert cell[0, 0, 3] == 0
        left = 2 + entry["index"] * 4
        assert cell[22, left + 6, 3] == 255  # coordinates really point at that frame
        assert cell[22, left + 6, 0] > 200


def test_all_requested_formats_are_written_and_validated(clip: Path, tmp_path: Path) -> None:
    formats = ("webm", "apng", "gif", "png_sequence", "spritesheet")
    result = convert_video(clip, _settings(tmp_path, formats=formats, crop=True))
    names = {path.name for path in result.files}
    assert {
        "red_square_transparent.webm",
        "red_square_transparent.apng",
        "red_square_preview.gif",
        "red_square_frames",
        "red_square_spritesheet.png",
        "red_square_spritesheet.json",
    } <= names
    assert any("GIF does not preserve" in warning for warning in result.warnings)
    sizes = {Image.open(p).size for p in (tmp_path / "out" / "red_square_frames").glob("*.png")}
    assert len(sizes) == 1


def test_crop_uses_one_box_for_every_frame(clip: Path, tmp_path: Path) -> None:
    settings = _settings(tmp_path, formats=("png_sequence",), crop=True, padding=4, center=True)
    result = convert_video(clip, settings)
    frames = sorted(result.outputs["png_sequence"][0].glob("*.png"))
    assert len(frames) == FRAMES
    assert {Image.open(f).size for f in frames} == {(12 + (FRAMES - 1) * 4 + 8, 12 + 8)}
    first = np.asarray(Image.open(frames[0]))
    last = np.asarray(Image.open(frames[-1]))
    # The square moves 4 px per frame inside the fixed canvas; it was not re-centered.
    assert first[10, 4 + 1, 3] == 255 and first[10, 4 + 1 + FRAMES * 4 - 4, 3] == 0
    assert last[10, 4 + 1 + (FRAMES - 1) * 4, 3] == 255


def test_resize_keeps_alpha_and_changes_size(clip: Path, tmp_path: Path) -> None:
    settings = _settings(
        tmp_path, formats=("png_sequence",), resize=ResizeSettings(mode="scale", scale=2.0)
    )
    result = convert_video(clip, settings)
    frames = sorted(result.outputs["png_sequence"][0].glob("*.png"))
    image = Image.open(frames[0])
    assert image.size == (128, 96) and image.mode == "RGBA"
    alpha = np.asarray(image)[..., 3]
    assert alpha.min() == 0 and alpha.max() == 255


def test_odd_canvas_size_still_encodes_webm(tmp_path: Path) -> None:
    frames = [green_screen_frame(i, size=(63, 47)) for i in range(4)]
    source = make_video(tmp_path / "odd.mp4", frames, fps=10, codec_args=[
        "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "12",
    ])  # fmt: skip
    result = convert_video(source, _settings(tmp_path, formats=("webm",)))
    info = inspect_video(result.outputs["webm"][0])
    assert (info.width, info.height) == (63, 47)


def test_explicit_fps_changes_frame_count_as_planned(clip: Path, tmp_path: Path) -> None:
    result = convert_video(clip, _settings(tmp_path, formats=("png_sequence",), fps=5))
    assert result.frame_count == 5
    assert sum(result.durations_ms) == 1000
    files = list(result.outputs["png_sequence"][0].glob("*.png"))
    assert len(files) == 5


def test_trim_selects_a_range(clip: Path, tmp_path: Path) -> None:
    settings = _settings(tmp_path, formats=("png_sequence",), start_sec=0.2, end_sec=0.6)
    result = convert_video(clip, settings)
    assert result.frame_count == 4
    first = np.asarray(Image.open(sorted(result.outputs["png_sequence"][0].glob("*.png"))[0]))
    assert first[22, 2 + 2 * 4 + 6, 3] == 255  # starts at source frame 2


def test_gif_source(tmp_path: Path) -> None:
    frames = [green_screen_frame(i) for i in range(6)]
    source = make_video(tmp_path / "blob.gif", frames, fps=10)
    result = convert_video(source, _settings(tmp_path, formats=("png_sequence",)))
    assert result.frame_count == 6
    assert result.background_label.startswith("AUTO -> CHROMA KEY")


def test_existing_alpha_is_preserved(tmp_path: Path) -> None:
    directory = tmp_path / "rgba"
    directory.mkdir()
    for index in range(4):
        frame = np.zeros((32, 32, 4), dtype=np.uint8)
        frame[8:24, 4 + index * 4 : 20 + index * 4] = (30, 90, 220, 255)
        Image.fromarray(frame).save(directory / f"{index:06d}.png")
    source = tmp_path / "alpha.webm"
    encode_sequence(
        sorted(directory.glob("*.png")),
        [100] * 4,
        source,
        fmt="webm",
        loop=0,
        crf=12,
        webm_pix_fmt="yuva420p",
    )
    assert inspect_video(source).has_alpha
    result = convert_video(source, _settings(tmp_path, formats=("png_sequence",)))
    assert result.background_label == "AUTO -> PRESERVE ALPHA"
    frame = np.asarray(Image.open(sorted(result.outputs["png_sequence"][0].glob("*.png"))[0]))
    assert frame[0, 0, 3] == 0
    assert frame[16, 16, 3] > 240
    assert abs(int(frame[16, 16, 2]) - 220) < 25  # color survived


def test_variable_frame_durations_reach_the_spritesheet(tmp_path: Path) -> None:
    directory = tmp_path / "vfr_frames"
    write_png_frames(directory, [green_screen_frame(i) for i in range(5)])
    durations = [100, 100, 300, 100, 200]
    source = tmp_path / "vfr.webm"
    encode_sequence(
        sorted(directory.glob("*.png")), durations, source, fmt="webm", loop=0, crf=12,
        webm_pix_fmt="yuva420p",
    )  # fmt: skip
    result = convert_video(source, _settings(tmp_path, formats=("spritesheet", "webm")))
    data = json.loads(result.outputs["spritesheet"][-2].read_text(encoding="utf-8"))
    assert [f["duration_ms"] for f in data["frames"]] == durations
    assert data["duration_ms"] == sum(durations)
    assert inspect_video(result.outputs["webm"][0]).durations_ms == durations


def test_progress_reports_every_frame(clip: Path, tmp_path: Path) -> None:
    seen: list[tuple[str, int | None, int | None]] = []

    def listener(event: PipelineProgressEvent) -> None:
        seen.append((event.stage, event.frames_done, event.frames_total))

    _run_with_bus(clip, _settings(tmp_path, formats=("png_sequence",)), listener)
    removing = [(done, total) for stage, done, total in seen if stage == STAGE_REMOVE and done]
    assert [done for done, _total in removing] == list(range(1, FRAMES + 1))
    assert {total for _done, total in removing} == {FRAMES}
    stages = []
    for stage, _done, _total in seen:
        if not stages or stages[-1] != stage:
            stages.append(stage)
    assert stages[0] == "Inspect input" and stages[-1] == "Finished"
    assert stages.index("Decode video") < stages.index(STAGE_REMOVE) < stages.index("Export")


def test_live_preview_is_throttled_and_final_frame_is_shown(clip: Path, tmp_path: Path) -> None:
    previews: list[tuple[Path, str]] = []
    observer = RunObserver(preview=lambda path, caption: previews.append((path, caption)))
    convert_video(clip, _settings(tmp_path, formats=("png_sequence",)), observer)
    assert 1 <= len(previews) < FRAMES
    assert previews[-1][1].startswith("Latest processed frame")
    assert previews[-1][1].endswith(f"{FRAMES} / {FRAMES}")
    assert previews[0][0].suffix == ".png"


def test_cancel_stops_between_frames_and_leaves_no_output(clip: Path, tmp_path: Path) -> None:
    bus = ProgressBus()

    def listener(event: PipelineProgressEvent) -> None:
        if event.stage == STAGE_REMOVE and (event.frames_done or 0) >= 3:
            bus.cancel_event.set()

    bus.add_listener(listener)
    token = bind_bus(bus)
    settings = _settings(tmp_path, formats=("webm", "spritesheet"))
    messages: list[str] = []
    try:
        with pytest.raises(JobCancelled):
            convert_video(clip, settings, RunObserver(log=messages.append))
    finally:
        reset_bus(token)
    assert bus.snapshot().status == "CANCELLED"
    assert not list((tmp_path / "out").glob("*.webm"))
    assert not list((tmp_path / "out").glob("*spritesheet*"))
    log = (tmp_path / "out" / "logs" / "red_square_converter.log").read_text(encoding="utf-8")
    assert "CANCELLED" in log  # the log is kept
    assert not (tmp_path / "work" / "red_square_converter").exists()  # intermediates cleaned


def test_cancel_keeps_workdir_when_asked(clip: Path, tmp_path: Path) -> None:
    bus = ProgressBus()
    bus.add_listener(
        lambda e: bus.cancel_event.set() if e.stage == STAGE_REMOVE and e.frames_done else None
    )
    token = bind_bus(bus)
    try:
        with pytest.raises(JobCancelled):
            convert_video(clip, _settings(tmp_path, keep_workdir=True))
    finally:
        reset_bus(token)
    assert (tmp_path / "work" / "red_square_converter").exists()


def test_failure_removes_partial_outputs_and_reports(clip: Path, tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        formats=("webm", "spritesheet"),
        sheet=SpritesheetSettings(max_size=16),  # a 64x48 frame cannot fit
    )
    with pytest.raises(Exception, match="does not fit"):
        convert_video(clip, settings)
    assert not list((tmp_path / "out").glob("*.webm"))


def test_ai_segmentation_with_mocked_model_runs_the_stabilize_stage(
    tmp_path: Path,
) -> None:
    register_remover("fake", FakeRemover)
    gray = [green_screen_frame(i, background=(90, 90, 90)) for i in range(6)]
    source = make_video(tmp_path / "gray.mp4", gray, fps=10)
    settings = _settings(
        tmp_path,
        formats=("png_sequence",),
        background=BackgroundSettings(mode="auto", ai_backend="fake", ai_model="tiny"),
        temporal="medium",
    )
    stages: list[str] = []
    result = _run_with_bus(
        source, settings, lambda e: stages.append(e.stage) if e.stage not in stages[-1:] else None
    )
    assert result.background_label == "AUTO -> AI SEGMENTATION"
    assert "Stabilize mask" in stages
    frame = np.asarray(Image.open(sorted(result.outputs["png_sequence"][0].glob("*.png"))[3]))
    assert frame[0, 0, 3] == 0 and frame[22, 2 + 3 * 4 + 6, 3] == 255


def test_no_comfyui_is_needed(clip: Path, tmp_path: Path) -> None:
    # The conversion above ran with no ComfyUI client, URL, or workflow configured at all.
    import midnight_upscale.video_convert as module

    assert "comfy" not in " ".join(sorted(vars(module)))
    assert GREEN == (0, 177, 64)


def test_changing_only_the_outputs_reuses_cached_frames(clip: Path, tmp_path: Path) -> None:
    first = convert_video(clip, _settings(tmp_path, formats=("webm",)))
    assert first.outputs["webm"][0].is_file()
    seen: list[str] = []
    messages: list[str] = []
    second_settings = _settings(tmp_path, formats=("spritesheet", "apng"))
    result = _run_with_bus(
        clip,
        second_settings,
        lambda e: seen.append(e.message),
        RunObserver(log=messages.append),
    )
    assert any("Reusing cached" in m for m in messages)
    assert not [m for m in seen if m.startswith("Frame ")]  # nothing was reprocessed
    assert result.frame_count == FRAMES
    assert {p.name for p in result.files} >= {
        "red_square_spritesheet.png",
        "red_square_transparent.apng",
    }
    assert result.background_label == first.background_label


def test_changed_processing_settings_invalidate_the_cache(clip: Path, tmp_path: Path) -> None:
    convert_video(clip, _settings(tmp_path, formats=("png_sequence",)))
    messages: list[str] = []
    changed = _settings(tmp_path, formats=("png_sequence",), crop=True)
    result = convert_video(clip, changed, RunObserver(log=messages.append))
    assert not any("Reusing cached" in m for m in messages)
    assert Image.open(next(result.outputs["png_sequence"][0].glob("*.png"))).size != (64, 48)
    messages.clear()
    off = _settings(tmp_path, formats=("png_sequence",), crop=True, use_cache=False)
    convert_video(clip, off, RunObserver(log=messages.append))
    assert not any("Reusing cached" in m for m in messages)


def test_edge_pixels_do_not_keep_the_old_background_color() -> None:
    from midnight_upscale.video_transform import fill_transparent_rgb

    rgba = np.zeros((9, 9, 4), dtype=np.uint8)
    rgba[:, :4] = (120, 40, 30, 255)  # dark foreground
    rgba[:, 4] = (255, 255, 255, 120)  # semi-transparent edge still carrying white
    rgba[:, 5:, :3] = (255, 255, 255)  # transparent white background
    fused = fill_transparent_rgb(rgba)
    assert np.array_equal(fused[..., 3], rgba[..., 3])
    assert fused[4, 4, 0] < 160 and fused[4, 5, 0] < 160  # edge color now comes from the body
