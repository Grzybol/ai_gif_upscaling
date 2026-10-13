from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from midnight_upscale.video_convert import ConvertSettings, convert_video
from midnight_upscale.video_decode import decode_plan, decode_single_frame, plan_frames
from midnight_upscale.video_export import export_spritesheet, zip_spritesheet
from midnight_upscale.video_inspect import inspect_video
from midnight_upscale.spritesheet import SpritesheetSettings
from midnight_upscale.utils import PipelineError
from midnight_upscale import spritesheet_input


def _upscale_style_zip(tmp_path: Path) -> Path:
    frames = []
    for index, color in enumerate(((220, 30, 40), (30, 210, 50), (50, 60, 230))):
        pixels = np.zeros((16, 16, 4), dtype=np.uint8)
        pixels[3:13, 3:13, :3] = color
        pixels[3:13, 3:13, 3] = 255
        path = tmp_path / f"frame_{index}.png"
        Image.fromarray(pixels).save(path)
        frames.append(path)
    files, _ = export_spritesheet(
        frames, [80, 120, 200], tmp_path, "upscaled",
        SpritesheetSettings(columns=2, padding=0, max_size=64),
    )
    return zip_spritesheet(files, tmp_path / "upscaled.zip")


def test_converter_reads_upscale_spritesheet_zip_with_timing_and_alpha(tmp_path: Path):
    source = _upscale_style_zip(tmp_path)
    info = inspect_video(source)
    assert info.frame_count == 3
    assert info.durations_ms == [80, 120, 200]
    assert info.duration_sec == 0.4
    assert info.has_alpha
    assert np.asarray(decode_single_frame(info, 1))[8, 8, 1] == 210

    plan = plan_frames(info)
    decoded = decode_plan(info, plan, tmp_path / "decoded", scratch=tmp_path / "scratch")
    assert len(decoded) == 3
    with Image.open(decoded[2]) as image:
        assert image.size == (16, 16)
        assert image.getpixel((0, 0))[3] == 0

    result = convert_video(
        source,
        ConvertSettings(formats=("png_sequence",), output_dir=tmp_path / "output",
                        work_dir=tmp_path / "work", use_cache=False),
    )
    assert result.frame_count == 3
    assert result.durations_ms == [80, 120, 200]
    assert result.background_label == "AUTO -> PRESERVE ALPHA"
    assert len(result.outputs["png_sequence"]) == 1
    output_frames = sorted(result.outputs["png_sequence"][0].glob("*.png"))
    assert len(output_frames) == 3
    with Image.open(output_frames[1]) as image:
        assert image.getpixel((0, 0))[3] == 0
        assert image.getpixel((8, 8))[1] == 210


def test_zip_without_spritesheet_metadata_reports_input_error(tmp_path: Path):
    source = tmp_path / "invalid.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("notes.txt", "not a spritesheet")
    with pytest.raises(PipelineError, match="JSON metadata"):
        inspect_video(source)


def test_gradio_upload_can_be_briefly_locked_on_windows(tmp_path: Path, monkeypatch):
    source = _upscale_style_zip(tmp_path)
    real_zip = spritesheet_input.ZipFile
    attempts = 0

    def locked_twice(path):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise PermissionError(13, "Permission denied", str(path))
        return real_zip(path)

    monkeypatch.setattr(spritesheet_input, "ZipFile", locked_twice)
    monkeypatch.setattr(spritesheet_input.time, "sleep", lambda _seconds: None)
    assert inspect_video(source).frame_count == 3
    assert attempts == 3


def test_persistent_upload_lock_has_actionable_error(tmp_path: Path, monkeypatch):
    source = _upscale_style_zip(tmp_path)

    def always_locked(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(spritesheet_input, "ZipFile", always_locked)
    monkeypatch.setattr(spritesheet_input.time, "sleep", lambda _seconds: None)
    with pytest.raises(PipelineError, match="still locked by Windows"):
        inspect_video(source)
