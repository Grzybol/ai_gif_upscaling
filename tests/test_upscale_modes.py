import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from midnight_upscale.upscale_modes import (
    DiskFrames,
    _combine,
    _export_mp4,
    _rgb_input,
    read_clip,
)
from midnight_upscale.video_inspect import inspect_video


def _sprite() -> np.ndarray:
    frame = np.zeros((8, 8, 4), dtype=np.uint8)
    frame[2:6, 2:6] = (220, 40, 50, 255)
    frame[:, :, :3][frame[:, :, 3] == 0] = 255  # hidden white matte
    return frame


def test_rgb_input_bleeds_visible_color_without_changing_alpha():
    source = _sprite()
    rgb = np.asarray(_rgb_input(source))
    assert tuple(rgb[1, 3]) == (220, 40, 50)
    assert source[1, 3, 3] == 0
    result = _combine(Image.fromarray(rgb), source, 2)
    assert result.shape == (16, 16, 4)
    assert result[0, 0, 3] == 0
    assert result[8, 8, 3] > 200


def test_spritesheet_zip_keeps_order_timing_and_alpha(tmp_path: Path):
    sheet = Image.new("RGBA", (16, 8))
    sheet.paste(Image.fromarray(_sprite()), (0, 0))
    second = _sprite()
    second[2:6, 2:6, :3] = (30, 190, 70)
    sheet.paste(Image.fromarray(second), (8, 0))
    source = tmp_path / "sheet.zip"
    with ZipFile(source, "w") as archive:
        import io
        data = io.BytesIO()
        sheet.save(data, format="PNG")
        archive.writestr("sheet.png", data.getvalue())
        archive.writestr("sheet.json", json.dumps({
            "frames": [
                {"index": i, "sheet": "sheet.png", "x": 8 * i, "y": 0,
                 "w": 8, "h": 8, "duration_ms": delay}
                for i, delay in enumerate([80, 120])
            ], "fps": None,
        }))
    clip = read_clip(source, tmp_path / "work")
    assert clip.durations_ms == [80, 120]
    assert tuple(clip.frames[0][3, 3, :3]) == (220, 40, 50)
    assert tuple(clip.frames[1][3, 3, :3]) == (30, 190, 70)
    assert clip.frames[0][0, 0, 3] == 0


def test_mp4_export_keeps_last_delay_and_frame_count(tmp_path: Path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is not installed")
    frames = []
    for index in range(3):
        path = tmp_path / f"{index:06d}.png"
        Image.fromarray(_sprite(), "RGBA").save(path)
        frames.append(path)
    destination = tmp_path / "result.mp4"
    _export_mp4(DiskFrames(frames), [80, 120, 200], destination, tmp_path / "work")
    info = inspect_video(destination)
    assert info.frame_count == 3
    assert info.width == 8 and info.height == 8
    assert abs(info.duration_sec - 0.4) < 0.02
    decoded = read_clip(destination, tmp_path / "decoded")
    assert decoded.durations_ms == [80, 120, 200]
