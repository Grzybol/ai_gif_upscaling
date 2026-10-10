"""White-key leaf edges and browser playback of exported spritesheets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from midnight_upscale.background import BackgroundSettings, remove_background, resolve_background
from midnight_upscale.spritesheet import SpritesheetSettings
from midnight_upscale.video_convert import ConvertSettings, convert_video
from midnight_upscale.white_background import remove_white_background
from tests.video_fixtures import make_video


def test_white_key_recovers_a_thin_leaf_and_enclosed_white_gap() -> None:
    color = np.array([40, 90, 20], dtype=np.float32)
    frame = np.full((32, 32, 4), 255, dtype=np.uint8)
    frame[4:28, 15:17, :3] = color
    frame[4:28, 14, :3] = np.rint(color * 0.4 + 255 * 0.6)
    frame[12:20, 17:27, :3] = color
    frame[14:18, 20:24, :3] = 255
    result = remove_white_background(frame)
    np.testing.assert_allclose(result[8, 14, :3], color, atol=3)
    assert abs(int(result[8, 14, 3]) - 102) <= 3
    assert result[15, 21, 3] == 0
    assert result[8, 15, 3] == 255
    assert result[0, 0, 3] == 0


def test_manual_white_mode_needs_no_ai_model() -> None:
    frame = np.full((12, 12, 4), 255, dtype=np.uint8)
    frame[3:9, 3:9, :3] = (30, 80, 20)
    resolved = resolve_background(
        BackgroundSettings(mode="white"), has_alpha=False, sample_frames=[frame]
    )
    result = remove_background(resolved, frame)
    assert resolved.label == "WHITE BACKGROUND"
    assert result[0, 0, 3] == 0
    assert result[5, 5, 3] == 255


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_spritesheet_player_uses_exported_atlas_and_timing(tmp_path: Path) -> None:
    frames = []
    for index in range(3):
        frame = np.full((48, 48, 4), 255, dtype=np.uint8)
        frame[10:38, 12 + index : 28 + index, :3] = (30, 80, 20)
        frames.append(frame)
    source = make_video(tmp_path / "plant.mp4", frames, fps=10)
    settings = ConvertSettings(
        background=BackgroundSettings(mode="white"),
        formats=("spritesheet",),
        sheet=SpritesheetSettings(columns=1, padding=0, max_size=48),
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        browser_preview=True,
    )
    result = convert_video(source, settings)
    player = result.browser_player
    assert player is not None and player.is_file()
    html = player.read_text(encoding="utf-8")
    payload = json.loads(html.split("const data=", 1)[1].split(";\n", 1)[0])
    atlas = json.loads(result.outputs["spritesheet"][-2].read_text(encoding="utf-8"))
    assert payload["images"] == [sheet["file"] for sheet in atlas["sheets"]]
    assert len(payload["images"]) == 3
    assert [item["image"] for item in payload["frames"]] == [0, 1, 2]
    assert [item["duration"] for item in payload["frames"]] == result.durations_ms
    with ZipFile(result.outputs["spritesheet"][-1]) as bundle:
        assert bundle.testzip() is None
        assert player.name in bundle.namelist()
    with Image.open(result.outputs["spritesheet"][0]) as image:
        rgba = np.asarray(image)
        assert rgba[0, 0, 3] == 0
        assert rgba[20, 20, 3] == 255
