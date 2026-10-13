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
from midnight_upscale.video_convert import ConvertSettings, convert_video, preview_one_frame
from midnight_upscale.video_export import composite_on_background
from midnight_upscale.video_inspect import inspect_video
from midnight_upscale.white_background import (
    add_comics_outline,
    draw_outward_border,
    force_remove_white_edges,
    remove_white_background,
    tighten_drawn_matte,
)
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


def test_comics_outline_is_one_pixel_inside_and_preserves_alpha() -> None:
    frame = np.zeros((9, 9, 4), dtype=np.uint8)
    frame[2:7, 2:7] = (200, 100, 40, 255)
    frame[1, 4] = (250, 250, 250, 8)
    result = add_comics_outline(frame)
    assert tuple(result[2, 4]) == (100, 50, 20, 255)
    assert tuple(result[3, 4]) == (200, 100, 40, 255)
    assert tuple(result[4, 4]) == (200, 100, 40, 255)
    assert result[1, 4, 3] == 0
    np.testing.assert_array_equal(result[..., 3], np.where(frame[..., 3] < 16, 0, frame[..., 3]))


def test_drawn_white_uses_white_key_with_separate_mode() -> None:
    frame = np.full((12, 12, 4), 255, dtype=np.uint8)
    frame[3:9, 3:9, :3] = (30, 80, 20)
    resolved = resolve_background(
        BackgroundSettings(mode="drawn_white"), has_alpha=False, sample_frames=[frame]
    )
    keyed = remove_background(resolved, frame)
    outlined = add_comics_outline(keyed)
    assert resolved.mode == "drawn_white"
    assert keyed[5, 5, 3] == outlined[5, 5, 3] == 255
    assert outlined[3, 5, 0] < keyed[3, 5, 0]
    assert outlined[5, 5, 0] == keyed[5, 5, 0]


def test_drawn_white_keys_only_exact_rgb_white() -> None:
    frame = np.array([[
        [255, 255, 255, 255],
        [255, 255, 254, 255],
        [250, 250, 250, 255],
        [255, 250, 250, 8],
    ]], dtype=np.uint8)
    drawn = resolve_background(
        BackgroundSettings(mode="drawn_white"), has_alpha=False, sample_frames=[frame]
    )
    result = remove_background(drawn, frame)
    np.testing.assert_array_equal(result[0, 0], (0, 0, 0, 0))
    np.testing.assert_array_equal(result[0, 1:], frame[0, 1:])
    outlined = add_comics_outline(result, discard_faint_alpha=False)
    assert outlined[0, 3, 3] == 8
    white = resolve_background(
        BackgroundSettings(mode="white"), has_alpha=False, sample_frames=[frame]
    )
    assert remove_background(white, frame)[0, 2, 3] == 0


def test_legacy_drawn_white_restores_near_white_key_and_outline() -> None:
    frame = np.full((9, 9, 4), 255, dtype=np.uint8)
    frame[..., :3] = (250, 250, 250)
    frame[2:7, 2:7, :3] = (160, 40, 25)
    legacy = resolve_background(
        BackgroundSettings(mode="drawn_white_legacy"),
        has_alpha=False, sample_frames=[frame],
    )
    exact = resolve_background(
        BackgroundSettings(mode="drawn_white"),
        has_alpha=False, sample_frames=[frame],
    )
    legacy_keyed = remove_background(legacy, frame)
    exact_keyed = remove_background(exact, frame)
    assert legacy.mode == "drawn_white_legacy"
    assert legacy_keyed[0, 0, 3] == 0
    assert exact_keyed[0, 0, 3] == 255
    outlined = add_comics_outline(legacy_keyed)
    assert outlined[2, 4, 0] < legacy_keyed[2, 4, 0]
    assert outlined[4, 4, 0] == legacy_keyed[4, 4, 0]


def test_comics_gradient_fades_alpha_inward_by_depth() -> None:
    frame = np.zeros((11, 11, 4), dtype=np.uint8)
    frame[2:9, 2:9] = (120, 80, 40, 255)
    result = add_comics_outline(frame, fade_strength=0.8, fade_depth=3)
    assert result[2, 5, 3] == 51
    assert result[3, 5, 3] == 119
    assert result[4, 5, 3] == 187
    assert result[5, 5, 3] == 255
    assert result[2, 5, 0] == 60
    assert result[3, 5, 0] == 120


def test_forced_white_removal_respects_depth_shade_and_interior() -> None:
    frame = np.zeros((11, 11, 4), dtype=np.uint8)
    frame[1:10, 1:10] = (80, 40, 20, 255)
    frame[1, 5, :3] = (250, 250, 250)
    frame[2, 5, :3] = (230, 230, 230)
    frame[3, 5, :3] = (250, 250, 250)
    shallow = force_remove_white_edges(frame, depth=1, darkest_shade=245)
    assert shallow[1, 5, 3] == 0
    assert shallow[2, 5, 3] == 255
    assert shallow[3, 5, 3] == 255
    deeper = force_remove_white_edges(frame, depth=2, darkest_shade=225)
    assert deeper[1, 5, 3] == deeper[2, 5, 3] == 0
    assert deeper[3, 5, 3] == 255


def test_forced_white_removal_keys_border_canvas_in_none_mode() -> None:
    frame = np.full((9, 9, 4), 255, dtype=np.uint8)
    frame[2:7, 2:7, :3] = (30, 60, 90)
    frame[4, 4, :3] = (255, 255, 255)
    result = force_remove_white_edges(frame, depth=1, darkest_shade=245)
    assert result[0, 0, 3] == 0
    assert result[4, 4, 3] == 255
    assert result[2, 4, 3] == 255


def test_tight_drawn_matte_removes_tinted_halo_and_recovers_inner_color() -> None:
    frame = np.zeros((15, 15, 4), dtype=np.uint8)
    frame[2:13, 2:13] = (40, 30, 20, 255)
    frame[2, 2:13, :3] = (235, 205, 165)  # white mixed with skin/gold
    frame[3, 2:13, :3] = (180, 145, 105)  # still below the forced-white cutoff
    result = tighten_drawn_matte(frame)
    assert result[2, 7, 3] == 0
    assert 0 < result[3, 7, 3] <= 192
    assert result[3, 7, 0] < 100
    assert tuple(result[7, 7]) == (40, 30, 20, 255)


def test_tight_drawn_mode_resolves_separately() -> None:
    frame = np.full((12, 12, 4), 255, dtype=np.uint8)
    frame[3:9, 3:9, :3] = (30, 80, 20)
    resolved = resolve_background(
        BackgroundSettings(mode="drawn_white_tight"), has_alpha=False,
        sample_frames=[frame],
    )
    assert resolved.mode == "drawn_white_tight"
    assert remove_background(resolved, frame)[0, 0, 3] == 0


def test_outward_border_has_selected_color_and_preserves_opaque_artwork() -> None:
    frame = np.zeros((11, 11, 4), dtype=np.uint8)
    frame[2:9, 2:9] = (180, 120, 60, 255)
    frame[2, 5, 3] = 128
    result = draw_outward_border(frame, width=2, color=(10, 20, 200))
    pad = 4
    assert result.shape == (19, 19, 4)
    np.testing.assert_array_equal(result[pad + 3:pad + 8, pad + 3:pad + 8],
                                  frame[3:8, 3:8])
    assert tuple(result[pad + 1, pad + 5, :3]) == (10, 20, 200)
    assert result[pad + 1, pad + 5, 3] > 0
    assert tuple(result[pad + 4, pad + 5]) == (180, 120, 60, 255)
    assert tuple(result[0, 0]) == (0, 0, 0, 0)


def test_border_smoothing_blends_inward_without_changing_outward_shape() -> None:
    frame = np.zeros((15, 15, 4), dtype=np.uint8)
    frame[2:13, 2:13] = (200, 100, 40, 255)
    frame[2, 7, :3] = (0, 0, 0)  # intentional dark lineart
    plain = draw_outward_border(frame, width=1, color=(0, 0, 0))
    smooth = draw_outward_border(frame, width=1, color=(0, 0, 0), smooth_depth=4)
    pad = 3
    np.testing.assert_array_equal(plain[..., 3], smooth[..., 3])
    assert 0 < smooth[pad + 2, pad + 5, 0] < plain[pad + 2, pad + 5, 0]
    assert smooth[pad + 3, pad + 5, 0] > smooth[pad + 2, pad + 5, 0]
    np.testing.assert_array_equal(smooth[pad + 7, pad + 7], frame[7, 7])
    np.testing.assert_array_equal(smooth[pad + 2, pad + 7, :3], (0, 0, 0))
    outside = np.pad(frame[..., 3] == 0, pad, constant_values=True)
    np.testing.assert_array_equal(plain[..., :3][outside], smooth[..., :3][outside])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_drawn_border_appears_in_preview_with_selected_background(tmp_path: Path) -> None:
    frame = np.full((48, 48, 4), 255, dtype=np.uint8)
    frame[10:38, 10:38, :3] = (60, 80, 100)
    source = make_video(tmp_path / "comic.mp4", [frame, frame], fps=10)
    settings = ConvertSettings(
        background=BackgroundSettings(
            mode="drawn_white", draw_border=True,
            border_width=2, border_color=(10, 20, 200),
        ),
    )
    _, mask, result, _ = preview_one_frame(inspect_video(source), settings, 0)
    border = np.all(result[..., :3] == (10, 20, 200), axis=2) & (mask > 0)
    assert np.any(border)
    assert result[24 + 4, 24 + 4, 0] != 10
    composed = np.asarray(composite_on_background(result, "magenta"))
    assert tuple(composed[0, 0]) == (255, 0, 255)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_outward_border_preview_matches_exported_frame(tmp_path: Path) -> None:
    frame = np.full((48, 48, 4), 255, dtype=np.uint8)
    frame[8:40, 8:40, :3] = (160, 90, 45)
    source = make_video(tmp_path / "outline.mp4", [frame, frame], fps=10)
    settings = ConvertSettings(
        background=BackgroundSettings(
            mode="drawn_white", draw_border=True, border_width=1,
            border_color=(0, 0, 0), border_smooth_depth=4,
        ),
        formats=("png_sequence",), output_dir=tmp_path / "out", work_dir=tmp_path / "work",
        use_cache=False,
    )
    _, _, preview, _ = preview_one_frame(inspect_video(source), settings, 0)
    result = convert_video(source, settings)
    frames = sorted(result.outputs["png_sequence"][0].glob("*.png"))
    with Image.open(frames[0]) as image:
        np.testing.assert_array_equal(np.asarray(image.convert("RGBA")), preview)
    assert preview.shape[:2] == (54, 54)


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
