"""Manual color key and fixed-position painted protection."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image

from midnight_upscale.background import BackgroundSettings, remove_background, resolve_background
from midnight_upscale.manual_key import parse_color_ranges, save_painted_mask
from midnight_upscale.utils import PipelineError
from midnight_upscale.video_convert import (
    ConvertSettings, _finalize_frame, convert_video, preview_one_frame,
)
from midnight_upscale.video_decode import decode_single_frame
from midnight_upscale.video_inspect import inspect_video
from midnight_upscale.video_transform import ResizeSettings
from tests.video_fixtures import make_video


def test_color_rules_accept_exact_range_and_tolerance() -> None:
    assert parse_color_ranges("#FFFFFF\n#F8F8F8..#FEFEFE\n#FAFAFA +/- 3") == (
        ((255, 255, 255), (255, 255, 255)),
        ((248, 248, 248), (254, 254, 254)),
        ((247, 247, 247), (253, 253, 253)),
    )
    with pytest.raises(PipelineError, match="line 1"):
        parse_color_ranges("#FFFFFF..#000000")


def test_painted_area_keeps_white_details_while_added_range_removes_stripes(
    tmp_path: Path,
) -> None:
    frame = np.full((8, 8, 4), (80, 40, 20, 255), dtype=np.uint8)
    frame[0, :] = (250, 250, 250, 255)  # compressed white stripe
    frame[4, 4] = (255, 255, 255, 255)  # tooth inside the subject
    layer = np.zeros_like(frame)
    layer[4, 4] = (0, 183, 255, 140)
    mask_path = save_painted_mask(
        {"background": frame, "layers": [layer], "composite": frame},
        tmp_path / "masks",
    )
    source = tmp_path / "character.mp4"
    settings = BackgroundSettings(
        mode="drawn_white", manual_colors=parse_color_ranges("#F8F8F8..#FEFEFE"),
        protection_mask=mask_path, protection_source=str(source),
    )
    resolved = resolve_background(
        settings, has_alpha=False, sample_frames=[frame], source_path=source,
    )
    keyed = remove_background(resolved, frame)
    assert keyed[0, 0, 3] == 0
    np.testing.assert_array_equal(keyed[4, 4], frame[4, 4])
    other = resolve_background(
        settings, has_alpha=False, sample_frames=[frame],
        source_path=tmp_path / "other.mp4",
    )
    assert remove_background(other, frame)[4, 4, 3] == 0


def test_protection_survives_forced_removal_and_border(tmp_path: Path) -> None:
    frame = np.zeros((12, 12, 4), dtype=np.uint8)
    frame[2:10, 2:10] = (120, 40, 30, 255)
    frame[2, 5] = (255, 255, 255, 255)
    mask = np.zeros((12, 12), dtype=bool)
    mask[2, 5] = True
    settings = BackgroundSettings(
        mode="drawn_white", draw_border=True, border_width=1,
        force_white_removal=True, force_white_depth=1, force_white_shade=245,
    )
    result = _finalize_frame(frame, settings, "drawn_white", protection=mask)
    np.testing.assert_array_equal(result[5, 8], frame[2, 5])


def test_empty_brush_is_disabled(tmp_path: Path) -> None:
    frame = np.zeros((4, 4, 4), dtype=np.uint8)
    assert save_painted_mask(
        {"background": frame, "layers": [frame], "composite": frame}, tmp_path
    ) == ""


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_painted_protection_preview_matches_resized_export(tmp_path: Path) -> None:
    frame = np.full((48, 48, 4), 255, dtype=np.uint8)
    frame[10:38, 10:38, :3] = (170, 40, 30)
    frame[20:25, 20:25, :3] = 255  # intentional white detail
    source = make_video(tmp_path / "comic.mp4", [frame, frame], fps=10)
    info = inspect_video(source)
    decoded = np.asarray(decode_single_frame(info, 0))
    layer = np.zeros_like(decoded)
    layer[19:26, 19:26] = (0, 183, 255, 140)
    mask_path = save_painted_mask(
        {"background": decoded, "layers": [layer], "composite": decoded},
        tmp_path / "masks",
    )
    settings = ConvertSettings(
        background=BackgroundSettings(
            mode="drawn_white", manual_colors=parse_color_ranges("#F0F0F0..#FFFFFF"),
            protection_mask=mask_path, protection_source=str(source.resolve()),
            draw_border=True, border_width=1,
        ),
        resize=ResizeSettings(mode="scale", scale=2),
        formats=("png_sequence",), output_dir=tmp_path / "out",
        work_dir=tmp_path / "work", use_cache=False,
    )
    _, _, preview, _ = preview_one_frame(info, settings, 0)
    output = convert_video(source, settings)
    frames = sorted(output.outputs["png_sequence"][0].glob("*.png"))
    with Image.open(frames[0]) as image:
        np.testing.assert_array_equal(np.asarray(image.convert("RGBA")), preview)
    assert preview[47, 47, 3] == 255  # protected white detail after 2× and 3 px gutter
    assert preview[5, 5, 3] == 0
