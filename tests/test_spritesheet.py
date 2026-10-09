import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from midnight_upscale.spritesheet import (
    SpritesheetSettings,
    plan_spritesheet,
    sheet_names,
    write_spritesheets,
)
from midnight_upscale.utils import PipelineError
from midnight_upscale.video_export import output_paths, resolve_output_stem


def _write_frames(directory: Path, count: int, size: tuple[int, int] = (10, 6)) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        frame = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        frame[..., 0] = index * 10
        frame[..., 3] = 255 if index % 2 == 0 else 128
        path = directory / f"{index:06d}.png"
        Image.fromarray(frame).save(path)
        paths.append(path)
    return paths


def test_single_sheet_coordinates_follow_padding() -> None:
    plan = plan_spritesheet(10, 6, 6, SpritesheetSettings(columns=3, padding=2))
    assert len(plan.sheets) == 1
    sheet = plan.sheets[0]
    assert (sheet.columns, sheet.rows) == (3, 2)
    assert sheet.cells[0] == (2, 2)
    assert sheet.cells[1] == (2 + 12, 2)
    assert sheet.cells[3] == (2, 2 + 8)
    assert (sheet.width, sheet.height) == (2 + 3 * 12, 2 + 2 * 8)


def test_auto_columns_make_a_roughly_square_sheet() -> None:
    plan = plan_spritesheet(32, 32, 16, SpritesheetSettings(padding=0))
    assert (plan.sheets[0].columns, plan.sheets[0].rows) == (4, 4)


def test_zero_padding_cells_touch() -> None:
    plan = plan_spritesheet(8, 8, 4, SpritesheetSettings(columns=2, padding=0))
    assert plan.sheets[0].cells == ((0, 0), (8, 0), (0, 8), (8, 8))
    assert (plan.sheets[0].width, plan.sheets[0].height) == (16, 16)


def test_splits_into_several_sheets_without_exceeding_max_size() -> None:
    settings = SpritesheetSettings(padding=2, max_size=64)
    plan = plan_spritesheet(14, 14, 20, settings)  # 3 columns x 3 rows = 9 per sheet
    assert len(plan.sheets) == 3
    assert [s.frame_count for s in plan.sheets] == [9, 9, 2]
    assert [s.first_frame for s in plan.sheets] == [0, 9, 18]
    for sheet in plan.sheets:
        assert sheet.width <= 64 and sheet.height <= 64
    assert plan.frame_count == 20


def test_frame_larger_than_texture_is_rejected() -> None:
    with pytest.raises(PipelineError, match="does not fit"):
        plan_spritesheet(100, 100, 2, SpritesheetSettings(max_size=64))


def test_too_many_custom_columns_is_rejected() -> None:
    with pytest.raises(PipelineError, match="columns"):
        plan_spritesheet(30, 30, 5, SpritesheetSettings(columns=10, max_size=128))


def test_power_of_two_pads_the_sheet_not_the_frames() -> None:
    plan = plan_spritesheet(10, 6, 6, SpritesheetSettings(columns=3, padding=2, power_of_two=True))
    sheet = plan.sheets[0]
    assert (sheet.width, sheet.height) == (64, 32)
    assert (plan.frame_width, plan.frame_height) == (10, 6)


def test_power_of_two_never_exceeds_a_non_power_of_two_limit() -> None:
    plan = plan_spritesheet(
        20, 20, 200, SpritesheetSettings(padding=0, max_size=3000, power_of_two=True)
    )
    for sheet in plan.sheets:
        assert sheet.width <= 3000 and sheet.height <= 3000


def test_sheet_names() -> None:
    assert sheet_names("woman_idle", 1) == (
        ["woman_idle_spritesheet.png"],
        "woman_idle_spritesheet.json",
    )
    pngs, json_name = sheet_names("woman_idle", 3)
    assert pngs == [f"woman_idle_spritesheet_{i:02d}.png" for i in range(3)]
    assert json_name == "woman_idle_spritesheet.json"


def test_write_spritesheet_pixels_json_and_durations(tmp_path: Path) -> None:
    frames = _write_frames(tmp_path / "frames", 5)
    durations = [100, 100, 300, 100, 200]
    png = tmp_path / "sheet.png"
    json_path = tmp_path / "sheet.json"
    plan = write_spritesheets(
        frames, durations, [png], json_path, SpritesheetSettings(columns=2, padding=2)
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["frame_width"] == 10 and data["frame_height"] == 6
    assert data["frame_count"] == 5
    assert data["duration_ms"] == 800
    assert data["fps"] == pytest.approx(5 / 0.8, abs=1e-3)
    assert data["columns"] == 2 and data["rows"] == 3
    assert data["loop"] is True
    assert data["variable_timing"] is True
    assert [f["duration_ms"] for f in data["frames"]] == durations
    assert all(f["w"] == 10 and f["h"] == 6 for f in data["frames"])

    sheet = Image.open(png)
    assert sheet.mode == "RGBA"
    assert sheet.size == (plan.sheets[0].width, plan.sheets[0].height)
    pixels = np.asarray(sheet)
    for entry in data["frames"]:
        x, y, index = entry["x"], entry["y"], entry["index"]
        cell = pixels[y : y + 6, x : x + 10]
        assert (cell[..., 0] == index * 10).all()  # each frame landed in its own cell
        assert (cell[..., 3] == (255 if index % 2 == 0 else 128)).all()  # alpha copied, not blended
    assert pixels[0, 0, 3] == 0  # the padding is transparent


def test_write_spritesheet_multi_sheet_metadata(tmp_path: Path) -> None:
    frames = _write_frames(tmp_path / "frames", 20, size=(14, 14))
    names, json_name = sheet_names("clip", 3)
    pngs = [tmp_path / n for n in names]
    write_spritesheets(
        frames,
        [50] * 20,
        pngs,
        tmp_path / json_name,
        SpritesheetSettings(padding=2, max_size=64),
    )
    data = json.loads((tmp_path / json_name).read_text(encoding="utf-8"))
    assert [s["file"] for s in data["sheets"]] == names
    assert [f["sheet"] for f in data["frames"]][:10] == [0] * 9 + [1]
    for path in pngs:
        assert max(Image.open(path).size) <= 64
    assert data["variable_timing"] is False


def test_spritesheet_rejects_frames_of_different_size(tmp_path: Path) -> None:
    frames = _write_frames(tmp_path / "frames", 3)
    Image.new("RGBA", (11, 6)).save(frames[1])
    with pytest.raises(PipelineError, match="same size"):
        write_spritesheets(
            frames, [10, 10, 10], [tmp_path / "s.png"], tmp_path / "s.json", SpritesheetSettings()
        )


def test_output_names_are_deterministic() -> None:
    out = Path("out")
    paths = output_paths(
        "woman_idle", ("webm", "spritesheet", "gif", "apng", "png_sequence"), out, 1
    )
    assert paths["webm"] == [out / "woman_idle_transparent.webm"]
    assert paths["spritesheet"] == [
        out / "woman_idle_spritesheet.png",
        out / "woman_idle_spritesheet.json",
    ]
    assert paths["gif"] == [out / "woman_idle_preview.gif"]
    assert paths["apng"] == [out / "woman_idle_transparent.apng"]
    assert paths["png_sequence"] == [out / "woman_idle_frames"]


def test_existing_outputs_get_a_version_suffix_unless_overwriting(tmp_path: Path) -> None:
    formats = ("webm", "spritesheet")
    (tmp_path / "clip_transparent.webm").write_bytes(b"old")
    assert resolve_output_stem("clip", formats, tmp_path, 1, overwrite=False) == "clip_v1"
    assert resolve_output_stem("clip", formats, tmp_path, 1, overwrite=True) == "clip"
    (tmp_path / "clip_v1_transparent.webm").write_bytes(b"old")
    assert resolve_output_stem("clip", formats, tmp_path, 1, overwrite=False) == "clip_v2"
