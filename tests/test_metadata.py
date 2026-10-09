from pathlib import Path

import pytest
from PIL import Image

from midnight_upscale.inspect import inspect_gif
from midnight_upscale.models import JobMetadata
from midnight_upscale.pipeline import prepare_asset
from midnight_upscale.utils import next_available_file
from tests.conftest import character_frame, save_rgba_gif


def test_variable_and_constant_durations(tmp_path: Path) -> None:
    variable = tmp_path / "variable.gif"
    frames = [character_frame(index, mark=(index + 1, 2)) for index in range(4)]
    save_rgba_gif(variable, frames, [40, 80, 40, 120])
    info = inspect_gif(variable)
    assert info.frame_count == 4
    assert info.frame_durations_ms == [40, 80, 40, 120]
    assert info.total_duration_ms == 280
    assert info.durations_constant is False
    assert info.estimated_fps == pytest.approx(4 / 0.280)
    assert info.width == 8
    assert info.height == 8
    assert info.gif_loop_count == 0
    assert info.has_transparency is True

    constant = tmp_path / "constant.gif"
    save_rgba_gif(constant, frames[:2], [50, 50])
    constant_info = inspect_gif(constant)
    assert constant_info.durations_constant is True
    assert constant_info.frame_durations_ms == [50, 50]


def test_missing_loop_extension_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "once.gif"
    save_rgba_gif(path, [character_frame(0), character_frame(1, mark=(2, 2))], [40, 40], loop=None)
    info = inspect_gif(path)
    assert info.gif_loop_count is None
    assert "play once" in info.format_report()


def test_metadata_roundtrip_keeps_required_fields(tmp_path: Path) -> None:
    source = tmp_path / "hero.gif"
    save_rgba_gif(source, [character_frame(0), character_frame(1, mark=(3, 3))], [40, 80])
    workdir = prepare_asset(
        source,
        scale=2,
        workdir=tmp_path / "job",
        alpha_mode="nearest",
        edge_cleanup="off",
        interpolate="none",
        overwrite=False,
    )
    original = JobMetadata.load(workdir / "metadata.json")
    payload = original.to_dict()
    for key in (
        "original_width",
        "original_height",
        "frame_count",
        "frame_durations_ms",
        "total_duration_ms",
        "gif_loop_count",
        "requested_scale",
        "target_width",
        "target_height",
    ):
        assert key in payload
    assert original.frame_durations_ms == [40, 80]
    assert original.requested_scale == 2
    assert (original.target_width, original.target_height) == (16, 16)
    restored = JobMetadata.load(workdir / "metadata.json")
    assert restored.to_dict() == payload


def test_zero_delay_is_recorded_and_not_rewritten_in_metadata(tmp_path: Path) -> None:
    source = tmp_path / "zero.gif"
    save_rgba_gif(source, [character_frame(0), character_frame(1, mark=(4, 1))], [0, 40])
    info = inspect_gif(source)
    assert info.frame_durations_ms == [0, 40]
    workdir = prepare_asset(
        source,
        scale=1,
        workdir=tmp_path / "job",
        alpha_mode="nearest",
        edge_cleanup="off",
        interpolate="none",
        overwrite=False,
    )
    metadata = JobMetadata.load(workdir / "metadata.json")
    assert metadata.frame_durations_ms == [0, 40]
    assert metadata.duration_warnings


def test_prepare_does_not_touch_the_source(tmp_path: Path) -> None:
    source = tmp_path / "hero.gif"
    save_rgba_gif(source, [character_frame(0), character_frame(1, mark=(2, 2))], [40, 40])
    before = source.read_bytes()
    stamp = source.stat().st_mtime_ns
    prepare_asset(
        source,
        scale=2,
        workdir=tmp_path / "job",
        alpha_mode="lanczos",
        edge_cleanup="off",
        interpolate="none",
        overwrite=False,
    )
    assert source.read_bytes() == before
    assert source.stat().st_mtime_ns == stamp
    with Image.open(tmp_path / "job" / "source_rgba" / "000000.png") as frame:
        assert frame.mode == "RGBA"
        assert frame.getpixel((0, 0))[3] == 255
        assert frame.getpixel((7, 7))[3] == 0


def test_reprocessing_the_same_gif_uses_the_next_workdir_version(tmp_path: Path) -> None:
    source = tmp_path / "download_1.gif"
    save_rgba_gif(source, [character_frame(0), character_frame(1, mark=(2, 2))], [30, 40])
    job = tmp_path / "download_1"

    def prepare(overwrite: bool) -> Path:
        return prepare_asset(
            source,
            scale=2,
            workdir=job,
            alpha_mode="nearest",
            edge_cleanup="off",
            interpolate="none",
            overwrite=overwrite,
        )

    first = prepare(False)
    second = prepare(False)
    third = prepare(False)
    assert first == job.resolve()
    assert second == (tmp_path / "download_1_v1").resolve()
    assert third == (tmp_path / "download_1_v2").resolve()
    assert (first / "metadata.json").is_file()
    assert (second / "metadata.json").is_file()
    replaced = prepare(True)
    assert replaced == first
    assert not (replaced / "rgb" / "000002.png").exists()


def test_existing_output_file_gets_the_next_version(tmp_path: Path) -> None:
    output = tmp_path / "download_1_2x_seedvr2-native.webm"
    output.write_bytes(b"a")
    (tmp_path / "download_1_2x_seedvr2-native_v1.webm").write_bytes(b"b")
    assert next_available_file(output) == tmp_path / "download_1_2x_seedvr2-native_v2.webm"
