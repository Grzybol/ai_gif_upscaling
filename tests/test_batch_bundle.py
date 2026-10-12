"""Batch converter downloads preserve every video's exports in one ZIP."""

from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from midnight_upscale.background import BackgroundSettings
from midnight_upscale.batch_bundle import create_batch_zip
from midnight_upscale.converter_logic import iter_converter_queue
from midnight_upscale.video_convert import ConvertSettings
from midnight_upscale.video_export import PNG_SEQUENCE
from tests.video_fixtures import make_video


def test_batch_zip_keeps_video_subdirectories_and_nested_frames(tmp_path: Path) -> None:
    first = tmp_path / "one.webm"
    second = tmp_path / "two.mp4"
    frames = tmp_path / "one_frames"
    frames.mkdir()
    (frames / "000000.png").write_bytes(b"frame")
    other = tmp_path / "two.webm"
    other.write_bytes(b"video")
    bundle = create_batch_zip(
        [(first, [frames]), (second, [other])], tmp_path, overwrite=False
    )
    with ZipFile(bundle) as archive:
        assert sorted(archive.namelist()) == [
            "one/one_frames/000000.png", "two/two.webm"
        ]
        assert archive.testzip() is None
    again = create_batch_zip([(first, [frames])], tmp_path, overwrite=False)
    assert again != bundle


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_multi_video_queue_emits_one_bundle(tmp_path: Path) -> None:
    frame = np.full((32, 32, 4), 255, dtype=np.uint8)
    frame[8:24, 8:24, :3] = (30, 90, 120)
    sources = [
        make_video(tmp_path / f"clip_{index}.mp4", [frame, frame], fps=10)
        for index in range(2)
    ]
    settings = ConvertSettings(
        background=BackgroundSettings(mode="white"),
        formats=(PNG_SEQUENCE,), output_dir=tmp_path / "output",
        work_dir=tmp_path / "work", browser_preview=False,
    )
    events = list(iter_converter_queue(sources, settings, bundle_batch=True))
    assert not [event.error for event in events if event.error]
    bundles = [event.bundle_path for event in events if event.bundle_path]
    assert len(bundles) == 1
    with ZipFile(bundles[0]) as archive:
        names = archive.namelist()
        assert any(name.startswith("clip_0/clip_0_frames/") for name in names)
        assert any(name.startswith("clip_1/clip_1_frames/") for name in names)
        assert archive.testzip() is None
