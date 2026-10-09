import shutil
from pathlib import Path

import pytest
from PIL import Image

from midnight_upscale.encode import (
    build_concat_lines,
    encode_sequence,
    probe_video,
    webm_command,
)
from midnight_upscale.inspect import inspect_gif
from midnight_upscale.models import JobMetadata
from midnight_upscale.utils import (
    ZERO_DELAY_ENCODED_MS,
    duration_tolerance_sec,
    encoded_duration_ms,
    expected_encoded_duration_ms,
)
from tests.conftest import character_frame, save_rgba_gif


def test_gif_variable_durations_survive_inspection(tmp_path: Path) -> None:
    path = tmp_path / "loop.gif"
    frames = [character_frame(index, mark=(1, index + 1)) for index in range(4)]
    save_rgba_gif(path, frames, [40, 80, 40, 120], loop=0)
    info = inspect_gif(path)
    assert info.frame_durations_ms == [40, 80, 40, 120]
    assert info.total_duration_ms == 280
    assert info.durations_constant is False


def test_concat_list_preserves_each_delay_and_repeats_the_last_frame() -> None:
    lines = build_concat_lines(["000000.png", "000001.png", "000002.png"], [40, 80, 120])
    assert lines[0] == "ffconcat version 1.0"
    assert lines[1:] == [
        "file '000000.png'",
        "option framerate 1000",
        "duration 0.040000",
        "file '000001.png'",
        "option framerate 1000",
        "duration 0.080000",
        "file '000002.png'",
        "option framerate 1000",
        "duration 0.120000",
        "file '000002.png'",
        "option framerate 1000",
    ]


def test_zero_delay_is_encoded_as_ten_milliseconds_without_changing_other_frames() -> None:
    assert encoded_duration_ms(0) == ZERO_DELAY_ENCODED_MS
    assert encoded_duration_ms(40) == 40
    assert expected_encoded_duration_ms([0, 40, 80]) == 10 + 40 + 80
    assert duration_tolerance_sec([40, 80, 40]) < 0.2


def test_webm_command_keeps_alpha_and_variable_frame_rate() -> None:
    command = webm_command(
        Path("concat.txt"),
        Path("out.webm"),
        crf=18,
        pix_fmt="yuva420p",
        duration_sec=0.28,
    )
    assert command[0] == "ffmpeg"
    assert "-fps_mode" in command and command[command.index("-fps_mode") + 1] == "vfr"
    assert "yuva420p" in command
    assert command[command.index("-auto-alt-ref") + 1] == "0"
    assert "alpha_mode=1" in command
    assert "-r" not in command
    assert command[command.index("-t") + 1] == "0.280000"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed"
)
def test_ffmpeg_preserves_variable_duration(tmp_path: Path) -> None:
    frames = []
    for index, color in enumerate(((255, 0, 0, 128), (0, 255, 0, 0), (0, 0, 255, 255))):
        image = Image.new("RGBA", (16, 16), color)
        path = tmp_path / f"{index:06d}.png"
        image.save(path)
        frames.append(path)
    durations = [40, 80, 120]
    output = tmp_path / "clip.webm"
    metadata = JobMetadata(
        source="synthetic",
        original_width=8,
        original_height=8,
        frame_count=3,
        frame_durations_ms=durations,
        total_duration_ms=240,
        gif_loop_count=0,
        durations_constant=False,
        estimated_fps=12.5,
        has_transparency=True,
        has_semitransparency=True,
        requested_scale=2,
        target_width=16,
        target_height=16,
        alpha_mode="nearest",
        edge_cleanup="off",
        interpolate="none",
    )
    encode_sequence(
        frames,
        durations,
        output,
        fmt="webm",
        loop=0,
        crf=30,
        webm_pix_fmt="yuva420p",
    )
    probed = probe_video(output)
    assert probed["width"] == 16
    assert probed["height"] == 16
    assert str(probed["pix_fmt"]).startswith("yuva")
    assert probed["duration_sec"] == pytest.approx(0.240, abs=0.05)
    from midnight_upscale.encode import assert_encoded_stream

    assert_encoded_stream(probed, metadata, fmt="webm", webm_pix_fmt="yuva420p")
