import shutil
from pathlib import Path

import pytest

from midnight_upscale.encode import encode_sequence
from midnight_upscale.utils import PipelineError
from midnight_upscale.video_decode import plan_frames
from midnight_upscale.video_inspect import (
    durations_from_times,
    format_video_card,
    inspect_video,
    parse_probe,
    parse_rate,
)
from tests.video_fixtures import green_screen_frame, make_video, write_png_frames

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def _payload(
    times: list[float], *, pix_fmt: str = "yuv420p", tags: dict | None = None, last: float = 0.04
) -> dict:
    stream: dict = {
        "codec_name": "h264",
        "width": 320,
        "height": 240,
        "pix_fmt": pix_fmt,
        "avg_frame_rate": "30000/1001",
        "nb_frames": str(len(times)),
    }
    if tags:
        stream["tags"] = tags
    packets = [{"pts_time": f"{t:.6f}", "duration_time": f"{last:.6f}"} for t in times]
    return {"streams": [stream], "packets": packets, "format": {"format_name": "mov,mp4"}}


def test_parse_rate() -> None:
    assert parse_rate("30000/1001") == pytest.approx(29.97, abs=0.001)
    assert parse_rate("25") == 25.0
    assert parse_rate("0/0") == 0.0
    assert parse_rate("N/A") == 0.0


def test_parse_probe_reads_fields_and_constant_timing() -> None:
    times = [i * 0.04 for i in range(25)]
    info = parse_probe(_payload(times), Path("clip.mp4"))
    assert (info.width, info.height, info.codec, info.pix_fmt) == (320, 240, "h264", "yuv420p")
    assert info.frame_count == 25
    assert info.fps == pytest.approx(25.0)
    assert info.duration_sec == pytest.approx(1.0)
    assert not info.variable_timing
    assert not info.alpha_declared


def test_parse_probe_keeps_variable_frame_timing() -> None:
    times = [0.0, 0.1, 0.2, 0.5, 0.6]
    info = parse_probe(_payload(times), Path("vfr.mp4"))
    assert info.variable_timing
    assert info.durations_ms[:4] == [100, 100, 300, 100]
    assert info.frame_count == 5


def test_parse_probe_sorts_b_frame_order_and_rebases_start() -> None:
    info = parse_probe(_payload([1.08, 1.0, 1.04, 1.12]), Path("b.mp4"))
    assert info.frame_times == [0.0, 0.04, 0.08, 0.12]


def test_parse_probe_detects_alpha_formats_and_webm_tag() -> None:
    assert parse_probe(_payload([0.0], pix_fmt="yuva420p"), Path("a.mov")).alpha_declared
    assert parse_probe(_payload([0.0], pix_fmt="bgra"), Path("a.gif")).alpha_declared
    tagged = parse_probe(_payload([0.0], tags={"ALPHA_MODE": "1"}), Path("a.webm"))
    assert tagged.alpha_declared


def test_parse_probe_rejects_files_without_video() -> None:
    with pytest.raises(PipelineError):
        parse_probe({"streams": [], "format": {}}, Path("x.mp4"))


def test_durations_sum_to_span_without_rounding_drift() -> None:
    times = [i / 29.97 for i in range(300)]
    durations = durations_from_times(times, 1 / 29.97)
    assert sum(durations) == round(300 / 29.97 * 1000)


def test_plan_source_keeps_every_frame_and_its_timing() -> None:
    info = parse_probe(_payload([0.0, 0.1, 0.2, 0.5, 0.6]), Path("vfr.mp4"))
    plan = plan_frames(info)
    assert plan.source_indices == [0, 1, 2, 3, 4]
    assert plan.durations_ms == info.durations_ms
    assert plan.dropped == 0 and plan.duplicated == 0


def test_plan_trim_selects_frames_by_time() -> None:
    info = parse_probe(_payload([i * 0.1 for i in range(10)], last=0.1), Path("c.mp4"))
    plan = plan_frames(info, start_sec=0.3, end_sec=0.7)
    assert plan.source_indices == [3, 4, 5, 6]


def test_plan_explicit_fps_reports_dropped_and_repeated_frames() -> None:
    info = parse_probe(_payload([i * 0.1 for i in range(10)], last=0.1), Path("c.mp4"))
    down = plan_frames(info, fps=5)
    assert down.count == 5
    assert down.dropped == 5 and down.duplicated == 0
    assert sum(down.durations_ms) == 1000
    up = plan_frames(info, fps=20)
    assert up.count == 20
    assert up.dropped == 0 and up.duplicated == 10


def test_plan_rejects_bad_ranges() -> None:
    info = parse_probe(_payload([i * 0.1 for i in range(10)], last=0.1), Path("c.mp4"))
    with pytest.raises(PipelineError):
        plan_frames(info, start_sec=2.0)
    with pytest.raises(PipelineError):
        plan_frames(info, fps=0)


@needs_ffmpeg
def test_inspect_real_mp4_and_card(tmp_path: Path) -> None:
    frames = [green_screen_frame(i) for i in range(8)]
    video = make_video(tmp_path / "clip.mp4", frames, fps=10)
    info = inspect_video(video)
    assert (info.width, info.height, info.frame_count) == (64, 48, 8)
    assert info.codec == "h264"
    assert not info.has_alpha
    card = format_video_card(info)
    assert "Has alpha:    NO" in card
    assert "Frame count:  8" in card


@needs_ffmpeg
def test_inspect_rejects_unsupported_extension(tmp_path: Path) -> None:
    bogus = tmp_path / "clip.avi"
    bogus.write_bytes(b"x")
    with pytest.raises(PipelineError, match="not a supported input"):
        inspect_video(bogus)


@needs_ffmpeg
def test_inspect_variable_frame_rate_webm(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    write_png_frames(frames_dir, [green_screen_frame(i) for i in range(5)])
    durations = [100, 100, 300, 100, 200]
    paths = sorted(frames_dir.glob("*.png"))
    out = tmp_path / "vfr.webm"
    encode_sequence(paths, durations, out, fmt="webm", loop=0, crf=18, webm_pix_fmt="yuva420p")
    info = inspect_video(out)
    assert info.variable_timing
    assert info.durations_ms == durations
    assert info.alpha_declared
    assert not info.has_alpha  # alpha plane exists but every pixel is opaque
