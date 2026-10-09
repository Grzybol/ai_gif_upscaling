"""Encode a PNG sequence without resampling variable GIF delays to a fixed FPS."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from midnight_upscale.models import JobMetadata
from midnight_upscale.utils import (
    PipelineError,
    duration_tolerance_sec,
    encoded_duration_ms,
    expected_encoded_duration_ms,
    require_binary,
    tail_text,
)
from midnight_upscale.validate import assert_duration_close

WEBM_PIX_FMTS = ("yuva420p", "yuva444p")


def build_concat_lines(frame_names: list[str], durations_ms: list[int]) -> list[str]:
    """Build an ffmpeg concat demuxer list.

    The last file is repeated with no duration of its own. The concat demuxer
    applies each ``duration`` line to the file above it and ignores a trailing
    duration, so without the extra line the last frame's delay is dropped.
    """

    if len(frame_names) != len(durations_ms):
        raise PipelineError(
            f"Concat list has {len(frame_names)} files and {len(durations_ms)} durations"
        )
    if not frame_names:
        raise PipelineError("No frames to encode")

    lines = ["ffconcat version 1.0"]
    for name, duration in zip(frame_names, durations_ms, strict=True):
        lines.append(f"file {_quote_concat_name(name)}")
        # PNG inputs default to 25 fps. A 30 ms delay then shares a timestamp
        # with the next frame and the encoder drops it. 1000 fps is a 1 ms clock.
        lines.append("option framerate 1000")
        lines.append(f"duration {encoded_duration_ms(duration) / 1000.0:.6f}")
    lines.append(f"file {_quote_concat_name(frame_names[-1])}")
    lines.append("option framerate 1000")
    return lines


def write_concat_file(frame_paths: list[Path], durations_ms: list[int], destination: Path) -> Path:
    lines = build_concat_lines([path.name for path in frame_paths], durations_ms)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def encode_sequence(
    frame_paths: list[Path],
    durations_ms: list[int],
    output: Path,
    *,
    fmt: str,
    loop: int | None,
    crf: int,
    webm_pix_fmt: str,
) -> None:
    if fmt not in {"webm", "apng", "gif"}:
        raise PipelineError(f"Unknown format {fmt!r}. Choose webm, apng, or gif")
    if len(frame_paths) != len(durations_ms):
        raise PipelineError("Frame count and duration count differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_path = frame_paths[0].parent / "concat.txt"
    write_concat_file(frame_paths, durations_ms, concat_path)
    ffmpeg = require_binary("ffmpeg")
    duration_sec = expected_encoded_duration_ms(durations_ms) / 1000.0
    if fmt == "webm":
        _run(
            webm_command(
                concat_path,
                output,
                crf=crf,
                pix_fmt=webm_pix_fmt,
                ffmpeg=ffmpeg,
                duration_sec=duration_sec,
                frame_count=len(frame_paths),
            )
        )
    elif fmt == "apng":
        _run(
            apng_command(
                concat_path,
                output,
                loop=loop,
                ffmpeg=ffmpeg,
                duration_sec=duration_sec,
                frame_count=len(frame_paths),
            )
        )
    else:
        palette = frame_paths[0].parent / "palette.png"
        _run(
            gif_palette_command(
                concat_path,
                palette,
                ffmpeg=ffmpeg,
                duration_sec=duration_sec,
                frame_count=len(frame_paths),
            )
        )
        _run(
            gif_command(
                concat_path,
                palette,
                output,
                loop=loop,
                ffmpeg=ffmpeg,
                duration_sec=duration_sec,
                frame_count=len(frame_paths),
            )
        )


def _concat_prefix(ffmpeg: str, concat_path: Path) -> list[str]:
    """Open the concat list and restore each frame's real duration.

    FFmpeg 9's concat demuxer advances PTS correctly but leaves every packet
    duration at the first delay. ``setts`` copies the gap to the next PTS.
    """

    return [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-bsf:v",
        "setts=duration=if(eq(NEXT_PTS\\,NOPTS)\\,DURATION\\,NEXT_PTS-PTS)",
        "-i",
        str(concat_path),
    ]


def _drop_concat_sentinel(frame_count: int | None) -> str | None:
    """Drop the extra last file that exists only so the final delay is kept."""

    if frame_count is None:
        return None
    return f"select=lt(n\\,{frame_count}),setpts=PTS-STARTPTS"


def webm_command(
    concat_path: Path,
    output: Path,
    *,
    crf: int,
    pix_fmt: str,
    ffmpeg: str = "ffmpeg",
    duration_sec: float | None = None,
    frame_count: int | None = None,
) -> list[str]:
    if pix_fmt not in WEBM_PIX_FMTS:
        raise PipelineError(
            f"WebM pixel format must be one of {', '.join(WEBM_PIX_FMTS)}, got {pix_fmt!r}"
        )
    if not 0 <= crf <= 63:
        raise PipelineError(f"CRF must be between 0 and 63, got {crf}")
    command = _concat_prefix(ffmpeg, concat_path)
    sentinel = _drop_concat_sentinel(frame_count)
    if sentinel is not None:
        command.extend(["-vf", sentinel])
    command.extend(
        [
            "-fps_mode",
            "vfr",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            pix_fmt,
            "-auto-alt-ref",
            "0",
            "-lag-in-frames",
            "0",
            "-crf",
            str(crf),
            "-b:v",
            "0",
            "-an",
            "-row-mt",
            "1",
            "-video_track_timescale",
            "1000",
            "-metadata:s:v:0",
            "alpha_mode=1",
        ]
    )
    _append_duration_limit(command, duration_sec)
    command.append(str(output))
    return command


def apng_command(
    concat_path: Path,
    output: Path,
    *,
    loop: int | None,
    ffmpeg: str = "ffmpeg",
    duration_sec: float | None = None,
    frame_count: int | None = None,
) -> list[str]:
    command = _concat_prefix(ffmpeg, concat_path)
    sentinel = _drop_concat_sentinel(frame_count)
    if sentinel is not None:
        command.extend(["-vf", sentinel])
    command.extend(
        [
            "-fps_mode",
            "vfr",
            "-c:v",
            "apng",
            "-pix_fmt",
            "rgba",
            "-plays",
            str(_apng_plays(loop)),
            "-video_track_timescale",
            "1000",
        ]
    )
    _append_duration_limit(command, duration_sec)
    command.append(str(output))
    return command


def gif_palette_command(
    concat_path: Path,
    palette: Path,
    *,
    ffmpeg: str = "ffmpeg",
    duration_sec: float | None = None,
    frame_count: int | None = None,
) -> list[str]:
    palette_filter = "palettegen=reserve_transparent=1:stats_mode=diff"
    sentinel = _drop_concat_sentinel(frame_count)
    if sentinel is not None:
        palette_filter = f"{sentinel},{palette_filter}"
    command = _concat_prefix(ffmpeg, concat_path)
    command.extend(["-vf", palette_filter])
    _append_duration_limit(command, duration_sec)
    command.append(str(palette))
    return command


def gif_command(
    concat_path: Path,
    palette: Path,
    output: Path,
    *,
    loop: int | None,
    ffmpeg: str = "ffmpeg",
    duration_sec: float | None = None,
    frame_count: int | None = None,
) -> list[str]:
    # GIF has one transparent index, so semitransparent alpha cannot survive.
    # alpha_threshold is only for this preview file.
    sentinel = _drop_concat_sentinel(frame_count)
    if sentinel is None:
        graph = "[0:v][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle:alpha_threshold=128"
    else:
        graph = (
            f"[0:v]{sentinel}[src];"
            "[src][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle:alpha_threshold=128"
        )
    command = _concat_prefix(ffmpeg, concat_path)
    command.extend(
        [
            "-i",
            str(palette),
            "-filter_complex",
            graph,
            "-fps_mode",
            "vfr",
            "-loop",
            _gif_loop(loop),
        ]
    )
    _append_duration_limit(command, duration_sec)
    command.append(str(output))
    return command


def probe_video(path: Path) -> dict[str, object]:
    command = [
        require_binary("ffprobe"),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,duration,nb_read_frames:stream_tags=ALPHA_MODE",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = _run(command)
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise PipelineError(f"ffprobe did not return JSON for {path}") from exc
    streams = payload.get("streams") or []
    if not streams:
        raise PipelineError(f"ffprobe found no video stream in {path}")
    stream = streams[0]
    format_info = payload.get("format") or {}
    duration_text = stream.get("duration") or format_info.get("duration")
    if duration_text in (None, "", "N/A"):
        raise PipelineError(f"ffprobe did not report a duration for {path}")
    frame_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    parsed_count = int(frame_count) if frame_count not in (None, "", "N/A") else None
    tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
    pix_fmt = str(stream.get("pix_fmt") or "")
    # FFmpeg 9 ffprobe reports a VP9 alpha WebM as yuv420p plus ALPHA_MODE=1.
    if str(tags.get("ALPHA_MODE") or "") == "1" and pix_fmt in {"yuv420p", "yuv444p"}:
        pix_fmt = "yuva" + pix_fmt[3:]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pix_fmt": pix_fmt,
        "duration_sec": float(duration_text),
        "frame_count": parsed_count,
    }


def assert_encoded_stream(
    probed: dict[str, object],
    metadata: JobMetadata,
    *,
    fmt: str,
    webm_pix_fmt: str,
) -> None:
    expected = expected_encoded_duration_ms(metadata.frame_durations_ms) / 1000.0
    assert_duration_close(
        float(probed["duration_sec"]),
        expected,
        duration_tolerance_sec(metadata.frame_durations_ms),
    )
    if probed["width"] != metadata.target_width or probed["height"] != metadata.target_height:
        raise PipelineError(
            f"Encoded video is {probed['width']}x{probed['height']}, "
            f"expected {metadata.target_width}x{metadata.target_height}"
        )
    if probed["frame_count"] not in (None, metadata.frame_count):
        raise PipelineError(
            f"Encoded video has {probed['frame_count']} frames, expected {metadata.frame_count}"
        )
    pix_fmt = str(probed["pix_fmt"])
    if fmt == "webm" and not pix_fmt.startswith("yuva"):
        raise PipelineError(
            f"WebM pixel format is {pix_fmt or 'unknown'}, expected {webm_pix_fmt} with alpha. "
            "The alpha plane was dropped."
        )
    if fmt == "apng" and pix_fmt not in {"rgba", "yuva420p", "yuva444p", "pal8", "rgba64be"}:
        raise PipelineError(
            f"APNG pixel format is {pix_fmt or 'unknown'}, expected an alpha format"
        )


def _append_duration_limit(command: list[str], duration_sec: float | None) -> None:
    # Stops the concat demuxer's repeated last-file entry from extending the clip.
    if duration_sec is not None:
        command.extend(["-t", f"{duration_sec:.6f}"])


def _quote_concat_name(name: str) -> str:
    return "'" + name.replace("'", r"'\''") + "'"


def _apng_plays(loop: int | None) -> int:
    # APNG plays is the total play count. 0 means infinite.
    # A GIF loop count is the number of repeats after the first play.
    if loop is None:
        return 1
    if loop == 0:
        return 0
    return loop + 1


def _gif_loop(loop: int | None) -> str:
    # ffmpeg's GIF muxer uses 0 for infinite and -1 for play once.
    if loop is None:
        return "-1"
    if loop == 0:
        return "0"
    return str(loop)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        program = Path(command[0]).name
        detail = tail_text(completed.stderr)
        raise PipelineError(f"{program} failed with exit code {completed.returncode}.\n{detail}")
    return completed
