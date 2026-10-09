"""Inspect an animation source (MP4, MOV, WebM, GIF) with ffprobe.

Frame timing comes from the packet timestamps, so a variable frame rate source
keeps one real duration per frame. Nothing here assumes a constant FPS.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from midnight_upscale.utils import PipelineError, require_binary, tail_text

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".gif")

# Pixel formats that carry an alpha plane.
ALPHA_PIX_FMTS = frozenset(
    {
        "rgba",
        "bgra",
        "argb",
        "abgr",
        "ya8",
        "ya16le",
        "ya16be",
        "rgba64le",
        "rgba64be",
        "bgra64le",
        "bgra64be",
        "pal8a",
    }
)
_ALPHA_PREFIXES = ("yuva", "gbrap")

# Decoders that read the alpha plane of a codec whose native decoder drops it.
ALPHA_DECODERS = {"vp9": "libvpx-vp9", "vp8": "libvpx"}

# Duration jitter from rounding 1/29.97 s style timestamps to whole milliseconds.
TIMING_JITTER_MS = 2


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    codec: str
    pix_fmt: str
    container: str
    duration_sec: float
    fps: float
    frame_count: int
    frame_times: list[float] = field(default_factory=list)
    durations_ms: list[int] = field(default_factory=list)
    variable_timing: bool = False
    alpha_declared: bool = False
    has_alpha: bool = False

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def is_gif(self) -> bool:
        return self.path.suffix.lower() == ".gif"

    @property
    def total_duration_ms(self) -> int:
        return sum(self.durations_ms)


def pix_fmt_has_alpha(pix_fmt: str) -> bool:
    return pix_fmt in ALPHA_PIX_FMTS or pix_fmt.startswith(_ALPHA_PREFIXES)


def ffprobe_json(path: Path) -> dict[str, Any]:
    command = [
        require_binary("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,duration"
        ":stream_tags=ALPHA_MODE"
        ":format=format_name,duration"
        ":packet=pts_time,duration_time",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise PipelineError(
            f"ffprobe could not read {path.name}.\n{tail_text(completed.stderr or '')}"
        )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise PipelineError(f"ffprobe did not return JSON for {path.name}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"ffprobe returned unexpected output for {path.name}")
    return payload


def parse_rate(text: object) -> float:
    """Parse ``30000/1001`` or ``25``. Returns 0.0 when unknown."""

    if not isinstance(text, str) or not text or text == "N/A":
        return 0.0
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            den = float(denominator)
            return 0.0 if den == 0 else float(numerator) / den
        return float(text)
    except ValueError:
        return 0.0


def _as_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def durations_from_times(times: list[float], last_duration: float | None) -> list[int]:
    """Per-frame durations in whole milliseconds that add up to the real span.

    Each boundary is rounded once, so rounding error never accumulates.
    """

    if not times:
        return []
    boundaries = list(times)
    tail = last_duration
    if tail is None or tail <= 0:
        gaps = [b - a for a, b in zip(times, times[1:], strict=False) if b > a]
        tail = statistics.median(gaps) if gaps else 0.0
    boundaries.append(times[-1] + tail)
    origin = times[0]
    rounded = [round((value - origin) * 1000.0) for value in boundaries]
    return [max(0, b - a) for a, b in zip(rounded, rounded[1:], strict=False)]


def parse_probe(payload: dict[str, Any], path: Path) -> VideoInfo:
    """Build a ``VideoInfo`` from ffprobe JSON. ``has_alpha`` still needs a pixel check."""

    streams = payload.get("streams") or []
    if not streams:
        raise PipelineError(f"{path.name} has no video stream")
    stream = streams[0]
    format_info = payload.get("format") or {}
    tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise PipelineError(f"{path.name} reports no frame size")
    pix_fmt = str(stream.get("pix_fmt") or "")
    codec = str(stream.get("codec_name") or "")

    packets = payload.get("packets") or []
    packet_times: list[tuple[float, float | None]] = []
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        when = _as_float(packet.get("pts_time"))
        if when is None:
            continue
        packet_times.append((when, _as_float(packet.get("duration_time"))))
    packet_times.sort(key=lambda item: item[0])

    nb_frames = _as_float(stream.get("nb_frames"))
    declared_rate = parse_rate(stream.get("avg_frame_rate")) or parse_rate(
        stream.get("r_frame_rate")
    )
    stream_duration = _as_float(stream.get("duration")) or _as_float(format_info.get("duration"))

    if packet_times:
        times = [item[0] for item in packet_times]
        durations = durations_from_times(times, packet_times[-1][1])
        origin = times[0]
        frame_times = [round(t - origin, 6) for t in times]
    else:
        count = int(nb_frames) if nb_frames else 0
        if count <= 0 and stream_duration and declared_rate:
            count = max(1, round(stream_duration * declared_rate))
        if count <= 0:
            raise PipelineError(f"{path.name} reports no frame timing")
        rate = declared_rate or (count / stream_duration if stream_duration else 0.0)
        if rate <= 0:
            raise PipelineError(f"{path.name} reports no frame rate")
        step = 1.0 / rate
        frame_times = [round(i * step, 6) for i in range(count)]
        durations = durations_from_times(frame_times, step)

    total_ms = sum(durations)
    frame_count = len(frame_times)
    fps = frame_count / (total_ms / 1000.0) if total_ms > 0 else declared_rate
    variable = bool(durations) and (max(durations) - min(durations) > TIMING_JITTER_MS)

    # FFmpeg reports a VP9/VP8 alpha WebM as yuv420p plus an ALPHA_MODE tag.
    tagged_alpha = str(tags.get("ALPHA_MODE") or "") == "1"
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        codec=codec,
        pix_fmt=pix_fmt,
        container=str(format_info.get("format_name") or ""),
        duration_sec=total_ms / 1000.0,
        fps=fps,
        frame_count=frame_count,
        frame_times=frame_times,
        durations_ms=durations,
        variable_timing=variable,
        alpha_declared=tagged_alpha or pix_fmt_has_alpha(pix_fmt),
        has_alpha=False,
    )


def decoder_args(info: VideoInfo) -> list[str]:
    """Input options that make FFmpeg read the alpha plane of a WebM source."""

    decoder = ALPHA_DECODERS.get(info.codec)
    if decoder and info.alpha_declared and info.pix_fmt.startswith("yuv"):
        return ["-c:v", decoder]
    return []


def inspect_video(
    path: Path, *, check_alpha: bool = True, any_extension: bool = False
) -> VideoInfo:
    """Probe ``path``. When the format can hold alpha, sample frames to see whether it does.

    ``any_extension`` lets the exporter probe its own APNG output.
    """

    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"Input file not found: {path}")
    if not any_extension and path.suffix.lower() not in VIDEO_EXTENSIONS:
        allowed = ", ".join(VIDEO_EXTENSIONS)
        raise PipelineError(f"{path.name} is not a supported input ({allowed})")
    info = parse_probe(ffprobe_json(path), path)
    if info.alpha_declared and check_alpha:
        from midnight_upscale.video_decode import sample_has_alpha

        info.has_alpha = sample_has_alpha(info)
    return info


def format_video_card(info: VideoInfo) -> str:
    timing = "variable" if info.variable_timing else "constant"
    lines = [
        "Source",
        "------",
        f"Filename:     {info.filename}",
        f"Resolution:   {info.width} x {info.height}",
        f"Duration:     {info.duration_sec:.3f} s",
        f"FPS:          {info.fps:.3f} ({timing} frame timing)",
        f"Frame count:  {info.frame_count}",
        f"Codec:        {info.codec or 'unknown'}",
        f"Pixel format: {info.pix_fmt or 'unknown'}",
        f"Has alpha:    {'YES' if info.has_alpha else 'NO'}",
    ]
    if info.alpha_declared and not info.has_alpha:
        lines.append("              (alpha plane is present but every sampled pixel is opaque)")
    return "\n".join(lines)
