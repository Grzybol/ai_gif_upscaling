"""Command line for the Midnight Lounge upscaling pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from midnight_upscale import __version__
from midnight_upscale.alpha import ALPHA_MODES, EDGE_CLEANUP_CHOICES
from midnight_upscale.comfy import known_seedvr2_status
from midnight_upscale.inspect import inspect_gif
from midnight_upscale.pipeline import (
    FORMATS,
    finalize_job,
    prepare_asset,
    process_asset,
    upscale_job,
)
from midnight_upscale.seedvr2 import validate_seedvr2_batch_size, validate_temporal_overlap
from midnight_upscale.utils import PipelineError, configure_logging, load_config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "verbose", False))
    try:
        args.func(args)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="midnight-upscale",
        description=(
            "Decode a transparent character GIF, upscale RGB with SeedVR2, "
            "resize alpha separately, and encode a transparent WebM."
        ),
    )
    parser.add_argument("--version", action="version", version=f"midnight-upscale {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verbose", action="store_true", help="Log debug details")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser(
        "inspect", parents=[common], help="Report GIF size, timing, and transparency"
    )
    inspect.add_argument("source", type=Path)
    inspect.add_argument(
        "--json", action="store_true", dest="as_json", help="Print JSON instead of text"
    )
    inspect.set_defaults(func=_cmd_inspect)

    prepare = sub.add_parser(
        "prepare",
        parents=[common],
        help="Decode RGBA frames, split RGB and alpha, and upscale alpha",
    )
    _add_source(prepare)
    _add_prepare_flags(prepare)
    prepare.set_defaults(func=_cmd_prepare)

    upscale = sub.add_parser("upscale", parents=[common], help="Upscale the prepared RGB frames")
    upscale.add_argument("workdir", type=Path, help="Job directory created by prepare")
    _add_upscale_flags(upscale)
    upscale.set_defaults(func=_cmd_upscale)

    finalize = sub.add_parser(
        "finalize",
        parents=[common],
        help="Recombine RGB and alpha, then encode",
    )
    finalize.add_argument("workdir", type=Path)
    _add_encode_flags(finalize)
    finalize.set_defaults(func=_cmd_finalize)

    process = sub.add_parser("process", parents=[common], help="Run prepare, upscale, and finalize")
    _add_source(process)
    _add_prepare_flags(process, overwrite=False)
    _add_upscale_flags(process, overwrite=False, config=True)
    _add_encode_flags(process, overwrite=True, config=False)
    process.add_argument(
        "--keep-workdir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep intermediate files after a successful process (default: keep)",
    )
    process.set_defaults(func=_cmd_process)

    info = sub.add_parser(
        "comfy-info",
        parents=[common],
        help="Save ComfyUI /object_info for workflow mapping",
    )
    info.add_argument("--comfy-url", default=None)
    info.add_argument("--config", type=Path, default=None)
    info.add_argument("--output", type=Path, default=Path("comfy_object_info.json"))
    info.add_argument("--timeout", type=float, default=None)
    info.set_defaults(func=_cmd_comfy_info)

    check = sub.add_parser(
        "comfy-check",
        parents=[common],
        help="Check native SeedVR2 nodes, models, ffmpeg, and workflow",
    )
    check.add_argument("--comfy-url", default=None)
    check.add_argument("--config", type=Path, default=None)
    check.set_defaults(func=_cmd_comfy_check)
    return parser


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", type=Path, help="Input GIF")


def _add_prepare_flags(parser: argparse.ArgumentParser, *, overwrite: bool = True) -> None:
    parser.add_argument("--scale", type=_positive_int, required=True, help="Integer upscale factor")
    parser.add_argument("--workdir", type=Path, default=None, help="Defaults to work/<gif name>")
    parser.add_argument("--alpha-mode", choices=ALPHA_MODES, default="lanczos")
    parser.add_argument("--edge-cleanup", choices=EDGE_CLEANUP_CHOICES, default="auto")
    parser.add_argument("--interpolate", default="none", help="Only 'none' is supported")
    if overwrite:
        parser.add_argument("--overwrite", action="store_true")


def _add_upscale_flags(
    parser: argparse.ArgumentParser,
    *,
    overwrite: bool = True,
    config: bool = True,
) -> None:
    parser.add_argument(
        "--backend",
        choices=("auto", "seedvr2-native", "seedvr2", "frame-upscale"),
        default="auto",
        help=(
            "auto uses native ComfyUI SeedVR2 when those nodes are installed, "
            "otherwise the numz custom node. seedvr2 keeps the mapped workflow."
        ),
    )
    parser.add_argument("--comfy-url", default=None, help="Defaults to http://127.0.0.1:8188")
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=5,
        help="SeedVR2 temporal window. Must be 4n+1 and at least 5 (5, 9, 13, ...).",
    )
    parser.add_argument(
        "--temporal-overlap",
        type=_non_negative_int,
        default=1,
        help=(
            "SeedVR2 temporal overlap. Default 1. Sent only when the installed node "
            "exposes that input. Suggested: 1 with batch 5; 1 or 2 with batch 9."
        ),
    )
    parser.add_argument(
        "--temporal-mode",
        choices=("auto", "unchunked", "chunked"),
        default="auto",
        help=(
            "Native SeedVR2 only. auto keeps short clips unchunked. The numz workflow ignores this."
        ),
    )
    if config:
        parser.add_argument(
            "--config",
            type=Path,
            default=None,
            help="YAML config. Defaults to config.yaml if present",
        )
    parser.add_argument(
        "--timeout", type=float, default=None, help="Seconds to wait for one ComfyUI prompt"
    )
    if overwrite:
        parser.add_argument("--overwrite", action="store_true")


def _add_encode_flags(
    parser: argparse.ArgumentParser,
    *,
    overwrite: bool = True,
    config: bool = True,
) -> None:
    parser.add_argument("--format", choices=FORMATS, default="webm")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--crf", type=int, default=None, help="VP9 CRF, 0-63. Default 18")
    parser.add_argument("--webm-pix-fmt", choices=("yuva420p", "yuva444p"), default=None)
    if config:
        parser.add_argument("--config", type=Path, default=None)
    if overwrite:
        parser.add_argument("--overwrite", action="store_true")


def _cmd_inspect(args: argparse.Namespace) -> None:
    info = inspect_gif(args.source)
    if args.as_json:
        print(json.dumps(info.to_dict(), indent=2))
    else:
        print(info.format_report(), end="")


def _cmd_prepare(args: argparse.Namespace) -> None:
    workdir = prepare_asset(
        args.source,
        scale=args.scale,
        workdir=args.workdir,
        alpha_mode=args.alpha_mode,
        edge_cleanup=args.edge_cleanup,
        interpolate=args.interpolate,
        overwrite=args.overwrite,
    )
    print(workdir)


def _cmd_upscale(args: argparse.Namespace) -> None:
    _reject_invalid_seedvr2_controls(args)
    upscale_job(
        args.workdir,
        backend=args.backend,
        comfy_url=args.comfy_url,
        batch_size=args.batch_size,
        config_path=args.config,
        timeout_sec=args.timeout,
        overwrite=args.overwrite,
        temporal_overlap=args.temporal_overlap,
        temporal_mode=args.temporal_mode,
    )


def _cmd_finalize(args: argparse.Namespace) -> None:
    finalize_job(
        args.workdir,
        fmt=args.format,
        output=args.output,
        overwrite=args.overwrite,
        crf=args.crf,
        webm_pix_fmt=args.webm_pix_fmt,
        config_path=args.config,
    )


def _cmd_process(args: argparse.Namespace) -> None:
    _reject_invalid_seedvr2_controls(args)
    process_asset(
        args.source,
        scale=args.scale,
        backend=args.backend,
        comfy_url=args.comfy_url,
        batch_size=args.batch_size,
        fmt=args.format,
        output=args.output,
        alpha_mode=args.alpha_mode,
        edge_cleanup=args.edge_cleanup,
        interpolate=args.interpolate,
        workdir=args.workdir,
        config_path=args.config,
        timeout_sec=args.timeout,
        overwrite=args.overwrite,
        keep_workdir=args.keep_workdir,
        crf=args.crf,
        webm_pix_fmt=args.webm_pix_fmt,
        temporal_overlap=args.temporal_overlap,
        temporal_mode=args.temporal_mode,
    )


def _cmd_comfy_check(args: argparse.Namespace) -> None:
    from midnight_upscale.seedvr2_native import run_comfy_check

    text, ready = run_comfy_check(args.comfy_url, args.config)
    print(text, flush=True)
    if not ready:
        raise PipelineError("SeedVR2 native is not ready.")


def _cmd_comfy_info(args: argparse.Namespace) -> None:
    from midnight_upscale.comfy import ComfyClient

    config, _loaded = load_config(args.config)
    comfy_cfg = config["comfyui"]
    client = ComfyClient(
        args.comfy_url or str(comfy_cfg["url"]),
        timeout_sec=float(args.timeout if args.timeout is not None else comfy_cfg["timeout_sec"]),
        poll_interval_sec=float(comfy_cfg["poll_interval_sec"]),
    )
    try:
        info = client.object_info()
    finally:
        client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    names = sorted(info)
    print(f"Wrote {len(names)} nodes to {args.output}")
    print("SeedVR2 class types this tool knows how to look for:")
    for line in known_seedvr2_status(info):
        print(f"  {line}")
    interesting = [
        name
        for name in names
        if any(token in name.lower() for token in ("seedvr", "loadimage", "saveimage", "vhs"))
    ]
    print("Matching node class types:")
    for name in interesting[:80]:
        print(f"  {name}")
    if len(interesting) > 80:
        print(f"  ... {len(interesting) - 80} more in {args.output}")


def _reject_invalid_seedvr2_controls(args: argparse.Namespace) -> None:
    if args.backend != "seedvr2":
        return
    validate_seedvr2_batch_size(args.batch_size)
    validate_temporal_overlap(args.temporal_overlap, args.batch_size)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
