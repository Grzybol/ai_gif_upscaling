from pathlib import Path

import numpy as np
from PIL import Image

from midnight_upscale.alpha import (
    edge_cleanup_simple,
    recombine_rgba,
    split_rgba,
    upscale_alpha,
)
from midnight_upscale.models import JobMetadata
from midnight_upscale.pipeline import prepare_asset
from midnight_upscale.utils import PipelineError
from tests.conftest import character_frame, save_rgba_gif, solid_frame


def test_split_and_recombine_keep_straight_rgb_and_partial_alpha() -> None:
    original = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    original.putpixel((0, 0), (255, 0, 0, 128))
    original.putpixel((1, 0), (1, 2, 3, 0))
    original.putpixel((0, 1), (9, 8, 7, 255))

    rgb, alpha = split_rgba(original)
    assert rgb.mode == "RGB"
    assert alpha.mode == "L"
    assert rgb.getpixel((0, 0)) == (255, 0, 0)
    assert alpha.getpixel((0, 0)) == 128
    # The transparent pixel keeps its RGB. It is not flattened onto black or white.
    assert rgb.getpixel((1, 0)) == (1, 2, 3)
    assert alpha.getpixel((1, 0)) == 0

    merged = recombine_rgba(rgb, alpha)
    assert merged.mode == "RGBA"
    assert np.array_equal(np.asarray(merged), np.asarray(original))


def test_recombine_rejects_different_sizes() -> None:
    rgb = Image.new("RGB", (2, 2), (1, 2, 3))
    alpha = Image.new("L", (3, 2), 128)
    try:
        recombine_rgba(rgb, alpha)
    except PipelineError as exc:
        assert "Cannot recombine" in str(exc)
    else:
        raise AssertionError("size mismatch was accepted")


def test_alpha_resize_dimensions_and_no_threshold() -> None:
    mask = Image.new("L", (8, 8), 0)
    for x in range(4, 8):
        for y in range(8):
            mask.putpixel((x, y), 255)
    mask.putpixel((1, 1), 128)

    lanczos = upscale_alpha(mask, 2, "lanczos")
    bicubic = upscale_alpha(mask, 2, "bicubic")
    nearest = upscale_alpha(mask, 2, "nearest")
    assert lanczos.size == bicubic.size == nearest.size == (16, 16)
    assert nearest.mode == "L"

    lanczos_values = set(np.asarray(lanczos).reshape(-1).tolist())
    nearest_values = set(np.asarray(nearest).reshape(-1).tolist())
    assert any(0 < value < 255 for value in lanczos_values)
    assert nearest_values <= {0, 128, 255}
    assert 128 in nearest_values


def test_edge_cleanup_recolors_only_invisible_neighbors() -> None:
    image = Image.new("RGBA", (9, 9), (0, 255, 0, 0))
    image.putpixel((4, 4), (255, 0, 0, 255))
    image.putpixel((4, 5), (0, 0, 255, 0))
    image.putpixel((5, 4), (0, 0, 255, 10))

    cleaned = edge_cleanup_simple(image)
    assert np.array_equal(np.asarray(cleaned)[:, :, 3], np.asarray(image)[:, :, 3])
    assert cleaned.getpixel((4, 4)) == (255, 0, 0, 255)
    assert cleaned.getpixel((4, 5)) == (255, 0, 0, 0)
    assert cleaned.getpixel((5, 4)) == (255, 0, 0, 10)
    assert cleaned.getpixel((0, 0)) == (0, 255, 0, 0)


def test_prepare_writes_unthresholded_alpha(tmp_path: Path) -> None:
    source = tmp_path / "hero.gif"
    save_rgba_gif(source, [character_frame(0), character_frame(1, mark=(2, 2))], [40, 40])
    workdir = prepare_asset(
        source,
        scale=2,
        workdir=tmp_path / "job",
        alpha_mode="nearest",
        edge_cleanup="off",
        interpolate="none",
        overwrite=False,
    )
    with (
        Image.open(workdir / "alpha" / "000000.png") as alpha,
        Image.open(workdir / "rgb" / "000000.png") as rgb,
        Image.open(workdir / "upscaled_alpha" / "000000.png") as upscaled,
    ):
        assert alpha.mode == "L"
        assert rgb.mode == "RGB"
        assert alpha.getpixel((7, 7)) == 0
        assert alpha.getpixel((0, 0)) == 255
        assert rgb.getpixel((0, 0)) == (255, 0, 0)
        assert upscaled.size == (16, 16)
        assert upscaled.getpixel((14, 14)) == 0
        assert upscaled.getpixel((0, 0)) == 255


def test_edge_cleanup_auto_uses_simple_only_when_transparent(tmp_path: Path) -> None:
    transparent = tmp_path / "transparent.gif"
    save_rgba_gif(
        transparent,
        [character_frame(0), character_frame(1, mark=(2, 2))],
        [40, 40],
    )
    transparent_job = prepare_asset(
        transparent,
        scale=2,
        workdir=tmp_path / "transparent-job",
        alpha_mode="nearest",
        edge_cleanup="auto",
        interpolate="none",
        overwrite=False,
    )
    assert JobMetadata.load(transparent_job / "metadata.json").edge_cleanup == "simple"

    opaque = tmp_path / "opaque.gif"
    save_rgba_gif(
        opaque,
        [solid_frame((8, 8), (255, 0, 0, 255)), solid_frame((8, 8), (0, 255, 0, 255))],
        [40, 40],
    )
    opaque_job = prepare_asset(
        opaque,
        scale=2,
        workdir=tmp_path / "opaque-job",
        alpha_mode="nearest",
        edge_cleanup="auto",
        interpolate="none",
        overwrite=False,
    )
    assert JobMetadata.load(opaque_job / "metadata.json").edge_cleanup == "off"


def test_red_object_on_transparent_black_keeps_a_red_soft_edge() -> None:
    canvas = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 22):
        for x in range(10, 22):
            canvas.putpixel((x, y), (255, 0, 0, 255))

    cleaned = edge_cleanup_simple(canvas)
    source = np.asarray(canvas)
    prepared = np.asarray(cleaned)
    assert np.array_equal(prepared[:, :, 3], source[:, :, 3])
    assert tuple(prepared[0, 0]) == (0, 0, 0, 0)
    assert tuple(prepared[16, 16]) == (255, 0, 0, 255)
    changed = np.any(prepared[:, :, :3] != source[:, :, :3], axis=2)
    assert changed.any()
    assert int(changed.sum()) < 400
    assert not np.any(changed & (source[:, :, 3] > 16))

    def recombined(image: Image.Image) -> np.ndarray:
        rgb, alpha = split_rgba(image)
        upscaled_rgb = rgb.resize((128, 128), Image.Resampling.LANCZOS)
        upscaled_alpha = upscale_alpha(alpha, 4, "lanczos")
        return np.asarray(recombine_rgba(upscaled_rgb, upscaled_alpha))

    merged = recombined(cleaned)
    raw = recombined(canvas)
    soft = (merged[:, :, 3] > 16) & (merged[:, :, 3] < 240)
    raw_soft = (raw[:, :, 3] > 16) & (raw[:, :, 3] < 240)
    assert soft.any()
    assert raw_soft.any()
    soft_rgb = merged[soft][:, :3].astype(np.int16)
    raw_rgb = raw[raw_soft][:, :3].astype(np.int16)
    assert int(soft_rgb[:, 0].mean()) > 180
    assert int(soft_rgb[:, 1].mean()) < 40
    assert int(soft_rgb[:, 2].mean()) < 40
    assert int(soft_rgb[:, 0].mean()) > int(raw_rgb[:, 0].mean()) + 40
