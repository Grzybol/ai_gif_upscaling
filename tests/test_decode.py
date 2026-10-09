from pathlib import Path

from PIL import Image

from midnight_upscale.decode import iter_composited_frames
from midnight_upscale.interpolate import resolve_interpolation
from midnight_upscale.utils import PipelineError
from tests.conftest import save_rgba_gif


def _disposal_frames() -> list[Image.Image]:
    first = Image.new("RGBA", (6, 6), (0, 0, 0, 0))
    first.putpixel((0, 0), (255, 0, 0, 255))
    second = Image.new("RGBA", (6, 6), (0, 0, 0, 0))
    second.putpixel((5, 5), (0, 255, 0, 255))
    third = second.copy()
    third.putpixel((2, 2), (0, 0, 255, 255))
    return [first, second, third]


def test_partial_frames_and_disposal_are_composited(tmp_path: Path) -> None:
    path = tmp_path / "disposal.gif"
    save_rgba_gif(path, _disposal_frames(), [40, 80, 120], disposal=[2, 1, 1])
    decoded = list(iter_composited_frames(path))
    assert [frame.duration_ms for frame in decoded] == [40, 80, 120]
    assert len(decoded) == 3

    # Frame 0 shows the red pixel on a transparent canvas.
    assert decoded[0].image.getpixel((0, 0)) == (255, 0, 0, 255)
    assert decoded[0].image.getpixel((5, 5))[3] == 0

    # Disposal 2 clears that red pixel before frame 1 is shown.
    assert decoded[1].image.getpixel((0, 0))[3] == 0
    assert decoded[1].image.getpixel((5, 5)) == (0, 255, 0, 255)

    # Frame 2 is a partial update: the green pixel remains and blue is added.
    assert decoded[2].image.getpixel((0, 0))[3] == 0
    assert decoded[2].image.getpixel((5, 5)) == (0, 255, 0, 255)
    assert decoded[2].image.getpixel((2, 2)) == (0, 0, 255, 255)
    for frame in decoded:
        assert frame.image.size == (6, 6)
        assert frame.image.mode == "RGBA"


def test_do_not_dispose_keeps_earlier_pixels(tmp_path: Path) -> None:
    path = tmp_path / "keep.gif"
    frames = _disposal_frames()[:2]
    # Second frame also contains the first frame's red pixel, plus green.
    frames[1].putpixel((0, 0), (255, 0, 0, 255))
    save_rgba_gif(path, frames, [30, 70], disposal=1)
    decoded = list(iter_composited_frames(path))
    assert decoded[1].image.getpixel((0, 0)) == (255, 0, 0, 255)
    assert decoded[1].image.getpixel((5, 5)) == (0, 255, 0, 255)
    assert decoded[1].image.getpixel((3, 3))[3] == 0


def test_non_gif_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "still.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 0)).save(path)
    try:
        list(iter_composited_frames(path))
    except PipelineError as exc:
        assert "expected an animated GIF" in str(exc)
    else:
        raise AssertionError("png was decoded as a gif")


def test_interpolate_none_is_the_only_mode() -> None:
    assert resolve_interpolation("none") == "none"
    try:
        resolve_interpolation("rife")
    except PipelineError as exc:
        assert "none" in str(exc)
    else:
        raise AssertionError("rife was accepted")
