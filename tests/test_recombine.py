from pathlib import Path

import pytest
from PIL import Image

from midnight_upscale.recombine import recombine_directories
from midnight_upscale.utils import ValidationError, frame_path, list_indexed_frames
from midnight_upscale.validate import assert_alpha_preserved, assert_frame_count


def _write_pair(
    directory: Path, index: int, rgb: tuple[int, int, int], alpha: int, size: int = 2
) -> None:
    rgb_dir = directory / "rgb"
    alpha_dir = directory / "alpha"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), rgb).save(frame_path(rgb_dir, index))
    Image.new("L", (size, size), alpha).save(frame_path(alpha_dir, index))


def test_recombine_keeps_order_count_and_partial_alpha(tmp_path: Path) -> None:
    _write_pair(tmp_path, 0, (255, 0, 0), 128)
    _write_pair(tmp_path, 1, (0, 255, 0), 0)
    written = recombine_directories(tmp_path / "rgb", tmp_path / "alpha", tmp_path / "final")
    assert [path.name for path in written] == ["000000.png", "000001.png"]
    with Image.open(written[0]) as first, Image.open(written[1]) as second:
        assert first.getpixel((0, 0)) == (255, 0, 0, 128)
        assert second.getpixel((0, 0)) == (0, 255, 0, 0)
        assert first.mode == "RGBA"


def test_recombine_aborts_when_a_frame_is_missing(tmp_path: Path) -> None:
    _write_pair(tmp_path, 0, (1, 2, 3), 255)
    _write_pair(tmp_path, 1, (4, 5, 6), 255)
    (tmp_path / "alpha" / "000001.png").unlink()
    with pytest.raises(ValidationError, match="Refusing to drop"):
        recombine_directories(tmp_path / "rgb", tmp_path / "alpha", tmp_path / "final")


def test_recombine_aborts_on_size_mismatch(tmp_path: Path) -> None:
    _write_pair(tmp_path, 0, (1, 2, 3), 200, size=2)
    Image.new("L", (4, 4), 200).save(tmp_path / "alpha" / "000000.png")
    with pytest.raises(ValidationError, match="RGB"):
        recombine_directories(tmp_path / "rgb", tmp_path / "alpha", tmp_path / "final")


def test_frame_count_validation_does_not_fill_gaps(tmp_path: Path) -> None:
    directory = tmp_path / "frames"
    directory.mkdir()
    Image.new("RGBA", (1, 1)).save(frame_path(directory, 0))
    Image.new("RGBA", (1, 1)).save(frame_path(directory, 2))
    with pytest.raises(ValidationError, match="not contiguous"):
        list_indexed_frames(directory)
    Image.new("RGBA", (1, 1)).save(frame_path(directory, 1))
    (tmp_path / "frames" / "000002.png").unlink()
    frames = assert_frame_count(directory, 2, "frames")
    assert len(frames) == 2
    with pytest.raises(ValidationError, match="expected 3"):
        assert_frame_count(directory, 3, "frames")


def test_semitransparent_alpha_check_catches_a_flattened_mask() -> None:
    source = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
    flattened = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
    with pytest.raises(ValidationError, match="semitransparent"):
        assert_alpha_preserved(source, flattened, "frame 000000")

    binary = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
    with pytest.raises(ValidationError, match="semitransparent"):
        assert_alpha_preserved(source, binary, "frame 000000")

    kept = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
    assert_alpha_preserved(source, kept, "frame 000000")
