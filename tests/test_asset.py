import numpy as np
import pytest
from PIL import Image

from midnight_upscale.white_background import remove_white_background
from midnight_upscale.gui_asset import export_asset, prepare_asset, preview_asset


def test_asset_risk_uses_converter_and_excludes_invisible_white(tmp_path):
    pixels = np.array(
        [[[255, 255, 255, 255], [255, 255, 255, 0], [230, 230, 230, 255], [20, 30, 40, 128]]],
        dtype=np.uint8,
    )
    source = tmp_path / "asset.png"
    Image.fromarray(pixels).save(source)
    adjusted, marked, removed, report = prepare_asset(str(source), 0)
    np.testing.assert_array_equal(adjusted, pixels)
    np.testing.assert_array_equal(removed, remove_white_background(pixels))
    assert "**1 → 1**" in report
    assert "Fully removed (pink): **1**" in report
    assert "Partially transparent (orange): **1**" in report
    assert marked[0, 0].tolist() == [255, 0, 100, 255]
    assert marked[0, 2].tolist() == [255, 170, 0, 255]


def test_warmth_protects_white_and_export_preserves_alpha(tmp_path):
    source = tmp_path / "asset.png"
    pixels = np.full((16, 16, 4), 255, dtype=np.uint8)
    pixels[0, 0, 3] = 0
    pixels[1, 1, 3] = 128
    Image.fromarray(pixels).save(source)
    adjusted, _, removed, report = prepare_asset(str(source), 100)
    np.testing.assert_array_equal(adjusted[:, :, 3], pixels[:, :, 3])
    np.testing.assert_array_equal(removed[:, :, 3], pixels[:, :, 3])
    assert "**255 → 0**" in report
    assert "No visible pixels affected" in report
    exported = export_asset(str(source), 100)
    with Image.open(exported) as image:
        np.testing.assert_array_equal(np.array(image), adjusted)
    # A subsequent reset reads the upload, not the previous warmed result.
    np.testing.assert_array_equal(prepare_asset(str(source), 0)[0], pixels)


def test_empty_and_invalid_upload(tmp_path):
    assert preview_asset(None, 0)[:3] == (None, None, None)
    source = tmp_path / "fake.png"
    Image.new("RGB", (2, 2)).save(source, format="JPEG")
    with pytest.raises(ValueError, match="single-frame PNG"):
        prepare_asset(str(source), 0)


def test_asset_prep_remains_in_main_gui():
    pytest.importorskip("gradio")
    from midnight_upscale.gui import build_demo

    demo = build_demo()
    labels = [component.get("props", {}).get("label") for component in demo.config["components"]]
    assert "PNG ASSET PREP" in labels
