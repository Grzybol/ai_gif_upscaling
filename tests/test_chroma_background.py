import numpy as np
import pytest

from midnight_upscale import segmentation
from midnight_upscale.background import (
    BackgroundSettings,
    remove_background,
    resolve_background,
)
from midnight_upscale.chroma import (
    ChromaSettings,
    chroma_alpha,
    chroma_key,
    estimate_key_color,
    looks_chroma_green,
    parse_hex_color,
)
from midnight_upscale.mask_temporal import get_level, smooth_frame, smooth_masks
from midnight_upscale.segmentation import (
    BackgroundRemoverUnavailable,
    ai_available,
    create_remover,
    register_remover,
)
from midnight_upscale.utils import PipelineError
from midnight_upscale.video_transform import (
    ResizeSettings,
    alpha_bbox,
    apply_crop,
    fill_transparent_rgb,
    plan_crop,
    resize_rgba,
    target_size,
    union_box,
)
from tests.video_fixtures import GREEN, green_screen_frame


def _rgba(frame: np.ndarray) -> np.ndarray:
    out = np.full(frame.shape[:2] + (4,), 255, dtype=np.uint8)
    out[..., :3] = frame
    return out


def test_parse_hex_color() -> None:
    assert parse_hex_color("#00ff00") == (0, 255, 0)
    assert parse_hex_color("0f0") == (0, 255, 0)
    assert parse_hex_color("rgb(0, 177, 64)") == (0, 177, 64)
    with pytest.raises(ValueError):
        parse_hex_color("#12")


def test_chroma_key_removes_green_and_keeps_object() -> None:
    frame = green_screen_frame(0)
    keyed = chroma_key(frame, ChromaSettings(key_color=GREEN))
    assert keyed.shape == (48, 64, 4)
    assert keyed[0, 0, 3] == 0  # background corner
    assert keyed[20, 6, 3] == 255  # inside the red square
    assert tuple(keyed[20, 6, :3]) == (220, 20, 20)


def test_chroma_key_leaves_no_green_visible() -> None:
    keyed = chroma_key(green_screen_frame(3), ChromaSettings(key_color=GREEN))
    visible = keyed[keyed[..., 3] > 8]
    greenish = (visible[:, 1].astype(int) - np.maximum(visible[:, 0], visible[:, 2])) > 20
    assert not greenish.any()


def test_chroma_tolerance_decides_what_is_removed() -> None:
    # A green that is close to, but not exactly, the key color.
    frame = np.full((8, 8, 3), (40, 200, 60), dtype=np.uint8)
    strict = chroma_alpha(frame, ChromaSettings(key_color=(0, 255, 0), tolerance=0.02, softness=0))
    loose = chroma_alpha(frame, ChromaSettings(key_color=(0, 255, 0), tolerance=0.6, softness=0))
    assert strict.min() == 1.0
    assert loose.max() == 0.0


def test_chroma_alpha_is_feathered_not_binary() -> None:
    ramp = np.zeros((1, 256, 3), dtype=np.uint8)
    ramp[0, :, 1] = np.linspace(255, 0, 256).astype(np.uint8)  # green fading to black
    ramp[0, :, 0] = np.linspace(0, 200, 256).astype(np.uint8)
    soft = chroma_alpha(ramp, ChromaSettings(key_color=(0, 255, 0), softness=0.4))
    middle = soft[(soft > 0.05) & (soft < 0.95)]
    assert middle.size > 10
    hard = chroma_key(ramp, ChromaSettings(key_color=(0, 255, 0), softness=0.4, hard_mask=True))
    assert set(np.unique(hard[..., 3])) <= {0, 255}


def test_despill_pulls_green_out_of_foreground() -> None:
    frame = np.full((6, 6, 3), (200, 150, 120), dtype=np.uint8)
    frame[:, :3] = (120, 190, 110)  # skin with green spill
    off = chroma_key(frame, ChromaSettings(key_color=GREEN, tolerance=0.0, softness=0, spill=0.0))
    on = chroma_key(frame, ChromaSettings(key_color=GREEN, tolerance=0.0, softness=0, spill=1.0))
    assert on[0, 0, 1] < off[0, 0, 1]
    assert on[0, 0, 1] <= max(on[0, 0, 0], on[0, 0, 2])


def test_estimate_key_color_and_green_detection() -> None:
    color, coverage = estimate_key_color([green_screen_frame(0), green_screen_frame(5)])
    assert color == GREEN
    assert coverage > 0.95
    assert looks_chroma_green(color, coverage)
    assert not looks_chroma_green((120, 120, 120), 1.0)
    noisy = np.random.default_rng(1).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    assert not looks_chroma_green(*estimate_key_color([noisy]))


def test_auto_preserves_alpha_before_anything_else() -> None:
    frame = _rgba(green_screen_frame(0))
    frame[0:4, 0:4, 3] = 0
    resolved = resolve_background(
        BackgroundSettings(), has_alpha=True, sample_frames=[green_screen_frame(0)]
    )
    assert resolved.label == "AUTO -> PRESERVE ALPHA"
    out = remove_background(resolved, frame)
    assert np.array_equal(out, frame)


def test_auto_resolves_to_chroma_key_for_green_border() -> None:
    resolved = resolve_background(
        BackgroundSettings(), has_alpha=False, sample_frames=[green_screen_frame(0)]
    )
    assert resolved.label == "AUTO -> CHROMA KEY"
    assert resolved.chroma is not None and resolved.chroma.key_color == GREEN


def test_auto_without_ai_installed_fails_with_install_help(monkeypatch) -> None:
    monkeypatch.setitem(segmentation._AVAILABILITY, "rembg", lambda: False)
    gray = np.full((32, 32, 3), 128, dtype=np.uint8)
    with pytest.raises(BackgroundRemoverUnavailable) as caught:
        resolve_background(BackgroundSettings(), has_alpha=False, sample_frames=[gray])
    assert "AI background removal is not installed." in str(caught.value)
    assert 'pip install -e ".[bgremove]"' in str(caught.value)


class FakeRemover:
    name = "fake"

    def __init__(self, model: str) -> None:
        self.model = model
        self.closed = False

    def alpha(self, rgb: np.ndarray) -> np.ndarray:
        return np.where(rgb[..., 0] > 150, 255, 0).astype(np.uint8)

    def close(self) -> None:
        self.closed = True


def test_ai_backend_is_pluggable_and_mocked() -> None:
    register_remover("fake", FakeRemover)
    assert ai_available("fake")
    settings = BackgroundSettings(mode="ai", ai_backend="fake", ai_model="tiny")
    resolved = resolve_background(settings, has_alpha=False, sample_frames=[])
    assert resolved.label == "AI SEGMENTATION"
    remover = create_remover("fake", "tiny")
    frame = _rgba(green_screen_frame(0))
    out = remove_background(resolved, frame, remover)
    assert out[20, 6, 3] == 255 and out[0, 0, 3] == 0
    assert np.array_equal(out[..., :3], frame[..., :3])  # RGB is never altered


def test_unknown_ai_backend_is_an_error() -> None:
    with pytest.raises(PipelineError):
        create_remover("does-not-exist")


def test_manual_mode_keeps_existing_transparency() -> None:
    frame = _rgba(green_screen_frame(0))
    frame[20, 6, 3] = 100  # a half-transparent pixel inside the object
    resolved = resolve_background(
        BackgroundSettings(mode="chroma"), has_alpha=True, sample_frames=[]
    )
    out = remove_background(resolved, frame)
    assert out[20, 6, 3] == 100


def test_temporal_smoothing_reduces_flicker_without_changing_shape() -> None:
    steady = np.full((4, 4), 200, dtype=np.uint8)
    flicker = steady.copy()
    flicker[1, 1] = 120
    window = [steady, flicker, steady]
    smoothed = smooth_frame(window, 1, get_level("low"))
    assert smoothed.shape == flicker.shape and smoothed.dtype == np.uint8
    assert smoothed[1, 1] > flicker[1, 1]
    assert smoothed[0, 0] == 200
    assert np.array_equal(smooth_frame(window, 1, get_level("off")), flicker)


def test_temporal_smoothing_does_not_ghost_real_motion() -> None:
    left = np.zeros((4, 8), dtype=np.uint8)
    left[:, :2] = 255
    right = np.zeros((4, 8), dtype=np.uint8)
    right[:, 6:] = 255
    out = smooth_frame([left, right, right], 1, get_level("low"))
    assert np.array_equal(out, right)  # motion is left alone: no ghost trail


def test_smooth_masks_streams_every_frame_once() -> None:
    masks = [np.full((2, 2), v, dtype=np.uint8) for v in (0, 255, 0, 255, 0)]
    loads: list[int] = []

    def load(index: int) -> np.ndarray:
        loads.append(index)
        return masks[index]

    out = list(smooth_masks(load, 5, get_level("medium")))
    assert len(out) == 5
    assert sorted(loads) == [0, 1, 2, 3, 4]
    with pytest.raises(PipelineError):
        get_level("extreme")


def test_common_bounding_box_covers_every_frame() -> None:
    frames = [chroma_key(green_screen_frame(i), ChromaSettings(key_color=GREEN)) for i in range(4)]
    boxes = [alpha_bbox(frame) for frame in frames]
    assert len(set(boxes)) == 4  # the object moves, so per-frame boxes differ
    union = union_box(boxes)
    assert union == (2, 16, 2 + 3 * 4 + 12, 28)
    assert alpha_bbox(np.zeros((4, 4, 4), dtype=np.uint8)) is None


def test_crop_with_padding_gives_one_fixed_canvas() -> None:
    frames = [chroma_key(green_screen_frame(i), ChromaSettings(key_color=GREEN)) for i in range(4)]
    union = union_box([alpha_bbox(f) for f in frames])
    plan = plan_crop(union, (64, 48), padding=4, center=True)
    cropped = [apply_crop(f, plan) for f in frames]
    assert {c.shape for c in cropped} == {(plan.height, plan.width, 4)}
    assert plan.width == (union[2] - union[0]) + 8
    # The square keeps its place in the canvas: frame 0 at the left, frame 3 at the right.
    assert cropped[0][4 + 4, 4 + 1, 3] == 255
    assert cropped[3][4 + 4, 4 + 1 + 12, 3] == 255
    clamped = plan_crop(union, (64, 48), padding=100, center=False)
    assert clamped.source_box == (0, 0, 64, 48)


def test_resize_target_sizes_and_aspect() -> None:
    assert target_size(100, 50, ResizeSettings()) == (100, 50)
    assert target_size(100, 50, ResizeSettings(mode="scale", scale=0.5)) == (50, 25)
    assert target_size(100, 50, ResizeSettings(mode="scale", scale=2.0)) == (200, 100)
    custom = ResizeSettings(mode="custom", width=200)
    assert target_size(100, 50, custom) == (200, 100)
    both = ResizeSettings(mode="custom", width=200, height=200)
    assert target_size(100, 50, both) == (200, 100)  # fits the box, keeps the ratio
    free = ResizeSettings(mode="custom", width=200, height=200, keep_aspect=False)
    assert target_size(100, 50, free) == (200, 200)


def test_resize_uses_premultiplied_alpha_so_edges_stay_clean() -> None:
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :4] = (255, 0, 0, 255)  # red half
    rgba[:, 4:] = (0, 255, 0, 0)  # transparent half with green RGB
    out = resize_rgba(rgba, (16, 16))
    edge = out[8, 7:9]
    visible = edge[edge[:, 3] > 20]
    assert visible.size and (visible[:, 1] < 40).all()  # no green bleeds into visible pixels


def test_fill_transparent_rgb_spreads_color_but_not_alpha() -> None:
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :4] = (255, 0, 0, 255)
    rgba[:, 4:, :3] = (0, 255, 0)  # green under full transparency
    filled = fill_transparent_rgb(rgba)
    assert np.array_equal(filled[..., 3], rgba[..., 3])
    assert filled[4, 4, 0] > 200 and filled[4, 4, 1] < 40  # next to red: now red
