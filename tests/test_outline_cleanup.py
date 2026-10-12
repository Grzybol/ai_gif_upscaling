import numpy as np

from midnight_upscale.outline_cleanup import OutlineCleanupSettings, clean_outline


def _outlined_square():
    frame = np.zeros((16, 16, 4), dtype=np.uint8)
    frame[3:13, 3:13] = (220, 80, 70, 255)
    frame[3, 3:13] = (20, 20, 20, 255)
    frame[12, 3:13] = (20, 20, 20, 255)
    frame[3:13, 3] = (20, 20, 20, 255)
    frame[3:13, 12] = (20, 20, 20, 255)
    frame[1, 8] = (10, 10, 10, 210)  # isolated detached residue
    return frame


def test_disabled_cleanup_is_exact_noop():
    original = _outlined_square()
    assert np.array_equal(clean_outline(original, OutlineCleanupSettings()), original)


def test_continuous_dark_outline_survives_and_sparse_edge_is_softened():
    original = _outlined_square()
    cleaned = clean_outline(original, OutlineCleanupSettings(enabled=True))
    assert np.array_equal(cleaned[3, 5:11, :3], original[3, 5:11, :3])
    assert np.all(cleaned[5:11, 5:11] == original[5:11, 5:11])
    assert cleaned[1, 8, 3] < original[1, 8, 3]
    assert np.any(cleaned != original)
