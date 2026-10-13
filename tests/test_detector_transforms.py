import numpy as np

from detector.transforms import augment, fisheye_warp, letterbox


def _img(h=300, w=500, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_letterbox_keeps_whole_image_and_aspect():
    out = letterbox(np.full((300, 600, 3), 255, np.uint8), 384)
    assert out.shape == (384, 384, 3)
    rows = np.flatnonzero(out[:, 192, 0] > 0)
    assert abs((rows[-1] - rows[0] + 1) - 192) <= 1  # 600x300 -> 384x192, centred
    assert out[0, 0].sum() == 0 and out[-1, -1].sum() == 0


def test_fisheye_outside_circle_is_black_and_centre_kept():
    img = np.full((400, 600, 3), 200, np.uint8)
    out = fisheye_warp(img, np.random.default_rng(1), size=256)
    assert out.shape == (256, 256, 3)
    assert out[0, 0].sum() == 0 and out[255, 255].sum() == 0
    assert out[128, 128].sum() > 0


def test_augment_is_reproducible_and_sized():
    a, ops_a = augment(_img(), np.random.default_rng(7), is_fisheye=False)
    b, ops_b = augment(_img(), np.random.default_rng(7), is_fisheye=False)
    assert ops_a == ops_b and np.array_equal(a, b) and a.shape == (384, 384, 3)


def test_fisheye_sources_are_never_rewarped():
    for s in range(50):
        _, ops = augment(_img(seed=s), np.random.default_rng(s), is_fisheye=True)
        assert "fisheye_warp" not in ops
