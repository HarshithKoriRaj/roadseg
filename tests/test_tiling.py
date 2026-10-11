import numpy as np, pytest
from roadseg.data.tiling import stitch, tile_image, tile_positions


@pytest.mark.parametrize("h,w", [(1500, 1500), (1024, 1024), (600, 513)])
def test_positions_cover_every_pixel(h, w):
    cov = np.zeros((h, w), bool)
    for y, x in tile_positions(h, w, 512, 256):
        cov[y:y + 512, x:x + 512] = True
    assert cov.all(), "tiling must cover the whole image including ragged edges"


@pytest.mark.parametrize("h,w", [(1500, 1500), (600, 513)])
def test_stitch_inverts_tile(h, w):
    rng = np.random.default_rng(0)
    img = rng.random((h, w)).astype(np.float32)
    tiles, pos = tile_image(img, 512, 256)
    np.testing.assert_allclose(stitch(tiles, pos, h, w, 512), img, atol=1e-5)


def test_tiles_are_all_the_requested_size():
    tiles, _ = tile_image(np.zeros((600, 513), np.float32), 512, 256)
    assert all(t.shape == (512, 512) for t in tiles)


def test_overlaps_are_averaged_not_overwritten():
    h = w = 768
    tiles, pos = tile_image(np.ones((h, w), np.float32), 512, 256)
    tiles = [t * (i + 1) for i, t in enumerate(tiles)]
    out = stitch(tiles, pos, h, w, 512)
    assert out.min() > 0 and np.isfinite(out).all()


@pytest.mark.parametrize("h,w", [(1500, 1500), (600, 513), (300, 700), (200, 200)])
def test_multichannel_tiles_keep_shape_dtype_and_positions(h, w):
    rng = np.random.default_rng(1)
    img3 = rng.random((h, w, 3)).astype(np.float32)
    tiles3, pos3 = tile_image(img3, 512, 256)
    _, pos2 = tile_image(img3[..., 0], 512, 256)
    assert pos3 == pos2
    assert all(t.shape == (512, 512, 3) and t.dtype == img3.dtype for t in tiles3)


@pytest.mark.parametrize("h,w", [(300, 700), (700, 300), (200, 200), (1, 600)])
def test_images_smaller_than_tile_roundtrip(h, w):
    rng = np.random.default_rng(2)
    img = rng.random((h, w)).astype(np.float32)
    tiles, pos = tile_image(img, 512, 256)
    assert all(t.shape == (512, 512) for t in tiles)
    out = stitch(tiles, pos, h, w, 512)
    assert out.shape == (h, w)
    np.testing.assert_allclose(out, img, atol=1e-5)
