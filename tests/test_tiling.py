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
    assert pos == [(0, 0), (0, 256), (256, 0), (256, 256)]
    tiles = [np.full((512, 512), i + 1, np.float32) for i in range(len(tiles))]
    out = stitch(tiles, pos, h, w, 512)
    assert out[10, 10] == 1.0                      # tile 0 only
    assert out[384, 384] == pytest.approx(2.5)     # all four: (1+2+3+4)/4
    assert out[10, 300] == pytest.approx(1.5)      # tiles 0 and 1
    assert out[300, 10] == pytest.approx(2.0)      # tiles 0 and 2
    acc = np.zeros((h, w)); cnt = np.zeros((h, w))
    for t, (y, x) in zip(tiles, pos):
        acc[y:y + 512, x:x + 512] += t
        cnt[y:y + 512, x:x + 512] += 1
    np.testing.assert_allclose(out, acc / cnt, atol=1e-6)


@pytest.mark.parametrize("stride", [0, -1, 513])
def test_tile_positions_rejects_bad_stride(stride):
    with pytest.raises(ValueError):
        tile_positions(1024, 1024, 512, stride)


def test_stitch_rejects_uncovered_pixels():
    img = np.ones((1024, 1024), np.float32)
    tiles, pos = tile_image(img, 512, 256)
    with pytest.raises(ValueError):
        stitch(tiles[1:], pos[1:], 1024, 1024, 512)


@pytest.mark.parametrize("h,w", [(200, 200), (300, 700)])
def test_padded_pixels_cannot_leak_into_output(h, w):
    rng = np.random.default_rng(3)
    img = rng.random((h, w)).astype(np.float32)
    tiles, pos = tile_image(img, 512, 256)
    leaked = []
    for t, (y, x) in zip(tiles, pos):
        t = t.copy()
        t[max(h - y, 0):, :] = 1e6
        t[:, max(w - x, 0):] = 1e6
        leaked.append(t)
    np.testing.assert_allclose(stitch(leaked, pos, h, w, 512), img, atol=1e-5)


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
