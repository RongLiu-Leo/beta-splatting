"""Smoke tests for the Track A soft rasterizer.

Verifies:
1. Single primitive at image center → bright disk on background.
2. Empty scene → pure background.
3. Two overlapping opaque primitives → front one wins (depth sort works).
4. Gradient flows through rasterization to primitive parameters.
5. Memory bounded when scene has many primitives.

Run: `python -m mlx_impl.tests.test_rasterizer`
"""

from __future__ import annotations
import numpy as np
import mlx.core as mx

from mlx_impl.rasterizer.slow import rasterize_soft


def _isotropic_conic(sigma_pix: float) -> mx.array:
    """conic packed [a, 0, c] with a = c = 1 / sigma_pix²."""
    v = 1.0 / (sigma_pix * sigma_pix)
    return mx.array([v, 0.0, v])


def test_single_primitive_center():
    """One bright red primitive at the center → red disk on black bg."""
    H = W = 64
    means_2d = mx.array([[32.0, 32.0]])
    conic = _isotropic_conic(6.0)[None, :]                # spread of ~6 pixels
    opacity = mx.array([0.9])
    beta = mx.array([2.0])
    colors = mx.array([[1.0, 0.0, 0.0]])                  # red
    bg = mx.array([0.0, 0.0, 0.0])

    img, alpha = rasterize_soft(
        means_2d, conic, opacity, beta, colors,
        W, H, bg, depths=mx.array([1.0]),
    )
    assert img.shape == (H, W, 3), img.shape
    assert alpha.shape == (H, W), alpha.shape

    # Center pixel should be bright red.
    center = img[32, 32]
    assert float(center[0]) > 0.5, f"center red channel too dim: {float(center[0])}"
    assert float(center[1]) < 0.05, float(center[1])
    assert float(center[2]) < 0.05, float(center[2])
    # Corner pixel should be bg (black).
    corner = img[0, 0]
    assert float(mx.abs(corner).max()) < 0.01, corner.tolist()
    print(f"OK   single primitive at center: red_max={float(img[..., 0].max()):.3f}, "
          f"alpha_max={float(alpha.max()):.3f}")


def test_empty_scene():
    H = W = 32
    means_2d = mx.zeros((0, 2))
    conic = mx.zeros((0, 3))
    opacity = mx.zeros((0,))
    beta = mx.zeros((0,))
    colors = mx.zeros((0, 3))
    bg = mx.array([0.5, 0.5, 0.5])
    img, alpha = rasterize_soft(means_2d, conic, opacity, beta, colors, W, H, bg)
    assert img.shape == (H, W, 3)
    assert float(mx.abs(img - 0.5).max().item()) < 1e-6
    assert float(alpha.max().item()) < 1e-6
    print("OK   empty scene → background")


def test_depth_sort_front_wins():
    """Two overlapping opaque primitives: closer one wins."""
    H = W = 32
    # Both at pixel (16, 16), close and far.
    means_2d = mx.array([[16.0, 16.0], [16.0, 16.0]])
    conic = mx.stack([_isotropic_conic(4.0), _isotropic_conic(4.0)])
    opacity = mx.array([0.99, 0.99])
    beta = mx.array([2.0, 2.0])
    colors = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # red front, green back
    bg = mx.array([0.0, 0.0, 0.0])

    # depths: red at 1.0 (closer), green at 5.0.
    img, _ = rasterize_soft(
        means_2d, conic, opacity, beta, colors,
        W, H, bg, depths=mx.array([1.0, 5.0]),
    )
    center = img[16, 16]
    # With opacity 0.99 and beta_alpha, front should dominate.
    assert float(center[0]) > 0.9, f"red should dominate front: {center.tolist()}"
    assert float(center[1]) < 0.1, f"green should be occluded: {center.tolist()}"

    # Reverse depth ordering.
    img2, _ = rasterize_soft(
        means_2d, conic, opacity, beta, colors,
        W, H, bg, depths=mx.array([5.0, 1.0]),
    )
    center2 = img2[16, 16]
    assert float(center2[1]) > 0.9, f"green should now dominate: {center2.tolist()}"
    print("OK   depth sort: front primitive wins")


def test_gradient_flows():
    """value_and_grad through the rasterizer must produce gradients w.r.t.
    every parameter fed to it. Place primitive off-center to avoid symmetry
    giving trivially-zero grad on means_2d."""
    H = W = 32
    means_2d_v = mx.array([[10.0, 12.0]])                              # OFF-CENTER
    conic_v = _isotropic_conic(4.0)[None, :]
    opacity_v = mx.array([0.7])
    beta_v = mx.array([2.0])
    colors_v = mx.array([[0.5, 0.4, 0.3]])
    bg = mx.array([0.0, 0.0, 0.0])

    def loss_fn(m2, cn, op, bt, col):
        img, _ = rasterize_soft(m2, cn, op, bt, col, W, H, bg,
                                depths=mx.array([1.0]))
        # Target: all-white image. Loss = MSE.
        target = mx.ones((H, W, 3))
        return ((img - target) ** 2).mean()

    l, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(
        means_2d_v, conic_v, opacity_v, beta_v, colors_v
    )
    for i, g in enumerate(grads):
        assert g is not None
        finite = bool(mx.all(mx.isfinite(g)).item())
        nz = bool(mx.any(g != 0).item())
        name = ["means_2d", "conic", "opacity", "beta", "colors"][i]
        assert finite, f"{name} grad has non-finite entries"
        assert nz, f"{name} grad is all zeros"
    print(f"OK   gradient flow: loss={float(l):.4f}, all 5 param grads finite and non-zero")


def test_many_primitives_memory():
    """Many random primitives, check memory + speed + no NaN."""
    import time
    H = W = 128
    N = 5000
    rng = np.random.default_rng(7)
    means_2d = mx.array((rng.random((N, 2)) * np.array([W, H])).astype(np.float32))
    conic = mx.broadcast_to(_isotropic_conic(3.0)[None, :], (N, 3))
    opacity = mx.array(rng.uniform(0.1, 0.5, N).astype(np.float32))
    beta = mx.array(2.0 * np.ones(N, dtype=np.float32))
    colors = mx.array(rng.uniform(0, 1, (N, 3)).astype(np.float32))
    depths = mx.array(rng.uniform(1.0, 10.0, N).astype(np.float32))
    bg = mx.array([1.0, 1.0, 1.0])

    mem_before = mx.get_active_memory() / 1e6
    t0 = time.time()
    img, alpha = rasterize_soft(means_2d, conic, opacity, beta, colors,
                                 W, H, bg, depths=depths, chunk_size=512)
    mx.eval(img, alpha)
    dt = time.time() - t0
    mem_after = mx.get_active_memory() / 1e6
    peak_mb = mx.get_peak_memory() / 1e6

    assert bool(mx.all(mx.isfinite(img)).item()), "NaN/Inf in image"
    print(f"OK   N={N}, {W}x{H}: {dt*1000:.0f} ms, "
          f"mem {mem_before:.0f}→{mem_after:.0f} MB, peak {peak_mb:.0f} MB")


def main():
    print("=== rasterizer ===")
    test_single_primitive_center()
    test_empty_scene()
    test_depth_sort_front_wins()
    test_gradient_flows()
    test_many_primitives_memory()


if __name__ == "__main__":
    main()
