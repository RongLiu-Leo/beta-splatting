"""Geometry pipeline smoke + parity tests.

Verifies:
- build_covariance_3d produces SPD matrices with correct scale.
- world_to_cam: identity view → mean_c == mean_w.
- persp_proj: a point at (0,0,z) projects to (cx, cy). Point at (1,0,z) projects to (cx + fx/z, cy).
- project() on a small fake model returns coherent shapes and a reasonable valid mask.
- beta_alpha: at dx=dy=0 returns opacity * 1^beta = opacity. At sigma > 1 returns 0.

Run: `python -m mlx_impl.tests.test_geometry`
"""

from __future__ import annotations
import numpy as np
import mlx.core as mx


def test_build_covariance_3d():
    from mlx_impl.geometry.projection import build_covariance_3d
    # Identity rotation, unit scales → covariance = I.
    N = 5
    s = mx.ones((N, 3))
    r = mx.zeros((N, 4))
    r[:, 0] = 1.0
    cov = build_covariance_3d(s, r)
    assert cov.shape == (N, 3, 3)
    diff = float(mx.abs(cov - mx.eye(3)).max().item())
    assert diff < 1e-6, diff
    # Anisotropic scale → diagonal covariance = diag(s^2).
    s2 = mx.array([[2.0, 3.0, 4.0]])
    r2 = mx.array([[1.0, 0.0, 0.0, 0.0]])
    cov2 = build_covariance_3d(s2, r2)
    expected = mx.diag(mx.array([4.0, 9.0, 16.0]))
    diff = float(mx.abs(cov2[0] - expected).max().item())
    assert diff < 1e-5, diff
    print(f"OK   build_covariance_3d: SPD diag matches diag(s^2)")


def test_world_to_cam_identity():
    from mlx_impl.geometry.projection import world_to_cam
    means = mx.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    covars = mx.broadcast_to(mx.eye(3)[None], (2, 3, 3))
    viewmat = mx.eye(4)
    m_c, c_c = world_to_cam(means, covars, viewmat)
    assert float(mx.abs(m_c - means).max().item()) < 1e-6
    assert float(mx.abs(c_c - covars).max().item()) < 1e-6
    print(f"OK   world_to_cam(identity) = identity")


def test_persp_proj_center_point():
    from mlx_impl.geometry.projection import persp_proj
    fx = fy = 500.0
    cx = cy = 400.0
    W = H = 800
    # Point at (0, 0, 5) → pixel (cx, cy) = (400, 400).
    means_c = mx.array([[0.0, 0.0, 5.0]])
    covars_c = mx.eye(3)[None]
    m_2d, cov_2d = persp_proj(means_c, covars_c, fx, fy, cx, cy, W, H)
    assert abs(float(m_2d[0, 0]) - cx) < 1e-4, float(m_2d[0, 0])
    assert abs(float(m_2d[0, 1]) - cy) < 1e-4, float(m_2d[0, 1])
    # Point at (1, 0, 5) → x_pixel = cx + fx * 1/5 = 400 + 100 = 500.
    means_c2 = mx.array([[1.0, 0.0, 5.0]])
    m2, _ = persp_proj(means_c2, covars_c, fx, fy, cx, cy, W, H)
    assert abs(float(m2[0, 0]) - 500.0) < 1e-3, float(m2[0, 0])
    print(f"OK   persp_proj: centered point → (cx, cy); shifted → (500, cy)")


def test_project_full():
    from mlx_impl.geometry.projection import project
    N = 100
    rng = np.random.default_rng(42)
    means_w = mx.array(rng.normal(size=(N, 3)).astype(np.float32) * 0.5)
    # Push them in front of camera by shifting z.
    means_w = means_w + mx.array([0.0, 0.0, 5.0])
    scaling = mx.array(0.05 * np.ones((N, 3), dtype=np.float32))
    rotation = mx.zeros((N, 4))
    rotation[:, 0] = 1.0
    viewmat = mx.eye(4)
    W = H = 800
    K = mx.array([[500.0, 0, 400.0], [0, 500.0, 400.0], [0, 0, 1.0]])

    r = project(
        means_w, scaling, rotation, viewmat,
        fx=500.0, fy=500.0, cx=400.0, cy=400.0,
        width=W, height=H,
    )
    assert r["means_2d"].shape == (N, 2)
    assert r["conic"].shape == (N, 3)
    assert r["depths"].shape == (N,)
    assert r["radii"].shape == (N,)
    assert r["valid"].shape == (N,)
    n_valid = int(r["valid"].sum().item())
    n_depth_ok = int((r["depths"] > 0.01).sum().item())
    print(f"OK   project: {n_valid}/{N} primitives valid, {n_depth_ok} in-depth")
    assert n_valid > N // 2, f"expected most primitives visible, got {n_valid}/{N}"


def test_beta_alpha():
    from mlx_impl.geometry.beta_kernel import beta_alpha
    # At center: dx=dy=0 → sigma=0 → alpha = opacity * 1^beta = opacity.
    conic = mx.array([1.0, 0.0, 1.0])  # isotropic inv-cov=I
    opacity = mx.array(0.5)
    beta = mx.array(2.0)
    a = beta_alpha(mx.array(0.0), mx.array(0.0), conic, opacity, beta)
    assert abs(float(a) - 0.5) < 1e-5, float(a)
    # sigma = 1 → alpha = 0 (edge of support).
    a2 = beta_alpha(mx.array(1.0), mx.array(0.0), conic, opacity, beta)
    assert abs(float(a2)) < 1e-6, float(a2)
    # sigma > 1 → alpha = 0.
    a3 = beta_alpha(mx.array(2.0), mx.array(0.0), conic, opacity, beta)
    assert abs(float(a3)) < 1e-6, float(a3)
    # Halfway (sigma=0.5) → alpha = opacity * 0.5^beta = 0.5 * 0.25 = 0.125.
    a4 = beta_alpha(mx.array(mx.sqrt(mx.array(0.5)).item()), mx.array(0.0), conic, opacity, beta)
    assert abs(float(a4) - 0.125) < 1e-4, float(a4)
    print(f"OK   beta_alpha: center=opacity, boundary=0, outside=0, sigma=0.5 checks")


def main():
    print("=== geometry ===")
    test_build_covariance_3d()
    test_world_to_cam_identity()
    test_persp_proj_center_point()
    test_project_full()
    test_beta_alpha()


if __name__ == "__main__":
    main()
