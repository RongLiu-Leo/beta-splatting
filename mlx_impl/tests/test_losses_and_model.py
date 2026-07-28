"""Smoke tests for losses and model init/save/load.

Verifies:
- l1_loss / ssim / psnr in MLX match reference computations.
- BetaModel.create_from_pcd runs and produces the expected shapes for both
  C=3 and C=4.
- save_ply / load_ply roundtrip preserves parameter values exactly.

Run: `python -m mlx_impl.tests.test_losses_and_model`
"""

from __future__ import annotations
import os
import tempfile
import numpy as np


def test_losses():
    import mlx.core as mx
    from mlx_impl.losses import l1_loss, ssim, psnr

    rng = np.random.default_rng(7)
    for C in (3, 4):
        a_np = rng.random((1, C, 64, 64)).astype(np.float32)
        b_np = a_np + rng.normal(size=(1, C, 64, 64)).astype(np.float32) * 0.05
        a, b = mx.array(a_np), mx.array(b_np)

        l1 = float(l1_loss(a, b))
        l1_ref = float(np.abs(a_np - b_np).mean())
        assert abs(l1 - l1_ref) < 1e-5, (l1, l1_ref)
        print(f"OK   l1 C={C}: {l1:.5f} (ref {l1_ref:.5f})")

        p = float(psnr(a, b))
        mse = ((a_np - b_np) ** 2).mean()
        p_ref = 20 * np.log10(1 / np.sqrt(mse))
        assert abs(p - p_ref) < 1e-3, (p, p_ref)
        print(f"OK   psnr C={C}: {p:.2f} dB (ref {p_ref:.2f})")

        # SSIM should be in [0, 1] and near 1 for near-identical images.
        s = float(ssim(a, b))
        assert 0.0 <= s <= 1.0, s
        assert s > 0.5, f"SSIM unexpectedly low for near-identical images: {s}"
        print(f"OK   ssim C={C}: {s:.4f}")


def test_model_init_and_ply_roundtrip():
    from mlx_impl.beta_model import BetaModel

    for C in (3, 4):
        m = BetaModel(sh_degree=1, sb_number=2, color_channels=C)
        rng = np.random.default_rng(0)
        N = 100
        points = rng.random((N, 3)).astype(np.float32) * 2 - 1
        colors = rng.random((N, 3)).astype(np.float32)  # always init from RGB
        m.create_from_pcd(points, colors, spatial_lr_scale=1.0)

        # Shape checks.
        assert m._xyz.shape == (N, 3)
        assert m._sh0.shape == (N, 1, C), m._sh0.shape
        K_sh = (m.max_sh_degree + 1) ** 2
        assert m._shN.shape == (N, K_sh - 1, C), m._shN.shape
        assert m._sb_params.shape == (N, 2, C + 3), m._sb_params.shape
        assert m._opacity.shape == (N, 1)
        assert m._scaling.shape == (N, 3)
        assert m._rotation.shape == (N, 4)

        # Activations produce valid values.
        opacity = m.get_opacity
        assert opacity.min().item() > 0 and opacity.max().item() < 1
        scaling = m.get_scaling
        assert scaling.min().item() > 0
        rot = m.get_rotation
        # Quats should be unit-norm.
        import mlx.core as mx
        rot_norm = mx.sqrt((rot * rot).sum(axis=-1))
        assert abs(float(rot_norm.mean()) - 1.0) < 1e-5, float(rot_norm.mean())

        # Save + reload round-trip.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "roundtrip.ply")
            m.save_ply(path)
            assert os.path.exists(path)

            m2 = BetaModel(sh_degree=1, sb_number=2, color_channels=C)
            m2.load_ply(path)

            def close(a, b, tag):
                a, b = np.array(a), np.array(b)
                d = float(np.max(np.abs(a - b)))
                assert d < 1e-6, f"{tag} diff {d:.2e}"
                return d

            close(m._xyz, m2._xyz, "xyz")
            close(m._sh0, m2._sh0, "sh0")
            close(m._shN, m2._shN, "shN")
            close(m._sb_params, m2._sb_params, "sb_params")
            close(m._opacity, m2._opacity, "opacity")
            close(m._beta, m2._beta, "beta")
            close(m._scaling, m2._scaling, "scaling")
            close(m._rotation, m2._rotation, "rotation")
        print(f"OK   BetaModel init + PLY roundtrip C={C} (N={N}, sh_deg=1, K={m.sb_number})")


def main():
    print("=== losses ===")
    test_losses()
    print("=== model init + PLY roundtrip ===")
    test_model_init_and_ply_roundtrip()


if __name__ == "__main__":
    main()
