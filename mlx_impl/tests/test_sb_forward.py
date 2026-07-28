"""Verify the MLX SB kernel matches a numpy reference of the CUDA math.

Run: `python -m mlx_impl.tests.test_sb_forward`

The reference implementation below is a direct numpy port of
spherical_beta.cuh::spherical_beta_isotropic_fwd (191 lines in CUDA →
~30 lines in numpy). This test does not need a GPU, MLX Metal, or a trained
PLY — pure math parity.
"""

from __future__ import annotations
import numpy as np


def _sb_reference_numpy(
    c0: np.ndarray,          # (N, C)
    sb_params: np.ndarray,   # (N, K, C + 3)
    dirs: np.ndarray,        # (N, 3)
) -> np.ndarray:
    """Reference implementation — direct translation of the CUDA loop."""
    N, K, slot = sb_params.shape
    C = c0.shape[-1]
    assert slot == C + 3

    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-20)
    out = c0.copy().astype(np.float64)

    for i in range(N):
        for k in range(K):
            color = sb_params[i, k, :C]
            theta = sb_params[i, k, C]
            phi = sb_params[i, k, C + 1]
            beta = sb_params[i, k, C + 2]
            mu = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            dot = np.dot(dirs[i], mu)
            if dot > 0:
                beta_term = dot ** (4.0 * np.exp(beta))
                out[i] += color * beta_term
    return out.astype(np.float32)


def main():
    import mlx.core as mx
    from mlx_impl.color.spherical_beta import spherical_beta_forward

    rng = np.random.default_rng(42)
    for C in (3, 4):
        N, K = 32, 2
        c0 = rng.normal(size=(N, C)).astype(np.float32) * 0.1
        sb = rng.normal(size=(N, K, C + 3)).astype(np.float32) * 0.1
        # Keep theta in [0, pi], phi in [0, 2*pi], beta modest.
        sb[..., C] = np.pi * rng.random((N, K)).astype(np.float32)
        sb[..., C + 1] = 2 * np.pi * rng.random((N, K)).astype(np.float32)
        dirs = rng.normal(size=(N, 3)).astype(np.float32)

        ref = _sb_reference_numpy(c0, sb, dirs)
        got = np.array(spherical_beta_forward(mx.array(c0), mx.array(sb), mx.array(dirs)))

        max_abs = float(np.max(np.abs(got - ref)))
        max_rel = float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-6)))
        ok = max_abs < 1e-4
        status = "OK  " if ok else "FAIL"
        print(f"{status} SB C={C}: max_abs={max_abs:.2e}  max_rel={max_rel:.2e}")
        assert ok, f"SB parity failed for C={C}"


if __name__ == "__main__":
    main()
