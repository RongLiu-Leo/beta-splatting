"""MCMC densification helpers — quaternion rotation, covariance construction,
position noise term, and regularization losses.

Ports the following from the reference:
- utils/general_utils.py :: build_rotation (L86)
- utils/general_utils.py :: build_scaling_rotation (L112)
- train.py :: noise = randn * (1 - opacity)^100 * noise_lr * xyz_lr;
              noise = actual_covariance @ noise (L147-155)
- train.py :: loss += opacity_reg * abs(get_opacity).mean();
              loss += scale_reg * abs(get_scaling).mean() (L112-114)

The MCMC-invariance opacity update `new_op = 1 - (1 - op)^(1/(ratio+1))`
lives on BetaModel._update_params — not here — because it needs access to
the model's activation functions.
"""

from __future__ import annotations
import mlx.core as mx


def build_rotation(q: mx.array) -> mx.array:
    """Quaternion → 3x3 rotation matrix. q: (N, 4) with [w, x, y, z] layout.

    Direct port of utils/general_utils.py:build_rotation.
    """
    norm = mx.rsqrt((q * q).sum(axis=-1, keepdims=True) + 1e-20)
    q = q * norm
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rows = mx.stack(
        [
            mx.stack([1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)], axis=-1),
            mx.stack([2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)], axis=-1),
            mx.stack([2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )
    return rows  # (N, 3, 3)


def build_scaling_rotation(s: mx.array, q: mx.array) -> mx.array:
    """Returns L = R @ diag(s). Shape (N, 3, 3).

    Matches utils/general_utils.py:build_scaling_rotation. `actual_covariance =
    L @ L.T` recovers the SPD covariance the reference uses for the MCMC noise
    projection.
    """
    R = build_rotation(q)                # (N, 3, 3)
    # diag(s) via broadcasting.
    return R * s[:, None, :]             # (N, 3, 3), scales columns


def apply_position_noise(
    xyz: mx.array,          # (N, 3)   pre-activation positions
    scaling: mx.array,      # (N, 3)   post-activation scales
    rotation: mx.array,     # (N, 4)   raw quaternions
    opacity: mx.array,      # (N, 1)   post-sigmoid opacities in [0, 1]
    xyz_lr: float,
    noise_lr: float,
    key: mx.array | None = None,
) -> mx.array:
    """Return `xyz + noise` where noise follows the reference MCMC schedule:

        noise = randn(N, 3) * (1 - opacity)^100 * noise_lr * xyz_lr
        noise = actual_covariance @ noise             # rotate/scale by covariance

    Matches train.py:147-155.
    """
    L = build_scaling_rotation(scaling, rotation)              # (N, 3, 3)
    cov = L @ mx.transpose(L, (0, 2, 1))                        # (N, 3, 3)
    n = mx.random.normal(xyz.shape, key=key)                    # (N, 3)
    scale = mx.power(1.0 - opacity, 100) * (noise_lr * xyz_lr)  # (N, 1)
    n = n * scale                                                # (N, 3)
    n = (cov @ n[..., None])[..., 0]                             # (N, 3)
    return xyz + n


def regularization_loss(
    opacity: mx.array,      # post-sigmoid, (N, 1)
    scaling: mx.array,      # post-exp,    (N, 3)
    opacity_reg: float,
    scale_reg: float,
) -> mx.array:
    """Two L1 regularizers inside the densification window.

    Matches train.py:112-114:
        loss += opacity_reg * abs(get_opacity).mean()
        loss += scale_reg   * abs(get_scaling).mean()
    """
    return opacity_reg * mx.abs(opacity).mean() + scale_reg * mx.abs(scaling).mean()
