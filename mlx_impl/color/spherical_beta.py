"""Spherical Betas color evaluation, pure MLX.

Direct translation of submodules/gsplat/cuda/csrc/spherical_beta.cuh.

Forward math (from the CUDA header, verbatim):

    C = c0 + Σᵢ cᵢ · max(dot(μᵢ, v), 0)^(4 · exp(βᵢ))
    where μᵢ = (sin θᵢ cos φᵢ, sin θᵢ sin φᵢ, cos θᵢ)

Because this is element-wise arithmetic across primitives and lobes, MLX
autograd handles the backward pass without a custom vjp — the CUDA bwd
kernel exists in the fork purely for speed, not because autograd can't
derive it. The math it computes is:

    dL/dr_i = v_color · betaTerm
    dL/dθ_i = v_color · (Σ_c c_i) · exponent · dot^(exponent-1) · d(dot)/dθ
    dL/dφ_i = v_color · (Σ_c c_i) · exponent · dot^(exponent-1) · d(dot)/dφ
    dL/dβ_i = v_color · (Σ_c c_i) · exponent · dot^(exponent) · ln(dot)

We rely on mx.grad to rediscover these.

Layout:
- sb_params: (N, K, 6) with slots [r, g, b, theta, phi, beta]
- c0:        (N, C)     DC color per primitive (C = color_channels)
- dirs:      (N, 3)     unit-normalized view directions per primitive
- returns:   (N, C)     per-primitive colors

For the 4-channel (RGBA) extension (Phase 8), sb_params grows to (N, K, 7)
with slots [r, g, b, a, theta, phi, beta]. This module is written so
color_channels is inferred from the tensor shape — no code change needed
for 4c beyond passing the right-shaped input.
"""

from __future__ import annotations

import mlx.core as mx


def _direction_from_angles(theta: mx.array, phi: mx.array) -> mx.array:
    """(theta, phi) → 3D unit direction, shape (..., 3)."""
    sin_theta = mx.sin(theta)
    return mx.stack(
        [sin_theta * mx.cos(phi), sin_theta * mx.sin(phi), mx.cos(theta)],
        axis=-1,
    )


def spherical_beta_forward(
    c0: mx.array,          # (N, C)
    sb_params: mx.array,   # (N, K, 3 + 3)  or  (N, K, C + 3) for RGBA
    dirs: mx.array,        # (N, 3)
) -> mx.array:
    """SB color evaluation for N primitives, K lobes each.

    The last-dim layout of sb_params is:
        [color_0, color_1, ..., color_{C-1}, theta, phi, beta]

    C is inferred from c0.shape[-1] and validated against sb_params.
    """
    C = c0.shape[-1]
    K_slot_count = sb_params.shape[-1]
    assert K_slot_count == C + 3, (
        f"sb_params last-dim must be color_channels + 3 (θ,φ,β), "
        f"got {K_slot_count} with C={C}"
    )

    # Split lobe params along the last axis.
    lobe_colors = sb_params[..., :C]              # (N, K, C)
    theta = sb_params[..., C]                     # (N, K)
    phi = sb_params[..., C + 1]                   # (N, K)
    beta = sb_params[..., C + 2]                  # (N, K)

    # Normalize per-primitive direction.
    dirs_norm = dirs * mx.rsqrt(
        (dirs * dirs).sum(axis=-1, keepdims=True) + 1e-20
    )

    # Lobe mean directions from spherical angles.
    mu = _direction_from_angles(theta, phi)       # (N, K, 3)

    # dot(dirs, mu_k) broadcast over lobes.
    dot = (dirs_norm[:, None, :] * mu).sum(axis=-1)  # (N, K)

    # Beta term: max(dot, 0) ** (4 * exp(beta)). Clamp interior of pow.
    dot_pos = mx.maximum(dot, 0.0)
    exponent = 4.0 * mx.exp(beta)
    # Avoid pow(0, x) NaN via mask: where dot<=0, betaTerm=0.
    safe_base = mx.maximum(dot_pos, 1e-20)
    beta_term = mx.where(dot > 0, mx.power(safe_base, exponent), mx.zeros_like(dot))

    # Sum over lobes: (N, K, C) * (N, K, 1) -> (N, C)
    lobe_contrib = lobe_colors * beta_term[..., None]
    color = c0 + lobe_contrib.sum(axis=1)
    return color
