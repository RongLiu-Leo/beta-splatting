"""Spherical Harmonics color evaluation, pure MLX.

Direct translation of the sh_coeffs_to_color_fast math in
submodules/gsplat/cuda/csrc/spherical_harmonics.cuh, but generalized so
the channel count is inferred from the coefficient tensor shape (not
hardcoded to 3 as in the CUDA path).

Autograd handles the backward — no custom vjp needed.

Layout:
- coeffs : (N, K_sh, C)     where K_sh = (degree + 1)^2, C = 3 or 4
- dirs   : (N, 3)           unit direction from primitive to camera
- degree : int              active SH degree (0..3)
- returns : (N, C)          per-primitive color

The SH basis constants below are copied from spherical_harmonics.cuh with
no changes. They're standard real spherical harmonic normalizations.
"""

from __future__ import annotations
import mlx.core as mx


# Real SH basis coefficients up to degree 3, matching the CUDA header.
_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def sh_forward(
    coeffs: mx.array,   # (N, K_sh, C)
    dirs: mx.array,     # (N, 3)
    degree: int,
) -> mx.array:
    """Evaluate SH color from coefficients along per-primitive directions.

    coeffs.shape[-1] determines the output channel count (3 or 4). Higher-order
    coefficients are ignored if degree < max supported.
    """
    C = coeffs.shape[-1]
    K_sh = coeffs.shape[-2]
    assert K_sh >= (degree + 1) ** 2, (
        f"coeffs has {K_sh} basis functions, need at least {(degree + 1) ** 2} for degree {degree}"
    )

    result = _C0 * coeffs[:, 0, :]  # (N, C)

    if degree >= 1:
        x = dirs[:, 0:1]
        y = dirs[:, 1:2]
        z = dirs[:, 2:3]
        result = result + _C1 * (-y * coeffs[:, 1, :] + z * coeffs[:, 2, :] - x * coeffs[:, 3, :])

    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + (
            _C2[0] * xy * coeffs[:, 4, :]
            + _C2[1] * yz * coeffs[:, 5, :]
            + _C2[2] * (2.0 * zz - xx - yy) * coeffs[:, 6, :]
            + _C2[3] * xz * coeffs[:, 7, :]
            + _C2[4] * (xx - yy) * coeffs[:, 8, :]
        )

    if degree >= 3:
        result = result + (
            _C3[0] * y * (3.0 * xx - yy) * coeffs[:, 9, :]
            + _C3[1] * xy * z * coeffs[:, 10, :]
            + _C3[2] * y * (4.0 * zz - xx - yy) * coeffs[:, 11, :]
            + _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeffs[:, 12, :]
            + _C3[4] * x * (4.0 * zz - xx - yy) * coeffs[:, 13, :]
            + _C3[5] * z * (xx - yy) * coeffs[:, 14, :]
            + _C3[6] * x * (xx - 3.0 * yy) * coeffs[:, 15, :]
        )

    return result
