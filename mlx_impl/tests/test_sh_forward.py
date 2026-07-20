"""Parity test for the MLX SH forward vs a numpy reference of the CUDA math.

Runs both C=3 and C=4 (the CUDA path is fixed to C=3; C=4 is the RGBA
extension supported natively on the MLX side).

Run: `python -m mlx_impl.tests.test_sh_forward`
"""

from __future__ import annotations
import numpy as np

_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
       -1.0925484305920792, 0.5462742152960396]
_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
       0.3731763325901154, -0.4570457994644658, 1.445305721320277,
       -0.5900435899266435]


def _sh_reference_numpy(coeffs: np.ndarray, dirs: np.ndarray, degree: int) -> np.ndarray:
    """Direct numpy translation of sh_coeffs_to_color_fast (spherical_harmonics.cuh)."""
    N, K, C = coeffs.shape
    x = dirs[:, 0:1]; y = dirs[:, 1:2]; z = dirs[:, 2:3]

    r = _C0 * coeffs[:, 0, :]
    if degree >= 1:
        r = r + _C1 * (-y * coeffs[:, 1, :] + z * coeffs[:, 2, :] - x * coeffs[:, 3, :])
    if degree >= 2:
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        r = r + (
            _C2[0]*xy * coeffs[:, 4, :]
            + _C2[1]*yz * coeffs[:, 5, :]
            + _C2[2]*(2*zz - xx - yy) * coeffs[:, 6, :]
            + _C2[3]*xz * coeffs[:, 7, :]
            + _C2[4]*(xx - yy) * coeffs[:, 8, :]
        )
    if degree >= 3:
        r = r + (
            _C3[0] * y*(3*xx - yy) * coeffs[:, 9, :]
            + _C3[1] * xy*z * coeffs[:, 10, :]
            + _C3[2] * y*(4*zz - xx - yy) * coeffs[:, 11, :]
            + _C3[3] * z*(2*zz - 3*xx - 3*yy) * coeffs[:, 12, :]
            + _C3[4] * x*(4*zz - xx - yy) * coeffs[:, 13, :]
            + _C3[5] * z*(xx - yy) * coeffs[:, 14, :]
            + _C3[6] * x*(xx - 3*yy) * coeffs[:, 15, :]
        )
    return r


def main():
    import mlx.core as mx
    from mlx_impl.color.spherical_harmonics import sh_forward

    rng = np.random.default_rng(1234)
    for C in (3, 4):
        for degree in (0, 1, 2, 3):
            N = 64
            K = 16   # supports all degrees
            coeffs = rng.normal(size=(N, K, C)).astype(np.float32) * 0.3
            dirs = rng.normal(size=(N, 3)).astype(np.float32)
            dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

            ref = _sh_reference_numpy(coeffs, dirs, degree)
            got = np.array(sh_forward(mx.array(coeffs), mx.array(dirs), degree))

            max_abs = float(np.max(np.abs(got - ref)))
            ok = max_abs < 1e-4
            status = "OK  " if ok else "FAIL"
            print(f"{status} SH C={C} deg={degree}: max_abs={max_abs:.2e}")
            assert ok, f"SH parity failed for C={C} deg={degree}"


if __name__ == "__main__":
    main()
