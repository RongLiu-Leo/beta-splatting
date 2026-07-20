"""3D → 2D projection for DBS primitives, pure MLX.

Ports:
- utils.cuh :: pos_world_to_cam  (L375)
- utils.cuh :: covar_world_to_cam (L406)
- utils.cuh :: persp_proj         (L253)  — EWA perspective covariance projection
- utils.cuh :: quat_scale_to_covar_preci  (L65)
- utils.cuh :: add_blur           (L459)
- fully_fused_projection_fwd.cu — overall pipeline

Single-camera version (C=1). Extensible to batched cameras later; DBS's
training loop renders one view at a time, so C=1 is the hot path.

Returned quantities per primitive:
- means_2d : (N, 2)    pixel-space centers
- conics   : (N, 3)    inv(cov_2d) packed as [a, b, c] where cov_2d_inv = [[a,b],[b,c]]
- depths   : (N,)      camera-space z
- radii    : (N,)      integer bounding radius (non-differentiable, for tile assignment)
- valid    : (N,) bool True if primitive should contribute (in-frustum, non-degenerate)
"""

from __future__ import annotations
from typing import Tuple

import mlx.core as mx


def build_covariance_3d(scaling: mx.array, rotation: mx.array) -> mx.array:
    """R diag(s^2) R^T. Matches utils.cuh:quat_scale_to_covar (SPD form, no
    lower-triangular strip).

    scaling: (N, 3), rotation: (N, 4). Returns (N, 3, 3).
    """
    from mlx_impl.densification import build_scaling_rotation
    L = build_scaling_rotation(scaling, rotation)                # (N, 3, 3)  = R @ diag(s)
    return L @ mx.transpose(L, (0, 2, 1))                        # (N, 3, 3)  SPD


def world_to_cam(
    means_w: mx.array,       # (N, 3)
    covars_w: mx.array,      # (N, 3, 3)
    viewmat: mx.array,       # (4, 4) row-major world→cam
) -> Tuple[mx.array, mx.array]:
    """Transform primitive centers and covariances into camera space.

    viewmat is the 4×4 world-to-camera transform in row-major layout (matches
    the reference's `viewpoint_camera.world_view_transform.transpose(0, 1)`
    which produces the same convention).
    """
    R = viewmat[:3, :3]                                          # (3, 3)
    t = viewmat[:3, 3]                                           # (3,)
    means_c = means_w @ mx.transpose(R) + t                      # (N, 3)
    # Cov_c = R @ Cov_w @ R^T
    Rt = mx.transpose(R)                                          # (3, 3)
    covars_c = R @ covars_w @ Rt                                 # (N, 3, 3)  broadcast
    return means_c, covars_c


def persp_proj(
    means_c: mx.array,       # (N, 3)
    covars_c: mx.array,      # (N, 3, 3)
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
) -> Tuple[mx.array, mx.array]:
    """EWA perspective projection. Ports utils.cuh:persp_proj.

    Returns (means_2d in pixel coords, cov_2d).
    """
    x = means_c[:, 0]
    y = means_c[:, 1]
    z = means_c[:, 2]

    # Clamp x/z and y/z to a slightly-expanded frustum to avoid divergence
    # near/beyond the FoV boundaries — matches the CUDA reference's 0.3*tan
    # margin.
    tan_fovx = 0.5 * width / fx
    tan_fovy = 0.5 * height / fy
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy

    rz = 1.0 / z
    rz2 = rz * rz
    tx = z * mx.minimum(mx.array(lim_x_pos), mx.maximum(mx.array(-lim_x_neg), x * rz))
    ty = z * mx.minimum(mx.array(lim_y_pos), mx.maximum(mx.array(-lim_y_neg), y * rz))

    # Jacobian of pixel = (fx * X/Z + cx, fy * Y/Z + cy) w.r.t. (X, Y, Z).
    # Shape per primitive: (2, 3). We build the batch as (N, 2, 3).
    zero = mx.zeros_like(z)
    J_row0 = mx.stack([fx * rz, zero, -fx * tx * rz2], axis=-1)   # (N, 3)
    J_row1 = mx.stack([zero, fy * rz, -fy * ty * rz2], axis=-1)   # (N, 3)
    J = mx.stack([J_row0, J_row1], axis=1)                        # (N, 2, 3)

    # cov_2d = J @ cov_c @ J^T  → (N, 2, 2)
    cov_2d = J @ covars_c @ mx.transpose(J, (0, 2, 1))

    means_2d = mx.stack([fx * x * rz + cx, fy * y * rz + cy], axis=-1)  # (N, 2)
    return means_2d, cov_2d


def add_blur_and_invert(
    cov_2d: mx.array,        # (N, 2, 2)
    eps2d: float = 0.3,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Add a small isotropic blur to cov_2d (anti-alias), invert, return conic.

    conic packs the symmetric 2x2 inverse as (a, b, c) where inv = [[a,b],[b,c]].
    Also returns det and a validity mask (det > 0 after blur → invertible).

    Matches utils.cuh:add_blur + inverse().
    """
    a = cov_2d[:, 0, 0] + eps2d
    b = cov_2d[:, 0, 1]
    c = cov_2d[:, 1, 1] + eps2d
    det = a * c - b * b
    valid = det > 1e-12
    # Guard det for the division; invalid entries will be masked out downstream.
    inv_det = 1.0 / mx.where(valid, det, mx.ones_like(det))
    conic_a = c * inv_det
    conic_b = -b * inv_det
    conic_c = a * inv_det
    conic = mx.stack([conic_a, conic_b, conic_c], axis=-1)        # (N, 3)
    return conic, det, valid


def compute_radius(cov_2d: mx.array, det: mx.array) -> mx.array:
    """1-sigma radius from cov_2d eigenvalues (non-differentiable).

    Matches the reference:
        b = 0.5 * (cov_2d[0,0] + cov_2d[1,1])
        radius = ceil(sqrt(b + sqrt(max(0.01, b*b - det))))
    """
    b = 0.5 * (cov_2d[:, 0, 0] + cov_2d[:, 1, 1])
    v1 = b + mx.sqrt(mx.maximum(mx.array(0.01), b * b - det))
    return mx.ceil(mx.sqrt(v1))                                  # (N,) float; caller casts if needed


def project(
    means_w: mx.array,       # (N, 3)
    scaling: mx.array,       # (N, 3)  post-activation
    rotation: mx.array,      # (N, 4)  post-normalization
    viewmat: mx.array,       # (4, 4)
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    near: float = 0.01, far: float = 100.0,
    eps2d: float = 0.3, radius_clip: float = 0.0,
):
    """One-shot projection wrapper. Returns a dict of per-primitive quantities."""
    covars_w = build_covariance_3d(scaling, rotation)             # (N, 3, 3)
    means_c, covars_c = world_to_cam(means_w, covars_w, viewmat)  # (N, 3), (N, 3, 3)

    # Frustum clip on depth.
    depths = means_c[:, 2]
    in_depth = (depths > near) & (depths < far)

    means_2d, cov_2d = persp_proj(means_c, covars_c, fx, fy, cx, cy, width, height)
    conic, det, cov_ok = add_blur_and_invert(cov_2d, eps2d)
    radius = compute_radius(cov_2d, det)

    # Image-region cull.
    in_image = (
        (means_2d[:, 0] + radius > 0) & (means_2d[:, 0] - radius < width) &
        (means_2d[:, 1] + radius > 0) & (means_2d[:, 1] - radius < height)
    )
    big_enough = radius > radius_clip
    valid = in_depth & cov_ok & in_image & big_enough

    return dict(
        means_2d=means_2d,       # (N, 2)   pixel coords
        conic=conic,             # (N, 3)   inv-cov2d packed (a, b, c)
        depths=depths,           # (N,)     camera-space z
        radii=radius,            # (N,)     float; use for tile assignment
        valid=valid,             # (N,)     bool
    )
