"""DBS Beta kernel evaluation, pure MLX.

Ports the per-pixel alpha computation from rasterize_to_pixels_fwd.cu:145-155:

    sigma = conic.x * dx*dx + conic.z * dy*dy + 2 * conic.y * dx*dy
    alpha = min(0.999, opacity * pow(max(0, 1 - sigma), beta))

sigma < 1  → inside the Beta kernel support
sigma >= 1 → contribution is zero (kernel bounded support, DBS's key change)

This is evaluated inside the rasterizer's inner loop for each primitive
contributing to each pixel. Wrapping it as its own function makes the
rasterizer readable and the math independently testable.
"""

from __future__ import annotations
import mlx.core as mx


def beta_alpha(
    dx: mx.array,        # (..., ) x-offset from primitive center in pixels
    dy: mx.array,        # (..., )
    conic: mx.array,     # (..., 3) inv-cov2d packed [a, b, c]
    opacity: mx.array,   # (..., 1) or scalar; broadcasts
    beta: mx.array,      # (..., 1) or scalar; broadcasts
) -> mx.array:
    """Return per-primitive per-pixel alpha contribution.

    Shapes: broadcasts naturally. Typical use in the rasterizer has
    dx/dy of shape (H, W, N) or (N, H, W) and conic/opacity/beta of (N, ...).
    """
    a = conic[..., 0]
    b = conic[..., 1]
    c = conic[..., 2]
    sigma = a * dx * dx + c * dy * dy + 2.0 * b * dx * dy

    # (1 - sigma) clamped to [0, ∞); pow with negative base → NaN, so clamp.
    base = mx.maximum(1.0 - sigma, 0.0)
    beta_term = mx.power(base, beta)
    alpha = mx.minimum(0.999, opacity * beta_term)
    # Force zero exactly at/outside support.
    alpha = mx.where(sigma < 1.0, alpha, mx.zeros_like(alpha))
    return alpha
