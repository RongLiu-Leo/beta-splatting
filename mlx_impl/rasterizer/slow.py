"""Track A — pure-MLX soft rasterizer, slow but differentiable.

Strategy: chunked front-to-back alpha compositing. All ops are MLX-native, so
mx.grad flows through the entire renderer for training.

Not fast:
- No tile-based primitive-to-pixel assignment (every primitive is evaluated
  at every pixel of every chunk).
- Sort happens once globally per view, on numpy (small overhead, no gradient).
- Chunking amortizes memory by processing primitives in groups.

Correctness is the goal, not throughput. On M4 Pro at 128×128 with 30k
primitives, this renders in seconds per view — enough to close the training
loop and verify the pipeline, then Track B (Metal kernels) takes it fast.

Per-chunk memory scales as CHUNK * H * W * 4 bytes. Defaults CHUNK=512 give
~32 MB per chunk at 128×128.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import mlx.core as mx

from mlx_impl.geometry.beta_kernel import beta_alpha


def _depth_sort_indices(depths: mx.array) -> mx.array:
    """Front-to-back sort by depth. Non-differentiable; runs on numpy."""
    d = np.array(depths)
    idx = np.argsort(d).astype(np.int32)
    return mx.array(idx)


def rasterize_soft(
    means_2d: mx.array,      # (N, 2) pixel coords
    conic: mx.array,         # (N, 3) packed inv-cov2d
    opacity: mx.array,       # (N,)   post-sigmoid geometric opacity
    beta: mx.array,          # (N,)   post-activation (4 * exp(_beta))
    colors: mx.array,        # (N, C) per-primitive per-view colors
    width: int, height: int,
    background: mx.array,    # (C,)
    valid: mx.array | None = None,  # (N,) bool; if provided, invalid primitives excluded
    depths: mx.array | None = None, # (N,) if provided, front-to-back sort applied first
    chunk_size: int = 512,
    trans_eps: float = 1e-4, # early-terminate accumulation when T falls below this
    checkpoint: bool = True, # recompute chunk intermediates during backward
) -> Tuple[mx.array, mx.array]:
    """Return (image, alpha). image: (H, W, C), alpha: (H, W)."""
    N = means_2d.shape[0]
    C = colors.shape[-1]

    # 1. Filter to valid primitives (or index them out via mask indexing).
    if valid is not None:
        keep = mx.array(np.where(np.array(valid))[0].astype(np.int32))
        means_2d = means_2d[keep]
        conic = conic[keep]
        opacity = opacity[keep]
        beta = beta[keep]
        colors = colors[keep]
        if depths is not None:
            depths = depths[keep]
        N = int(keep.shape[0])

    if N == 0:
        img = mx.broadcast_to(background[None, None, :], (height, width, C))
        alpha = mx.zeros((height, width))
        return img, alpha

    # 2. Depth sort (front to back) — index-based, no gradient impact.
    if depths is not None:
        order = _depth_sort_indices(depths)
        means_2d = means_2d[order]
        conic = conic[order]
        opacity = opacity[order]
        beta = beta[order]
        colors = colors[order]

    # 3. Pixel grid. Note: means_2d are in pixel coords with (0,0) at top-left.
    ys, xs = mx.meshgrid(mx.arange(height, dtype=mx.float32),
                         mx.arange(width, dtype=mx.float32), indexing="ij")
    # xs, ys: (H, W); add 0.5 for pixel-center convention.
    xs = xs + 0.5
    ys = ys + 0.5

    # 4. Chunked front-to-back accumulation.
    image = mx.zeros((height, width, C))
    T = mx.ones((height, width))

    # Per-chunk step wrapped in mx.checkpoint so autograd recomputes the
    # per-pixel intermediates (dx, dy, sigma, base, alpha, one_minus, T_within,
    # w — 8 tensors of shape (H, W, k)) during backward instead of storing them.
    # For 100x100 with chunk=256 that's ~80 MB of intermediates per chunk that
    # we no longer keep in memory across all chunks. See tests/test_rasterizer.py
    # for the peak-memory delta.
    def _one_chunk(image, T, m_chunk, cn_chunk, op_chunk, bt_chunk, col_chunk):
        dx = xs[..., None] - m_chunk[:, 0][None, None, :]
        dy = ys[..., None] - m_chunk[:, 1][None, None, :]
        a = cn_chunk[:, 0][None, None, :]
        b = cn_chunk[:, 1][None, None, :]
        c = cn_chunk[:, 2][None, None, :]
        sigma = a * dx * dx + c * dy * dy + 2.0 * b * dx * dy
        base = mx.maximum(1.0 - sigma, 0.0)
        alpha = mx.minimum(
            0.999,
            op_chunk[None, None, :] * mx.power(base, bt_chunk[None, None, :]),
        )
        alpha = mx.where(sigma < 1.0, alpha, mx.zeros_like(alpha))
        one_minus = 1.0 - alpha
        k = alpha.shape[-1]
        if k > 1:
            cp = mx.cumprod(one_minus[..., :-1], axis=-1)
            T_within = mx.concatenate([mx.ones((height, width, 1)), cp], axis=-1)
        else:
            T_within = mx.ones((height, width, 1))
        w = alpha * T_within * T[..., None]
        new_image = image + w @ col_chunk
        new_T = T * mx.prod(one_minus, axis=-1)
        return new_image, new_T

    _chunk_ckpt = mx.checkpoint(_one_chunk) if checkpoint else _one_chunk

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        image, T = _chunk_ckpt(
            image, T,
            means_2d[start:end], conic[start:end],
            opacity[start:end], beta[start:end], colors[start:end],
        )
        # Early terminate if all pixels are opaque. Note: reading T.max()
        # forces a materialization and severs any pending lazy work — do this
        # only outside the inner accumulation graph.
        if trans_eps > 0 and float(T.max().item()) < trans_eps:
            break

    # 5. Composite background under the accumulated color.
    image = image + T[..., None] * background[None, None, :]
    alpha_out = 1.0 - T
    return image, alpha_out
