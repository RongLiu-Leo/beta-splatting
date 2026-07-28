"""Loss functions in MLX.

Drop-in replacements for utils/loss_utils.py + fused_ssim. L1 is trivial;
SSIM is a direct translation of the standard Gaussian-window SSIM from
utils/loss_utils.py — we do NOT depend on the CUDA fused_ssim package.

For inputs in NCHW layout, matching the existing training code.
"""

from __future__ import annotations
from math import exp

import mlx.core as mx


def l1_loss(pred: mx.array, target: mx.array) -> mx.array:
    return mx.abs(pred - target).mean()


def l2_loss(pred: mx.array, target: mx.array) -> mx.array:
    return ((pred - target) ** 2).mean()


def _gaussian_1d(window_size: int, sigma: float) -> mx.array:
    center = window_size // 2
    vals = [exp(-((x - center) ** 2) / (2 * sigma ** 2)) for x in range(window_size)]
    w = mx.array(vals, dtype=mx.float32)
    return w / w.sum()


def _gaussian_2d_window(window_size: int, channel: int) -> mx.array:
    """Return (channel, 1, window_size, window_size) Gaussian window."""
    g1 = _gaussian_1d(window_size, 1.5)                      # (W,)
    g2 = g1[:, None] * g1[None, :]                           # (W, W)
    w = g2[None, None, :, :]                                 # (1, 1, W, W)
    return mx.broadcast_to(w, (channel, 1, window_size, window_size))


def ssim(
    img1: mx.array,        # (N, C, H, W)  or  (C, H, W)
    img2: mx.array,        # same shape
    window_size: int = 11,
    size_average: bool = True,
) -> mx.array:
    """Structural similarity, matching utils/loss_utils.py math."""
    if img1.ndim == 3:
        img1 = img1[None, :, :, :]
        img2 = img2[None, :, :, :]

    channel = img1.shape[1]
    window = _gaussian_2d_window(window_size, channel)  # (C, 1, W, W)
    pad = window_size // 2

    # MLX conv2d expects (N, H, W, C_in) input, (C_out, KH, KW, C_in/groups) weight.
    # We use groups=channel (depthwise). Reshape img to NHWC and window to (C, KH, KW, 1).
    img1_hwc = mx.transpose(img1, (0, 2, 3, 1))
    img2_hwc = mx.transpose(img2, (0, 2, 3, 1))
    weight = mx.transpose(window, (0, 2, 3, 1))  # (C, KH, KW, 1)

    def dconv(x: mx.array) -> mx.array:
        return mx.conv2d(x, weight, stride=1, padding=pad, groups=channel)

    mu1 = dconv(img1_hwc)
    mu2 = dconv(img2_hwc)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = dconv(img1_hwc * img1_hwc) - mu1_sq
    sigma2_sq = dconv(img2_hwc * img2_hwc) - mu2_sq
    sigma12 = dconv(img1_hwc * img2_hwc) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(axis=(1, 2, 3))


def psnr(img1: mx.array, img2: mx.array) -> mx.array:
    """PSNR assuming inputs in [0, 1]. Match utils/image_utils.py."""
    mse = ((img1 - img2) ** 2).mean(axis=(-3, -2, -1))
    return 20.0 * mx.log10(1.0 / mx.sqrt(mse + 1e-20))
