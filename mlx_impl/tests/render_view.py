"""Render one training view from a trained PLY, save PNG side-by-side with GT.

Usage:
    python -m mlx_impl.tests.render_view --ply out/lego_1k.ply --cam-idx 5
"""

from __future__ import annotations
import argparse
import numpy as np
from PIL import Image
import mlx.core as mx

from mlx_impl.beta_model import BetaModel
from mlx_impl.dataset import load_nerf_synthetic
from mlx_impl.train import render_view


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--scene", default="lego")
    p.add_argument("--downscale", type=int, default=8)
    p.add_argument("--cam-idx", type=int, default=0)
    p.add_argument("--sh-degree", type=int, default=0)
    p.add_argument("--sb-number", type=int, default=2)
    p.add_argument("--output", default="rendered.png")
    args = p.parse_args()

    m = BetaModel(sh_degree=args.sh_degree, sb_number=args.sb_number, color_channels=3)
    m.load_ply(args.ply)
    print(f"loaded {int(m._xyz.shape[0])} primitives")

    cams = load_nerf_synthetic(args.scene, "train", downscale=args.downscale)
    cam = cams[args.cam_idx]
    print(f"camera {args.cam_idx}: {cam.width}x{cam.height}")

    bg = mx.array([1.0, 1.0, 1.0])
    img, alpha, _ = render_view(m, cam, bg, chunk_size=256)
    img = np.clip(np.array(img), 0, 1)
    gt = np.clip(np.array(cam.image), 0, 1)

    # Side-by-side.
    combined = np.concatenate([gt, img], axis=1)
    Image.fromarray((combined * 255).astype(np.uint8)).save(args.output)
    mse = float(((img - gt) ** 2).mean())
    psnr = 20 * np.log10(1 / (np.sqrt(mse) + 1e-12))
    print(f"saved {args.output}  (GT | Rendered)  PSNR={psnr:.2f}")


if __name__ == "__main__":
    main()
