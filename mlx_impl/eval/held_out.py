"""Held-out eval for a trained MLX-DBS PLY on NeRF-synthetic.

Renders every camera in transforms_test.json and reports mean/best/worst
PSNR + SSIM across the split. Writes a JSON summary + optional side-by-sides
for the worst K views.

Usage:
    python -m mlx_impl.eval.held_out --ply out/lego_v4.ply \
        --scene lego --downscale 8 --split test \
        --out-json eval/lego_v4_test.json --save-worst 3
"""

from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
from PIL import Image
import mlx.core as mx

from mlx_impl.beta_model import BetaModel
from mlx_impl.dataset import load_nerf_synthetic
from mlx_impl.losses import psnr, ssim
from mlx_impl.train import render_view


def _to_nchw(img_hwc: mx.array) -> mx.array:
    return mx.transpose(img_hwc[None], (0, 3, 1, 2))


def evaluate(args) -> dict:
    t0 = time.time()
    print(f"[cfg] ply={args.ply} scene={args.scene} split={args.split} "
          f"downscale={args.downscale}")

    model = BetaModel(sh_degree=args.sh_degree, sb_number=args.sb_number,
                      color_channels=3)
    model.load_ply(args.ply)
    n_prim = int(model._xyz.shape[0])
    print(f"[model] {n_prim} primitives")

    cams = load_nerf_synthetic(args.scene, split=args.split,
                               downscale=args.downscale, white_bg=True,
                               max_cameras=args.max_cameras)
    print(f"[data] {len(cams)} views at {cams[0].width}x{cams[0].height}")

    bg = mx.array([1.0, 1.0, 1.0])

    per_view = []
    for i, cam in enumerate(cams):
        img, _, _ = render_view(model, cam, bg, chunk_size=args.chunk_size)
        img_c = mx.clip(img, 0.0, 1.0)
        gt = cam.image
        img_nchw = _to_nchw(img_c)
        gt_nchw = _to_nchw(gt)
        p = float(psnr(img_nchw, gt_nchw))
        s = float(ssim(img_nchw, gt_nchw))
        per_view.append({"idx": i, "psnr": p, "ssim": s})
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[eval] view {i+1:3d}/{len(cams)}  PSNR={p:.2f}  SSIM={s:.4f}",
                  flush=True)

        # Free per-view arrays before the next projection allocates.
        del img, img_c, img_nchw, gt_nchw

    psnrs = np.array([v["psnr"] for v in per_view])
    ssims = np.array([v["ssim"] for v in per_view])

    summary = {
        "ply": args.ply,
        "scene": args.scene,
        "split": args.split,
        "downscale": args.downscale,
        "n_primitives": n_prim,
        "n_views": len(cams),
        "psnr": {
            "mean": float(psnrs.mean()),
            "median": float(np.median(psnrs)),
            "min": float(psnrs.min()),
            "max": float(psnrs.max()),
        },
        "ssim": {
            "mean": float(ssims.mean()),
            "median": float(np.median(ssims)),
            "min": float(ssims.min()),
            "max": float(ssims.max()),
        },
        "elapsed_s": time.time() - t0,
        "per_view": per_view,
    }

    print("")
    print(f"[summary] mean PSNR {summary['psnr']['mean']:.2f}   "
          f"mean SSIM {summary['ssim']['mean']:.4f}   "
          f"({len(cams)} views, {summary['elapsed_s']:.1f}s)")
    print(f"[summary] PSNR min={summary['psnr']['min']:.2f}  "
          f"max={summary['psnr']['max']:.2f}  "
          f"median={summary['psnr']['median']:.2f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[save] {args.out_json}")

    if args.save_worst > 0:
        os.makedirs(args.image_dir, exist_ok=True)
        worst = sorted(per_view, key=lambda v: v["psnr"])[: args.save_worst]
        for v in worst:
            cam = cams[v["idx"]]
            img, _, _ = render_view(model, cam, bg, chunk_size=args.chunk_size)
            img_np = np.clip(np.array(img), 0, 1)
            gt_np = np.clip(np.array(cam.image), 0, 1)
            side = np.concatenate([gt_np, img_np], axis=1)
            path = os.path.join(
                args.image_dir,
                f"worst_v{v['idx']:03d}_psnr{v['psnr']:.2f}.png",
            )
            Image.fromarray((side * 255).astype(np.uint8)).save(path)
            print(f"[save] {path}")

    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--scene", default="lego")
    p.add_argument("--split", default="test", choices=["train", "test", "val"])
    p.add_argument("--downscale", type=int, default=8)
    p.add_argument("--sh-degree", type=int, default=0)
    p.add_argument("--sb-number", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--max-cameras", type=int, default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--save-worst", type=int, default=0)
    p.add_argument("--image-dir", default="out/eval")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
