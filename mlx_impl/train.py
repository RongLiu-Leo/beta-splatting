"""MLX training loop for DBS on a NeRF-synthetic scene.

Mirror of ../train.py but MLX-native. Uses:
- mlx_impl.beta_model.BetaModel (state + activations + MCMC densification)
- mlx_impl.geometry.projection.project (world→cam→2D + covariance)
- mlx_impl.color.spherical_beta.spherical_beta_forward (color eval)
- mlx_impl.color.spherical_harmonics.sh_forward (SH residual — degree 0 skips it)
- mlx_impl.rasterizer.slow.rasterize_soft (Track A pure-MLX renderer)
- mlx_impl.optimizer.MutableAdam (growable optimizer state)
- mlx_impl.losses.l1_loss, ssim, psnr

Small by design: 100x100 images, 20k primitives to start, cap_max 100k. This
is a "prove the pipeline converges" run — paper-quality training needs
Track B (Metal shaders) for the required throughput.

Run: `python -m mlx_impl.train`
Optional args parsed via argparse.
"""

from __future__ import annotations
import argparse
import math
import os
import random
import time

import numpy as np
import mlx.core as mx

from mlx_impl.beta_model import BetaModel
from mlx_impl.optimizer import MutableAdam
from mlx_impl.dataset import load_nerf_synthetic, initialize_point_cloud
from mlx_impl.geometry.projection import project
from mlx_impl.color.spherical_beta import spherical_beta_forward
from mlx_impl.color.spherical_harmonics import sh_forward
from mlx_impl.rasterizer.slow import rasterize_soft
from mlx_impl.losses import l1_loss, ssim, psnr
from mlx_impl.densification import apply_position_noise, regularization_loss


def _param_lrs(spatial_lr_scale: float = 1.0):
    """Match arguments/__init__.py:OptimizationParams defaults."""
    return dict(
        xyz=1.6e-4 * spatial_lr_scale,
        sh0=2.5e-3,
        shN=2.5e-3 / 20.0,
        sb_params=2.5e-3,
        opacity=5e-2,
        beta=1e-3,
        scaling=5e-3,
        rotation=1e-3,
    )


def render_view(
    model: BetaModel,
    cam,
    background: mx.array,
    chunk_size: int = 512,
):
    """Full DBS forward: project + color + rasterize. Returns (image, alpha, meta)."""
    # 1. Project every primitive to this view.
    proj = project(
        model._xyz, model.get_scaling, model.get_rotation, cam.viewmat,
        fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
        width=cam.width, height=cam.height,
    )

    # 2. Per-primitive view direction in world space: from primitive to camera.
    # cam center = -R^T @ t of viewmat.
    R = cam.viewmat[:3, :3]
    t = cam.viewmat[:3, 3]
    cam_center = -mx.transpose(R) @ t                    # (3,)
    dirs = model._xyz - cam_center[None, :]              # (N, 3)

    # 3. Color evaluation: SB + SH DC.
    # sh0: (N, 1, C) → (N, C).
    c0 = model._sh0[:, 0, :]
    if model.max_sh_degree > 0 and model.active_sh_degree > 0:
        # Concatenate DC + residual for SH forward.
        shs = model.get_shs                              # (N, K_sh, C)
        sh_color = sh_forward(shs, dirs / (mx.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-20),
                              model.active_sh_degree)
        c0_effective = sh_color                          # SH is standalone; overrides c0
    else:
        c0_effective = c0

    colors = spherical_beta_forward(c0_effective, model.get_sb_params, dirs)
    # Clamp to [0, 1] softly via sigmoid-like — but paper uses raw + softplus on SB
    # intensities; the sum can go above 1. Rasterizer just uses linear composite.

    # 4. Rasterize.
    image, alpha = rasterize_soft(
        proj["means_2d"], proj["conic"], model.get_opacity[:, 0],
        model.get_beta[:, 0], colors,
        cam.width, cam.height, background,
        valid=proj["valid"], depths=proj["depths"],
        chunk_size=chunk_size,
    )
    return image, alpha, proj


def train(args):
    print(f"[cfg] scene={args.scene} downscale={args.downscale} "
          f"init_pts={args.initial_points} cap_max={args.cap_max}")
    print(f"[cfg] iters={args.iterations} lambda_ssim={args.lambda_dssim} "
          f"densify=({args.densify_from},{args.densify_until},every {args.densify_every})")

    # --- Load dataset -----------------------------------------------------
    t0 = time.time()
    cams = load_nerf_synthetic(args.scene, split="train",
                               downscale=args.downscale, white_bg=True,
                               max_cameras=args.max_cameras)
    print(f"[data] loaded {len(cams)} training cameras "
          f"at {cams[0].width}x{cams[0].height} in {time.time()-t0:.1f}s")

    # --- Init model + optimizer ------------------------------------------
    points, colors = initialize_point_cloud(args.initial_points, extent=1.5, seed=args.seed)
    model = BetaModel(sh_degree=args.sh_degree, sb_number=args.sb_number,
                      color_channels=3)
    model.create_from_pcd(points, colors, spatial_lr_scale=1.0)

    opt = MutableAdam(_param_lrs(spatial_lr_scale=1.0))
    opt.init_state(model.parameters())

    background = mx.array([1.0, 1.0, 1.0])       # white bg matches lego alpha-composite
    model.background = background

    # --- Training loop ---------------------------------------------------
    def loss_fn(params_dict, cam):
        model.set_parameters(params_dict)
        image, _, _ = render_view(model, cam, background, chunk_size=args.chunk_size)
        gt = cam.image                              # (H, W, 3)
        l1 = mx.abs(image - gt).mean()
        # SSIM expects NCHW; our images are HWC.
        img_nchw = mx.transpose(image[None], (0, 3, 1, 2))
        gt_nchw = mx.transpose(gt[None], (0, 3, 1, 2))
        ss = ssim(img_nchw, gt_nchw)
        loss = (1.0 - args.lambda_dssim) * l1 + args.lambda_dssim * (1.0 - ss)
        # Regularizers inside densify window (checked by caller — reg=0 outside).
        return loss

    def loss_with_reg(params_dict, cam, opacity_reg, scale_reg):
        model.set_parameters(params_dict)
        image, _, _ = render_view(model, cam, background, chunk_size=args.chunk_size)
        gt = cam.image
        l1 = mx.abs(image - gt).mean()
        img_nchw = mx.transpose(image[None], (0, 3, 1, 2))
        gt_nchw = mx.transpose(gt[None], (0, 3, 1, 2))
        ss = ssim(img_nchw, gt_nchw)
        base = (1.0 - args.lambda_dssim) * l1 + args.lambda_dssim * (1.0 - ss)
        reg = regularization_loss(
            model.get_opacity, model.get_scaling, opacity_reg, scale_reg,
        )
        return base + reg

    grad_fn = mx.value_and_grad(loss_with_reg)

    log_lines = []
    ema_loss = 0.0
    t_start = time.time()

    for it in range(1, args.iterations + 1):
        cam = cams[random.randrange(len(cams))]

        in_window = args.densify_from < it < args.densify_until
        op_reg = args.opacity_reg if in_window else 0.0
        sc_reg = args.scale_reg if in_window else 0.0

        loss_val, grads = grad_fn(model.parameters(), cam, op_reg, sc_reg)

        # Apply Adam step.
        new_params = opt.step(model.parameters(), grads)
        model.set_parameters(new_params)

        loss_scalar = float(loss_val.item())
        ema_loss = 0.4 * loss_scalar + 0.6 * ema_loss if it > 1 else loss_scalar

        # MCMC densification.
        did_densify = False
        if in_window and it % args.densify_every == 0:
            dead_mask = (model.get_opacity <= 0.005).squeeze(-1)
            model.relocate_gs(dead_mask=dead_mask, optimizer=opt)
            added = model.add_new_gs(cap_max=args.cap_max, optimizer=opt)
            model._xyz = apply_position_noise(
                model._xyz, model.get_scaling, model.get_rotation, model.get_opacity,
                xyz_lr=opt.lrs["xyz"], noise_lr=args.noise_lr,
            )
            did_densify = True

        if it % args.log_every == 0 or it == 1:
            elapsed = time.time() - t_start
            n_prim = int(model._xyz.shape[0])
            mem_mb = mx.get_active_memory() / 1e6
            peak_mb = mx.get_peak_memory() / 1e6
            marker = " D" if did_densify else "  "
            line = (f"[iter {it:5d}]{marker} loss={ema_loss:.5f} "
                    f"prim={n_prim:6d} mem={mem_mb:.0f}MB peak={peak_mb:.0f}MB "
                    f"t={elapsed:.1f}s ({it/elapsed:.1f} it/s)")
            print(line, flush=True)
            log_lines.append(line)

    # --- Save PLY --------------------------------------------------------
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save_ply(out_path)
    print(f"[save] {out_path}  ({int(model._xyz.shape[0])} primitives)")

    # --- Eval on training cams (quick) -----------------------------------
    if args.eval:
        psnrs = []
        for cam in cams[:min(20, len(cams))]:
            img, _, _ = render_view(model, cam, background, chunk_size=args.chunk_size)
            img_c = mx.clip(img, 0.0, 1.0)
            p = float(psnr(mx.transpose(img_c[None], (0, 3, 1, 2)),
                           mx.transpose(cam.image[None], (0, 3, 1, 2))))
            psnrs.append(p)
        print(f"[eval] mean PSNR on {len(psnrs)} training views: {np.mean(psnrs):.2f}")

    # --- Save log --------------------------------------------------------
    if args.log_file:
        with open(args.log_file, "w") as f:
            f.write("\n".join(log_lines))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="lego")
    p.add_argument("--downscale", type=int, default=8)          # 100x100 for smoke
    p.add_argument("--initial-points", type=int, default=20000)
    p.add_argument("--cap-max", type=int, default=100000)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--sh-degree", type=int, default=0)
    p.add_argument("--sb-number", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--max-cameras", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--densify-from", type=int, default=300)
    p.add_argument("--densify-until", type=int, default=800)
    p.add_argument("--densify-every", type=int, default=100)
    p.add_argument("--lambda-dssim", type=float, default=0.2)
    p.add_argument("--opacity-reg", type=float, default=0.01)
    p.add_argument("--scale-reg", type=float, default=0.01)
    p.add_argument("--noise-lr", type=float, default=5e4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="mlx_lego.ply")
    p.add_argument("--log-file", default=None)
    p.add_argument("--eval", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    train(args)
