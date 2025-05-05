#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint

from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim
import sys
from scene import Scene, BetaModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, ViewerParams, OptimizationParams
from scene.beta_model import build_scaling_rotation
import viser
from scene.beta_viewer import BetaViewer
import time
import json


def training(args):
    first_iter = 0
    prepare_output_and_logger(args)
    beta_model = BetaModel(args.sh_degree, args.sb_number)
    scene = Scene(args, beta_model)
    beta_model.training_setup(args)
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        beta_model.restore(model_params, args)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not args.disable_viewer:
        server = viser.ViserServer(port=args.port, verbose=False)
        viewer = BetaViewer(
            server=server,
            render_fn=lambda camera_state, render_tab_state: (
                lambda mask: beta_model.view(
                    camera_state, render_tab_state, viewer.gui_dropdown.value, mask
                )
            )(
                torch.logical_and(
                    beta_model._beta >= viewer.gui_multi_slider.value[0],
                    beta_model._beta <= viewer.gui_multi_slider.value[1],
                ).squeeze()
            ),
            mode="training",
        )

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    # Patience-related variables for evaluation mode
    patience = 20
    patience_counter = 0

    # Initialize iteration and progress_bar for logging
    iteration = first_iter + 1
    if args.cap_max < beta_model._xyz.shape[0]:
        print(
            f"Warning: cap_max ({args.cap_max}) is smaller than the number of points initialized ({beta_model._xyz.shape[0]}). Resetting cap_max to the number of points initialized."
        )
        args.cap_max = beta_model._xyz.shape[0]
    if not args.eval:
        progress_bar = tqdm(
            range(first_iter, args.iterations), desc="Training progress"
        )
    else:
        progress_bar = tqdm(desc="Training progress")

    while True:
        # For non-eval mode, break when reaching the specified iterations
        if not args.eval and iteration > args.iterations:
            break

        iter_start.record()
        if not args.disable_viewer:
            while viewer.state == "paused":
                time.sleep(0.01)
            viewer.lock.acquire()
            tic = time.time()

        xyz_lr = beta_model.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            beta_model.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        beta_model.background = (
            torch.rand((3), device="cuda") if args.random_background else background
        )
        render_pkg = beta_model.render(viewpoint_cam)
        image = render_pkg["render"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (
            1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        )
        if args.densify_from_iter < iteration < args.densify_until_iter:
            loss += args.opacity_reg * torch.abs(beta_model.get_opacity).mean()
            loss += args.scale_reg * torch.abs(beta_model.get_scaling).mean()

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            progress_bar.set_postfix(
                {
                    "Iter": iteration,
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "Beta": f"{beta_model._beta.mean().item():.2f}",
                }
            )
            progress_bar.update(1)

            if iteration in args.save_iterations and not args.eval:
                print(f"\n[ITER {iteration}] Saving beta_model")
                scene.save(iteration)

            if (
                iteration < args.densify_until_iter
                and iteration > args.densify_from_iter
                and iteration % args.densification_interval == 0
            ):
                dead_mask = (beta_model.get_opacity <= 0.005).squeeze(-1)
                beta_model.relocate_gs(dead_mask=dead_mask)
                beta_model.add_new_gs(cap_max=args.cap_max)

                L = build_scaling_rotation(
                    beta_model.get_scaling, beta_model.get_rotation
                )
                actual_covariance = L @ L.transpose(1, 2)

                noise = (
                    torch.randn_like(beta_model._xyz)
                    * (torch.pow(1 - beta_model.get_opacity, 100))
                    * args.noise_lr
                    * xyz_lr
                )
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                beta_model._xyz.add_(noise)

            beta_model.optimizer.step()
            beta_model.optimizer.zero_grad(set_to_none=True)

            if not args.disable_viewer:
                num_train_rays_per_step = (
                    gt_image.numel()
                )  # Total number of rays in the image
                viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic + 1e-8)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                viewer.update(iteration, num_train_rays_per_step)

            # Patience-based best model saving in eval mode
            if args.eval and iteration % 500 == 0 and iteration >= 15_000:
                if scene.save_best_model():
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping.")
                    break

        iteration += 1

    progress_bar.close()

    print("\nTraining complete.\n")

    if args.eval:
        print("\nEvaluating Best Model Performance\n")
        scene = Scene(args, beta_model, "best")
        result = scene.eval()
        with open(
            os.path.join(scene.model_path, "point_cloud/iteration_best/metrics.json"),
            "w",
        ) as f:
            json.dump(result, f, indent=True)

    if args.compress:
        print("Compressing model...")
        beta_model.save_png(
            os.path.join(scene.model_path, "point_cloud/iteration_best")
        )

    if not args.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", os.path.basename(args.source_path))

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    ModelParams(parser), OptimizationParams(parser), ViewerParams(parser)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--compress", type=bool, default=True)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    training(args)
