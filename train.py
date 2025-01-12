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
from os import makedirs
import json
import torch
from random import randint

import torchvision
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation

def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint):

    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sb_number)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))

        loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Beta":f"{gaussians._beta.mean().item():.2f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration % 500 == 0 and iteration > 15_000 and dataset.eval:
                save_best_model(scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                gaussians.relocate_gs(dead_mask=dead_mask)
                gaussians.add_new_gs(cap_max=args.cap_max)

                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)
                
                noise = torch.randn_like(gaussians._xyz) * (torch.pow(1 - gaussians.get_opacity, 100)) * args.noise_lr * xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if dataset.eval:
        print("\nEvaluating Best Model Performance\n")
        eval(Scene(dataset, gaussians, "best"), render, (pipe, background))

def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", os.path.basename(args.source_path))
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def save_best_model(scene : Scene, renderFunc, renderArgs):
    torch.cuda.empty_cache()
    psnr_test = 0.0
    test_view_stack = scene.getTestCameras()
    for idx, viewpoint in enumerate(test_view_stack):
        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        psnr_test += psnr(image, gt_image).mean()
    psnr_test /= len(test_view_stack)
    if psnr_test > scene.best_psnr:
        print(f"save best model. PSNR: {psnr_test}")
        scene.save("best")
        scene.best_psnr = psnr_test
    torch.cuda.empty_cache()

def eval(scene : Scene, renderFunc, renderArgs):
    gt_path = os.path.join(args.model_path, "point_cloud/iteration_best/gt")
    render_path = os.path.join(args.model_path, "point_cloud/iteration_best/render")
    makedirs(gt_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    with torch.no_grad():
        torch.cuda.empty_cache()
        psnr_test = 0.0
        ssim_test = 0.0
        lpips_test = 0.0
        test_view_stack = scene.getTestCameras()
        for idx, viewpoint in enumerate(test_view_stack):
            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt_image, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
            psnr_test += psnr(image, gt_image).mean()
            ssim_test += fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean()
            lpips_test += lpips(image, gt_image, net_type='vgg').mean()
        psnr_test /= len(test_view_stack)
        ssim_test /= len(test_view_stack)
        lpips_test /= len(test_view_stack)
        torch.cuda.empty_cache()

        result = {
            "ours_best": {
                "SSIM": ssim_test.item(),
                "PSNR": psnr_test.item(),
                "LPIPS": lpips_test.item()
            }
        }
        with open(os.path.join(args.model_path, "point_cloud/iteration_best/metrics.json"), "w") as f:
            json.dump(result, f, indent=True)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
