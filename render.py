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
import sys
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from PIL import Image
import time

@torch.no_grad
def rendering(args):
    gaussians = GaussianModel(args.sh_degree, args.sb_number)
    scene = Scene(args, gaussians, args.loading_iteration)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    gaussians.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cameras = scene.getTrainCameras() if hasattr(scene, 'getTrainCameras') else []
    test_cameras = scene.getTestCameras() if hasattr(scene, 'getTestCameras') else []

    output_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{args.loading_iteration}", "renders")
    os.makedirs(output_dir, exist_ok=True)

    def render_and_save(cameras, prefix, render_mode):
        for i, cam in enumerate(cameras):
            # Render the scene for the current camera
            render_pkg = gaussians.render(cam, render_mode=render_mode)
            image = render_pkg["render"]
            # Convert to CPU and save image using an image library like PIL or OpenCV
            # Assuming image is a torch tensor with shape [C, H, W] and values [0,1]
            image_cpu = image.clamp(0, 1).cpu().permute(1, 2, 0).numpy()  # [H, W, C]
            im = Image.fromarray((image_cpu * 255).astype('uint8'))
            im.save(os.path.join(output_dir, f"{prefix}_{render_mode}_{i}.png"))
            print(f"Saved {prefix}_{render_mode}_{i}.png")

    # Render and save training view images
    if train_cameras and not args.skip_train_view:
        print("Rendering training views...")
        render_and_save(train_cameras, "train", args.render_mode)

    # Render and save testing view images
    if test_cameras:
        print("Rendering testing views...")
        render_and_save(test_cameras, "test", args.render_mode)

    print("Rendering complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    ModelParams(parser), PipelineParams(parser)

    parser.add_argument("--loading_iteration", type=str, default="best", help="Loading iteration for rendering")
    parser.add_argument("--render_mode", type=str, choices=["RGB", "Diffuse", "Specular"], 
                        default="RGB", help="Rendering mode to use")
    parser.add_argument("--skip_train_view", type=bool, default=True, help="Whether skip training view")
    args = parser.parse_args(sys.argv[1:])

    print("Loading model from " + args.model_path)
    rendering(args)
