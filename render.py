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
from utils.camera_utils import generate_interpolated_path, generate_ellipse_path_z
import numpy as np
import imageio


@torch.no_grad
def rendering(args):
    gaussians = GaussianModel(args.sh_degree, args.sb_number)
    scene = Scene(
        args, gaussians, args.loading_iteration, shuffle=False, center_and_z_up=True
    )

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    gaussians.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cameras = []
    cameras.extend(scene.getTrainCameras() if hasattr(scene, "getTrainCameras") else [])
    cameras.extend(scene.getTestCameras() if hasattr(scene, "getTestCameras") else [])

    output_dir = os.path.join(
        args.model_path, "point_cloud", f"iteration_{args.loading_iteration}", "renders"
    )
    os.makedirs(output_dir, exist_ok=True)

    def render_image(cameras, render_mode):
        for i, cam in enumerate(cameras):
            render_pkg = gaussians.render(cam, render_mode=render_mode)
            image = render_pkg["render"]
            image_cpu = image.clamp(0, 1).cpu().permute(1, 2, 0).numpy()  # [H, W, C]
            im = Image.fromarray((image_cpu * 255).astype("uint8"))
            im.save(os.path.join(output_dir, f"{render_mode}_{i}.png"))
            print(f"Saved {render_mode}_{i}.png")

    def render_video(cameras, render_mode, render_traj_path_type="ellipse", fps=30):
        if render_traj_path_type == "ellipse":
            trajectory_cameras = generate_ellipse_path_z(cameras)
        elif render_traj_path_type == "interp":
            trajectory_cameras = generate_interpolated_path(cameras)
        else:
            raise ValueError(f"Unknown trajectory type: {render_traj_path_type}")

        # Set up video writer
        video_path = os.path.join(output_dir, f"{render_mode}.mp4")
        writer = imageio.get_writer(video_path, fps=fps)

        # Render each frame and write directly to the video
        for i, cam in enumerate(trajectory_cameras):
            render_pkg = gaussians.render(cam, render_mode=render_mode)
            image = render_pkg["render"]
            # Convert image to uint8 format
            image_cpu = image.clamp(0, 1).cpu().permute(1, 2, 0).numpy()  # [H, W, C]
            frame = (image_cpu * 255).astype(np.uint8)
            writer.append_data(frame)
        writer.close()
        print(f"Video saved to {video_path}")

    if args.image:
        print("Rendering images...")
        render_image(cameras, args.render_mode)

    print("Rendering videos...")
    render_video(cameras, args.render_mode, args.render_traj_path)

    print("Rendering complete.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    ModelParams(parser), PipelineParams(parser)

    parser.add_argument(
        "--loading_iteration",
        type=str,
        default="best",
        help="Loading iteration for rendering",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        choices=["RGB", "Diffuse", "Specular"],
        default="RGB",
        help="Rendering mode to use",
    )
    parser.add_argument(
        "--render_traj_path",
        type=str,
        choices=["ellipse", "interp"],
        default="ellipse",
        help="Rendering trajectory path to use",
    )
    parser.add_argument("--image", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    print("Loading model from " + args.model_path)
    rendering(args)
