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
import nerfview
import time
import torch.nn.functional as F


def training(args):
    beta_model = BetaModel(args.sh_degree, args.sb_number)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    scene = Scene(args, beta_model)
    beta_model.load_ply(os.path.join(
                    args.model_path,
                    "point_cloud",
                    "iteration_best",
                    "point_cloud.ply",
                ))
    if args.eval:
        scene.eval()
    
    beta_model.save_png(os.path.join(
            args.model_path, "point_cloud/iteration_compress"
        ))

    beta_model.load_png(os.path.join(
                    args.model_path,
                    "point_cloud",
                    "iteration_compress",
                    "png",
                ))

    if args.eval:
        print("\nEvaluating Compress Model Performance\n")
        scene.eval()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Compressing script parameters")
    ModelParams(parser), OptimizationParams(parser), ViewerParams(parser)
    args = parser.parse_args(sys.argv[1:])

    print("Compressing " + args.model_path)

    training(args)
