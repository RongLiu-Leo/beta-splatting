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


def compressing(args):
    beta_model = BetaModel()
    beta_model.load_ply(args.ply)
    start_time = time.time()
    beta_model.save_png(os.path.dirname(args.ply))
    end_time = time.time()
    print(f"Compression time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Compressing script parameters")
    parser.add_argument(
        "--ply", type=str, default=None, help="path to the .ply file"
    )
    args = parser.parse_args(sys.argv[1:])

    print("Compressing " + args.ply)

    compressing(args)
