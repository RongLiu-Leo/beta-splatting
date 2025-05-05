import os
import torch
import sys
from scene import Scene, BetaModel
from argparse import ArgumentParser
from arguments import ModelParams

def training(args):
    beta_model = BetaModel(args.sh_degree, args.sb_number)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    scene = Scene(args, beta_model)
    ply_path = os.path.join(args.model_path, "point_cloud", "iteration_"+args.iteration, "point_cloud.ply")
    if os.path.exists(ply_path):
        print("Evaluating " + ply_path)
        beta_model.load_ply(ply_path)
        scene.eval()
    png_path = os.path.join(args.model_path, "point_cloud", "iteration_"+args.iteration, "png")
    if os.path.exists(png_path):
        print("Evaluating " + png_path)
        beta_model.load_png(png_path)
        scene.eval()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluating script parameters")
    ModelParams(parser)
    parser.add_argument("--iteration", default="best", type=str, help="Iteration to evaluate")
    args = parser.parse_args(sys.argv[1:])
    args.eval = True

    print("Evaluating " + args.model_path)

    training(args)
