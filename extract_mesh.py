from scene import Scene, BetaModel
from argparse import ArgumentParser
from arguments import ModelParams, ViewerParams
import sys
import torch

def test_mesh_extraction(args):
    ply_file = args.ply #r'D:\Work\GS_Research\beta-splatting\output\lego\point_cloud\iteration_best\point_cloud.ply'
    beta_model = BetaModel(args.sh_degree, args.sb_number)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scene = Scene(args, beta_model)

    beta_model.load_ply(args.ply)

    train_cameras = scene.getTrainCameras()
    for cam in train_cameras:
        render_pkg = beta_model.render(cam, render_mode="RGB+D")
        pass

    pass


if __name__ == '__main__':
    parser = ArgumentParser(description="Viewing script parameters")
    parser.add_argument("--ply", type=str, default=None, help="path to the .ply file")
    ModelParams(parser)
    args = parser.parse_args(sys.argv[1:])
    test_mesh_extraction(args)