import time
import torch
import viser
from argparse import ArgumentParser
from arguments import ModelParams, ViewerParams
from scene import BetaModel
from scene.beta_viewer import BetaViewer


def viewing(args):
    beta_model = BetaModel(args.sh_degree, args.sb_number)
    if args.ply:
        beta_model.load_ply(args.ply)
    elif args.png:
        beta_model.load_png(args.png)
    else:
        raise ValueError("You must provide either a .ply file or a .png folder")
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = BetaViewer(
        server=server,
        render_fn=lambda camera_state, render_tab_state: beta_model.view(
            camera_state, render_tab_state
        ),
        mode="rendering",
    )

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    parser = ArgumentParser(description="Viewing script parameters")
    ModelParams(parser), ViewerParams(parser)
    parser.add_argument("--ply", type=str, default=None, help="path to the .ply file")
    parser.add_argument("--png", type=str, default=None, help="path to the png folder")

    args = parser.parse_args()

    viewing(args)
