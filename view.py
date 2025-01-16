import time

import nerfview
import torch
import viser

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import GaussianModel


def viewing(args):
    gaussians = GaussianModel(args.sh_degree, args.sb_number)
    gaussians.load_ply(args.ply)
    raw_betas = gaussians._beta
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    gaussians.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = nerfview.Viewer(
        server=server,
        render_fn=lambda camera_state, img_wh: (
            lambda mask: gaussians.view(camera_state, img_wh, gui_dropdown.value, mask)
        )(
            torch.logical_and(
                gaussians._beta >= gui_multi_slider.value[0],
                gaussians._beta <= gui_multi_slider.value[1],
            ).squeeze()
        ),
        mode="rendering",
    )
    with server.gui.add_folder("Geometry Complexity Control"):
        beta_stats = f"""<sub>
                    Beta Stats:\\
                    Beta Min: {raw_betas.min():.2f}\\
                    Beta Max: {raw_betas.max():.2f}\\
                    Beta Mean: {raw_betas.mean():.2f}\\
                    Beta Std: {raw_betas.std():.2f}
                    </sub>"""
        server.gui.add_markdown(beta_stats)
        gui_multi_slider = server.gui.add_multi_slider(
            "Beta Range",
            min=raw_betas.min().floor().item(),
            max=raw_betas.max().ceil().item(),
            step=0.01,
            initial_value=(
                raw_betas.min().floor().item(),
                raw_betas.max().ceil().item(),
            ),
        )
        gui_multi_slider.on_update(viewer.rerender)
    with server.gui.add_folder("Render Mode"):
        gui_dropdown = server.gui.add_dropdown(
            "Mode",
            ["RGB", "Diffuse", "Specular", "Depth"],
            initial_value="RGB",
        )
        gui_dropdown.on_update(viewer.rerender)
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    parser = ArgumentParser(description="Viewing script parameters")
    ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument(
        "--ply", type=str, required=True, default=None, help="path to the .ply file"
    )

    args = parser.parse_args()

    viewing(args)
