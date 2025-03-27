import imageio.v2 as imageio
import os
import torch
import numpy as np
from torch import Tensor

def compress_png(compress_dir: str, param_name: str, params:Tensor, n_sidelen: int):
    grid = params.reshape((n_sidelen, n_sidelen, -1)).squeeze()
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)

    img = (grid_norm.detach().cpu().numpy() * (2**8 - 1)).round().astype(np.uint8)

    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta
    

def decompress_png(compress_dir: str, param_name: str, meta: dict):
    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img_norm = img / (2**8 - 1)

    mins = np.array(meta["mins"], dtype=np.float32)
    maxs = np.array(meta["maxs"], dtype=np.float32)
    grid = img_norm * (maxs - mins) + mins

    params = np.reshape(grid, meta["shape"])
    params = params.astype(meta["dtype"])
    return params
