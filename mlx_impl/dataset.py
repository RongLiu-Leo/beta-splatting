"""NeRF-synthetic dataset loader for MLX training.

Loads transforms_train.json + PNG frames from a NeRF-synthetic scene folder
like ../lego/. Returns cameras (viewmat, K, W, H) + images pre-composited
over a white background.

Simplified compared to scene/dataset_readers.py: no COLMAP, no test/train
split via file paths (just uses transforms_train.json for training and
transforms_test.json for eval), no LOD, no camera intrinsics variation.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import os
import math
from typing import List
import numpy as np
from PIL import Image
import mlx.core as mx


@dataclass
class Camera:
    viewmat: mx.array   # (4, 4) world → cam, row-major
    K: mx.array         # (3, 3) intrinsics
    width: int
    height: int
    image: mx.array     # (H, W, 3) RGB in [0,1], alpha-composited over bg
    fx: float
    fy: float
    cx: float
    cy: float


def _load_image(path: str, target_wh: tuple | None, white_bg: bool) -> np.ndarray:
    """Return an RGB image in [0, 1] with alpha composited over the bg."""
    im = Image.open(path)
    if target_wh is not None and im.size != target_wh:
        im = im.resize(target_wh, Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255.0
    if arr.shape[-1] == 4:
        rgb = arr[..., :3]
        alpha = arr[..., 3:4]
        bg = 1.0 if white_bg else 0.0
        arr = rgb * alpha + bg * (1.0 - alpha)
    return arr  # (H, W, 3)


def load_nerf_synthetic(
    root: str,
    split: str = "train",
    downscale: int = 4,
    white_bg: bool = True,
    max_cameras: int | None = None,
) -> List[Camera]:
    """Load a NeRF-synthetic scene. Returns list of Camera.

    downscale: divide image resolution by this (both dims).
    max_cameras: limit for smoke runs.
    """
    js = os.path.join(root, f"transforms_{split}.json")
    with open(js) as f:
        data = json.load(f)
    cam_angle_x = data["camera_angle_x"]

    # We need the original image size to compute intrinsics.
    frames = data["frames"]
    if max_cameras is not None:
        frames = frames[:max_cameras]

    # Read one image to get dims.
    first_path = os.path.join(root, frames[0]["file_path"] + ".png")
    with Image.open(first_path) as im:
        W0, H0 = im.size
    W, H = W0 // downscale, H0 // downscale
    fx = 0.5 * W / math.tan(0.5 * cam_angle_x)
    fy = fx
    cx = W / 2.0
    cy = H / 2.0
    K_np = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    K = mx.array(K_np)

    cams: List[Camera] = []
    for fr in frames:
        img_path = os.path.join(root, fr["file_path"] + ".png")
        arr = _load_image(img_path, (W, H), white_bg)                 # (H, W, 3)

        # NeRF-synthetic transform_matrix is c2w in OpenGL convention
        # (camera looks down -z). We want a viewmat in world→cam with the
        # gsplat convention (camera looks down +z, +y down). So we:
        #   1. Invert c2w → w2c
        #   2. Flip y and z axes to match the gsplat/CUDA convention
        # This matches what scene/cameras.py does in the reference.
        c2w = np.array(fr["transform_matrix"], dtype=np.float32)
        # OpenGL → OpenCV/COLMAP camera axes: flip Y and Z.
        c2w[:3, 1:3] *= -1.0
        w2c = np.linalg.inv(c2w)
        viewmat = mx.array(w2c.astype(np.float32))

        cams.append(Camera(
            viewmat=viewmat, K=K, width=W, height=H,
            image=mx.array(arr),
            fx=fx, fy=fy, cx=cx, cy=cy,
        ))
    return cams


def initialize_point_cloud(num_points: int, extent: float = 1.5, seed: int = 0):
    """Random uniform init inside [-extent, extent]^3 (NeRF-synthetic bounds).

    Returns (points, colors) numpy arrays for BetaModel.create_from_pcd.
    """
    rng = np.random.default_rng(seed)
    points = (rng.random((num_points, 3)).astype(np.float32) * 2 - 1) * extent
    colors = rng.random((num_points, 3)).astype(np.float32) * 0.5 + 0.25  # gray-ish
    return points, colors
