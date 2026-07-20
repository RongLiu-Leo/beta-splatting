"""BetaModel — MLX version.

Direct translation of scene/beta_model.py's parameter storage, activations,
and PLY I/O. Optimizer/densification/rasterizer bindings are stubbed and
land in later phases (see docs/mlx_port_plan.md).

Design notes:
- MLX doesn't need explicit device placement (unified memory). Every
  parameter lives in Metal-addressable memory automatically.
- Parameters are `mx.array` instances stored as attributes. MLX's optimizer
  updates them via a flattened tree.
- Channel count is inferred from tensor shape wherever possible so the same
  code paths handle 3c (RGB) and 4c (RGBA) without preprocessor branches.

For the training loop (docs/mlx_port_plan.md Phase 4), this class exposes
`.parameters()` (a dict) and `.set_parameters(dict)` — matching MLX's
optimizer contract.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import numpy as np
import mlx.core as mx
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors


# --- Activations (pure functions; MLX arrays) --------------------------------

def _softplus(x: mx.array, beta: float = 1.0) -> mx.array:
    """Numerically stable softplus with slope parameter beta.

    Matches torch.nn.functional.softplus(x, beta): (1/beta) * log(1 + exp(beta * x))
    """
    return (1.0 / beta) * mx.log1p(mx.exp(beta * x))


def _inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    """Inverse of sigmoid. Applied at init time on numpy inputs."""
    return np.log(x / (1.0 - x))


def _rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    """RGB → SH DC term. Matches utils/sh_utils.py RGB2SH."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


# --- Model -------------------------------------------------------------------

class BetaModel:
    """MLX BetaModel. Mirrors scene/beta_model.py:BetaModel.

    Parameters (per primitive N, K = sb_number lobes, C = color_channels):
        _xyz         : (N, 3)             positions
        _sh0         : (N, 1, C)          SH DC term
        _shN         : (N, K_sh - 1, C)   SH residual (K_sh = (max_sh_degree+1)**2)
        _sb_params   : (N, K, C + 3)      SB lobes [color..., theta, phi, beta]
        _opacity     : (N, 1)             pre-sigmoid geometric opacity
        _beta        : (N, 1)             pre-activation geometry shape (via 4*exp)
        _scaling     : (N, 3)             pre-exp scales
        _rotation    : (N, 4)             pre-normalize quaternions

    In 3-channel mode C=3. In 4-channel mode C=4 (alpha rides on the color
    parameters — see docs/rgba_extension.md).
    """

    def __init__(
        self,
        sh_degree: int = 0,
        sb_number: int = 2,
        color_channels: int = 3,
    ):
        assert color_channels in (3, 4), "color_channels must be 3 or 4"
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.sb_number = sb_number
        self.color_channels = color_channels

        # Empty placeholders. Populated by create_from_pcd or load_ply.
        self._xyz: mx.array = mx.zeros((0, 3))
        self._sh0: mx.array = mx.zeros((0, 1, color_channels))
        self._shN: mx.array = mx.zeros((0, 0, color_channels))
        self._sb_params: mx.array = mx.zeros((0, sb_number, color_channels + 3))
        self._scaling: mx.array = mx.zeros((0, 3))
        self._rotation: mx.array = mx.zeros((0, 4))
        self._opacity: mx.array = mx.zeros((0, 1))
        self._beta: mx.array = mx.zeros((0, 1))

        self.background: mx.array = mx.zeros((color_channels,))
        self.spatial_lr_scale: float = 0.0

    # --- activations exposed as properties ---------------------------------

    @property
    def get_xyz(self) -> mx.array:
        return self._xyz

    @property
    def get_scaling(self) -> mx.array:
        return mx.exp(self._scaling)

    @property
    def get_rotation(self) -> mx.array:
        # Normalize quaternions to unit length.
        norm = mx.rsqrt((self._rotation * self._rotation).sum(axis=-1, keepdims=True) + 1e-20)
        return self._rotation * norm

    @property
    def get_opacity(self) -> mx.array:
        return mx.sigmoid(self._opacity)

    @property
    def get_beta(self) -> mx.array:
        return 4.0 * mx.exp(self._beta)

    @property
    def get_shs(self) -> mx.array:
        """Concatenate SH DC + residual along the (K_sh) axis."""
        return mx.concatenate([self._sh0, self._shN], axis=1)

    @property
    def get_sb_params(self) -> mx.array:
        """Apply the softplus to the color slots of SB params.

        Matches sb_params_activation in scene/beta_model.py:47-50 — softplus with
        beta = log(2) * 10, applied only to the color channels, θ/φ/β pass through.
        """
        C = self.color_channels
        colors = _softplus(self._sb_params[..., :C], beta=math.log(2) * 10)
        rest = self._sb_params[..., C:]
        return mx.concatenate([colors, rest], axis=-1)

    # --- construction ------------------------------------------------------

    def create_from_pcd(self, points: np.ndarray, colors: np.ndarray, spatial_lr_scale: float):
        """Initialize from a point cloud (numpy arrays).

        points : (N, 3), colors : (N, 3) or (N, 4) in [0, 1].
        Matches scene/beta_model.py:create_from_pcd, but MLX-native.
        """
        self.spatial_lr_scale = spatial_lr_scale
        C = self.color_channels
        N = points.shape[0]
        print(f"Number of points at initialisation : {N}")

        # SH init: DC term from RGB. If 4c and colors is 3c, default alpha=1.
        if colors.shape[-1] == 3 and C == 4:
            alpha_col = np.ones((N, 1), dtype=np.float32)
            colors = np.concatenate([colors, alpha_col], axis=-1)
        elif colors.shape[-1] == 4 and C == 3:
            colors = colors[..., :3]

        fused_color = _rgb_to_sh(colors.astype(np.float32))  # (N, C)

        K_sh = (self.max_sh_degree + 1) ** 2
        shs = np.zeros((N, K_sh, C), dtype=np.float32)
        shs[:, 0, :] = fused_color  # DC term
        # Higher-order SH stays zero at init (matches PyTorch path).

        # Nearest-neighbor distances → initial isotropic scale.
        nn_model = NearestNeighbors(n_neighbors=4, metric="euclidean").fit(points)
        dists, _ = nn_model.kneighbors(points)
        dist2 = (dists[:, 1:] ** 2).mean(axis=-1)
        scales = np.log(np.sqrt(dist2 + 1e-20))[:, None].repeat(3, axis=1).astype(np.float32)

        rots = np.zeros((N, 4), dtype=np.float32)
        rots[:, 0] = 1.0

        opacities = _inverse_sigmoid(0.5 * np.ones((N, 1), dtype=np.float32))
        betas = np.zeros_like(opacities)

        # SB params: (N, K, C + 3) = [color_channels..., theta, phi, beta].
        sb_params = np.zeros((N, self.sb_number, C + 3), dtype=np.float32)
        # For RGBA, alpha init is 0 pre-softplus → post-softplus roughly 0.7 with
        # our beta=log(2)*10 slope — matches the RGB init behavior; alpha starts
        # non-trivial but sub-1 and the model can push it up.
        rng = np.random.default_rng(0)
        theta = np.pi * rng.random((N, self.sb_number)).astype(np.float32)
        phi = 2 * np.pi * rng.random((N, self.sb_number)).astype(np.float32)
        sb_params[:, :, C] = theta       # slot layout: [...C], theta, phi, beta
        sb_params[:, :, C + 1] = phi

        # Wrap in MLX arrays. No `.requires_grad_(True)` — mx.grad picks up any
        # array in the parameter dict.
        self._xyz = mx.array(points.astype(np.float32))
        self._sh0 = mx.array(shs[:, 0:1, :])
        self._shN = mx.array(shs[:, 1:, :])
        self._sb_params = mx.array(sb_params)
        self._scaling = mx.array(scales)
        self._rotation = mx.array(rots)
        self._opacity = mx.array(opacities)
        self._beta = mx.array(betas)

    # --- parameter registry (for the MLX optimizer, later phase) ----------

    def parameters(self) -> Dict[str, mx.array]:
        return {
            "xyz": self._xyz,
            "sh0": self._sh0,
            "shN": self._shN,
            "sb_params": self._sb_params,
            "opacity": self._opacity,
            "beta": self._beta,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }

    def set_parameters(self, params: Dict[str, mx.array]):
        self._xyz = params["xyz"]
        self._sh0 = params["sh0"]
        self._shN = params["shN"]
        self._sb_params = params["sb_params"]
        self._opacity = params["opacity"]
        self._beta = params["beta"]
        self._scaling = params["scaling"]
        self._rotation = params["rotation"]

    def prune(self, live_mask: mx.array):
        """Keep only primitives where live_mask is True."""
        # MLX supports boolean indexing.
        self._xyz = self._xyz[live_mask]
        self._sh0 = self._sh0[live_mask]
        self._shN = self._shN[live_mask]
        self._sb_params = self._sb_params[live_mask]
        self._scaling = self._scaling[live_mask]
        self._rotation = self._rotation[live_mask]
        self._opacity = self._opacity[live_mask]
        self._beta = self._beta[live_mask]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # --- PLY I/O -----------------------------------------------------------
    # I/O runs on numpy through plyfile; MLX arrays are converted at the
    # boundary. Compatible with CUDA-trained PLYs since the on-disk layout
    # matches scene/beta_model.py's construct_list_of_attributes().

    def _attribute_names(self) -> list:
        names = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._sh0.shape[1] * self._sh0.shape[2]):
            names.append(f"sh0_{i}")
        for i in range(self._shN.shape[1] * self._shN.shape[2]):
            names.append(f"shN_{i}")
        for i in range(self._sb_params.shape[1] * self._sb_params.shape[2]):
            names.append(f"sb_params_{i}")
        names.append("opacity")
        names.append("beta")
        for i in range(self._scaling.shape[1]):
            names.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            names.append(f"rot_{i}")
        return names

    def save_ply(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = np.array(self._xyz)
        normals = np.zeros_like(xyz)
        # CUDA layout: sh0 stored transposed to (N, C, K), flattened.
        # For MLX we hold (N, K, C), so match CUDA byte layout on save.
        sh0 = np.array(mx.transpose(self._sh0, (0, 2, 1))).reshape(xyz.shape[0], -1)
        shN = np.array(mx.transpose(self._shN, (0, 2, 1))).reshape(xyz.shape[0], -1)
        sb_params = np.array(mx.transpose(self._sb_params, (0, 2, 1))).reshape(xyz.shape[0], -1)
        opacity = np.array(self._opacity)
        beta = np.array(self._beta)
        scale = np.array(self._scaling)
        rotation = np.array(self._rotation)

        dtype_full = [(a, "f4") for a in self._attribute_names()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, sh0, shN, sb_params, opacity, beta, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path: str):
        """Load a CUDA-trained PLY into this MLX model.

        Assumes the on-disk layout produced by scene/beta_model.py:save_ply().
        Channel count is inferred from the number of `sh0_*` fields:
            #sh0 fields = 1 * C     → C = #sh0 fields
        We validate against self.color_channels.
        """
        plydata = PlyData.read(path)
        v = plydata["vertex"]
        N = v.count
        xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1)

        sh0_names = sorted([n for n in v.data.dtype.names if n.startswith("sh0_")],
                           key=lambda s: int(s.split("_")[1]))
        shN_names = sorted([n for n in v.data.dtype.names if n.startswith("shN_")],
                           key=lambda s: int(s.split("_")[1]))
        sb_names = sorted([n for n in v.data.dtype.names if n.startswith("sb_params_")],
                          key=lambda s: int(s.split("_")[2]))

        C = len(sh0_names) // 1  # sh0 has K=1 element per channel
        if C != self.color_channels:
            raise ValueError(
                f"PLY has {C} color channels but model was constructed with "
                f"color_channels={self.color_channels}. Reconstruct the model to match."
            )
        K_shN = len(shN_names) // C
        K_sb_slot = len(sb_names) // self.sb_number  # per-lobe slot count
        assert K_sb_slot == C + 3, (
            f"SB slot count mismatch: PLY has {K_sb_slot} per lobe, expected {C + 3}."
        )

        sh0 = np.stack([np.asarray(v[n]) for n in sh0_names], axis=1).reshape(N, C, 1)
        shN_flat = np.stack([np.asarray(v[n]) for n in shN_names], axis=1)
        shN = shN_flat.reshape(N, C, K_shN)
        sb_flat = np.stack([np.asarray(v[n]) for n in sb_names], axis=1)
        sb = sb_flat.reshape(N, K_sb_slot, self.sb_number)  # CUDA transposes to (N, slot, K)

        opacity = np.asarray(v["opacity"])[:, None]
        beta = np.asarray(v["beta"])[:, None]
        scale = np.stack([np.asarray(v[f"scale_{i}"]) for i in range(3)], axis=1)
        rot = np.stack([np.asarray(v[f"rot_{i}"]) for i in range(4)], axis=1)

        # Undo the CUDA transpose to get MLX layout (N, K, C) / (N, K, slot).
        self._xyz = mx.array(xyz.astype(np.float32))
        self._sh0 = mx.array(np.transpose(sh0, (0, 2, 1)).astype(np.float32))
        self._shN = mx.array(np.transpose(shN, (0, 2, 1)).astype(np.float32))
        self._sb_params = mx.array(np.transpose(sb, (0, 2, 1)).astype(np.float32))
        self._opacity = mx.array(opacity.astype(np.float32))
        self._beta = mx.array(beta.astype(np.float32))
        self._scaling = mx.array(scale.astype(np.float32))
        self._rotation = mx.array(rot.astype(np.float32))
        self.active_sh_degree = self.max_sh_degree
