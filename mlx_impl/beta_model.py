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

    def prune(self, live_mask: mx.array, optimizer=None):
        """Keep only primitives where live_mask is True.

        If optimizer is provided (MutableAdam), its state is pruned in step.
        MLX 0.31.2 doesn't support boolean indexing, so we convert to int
        indices via numpy.
        """
        keep = mx.array(np.where(np.array(live_mask))[0].astype(np.int32))
        self._xyz = self._xyz[keep]
        self._sh0 = self._sh0[keep]
        self._shN = self._shN[keep]
        self._sb_params = self._sb_params[keep]
        self._scaling = self._scaling[keep]
        self._rotation = self._rotation[keep]
        self._opacity = self._opacity[keep]
        self._beta = self._beta[keep]
        if optimizer is not None:
            for name in optimizer.state:
                optimizer.prune_state(name, keep)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # --- MCMC densification -------------------------------------------------
    # Ported from scene/beta_model.py (relocate_gs at L729, add_new_gs at L762,
    # _update_params at L700, _sample_alives at L721, densification_postfix at
    # L618, replace_tensors_to_optimizer at L651). Optimizer state grows /
    # shrinks in lockstep with the parameter tensors — this is the usual place
    # MLX ports break.

    def _update_params(self, idxs: mx.array, ratio: mx.array):
        """Gather params at `idxs`, adjust opacity per the MCMC-invariance rule.

        The opacity update `new = 1 - (1 - op)^(1/(ratio+1))` preserves the
        combined alpha contribution when one primitive is split into ratio+1
        copies: (1 - new)^(ratio+1) = 1 - op. This is what makes MCMC
        relocation distribution-preserving in the DBS paper.

        Returns the pre-activation opacity (via inverse_sigmoid) since the
        caller stores raw parameters and applies sigmoid at read time.
        """
        op = self.get_opacity[idxs, 0]                                     # (K,)
        ratio_f = ratio.astype(mx.float32)
        new_op = 1.0 - mx.power(1.0 - op, 1.0 / (ratio_f + 1.0))          # (K,)
        new_op = mx.clip(new_op[..., None], 0.005, 1.0 - 1e-7)             # (K, 1)
        # Store pre-activation.
        new_op_raw = mx.log(new_op / (1.0 - new_op))
        return (
            self._xyz[idxs],
            self._sh0[idxs],
            self._shN[idxs],
            self._sb_params[idxs],
            new_op_raw,
            self._beta[idxs],
            self._scaling[idxs],
            self._rotation[idxs],
        )

    def _sample_alives(
        self,
        probs: mx.array,
        num: int,
        alive_indices: mx.array | None = None,
    ):
        """Multinomial sample of `num` indices weighted by `probs`, with the
        per-sample duplication count (ratio) returned alongside.

        Matches scene/beta_model.py:_sample_alives. `alive_indices`, when
        provided, maps sampled positions back into the model's index space.
        """
        probs = probs / (probs.sum() + 1e-7)
        # mx.random.categorical wants logits over the last axis.
        log_probs = mx.log(probs + 1e-20)
        sampled = mx.random.categorical(log_probs, num_samples=num)        # (num,)
        if alive_indices is not None:
            sampled = alive_indices[sampled]
        # Bincount → per-sample duplication count.
        N = int(self._opacity.shape[0])
        counts = mx.zeros(N, dtype=mx.int32)
        counts[sampled] = counts[sampled] + mx.ones(sampled.shape[0], dtype=mx.int32)
        # ^ scatter-add via the working index-write API verified in tests
        ratio = counts[sampled]
        return sampled, ratio

    def relocate_gs(self, dead_mask: mx.array, optimizer):
        """Replace `dead_mask` primitives with copies of live ones sampled
        proportional to opacity. In-place on the model; optimizer momentum at
        the source indices is reset.

        Matches scene/beta_model.py:relocate_gs (L729).
        """
        dead_count = int(dead_mask.sum().item())
        if dead_count == 0:
            return
        # np.where via numpy for portability of nonzero-index extraction.
        import numpy as np
        dm_np = np.array(dead_mask)
        dead_indices = mx.array(np.where(dm_np)[0].astype(np.int32))
        alive_indices = mx.array(np.where(~dm_np)[0].astype(np.int32))
        if alive_indices.shape[0] == 0:
            return

        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            probs=probs, num=dead_count, alive_indices=alive_indices,
        )

        new_xyz, new_sh0, new_shN, new_sb, new_op, new_beta, new_sc, new_rot = \
            self._update_params(reinit_idx, ratio=ratio)

        # Overwrite dead slots.
        self._xyz[dead_indices] = new_xyz
        self._sh0[dead_indices] = new_sh0
        self._shN[dead_indices] = new_shN
        self._sb_params[dead_indices] = new_sb
        self._opacity[dead_indices] = new_op
        self._beta[dead_indices] = new_beta
        self._scaling[dead_indices] = new_sc
        self._rotation[dead_indices] = new_rot

        # Reference also copies new opacity to source (reinit_idx) — this is
        # the MCMC-invariance step that keeps combined contribution constant.
        self._opacity[reinit_idx] = self._opacity[dead_indices]

        # Reset Adam moments at source indices only (relocation).
        for name in optimizer.state:
            optimizer.reinit_state_at(name, reinit_idx)

    def add_new_gs(self, cap_max: int, optimizer) -> int:
        """Grow primitive count by up to 5%, capped at `cap_max`. Returns the
        number of new primitives added.

        Matches scene/beta_model.py:add_new_gs (L762).
        """
        current = int(self._opacity.shape[0])
        target = min(cap_max, int(1.05 * current))
        num_new = max(0, target - current)
        if num_new <= 0:
            return 0

        probs = self.get_opacity[:, 0]
        add_idx, ratio = self._sample_alives(probs=probs, num=num_new)

        new_xyz, new_sh0, new_shN, new_sb, new_op, new_beta, new_sc, new_rot = \
            self._update_params(add_idx, ratio=ratio)

        # Update the SOURCE primitives' opacity to the invariance-preserving
        # value (source keeps the same shape, gets shared new_op).
        self._opacity[add_idx] = new_op

        # Append new primitives to every tensor.
        self._xyz = mx.concatenate([self._xyz, new_xyz], axis=0)
        self._sh0 = mx.concatenate([self._sh0, new_sh0], axis=0)
        self._shN = mx.concatenate([self._shN, new_shN], axis=0)
        self._sb_params = mx.concatenate([self._sb_params, new_sb], axis=0)
        self._opacity = mx.concatenate([self._opacity, new_op], axis=0)
        self._beta = mx.concatenate([self._beta, new_beta], axis=0)
        self._scaling = mx.concatenate([self._scaling, new_sc], axis=0)
        self._rotation = mx.concatenate([self._rotation, new_rot], axis=0)

        # Grow optimizer state by num_new zeros for every param.
        for name in optimizer.state:
            optimizer.grow_state(name, num_new)

        return num_new

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
