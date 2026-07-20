"""MutableAdam — an Adam optimizer whose moment tensors we can resize/reset.

MLX ships mlx.optimizers.Adam, but its state initialization is coupled to the
initial parameter shapes. When DBS densifies (relocate_gs / add_new_gs) the
parameter tensors change shape mid-training, and the optimizer state must
follow. So we implement Adam directly with a state dict we own, exposing
resize/reset hooks the densification code calls.

Semantics match torch.optim.Adam with per-parameter lrs and eps=1e-15
(matching the reference training loop in ../train.py:256).

Only implements the pieces the training loop needs — no weight decay, no
amsgrad, no LR scheduling (LR scheduling is handled at the call site via
set_lr(), matching how the reference updates position_lr per iteration).
"""

from __future__ import annotations
from typing import Dict, Tuple
import mlx.core as mx


class MutableAdam:
    def __init__(
        self,
        param_lrs: Dict[str, float],
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-15,
    ):
        self.lrs: Dict[str, float] = dict(param_lrs)
        self.b1, self.b2 = betas
        self.eps = eps
        self.t: int = 0
        # state[name] = {'m': mx.array, 'v': mx.array}
        self.state: Dict[str, Dict[str, mx.array]] = {}

    def init_state(self, params: Dict[str, mx.array]):
        """Initialize m/v to zero-like each parameter."""
        for name, p in params.items():
            self.state[name] = {
                "m": mx.zeros_like(p),
                "v": mx.zeros_like(p),
            }
        # Reset time step when re-initializing.
        self.t = 0

    def set_lr(self, name: str, lr: float):
        self.lrs[name] = lr

    def step(
        self,
        params: Dict[str, mx.array],
        grads: Dict[str, mx.array],
    ) -> Dict[str, mx.array]:
        """One Adam step. Returns updated params dict."""
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        new_params = {}
        for name, p in params.items():
            g = grads.get(name)
            if g is None:
                new_params[name] = p
                continue
            s = self.state[name]
            s["m"] = self.b1 * s["m"] + (1.0 - self.b1) * g
            s["v"] = self.b2 * s["v"] + (1.0 - self.b2) * g * g
            m_hat = s["m"] / bc1
            v_hat = s["v"] / bc2
            new_params[name] = p - self.lrs[name] * m_hat / (mx.sqrt(v_hat) + self.eps)
        return new_params

    # --- Resize / reset hooks called by densification -------------------

    def reinit_state_at(self, name: str, indices: mx.array):
        """Zero m/v at the given indices (rows).

        Called after relocate_gs: the source primitives that were duplicated
        into dead slots get their optimizer momentum reset — matching the
        reference's stored_state[exp_avg][inds] = 0 pattern
        (scene/beta_model.py:675-676).
        """
        s = self.state[name]
        zero_slice = mx.zeros((indices.shape[0],) + s["m"].shape[1:], dtype=s["m"].dtype)
        s["m"][indices] = zero_slice
        s["v"][indices] = zero_slice

    def prune_state(self, name: str, keep_indices: mx.array):
        """Keep only rows at keep_indices. Called by prune()."""
        s = self.state[name]
        s["m"] = s["m"][keep_indices]
        s["v"] = s["v"][keep_indices]

    def grow_state(self, name: str, extra_count: int):
        """Append `extra_count` zero rows to m/v. Called by add_new_gs()."""
        if extra_count <= 0:
            return
        s = self.state[name]
        new_shape = (extra_count,) + s["m"].shape[1:]
        z_m = mx.zeros(new_shape, dtype=s["m"].dtype)
        z_v = mx.zeros(new_shape, dtype=s["v"].dtype)
        s["m"] = mx.concatenate([s["m"], z_m], axis=0)
        s["v"] = mx.concatenate([s["v"], z_v], axis=0)
