"""Standalone tests for MCMC densification — no rasterizer needed.

Verifies:
1. relocate_gs replaces dead slots with copies of live ones; opacity > 0.005 after.
2. add_new_gs grows tensors by 5% (capped at cap_max) and grows optimizer state.
3. MCMC opacity-invariance: (1 - new_op)^(ratio+1) ≈ 1 - source_op.
4. Optimizer state resize/reset behaves correctly.
5. Position noise perturbs positions in-plane with covariance.
6. Regularization loss is finite and > 0.
7. prune() also prunes optimizer state.
8. Adam step converges a simple problem (sanity check on our custom Adam).

Run: `python -m mlx_impl.tests.test_densification`
"""

from __future__ import annotations
import numpy as np
import mlx.core as mx

from mlx_impl.beta_model import BetaModel, _inverse_sigmoid
from mlx_impl.optimizer import MutableAdam
from mlx_impl.densification import (
    apply_position_noise,
    build_scaling_rotation,
    regularization_loss,
)


def _make_model(N=1000, sh_degree=1, sb_number=2, color_channels=3,
                dead_count=100, seed=0) -> BetaModel:
    """Fake-initialize a BetaModel bypassing create_from_pcd (which needs a PCD)."""
    m = BetaModel(sh_degree=sh_degree, sb_number=sb_number, color_channels=color_channels)
    rng = np.random.default_rng(seed)
    m._xyz = mx.array(rng.normal(size=(N, 3)).astype(np.float32))
    m._sh0 = mx.array(rng.normal(size=(N, 1, color_channels)).astype(np.float32) * 0.1)
    K_sh = (sh_degree + 1) ** 2
    m._shN = mx.array(rng.normal(size=(N, K_sh - 1, color_channels)).astype(np.float32) * 0.1)
    m._sb_params = mx.array(rng.normal(size=(N, sb_number, color_channels + 3)).astype(np.float32) * 0.1)
    m._beta = mx.zeros((N, 1))
    m._scaling = mx.array(np.log(0.05 * np.ones((N, 3), dtype=np.float32)))  # scale=0.05

    # Rotations = identity quats [1, 0, 0, 0].
    rots = np.zeros((N, 4), dtype=np.float32)
    rots[:, 0] = 1.0
    m._rotation = mx.array(rots)

    # Opacities: random ~0.5, but first dead_count primitives dead (pre-sigmoid = -100).
    op = _inverse_sigmoid(0.5 * np.ones((N, 1), dtype=np.float32))
    op[:dead_count] = -100.0
    m._opacity = mx.array(op)
    return m


def _param_lrs():
    return dict(xyz=1e-4, sh0=2.5e-3, shN=2.5e-3 / 20, sb_params=2.5e-3,
                opacity=5e-2, beta=1e-3, scaling=5e-3, rotation=1e-3)


def test_relocate():
    m = _make_model(N=1000, dead_count=100)
    opt = MutableAdam(_param_lrs())
    opt.init_state(m.parameters())

    dead_mask = (m.get_opacity <= 0.005).squeeze(-1)
    dead_before = int(dead_mask.sum().item())
    assert dead_before == 100, dead_before

    m.relocate_gs(dead_mask, opt)

    # Size unchanged after relocate.
    assert m._xyz.shape == (1000, 3), m._xyz.shape
    # No slots dead anymore.
    dead_after = int((m.get_opacity <= 0.005).sum().item())
    assert dead_after == 0, f"expected 0 dead after relocate, got {dead_after}"
    # Optimizer state size unchanged.
    assert opt.state["xyz"]["m"].shape == (1000, 3)
    print(f"OK   relocate: {dead_before} dead → 0 dead; N stable at 1000")


def test_add_new_gs_and_optimizer_growth():
    m = _make_model(N=1000, dead_count=0)
    opt = MutableAdam(_param_lrs())
    opt.init_state(m.parameters())

    added = m.add_new_gs(cap_max=1500, optimizer=opt)
    # 1.05 * 1000 = 1050 → 50 new
    assert added == 50, f"expected 50 added, got {added}"
    assert m._xyz.shape == (1050, 3)
    assert m._sh0.shape[0] == 1050
    assert m._shN.shape[0] == 1050
    assert m._sb_params.shape[0] == 1050
    assert m._opacity.shape == (1050, 1)

    # Optimizer state grew for every param.
    for name in ("xyz", "sh0", "shN", "sb_params", "opacity", "beta", "scaling", "rotation"):
        assert opt.state[name]["m"].shape[0] == 1050, (name, opt.state[name]["m"].shape)
        assert opt.state[name]["v"].shape[0] == 1050

    # Cap enforced.
    m2 = _make_model(N=1000, dead_count=0)
    opt2 = MutableAdam(_param_lrs())
    opt2.init_state(m2.parameters())
    added2 = m2.add_new_gs(cap_max=1020, optimizer=opt2)
    assert added2 == 20, added2
    assert m2._xyz.shape == (1020, 3)
    print(f"OK   add_new_gs: +50 with cap 1500, +20 with cap 1020; optimizer state tracked")


def test_mcmc_opacity_invariance():
    """After splitting one primitive into (ratio+1) copies, combined
    contribution should be preserved: (1 - new_op)^(ratio+1) = 1 - old_op."""
    m = _make_model(N=100, dead_count=0)
    # Force known opacity so we can check the arithmetic.
    op_np = np.full((100, 1), _inverse_sigmoid(np.float32(0.3)), dtype=np.float32)
    m._opacity = mx.array(op_np)

    idxs = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)
    ratio = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)  # split into 1, 2, 3, 4, 5
    _, _, _, _, new_op_raw, _, _, _ = m._update_params(idxs, ratio)
    new_op = 1.0 / (1.0 + mx.exp(-new_op_raw))  # back through sigmoid

    old = 0.3
    for i, r in enumerate([0, 1, 2, 3, 4]):
        new = float(new_op[i, 0].item())
        lhs = (1.0 - new) ** (r + 1)
        rhs = 1.0 - old
        assert abs(lhs - rhs) < 1e-4, (i, r, new, lhs, rhs)
    print(f"OK   MCMC opacity invariance: (1-new)^(r+1) = 1-old across ratios 0..4")


def test_position_noise():
    """(1-op)^100 vanishes fast; low-opacity primitives get noise, high-op ones don't.

    Mixed opacity: half at 0.02 (exploring, should move), half at 0.9 (settled,
    should barely move). Verify low-op mean displacement > high-op mean.
    """
    N = 200
    m = _make_model(N=N, dead_count=0)
    op = np.zeros((N, 1), dtype=np.float32)
    op[:N // 2] = _inverse_sigmoid(np.float32(0.02))    # exploring
    op[N // 2:] = _inverse_sigmoid(np.float32(0.9))     # settled
    m._opacity = mx.array(op)

    xyz_before = mx.array(np.array(m._xyz))
    new_xyz = apply_position_noise(
        m._xyz, m.get_scaling, m.get_rotation, m.get_opacity,
        xyz_lr=1e-4, noise_lr=5e4,
    )
    delta = np.array(mx.abs(new_xyz - xyz_before))
    low_disp = float(delta[:N // 2].mean())
    high_disp = float(delta[N // 2:].mean())
    assert low_disp > 1e-6, f"low-opacity primitives got no noise ({low_disp})"
    assert high_disp < 1e-10, f"high-opacity primitives got too much noise ({high_disp})"
    assert low_disp > high_disp * 1e6, (low_disp, high_disp)
    print(f"OK   position noise: low-op shift {low_disp:.2e}, high-op shift {high_disp:.2e}")


def test_regularization():
    m = _make_model(N=200)
    reg = regularization_loss(m.get_opacity, m.get_scaling, 0.01, 0.01)
    r = float(reg.item())
    assert r > 0 and r < 1.0, r
    print(f"OK   regularization_loss: {r:.4e}")


def test_prune_with_optimizer():
    m = _make_model(N=500, dead_count=0)
    opt = MutableAdam(_param_lrs())
    opt.init_state(m.parameters())
    keep = mx.array([True] * 300 + [False] * 200)
    m.prune(keep, optimizer=opt)
    assert m._xyz.shape == (300, 3)
    for name in ("xyz", "sh0", "shN", "sb_params", "opacity", "beta", "scaling", "rotation"):
        assert opt.state[name]["m"].shape[0] == 300, (name, opt.state[name]["m"].shape)
    print(f"OK   prune: 500 → 300 primitives; optimizer state pruned in step")


def test_adam_converges():
    """Sanity check the MutableAdam implementation — minimize (x - 3)^2."""
    x = mx.array([0.0])
    opt = MutableAdam({"x": 0.1})
    opt.init_state({"x": x})

    def loss_fn(params):
        return ((params["x"] - 3.0) ** 2).sum()

    for _ in range(300):
        _, grads = mx.value_and_grad(loss_fn)({"x": x})
        params = opt.step({"x": x}, {"x": grads["x"]})
        x = params["x"]
    final = float(x.item())
    assert abs(final - 3.0) < 0.05, f"Adam failed to converge: x={final}"
    print(f"OK   MutableAdam converges: x = {final:.4f} (target 3.0)")


def test_full_densify_cycle():
    """Mirror one iteration of the reference training loop's densify block."""
    m = _make_model(N=1000, dead_count=50)
    opt = MutableAdam(_param_lrs())
    opt.init_state(m.parameters())

    cap_max = 300_000  # lego setting from the paper

    # Reference block from train.py:112-155
    dead_mask = (m.get_opacity <= 0.005).squeeze(-1)
    m.relocate_gs(dead_mask=dead_mask, optimizer=opt)
    added = m.add_new_gs(cap_max=cap_max, optimizer=opt)
    m._xyz = apply_position_noise(
        m._xyz, m.get_scaling, m.get_rotation, m.get_opacity,
        xyz_lr=1e-4, noise_lr=5e4,
    )

    assert m._xyz.shape[0] == 1050, m._xyz.shape
    assert opt.state["xyz"]["m"].shape[0] == 1050
    print(f"OK   full densify cycle: 1000 → 1050 (+{added}); state coherent")


def main():
    print("=== MCMC densification ===")
    test_relocate()
    test_add_new_gs_and_optimizer_growth()
    test_mcmc_opacity_invariance()
    test_position_noise()
    test_regularization()
    test_prune_with_optimizer()
    print("=== MutableAdam ===")
    test_adam_converges()
    print("=== end-to-end ===")
    test_full_densify_cycle()


if __name__ == "__main__":
    main()
