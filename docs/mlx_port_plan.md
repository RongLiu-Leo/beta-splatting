# MLX port plan — Deformable Beta Splatting on Apple Silicon

**Living document.** This supersedes the CUDA-side plan in `implementation_plan.md` for this branch. When they conflict, this doc wins.

## Executive summary

We are porting Deformable Beta Splatting from CUDA to Apple MLX so it runs locally on the M4 Pro (24 GB unified memory). This is a real engineering project — a rough, honest estimate is **4–8 weeks of focused work** to reach parity on training + inference. Do not treat it as a weekend refactor.

The port is **worth doing** because:
- We stop paying Colab session limits + wheel-cache warmup on every iteration.
- Unified memory + Metal is fast enough on M4 that we don't need to trade quality for tractability (msplat trains full Mip-NeRF 360 in ~70 s on M4 Max — Beta rasterization is comparable-cost work).
- It unblocks the 4-channel work: MLX code is easier to make channel-agnostic than CUDA is.
- It puts us on the same substrate as `msplat`, `splat-apple`, and future Apple-Silicon 3DGS research.

## Verified MLX capabilities

From the MLX docs and existing MLX-based rasterizers (`splat-apple`, `mlx-splat`, `msplat`):

| Capability                              | Verified? | Notes                                                                                        |
|------------------------------------------|-----------|----------------------------------------------------------------------------------------------|
| Autograd via `mx.grad`/`mx.value_and_grad`| Yes       | Works on most primitive ops.                                                                 |
| Custom Metal kernels via `mx.fast.metal_kernel` | Yes | You write the kernel body; MLX generates the signature.                                     |
| Custom VJPs via `@mx.custom_function`   | Yes       | Standard pattern; hand-derive backward and register it.                                      |
| Atomic gradient accumulation in Metal   | Yes       | `atomic_outputs=True, init_value=0` — the mechanism msplat uses.                             |
| Adam optimizer (`mlx.optimizers.Adam`)  | Yes       | Same knobs as PyTorch.                                                                        |
| `nn.Module`-style parameter tracking    | Yes       | Different syntax from PyTorch but equivalent.                                                 |
| MLX autograd through sort/scatter       | **No / partial** | Docs warn about missing vmap on some primitives. Sort in particular is non-differentiable — needs custom vjp. |
| MLX ↔ NumPy interop                     | Yes       | `mx.array(np_arr)`, `np.array(mx_arr)`. Zero-copy on Apple Silicon.                          |

Two things that meaningfully shape the plan:

1. **The rasterizer must be a custom kernel with hand-written vjp.** Not a pure-MLX composition. Autograd through tile sort + scatter isn't reliable.
2. **The per-primitive math (Beta kernel eval, SB color) does not need a custom kernel.** It's pure element-wise arithmetic — MLX autograd handles it natively.

## What we're porting (audit of CUDA sources)

Full list from `submodules/gsplat/cuda/csrc/` — 5,516 lines total:

| File                                              | Lines | Purpose                                       | Port target        |
|---------------------------------------------------|-------|-----------------------------------------------|--------------------|
| `fully_fused_projection_fwd.cu` + `_bwd.cu`       | 606   | World → screen projection + covariance        | **Metal kernel**   |
| `rasterize_to_pixels_fwd.cu` + `_bwd.cu`          | 876   | Tile-based front-to-back compositing          | **Metal kernel**   |
| `isect_tiles.cu`                                  | 391   | Primitive-to-tile assignment                  | **Metal kernel**   |
| `fully_fused_projection_packed_fwd.cu` + `_bwd.cu`| 792   | Packed variant of projection                  | **Defer**          |
| `quat_scale_to_covar_preci_fwd.cu` + `_bwd.cu`    | 317   | Quat/scale → 3D covariance + precision matrix | Pure MLX (small)   |
| `spherical_harmonics.cuh`                         | 369   | SH color (fwd + bwd inline)                   | Pure MLX           |
| `spherical_beta.cuh`                              | 191   | **SB color (DBS's key contribution)** — fwd + bwd inline | Pure MLX     |
| `compute_sh_fwd.cu` + `_bwd.cu`                   | 169   | SH dispatch wrappers                          | Pure MLX           |
| `compute_sb_fwd.cu` + `_bwd.cu`                   | 183   | SB dispatch wrappers                          | Pure MLX           |
| `world_to_cam_fwd.cu` + `_bwd.cu`                 | 323   | Camera transform                              | Pure MLX (small)   |
| `proj_fwd.cu` + `_bwd.cu`                         | 287   | Projection helper                             | Pure MLX (small)   |
| `rasterize_to_indices_in_range.cu`                | 309   | Rasterization-adjacent utility                | Defer              |
| `helpers.cuh`, `types.cuh`, `utils.cuh`           | 703   | Common code (float ops, memory)               | Rewritten inline   |

**Bottom line:** ~2,600 lines land in Metal kernels; ~1,300 lines can be pure MLX; ~1,600 lines are helpers / packed variants we can rewrite trivially or defer.

## Two-track strategy

We run two tracks in parallel because they de-risk each other:

**Track A — Pure MLX prototype (correctness first).** Slow but complete implementation entirely in MLX ops, including a soft-rasterization variant that doesn't need custom Metal. Used to:
- Verify the DBS math end-to-end.
- Serve as numerical ground truth when the Metal path is being built.
- Fall back to if the Metal port stalls.
- Estimated performance: 10–100× slower than msplat. Trains a 200-iter smoke run in minutes, not seconds. Not deployable, still useful.

**Track B — Metal-kernel rasterizer (performance).** Custom Metal shaders for the projection + tile assignment + rasterization + rasterization backward. Wired via `mx.fast.metal_kernel` + `@mx.custom_function`. Reuses ideas (not code, licensing-permitting) from msplat and splat-apple. Where Track A is correct-but-slow, Track B is fast-but-error-prone. Track A validates Track B.

Both tracks share:
- The MLX-native `BetaModel` (parameter storage, activations, save/load).
- The SB and SH color kernels in pure MLX.
- The training loop, optimizer, losses, dataset loaders.

## Module structure

New directory `mlx_impl/` alongside the existing (unchanged) CUDA code. Both trees coexist so we can compare outputs; we do not delete the CUDA path until the MLX path passes verification.

```
mlx_impl/
  __init__.py
  beta_model.py            # BetaModel translated to MLX (parameters + activations + save/load)
  losses.py                # l1_loss, ssim, psnr in MLX
  optimizer.py             # thin wrapper around mlx.optimizers.Adam matching our param groups
  color/
    spherical_harmonics.py # SH forward+backward in pure MLX (autograd handles bwd)
    spherical_beta.py      # SB forward+backward in pure MLX
  geometry/
    quat_scale_to_covar.py # quat + scale → 3D covariance
    world_to_cam.py        # camera transforms
    projection.py          # 3D → 2D projection with 2D covariance
    beta_kernel.py         # Beta kernel evaluation
  rasterizer/
    slow.py                # Track A — pure MLX soft rasterizer, differentiable via autograd
    fast.py                # Track B — Metal-kernel rasterizer, hand-written vjp
    metal_kernels/         # .metal shader sources
      project.metal
      isect_tiles.metal
      rasterize_fwd.metal
      rasterize_bwd.metal
  render.py                # top-level render() — picks slow vs fast based on config
  train.py                 # MLX training loop (mirror of ../train.py)
  eval.py                  # MLX eval loop
  compat/
    ply_io.py              # load CUDA-trained PLYs into MLX BetaModel
    numpy_bridge.py        # numpy ↔ mlx.array helpers
  tests/
    test_sb_forward.py     # cross-check SB output vs CUDA reference (needs PLY + one view)
    test_sh_forward.py
    test_projection.py
    test_render_from_ply.py # full render comparison
```

The existing `scene/`, `arguments/`, `utils/` stay mostly untouched — we hook the MLX rasterizer in via a config flag (`args.backend ∈ {"cuda", "mlx"}`).

## Phased plan

Each phase ends at a runnable, verifiable state. Do not skip.

### Phase 0 — Environment + baseline (blocking; needs your machine)

- [ ] `xcode-select --install` to get Metal toolchain.
- [ ] Fresh Python 3.11 or 3.12 venv at `.venv-mlx/`.
- [ ] `pip install mlx numpy plyfile Pillow imageio tqdm tyro tensorboard opencv-python matplotlib pandas tabulate scikit-learn`. (Note: **no PyTorch, no CUDA, no fused_ssim, no gsplat**.)
- [ ] Verify `python -c "import mlx.core as mx; print(mx.metal.is_available())"` prints `True`.
- [ ] Verify `python -c "import mlx.core as mx; a = mx.random.normal((1000, 1000)); b = a @ a; mx.eval(b); print('GEMM OK')"`.
- [ ] Optional: download one trained DBS PLY from a Colab run into `references/lego_trained.ply` for cross-validation. If we haven't trained one on Colab yet, we can defer this to Phase 4 and use random init instead.

**Exit criterion:** MLX runs. Metal is detected. We have (or plan to have) a reference PLY to compare against.

### Phase 1 — MLX BetaModel + color kernels (Track A pieces, no rasterizer yet)

Deliverables:
- `mlx_impl/beta_model.py` — parameter tensors, activations (softplus, sigmoid, exp), `get_scaling`/`get_rotation`/etc. properties. `create_from_pcd`, `save_ply`, `load_ply` (via `plyfile` on numpy, then convert). This is a mechanical PyTorch→MLX translation.
- `mlx_impl/color/spherical_beta.py` — pure MLX implementation of the SB kernel. **The forward math is:** `C = c0 + Σ_i c_i · max(dot(μ_i, v), 0)^(4·exp(β_i))` where `μ_i = (sin θ cos φ, sin θ sin φ, cos θ)`. Because it's element-wise arithmetic across primitives and lobes, MLX autograd handles the backward for free — no custom vjp needed.
- `mlx_impl/color/spherical_harmonics.py` — pure MLX SH forward. Standard basis functions up to degree 3. Autograd handles backward.
- `mlx_impl/losses.py` — L1 loss trivial; SSIM implemented as `conv2d` with a Gaussian window (same math as `utils/loss_utils.py`, drop-in translated). Replaces `fused_ssim`. This is 20 lines of MLX.

**Verification:**
- Load a trained CUDA PLY via numpy → `BetaModel` in MLX. Print tensor shapes. Sanity-check.
- Given a fixed direction, compute SB output in MLX. Compare against a Python reference of the CUDA math (`spherical_beta.cuh` translated to numpy). Should match to 1e-5.
- Run SSIM in MLX on a random 4×3×64×64 image pair vs the pure-PyTorch SSIM from `utils/loss_utils.py`. Should match to 1e-4.

**Exit criterion:** we can load a CUDA-trained model into MLX, evaluate its color at any direction, and match the CUDA math within float tolerance.

### Phase 2 — Geometry pipeline in MLX

Deliverables:
- `mlx_impl/geometry/quat_scale_to_covar.py` — quaternion + scale → 3D covariance matrix. Trivial linear algebra.
- `mlx_impl/geometry/world_to_cam.py` — transforms primitives to camera space.
- `mlx_impl/geometry/projection.py` — 3D → 2D projection with covariance projection (Zwicker et al. EWA math).
- `mlx_impl/geometry/beta_kernel.py` — Beta kernel bounded evaluation. `k(x; μ, Σ, β) = (1 - x^T Σ^{-1} x / thresh)^β` clamped to `[0, ∞)`; thresh chosen so support radius is finite.

**Verification:** compare projected 2D centers and 2D covariances against CUDA output for a set of test primitives and a fixed camera. Within 1e-5.

**Exit criterion:** we can project a CUDA-trained model's primitives to 2D and match CUDA's projection output.

### Phase 3 — Track A: Pure-MLX soft rasterizer

Deliverables:
- `mlx_impl/rasterizer/slow.py` — a differentiable rasterizer written entirely in MLX ops.

**Approach for differentiability without a custom kernel:**
- No tile sorting. Instead, for each pixel, compute contribution from *all* primitives whose 2D center is within a max-support radius. Cost is roughly O(N·pixels_touched), much worse than the CUDA path, but MLX autograd handles it because it's a masked reduce, not a sort-then-composite.
- Front-to-back compositing is approximated with a soft ordering weighted by depth. Numerically differs from front-to-back but preserves gradient flow.

Expected performance on M4 Pro: single 800×800 view with 100k primitives — maybe 5–30 seconds/render. Too slow for real training. Fine for correctness verification.

**Verification:**
- `mlx_impl/tests/test_render_from_ply.py`: load a trained CUDA PLY, render one view with `slow.py`, render the same view with CUDA (via Colab), compare MSE. Target: soft rasterizer's output within visual similarity — not bit-exact (different compositing math), but PSNR ≥ 35 dB against CUDA reference.

**Exit criterion:** end-to-end MLX render of a CUDA-trained model that produces a recognizable image of the scene. Confirms the geometry + color pipeline is wired correctly. Slow but honest.

### Phase 4 — Training loop (Track A)

Deliverables:
- `mlx_impl/train.py` — mirror of `train.py`, uses MLX BetaModel + slow rasterizer + MLX Adam.
- MCMC relocation (`relocate_gs`, `add_new_gs`) ported. These operate on parameter tensors only (no rasterizer) so they translate directly.
- Densification: reuse the same math; the `noise` addition and `1 - opacity` scaling are trivial MLX ops.

**Verification:** train `lego` at very reduced settings (32×32 images, 100 primitives, 500 iters) end-to-end. Confirm loss decreases. Do not aim for quality — this is a smoke test that the training loop closes.

**Exit criterion:** we can train an MLX model from scratch. Slowly. On a tiny scene. Correctness verified.

### Phase 5 — Track B: Metal-kernel rasterizer forward

Deliverables:
- `mlx_impl/rasterizer/metal_kernels/project.metal` — 3D → 2D projection kernel.
- `mlx_impl/rasterizer/metal_kernels/isect_tiles.metal` — primitive-to-tile assignment.
- `mlx_impl/rasterizer/metal_kernels/rasterize_fwd.metal` — tile-based front-to-back compositing with Beta kernel.
- `mlx_impl/rasterizer/fast.py` — Python wrapper using `mx.fast.metal_kernel`.

Design source: study `msplat` (rayanht) and `splat-apple` (ghif) rasterizer implementations. **Do not copy code without license verification** — read them to understand the shader structure, then write ours from scratch with the Beta kernel + SB color specifics.

**Verification:** identical output to Track A slow rasterizer, within float tolerance. Speed target: ≥ 5× msplat's baseline 3DGS number as a first pass (we care about correctness before optimization).

**Exit criterion:** a Metal-backed forward render that matches the pure-MLX render.

### Phase 6 — Track B backward: hand-written vjp

Deliverables:
- `mlx_impl/rasterizer/metal_kernels/rasterize_bwd.metal` — backward pass. Gradient accumulation via `atomic_outputs=True` per Metal semantics. Math derived from `rasterize_to_pixels_bwd.cu`.
- `@mx.custom_function` wiring in `fast.py` so `mx.grad` sees the fast rasterizer as an atomic op with custom vjp.

**Verification:**
- Finite-difference gradient check on a small scene (10 primitives, 32×32 image). Analytic gradient vs numerical gradient should match within 1e-3.
- Full training on `lego` (small settings): loss curve matches Track A within a small margin (small margin because sort order differences do produce slightly different gradients).

**Exit criterion:** Track B is a full training backend. Track A becomes a reference implementation kept alive for regression testing.

### Phase 7 — Full training run at scale

- [ ] Train `lego --cap_max 300000 --iterations 30000` on M4 Pro with Track B. Log wall time, PSNR/SSIM/LPIPS.
- [ ] Compare against the Phase 0 CUDA baseline (from `docs/journal.md`).
- [ ] Target: PSNR within ±0.1 of CUDA baseline. Wall time within 2× of msplat's baseline 3DGS (the DBS math is roughly comparable in cost per primitive).

**Exit criterion:** we have a local M4 Pro training pipeline for DBS that matches the paper's numbers.

### Phase 8 — Layer the 4-channel work on top

Once Track B is stable at 3c, the RGBA extension from `implementation_plan.md` and `rgba_extension.md` gets folded in. On the MLX side this is much cheaper than on CUDA — everything is templated on tensor shape, not on a preprocessor constant. Expected work: 1–2 weeks after Phase 7.

### Phase 9 — Inference optimization

Everything in `inference_optimization.md` applies once we have a Metal-backed renderer. Priorities: (1) frustum culling in Metal, (5) occlusion early termination, (2) lobe pruning, (9) MLX's compiled graphs.

## Risk register

Ranked by how badly it hurts if it materializes.

| Risk                                                                 | Impact  | Mitigation                                                                                              |
|----------------------------------------------------------------------|---------|--------------------------------------------------------------------------------------------------------|
| Metal kernel backward pass numerically wrong                         | High    | Finite-difference gradient check every commit. Track A serves as reference.                             |
| MLX autograd unable to backprop through some MLX op we use in Track A| High    | Prototype early (Phase 1). If we hit it, use Track B directly and skip Track A convergence guarantees. |
| Metal shader compilation fails on macOS 25.4 (we're not on Tahoe 26) | Medium  | msplat works on macOS 14+; we should be fine on 25.4. If it breaks, we upgrade.                        |
| Performance not competitive with msplat at parity                    | Medium  | Track B optimization pass (Phase 9). Not a correctness issue.                                          |
| PLY compat — CUDA-trained PLYs don't load into MLX due to layout     | Low     | `mlx_impl/compat/ply_io.py` does the translation. Tensor layout is documented in `beta_model.py`.       |
| Random-seed reproducibility (MLX vs PyTorch RNG differ)              | Low     | Not a research issue; only affects strict reproducibility of intermediate values.                       |
| Licensing on msplat / splat-apple / gsplat-mps prevents code reuse   | Low-med | We reference structure, not code. Write our own kernels.                                               |
| 24 GB unified memory tight for MipNeRF360 (1M primitives)            | Low     | Should fit — expected peak 5–10 GB. If not, work with reduced cap_max.                                 |

## Fallbacks

If Phase 5 (Metal kernels) stalls significantly, fallbacks in order:

1. **Fork msplat.** Take msplat as a baseline rasterizer; graft the Beta kernel + SB into its Metal shaders. Delivery of DBS-on-Metal, at the cost of msplat's API constraints and license terms.
2. **PyTorch MPS backend.** Run the existing PyTorch code with MPS device. Custom CUDA kernels won't work — we'd need to fall back to a slow pure-PyTorch rasterizer. Poor perf, but functional.
3. **Stay on CUDA/Colab.** The docs and CUDA plan remain valid; nothing is lost.

## What can be scaffolded before you free memory

I can write these now without running MLX (they translate mechanically from PyTorch):
- `mlx_impl/beta_model.py` scaffold (parameters + activations + I/O).
- `mlx_impl/color/spherical_beta.py` and `spherical_harmonics.py` (pure math translations of the CUDA `.cuh` files).
- `mlx_impl/losses.py` (SSIM translation from `utils/loss_utils.py`).
- Directory structure and README.

These will need MLX to actually import and run. Consider them "code review-ready" but not verified.

I should NOT scaffold before running:
- The rasterizer (both slow and fast) — too many unknowns without a machine to test.
- Metal shaders — need iterative compile/run.
- The training loop — depends on the rasterizer.

## Prerequisites for the memory-freed session

When you free memory, you'll need:
- **Disk:** ~5–10 GB for the venv, MLX, numpy, imageio, opencv, plus one trained-scene checkout.
- **Xcode command-line tools:** run `xcode-select --install` in Terminal.
- **Python:** 3.11 or 3.12. If you don't have it, `brew install python@3.12`.
- **macOS:** you're on Darwin 25.4 which is macOS 15 Sequoia. MLX 0.x supports macOS 14+, so we're fine. Metal 3 is available on M-series unconditionally.
- **Optional:** if you have a CUDA-trained PLY from Colab, drop it in `references/`. If not, we'll skip cross-validation until you do.

Once we're set up: install commands are in Phase 0. First run is Phase 1 verification (SB kernel forward parity vs numpy reference).

## Success criteria for the port

The port is "done" when:
1. `python -m mlx_impl.train -s lego --iterations 30000 --eval` completes on M4 Pro.
2. Final PSNR within ±0.1 of the CUDA baseline recorded in `docs/journal.md`.
3. Both 3-channel and 4-channel modes work (Phase 8).
4. `python -m mlx_impl.eval -m <trained model>` produces the same metrics table as CUDA `eval.py`.
5. Wall-time for the full 30k-iter lego run is < 2× msplat's baseline 3DGS run on the same hardware.

Anything short of #1 is incomplete. #2 is the honesty check. #5 is the "we didn't waste our time" check.
