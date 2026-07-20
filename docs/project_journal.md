# project_journal.md

Definitive one-page status. Longer dated log in `journal.md`; deep design docs elsewhere in `docs/`. Update this when a phase changes state.

## Why

Local Deformable Beta Splatting (DBS) on Apple Silicon, extended to 4-channel RGBA, with inference optimized for real-time. Motivations: kill Colab dependency; give alpha the same view-dependent freedom SB gives RGB; ship a viable renderer, not just paper numbers.

## Status (2026-07-20)

Branch: `feat/rgba-inference`. Scaffold + Phase-1 parity landed. No differentiable renderer yet, so no MLX training or MLX-rendered images.

## Done

- **Docs + memory bootstrapped.** `docs/mlx_port_plan.md` is the source-of-truth plan; `docs/rgba_extension.md` locks Design A2 (alpha = 4th SB intensity, multiplicative with geometric opacity). Memory tracks user profile + Apple Silicon preference + project state.
- **Apple Silicon landscape verified.** msplat, gsplat-mps, splat-apple, MetalSplatter, RadianceKit exist and were evaluated. None cover DBS's Beta kernel / SB color / MCMC ops — port must be written.
- **MLX scaffold (`mlx_impl/`, 684 lines) runs on M4 Pro + Metal.** `BetaModel` state + activations + PLY I/O, pure-MLX SB and SH forward, MLX SSIM/L1/PSNR — all channel-agnostic from line 1.
- **Phase-1 parity tests pass on-device.** SB max_abs 5e-8 vs numpy ref (C=3 and C=4). SH bit-exact across degrees 0–3, C ∈ {3, 4}. Losses bit-exact. PLY save/load roundtrip < 1e-6.
- **Real 3D on-device via msplat (baseline, not DBS).** Trained lego in 31 s, 7000 iters, 51,907 splats, PSNR 25.09 / SSIM 0.9174 → `lego_ns_7k.ply`. Confirms M4 Pro can carry DBS-scale loads and gives a numerical reference point.

## Doing

Nothing in flight — waiting on go-ahead for the next phase.

## Next (in order)

1. **Geometry pipeline (~1 week).** quat→covar, world→cam, 3D→2D projection with EWA covariance, Beta-kernel evaluation. Pure MLX, autograd handles bwd. Verifiable by projecting a CUDA-trained PLY and diffing against CUDA output.
2. **Track A: pure-MLX slow rasterizer (~1–2 weeks).** Soft rasterizer written entirely in MLX ops. 10–100× slower than msplat, but differentiable via autograd → training loop closes. Correctness reference for Track B.
3. **Training loop + MCMC densification (~1 week).** Mirror `train.py` in MLX. Smoke-train `lego` end-to-end.
4. **Track B: Metal-shader rasterizer + hand-written vjp (~5–7 weeks).** Custom Metal via `mx.fast.metal_kernel` + `@mx.custom_function`. Design from scratch, using msplat as shader-layout reference (not fork, not upstream — DBS changes core assumptions everywhere).
5. **4-channel end-to-end (~1–2 weeks after Phase 4).** Flip `color_channels=4` runs; MLX code is already channel-agnostic, so the work is mostly verification + eval matrix.
6. **Inference optimizations (parallel with 5).** Ranked list in `docs/inference_optimization.md`. Top: frustum culling audit, occlusion-ε tuning, PLAS-sorted PLYs, lobe pruning.

## Rejected / parked

- **Fork msplat.** Would inherit its 3DGS-vanilla design and permanent merge burden. Also license-adjacent friction.
- **Upstream DBS to msplat.** Personal research repo, unlikely to absorb the scope change.
- **Design A1 (alpha replaces geometric opacity).** Breaks DBS's opacity-only MCMC invariance proof.
- **Design C (per-primitive scalar alpha).** No view dependence — reduces to a second opacity.
- **CUDA/Colab path as primary.** Pivoted 2026-07-10; kept as fallback in `implementation_plan.md`.

## References

- Full plan: `docs/mlx_port_plan.md`
- Long log: `docs/journal.md`
- 4c design: `docs/rgba_extension.md`
- Method notes: `docs/method_dbs.md`
- Apple Silicon catalog: `docs/apple_silicon_options.md`
- Inference plan: `docs/inference_optimization.md`
