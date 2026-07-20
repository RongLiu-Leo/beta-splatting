# project_journal.md

Definitive one-page status. Longer dated log in `journal.md`; deep design docs elsewhere in `docs/`. Update this when a phase changes state.

## Why

Local Deformable Beta Splatting (DBS) on Apple Silicon, extended to 4-channel RGBA, with inference optimized for real-time. Motivations: kill Colab dependency; give alpha the same view-dependent freedom SB gives RGB; ship a viable renderer, not just paper numbers.

## Status (2026-07-20)

Branch: `feat/rgba-inference`. **Track A works end-to-end and converges to good quality.** MLX-DBS on lego at 100×100 hits **PSNR 30.34 dB** on training views after 4000 total iters (1000 densify + 3000 refinement, via resume). 3827 primitives. Real 3D on-device, no CUDA. Best single-view PSNR: 31.25.

## Done

- **Docs + memory bootstrapped.** `docs/mlx_port_plan.md` is the source-of-truth plan; `docs/rgba_extension.md` locks Design A2 (alpha = 4th SB intensity, multiplicative with geometric opacity). Memory tracks user profile + Apple Silicon preference + project state.
- **Apple Silicon landscape verified.** msplat, gsplat-mps, splat-apple, MetalSplatter, RadianceKit exist and were evaluated. None cover DBS's Beta kernel / SB color / MCMC ops — port must be written.
- **MLX scaffold (`mlx_impl/`, 684 lines) runs on M4 Pro + Metal.** `BetaModel` state + activations + PLY I/O, pure-MLX SB and SH forward, MLX SSIM/L1/PSNR — all channel-agnostic from line 1.
- **Phase-1 parity tests pass on-device.** SB max_abs 5e-8 vs numpy ref (C=3 and C=4). SH bit-exact across degrees 0–3, C ∈ {3, 4}. Losses bit-exact. PLY save/load roundtrip < 1e-6.
- **Real 3D on-device via msplat (baseline, not DBS).** Trained lego in 31 s, 7000 iters, 51,907 splats, PSNR 25.09 / SSIM 0.9174 → `lego_ns_7k.ply`. Confirms M4 Pro can carry DBS-scale loads and gives a numerical reference point.
- **MCMC densification ported to MLX.** `relocate_gs`, `add_new_gs`, `_sample_alives`, `_update_params`, position-noise term, opacity/scale regularizers. Custom `MutableAdam` optimizer whose state grows/shrinks with the parameter tensors. 8/8 standalone tests pass, including the (1-new_op)^(ratio+1) = 1-old_op invariance. This is what makes DBS actually grow toward `cap_max` instead of pruning-only like msplat.
- **Geometry pipeline in MLX.** `build_covariance_3d`, `world_to_cam`, EWA `persp_proj`, `add_blur_and_invert`, `project()`, plus `beta_alpha` (bounded-support Beta kernel per-pixel eval). 5/5 tests pass.
- **Track A soft rasterizer.** Pure MLX, chunked front-to-back compositing, differentiable via autograd — no custom vjp. 5/5 tests pass including gradient flow through all inputs.
- **Dataset loader + training loop.** NeRF-synthetic → MLX cameras with OpenGL→OpenCV axis flip handled. Full training loop wires model + render + loss + reg + Adam + MCMC densify.
- **First working end-to-end DBS training on Apple Silicon.** 1000 iters on lego at 100×100: loss 0.77 → 0.03, primitives 3000 → 5650, **PSNR 27.34 dB on training views, 4 min wall time, 21 GB peak memory.** Rendered output visually recognizable — see `out/lego_1k_view5.png`.
- **Checkpointed rasterizer + periodic saves + `--resume`.** `mx.checkpoint` in the rasterizer drops active memory during training from 1.6 GB → 15 MB. Periodic PLY saves via `--save-every` rescue mid-run OOMs. `--resume` reloads a PLY for continued training. Rendered `out/lego_v4_final_view5.png`.
- **Resume + refinement-only strategy → PSNR 30.34.** After densify-heavy runs kept OOMing at ~5,650 primitives, switched to: densify to 3,827 primitives, save, then resume and refine 3,000 more iters with no growth. Peak memory plateaus at 15 GB (safe). Final: **mean PSNR 30.34 dB on 20 training views, best single-view 31.25 dB.**

## Doing

Nothing in flight — waiting on go-ahead for the next phase.

## Next (in order)

1. **Longer/higher-res Track A runs.** 3000-5000 iters, 200×200 or 400×400 images. May need pixel-tiling in the rasterizer to fit 24 GB. Push PSNR toward paper numbers.
2. **Held-out eval matrix.** Load transforms_test.json, render, compute test-view PSNR/SSIM/LPIPS. First honest comparison vs msplat baseline.
3. **Track B: Metal-shader rasterizer + hand-written vjp (~5–7 weeks).** Custom Metal via `mx.fast.metal_kernel` + `@mx.custom_function`. Track A stays as reference implementation. This is what unlocks paper-scale training on-device.
4. **4-channel end-to-end runs.** Model + rasterizer + losses are already channel-agnostic — just needs a `--color-channels 4` flag pass-through and an alpha-aware dataset loader for NeRF-synthetic (which has real alpha in the PNGs).
5. **Inference optimizations.** Ranked list in `docs/inference_optimization.md`. Top: frustum culling audit, occlusion-ε tuning, PLAS-sorted PLYs, lobe pruning.

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
