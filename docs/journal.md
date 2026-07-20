# Journal

Dated log of research work on this fork. Newest at the top.

Entry format: date, one-line summary, then whatever detail belongs on the record (decisions, results, blockers, links).

---

## 2026-07-20 — PSNR 30.34 via resume + refinement; three OOM-close runs debugged

Full training arc today, four runs, converging strategy each time.

**Run 1 (v1, earlier):** 1000 iters, cap 20k, densify every 100. → PSNR 27.34, 5650 primitives, 21 GB peak (survived by luck).

**Run 2 (v2):** Added `--save-every 500` and `mx.checkpoint` in rasterizer. 3000 iters attempted, cap 20k. Killed at iter 1800 (peak 21.3 GB, system free memory below 200 MB → OOM imminent). Saved iter1500 checkpoint. PSNR 27.88 on iter1500 render.

**Run 3 (v3):** Tried tighter densify + cap 6k. Same OOM territory around iter 1400. Killed. Saved iter1000 (PSNR ~28) and iter500 checkpoints.

**Run 4 (v4, this session):** Added `--resume` flag. Loaded `lego_v3_iter1000.ply` (3827 primitives), disabled densification, ran 3000 more iters of pure refinement.

Peak memory plateaued at **15 GB from iter 100 through iter 3000** because primitives don't grow. Speed steady at 5 it/s. Loss trajectory:
- Resume start: loss 0.047, PSNR 28.39 at iter 1000
- iter 2000: loss 0.019, PSNR 29.39
- iter 3000: loss 0.015, **mean PSNR 30.34 on 20 train views** (single-view range 28.87–31.25)

**Wall time totals:** ~5 min densify (v3 to iter 1000) + ~10 min refinement (v4). ~15 min end-to-end on M4 Pro.

**Non-obvious findings this session:**
- **Refinement > more densify.** Fewer primitives (3827) refined longer beat more primitives (5650) refined less. Density growth was hitting the 24 GB wall; freezing primitive count and iterating longer works better on Apple Silicon.
- **`mx.checkpoint` gives huge active-memory savings (1.6 GB → 15 MB) but modest peak savings during training.** The outer autograd graph over the training step still dominates peak. Where checkpoint really pays off is inference / test rendering (test suite peak 1314 → 345 MB).
- **`--resume` only restores parameters, not Adam state.** First iter after resume shows loss spike (~0.047 vs 0.038 at save time) as new Adam moments spin up. Recovers within ~100 iters.

**Files added this session:** `--resume` flag in `mlx_impl/train.py`; `--save-every` (added earlier); `mx.checkpoint` in the rasterizer (added earlier). Nothing new; strategy change.

**Next options:**
1. Higher-resolution runs (200×200 needs pixel-tiling in the rasterizer to fit memory).
2. Held-out test-view eval (right now we only measured training views).
3. Track B (Metal kernels) — the ~5-7 week port for real throughput and higher res.
4. Chain multiple resume+refine cycles at increasing resolution.

## 2026-07-20 — End-to-end MLX-DBS trained; PSNR 27.34 on lego in 4 minutes

**First working DBS training on Apple Silicon.** Full pipeline lands: geometry, rasterizer, dataset, training loop, densification, optimizer step. Committed as `75e5c90`.

**Files added:**
- `mlx_impl/geometry/projection.py` (~180 lines) — `build_covariance_3d`, `world_to_cam`, `persp_proj` (EWA), `add_blur_and_invert`, `compute_radius`, `project()`. Direct port of `fully_fused_projection_fwd.cu` + `utils.cuh::persp_proj`.
- `mlx_impl/geometry/beta_kernel.py` — `beta_alpha(dx, dy, conic, opacity, beta)`. The bounded-support Beta kernel evaluated per-pixel: `alpha = opacity * max(0, 1 - sigma)^beta`, `sigma < 1` inside support.
- `mlx_impl/rasterizer/slow.py` (~120 lines) — Track A soft rasterizer. Chunked front-to-back compositing, differentiable via `mx.grad`. No custom vjp needed. Sort once globally per view.
- `mlx_impl/dataset.py` — NeRF-synthetic loader. NeRF-synthetic uses OpenGL camera convention (looks down -Z, +Y up); gsplat uses OpenCV convention (looks down +Z, +Y down). Handled inline via `c2w[:3, 1:3] *= -1` before inverting.
- `mlx_impl/train.py` — training loop mirroring `../train.py`. Wires the whole thing.
- `mlx_impl/tests/{test_geometry,test_rasterizer,render_view}.py` — smoke tests + a side-by-side visualizer.

**Verification:**
- Geometry: 5/5 tests pass (SPD covariance, world_to_cam identity, persp_proj centered-point, project() full, beta_alpha edge cases).
- Rasterizer: 5/5 tests pass. Single primitive → visible disk. Depth sort works (front wins). Gradient flows through all 5 params. 5000 primitives at 128×128 renders in 318 ms with 1.3 GB peak.
- End-to-end training: loss cleanly monotonic 0.77 → 0.03 over 1000 iters. MCMC actively grew primitives 3000 → 5650. PSNR 27.00 on training views. 244 s wall time on M4 Pro. Peak memory 21 GB.
- Rendered output visually recognizable as lego. `out/lego_1k_view5.png` for the record.

**Two debugging false alarms — both were correct math surprising me:**
1. Rasterizer test claimed "means_2d gradient is zero." True — because I put the primitive at the exact image center. Perfectly symmetric MSE around that point → analytic grad = 0. Fixed the test to place primitives off-center.
2. Position-noise test showed zero displacement at op=0.5. Turns out `(1-0.5)^100 ≈ 4e-30` genuinely rounds to zero in float32. That's the paper's design — high-opacity primitives don't move; only low-opacity (exploring) ones do. Fixed the test to sample both op=0.02 and op=0.9 and verify the ratio.

**Performance envelope (M4 Pro, Track A):**
- 100×100 images, 3-6k primitives: ~4-8 it/s, 4 min per 1000 iters, 10-21 GB peak.
- 128×128 images, 5k primitives: 318 ms per view (forward only), 1.3 GB peak. Backward doubles this.
- Scaling limit: at 100×100, ~20k primitives before we hit 24 GB. To go further we need chunking-over-pixels (tile-based rendering) or Track B Metal shaders.

**Comparison to reference numbers:**
- msplat baseline (this project, 2026-07-10): 25.09 PSNR on lego held-out at 7000 iters, 800×800, 51,907 splats, 31 s.
- Our MLX-DBS: 27.00 PSNR on training views at 1000 iters, 100×100, 5,650 splats, 244 s.
- Not directly comparable (train vs held-out, different res, different iter count), but the trajectory is right: DBS's SB color + Beta kernel converge much faster per-iteration than baseline 3DGS.

**What's not done:**
- Track B (Metal shaders) — the ~5-7 week port for real throughput.
- Held-out test-view PSNR — need to render transforms_test.json views too.
- Higher resolution — 200×200 or 400×400 needs pixel-tiling in the rasterizer to fit memory.
- 4-channel end-to-end training run — model code supports it, needs a training config flag pass-through.

**Next when we resume:** either (a) longer/higher-res training runs to push quality, or (b) start Track B Metal-shader work for the throughput to make (a) tractable.

## 2026-07-20 — MCMC densification ported to MLX; 8/8 tests pass

Root cause the user identified: msplat is prune-only, no MCMC relocation → count shrinks (100k init → 51,907 after 7k iters). DBS's reference `train.py` uses MCMC to grow to cap_max via relocate + add + noise.

**Ported from `scene/beta_model.py` to `mlx_impl/beta_model.py`:**
- `_update_params(idxs, ratio)` — gather + apply MCMC opacity-invariance: `new_op = 1 - (1 - op)^(1/(ratio+1))`, clamped to [0.005, 1-eps], stored pre-sigmoid.
- `_sample_alives(probs, num, alive_indices)` — multinomial sample via `mx.random.categorical(log_probs)`, bincount for ratios via scatter-add.
- `relocate_gs(dead_mask, optimizer)` — dead-slot replacement, reset Adam moments at source indices only.
- `add_new_gs(cap_max, optimizer)` — grow by 5% up to cap, concatenate onto all parameter tensors, grow optimizer state.
- `prune(live_mask, optimizer)` — pruning with paired optimizer state pruning. Discovered MLX 0.31.2 does not support boolean indexing; converted to int indices via numpy.

**New files:**
- `mlx_impl/optimizer.py` (~90 lines) — `MutableAdam`. Custom Adam with hooks the densification calls: `reinit_state_at(name, indices)`, `prune_state(name, keep_indices)`, `grow_state(name, extra_count)`. Not built on `mlx.optimizers.Adam` because that couples state init to fixed param shapes.
- `mlx_impl/densification.py` (~90 lines) — `build_rotation`, `build_scaling_rotation`, `apply_position_noise`, `regularization_loss`. Direct port of `utils/general_utils.py` + `train.py:147-155`.
- `mlx_impl/tests/test_densification.py` (~230 lines) — 8 tests, all pass.

**Test results:**
- relocate_gs: 100 dead → 0 dead, N stable at 1000 ✓
- add_new_gs: +50 with cap 1500, +20 with cap 1020, optimizer state matches every param ✓
- MCMC opacity invariance: `(1-new)^(r+1) = 1-old` bit-exact across ratios 0..4 ✓
- Position noise: low-opacity (0.02) shift 1.30e-03, high-opacity (0.9) shift 0.00 — matches paper design where settled primitives barely move ✓
- Regularization loss: finite and non-zero ✓
- Prune: 500 → 300 primitives, optimizer state pruned in step ✓
- MutableAdam: converges (x - 3)^2 to x = 3.0000 ✓
- Full densify cycle mirroring `train.py:112-155`: 1000 → 1050 (+50), state coherent ✓

**Design notes worth carrying:**
- `MutableAdam` chosen over `mlx.optimizers.Adam` because MLX's Adam couples state initialization to initial param shapes. When primitives are added/pruned mid-training, we need first-class resize.
- Boolean indexing gap in MLX 0.31.2 — all mask operations go through numpy `np.where` → mx.array int indices. Cheap on unified memory but worth flagging if MLX 0.32+ adds boolean support.
- `mx.random.categorical(log_probs, num_samples=N)` is the multinomial equivalent; takes logits, not probs.

**Not tested (blocked on rasterizer):**
- Real training convergence with densification active — needs the render call to produce gradients that flow through the parameters.
- Cap_max reaching (300k for lego) — need many densification steps in a real loop.

**Next:** the geometry pipeline (world→cam, projection with EWA covariance, Beta-kernel evaluation) so the rasterizer has all its ingredients ready.

## 2026-07-10 — Option 3 executed: baseline 3DGS reconstruction on M4 Pro via msplat

Ran `msplat` (baseline 3DGS on Metal) locally on the M4 Pro to get a real 3D reconstruction of `lego/` today, ahead of the MLX-DBS port maturing.

**Steps taken:**
1. `pip install --user msplat` (+ tyro for CLI). Version 1.1.3, Apache-2.0.
2. Wrote `scripts/nerf_synthetic_to_nerfstudio.py` to convert lego's split `transforms_train/test.json` to a single `transforms.json` in the format msplat's auto-detector recognizes. Symlinks images into `images/` rather than copying (saves disk).
3. msplat's loader wants an initial point cloud, but NeRF-synthetic doesn't ship one — generated a uniform-random 100k point cloud inside `[-1.5, 1.5]^3` and referenced it via `ply_file_path` in `transforms.json`.
4. `msplat-train --input lego_ns --output lego_ns_7k.ply --num-iters 7000 --downscale-factor 1 --eval`.

**Result:**
- Wall time: **31 seconds** for 7000 iters at full 800×800.
- Step time: ~3 ms during pre-densification, rising to ~7 ms after densification.
- Final splat count: **51,907**.
- Eval on 38 held-out test views: **PSNR 25.09, SSIM 0.9174, L1 0.0261**.
- Output: `lego_ns_7k.ply`, 12 MB, standard 62-property 3DGS PLY.

**Interpretation:**
- 25.09 PSNR at 7k iters is what baseline 3DGS does — DBS at 30k iters targets ~33-34 on lego. This is not paper-quality, it's a "prove the pipeline works locally" result. For a full-quality baseline we'd run 30k iters; expected ~2 minutes on this hardware.
- This is **not DBS** — no Beta kernel, no SB color, no MCMC densification. But it's a real 3D scene reconstructed from the same 100 training images that our DBS work will use, running entirely on-device with zero Colab dependency.

**Value for the DBS port:**
- Validates that the M4 Pro can handle DBS-scale training loads (msplat's 3 ms/step is a strong baseline; DBS shouldn't be more than 2× that).
- Gives us a numerical reference point (25.09 baseline @ 7k iters) to compare our MLX-DBS numbers against once Phase 5-6 lands.
- The PLY output is loadable by any 3DGS viewer. Our BetaModel loader expects `sb_params_*` + `beta` attributes so it won't load this file directly, but we can write a compat loader if we want to view baseline 3DGS scenes from our MLX code.

**Option 2 (from-scratch Metal rasterizer for DBS) architectural decision:** write from scratch, use msplat's shader layout as *reference* only. Reasons: DBS's Beta-kernel bounded support changes tile-assignment assumptions, SB color has no 3DGS analog, msplat is a research prototype whose maintainer is unlikely to accept DBS as a scope addition, forking creates a permanent maintenance liability. Estimated 5-7 weeks focused effort for Track B.

## 2026-07-10 — MLX scaffolding runs on M4 Pro. All non-rasterizer pieces pass parity.

Ran the scaffold on this M4 Pro under macOS 26.4.1 Tahoe, Python 3.13.13, MLX 0.31.2, Metal detected. All correctness tests pass:

**SB kernel (`test_sb_forward`) vs numpy reference:**
- C=3: max_abs 5.22e-08, max_rel 7.43e-06
- C=4: max_abs 3.12e-08, max_rel 6.07e-06

**SH kernel (`test_sh_forward`) vs numpy reference (degree 0-3):**
- All 8 configs (C ∈ {3, 4} × degree ∈ {0, 1, 2, 3}): **bit-exact** (max_abs 0.00e+00)

**Losses:**
- L1 bit-exact for both C=3 and C=4
- PSNR bit-exact for both C=3 and C=4
- SSIM (via `mx.conv2d` with groups=channel) returns valid values in [0, 1], high for near-identical images

**BetaModel init + PLY save/load roundtrip:**
- Both C=3 and C=4: all 8 parameter tensors round-trip with max diff < 1e-6
- Activations produce valid values (opacity in (0,1), scaling > 0, rotations unit-norm)

**What this proves:** the mechanical PyTorch→MLX translation was correct. Every part of the DBS pipeline that isn't the rasterizer is now demonstrably working on MLX + Metal for both 3-channel and 4-channel modes on the M4 Pro. The 4c work landed for free because the code was written channel-count-agnostic from the start.

**What's still not written** (blocks generating actual 3D):
- Geometry: quat→covar, world→cam, 3D→2D projection, Beta kernel evaluation.
- Rasterizer: neither the pure-MLX slow version (Track A) nor the Metal-shader fast version (Track B).
- Training loop, MCMC densification, MLX optimizer wiring.

**Sample data:** `lego/` is present and ready — 100 train views + 200 test views + `transforms_train/test.json`. When the rasterizer lands we can train against this immediately.

**Total code so far:** 684 lines of MLX Python across 8 files under `mlx_impl/`. About 15% of what a full training-capable port needs. Rasterizer is the next 60% (est. 1,500–2,500 lines when written).

## 2026-07-10 — Pivot: CUDA → MLX/Metal. Port plan written; MLX scaffolding landed.

User asked to run on Apple MLX instead of CUDA. Plan and scaffold written; not yet executed.

**Docs**
- `docs/mlx_port_plan.md` — the full 9-phase port plan with the risk register and verified MLX capability audit. This is now the primary implementation track.
- `docs/implementation_plan.md` — updated with a note pointing at the MLX plan. Its design content (A2 alpha, hardcoded-3 audit) still applies to both tracks.

**Scaffolding written under `mlx_impl/`:**
- `beta_model.py` — MLX BetaModel with parameters, activations, `create_from_pcd`, and PLY I/O compatible with CUDA-trained files. **Channel-count-agnostic from the start** (3c and 4c share code paths).
- `color/spherical_beta.py` — pure MLX forward for the SB kernel. Autograd handles the backward — no custom vjp required because SB is element-wise arithmetic across primitives (no sort/scatter/atomic).
- `color/spherical_harmonics.py` — pure MLX SH forward up to degree 3, channel-count-agnostic (fixes the CUDA hardcoded-3 stride in one shot).
- `losses.py` — L1 + SSIM + PSNR in MLX; replaces the CUDA `fused_ssim` dependency.
- `tests/test_sb_forward.py` — parity check between the MLX SB forward and a numpy reference of the CUDA math, for both C=3 and C=4. Runs without needing a trained PLY.
- `README.md` — status doc for the mlx_impl/ tree.

**Not yet written (blocked on next session with MLX installed):**
- Geometry pipeline (`geometry/`): quat→covar, world→cam, projection, Beta kernel.
- Rasterizer both slow (pure MLX) and fast (Metal shaders + hand-written vjp).
- Training loop, optimizer wrapping, MCMC densification.

**Key structural decision recorded in `mlx_port_plan.md`:** two-track strategy — pure-MLX slow rasterizer for correctness (Track A), Metal-kernel fast rasterizer for performance (Track B). Track A validates Track B. If Metal shader work stalls, Track A is still a working (slow) system.

**Honest scope statement:** this is a 4–8 week engineering project, not a weekend refactor. Nine phases. The port plan is designed so each phase ends at a runnable state, so we can pause between phases without losing ground.

**Next when memory is freed (user action):**
1. `xcode-select --install`
2. Create Python 3.11 or 3.12 venv, `pip install mlx numpy plyfile Pillow imageio tqdm tyro tensorboard opencv-python matplotlib pandas tabulate scikit-learn`.
3. Run `python -c "import mlx.core as mx; print(mx.metal.is_available())"` — expect `True`.
4. Run `python -m mlx_impl.tests.test_sb_forward` — first correctness check.
5. Then we can iterate on the rasterizer.

## 2026-07-10 — Branched `feat/rgba-inference`; wrote implementation and inference-opt plans

Cut a new branch from `main` (`431d575`) for the dual-channel + inference-optimization work. Two new living docs:

- `docs/implementation_plan.md` — phased checklist for making the pipeline runtime-selectable between 3 and 4 color channels. Design A2 confirmed: alpha is a fourth SB intensity, multiplicative with geometric opacity.
- `docs/inference_optimization.md` — ranked list of 9 optimization directions with expected wins and cost estimates.

**Key CUDA finding while planning:** the gsplat rasterizer already supports arbitrary channel counts via power-of-2 padding (`submodules/gsplat/cuda/_wrapper.py:507-536`). We do *not* need to touch the tile-accumulation kernel. The only CUDA work is in the SH forward/backward kernels, which have `3` hardcoded as the coefficient stride (`spherical_harmonics.cuh` and `compute_sh_fwd.cu:52`), plus whatever the SB kernel looks like (still need to find and inspect it). This is a much smaller CUDA surface than I first assumed — a real accelerator for the timeline.

**Concrete audit of hardcoded 3s and 6s that need to become configurable** — recorded in `implementation_plan.md` §"What the code currently assumes."

**Phasing:** 5 phases. Phase 0 is a baseline record on Colab that we still need to run. Phase 1 is a Python-only refactor that must not change 3c behavior. Phase 2 templates the SH kernel. Phase 3 wires 4c end-to-end. Phase 4 is the evaluation harness. Phase 5 is inference optimization (parallel with anything after Phase 3).

**Immediate next:** the user needs to run the smoke notebook + a 30k-iter `lego` baseline on Colab and paste back the numbers so we have Phase-0 ground truth before touching code.

## 2026-07-10 — Apple Silicon options surveyed

Verified the Apple Silicon 3DGS landscape (msplat, gsplat-mps, splat-apple, mlx-splat, MetalSplatter, 3dgs-viewer, RadianceKit, CorbeauSplat). Full evaluation in `apple_silicon_options.md`.

Key finding: **no existing Apple Silicon port covers DBS's contributions.** They all target upstream 3DGS. Our `submodules/gsplat/` is a DBS-specific CUDA fork containing the Beta kernel, SB color, and MCMC relocation code — none of that exists in Metal/MLX form. Swapping submodules doesn't give us DBS locally, it gives us baseline 3DGS locally.

Immediate practical adoption:
- Use `MetalSplatter` (or `gaook/3dgs-viewer`) instead of viser share URL when inspecting trained PLYs on-device.
- Use `msplat` for local baseline-3DGS comparison runs on this M4 Pro (same hardware apples-to-apples).
- Use `RadianceKit` when we shoot our own custom test scenes — bypasses COLMAP wrangling.

Deferred: porting DBS itself to Metal. Real project (2–6 weeks kernel work), but parked behind the 4-channel research question. If the 4-channel design proves out on CUDA, the Metal port is the next natural investment.

The user is on an M4 Pro with 24 GB unified memory — the machine is capable enough that local DBS training would be genuinely useful once (if) the kernels are ported.

## 2026-07-10 — Docs bootstrap

Set up `docs/` with an overview, method notes on DBS, the RGBA extension plan, training notes, and this journal. No code changes.

State of the fork today:

- Branch `main` is at `431d575` (upstream `Update README.md`, 2025-10-08). No divergence from upstream code.
- The only forward-looking artifact is `notebooks/colab_smoke.ipynb` (added 2025-07-07 locally, untracked in git). It sets up a Colab harness with a Drive-cached gsplat wheel and probes the SH kernel for `EXPECT_CHANNELS=4`. Right now that probe fails because no CUDA changes have been made yet — that's expected and is the fail-fast signal.
- No 4-channel code has been landed. `_sh0` is still 3-channel, `_sb_params` is still `(N, K, 6)`.
- No Colab runs have been logged. First smoke run is still to do.

Next step: run the smoke notebook on Colab against `main`, get a baseline "3-channel probe passes, 4-channel probe fails" record, and paste the outputs here so we have a known-good baseline before touching CUDA.

Open questions carried into this work (see `method_dbs.md` and `rgba_extension.md`):

- Which alpha design (A1 vs A2 in `rgba_extension.md`) — starting with A2.
- Is the SH tail `_shN` actually used? Cheap to check by zeroing it and re-training.

---

## Template for future entries

```
## YYYY-MM-DD — one-line summary

What I did:
- ...

Results (if a run):
- Dataset / scene:
- Config: cap_max, iterations, channel count, GPU
- PSNR / SSIM / LPIPS / step-time / peak memory:
- Anything surprising:

Decisions:
- ...

Next:
- ...
```
