# Journal

Dated log of research work on this fork. Newest at the top.

Entry format: date, one-line summary, then whatever detail belongs on the record (decisions, results, blockers, links).

---

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
