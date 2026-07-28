# Implementation plan — dual 3c/4c support

> **Note (2026-07-10):** the CUDA port plan below is preserved for its design-decision content (Design A2 for alpha, hardcoded-3 audit, hyperparameter mapping), but the primary implementation track is now MLX — see `mlx_port_plan.md`. When the two conflict, MLX wins. The 4-channel design (Design A2 — alpha as a fourth SB intensity, multiplicative with geometric opacity) is unchanged and applies to both tracks; it just lands in `mlx_impl/` first.

This is a **living document.** It is the working checklist for landing dual 3-channel / 4-channel support on `feat/rgba-inference`. Keep refining it as we learn things. When an item is done, mark it `[x]` and add the commit hash. When we discover the plan is wrong, edit the plan — don't work around it.

**Related docs:**
- Rationale + design-space discussion: `rgba_extension.md`.
- Method-side context: `method_dbs.md`.
- Inference optimization plan: `inference_optimization.md`.

## Scope

Make the DBS pipeline **channel-count-agnostic** at runtime, with `color_channels ∈ {3, 4}` selectable per training run:

- `color_channels=3`: exact behavior of upstream DBS (regression must be zero, up to build noise).
- `color_channels=4`: RGBA. Alpha is a fourth SB intensity per lobe, *multiplied* with the geometric opacity at rasterization time (Design A2 in `rgba_extension.md`). Alpha is a color-space quantity, not a replacement for the geometric opacity that MCMC relocation depends on.

Non-goals for this branch:
- Porting to Metal/MLX. That's a separate project (`apple_silicon_options.md`).
- The `sb_number` progressive schedule (K=1 → 2 → 4). Separate ablation.

## What the code currently assumes (as of `main@431d575`)

Grepped and audited. These are the hardcoded 3-channel or hardcoded 6-slot spots we have to touch:

### `submodules/gsplat/cuda/csrc/`
- **`spherical_harmonics.cuh`** — SH coefficient stride hardcoded to `3` in every `coeffs[* * 3 + c]` index (`sh_coeffs_to_color_fast`, `sh_coeffs_to_color_fast_vjp`). Fix: template on `CDIM` (channel dim) or add an overload with a runtime stride.
- **`compute_sh_fwd.cu`** — `TORCH_CHECK(coeffs.size(-1) == 3, ...)` on line 52; `coeffs + elem_id * K * 3` on line 35. Fix: relax the check to `∈ {3, 4}`, replace `3` with the actual channel count from `coeffs.size(-1)`.
- **`compute_sh_bwd.cu`** — mirror of fwd; same fix.
- **SB evaluation kernel** — need to inspect. If it also assumes 6 slots per lobe, has to be templated similarly.

### `submodules/gsplat/cuda/_wrapper.py`
- Rasterization (`rasterization()`, `_wrapper.py:445-577`): **already handles arbitrary channels via power-of-2 padding** (line 507-536). No CUDA change needed for tile accumulation. This is a big win.
- SH wrapper (line 1235): passes `coeffs` through unmodified. Should work once the CUDA side is templated.

### `scene/beta_model.py`
- **`create_from_pcd` (L156-212)**: `sb_params` shape hardcoded to `(N, K, 6)`; `theta` at `[..., 3]`, `phi` at `[..., 4]`, no beta init (β=0 → activation gives 4). For 4c, layout becomes `(N, K, 7)` = `[R, G, B, A, θ, φ, β]`; θ moves to `[..., 4]`, φ to `[..., 5]`. **`_sh0`** shape gets a channel dim of `color_channels`.
- **`sb_params_activation` (L47-50)**: softplus over `[..., :3]` — must be `[..., :color_channels]` in 4c mode.
- **`construct_list_of_attributes` (L272-287)**: derives PLY attribute names from tensor shapes. Actually already dynamic — good, no change beyond the tensor shapes themselves.
- **`load_png` (L502-509)**: hardcoded "expecting 3 channels each" comment + hardcoded 6 in `np.zeros((..., 6, ...))`. Must be conditional on `color_channels`.
- **`render` (L802-846)**: passes `self.get_shs` and `self.get_sb_params` to `rasterization()`. If gsplat's rasterization respects channel dims (it does), the output `rgbs` becomes 4-channel naturally.
- **`__init__` (L68-83)**: add `color_channels` param; store it; use it everywhere the shape decisions above depend on it.

### `train.py`
- **Loss (L107-114)**: `l1_loss(image, gt_image)` and SSIM — both assume matched channel counts. For 4c we compare RGBA-to-RGBA on synthetic scenes; for 3c we drop alpha even if the dataset has one.
- **Background (L41-42, L101-103)**: `bg_color = [1, 1, 1]` — needs a 4th element for 4c (alpha = 1 for opaque background, 0 for transparent).

### `scene/dataset_readers.py`
- Need to inspect. For NeRF-synthetic PNGs, alpha is in the file. For COLMAP/MipNeRF360, it isn't. Loader must return a `(H, W, C)` image where `C` matches the model's `color_channels`. When the dataset provides a channel count that doesn't match, composite over the model's background color.

### `arguments/__init__.py`
- Add `color_channels: int = 3` to `ModelParams`. Wired everywhere via `args.color_channels`.

### `compress.py` / `utils/compress_utils.py`
- `sort_param_dict` and `compress_png` may hardcode per-channel splits. Verify; adjust if needed.

## Phased plan

Phases are ordered so we can end each one at a **runnable, testable** state. Don't skip ahead — the smoke probe from `notebooks/colab_smoke.ipynb` gives us fast feedback at each stage.

### Phase 0 — Baseline record (blocking)
- [ ] Run the smoke notebook on Colab against `main` (or this branch pre-changes, they're byte-identical).
- [ ] Log: `nvidia-smi -L`, section 7 probe (`C=3` OK, `C=4` FAIL), section 9 tensor shapes.
- [ ] Paste results into `docs/journal.md`.
- [ ] Also run a full 30k-iter `train.py -s lego --white_background --eval` on Colab L4 to lock in reference PSNR/SSIM/LPIPS numbers before any code changes. These become the 3-channel regression benchmark.

**Exit criterion:** we have numeric "before" numbers we can compare 3c-after-refactor and 4c-after-refactor against.

### Phase 1 — Python-only refactor for 3c (safe, no behavior change)
Goal: introduce `color_channels` plumbing without changing behavior at `color_channels=3`.

- [ ] Add `color_channels: int = 3` to `ModelParams`.
- [ ] `BetaModel.__init__` accepts `color_channels`, stores it as `self.color_channels`.
- [ ] Replace every hardcoded `3` in `beta_model.py` shape math with `self.color_channels`, but only *after* verifying each `3` we found is actually a channel-count `3` (some `3`s in the file are xyz dims — don't touch those).
- [ ] Replace every hardcoded `6` for SB slot count with `self.color_channels + 3` (RGB + θφβ → RGBA + θφβ).
- [ ] Rename `sb_params_activation` softplus slice from `[..., :3]` to `[..., :self.color_channels]`.
- [ ] `load_ply` / `save_ply` / `load_png` / `save_png` — infer channel count from tensor shape on load; write it consistently on save. Update the "expecting 3 channels each" comment.
- [ ] Add a `color_channels` field to the meta.json produced by `save_png`.
- [ ] `train.py` background tensor sized to `color_channels`; when `color_channels=4`, default alpha=1.

**Verification:** run the smoke notebook on this branch with `EXPECT_CHANNELS=3`. Kernel probe passes (unchanged). 200-iter smoke train produces byte-identical loss curve to Phase-0 baseline (or within float noise). Full 30k-iter run reproduces Phase-0 metrics within ±0.02 PSNR.

**Exit criterion:** `color_channels=3` runs behave exactly like `main` did. No CUDA changes yet.

### Phase 2 — CUDA changes to SH kernel to accept 4 channels
Goal: gsplat's `spherical_harmonics` accepts `C ∈ {3, 4}` without regressing `C=3`.

- [ ] `spherical_harmonics.cuh`: replace hardcoded `3` in `coeffs[i * 3 + c]` with a template parameter or a `stride` argument. Decide up front: **template on `CDIM`** — the kernel launches for a known channel count per call, and templating gives us max perf with no branching in-loop. Instantiate for `CDIM=3` and `CDIM=4`.
- [ ] `compute_sh_fwd.cu`: relax the `TORCH_CHECK` to `coeffs.size(-1) ∈ {3, 4}`. Dispatch on channel count to the correct template instantiation. Output tensor shape gets the right channel dim.
- [ ] `compute_sh_bwd.cu`: same treatment.
- [ ] `_wrapper.py`: no change expected (already tensor-shape-driven), but add an assert that `coeffs.size(-1) in (3, 4)`.
- [ ] Locate SB CUDA kernels (grep for `sb_params` or `spherical_beta` in `submodules/gsplat/cuda/csrc/`). Same treatment: template the slot layout on `CDIM+3`.

**Verification:** run the smoke notebook with `EXPECT_CHANNELS=4`. Section 7 kernel probe flips from FAIL → OK. Section 7 also still passes with `EXPECT_CHANNELS=3`. Bit-exact PSNR for the `C=3` path vs Phase 1 (a good sanity check that we didn't accidentally slow the 3-channel path).

**Exit criterion:** SH + SB kernels accept `C ∈ {3, 4}` at the C API layer; no Python-side work has flipped model shapes yet.

### Phase 3 — Wire `color_channels=4` end-to-end in Python
Goal: `python train.py -s lego --color_channels 4 --white_background --eval` runs and produces an RGBA render.

- [ ] `create_from_pcd`: init `sb_params` with the extra alpha slot. Alpha init = inverse-softplus(1.0) ≈ log(e - 1) so post-activation alpha starts at 1.0 (fully opaque, matches upstream behavior at t=0).
- [ ] `_sh0` gets 4 channels; init the alpha channel from the ground-truth alpha at init points if available, else from 1.0.
- [ ] `render` composes `rgba = rasterize(color_channels=4)`. Background tensor has 4 elements.
- [ ] `train.py` loss: L1 + SSIM over all 4 channels for NeRF-synthetic; drop the alpha channel from loss when the training image is 3-channel (COLMAP/MipNeRF360).
- [ ] Alpha multiplicative rule (**Design A2**): at loss time, the model's rendered image has already been composited with `alpha_geometric * alpha_color`. Don't apply the geometric opacity again.
- [ ] Dataset reader: return a 4-channel image when the source has one (NeRF-synthetic PNGs), else return 3-channel and let the loss handle it.

**Verification:**
- Smoke run: `--color_channels 4` on `lego`, 200 iters, no crashes; section 9 shows `sh0` last-dim = 4 and `sb_params` slot count = 7.
- 30k-iter run on `lego --color_channels 4 --eval`: PSNR ≥ 3c-baseline within tolerance; alpha-MSE against ground-truth alpha lower than "flat alpha=1" baseline.

**Exit criterion:** both `--color_channels 3` and `--color_channels 4` train to completion. 3c metrics regression-free vs Phase 0; 4c metrics beat 3c on NeRF-synthetic alpha-MSE.

### Phase 4 — Evaluation harness for both modes
- [ ] `benchmark.py`: sweep both `color_channels` on the paper datasets. Add rows to the results table.
- [ ] `compress.py` / `save_png`: ensure PNG compression handles the 4-channel case; measure the compression ratio delta.
- [ ] `eval.py`: metrics matrix per channel count.

**Exit criterion:** results table in `rgba_extension.md` populated with 3c-baseline vs 4c numbers on NeRF-synthetic + at least one MipNeRF360 scene.

### Phase 5 — Inference optimization (see `inference_optimization.md`)
Not blocking on 3c/4c parity; can begin in parallel once Phase 3 lands.

## Open questions (to be resolved as we work)

- **SB CUDA kernel location.** I haven't yet grepped for the SB-specific kernel (as opposed to SH). Where is it? What does its channel assumption look like? Must inspect before Phase 2.
- **Alpha channel init from SfM point cloud.** The COLMAP init has no alpha info. Should we init alpha from a per-point occupancy heuristic (all opaque = 1) or from view-space alpha aggregation? Start with opaque=1; revisit if it hurts.
- **Does the SH `_shN` residual still make sense in 4c mode?** If `_shN` was carrying color-only high-frequency detail, extending it to alpha is defensible; if it was vestigial, we can skip that expansion. Test both: (a) `_shN` 4-channel, (b) `_shN` stays 3-channel and alpha only rides on `_sh0` + SB.
- **Rasterization background composition.** When we have both geometric opacity and color-space alpha, the compositing equation is `C_pixel = Σ_i T_i · (α_geom_i · α_color_i) · c_i + (1 - Σ_i T_i · α_geom_i · α_color_i) · c_bg`. Check that gsplat's rasterizer is doing this — it may only apply `α_geom` and leave `α_color` in the color channels for post-hoc composition.

## Journal reference

Each phase should land with a commit and a journal entry in `docs/journal.md`. Format:

```
## YYYY-MM-DD — Phase N: <one-line>
Commit: <hash>
What changed: ...
Verification: <smoke result, 30k-iter numbers if applicable>
Surprises: ...
Next: <next phase or open question>
```
