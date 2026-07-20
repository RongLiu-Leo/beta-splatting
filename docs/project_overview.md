# Project overview

## What this repo is

A local research fork of **Deformable Beta Splatting** (DBS), based on the [official repo](https://github.com/RongLiu-Leo/beta-splatting) by Rong Liu, Dylan Sun, Meida Chen, Yue Wang, and Andrew Feng (USC). Upstream paper: [arXiv:2501.18630](https://arxiv.org/abs/2501.18630).

DBS is a radiance-field method in the 3D Gaussian Splatting (3DGS) family. It replaces two of 3DGS's fixed choices with learnable, more expressive alternatives:

- **Geometry kernel:** Gaussian → **Beta kernel** (bounded support, adaptive frequency via a learned shape parameter β).
- **Color encoding:** low-order Spherical Harmonics → **Spherical Betas** (a small set of directional lobes parameterized by softplus-activated intensities plus θ, φ direction and β sharpness).

DBS also refactors densification: instead of the ad-hoc gradient/opacity heuristics from 3DGS, it uses opacity-only MCMC updates whose distribution invariance is kernel-independent. That gives DBS the same MCMC guarantees as 3DGS-MCMC while working with Beta kernels.

Reported upstream (Mip-NeRF 360, average across scenes): PSNR 28.75, ~45% fewer parameters than 3DGS-MCMC, ~1.5× faster rendering.

## What our fork is doing

**Goal:** replace the 3-channel (RGB) color pipeline with **4-channel (RGBA)** so alpha is learned per-primitive as a color-space quantity, not just as the geometric opacity.

**Why alpha as a color-space channel?**
- 3DGS/DBS conflates two things in "opacity": the geometric contribution of a primitive to a ray (occupancy) and the color-space transparency of the material (glass, foliage, hair). Learning a spatially varying, view-dependent alpha alongside RGB should give the model a place to store the second one.
- Downstream (compositing, matting, alpha-aware losses on RGBA training data like NeRF-synthetic) needs a real alpha, not a thresholded opacity.
- Cheapest possible research probe: does adding the fourth SH/SB channel improve NeRF-synthetic PSNR without hurting Mip-NeRF 360 performance? If yes, dig in.

**Status (2026-07-10):** none of the 4-channel code is landed yet. The only forward-looking artifact is `notebooks/colab_smoke.ipynb`, which sets up a Colab training harness with a Google-Drive-cached gsplat wheel and probes the SH kernel for the target channel count. When the CUDA changes exist, that probe flips from FAIL→OK on `EXPECT_CHANNELS=4`.

The main branch is unmodified from upstream except for the notebook. See `journal.md` for the dated log.

## Repo layout (research-relevant)

- `train.py` — training loop. L1 + SSIM loss, opacity/scale regularization inside the densification window, MCMC-style relocation + noise injection.
- `scene/beta_model.py` — the `BetaModel` class. Parameter tensors (`_xyz`, `_sh0`, `_shN`, `_sb_params`, `_scaling`, `_rotation`, `_opacity`, `_beta`) and their activations live here. **This is where the 4-channel work concentrates.**
- `scene/beta_viewer.py` — viser-based interactive viewer.
- `submodules/gsplat/` — CUDA kernels (rasterization, SH forward/backward). 4-channel work needs CUDA edits: `compute_sh_fwd.cu`, `spherical_harmonics.cuh`, `_wrapper.py`.
- `arguments/__init__.py` — hyperparameter groups (`ModelParams`, `OptimizationParams`, `ViewerParams`).
- `notebooks/colab_smoke.ipynb` — Colab bring-up + smoke train.
- `lego/` — NeRF-synthetic Lego scene, ships with the repo, has real RGBA PNGs — useful for the 4-channel work.
- `docs/` — this folder.

## Where we run things

- **Local (Mac, M4 Pro / 24 GB):** editing, git, docs. Cannot train DBS locally today because DBS's CUDA kernels have no Metal port. Can run baseline 3DGS locally via `msplat` for honest comparison; can view trained PLYs via `MetalSplatter`. See `apple_silicon_options.md`.
- **Colab:** training and all CUDA compilation for DBS itself. Free T4 for smoke tests; L4/A100 (Pro compute units) for real runs. Wheel cache lives on Google Drive keyed by the gsplat CUDA source hash — a Python-only edit reuses the cached wheel and finishes in ~1 min; a `.cu` edit triggers a 5–10 min rebuild.
- **Colab access from Claude Code:** none. Claude runs on the Mac and cannot open Drive, run notebook cells, or read GPU logs. Copy relevant outputs (`nvidia-smi`, training metrics, tracebacks) back into `journal.md` after a run so the docs stay grounded in observed reality.
- **Future direction:** porting DBS's Beta kernel + SB color + MCMC ops from CUDA to Metal/MLX would enable local DBS training on this machine. See `apple_silicon_options.md#porting-dbs-to-metal`. Not started; parked behind the 4-channel research question.

## Benchmarks we care about

- **NeRF-synthetic** (`lego`, `chair`, `drums`, `ficus`, `hotdog`, `materials`, `mic`, `ship`): small, RGBA, cheap. Primary testbed for the 4-channel work because alpha exists in the ground truth.
- **Mip-NeRF 360** (garden, bicycle, ...): the "does it still work on real scenes" check. Larger, RGB.
- **Tanks & Temples**, **Deep Blending**: only if there's a paper-track hypothesis worth defending.

## Success criteria for the 4-channel direction

Concrete numbers to hit before we call the 4-channel extension "worth landing":

1. **No regression on RGB scenes.** DBS-full 4c on Mip-NeRF 360 within ±0.05 PSNR of DBS-full 3c reported upstream (28.75).
2. **Meaningful gain on RGBA scenes.** DBS-4c on NeRF-synthetic average PSNR ≥ +0.3 over DBS-3c at matched primitive count, driven by cleaner alpha at silhouettes.
3. **No worse than +5% training time, no worse than +10% memory.** The fourth channel is cheap; anything worse means we've done something wrong.
4. **Compression still works.** `compress.py` produces a PNG-folder output whose reloaded metrics are within tolerance of upstream's compression loss (~0.15 PSNR).
