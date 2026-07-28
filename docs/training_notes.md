# Training notes — Colab workflow and hyperparameters

## Where things run

- **Editing / git / docs:** Mac (M4 Pro / 24 GB unified memory).
- **DBS training / CUDA compile:** Google Colab. See `notebooks/colab_smoke.ipynb`. Required because DBS's kernels are CUDA-only; no Metal port exists yet. See `apple_silicon_options.md` for the full landscape.
- **Local baseline 3DGS (for comparison only):** `msplat` on this Mac. Not a DBS substitute.
- **Local viewer:** `MetalSplatter` or `gaook/3dgs-viewer` for inspecting trained PLYs without spinning up viser.
- **Wheel cache:** Google Drive at `/content/drive/MyDrive/beta-splatting-4c/wheels/`. Wheel filename includes a 16-hex-char hash of every `.cu/.cuh/.cpp/.h/.py` file in `submodules/gsplat/`, plus torch + CUDA versions. Python-only edits → cache hit. Kernel edits → 5–10 min rebuild.
- **Model outputs:** ephemeral `/content/smoke_out` on Colab — models are big; don't dump them to Drive. If you need to keep one, `gsutil cp` or download the PLY manually.

## Colab session recipe

1. Open `notebooks/colab_smoke.ipynb` in Colab (upload it or fetch from GitHub after pushing).
2. Runtime → Change runtime type → **T4** for smoke; **L4/A100** (Pro compute units) for real runs.
3. Edit the Config cell — set `REPO_URL` to the fork, `REPO_BRANCH` to whatever branch you're testing, and `EXPECT_CHANNELS` to `3` (baseline) or `4` (RGBA build).
4. Runtime → Run all. First run: ~10 min including CUDA compile. Subsequent runs with cached wheel: ~1 min to smoke train.
5. When done, copy the interesting outputs — GPU name from `nvidia-smi`, the section 7 kernel-probe result, the section 9 tensor shapes, any PSNR you managed to log — into `docs/journal.md`.

Iteration budget you should expect:

| What changed                       | Colab time from Run All to result    |
|------------------------------------|--------------------------------------|
| Python-only, no kernel edit        | ~1 min (cache hit, 200-iter smoke)   |
| CUDA edit (`.cu`, `.cuh`, `.cpp`)  | ~7 min (cache miss, rebuild + smoke) |
| Full 30k-iter eval run (lego, L4)  | ~30–45 min                           |
| Full 30k-iter eval (garden, A100)  | ~45–90 min depending on cap_max      |

## Colab access from Claude Code

**Claude cannot open Colab, mount Drive, or execute cells.** Everything upstream of "here are the numbers" you have to do yourself. What you can hand back:

- `nvidia-smi -L` output (which GPU).
- The kernel-probe pass/fail from section 7.
- The tensor-shape dump from section 9.
- Any traceback.
- PSNR / SSIM / LPIPS / step-time from a real run.

Paste those into a message and Claude will fold them into `journal.md` under the run date.

## Hyperparameter reference

Defaults (from `arguments/__init__.py`, at the time of writing):

| Group          | Param                    | Default     | Notes                                             |
|----------------|--------------------------|-------------|---------------------------------------------------|
| Model          | `sh_degree`              | 0           | 0 means SH DC only; SB carries the color signal.  |
| Model          | `sb_number` (K lobes)    | 2           | Doubling roughly doubles color params.            |
| Model          | `cap_max`                | 1,000,000   | Hard primitive cap. 300k is the paper's NeRF-synth setting. |
| Optim          | `iterations`             | 30,000      | Full training.                                    |
| Optim          | `position_lr_init/final` | 1.6e-4 / 1.6e-6 | Exponential decay.                            |
| Optim          | `sh_lr`, `sb_params_lr`  | 2.5e-3 each | Color learning rates.                             |
| Optim          | `opacity_lr`             | 5e-2        | Fast; opacity is aggressive in MCMC.              |
| Optim          | `beta_lr`                | 1e-3        | Shape parameter.                                  |
| Optim          | `lambda_dssim`           | 0.2         | Loss = 0.8·L1 + 0.2·(1 − SSIM).                   |
| Optim          | `densify_from/until`     | 500 / 25000 | Densification window.                             |
| Optim          | `densification_interval` | 100         | MCMC relocation every 100 iters.                  |
| Optim          | `noise_lr`               | 5e4         | Position noise scale during MCMC.                 |
| Optim          | `opacity_reg`            | 0.01        | Load-bearing. Don't touch without a plan.         |
| Optim          | `scale_reg`              | 0.01        | Same.                                             |

## Common run recipes

```bash
# Smoke — 200 iters, no compress, no viewer
python train.py -s lego -m /tmp/smoke --iterations 200 --white_background --no-compress --quiet --disable_viewer

# NeRF-synthetic full run with eval
python train.py -s lego --cap_max 300000 --white_background --eval

# Mip-NeRF 360 scene
python train.py -s /path/to/garden --cap_max 1000000 --eval

# Benchmark suite (paper-style)
python benchmark.py -m360 /path/to/mipnerf360 -ns /path/to/nerfsynthetic
```

## Notebook maintenance

- The smoke notebook is checked into git. If you edit it in Colab, save-and-download it back into `notebooks/` and commit.
- **Do not commit a live GitHub PAT.** If you use one in `REPO_URL` for a private branch, strip it before saving.
- If the smoke notebook stops matching the codebase (e.g., a section 8 flag gets removed), the notebook is wrong — fix it, don't work around it.
