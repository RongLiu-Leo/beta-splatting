# beta-splatting-4c — documentation

This is a research working copy of **Deformable Beta Splatting** (DBS, Liu et al., USC, arXiv:2501.18630). The `-4c` suffix marks our fork's direction: extending the color model from 3-channel RGB to **4-channel RGBA** so alpha becomes a first-class per-primitive quantity alongside color.

The docs here are the working notebook for the research — not user-facing documentation for the DBS repo. The upstream README is `../README.md`.

## Contents

- **[`project_journal.md`](project_journal.md) — one-page definitive status. Read this first.**
- [`project_overview.md`](project_overview.md) — what we're building, why, and where it stands today.
- [`method_dbs.md`](method_dbs.md) — how DBS works, what it changes vs. 3DGS/3DGS-MCMC, and where the bottlenecks are.
- [`rgba_extension.md`](rgba_extension.md) — the 4-channel research direction: motivation, plan, open questions.
- [`training_notes.md`](training_notes.md) — Colab workflow, hyperparameters, run recipes.
- [`apple_silicon_options.md`](apple_silicon_options.md) — MLX/Metal ports catalog, what's usable now, and the DBS-to-Metal port question.
- [`mlx_port_plan.md`](mlx_port_plan.md) — **primary track.** Port from CUDA to MLX/Metal for local M4 Pro execution. Living document.
- [`implementation_plan.md`](implementation_plan.md) — dual 3c/4c support. Written CUDA-first before the MLX pivot; still valid for 3c/4c design decisions, but the MLX plan is what we're actually executing.
- [`inference_optimization.md`](inference_optimization.md) — **living** plan for inference-time speedups, ranked by ROI. Framework-agnostic.
- [`journal.md`](journal.md) — dated log of work, decisions, and results.

## How to keep this current

- When you finish a work session, add a dated entry to `journal.md` — even a two-line one.
- When you choose an approach over alternatives, write the alternatives you rejected and *why* in `rgba_extension.md` (or wherever fits). "What we picked" without "what we considered" is the failure mode.
- When a run finishes on Colab, paste the PSNR/SSIM/LPIPS/step-time numbers into `journal.md` under the date, and update the results table in `rgba_extension.md` if it's a milestone.
- If a piece of code stops being true (renamed function, moved file), update the doc that references it in the same commit.
