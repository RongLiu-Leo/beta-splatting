# Inference optimization plan

Also a **living document.** Refine as we measure.

## What we mean by "inference"

Two related but distinct workloads:

1. **Rendering a single view from a trained model** — the online case. Latency-critical. Metric: **ms/frame** and **FPS** at a fixed image resolution.
2. **Batch evaluation across many views** (`eval.py`, `benchmark.py`) — offline. Throughput-critical. Metric: **wall time to sweep all test views**.

Both benefit from the same underlying kernel optimizations; the online case additionally cares about first-frame latency (deserialization + warmup).

## Baseline we're optimizing against

Upstream DBS reports **1.5× faster than 3DGS-MCMC** at similar quality. That's already a strong baseline. We aim to hold or improve on this for the 3-channel path and to **close the gap that the 4-channel path opens** — a fourth channel is naively ~33% more color compute and ~33% more color memory bandwidth.

Concrete numbers to collect on the target GPU (Colab L4 for now, later local M4 Pro if the Metal port ever happens):

| Config                          | Scene   | Res       | Primitives | ms/frame | FPS  | Peak VRAM |
|---------------------------------|---------|-----------|------------|----------|------|-----------|
| DBS-3c (baseline)               | garden  | 1237×822  | 1.0 M      | ?        | ?    | ?         |
| DBS-3c compressed               | garden  | 1237×822  | 1.0 M      | ?        | ?    | ?         |
| DBS-4c (Phase 3)                | lego    | 800×800   | 300k       | ?        | ?    | ?         |
| DBS-4c + optimizations (target) | lego    | 800×800   | 300k       | ?        | ?    | ?         |

Populate as we run.

## Optimization directions, ranked by expected wins

Ordered by our estimate of ROI (return per hour of engineering time). Revise as we learn.

### 1. View-frustum + radius culling before rasterization

**Idea:** primitives outside the view frustum or with projected radius < 1 pixel contribute nothing. Skip them before tile assignment.

**Status:** partially in place. `radius_clip` already exists as a viewer argument (see `beta_viewer.py` and `rasterization()` call in `beta_model.py:897`). We should verify it's on by default in `train.py`/`eval.py`'s render calls too, and audit whether the frustum test happens tightly enough.

**Expected win:** 5–20% depending on scene. Larger for MipNeRF360 (wide FoV, lots of off-screen primitives) than for NeRF-synthetic (tight bounds).

**Cost:** small — mostly wiring existing args.

### 2. Lobe-count pruning based on intensity

**Idea:** SB lobes with softplus intensity below a threshold contribute negligibly to the color. At inference, skip them.

**Status:** not implemented. Sound because softplus intensities can go to near-zero during training when the model doesn't need a second lobe.

**Expected win:** if 20–40% of lobes are effectively zero (measurement pending), we get a linear speedup on the SB evaluation, which is ~30% of color-compute time. Ballpark 5–15% overall.

**Cost:** medium. Needs a per-primitive lobe mask computed once at load time, then plumbed into the SB CUDA kernel to short-circuit dead lobes.

### 3. Kernel fusion: SB + SH + composition

**Idea:** currently SH → SB → rasterization is three passes with two intermediate `[N, 3]` or `[N, 4]` writes to global memory. Fusing them into a single kernel that produces the composited color in registers eliminates those round-trips.

**Status:** not implemented; the DBS gsplat fork already fuses more than upstream gsplat, but SH+SB is a fusion opportunity that we don't think has been done.

**Expected win:** 10–20% on the color-compute side. More on the 4c path because more bytes to move.

**Cost:** high. Real CUDA work, careful register pressure management. Do this after Phase 3 (`implementation_plan.md`) so we can measure the *un*-fused 4c cost first.

### 4. FP16 / BF16 rendering path

**Idea:** at inference, drop the color pipeline from fp32 to fp16. Weights (SH/SB) also fp16 in-memory. Accumulation stays fp32 to avoid catastrophic cancellation in high-primitive-count tiles.

**Status:** not implemented. gsplat upstream has fp16 in some paths, DBS fork status unknown.

**Expected win:** 20–40% color-compute speedup and half the color memory bandwidth. Model size on disk halves too — a downstream win.

**Cost:** medium. Careful with SB β and θ/φ (they should stay fp32; they parameterize a rotation and a sharpness, both numerically sensitive). Only intensities and SH coeffs go fp16.

### 5. Occlusion-aware early termination

**Idea:** rasterization already does front-to-back accumulation and can early-exit a tile when accumulated transmittance drops below ε (typically ε=1/255). Ensure this is enabled and ε is tuned.

**Status:** upstream gsplat has this. Verify our fork hasn't disabled it. Consider raising ε at inference (paper uses 1/255, but 1/128 is often visually indistinguishable and saves ~10–15% of primitive touches on high-density scenes).

**Expected win:** 5–15% depending on scene density.

**Cost:** trivial (a flag change) if the mechanism is already there.

### 6. Precomputed view-independent color (for K=1 SB and low-order SH)

**Idea:** the DC term of SH and any SB lobe pointing at the camera contribute nearly constant color per view — precompute per-view.

**Status:** not implemented. Real win depends on how many lobes are actually view-dependent; if K=2 and both lobes are highly directional, the win is small.

**Expected win:** 3–10%. Not high priority.

### 7. Load-time PLAS reorder + tile-major layout

**Idea:** `compress.py` already sorts primitives by a PLAS space-filling curve for compression. That same ordering also improves cache locality during rasterization if the layout is preserved. Ensure the trained-and-then-saved PLY is sorted, not raw insertion order.

**Status:** compressed models are sorted; raw PLY is not.

**Expected win:** 3–8% render time for large scenes. Free if we already saved compressed.

**Cost:** trivial. Consider adding a `--sort` flag to `train.py` that runs the PLAS sort at the end of training even without compression.

### 8. Level-of-detail (LoD) rendering

**Idea:** primitives with projected radius > some threshold, or below some depth, use their coarser DC-only representation; skip SB evaluation for distant/small contributors.

**Status:** not implemented. Meaningful for very large scenes (MipNeRF360 outdoor).

**Expected win:** 10–30% for large scenes; 0% for NeRF-synthetic.

**Cost:** high. Only pursue if we care about outdoor MipNeRF360 rendering FPS specifically.

### 9. CUDA graphs for repeated-view rendering

**Idea:** capture the render call as a CUDA graph after warmup; replay for subsequent views. Eliminates Python overhead per frame.

**Status:** not implemented.

**Expected win:** major for offline batch eval (`benchmark.py`) — can be 20–40% wall time. Small for online rendering that already has a Python event loop.

**Cost:** medium. Requires the render call to be structurally identical across views (fixed primitive count, fixed image size).

## Priority order for this branch

Rough sequencing after `implementation_plan.md` Phase 3 lands:

1. **First**: (1) Frustum/radius culling audit + (5) occlusion ε tuning + (7) PLAS-sorted PLY. All are cheap and additive.
2. **Second**: (2) Lobe pruning. This is the biggest quality-preserving win for DBS specifically.
3. **Third**: (9) CUDA graphs for batch eval — big win for `benchmark.py` runtimes.
4. **Fourth**: (4) FP16 render path. Larger effort, larger win.
5. **Deferred**: (3) SB+SH kernel fusion; (6) precomputed view-independent; (8) LoD. Do only if the metrics table shows we still need speed.

## Measurement discipline

Every optimization commit should include:

- Before/after ms/frame on at least one scene, same GPU, same primitive count.
- Before/after PSNR/SSIM/LPIPS on the eval set. If quality drops more than 0.05 PSNR, the optimization is not free and we should discuss whether it's worth landing.
- Wall-time of `benchmark.py` on the same scene.

Log these in `journal.md` under the commit date.

## What we're NOT optimizing for

- **Training speed.** The 30k-iter training run cost isn't the bottleneck; inference is what matters at deployment.
- **First-time build time.** The wheel cache in `notebooks/colab_smoke.ipynb` already covers this.
- **Model size on disk.** `compress.py` already gives 6× reduction. Improvement here would be a separate research direction.
