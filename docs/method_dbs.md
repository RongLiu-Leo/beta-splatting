# Method notes: Deformable Beta Splatting

Reference: Liu, Sun, Chen, Wang, Feng. *Deformable Beta Splatting.* arXiv:2501.18630, 2025.

This is our working notes on how DBS actually functions, what it replaces from the 3DGS lineage, and where the sharp edges are. Read this alongside the paper — it doesn't replace it, it captures the parts we return to when making design decisions.

## Lineage

```
NeRF (2020)             volumetric field, per-ray MLP query, slow
  ↓
3DGS (Kerbl 2023)       explicit primitives, splatting rasterizer, real-time
  ↓                     but: Gaussian tails, low-order SH, heuristic densification
3DGS-MCMC (Kheradmand   reframes densification as MCMC over an opacity-weighted
  2024)                  posterior; distribution-preserving relocation
  ↓
DBS (Liu 2025)          swap kernel (Gaussian → Beta), swap color (SH → SB),
                        prove MCMC works for any kernel with opacity-only updates
```

## What DBS changes

### 1. Geometry kernel: Beta instead of Gaussian

**Gaussian in 3DGS.** Each primitive contributes $\alpha \cdot \exp(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu))$. Unbounded support — every primitive nominally contributes to every pixel. Practical rasterization uses a 3σ cutoff, but the tails still smear.

**Beta kernel in DBS.** Same $\mu, \Sigma$, but the falloff is a beta-distribution-shaped bump with a learned shape parameter $\beta$:
- Bounded support: contribution is exactly zero outside a compact region — no tail smearing, no per-pixel cost from irrelevant primitives.
- Adaptive frequency: small $\beta$ → wide, low-frequency bump; large $\beta$ → sharp, high-frequency bump. One primitive can represent flat regions or sharp edges by shifting $\beta$ during training.

In code: `_beta` is the learned per-primitive parameter; activation is `4.0 * exp(beta)` (`beta_activation`, `scene/beta_model.py:52-53`). The Beta kernel evaluation itself is in the gsplat CUDA extension.

**Why this matters for us:** with bounded support, primitives are localized. That makes the color model responsible for less spatial-blending work, which is one of the reasons the SB color model (below) can afford to be low-order.

### 2. Color encoding: Spherical Betas instead of Spherical Harmonics

**SH in 3DGS.** For each primitive, $(l+1)^2$ RGB coefficients up to degree $l$ (typically 3 → 16 coefficients × 3 channels = 48 numbers/primitive). SH are a fixed orthonormal basis on the sphere — good for smooth angular functions, bad for concentrated specular highlights, and their storage cost dominates the model size.

**SB in DBS.** A small number ($K$ = `sb_number`, default 2) of *learnable* directional lobes. Each lobe has:
- 3 (RGB) intensity components — post-softplus, so non-negative.
- 2 direction angles $(\theta, \phi)$.
- 1 shape parameter $\beta$ (yes, same symbol as the geometry kernel — separate parameter).

Total: $K \cdot 6$ numbers/primitive. With $K=2$ that's 12 vs SH's 48 — a 4× reduction that recovers or improves fidelity because the lobes are placed where they're needed instead of on a fixed basis.

In code: `_sb_params` has shape `(N, K, 6)` — `[R,G,B,θ,φ,β]` per lobe (`scene/beta_model.py:186-199`). Softplus is applied to the first three channels via `sb_params_activation`. There is also `_sh0` (DC term) and `_shN` (higher-order residual) — the current codebase keeps a small SH tail alongside SB, which the paper uses as the low-frequency baseline that SB adds to.

**Bottleneck (why we care):** SB is 3-channel by construction. Our 4-channel extension has to decide whether alpha is a fourth intensity per lobe (`(N, K, 7)` → RGBA + θ, φ, β) or a lobe-independent scalar per primitive with a separate spatial variation. See `rgba_extension.md`.

### 3. Densification: opacity-only MCMC

**3DGS.** Split/clone/prune primitives based on gradient magnitude and opacity heuristics. Fragile; hand-tuned per dataset.

**3DGS-MCMC.** Reinterpret training as MCMC sampling from an opacity-weighted posterior over primitive configurations. "Relocation" moves low-opacity primitives to high-error regions in a way that preserves the invariant distribution — with a correction term specific to the Gaussian kernel.

**DBS.** Proves the correction term depends only on opacity, not on the kernel shape. So the same MCMC relocation works for Beta kernels (or any bounded-support kernel) unmodified. In code:

```python
# train.py, inside densification block
dead_mask = (beta_model.get_opacity <= 0.005).squeeze(-1)
beta_model.relocate_gs(dead_mask=dead_mask)   # opacity-only MCMC relocation
beta_model.add_new_gs(cap_max=args.cap_max)   # grow up to cap_max primitives

# Post-relocation position noise, scaled by covariance and (1-opacity)^100
noise = randn * (1 - opacity)**100 * noise_lr * xyz_lr
noise = actual_covariance @ noise
_xyz.add_(noise)
```

The `(1 - opacity)^100` factor is the invariant-distribution weighting: high-opacity primitives (well-placed) barely move, low-opacity ones (still exploring) get large kicks.

## Reported bottlenecks and where our attention should go

From the paper's ablations and our reading of the code:

1. **`sb_number` is a modest lever.** Going from K=2 → K=4 improves specular scenes but roughly doubles the color-side memory. The paper's default K=2 is a good speed/quality balance.
2. **Beta kernel bounded support saves memory but costs CUDA complexity.** The gsplat fork has custom kernel code for evaluation and tile assignment; any 4-channel work has to touch these files. This is where our fork will spend most of the CUDA time.
3. **Opacity regularization is doing a lot of the work.** `opacity_reg=0.01` inside the densification window is what keeps the primitive count reasonable — the MCMC dynamics assume this. Increasing it kills quality fast; decreasing it explodes memory. Don't touch it without a plan.
4. **Compression is post-hoc.** `compress.py` sorts primitives by a PLAS space-filling curve and PNG-encodes attributes. The 6× compression is because the model is small; if we add channels, the compression ratio drops proportionally.

## Improvements over previous frameworks — one-sentence versions

- **vs. 3DGS:** better fidelity per parameter, bounded-support kernels, no hand-tuned densification. Same rendering paradigm.
- **vs. 3DGS-MCMC:** same MCMC guarantees but works for any kernel; ~45% fewer parameters and ~1.5× faster at matched quality.
- **vs. gsplat baseline:** DBS ships as a `submodules/gsplat/` fork with the Beta kernel and SB color code added; upstream gsplat has neither.

## Open questions we don't yet have answers to

- Does SB benefit from progressive lobe count (start K=1, grow to K=2, K=4)?
- Is the geometry `_beta` correlated with the color-lobe `_beta`? If yes, a shared parameterization could save 1 slot/primitive.
- Does the model actually use the SH tail (`_shN`), or is it vestigial? If unused, removing it is free memory.
- **Ours:** is a per-primitive scalar alpha enough, or does alpha need view-dependence like the RGB channels get from SB? See `rgba_extension.md`.
