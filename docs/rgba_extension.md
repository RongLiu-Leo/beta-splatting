# 4-channel (RGBA) extension — research direction

## The question

Does adding a learned, per-primitive **fourth color channel** (alpha, treated as a color-space quantity) to Deformable Beta Splatting improve fidelity on scenes with real transparency (glass, foliage, hair, RGBA training data) without regressing on opaque RGB scenes?

## Why this is a real question, not a cosmetic change

The DBS/3DGS color pipeline is RGB-only. What the model calls "opacity" is a scalar per primitive that controls its geometric contribution to a ray during alpha compositing — it is not a color-space property and does not vary with view direction. Two things that a per-primitive scalar cannot capture:

1. **View-dependent transparency.** Glass looks different at grazing angles. Wet foliage's specular highlight rides *on top of* a partially transparent leaf. The SB directional lobes already give RGB this freedom; alpha doesn't have it.
2. **Sub-primitive transparency detail.** A single Beta primitive at the edge of a translucent object needs different alpha at different pixels within its footprint. Right now, the primitive has one opacity value; sub-primitive alpha variation has to be faked by placing extra primitives.

Both of these are "obvious in retrospect once you look at what SB is doing for RGB and notice A is missing."

## Design space

Three plausible shapes for the extension. We should pick **A** first as the minimum viable change; B and C are follow-ups if A works.

### A. Alpha as a fourth SB intensity — MVP

`_sb_params` grows from `(N, K, 6)` to `(N, K, 7)`: each lobe gains an alpha intensity alongside R, G, B. `_sh0` grows from 3 channels to 4. The rasterizer accumulates a 4-channel image; the fourth channel is the composited alpha.

- **Pros:** minimal parameter add (1 slot × K lobes ≈ 2 numbers/primitive at K=2), reuses the SB softplus activation, gives alpha the same view-dependent lobes RGB gets.
- **Cons:** entangles alpha with color lobe positions — if the specular highlight is in a different direction than the transparent edge, one lobe has to compromise. Also: how does the fourth channel interact with the *geometric* opacity? Two options:
  - **A1:** alpha replaces geometric opacity entirely. `_opacity` goes away. Simpler but risks fighting the MCMC densification (which is opacity-weighted).
  - **A2:** alpha is *additional*. Rendered alpha = geometric opacity × color-space alpha. Keeps MCMC intact but you're learning two multiplicative quantities that can conspire against gradient flow.

**Recommend A2 first.** MCMC densification depends on the opacity-only invariance proof — messing with `_opacity` invalidates it. Adding a multiplicative color-space alpha on top is orthogonal to that proof.

### B. Alpha as a lobe-independent view-dependent scalar

`_sb_params` unchanged at `(N, K, 6)`, but add a separate `_alpha_sb` at `(N, K_α, 4)` — its own set of alpha-only spherical lobes. Composited alpha = softplus over the alpha lobes.

- **Pros:** alpha and RGB can point their lobes in different directions.
- **Cons:** parameter count grows more; more CUDA to write. Only worth it if A shows that lobe entanglement is actually the limiting factor.

### C. Alpha as a per-primitive scalar (no view dependence)

Just add a learned `_alpha` at `(N, 1)`. This is roughly "another opacity but for color space only."

- **Pros:** trivial to implement.
- **Cons:** doesn't give alpha view dependence — the whole motivation. This ends up looking like a second opacity term with no new expressive power. **Rejected** as the primary direction; noted here so we don't accidentally reinvent it.

## What needs to change in code (Path A2)

Roughly:

- `scene/beta_model.py`
  - `__init__`: change `_sh0` init shape channel count from 3 → 4; `_sb_params` last dim 6 → 7.
  - `create_from_pcd` (currently around L186): SB slot layout `[R,G,B,θ,φ,β]` → `[R,G,B,A,θ,φ,β]`. Update the theta/phi/beta index writes (currently `[..., 3]`, `[..., 4]`, `[..., 5]`) to `[..., 4]`, `[..., 5]`, `[..., 6]`.
  - `sb_params_activation`: softplus over `[..., :4]` instead of `[..., :3]`.
  - `render`: gsplat call needs to return a 4-channel image; unpack RGB and A separately.
  - `save_ply` / `load_ply`: expand the property list for the extra channel.
- `submodules/gsplat/`
  - `spherical_harmonics.cuh`, `compute_sh_fwd.cu`, `compute_sh_bwd.cu`: template on channel count or add a 4-channel specialization.
  - `_wrapper.py`: expose the 4-channel path.
  - Rasterization kernel: color accumulation loop iterates over 4 channels; ensure shared-memory tile buffers are sized correctly.
- `train.py`
  - Loss on the RGB slice for opaque datasets; loss on the full RGBA image when ground truth is 4-channel (NeRF-synthetic).
- `arguments/__init__.py`
  - Add a `color_channels` model param (default 3, set to 4 for the RGBA runs) so we can switch without a code branch.

Concrete diagnostics are already in `notebooks/colab_smoke.ipynb` cells 14 and 18: the SH kernel probe (`spherical_harmonics(0, dirs, coeffs)` with `C=4`) is the fast fail-fast check, and the post-train PLY inspector prints `sh0` channel dim and `sb_params` slot count.

## Evaluation plan

For each candidate design, run the same matrix:

| Dataset             | Metrics                          | Purpose                                  |
|---------------------|----------------------------------|------------------------------------------|
| NeRF-synthetic (8)  | PSNR, SSIM, LPIPS, alpha-MSE     | Does the alpha channel actually help?    |
| Mip-NeRF 360 (subset)| PSNR, SSIM, LPIPS               | No regression on opaque scenes           |
| Any 1 scene         | Step time, peak GPU memory       | Cost of the fourth channel               |
| Any 1 scene         | PSNR after compress.py          | Compression still works                  |

Baseline: DBS-3c at the same primitive cap (`--cap_max 300_000` for NeRF-synthetic, `--cap_max 1_000_000` for MipNeRF-360).

## Success criteria — see also `project_overview.md`

- +0.3 PSNR on NeRF-synthetic average over DBS-3c at matched primitive count.
- Within ±0.05 PSNR of DBS-3c on Mip-NeRF 360.
- ≤ +5% training time, ≤ +10% memory.
- Compression pipeline still runs without code changes beyond channel count.

## Alternatives considered and rejected

- **Design C** (per-primitive scalar alpha) — no view dependence, reduces to "another opacity." Rejected.
- **Design A1** (alpha replaces geometric opacity) — breaks the MCMC-invariance proof that opacity-only updates preserve the target distribution. Rejected as the starting point; may reconsider later if A2 shows the two opacities always co-vary.
- **Full RGBA per SH degree instead of SB** — expensive (16 × 4 = 64 vs 12 for K=2 SB with alpha). The paper already showed SB > SH at matched param count for RGB; no reason to expect the ordering to flip for RGBA. Rejected.
