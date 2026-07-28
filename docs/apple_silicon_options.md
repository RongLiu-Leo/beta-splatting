# Apple Silicon local-training options

Researched 2026-07-10. This machine is an **M4 Pro / 24 GB unified memory**. The 3DGS ecosystem on Apple Silicon has matured; several ports/apps now exist. This doc catalogs them and — critically — evaluates each against **our actual need**, which is running **DBS** (not baseline 3DGS) on this hardware.

## Bottom line first

Baseline 3D Gaussian Splatting has usable Apple Silicon ports. **Deformable Beta Splatting does not.** DBS's `submodules/gsplat/` is a CUDA fork containing the Beta kernel, Spherical-Betas color model, and MCMC relocation code — none of the Apple Silicon ports replicate those additions. Consequences:

- We **cannot** just swap the submodule for `gsplat-mps` or `msplat` and get DBS locally. That would give us upstream gsplat on Metal, without the paper's contributions.
- Colab (CUDA) stays the training path for DBS itself until (unless) we port the DBS-specific kernels to Metal/MLX. That is a real research direction on its own — see "Porting DBS to Metal" below.
- Apple Silicon options **are** useful right now for: (a) baseline-3DGS comparison runs, (b) local viewers, (c) COLMAP-free scene creation via RadianceKit.

## The projects, evaluated

### `rayanht/msplat` — Metal-native training engine
- **Repo:** https://github.com/rayanht/msplat
- **What it is:** Full 3DGS training pipeline as fused Metal compute shaders. No PyTorch, no CUDA. Python + Swift + C++ bindings, standalone CLI. Requires macOS 14+, Apple Silicon.
- **Reported speed:** trains a full Mip-NeRF 360 scene in ~70 s on M4 Max; renders ~350 FPS.
- **Usable for us for:**
  - Baseline 3DGS reference numbers on the same hardware, for honest local comparison.
  - Fast iteration on scene captures.
- **Not usable for:** DBS training. No Beta kernel, no SB color, no DBS-style MCMC.
- **Interesting for a port:** cleanest reference for how a Metal 3DGS training loop should look. If we ever port DBS to Metal, msplat is the codebase to study first.

### `iffyloop/gsplat-mps` — gsplat v0.1.3 on MPS
- **Repo:** https://github.com/iffyloop/gsplat-mps
- **What it is:** A PyTorch-friendly port of `gsplat` v0.1.3 (fairly old) to Apple MPS via Metal Performance Shaders.
- **Usable for us for:** MPS-backed gsplat rasterization if we can accept the old API.
- **Caveats:** v0.1.3 predates the DBS gsplat fork by a long time. The DBS repo is on a newer gsplat with different APIs. Bringing DBS forward onto gsplat-mps means porting *both* the DBS additions *and* rebasing to an older gsplat surface — worse than starting fresh.

### `ghif/splat-apple` — MLX + MPS backends
- **Repo:** https://github.com/ghif/splat-apple
- **What it is:** 3DGS with both MLX and PyTorch/MPS backends selectable at runtime.
- **Usable for us for:** Same as msplat — baseline reference. MLX backend is interesting if we want to lean into Apple's differentiable framework rather than Metal directly.

### `daikiad/mlx-splat` — MLX + Metal, render-only
- **Repo:** https://github.com/daikiad/mlx-splat
- **What it is:** Minimal, render-only 3DGS on MLX + Metal. **Does not train.**
- **Usable for us for:** Reading how someone else structured an MLX Gaussian rasterizer. Not a training path.

### `scier/MetalSplatter` — viewer
- **Repo:** https://github.com/scier/MetalSplatter
- **What it is:** Metal-based Gaussian Splat renderer for macOS/iOS/visionOS. **Render only, no training.**
- **Usable for us for:** Viewing trained PLYs locally without spinning up viser + a share URL. Native macOS window, feels faster than the web viewer.

### `gaook/3dgs-viewer` — Metal viewer, real-time
- **Repo:** https://github.com/gaook/3dgs-viewer
- **What it is:** Metal viewer, claims 1.7M gaussians at interactive frame rates.
- **Usable for us for:** Same slot as MetalSplatter — pick whichever handles our PLY layout best.

### RadianceKit — macOS app
- **Site:** https://www.radiancekit.de/
- **What it is:** Commercial-quality macOS app. Photo/video → COLMAP → 3DGS training → export. Native, no cloud. Requires **macOS 26 Tahoe** + Apple Silicon; 16 GB RAM recommended.
- **Usable for us for:** Bypassing COLMAP for our own scene captures. Import photos/video → export PLY → feed to the DBS eval pipeline. Massive quality-of-life win for creating custom test scenes.
- **Not usable for:** DBS training itself. It trains baseline 3DGS, not DBS.

### `freddewitt/CorbeauSplat`
- **Repo:** https://github.com/freddewitt/CorbeauSplat
- **What it is:** All-in-one automation tool for macOS Silicon — raw video → trained splat.
- **Usable for us for:** Similar to RadianceKit but open-source. Worth a look if RadianceKit's licensing is inconvenient.

## Immediate practical use

Given the caveats above, here is what we can actually adopt now without waiting on a port:

1. **Local viewer for DBS output.** Trade the viser share URL for `MetalSplatter` or `3dgs-viewer` when inspecting a trained PLY. Keeps everything on-device, snappier interaction.
2. **Local baseline 3DGS runs for honest comparison.** Use `msplat` on this machine to produce the "same hardware, same scene, baseline 3DGS" numbers we compare DBS against. Removes the "but Colab GPU was faster/slower" excuse from any speed claim.
3. **RadianceKit for custom scene captures.** For any test scene we shoot ourselves, RadianceKit handles the COLMAP-adjacent alignment step natively — export the point cloud + camera poses, hand to DBS on Colab.

## Porting DBS to Metal — the real project

If we're serious about local DBS training on Apple Silicon (and given a 24 GB unified-memory M4 Pro, we should at least think about it), the work is roughly:

**In scope:**
- Port `submodules/gsplat/` DBS additions from CUDA to Metal. That's:
  - Beta kernel forward/backward (evaluation + gradient).
  - SB color: forward + backward for the softplus-activated intensities plus (θ, φ, β) parameters.
  - MCMC relocation ops.
  - Rasterization: tile assignment + accumulation loop, adapted from either upstream gsplat's Metal fork or `msplat`'s fused compute shaders.
- Adapt `scene/beta_model.py` to a Metal backend — swap the `from gsplat.rendering import rasterization` import path for a Metal-backed equivalent behind a runtime flag.
- Verify numerical equivalence to the CUDA path on a fixed seed / same-scene run. Delta in PSNR should be below floating-point tolerance.

**Effort estimate (rough):** kernel-side work is on the order of 2–6 weeks depending on how much of `msplat`'s scaffolding can be reused for the rasterizer. The DBS-specific kernels are smaller than the full 3DGS kernel surface so the marginal work over `msplat` is not huge if we start there.

**Why we should not start this yet:** the 4-channel research direction (see `rgba_extension.md`) is the higher-priority open question, and it will need to be validated on CUDA before we spend port effort on it. Port after we know the 4-channel design is worth keeping.

**Journal this direction under:** `docs/journal.md` if we ever kick it off, plus a new `docs/metal_port.md` at that time.

## Open questions

- Does `msplat`'s per-primitive attribute layout make sense for our Beta + SB parameters? (Probably yes; the geometry parameters are the same shapes.)
- Is MLX's autograd expressive enough for the Beta-kernel backward, or do we need custom Metal shaders for the gradient path?
- Are there licensing constraints in `msplat` / `MetalSplatter` that would affect a downstream port that we intend to open-source alongside our research?
