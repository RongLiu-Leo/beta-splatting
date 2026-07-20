# mlx_impl — MLX/Metal port of Deformable Beta Splatting

Status: **scaffolding** (2026-07-10). Nothing here has run yet. This directory contains the mechanical PyTorch→MLX / CUDA→MLX translations that we're confident will import and execute once MLX is installed. The hard parts (rasterizer, Metal shaders) are stubbed and phased in later.

For the full plan see `../docs/mlx_port_plan.md`.

## Layout

```
mlx_impl/
  README.md                  ← you are here
  __init__.py                ← package marker
  beta_model.py              ← MLX BetaModel: parameters + activations + PLY I/O   [scaffold]
  losses.py                  ← L1, SSIM, PSNR in MLX                                [scaffold]
  color/
    __init__.py
    spherical_beta.py        ← SB kernel forward (autograd handles bwd)             [scaffold]
    spherical_harmonics.py   ← SH kernel forward (autograd handles bwd)             [TBD]
  geometry/                  ← quat→covar, world→cam, projection, Beta kernel       [TBD]
  rasterizer/
    slow.py                  ← pure-MLX soft rasterizer (Track A)                   [TBD]
    fast.py                  ← Metal-kernel rasterizer (Track B)                    [TBD]
    metal_kernels/           ← .metal shader sources                                [TBD]
  render.py                  ← top-level dispatch                                   [TBD]
  train.py                   ← MLX training loop                                    [TBD]
  eval.py                    ← MLX evaluation loop                                  [TBD]
  compat/
    ply_io.py                ← load CUDA-trained PLYs                               [TBD]
    numpy_bridge.py                                                                 [TBD]
  tests/
    test_sb_forward.py       ← first thing to run                                   [TBD]
```

## Marker meanings

- `[scaffold]` — Python written, not yet imported/run. Expected to work with minor fixups once MLX is installed. Review-ready.
- `[TBD]` — not written. Blocked on Phase X of the plan (see `../docs/mlx_port_plan.md`).

## Getting to first import (once memory is freed)

```bash
# 1. Xcode CLT for Metal toolchain (~1 GB, one-time)
xcode-select --install

# 2. Fresh venv (Python 3.11 or 3.12 required)
python3.12 -m venv .venv-mlx
source .venv-mlx/bin/activate

# 3. Deps (no PyTorch, no CUDA)
pip install mlx numpy plyfile Pillow imageio tqdm tyro tensorboard \
            opencv-python matplotlib pandas tabulate scikit-learn

# 4. Sanity check
python -c "import mlx.core as mx; print('metal:', mx.metal.is_available())"

# 5. First test — SB kernel forward parity
python -m mlx_impl.tests.test_sb_forward
```

## Design notes

- **Not a PyTorch shim.** We use MLX types directly (`mx.array`) — no wrapper types. This makes autograd behave and avoids the double-conversion overhead that plagues MPS-backed PyTorch.
- **Two rasterizer implementations coexist.** `slow.py` for correctness (pure MLX, differentiable via autograd); `fast.py` for performance (Metal shaders with hand-written vjp). Track A validates Track B.
- **The CUDA path stays intact.** `submodules/gsplat/`, `scene/beta_model.py`, `train.py` etc. are untouched. Users pick backend via a flag. Removing the CUDA path is a separate decision after the MLX path proves out.
- **Interop with the existing loaders.** `scene/dataset_readers.py` returns numpy arrays; we accept those. No changes needed to `arguments/` or the cameras code.

## Known unknowns

Documented in `../docs/mlx_port_plan.md#risk-register`. Highlights:
- Metal shader backward correctness (finite-diff check every commit).
- MLX autograd through some ops in Track A — may need workarounds discovered on first run.
- Macos version compatibility (should be fine on 25.4/Sequoia).
