# 2DGS Kernel Reading Notes

Notes from studying the reference implementations before writing our own kernels.

## Key files to read

### gsplat (cleaner, more modular)
- `../gsplat/gsplat/cuda/_2dgs_v2.py` — Python entry points / autograd Functions
- `../gsplat/gsplat/cuda/csrc/` — C++ dispatch layer
- Look for `2dgs` in kernel names

### 2d-gaussian-splatting (original paper code)
- `../2d-gaussian-splatting/submodules/diff-surfel-rasterization/` — the actual CUDA rasterizer
- `cuda_rasterizer/forward.cu` / `backward.cu` — forward and backward passes

## Concepts to understand before writing kernels

- [ ] How 3DGS projects 3D Gaussians to 2D (the covariance projection math)
- [ ] What changes in 2DGS: ray-splat intersection instead of projected ellipses
- [ ] Tile-based rasterization: sorting Gaussians by tile, parallel per-tile rendering
- [ ] Alpha compositing in front-to-back order
- [ ] Backward pass: how gradients flow through the alpha compositing
