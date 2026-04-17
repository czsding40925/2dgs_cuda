# TODO

Current state: end-to-end training works on real scenes (garden, bicycle).
Forward → loss → backward → Adam → densification all wired and producing
improving renders. Quality is decent but sub-par vs reference; main remaining
work is the 2DGS-specific regularization losses and longer-run tuning.

---

## Render quality

### 1. Distortion loss  ← next priority
**From the 2DGS paper. `dist_lambda=1e-2`, start at iter 3000.**

Penalizes the spread of depth weights along each ray (MipNeRF360 formula).
Prevents floaters and sharpens depth boundaries.

Formula accumulated per-pixel in the rasterizer inner loop (front-to-back):
```
distort += 2 * (vis * depth * (1 - T) - vis * accum_vis_depth)
accum_vis_depth += vis * depth
```

What needs to change:
- `kernels/rasterize_fwd.cu`: load per-Gaussian `depths[]` into shared memory;
  accumulate `distort` + `accum_vis_depth`; write `render_distort[pix_id]` output
- `ForwardBuffers` in `train.cu`: add `float* render_distort`
- `kernels/rasterize_bwd.cu`: backprop `d(distortion)/d(vis_i)` and `d(distortion)/d(depth_i)`
- `train.cu`: add `dist_lambda * mean(render_distort)` to loss, gated on `iter > 3000`

Reference: `gsplat/gsplat/cuda/csrc/RasterizeToPixels2DGSFwd.cu` lines 205–399

---

### 2. Normal consistency loss
**From the 2DGS paper. `normal_lambda=5e-2`, start at iter 7000.**

Penalizes inconsistency between Gaussian surface normals and normals estimated
from the rendered depth gradient. Encourages Gaussians to lie flat on surfaces.

What needs to change:
- `kernels/rasterize_fwd.cu`: accumulate per-pixel rendered normals (weighted by `vis`);
  write `render_normals[pix_id * 3]` — same pattern as color accumulation
- `ForwardBuffers`: add `float* render_normals`, `float* render_depths`
- `kernels/rasterize_bwd.cu`: backprop normals gradient → `grad_normals[g]`
- `kernels/projection_2dgs_bwd.cu`: accept `grad_normals`, propagate to rotation/scaling
- Loss: `normal_error = 1 - dot(render_normals, normals_from_depth_gradient)`;
  needs a small kernel computing central-difference depth gradients from `render_depths`

Reference: `gsplat/examples/simple_trainer_2dgs.py` lines 616–629

---

### 3. Gradient check validation
The backward kernels pass a basic smoke test but haven't been compared
numerically against gsplat on a shared scene. Worth doing once distortion/normal
losses are added (they add new gradient paths).

---

## Densification tuning

### 4. Randomized split placement
Currently splits place children deterministically at `±0.5 * axis_scale` along
the dominant tangent axis. The reference uses a random Gaussian sample offset
(sampled from the Gaussian distribution). May improve coverage in curved regions.

---

## Viewer

### 5. Direct Python render binding
`viser_splat_viewer.py` currently bridges via subprocess + PNG file. A thin
pybind11 or C API binding would remove file I/O latency and enable live
training-time viewing. Depends on Phase 2 (at::Tensor wrappers) being in place.

---

## Phase 2 — PyTorch integration

### 6. `at::Tensor` wrappers
Wrap each CUDA kernel launch to accept `at::Tensor` at the boundary.
Kernels unchanged — only the call site changes.
Reference: `gsplat/` Python binding layer.

### 7. pybind11 bindings (`python/bindings.cpp`)
Expose the `at::Tensor` wrappers to Python so the training loop can be written
in PyTorch with `autograd.Function`. Skeleton is in `python/bindings.cpp`.

---

## Phase 3 — Extensions

### 8. Billboard / Textured Gaussian support
See CLAUDE.md §Primitive type design. Add `PrimitiveType` enum and `if constexpr`
branches in the rasterizer for `Gaussian` / `Billboard` / `TexturedGaussian`.

### 9. OptiX BVH ray tracing (IRGS-style)
For relighting / indirect illumination. Reference: `IRGS/submodules/surfel_tracer/`.
Requires OptiX 7 SDK and RTX hardware.

---

## Completed

- [x] CUDA forward pipeline: `projection_2dgs` → `intersect_tile` → `rasterize_fwd`
- [x] Photometric loss: L1 + D-SSIM (`loss.cu`)
- [x] Adam optimizer with per-group learning rates (`adam.cu`)
- [x] Adam epsilon fixed to `1e-15` (matches gsplat/PyTorch 3DGS; `1e-8` over-damps updates)
- [x] PLY checkpoint loader (`SplatData::init_from_ply`)
- [x] Full SH evaluation (degrees 0–3, view-dependent color)
- [x] Backward rasterizer kernel (`rasterize_bwd.cu`)
- [x] Backward projection kernel (`projection_2dgs_bwd.cu`)
- [x] SH coefficient backward in `train.cu`
- [x] Adam step wired into the training loop
- [x] Finite-difference gradient check executable (`make run-gradcheck`)
- [x] Progress bar and periodic PNG previews in the training loop
- [x] `SplatData::reserve` and dynamic `N` support in the trainer
- [x] Densification: prune / clone / split with Adam/buffer remapping
- [x] Densification: per-Gaussian visibility count for gradient averaging (mirrors gsplat)
- [x] Densification: adaptive grad threshold (`mean_grad * mult`), `max_gaussians` ceiling
- [x] Densification: random camera selection, exponential position LR decay
- [x] Densification defaults tuned to match gsplat (`every=100`, `start=500`, `prune_alpha=0.005`)
- [x] Single-frame render (`--render out.png --cam N`)
- [x] Serve-render mode (`--serve-render`) for viser subprocess viewer
- [x] Camera orbit sequence (`--orbit N --orbit-out prefix`)
- [x] `viser_splat_viewer.py` — subprocess-based rendered viewer with flip correction
- [x] `gsplat_viewer_from_ply.py` — inline gsplat viewer with scene-aware initial camera
- [x] `train.cu` refactored: `Config`/`parse_args` → `train_config.hpp`, densify block → `kernels/densify.cuh`
- [x] Git repo initialized and pushed to github.com/czsding40925/2dgs_cuda
