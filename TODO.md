# TODO

Current state: end-to-end training works on real scenes (garden, bicycle).
Forward → loss → backward → Adam → densification all wired and producing
improving renders. Distortion loss, expected-depth normal consistency, and their
gradient checks are now in place. Quality is decent but still behind reference;
main remaining work is longer-run tuning and a few parity features.

---

## Render quality

### 1. Median-depth parity
Current normal regularization uses expected depth only. The remaining 2DGS parity
item is adding median depth / `depth_ratio` support so the rendered normal target
can interpolate between expected and median depth like the reference.

Reference: `gsplat/examples/simple_trainer_2dgs.py`

---

### 2. Longer-run tuning
The new regularizers are wired with the standard 2DGS defaults:
- `dist_lambda=1e-2`, `dist_start_iter=3000`
- `normal_lambda=5e-2`, `normal_start_iter=7000`

What remains is empirical tuning on longer runs:
- check whether `garden` and `bicycle` want the same start iterations
- verify the current weights versus floaters / over-smoothing tradeoff
- inspect rendered depth and normals during training, not just RGB

---

### 3. Cross-check against reference trainers
The local finite-difference checks pass, but it is still worth comparing a short
shared-scene run against `2d-gaussian-splatting` or `gsplat` and checking:
- loss magnitudes over time
- rendered normals / depth maps
- splat-count growth after densification

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
- [x] 2DGS distortion loss (`render_distort`, trainer wiring, backward path)
- [x] 2DGS expected-depth normal consistency loss
- [x] Finite-difference checks for rasterizer aux buffers and geometry-loss gradients
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
