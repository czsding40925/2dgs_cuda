# TODO

Current state: forward pass works end-to-end (project → tile sort → rasterize → loss).
The training loop now executes:
forward render → photometric loss → backward rasterizer → backward projection
→ SH coefficient backward → Adam step → basic prune/clone/split densification.
Checkpoint export, a live point-cloud viewer, and a rendered subprocess-based
viewer are now in place too.
The main remaining work is tightening the render viewer path, camera-flip cleanup,
longer-run training validation, and optional SH-direction gradients back into means.
Preview snapshots and the runnable `garden` training command are already in place.

---

## Critical path to a training loop

### 0. Interactive viewer
**Priority: top** — rendered viewing works; next step is making it cleaner and faster.

Current state:
- `train.cu` can now export standard Gaussian-splatting PLY checkpoints via
  `--save-ply` and `--save-ply-every`
- `python/viser_point_cloud_viewer.py` can watch that checkpoint and show the
  current Gaussian centers/colors live in a browser over `viser`
- `python/viser_splat_viewer.py` now renders actual splats by sending browser
  camera poses into a persistent CUDA subprocess over stdin/stdout
- this is enough for baseline live inspection on an AWS box with SSH port forwarding

Next viewer goal:
- use Python `viser` + `nerfview` as the UI / camera frontend
- replace the subprocess bridge with a thinner direct render API around the
  existing CUDA path
- return rendered RGB images from the CUDA backend to the viewer callback

Why this path:
- `viser` is Python-first, but it works well as a viewer shell around a CUDA renderer
- `gsplat` already uses `viser` / `nerfview` for interactive viewing
- 2DGS is compatible with this model because the viewer only needs camera controls
  plus a render callback; it does not need a native "2DGS primitive" in the viewer

Next slice:
- checkpoint viewer mode: keep the current rendered viewer but move it off the
  subprocess/file bridge
- training viewer mode: keep reflecting the current splat state live, but with
  lower latency and tighter binding
- expose RGB first, then add depth / normals as switchable render modes

Still needed:
- decide whether the direct render binding should be `pybind11` or a smaller C API / IPC layer
- define the minimal render API boundary: splat state, camera pose, intrinsics, image size
- confirm whether to reuse the current `render_to_file` path or factor out a dedicated
  reusable render function for Python access

---

### 1. Fix camera vertical flip
**Priority: high** — affects all rendered outputs.

Renders are vertically flipped vs ground truth (confirmed by MSE comparison on garden
cam 130: MSE=0.094 normal, MSE=0.076 against flipped GT). All upstream math has been
verified correct against gsplat — the bug is likely inside the rasterizer tile/pixel
index layout.

Things to check:
- `rasterize_fwd.cu`: `blockIdx.x` drives tile row — confirm pixel row = `blockIdx.x *
  TILE_SIZE + threadIdx.x`. Currently the grid is `dim3(tile_height, tile_width)` with
  `blockIdx.x` = row. Verify it matches how `tile_offsets` is indexed in `intersect_tile.cu`.
- Quick workaround: flip `render_colors` row-order in `render_to_file` before PNG write
  (`for row in 0..H/2: swap(row, H-1-row)`) and check if the result matches GT.

---

### 2. Backward rasterizer
**Priority: medium** — wired in `train.cu`; remaining work is validation.

Reference: `gsplat/RasterizeToPixels2DGSBwd.cu` and
`LichtFeld-Studio/src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu`.

Status:
- `kernels/rasterize_bwd.cu` implements the back-to-front replay and outputs
  `grad_ray_transforms`, `grad_opacities`, `grad_colors`, and screen-space mean grads.
- `kernels/rasterize_fwd.cu` now initializes `last_ids=-1` for empty pixels so the
  backward replay can safely skip tiles with no contributors.

Still needed:
- Compare gradients against `gsplat/RasterizeToPixels2DGSBwd.cu` on a small scene
- Tighten `make run-gradcheck` tolerances once GPU runtime testing is available

---

### 3. Backward projection
**Priority: medium** — wired in `train.cu`; next step is validation.

Reference: `gsplat/Projection2DGSPacked.cu` backward section or
`textured_gaussians/cuda/csrc/fully_fused_projection_2dgs_bwd.cu`.

Status:
- `kernels/projection_2dgs_bwd.cu` now maps
  `grad_ray_transforms + grad_means2d + grad_depths + grad_normals`
  to `grad_means`, `grad_rotation`, and `grad_scaling`.
- The implementation follows the local `projection_2dgs.cu` conventions and uses
  gsplat's `means2d -> ray_transforms` VJP as the reference.

Still needed:
- Add a richer numerical gradient check beyond the current smoke test
- Decide whether to keep `grad_depths/grad_normals` unused in the first trainer

---

### 4. Backward SH evaluation
**Priority: medium** — coefficient backward is now wired; direction gradients are still skipped.

Status:
- `train.cu` now includes `sh_backward_kernel`, which maps `grad_colors`
  to `grad_sh0` and `grad_shN` using the same basis functions as the forward pass.
- Clamp saturation is respected: channels clamped to 0 or 1 contribute zero SH gradient.

Still needed:
- Optionally propagate SH view-direction gradients into means for a more complete trainer
- Expand `make run-gradcheck` to cover more SH coefficients or clamp-edge cases

---

### 5. Wire up the optimizer and training loop
**Priority: medium** — the basic backward/step path is done.

Status:
- `train.cu` allocates per-Gaussian gradient buffers, calls the backward kernels,
  builds `SplatGradients`, and runs `optimizer.step(splats, grads, opt_cfg)`.
- The trainer now has terminal progress logging plus optional PNG previews via
  `--log-every`, `--preview-every`, and `--preview-out`.
- The README now includes a runnable `garden` training command.

Still needed:
- Run longer training on a GPU and inspect whether loss decreases stably
- Add lightweight logging for gradient norms or parameter deltas when debugging

---

### 6. Densification
**Priority: high** — basic prune/clone/split is in; tuning is still missing.

Reference: `LichtFeld-Studio/src/training/kernels/densification_kernels.cu`.

- Accumulate `grad_means2d_abs` over a configurable window
- **Clone**: small-scale, above-average accumulated screen-gradient Gaussians are duplicated
- **Split**: large-scale, above-average accumulated screen-gradient Gaussians are replaced
  by two smaller children along the dominant local tangent axis
- **Prune**: opacity below threshold → remove
- Run between iterations 250–15000, every 50 iters by default
- `SplatData`, `ForwardBuffers`, and `SplatAdam` are now rebuilt/remapped when `N` changes
- The adaptive densification gate is intentionally looser now (`0.5 * mean_grad`)
  and the per-pass action cap is higher so current runs can grow splat count faster

Still needed:
- Tune prune/clone/split thresholds against longer runs
- Decide whether split opacity/scale rules should stay deterministic or move closer to the
  randomized reference implementations
- Reconcile the absolute image-plane gradient scale with the reference trainers:
  the local `garden` run currently reports early `max_grad` around `1e-8`, so the literal
  `2e-4` / `6e-4` thresholds from 2DGS / gsplat do not transfer directly yet
- Add logging summaries for `N`, prune count, clone count, and split count over long runs

Next session note:
- Run a longer `garden` training check with the default schedule and inspect whether densification
  improves loss/preview quality instead of growing `N` too aggressively.
- If growth is too aggressive, start by lowering the action cap in `train.cu::maybe_densify`
  and/or raising the gradient threshold above the current `mean_grad` heuristic.
- If split placement looks too rigid, switch the deterministic tangent-axis offset to a small
  randomized offset modeled after the local LichtFeld-Studio reference.
- On the performance side, the next speed item is persistent tile-intersection scratch
  in `kernels/intersect_tile.cu`, followed immediately by CUDA event timing around SH eval,
  projection, tile intersection, rasterization, loss, backward, and Adam so the next
  optimization decision is data-driven.

---

## Render quality

### 7. Orbit render camera flip workaround
The orbit frames (`--orbit N`) show correct scene content from different angles but are
vertically flipped due to issue #1. Once #1 is fixed, orbit will look correct.
Alternatively, add a `--flip-y` flag to `render_orbit` that reverses pixel rows before saving.

### 8. Orbit start angle aligned to DSC07959
The first orbit frame starts at an arbitrary `theta=0` angle in the X-world direction.
To start from DSC07959's viewpoint (the reference "good view"):
- Project DSC07959's camera position onto the orbit plane
- Set `theta_0 = atan2(dot(pos, a2), dot(pos, a1))`

---

## Phase 2 — PyTorch integration

### 9. `at::Tensor` wrappers
Wrap each CUDA kernel launch to accept `at::Tensor` at the boundary:

```cpp
void launch_rasterize_fwd(at::Tensor means2d, at::Tensor ray_transforms,
                           at::Tensor opacities, at::Tensor colors, …)
```

Kernels unchanged — only the call site changes. Reference: `gsplat/` Python binding layer.

### 10. pybind11 bindings (`python/bindings.cpp`)
Expose the `at::Tensor` wrappers to Python so the training loop can be written in
PyTorch with `autograd.Function`. Skeleton is in `project/python/bindings.cpp`.

---

## Phase 3 — Extensions

### 11. Billboard / Textured Gaussian support
See CLAUDE.md §Primitive type design. Add `PrimitiveType` enum and `if constexpr`
branches in the rasterizer for `Gaussian` / `Billboard` / `TexturedGaussian`.

### 12. OptiX BVH ray tracing (IRGS-style)
For relighting / indirect illumination. Reference: `IRGS/submodules/surfel_tracer/`.
Requires OptiX 7 SDK and RTX hardware.

---

## Completed

- [x] CUDA forward pipeline: `projection_2dgs` → `intersect_tile` → `rasterize_fwd`
- [x] Photometric loss: L1 + D-SSIM (`loss.cu`)
- [x] Adam optimizer with per-group learning rates (`adam.cu`)
- [x] PLY checkpoint loader (`SplatData::init_from_ply`)
- [x] Full SH evaluation (degrees 0–3, view-dependent color)
- [x] Backward rasterizer kernel (`rasterize_bwd.cu`)
- [x] Backward projection kernel (`projection_2dgs_bwd.cu`)
- [x] SH coefficient backward in `train.cu`
- [x] Adam step wired into the training loop
- [x] Finite-difference gradient check executable (`make run-gradcheck`)
- [x] Runnable `garden` training command documented in `README.md`
- [x] Progress bar and periodic PNG previews in the training loop
- [x] `SplatData::reserve` and dynamic `N` support in the trainer
- [x] Basic prune/clone densification pass with Adam/buffer remapping
- [x] Basic split densification with parent replacement and fresh optimizer state
- [x] Single-frame render (`--render out.png --cam N`)
- [x] Camera orbit sequence (`--orbit N --orbit-out prefix`)
  - Ray-convergence scene center (better than mean camera position)
  - Height offset preserved from real camera distribution
  - `look_at` bug fixed: `down = fwd × right` not `right × fwd`
