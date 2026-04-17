# 2DGS CUDA

Learning-focused reimplementation of 2D Gaussian Splatting, built up from the CUDA kernels.
Pure C++/CUDA first — Python bindings added later once the kernels are solid.

## Structure

```
repo root/
├── train.cu                    # *** entry point — load scene, init SplatData, training loop
├── kernels/
│   ├── splat_data.cuh          # *** central data structure — owns all Gaussian GPU arrays
│   ├── projection_2dgs.cu      # kernel: means/quats/scales → ray_transforms T, depths
│   ├── projection_2dgs_bwd.cu  # kernel: grad T/means2d/depth/normals → grad means/rot/scale
│   ├── intersect_tile.cu       # kernel: projected AABBs → sorted tile/Gaussian pairs
│   ├── rasterize_fwd.cu        # kernel: tile-based forward alpha compositing
│   ├── rasterize_bwd.cu        # kernel: back-to-front replay → grad T, opacity, color
│   ├── gradient_checks.cu      # finite-difference validation for rasterize/projection/SH backward
│   ├── loss.cu                 # kernel: L1 + D-SSIM photometric loss
│   ├── adam.cu                 # kernel: raw-pointer Adam optimizer + SplatAdam state
│   ├── simple_knn.cuh          # helper: CUDA 3-NN distances for scale initialization
│   ├── simple_knn_test.cu      # test: deterministic KNN distance check
│   ├── quat_to_rotmat.cu       # kernel: quaternion → 3×3 rotation matrix (standalone)
│   ├── camera_projection.cu    # kernel: pinhole projection 3D → 2D pixels (standalone)
│   ├── colmap_reader.hpp       # header-only COLMAP binary reader (no deps)
│   ├── colmap_reader_test.cu   # test: synthetic COLMAP scene round-trip
│   └── splat_data_test.cu      # test: allocation, activations, move semantics
├── python/                     # Python-side utilities, viewer, and pybind11 experiments
│   ├── colmap_loader.py        # pure-Python COLMAP binary reader → numpy arrays
│   ├── bindings.cpp            # pybind11 wrapper exposing kernels to Python
│   ├── example.py              # end-to-end: load COLMAP → call CUDA kernel from Python
│   ├── viser_point_cloud_viewer.py  # live browser viewer for exported checkpoint PLYs
│   └── viewer_requirements.txt # Python dependency list for the viewer
├── notes/
│   └── 2dgs_kernel_reading.md  # reading notes on reference implementations
├── .vscode/launch.json         # Nsight cuda-gdb debug config
├── Makefile
└── build/                      # compiled binaries (gitignored)
```

## Workflow

### 1. Data flow from disk to GPU

```
COLMAP dataset (cameras.bin, images.bin, points3D.bin)
        │
        │  colmap_reader.hpp :: load_colmap()
        ▼
ColmapScene { cameras[], points3D[], point_colors[] }   ← host memory
        │
        │  SplatData::init_from_points()
        ▼
SplatData on GPU                                         ← device memory
  _means     [N, 3]   ← point cloud XYZ
  _rotation  [N, 4]   ← identity quaternion (1,0,0,0)
  _scaling   [N, 3]   ← log(nearest_neighbor_dist) isotropic
  _opacity   [N]      ← logit(0.1)
  _sh0       [N, 3]   ← (rgb − 0.5) / C₀   (degree-0 SH)
  _shN       [N, K]   ← zero   (higher degrees unlocked during training)
```

### 2. Architecture — two layers

```
SplatData (splat_data.cuh)          owns GPU memory, enforces storage conventions
     │  float* pointers
     ▼
CUDA kernels (.cu)                  pure math on raw pointers, apply activations inline
     │  at::Tensor wrappers (phase 2)
     ▼
Python / PyTorch training loop      feeds data, reads gradients, calls optimizer
```

Parameters are stored **raw** (pre-activation) and converted at read time inside kernels:

| Parameter | Stored as | Activation | Reason |
|-----------|-----------|------------|--------|
| opacity   | logit     | sigmoid    | unconstrained optimization → (0, 1) |
| scaling   | log-scale | exp        | unconstrained optimization → positive |
| rotation  | unnorm quat | normalize | avoids constrained optimization |

### 3. Training loop (train.cu)

Each iteration:
1. **Pick a camera** — random view from the loaded scene
2. **Forward pass** — project Gaussians, tile-intersect, rasterize → rendered image
3. **Loss** — load the matching RGB image, compute L1 + D-SSIM, then add the
   2DGS distortion and normal-consistency regularizers on their scheduled iterations
4. **Backward pass** — rasterizer → projection → SH coefficient backward
5. **Optimizer step** — Adam with per-group learning rates
6. **Densification** — prune, clone, and split by accumulated screen-space gradient

SH curriculum: `active_sh_degree` starts at 0 and steps up every 1000 iterations,
so the model first learns low-frequency color and gradually adds high-frequency detail.

### 4. Rasterization pipeline

Full pipeline from a projected set of Gaussians to a scalar loss, front-to-back
per tile. Current forward stages:

```
projection_2dgs          → ray_transforms T [N,3,3], depths [N], means2d [N,2], radii [N,2]
  ↓
tile intersection        → isect_ids [n_isects], flatten_ids [n_isects], tile_offsets [tile_h, tile_w]
  ↓  (radix sort on isect_ids)
forward rasterizer       → render_colors [H,W,C], render_alphas [H,W]
  ↓
render auxiliaries       → render_normals [H,W,3], render_depth_accum [H,W], render_distort [H,W]
  ↓
photometric + geometry loss
                         → L1 + D-SSIM + 2DGS regularizers against target image,
                           grad_render_color / grad_render_alpha / grad_render_normals / grad_render_depth / grad_render_distort
  ↓  backward rasterizer
backward rasterizer      → grad_ray_transforms, grad_opacities, grad_colors, grad_means2d
  ↓  backward projection
backward projection      → grad_means, grad_rotation, grad_scaling
  ↓
SH backward + Adam       → grad_sh0/grad_shN → update raw SplatData params
```

For backward validation, `make run-gradcheck` builds and runs a deterministic
finite-difference executable covering the active rasterizer, projection, and SH
coefficient gradients.

#### Stage 1 — Projection (`projection_2dgs.cu`)

For each Gaussian: `means/quats/scales → ray_transforms T, depths, means2d, radii`.

`T` is a `[3×3]` matrix (stored flat as 9 floats) — it maps any pixel ray `(px, py)`
into the Gaussian's local `(u, v)` frame:

```
h_u = px * T[2] − T[0]      ← homogeneous plane for u=const through pixel
h_v = py * T[2] − T[1]      ← homogeneous plane for v=const through pixel
ζ   = cross(h_u, h_v)       ← the unique point satisfying both planes
(u, v) = (ζ.x/ζ.z, ζ.y/ζ.z)  ← local UV on the splat
```

`T` rows are `u_cam`, `v_cam`, `p_cam` after applying the pinhole projection matrix.
`radii` [N,2] is the projected ellipse half-axes in pixels, used for AABB culling.

Reference: `gsplat/Projection2DGS.cuh`, `gsplat/Projection2DGSPacked.cu`

#### Stage 1b — Backward projection (`projection_2dgs_bwd.cu`)

The projection backward pass now reconstructs gradients from projected-space outputs
back to raw Gaussian parameters:

```
grad_ray_transforms + grad_means2d + grad_depths + grad_normals
    → grad_u/v/p_cam
    → grad_u/v/n_world + grad_mean
    → grad_rotation(raw quat), grad_scaling(raw log-scale), grad_means
```

Implementation notes:
- The `means2d → ray_transforms` derivative uses the same closed-form VJP as
  `gsplat/Projection2DGS.cuh`
- Quaternion gradients include the normalization Jacobian, so the optimizer still
  updates the raw unconstrained quaternion stored in `SplatData`
- Only `scale.x` and `scale.y` receive gradients for 2DGS; `scale.z` stays unused
- `train.cu` currently calls this with `grad_ray_transforms` and `grad_means2d`;
  depth/normal gradients are left at zero in the first end-to-end trainer

#### Stage 2 — Tile intersection (`IntersectTile.cu`)

The image is divided into `16×16` pixel tiles. For each Gaussian, its projected AABB
`[mean2d ± radius]` is clipped to the tile grid, producing one entry per
(Gaussian, tile) pair.

**Two-pass design** (avoids dynamic allocation):
- Pass 1 (`cum_tiles_per_gauss=nullptr`): count how many tiles each Gaussian touches
  → allocate `isect_ids[n_isects]` and `flatten_ids[n_isects]`
- Pass 2: fill in the actual (camera_id, tile_id, depth) tuples

Each intersection is packed into a 64-bit key:
```
isect_id = (camera_id << (32 + tile_n_bits)) | (tile_id << 32) | depth_bits
```
The depth is stored as its raw float32 bit pattern so an integer radix sort
simultaneously sorts by camera, tile, and depth.

**Radix sort** (`cub::DeviceRadixSort::SortPairs`) — sorts `isect_ids` and
`flatten_ids` together so Gaussians within each tile are ordered front-to-back.

**Tile offsets** (`intersect_offset_kernel`): after sorting, a second kernel scans
`isect_ids` looking for tile boundary transitions → `tile_offsets[tile_h, tile_w]`
gives `[range_start, range_end)` for each tile's Gaussians in the sorted array.

Reference: `gsplat/IntersectTile.cu`

##### Memory notes: tile list explosion

`launch_tile_intersect()` allocates buffers proportional to `n_isects`, the total
number of `(tile, Gaussian)` overlaps, then CUB radix sort needs additional
temporary storage. On the MipNeRF-360 `garden` scene at full resolution
(`5187×3361`, `138,766` points), the old `SplatData::init_from_points()` scale
initializer produced very large projected radii:

```
visible=86527/138766  avg_radius≈760 px  n_isects=766479362
radix temp≈8.63 GiB, free≈4.51 GiB on NVIDIA L4 23 GiB → OOM
```

Root cause: the initial nearest-neighbor scale estimate sorted points by
`x+y+z` and checked only a tiny local window, which badly overestimated distances
in COLMAP point clouds. The fix is the standalone CUDA helper in
`kernels/simple_knn.cuh`, wired through `SplatData::init_from_points()`. It
Morton-sorts points, computes mean squared distance to 3 nearest neighbors, then
caps isolated points at the p99 distance. The same first iteration now reports:

```
visible=76568/138766  avg_radius≈65 px  n_isects=7988522
```

If OOM returns, first inspect the `projection:` line printed on iteration 1. Large
average radii or a huge `n_isects` usually mean scale initialization, radius
capping, or camera resolution should be reduced before changing the sorter.

##### KNN references for scale initialization

Several reference implementations initialize Gaussian scale from local point-cloud
spacing:

- `gaussian-splatting/submodules/simple-knn/` and
  `2d-gaussian-splatting/submodules/simple-knn/` provide `distCUDA2(points)`.
  It Morton-sorts points on the GPU, searches nearby boxes, and returns mean
  squared distance to the 3 nearest neighbors. GraphDECO then uses
  `log(sqrt(distCUDA2(points)))` as the raw scale.
- `gsplat` does not vendor `simple-knn`; its example trainers use
  `sklearn.neighbors.NearestNeighbors` in `gsplat/examples/utils.py::knn()` and
  initialize scale from the average distance to the 3 nearest neighbors.
- `LichtFeld-Studio/src/core/splat_data.cpp` is the best pure C++ reference.
  It uses `nanoflann` on CPU in `compute_mean_neighbor_distances()` for standard
  initialization. Its `compute_mrnf_knn_log_scales()` path also clamps scale by a
  robust scene extent before taking `log()`, which is useful for avoiding huge
  isolated-point splats.

In this repo, `kernels/simple_knn.cuh` now provides the
`distCUDA2`-style path without Torch. It returns raw CUDA/host arrays and has a
deterministic smoke test in `kernels/simple_knn_test.cu`
(`make run-knn`).

#### Stage 3 — Forward rasterizer (`RasterizeToPixels2DGSFwd.cu`)

Grid: `C × tile_height × tile_width` blocks, each block = one tile = `tile_size²`
threads. Each thread owns one pixel.

Shared memory per block (`block_size` = `tile_size²` entries each):
```
id_batch        [block_size]   int32  — Gaussian index
xy_opacity_batch [block_size]  vec3   — (mean2d.x, mean2d.y, opacity)
u_Ms_batch      [block_size]   vec3   — row 0 of T
v_Ms_batch      [block_size]   vec3   — row 1 of T
w_Ms_batch      [block_size]   vec3   — row 2 of T
```

**Batch loading pattern**: threads cooperate — each thread loads *one Gaussian* from
the sorted list into shared memory, then all threads process the whole batch for
*their own pixel*. This is the standard "tile-based Gaussian splatting" trick that
amortizes global memory reads across the tile.

**Per-pixel inner loop** (front-to-back over sorted Gaussians in this tile):

```cpp
vec3 h_u = px * w_M - u_M;
vec3 h_v = py * w_M - v_M;
vec3 zeta = cross(h_u, h_v);
vec2 s = {zeta.x / zeta.z, zeta.y / zeta.z};  // (u, v)

float gauss_weight_3d = s.x*s.x + s.y*s.y;
float gauss_weight_2d = FILTER_INV_SQUARE * dot(d, d);   // 2D screen falloff
float sigma = 0.5f * min(gauss_weight_3d, gauss_weight_2d);

float alpha = min(0.999f, opacity * exp(-sigma));
if (alpha < ALPHA_THRESHOLD) continue;

float next_T = T * (1.f - alpha);
if (next_T <= 1e-4f) { done = true; break; }  // pixel fully covered

pix_out += color * alpha * T;   // volumetric compositing weight = alpha * T
T = next_T;
```

The `min(gauss_weight_3d, gauss_weight_2d)` merges the 3D ray-intersection kernel
with a 2D screen-space falloff; the 2D term prevents aliasing for very small Gaussians.

`T` is the running transmittance (initially 1.0, product of `(1−alpha)` for all
Gaussians so far). `last_ids` records the last Gaussian to contribute to each pixel
— needed by the backward pass to replay in reverse order.

Reference: `gsplat/RasterizeToPixels2DGSFwd.cu`

#### Stage 4 — Backward rasterizer (`RasterizeToPixels2DGSBwd.cu`)

Same grid/block structure. The backward pass runs **back-to-front** — it starts from
`last_ids[pix]` and walks backward through the sorted Gaussian list.

Why back-to-front: to recompute transmittance at each step during backward without
storing it for every Gaussian. Starting from the final `T_final = 1 − render_alpha`,
and going backward: `T[i] = T[i+1] / (1 − alpha[i])`.

Gradients computed per pixel per Gaussian:
```
dL/d(alpha_i) = (color_i − C_remaining) * T_i * dL/d(render_color)
              + (−1/(1−alpha_i)) * ...  (transmittance term)
dL/d(u,v)     via chain rule through sigma = 0.5*(u²+v²) → alpha
dL/d(T_rows)  via chain rule through the (h_u, h_v, cross, divide) ops
```

Gradients for `ray_transforms` are accumulated with `atomicAdd` since multiple
pixels within a tile share the same Gaussian.

Reference: `gsplat/RasterizeToPixels2DGSBwd.cu`

The implementation lives in [`kernels/rasterize_bwd.cu`](kernels/rasterize_bwd.cu).
One correctness fix is already applied on the forward side: pixels with no contributor
now store `last_ids = -1`, so the backward replay does not accidentally process the
first entry in a tile for empty pixels.

#### Stage 4b — SH backward (`train.cu`)

`sh_backward_kernel` reuses the same SH basis as the forward evaluation and maps
`grad_colors [N,3]` to:

```
grad_sh0 [N,3]
grad_shN [N,45]
```

Current scope:
- gradients are propagated to SH coefficients only
- gradients through the view direction `normalize(mean - cam_pos)` are intentionally
  dropped for now
- clamp saturation is respected, so fully clamped channels contribute zero gradient

#### Stage 5 — Optimizer (`adam.cu`)

`kernels/adam.cu` implements Adam over raw `float*` buffers, following the same
per-element update pattern as `gsplat/gsplat/cuda/csrc/AdamCUDA.cu`.

The low-level launch accepts any `[N, D]` parameter group:

```cpp
launch_adam_step(param, grad, exp_avg, exp_avg_sq,
                 N, D, lr, cfg, step, valid_mask);
```

`SplatAdam` owns moment buffers for each `SplatData` field:

| Parameter group | Shape | Default LR |
|-----------------|-------|------------|
| means | `[N,3]` | `1.6e-4` |
| rotation | `[N,4]` | `1.0e-3` |
| scaling | `[N,3]` | `5.0e-3` |
| opacity | `[N]` | `5.0e-2` |
| sh0 | `[N,3]` | `2.5e-3` |
| shN | `[N,K]` | `1.25e-4` |

The optimizer updates the raw, pre-activation parameters in `SplatData`; kernels
continue to apply `sigmoid`, `exp`, and quaternion normalization at read time.
`train.cu` now allocates `SplatAdam` alongside the forward buffers, builds
`SplatGradients` after the backward kernels, and applies `optimizer.step(...)`
every iteration.

## Known Issues

### Renders appear vertically flipped vs ground truth

When rendering a trained PLY checkpoint with `--render`, the output is vertically
flipped relative to the actual photograph. Confirmed on MipNeRF-360 `garden`:
MSE(render, GT) = 0.094, MSE(render, flip(GT)) = 0.076. The error is consistent,
not a stochastic artifact.

All mathematical components have been verified against the gsplat reference and match:
- `c2w_to_w2c` conversion (viewmat construction)
- `projection_2dgs` T-matrix layout and row convention
- Quaternion storage order (w,x,y,z)
- Sigmoid/exp activations on opacity and scale

Root cause not yet found. The rasterizer tile grid convention (`blockIdx.x` = tile row,
`blockIdx.y` = tile column) is a candidate but hasn't been confirmed. If images look
right but upside-down, flip `render_colors` row-order before writing the PNG as a
workaround: `for row in [0..H/2]: swap(row, H-1-row)`.

## Build

```bash
# All release binaries
make all

# Run the training entry point (loads garden scene, renders, computes loss)
make run-train            # → ./build/train --data ../data/360_v2/garden
./build/train --data ../data/360_v2/garden --images images_4 --iters 1

# Individual kernel tests
make run-knn              # standalone CUDA 3-NN scale initializer test
make run-projection       # projection_2dgs kernel, single-Gaussian sanity check
make run-gradcheck        # finite-difference validation for rasterize/projection/SH backward
make run-quat             # quat_to_rotmat kernel + tests (1M Gaussians, timed)
make run-camera           # camera_projection kernel + tests
make run-colmap-test      # COLMAP reader test (synthetic data, no dataset needed)
make run-splat            # SplatData allocation, activations, move semantics
make run-loss             # photometric loss (L1 + D-SSIM) sanity checks
make run-intersect        # tile intersection test
make run-rasterize        # projection → intersection → forward rasterizer test
make run-adam             # Adam update sanity check with valid-mask skip

# Debug builds (compiled with -G for cuda-gdb / Nsight)
make build/camera_projection_debug
make build/quat_to_rotmat_debug
make build/projection_2dgs_debug
make debug                # build + launch cuda-gdb for camera_projection
make debug-quat           # build + launch cuda-gdb for quat_to_rotmat
```

### Example training run

For a practical end-to-end run on the bundled MipNeRF-360 `garden` scene, use the
downsampled `images_4` folder first. This now works directly: the trainer inspects
the real image sizes, rescales the COLMAP intrinsics to match, and auto-caches the
training set when it is small enough.

```bash
./build/train \
  --data data/360_v2/garden \
  --images images_4 \
  --iters 1000 \
  --log-every 1 \
  --save-ply checkpoints/garden_latest.ply \
  --save-ply-every 50 \
  --densify-start 250 \
  --densify-every 50 \
  --densify-stop 15000 \
  --opacity-reset-every 3000 \
  --dist-lambda 1e-2 \
  --dist-start-iter 3000 \
  --normal-lambda 5e-2 \
  --normal-start-iter 7000 \
  --densify-prune-alpha 0.05 \
  --densify-grow-scale3d 0.01 \
  --densify-prune-scale3d 0.1
```

What you get:
- a terminal progress bar with loss, SH level, `n_isects`, throughput, and ETA
- a standard Gaussian-splatting checkpoint PLY at
  `checkpoints/garden_latest.ply`, overwritten every 50 iterations
- prune/clone/split densification runs automatically between iterations 250 and 15000,
  every 50 iterations by default
- automatic intrinsics matching for `images`, `images_2`, `images_4`, and `images_8`
- automatic byte-cache for smaller training sets such as `garden/images_4`
- 2DGS geometry regularization with distortion loss starting at iteration `3000`
  and depth-derived normal consistency starting at iteration `7000`

Performance note:
- reference 2DGS / gsplat setups usually train MipNeRF-360 on downsampled image
  folders (`images_4` for outdoor scenes, often `images_2` for indoor scenes)
- this CUDA trainer now rescales intrinsics to match those folders automatically
- full-resolution `--images images` still works; it just disables the startup image
  cache on `garden` because the raw dataset is too large
- the remaining optimization plan is tracked in
  [notes/performance_optimization_notes.md](notes/performance_optimization_notes.md)

Reference mapping used in this trainer:
- learning rates match 2D Gaussian Splatting: `xyz=1.6e-4`, `rotation=1e-3`,
  `scaling=5e-3`, `opacity=5e-2`, `sh0=2.5e-3`
- loss mix matches both 2DGS and gsplat: `lambda_dssim = 0.2`
- this repo now uses a more aggressive default than the references: start `250`,
  stop `15000`, every `50`, to grow splat count sooner on current runs
- opacity prune/reset matches the common 2DGS / gsplat 2D trainer setting: `prune_alpha=0.05`,
  opacity reset every `3000`
- clone/split/prune scale thresholds now use scene-scale-normalized values from gsplat:
  `grow_scale3d=0.01`, `prune_scale3d=0.1`

One caveat:
- the raw image-plane gradient magnitude in this CUDA trainer is currently much smaller than
  the literal `2e-4` / `6e-4` thresholds used in 2DGS / gsplat. Because of that, the default
  `--densify-grad-thresh` here is `0`, which means "use the trainer's adaptive mean-grad gate".
  If you want to experiment with an explicit threshold anyway, start around `1e-8` on `garden`,
  not `2e-4`.

Useful shorter smoke test:

```bash
./build/train \
  --data data/360_v2/garden \
  --images images_4 \
  --iters 5 \
  --log-every 1 \
  --preview-every 1 \
  --preview-out previews/garden_smoke/iter \
  --save-ply checkpoints/garden_smoke_latest.ply
```

If you want to force densification early for debugging, the trainer also supports:

```bash
./build/train \
  --data data/360_v2/garden \
  --images images_4 \
  --iters 3 \
  --log-every 1 \
  --densify-start 1 \
  --densify-every 1 \
  --densify-stop 3
```

That path is mainly for exercising prune/clone/split and dynamic reallocation quickly;
the default schedule is the safer one for actual training.

The VS Code `train (CUDA debug)` launch config in `.vscode/launch.json` uses the same
reference-flavored argument set.

If you want to force that configuration explicitly from the terminal, these are the knobs:

```bash
./build/train \
  --data data/360_v2/garden \
  --images images_4 \
  --iters 1000 \
  --log-every 1 \
  --save-ply checkpoints/garden_latest.ply \
  --save-ply-every 50 \
  --densify-start 250 \
  --densify-every 50 \
  --densify-stop 15000 \
  --opacity-reset-every 3000 \
  --dist-lambda 1e-2 \
  --dist-start-iter 3000 \
  --normal-lambda 5e-2 \
  --normal-start-iter 7000 \
  --densify-prune-alpha 0.05 \
  --densify-grow-scale3d 0.01 \
  --densify-prune-scale3d 0.1
```

### Live viewer

The trainer can now export a live-updating checkpoint PLY, and the repo includes
two browser viewers for it:
- `viser_splat_viewer.py`: rendered splat view driven by the CUDA renderer
- `viser_point_cloud_viewer.py`: lightweight point-cloud inspection view

The rendered viewer is the recommended one:

```bash
pip install -r python/viewer_requirements.txt

python3 python/viser_splat_viewer.py \
  --ply checkpoints/garden_latest.ply \
  --port 8080 \
  --resolution 1024
```

What this viewer does:
- keeps the checkpoint loaded in a persistent CUDA subprocess
- sends browser camera poses to that subprocess
- renders actual 2DGS splats and shows the result as the viewer background
- reloads the checkpoint automatically when training overwrites the `.ply`

Useful notes:
- for smoother live updates during training, lower `--save-ply-every` from `50`
  to something like `25`
- `--show-point-cloud` overlays the Gaussian centers for debugging
- the viewer currently flips the rendered image vertically by default to match
  the known rasterizer/viewer Y-orientation issue tracked in `TODO.md`

The older point-cloud-only viewer is still available:

```bash
python3 python/viser_point_cloud_viewer.py \
  --ply checkpoints/garden_latest.ply \
  --port 8080 \
  --poll-seconds 2.0
```

What the point-cloud viewer does:
- loads the trainer's latest `.ply` checkpoint
- shows Gaussian centers and base colors as a point cloud in the browser
- reloads automatically whenever training overwrites the checkpoint file

Current limitation:
- the rendered viewer currently uses a subprocess bridge to the CUDA binary,
  not a direct Python binding yet
- the next viewer step is replacing that bridge with a tighter CUDA render API
  and then adding depth / normals / pause-resume controls

You can also use the viewer that ships with `gsplat`, but it needs a `.pt`
checkpoint rather than a `.ply`. The adapter script
`python/gsplat_viewer_from_ply.py` converts our checkpoint and launches
`gsplat/examples/simple_viewer.py` unchanged:

```bash
python3 python/gsplat_viewer_from_ply.py \
  --ply checkpoints/garden_latest.ply \
  --gsplat-root /home/ubuntu/repos/gsplat \
  --port 8081
```

That path is now also wrapped as:

```bash
make run-viewer-gsplat
```

On AWS, use SSH local port forwarding from your laptop:

```bash
ssh -L 8080:localhost:8080 ubuntu@<your-instance>
```

Then open `http://localhost:8080` in your local browser.

## Data Loading

**`kernels/colmap_reader.hpp`** — header-only, zero deps, used by `train.cu`:
- Reads `cameras.bin`, `images.bin`, `points3D.bin` from `sparse/0/`
- Returns `ColmapScene { cameras, points3D, point_colors }`
- Each `CameraInfo` has `Intrinsics K`, `Mat4 camtoworld`, `std::string image_path`

`train.cu` now inspects the chosen image folder at startup, rescales the COLMAP
intrinsics to the actual image dimensions, and uploads ground-truth images through a
byte staging buffer on the GPU. Smaller datasets are auto-cached in host memory at
startup to avoid repeated `stb_image` decode work during training.

Practical usage:
- use `--images images_4` for faster MipNeRF-360 outdoor training, matching the
  original 2DGS evaluation setup
- use `--images images` if you explicitly want full-resolution training
- `images_2`, `images_4`, and `images_8` no longer require manually regenerated
  COLMAP intrinsics

**`python/colmap_loader.py`** — pure Python, for PyTorch integration (phase 2):
- Same data, returned as numpy float32 arrays
- `python/bindings.cpp` wraps CUDA kernel launches to accept `np.ndarray` directly

Supported dataset layouts:
```
# COLMAP (MipNeRF360, Tanks & Temples)   # Blender / NerfStudio
dataset/                                  dataset/
  sparse/0/                                 transforms.json
    cameras.bin                             images/
    images.bin
    points3D.bin
  images/                                  # must match cameras.bin dimensions
```

## Loss Function

`kernels/loss.cu` implements the standard photometric loss used in 3DGS/2DGS training.
`train.cu` now calls it after rasterization and logs `loss`, `l1`, and `dssim`:

```
iter     1 / 1  |  N=138766  sh=0/3  |  n_isects=7988522  loss=0.258281  l1=0.176346  dssim=0.586018
```

```
L = (1 − λ) · L1(render, gt) + λ · (1 − SSIM(render, gt))
```

λ = 0.2 by default (same as original 3DGS and gsplat).

### L1

Per-pixel mean absolute error. Simple, fast, gradient is ±1/N (sign of residual).

### SSIM

Structural Similarity Index — measures luminance, contrast, and structure similarity
in local 11×11 windows with a Gaussian-weighted kernel (σ = 1.5).

```
SSIM(x, y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
```

Unlike L1, SSIM is perceptually motivated: it penalizes structural differences rather
than treating all pixel errors equally. The 1−SSIM term means the loss goes to 0
when render and ground truth are identical.

### Gradient flow

The backward pass differentiates through each SSIM window back to the input image.
Each pixel `p` participates in all windows that contain it (up to 11×11 = 121).

Using the notation `A = μx² + μy² + C1`, `B = σx² + σy² + C2`, `C = 2μxμy + C1`,
`D = 2σxy + C2`:

```
dL/d(img1[p]) = Gauss★(dm/dμ1·dL) + 2·img1[p]·Gauss★(dm/dσ1²·dL) + img2[p]·Gauss★(dm/dσ12·dL)
```

where `★` denotes convolution with the 11-tap Gaussian kernel (same kernel used in
the forward pass), and the three partial derivatives of the SSIM map m w.r.t. local
statistics are:

```
dm/dμ1    = (2μ2/AB) · (1 − m)  −  (2μ1·C)/(A·B)   [combined with σ terms, see code]
dm/dσ1²   = −D / B²
dm/dσ12   =  C / AB
```

The LichtFeld-Studio trick folds correction terms into `dm_dmu1` so the backward only
needs raw `img1[p]` and `img2[p]` (not saved intermediate means/covariances). This
lets the full forward+backward run in 3 Gaussian filter passes.

### LPIPS

**No CUDA-native LPIPS kernel exists** anywhere in the reference repos. gsplat uses
Python-only LPIPS via `torchmetrics.LearnedPerceptualImagePatchSimilarity` (AlexNet
or VGG backbone through standard PyTorch autograd). This is fine as an eval metric
or auxiliary training loss once Python bindings are added in Phase 2.

### Wasserstein Distortion (future — Phase 2)

From the *Wasserstein Distortion for 3DGS* paper. Instead of pixel-space differences,
WD compares rendered vs. ground-truth images in **VGG feature space**, matching local
statistics (mean μ and std ν) under a Gaussian pooling kernel (σ = 4):

```
d_WD = sqrt((μ − μ̂)² + (ν − ν̂)²)
```

aggregated over VGG feature maps and spatial locations. In practice the recommended
variant is **WD-R** = WD + small weight of standard L1+SSIM, which prevents web-like
artifacts in low-splat-count regions.

Results from the paper: 2.3× human preference over standard loss, best LPIPS/DISTS/FID.
Warm-up: 3k–5k iterations of standard loss before introducing WD.

**Implementability:** The local-statistics formula is simple CUDA once features are
extracted, but VGG inference is the dependency — not trivially done in pure C++/CUDA.
Best added in Phase 2 when the LibTorch/Python boundary is established:
VGG already ships with `torchvision` and is trivially accessible from Python.

## Kernel Pipeline Status

### Phase 1 — 2DGS core

| Step | File | Status |
|------|------|--------|
| Quat → rotation matrix | `kernels/quat_to_rotmat.cu` | done |
| Camera projection (3D → 2D) | `kernels/camera_projection.cu` | done |
| Means/quats/scales → ray_transforms T, depths | `kernels/projection_2dgs.cu` | done |
| Tile–Gaussian intersection (AABB → tile ranges) | `kernels/intersect_tile.cu` | done |
| Forward rasterizer (depth sort + alpha composite) | `kernels/rasterize_fwd.cu` | done |
| Backward pass (gradients) | `train.cu`, `kernels/rasterize_bwd.cu`, `kernels/projection_2dgs_bwd.cu` | done |
| Adam optimizer | `kernels/adam.cu` | done |
| Densification (clone/split) | — | — |

### Phase 2 — Multi-primitive + textures

| Step | Notes |
|------|-------|
| `PrimitiveType` enum + template param on rasterizer | `Gaussian` / `Billboard` / `TexturedGaussian` |
| Texture arrays in `SplatData` (`texture_alpha`, `texture_color`) | `[N,H,W]` and `[N,3,H,W]`, optional (nullptr for Gaussian-only) |
| `sample_bilinear` device function | ~15 lines; shared by Billboard and TexturedGaussian paths |
| Billboard alpha path | `sample_bilinear(texture_alpha, u, v)` replaces Gaussian falloff |

### Phase 3 — Path tracing / relighting

| Step | Notes |
|------|-------|
| BVH construction over Gaussian icospheres | OptiX GAS or pure CUDA `TriangleBvh4` (see `IRGS/`) |
| Ray-primitive intersection kernel | `__anyhit__` sorts hits by depth |
| PBR rendering equation | base color + roughness + environment map |

## Reference Implementations

| What | Where |
|------|-------|
| 2DGS original | `../2d-gaussian-splatting/submodules/diff-surfel-rasterization/` |
| gsplat 2DGS kernels | `../gsplat/gsplat/cuda/csrc/` — `Projection2DGS*.cu`, `RasterizeToPixels2DGS*.cu` |
| LichtFeld-Studio (pure C++/CUDA 3DGS, no PyTorch) | `../LichtFeld-Studio/src/training/rasterization/gsplat/` |
| CUDA sandbox | `../cuda_pybind_torch_playground/` |
| Textured Gaussians (phase 2) | `../textured_gaussians/` |
| Path tracing / relighting (phase 3) | `../3dgrut/`, `../IRGS/` |

## Phases

- [x] **Phase 1 — 2DGS core**: forward/backward kernels plus Adam-wired training loop
- [ ] **Phase 2 — Textured Gaussians**: per-Gaussian UV texture maps
- [ ] **Phase 3 — Path tracing / relighting**: physically-based rendering extensions
