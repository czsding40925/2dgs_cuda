# Performance Optimization Notes

Notes for speeding up the current CUDA 2DGS trainer in a way that stays close to
the reference `2d-gaussian-splatting` and `gsplat` implementations.

## Reference baselines

- `2d-gaussian-splatting/scripts/m360_eval.py` trains outdoor MipNeRF-360 scenes
  on `images_4` and indoor scenes on `images_2`.
- `gsplat/examples/simple_trainer_2dgs.py` defaults to `data_factor=4`, which is
  the same idea: reduce training resolution first before chasing deeper kernel work.
- `gsplat` does use a fused 2DGS projection primitive (`fully_fused_projection_2dgs`),
  but its training stack is still staged. The main win is not "one giant kernel";
  it is less host orchestration, less temporary allocation, and less unnecessary work.

## Status after the first speed pass

The trainer now supports `--images images_2`, `images_4`, and `images_8` directly.

What changed:
- startup now inspects the actual image dimensions in the selected folder
- camera intrinsics are rescaled to match those dimensions
- training uploads ground-truth images through a byte staging buffer and converts to
  float on the GPU
- smaller datasets are auto-cached in host memory to avoid repeated image decode work

On `garden` this means:
- `images_4` is now the recommended baseline
- `images_4` auto-caches cleanly because the raw dataset is about `576.7 MiB`
- full-resolution `images` still works, but the cache is skipped because the raw
  dataset is about `9.01 GiB`

## Immediate operational advice

- Do not benchmark with multiple `./build/train` processes on the same GPU. That
  makes the trainer look slower than it is.
- Keep preview cadence modest during longer runs, for example `--preview-every 500`
  instead of every iteration.

## Recommended optimization order

### 1. Enable downsampled training inputs

Status: done.

- `images_2`, `images_4`, and `images_8` now work by rescaling intrinsics at startup.
- `images_4` should now be the default training choice for MipNeRF-360 outdoor scenes.

### 2. Cache target images instead of loading from disk every iteration

Status: partially done.

- training no longer decodes cached datasets every iteration
- the cache currently stores raw RGB bytes in host memory and uploads them through a
  GPU staging buffer
- the remaining improvement here would be pinned host memory and possibly a more
  explicit cache policy

### 3. Reuse tile-intersection scratch buffers

Next implementation target.

Current hot path:
- `kernels/intersect_tile.cu` allocates and frees `tiles_per_gauss`, prefix-sum
  buffers, sort buffers, and tile offsets every iteration.

Planned change:
- introduce a persistent workspace object for tile intersection
- grow capacity only when `N`, `n_isects`, or tile count exceed the current reservation

Expected benefit:
- lower allocator overhead
- fewer host-device synchronization points around temporary storage setup

### 4. Reuse loss scratch buffers

Status: done.

- `kernels/loss.cu` now reuses a persistent `LossWorkspace` instead of allocating SSIM
  and L1 scratch every iteration

### 5. Remove unnecessary `cudaDeviceSynchronize()` calls

Status: partially done.

- the training hot path no longer forces syncs after forward rasterization, backward
  rasterization, tile-offset generation, or SH backward
- remaining sync cleanup should be guided by timing rather than done blindly

### 6. Add CUDA event timing before larger refactors

Immediately after the tile-intersection workspace work lands, add this profiling pass
so the next optimization choice is based on measured stage times rather than guesswork.

Before deeper kernel work, instrument:
- SH evaluation
- projection
- tile intersection
- rasterization
- loss
- backward
- Adam step

This should tell us whether the next bottleneck is still host orchestration or if
the rasterizer becomes dominant once the easy wins land.

## Deeper follow-up work

These are worth doing after the steps above.

### 7. Compact visible-Gaussian or packed path

Reference direction:
- `gsplat` uses `fully_fused_projection_2dgs`
- it can also return packed outputs for only the visible entries

Potential local adaptation:
- compact valid projected Gaussians after projection
- run later stages on the compact active set instead of dense `N`
- optionally propagate densification stats from the packed outputs

Note:
- `gsplat` documents `packed` mainly as a memory tradeoff, not a guaranteed speedup,
  so this should be measured rather than assumed

### 8. Selective or fused Adam updates

Reference direction:
- `gsplat/gsplat/optimizers/selective_adam.py`

Potential local adaptation:
- fuse Adam math more aggressively
- optionally skip updates for splats that are inactive or invisible in the current step

This is lower priority than the image/downsample/workspace fixes above.
