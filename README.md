# 2DGS CUDA

From-scratch CUDA implementation of 2D Gaussian Splatting. Pure C++/CUDA — no
PyTorch, no LibTorch. Includes an MPI data-parallel training mode and an
optional neurally-textured ("Nexels") appearance pipeline. Built as the
final project for Stanford CME 213.

## Build

CUDA 12.x + a recent host C++ compiler. MPI build wants OpenMPI or MPICH
reachable as `mpicxx`. No CMake.

```bash
make all            # single-GPU trainer + tests
make train_mpi      # MPI multi-rank build
make run-gradcheck  # finite-difference gradient validation
```

## Train

End-to-end on the bundled MipNeRF-360 `garden` scene:

```bash
./build/train \
  --data data/360_v2/garden --images images_4 \
  --iters 1000 --log-every 1 \
  --save-ply checkpoints/garden_latest.ply --save-ply-every 50 \
  --densify-start 250 --densify-every 50 --densify-stop 15000 \
  --opacity-reset-every 3000 \
  --dist-lambda 1e-2 --dist-start-iter 3000 \
  --normal-lambda 5e-2 --normal-start-iter 7000 \
  --densify-prune-alpha 0.05 \
  --densify-grow-scale3d 0.01 --densify-prune-scale3d 0.1
```

Output: terminal progress (loss, PSNR, splat count, throughput), a periodic
PLY checkpoint, optional PNG previews via `--preview-every`. Defaults match
2DGS / gsplat: LRs `means 1.6e-4`, `rotation 1e-3`, `scaling 5e-3`,
`opacity 5e-2`, `sh0 2.5e-3`, `shN 1.25e-4`; `lambda_dssim = 0.2`;
scene-scale-normalised densification thresholds. Image folders `images`,
`images_2/4/8` are auto-handled — COLMAP intrinsics are rescaled to match the
chosen folder at startup.

### Distributed (MPI)

```bash
mpirun -np 4 ./build/train_mpi --data data/360_v2/garden \
       --images images_4 --iters 1000 --mpi-host-stage
```

One rank per GPU (or oversubscribed). Fully-replicated parameters; one
`MPI_Allreduce(SUM)` per parameter tensor after backward, plus one for the
densification accumulators per densify window. `--mpi-host-stage` routes the
allreduce through pinned host memory — drop it on a CUDA-aware MPI build.
Helper targets: `make run-train-mpi-{1,2,4}`, `run-bench-strong`,
`run-bench-weak`, `run-mpi-parity`.

### Nexels (neurally-textured 2DGS)

```bash
./build/train --use-nexel --nexel-view-dep \
  --hash-log-table-size 17 --mlp-hidden 64 [...]
```

Replaces per-Gaussian SH coefficients with a shared Instant-NGP hash grid +
2-layer MLP, queried at each Gaussian's mean position. `--nexel-view-dep`
makes the MLP emit packed SH coefficients (3 DC + 45 directional) for view
dependence. Smaller representation than full SH, slower per iter.

## Architecture

Two layers:

```
SplatData (kernels/splat_data.cuh)   owns GPU memory; raw pre-activation params
        │  float* pointers
        ▼
CUDA kernels (kernels/*.cu)          pure math on raw pointers; activations inline
```

Parameters are stored unconstrained and activated at read time: `opacity`
(logit → sigmoid), `scaling` (log → exp), `rotation` (unnorm quat → normalize).

### Forward / backward pipeline

```
projection_2dgs       means/quats/scales → ray_transforms T, means2d, depths, radii
intersect_tile        AABBs → sorted (tile, Gaussian) pairs via cub radix sort
rasterize_fwd         tile-based front-to-back alpha compositing → rgb + aux
loss                  L1 + D-SSIM + 2DGS distortion + depth-derived normals
rasterize_bwd         back-to-front replay → grad T, opacity, color, means2d
projection_2dgs_bwd   grad T → grad means/rotation/scaling
sh_backward           grad colors → grad sh0/shN
adam_step             raw-pointer Adam over each SplatData group
densify               prune / clone / split by accumulated screen-space gradient
```

### File layout

```
train.cu                       entry point — load scene, training loop, MPI hooks
train_config.hpp               CLI parsing + Config struct
kernels/
  splat_data.cuh               RAII GPU arrays + activations
  projection_2dgs{,_bwd}.cu    projection forward / backward
  intersect_tile.cu            tile/Gaussian pair generation + radix sort
  rasterize_{fwd,bwd}.cu       tile-based rasterizer
  loss.cu                      L1 + D-SSIM (Gaussian-windowed)
  adam.cu                      raw-pointer Adam + per-group LRs
  densify.cuh                  prune / clone / split with Adam-state remap
  gradient_checks.cu           finite-difference validation harness
  hash_grid.cu, mlp.cu         Nexels appearance backbone
  nexel_color.cu               AppearanceField + per-Gaussian color path
  appearance_adam.cuh          Adam state for Nexels params
  mpi_comm.cuh                 RAII MPI context + collective wrappers
  profiling.cuh                NVTX + cudaEvent profiling buckets
  colmap_reader.hpp            header-only COLMAP loader (no deps)
  simple_knn.cuh               Morton-sorted 3-NN scale initialiser
python/
  viser_splat_viewer.py        live browser viewer (subprocess to renderer)
  gsplat_viewer_from_ply.py    gsplat's viewer wrapping our PLY checkpoint
  bindings.cpp                 (phase 2) pybind11 wrappers
scripts/
  bench_{strong,weak}_scaling.sh   MPI scaling sweeps
  bench_comm_breakdown.py          stacked-bucket + efficiency plots
  test_mpi_parity.sh               solo vs np=1 vs np=2 parity
  compare_sh_vs_nexel.sh           three-way quality comparison
```

## Viewer

```bash
pip install -r python/viewer_requirements.txt
python3 python/viser_splat_viewer.py \
  --ply checkpoints/garden_latest.ply --port 8080 --resolution 1024
```

Keeps the checkpoint loaded in a persistent CUDA subprocess, sends browser
camera poses, renders actual 2DGS splats. Reloads automatically when the
trainer overwrites the PLY. On a remote box, forward the port:
`ssh -L 8080:localhost:8080 ubuntu@<host>`.

## Tests

```bash
make run-gradcheck      # finite-diff backward (projection, rasterize, SH, geom, nexels)
make run-knn            # CUDA simple-KNN smoke
make run-rasterize      # projection → intersection → forward rasterizer
make run-mpi-parity     # solo vs np=1 vs np=2 30-iter parity
```

All backward kernels have a corresponding centered-FD gradient check in
`kernels/gradient_checks.cu`; CI-style invocation is `make run-gradcheck`.

## References

| What | Path |
|------|------|
| 2DGS original (paper code) | `../2d-gaussian-splatting/` |
| gsplat 2DGS kernels | `../gsplat/gsplat/cuda/csrc/` |
| LichtFeld-Studio (pure C++/CUDA 3DGS) | `../LichtFeld-Studio/src/training/rasterization/gsplat/` |
| Nexels (neurally-textured surfels) | `../GS/nexels/` |
| Instant-NGP (hash grid) | `../neural_fields/instant-ngp/` |
