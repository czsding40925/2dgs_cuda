# TODO

## Resume here (set down 2026-05-20 evening)

Last session ended after the 3-way quality comparison
(SH / Nexels-RGB / Nexels-view-dep) finished and the final-report
scaffold was updated with the headline numbers. Pick up at:

- **docs/final_report.md** — markdown scaffold for the 6-page final
  report (due June 8). §5.3 quality table populated, §6 has a wall-
  time root-cause analysis. Promote to `docs/final_report.tex`
  whenever ready.
- **Optional outstanding experiments** (none blocking the scaffold):
  1. *Hash-bwd A/B at production T=2^19* — re-run the warp-dedup vs
     naive benchmark on the larger table. Same A/B harness as the
     one in TODO.md §"Hash-grid backward — warp-dedup variant".
     ~5 min wall.
  2. *Nsight Compute SpeedOfLight* on `rasterize_bwd` and
     `mlp_backward_kernel`. Backs the §3 + §6 bound-diagnosis with
     measured % peak FP32 / DRAM-BW / occupancy. ~10 min wall;
     `make run-bench-ncu-bwd` already wired for the rasterizer side.
  3. *30k-iter Nexels-VD run* — both Nexels variants were still
     climbing at iter 5k. Speculative but would test whether the
     remaining 0.9 dB gap is mostly "needed more iters" or
     architectural. ~3 hours wall.
- **MLP backward atomic dedup** — identified in this session as the
  biggest perf opportunity (Nexels-VD wall is 6.9× SH; W2 grad
  accumulator is the bottleneck). Same warp-dedup pattern as
  `hash_grid_backward_warpdedup_kernel`. Not started.

---

Current state: end-to-end training works on real scenes (garden, bicycle).
Forward → loss → backward → Adam → densification all wired and producing
improving renders. Distortion loss, expected-depth normal consistency, and their
gradient checks are now in place. Quality is decent but still behind reference;
main remaining work is longer-run tuning and a few parity features.

---

## Course milestones

### M3 (May 11) — performance analysis of the vanilla-2DGS CUDA kernels
**Scope decision (2026-05-11):** the nexels appearance pipeline (hash grid +
MLP, error-tensor densification) originally planned for M3 in
`docs/M2_design.md` §4 is **dropped from M3 scope**. M3 will report on the
existing vanilla 2DGS implementation plus per-kernel profiling, per the
plan in `notes/perf_plan.md`. Nexels appearance is deferred to
time-permitting work after M4 (see "Phase 3" below). Reasons: (i) M3's
rubric is "kernels + perf analysis", not feature completeness; the current
implementation already has 6 substantial kernels worth analysing; (ii)
the appearance plumbing through the rasterizer is the highest-risk part
of the original design and would put the M4 (MPI) deadline at risk.

Concrete M3 to-dos: see `notes/perf_plan.md` §3 (instrumentation) and §5
(experiments). At minimum, deliver (a) per-kernel-bucket time table,
(b) ncu-driven "bound" table for the top-3 kernels, (c) one figure (most
likely the resolution sweep).

### M4 (May 27) — distributed (MPI) data-parallel training
Per `docs/M2_design.md` §5. **Plan finalised 2026-05-20** — sequencing
decision: MPI first by May 27, Nexels appearance (Phase B below) lands
between M4 and the June 8 final report. Same scope-discipline rationale
that drove the M3 amendment.

**Hardware reality on this box.** 1× L4. Open MPI 4.1.7 is installed
but built **without** CUDA buffer support: `ompi_info` reports the
`cuda` MPI extension as present, yet `opal_built_with_cuda_support` is
`false`, so handing a device pointer to `MPI_Allreduce` segfaults in
the vader BTL. The MPI Makefile targets therefore pass
`--mpi-host-stage` so the trainer round-trips through pinned host
memory; on a node with a CUDA-aware MPI build the flag can be dropped.
Multi-rank runs oversubscribe the single GPU via `cudaSetDevice(0)`:
useful for code-path + correctness validation, but wall-clock numbers
will be much slower than `np=1` (~5× at np=4 in the smoke sweep) and
the M4 report says so explicitly.

**Status as of 2026-05-20.** A1 + A2 + A4 landed; A3 has the smoke
test in place, finite-diff split-batch check still pending.

#### A1. Data-parallel decomposition
- 1 rank ↔ 1 GPU via `cudaSetDevice(rank % n_gpus)`.
- Full replication of `SplatData` (+ later appearance field) on every rank.
- Each iter: rank 0 picks the per-rank disjoint camera subset + a shared
  RNG seed, broadcasts via `MPI_Bcast`.
- After backward, **one fused `MPI_Allreduce(SUM)`** over a flat gradient
  buffer (`grad_means | grad_rotation | grad_scaling | grad_opacity |
  grad_gamma | grad_sh0 | grad_shN`) before Adam. One collective per
  iter — keep it simple, profile, decide later whether to split.
- Densification: rank 0 decides from all-reduced visibility counts +
  grad-accum, broadcasts the prune/clone/split index list + new-primitive
  payload so replicas re-allocate identically. Reset-opacity ditto.

#### A2. Code wiring
- New `kernels/mpi_comm.cuh` — RAII init/finalize, fused-allreduce helper,
  optional host-staged fallback behind a build flag.
- `train.cu` — wrap post-`sh_bwd` block with allreduce; rank-0-decides
  shim for camera selection + RNG + densification + checkpoint I/O.
- `train_config.hpp` — `--mpi`, `--mpi-cuda-aware`, `--seed`,
  `--cameras-per-rank K` (effective batch = K × world_size).
- `Makefile` — `train_mpi` target linking `mpicxx -ccbin nvcc`; helper
  targets `run-train-mpi-{1,2,4}`.

#### A3. Correctness tests
- ✓ `scripts/test_mpi_parity.sh` — `solo` vs `mpi_np1` vs `mpi_np2` over
  30 iters with `--seed 42`; asserts `solo` vs `mpi_np1` per-iter loss
  agreement within `TOL_ABS=1e-4` (note: bit-identity is unreachable —
  even two repeat runs of `./build/train` drift by ~1e-5 at iter 24 due
  to non-deterministic `atomicAdd` in `rasterize_bwd`), and that
  `mpi_np2` stays finite and non-divergent. Both checks currently pass.
- ✓ `run_split_batch_linearity_check` in `kernels/gradient_checks.cu` —
  builds two synthetic "cameras" sharing one Gaussian (different
  means2d shift + different target render per camera), confirms
  `g_opacity_cam0 + g_opacity_cam1 ≈ d/d_opacity (loss_cam0 +
  loss_cam1)` via centered FD. Passes with abs_err ~1.7e-6, rel_err
  ~3.6e-4. This is the linearity property the MPI allreduce-SUM
  exploits — isolated from `rasterize_bwd` atomic noise that
  contributes the ~1e-5 drift in the end-to-end parity test.

#### A4. Benchmark scripts (delivers rubric §6 figures)
- ✓ `scripts/bench_strong_scaling.sh` — sweep `np ∈ {1, 2, 4}` keeping
  total camera-iters constant (iters_per_rank = total_iters / np). Per-
  bucket rows → `logs/perf/strong_scaling.csv`; per-run wall summary
  → `logs/perf/strong_summary.csv`. `make run-bench-strong` wraps it.
- ✓ `scripts/bench_weak_scaling.sh` — sweep `np` at fixed iters per
  rank (total work scales linearly). Outputs analogous to strong.
  `make run-bench-weak` wraps it.
- ✓ `scripts/bench_comm_breakdown.py` — given a breakdown CSV, emits
  `<basename>_bars.png` (stacked-bucket bars across np, with
  `mpi_allreduce*` highlighted red) and `<basename>_efficiency.png`
  (per-iter wall + parallel efficiency, both kernel-only and
  end-to-end-wall variants). Pairs with the `_summary.csv` for the
  wall-time numbers.
- ✓ New profiling bucket `mpi_allreduce` (and `mpi_allreduce_densify`),
  NVTX + cudaEvent, fed through the existing `prof::registry()`
  infrastructure.
- ✓ `run-bench-mpi-nsys` Makefile target — `nsys profile -t
  cuda,nvtx,osrt,mpi --mpi-impl openmpi` scoped to the `profile_window`
  NVTX range. Smoke-run pending.

**Full sweep results (2026-05-20, garden/images_4, oversubscribed L4,
host-staged allreduce, no densification in window):**

*Strong scaling — fixed total camera-iters = 200; per-rank window =
200/np iters:*

| np | wall_sec | it/s | per-iter ms (bucket-sum) | per-iter ms (wall) | `mpi_allreduce` ms | efficiency (wall) |
|---:|---------:|-----:|-------------------------:|-------------------:|-------------------:|------------------:|
| 1  | 29       | 6.90 |  95                      | 145                |   0                | 100 %             |
| 2  | 45       | 2.22 | 265                      | 450                |  72                |  16 %             |
| 4  | 61       | 0.82 | 535                      | 1220               | 152                |   4 %             |

*Weak scaling — fixed 50 iters per rank; total work scales linearly:*

| np | wall_sec | it/s | per-iter ms (bucket-sum) | per-iter ms (wall) | `mpi_allreduce` ms | efficiency (wall) |
|---:|---------:|-----:|-------------------------:|-------------------:|-------------------:|------------------:|
| 1  | 14       | 3.57 | 100                      | 280                |   0                | 100 %             |
| 2  | 32       | 1.56 | 275                      | 640                |  75                |  22 %             |
| 4  | 61       | 0.82 | 530                      | 1220               | 152                |   6 %             |

End-to-end wall sits ~2× the bucket-sum at np=1, climbing to ~2.3× at
np=4 — the gap is non-profiled work (image upload, host-side iter
scaffolding) + oversubscription scheduling. Parallel efficiency drops
sharply with np because (a) all ranks share one GPU so kernel time
multiplies with np, and (b) the host-staged allreduce adds D2H↔H2D for
every gradient buffer. Both effects are properties of this hardware
configuration; CUDA-aware MPI on a real multi-GPU box would keep
kernel time near constant and drop the allreduce slab to its true
ring-allreduce cost. The M4 report frames this signature honestly
rather than masking it.

PNGs: `logs/perf/{strong,weak}_scaling_{bars,efficiency}.png`.

#### A5. M4 report sections (~2 pages)
T1: strong-scaling table at `np ∈ {1, 2, 4}` (single-L4 oversubscribed,
documented as such). T2: comm-vs-compute stacked bar. T3:
single-rank-vs-2-rank loss-curve overlay. §7 honest-discussion section
that the all-reduce-vs-compute crossover only becomes meaningful on real
multi-GPU hardware.

---

## Render quality

### 1. Longer-run tuning
The new regularizers are wired with the standard 2DGS defaults:
- `dist_lambda=1e-2`, `dist_start_iter=3000`
- `normal_lambda=5e-2`, `normal_start_iter=7000`

What remains is empirical tuning on longer runs:
- check whether `garden` and `bicycle` want the same start iterations
- verify the current weights versus floaters / over-smoothing tradeoff
- inspect rendered depth and normals during training, not just RGB

---

### 2. Cross-check against reference trainers
The local finite-difference checks pass, but it is still worth comparing a short
shared-scene run against `2d-gaussian-splatting` or `gsplat` and checking:
- loss magnitudes over time
- rendered normals / depth maps
- splat-count growth after densification

---

## Densification tuning

### 3. Randomized split placement
Currently splits place children deterministically at `±0.5 * axis_scale` along
the dominant tangent axis. The reference uses a random Gaussian sample offset
(sampled from the Gaussian distribution). May improve coverage in curved regions.

---

## Viewer

### 4. Direct Python render binding
`viser_splat_viewer.py` currently bridges via subprocess + PNG file. A thin
pybind11 or C API binding would remove file I/O latency and enable live
training-time viewing. Depends on Phase 2 (at::Tensor wrappers) being in place.

---

## Phase 2 — PyTorch integration

### 5. `at::Tensor` wrappers
Wrap each CUDA kernel launch to accept `at::Tensor` at the boundary.
Kernels unchanged — only the call site changes.
Reference: `gsplat/` Python binding layer.

### 6. pybind11 bindings (`python/bindings.cpp`)
Expose the `at::Tensor` wrappers to Python so the training loop can be written
in PyTorch with `autograd.Function`. Skeleton is in `python/bindings.cpp`.

---

## Phase 3 — Extensions

### Nexels I-NGP appearance (post-M4, → June 8 final report)

**Plan finalised 2026-05-20.** Placement decision: **per-Gaussian**
(replace `sh_eval_kernel`). The hash-grid + MLP is queried once per
Gaussian at its mean position and writes into the existing
`fwd.colors[N*3]` buffer. Rasterizer is untouched. Loses the paper's
per-ray-hit neural texture, but ~10× smaller surgery and falls cleanly
out of the existing pipeline. Upgrade to per-hit only if PSNR demands
it (would be a stretch / future-work section in the final report).

`tiny-cuda-nn` is already submoduled at `submodules/tiny-cuda-nn`; we
read its kernel code but do **not** link against it (matches the
"port the kernels yourself" framing in M2).

Kernels to land:
- ✓ `kernels/hash_grid.cu` — fwd (trilinear interp, prime-hash) + bwd
  (8-corner `atomicAdd` scatter). Runtime-configurable `L`, `T`, `F`.
  Smoke test (`make run-hash-grid`) and finite-diff gradcheck
  (`make run-gradcheck` → `run_hash_grid_gradcheck`) both passing
  (46/46 touched grid entries within abs_err~1e-6, rel_err<1e-3).
  Grad-back-to-positions deferred — geometry already gets gradient
  from the rasterizer.
- ✓ `kernels/mlp.cu` — fused 2-layer MLP (`h = ReLU(W1·x + b1)`,
  `y = W2·h + b2`). Runtime-configurable `D_in`, `D_hid`, `D_out`.
  Weights and biases loaded once into shared memory per block;
  backward uses block-shared gradient accumulators flushed to global
  via one atomicAdd per parameter per block. Caches post-ReLU `hidden`
  for backward. Smoke (`make run-mlp`) passes `grad_b2 == N` exactly;
  finite-diff gradcheck (`run_mlp_gradcheck`) passes 8 spot-checks
  across all 5 gradient outputs (`grad_W1`, `grad_W2`, `grad_b1`,
  `grad_b2`, `grad_features`) within abs_err ~1e-5, rel_err <1e-3.
  ReLU-dead units correctly produce zero gradient on both sides.
  Note: `extern __shared__` array renamed to `mlp_smem` to avoid
  symbol clash with `rasterize_fwd.cu`'s `smem` when both are compiled
  into the same translation unit (`gradient_checks.cu`).
- ✓ `kernels/nexel_color.cu` — `AppearanceField` (params: grid + W1/b1 +
  W2/b2 + level_scales, owns lifecycle + `init_random` matching the
  Nexels paper defaults: grid=0.01, MLP weights = uniform/√fan_in,
  biases zero), `AppearanceWorkspace` (per-N transient `features`,
  `hidden`, `mlp_out` with capacity-style ensure-on-grow), and
  `AppearanceGrads` (5 parameter-grad buffers plus a `grad_features`
  scratch for the hash backward). `nexel_color_forward` chains
  `hash_grid_fwd` → `mlp_fwd` → +0.5/clamp activation; backward
  chains pass-through activation → `mlp_bwd` → `hash_grid_bwd`,
  re-using `mlp_out` as scratch for `grad_mlp_out` to save an
  allocation. Smoke (`make run-nexel-color`) and end-to-end
  gradcheck (`run_nexel_color_gradcheck`) both pass — 8 grid
  entries through the full forward/backward chain agree with FD to
  abs_err ~1e-5, rel_err <1.1e-2.
  Caveat: forward output is uniform across Gaussians at constant-0.01
  grid init (expected; paper-matched). Real distribution emerges once
  the grid actually learns.
- ✓ `AppearanceField` (params + lifecycle) and `AppearanceAdam`
  (5-tensor Adam state in `kernels/appearance_adam.cuh`, reuses
  `launch_adam_step`).
- ✓ MPI allreduce for `grad_grid + grad_W1/b1/W2/b2` wired into the
  existing post-backward block in `train.cu`, gated on
  `cfg.use_nexel`. Parity confirmed via `NEXEL=1 make run-mpi-parity`:
  solo vs mpi_np1 byte-identical, np=2 stable. Profile breakdown at
  `np=2, T=2^17, host_stage`: TOTAL 325 ms/iter — `rasterize_bwd`
  176 ms (54%), `mpi_allreduce` 93 ms (29%, up from ~70 ms pre-nexel
  reflecting the extra ~16 MB hash-grid payload), `nexel_color_bwd`
  36 ms (11%), `nexel_color_fwd` 4 ms (1.3%), `nexel_adam_step` 0.5 ms.

### View-dependent (SH-basis) Nexels path — landed 2026-05-20

`--nexel-view-dep` switches the MLP output from `D_out=3` (raw RGB) to
`D_out=48` (3 DC + 45 directional SH coefficients), and chains a new
`nexel_sh_eval_packed_fwd/bwd_kernel` pair into the orchestrator so the
final RGB is evaluated per Gaussian against the camera-to-mean
direction. Same view-dependence math as the legacy SH path, just with
coefficients sourced from the shared MLP instead of per-Gaussian
parameters.

Implementation highlights:
- `nexel_sh_eval_packed_*_kernel` in `kernels/nexel_color.cu`. Packed
  layout: `coeffs[i*48 + 0..2]` = sh0[r,g,b], `coeffs[i*48 + 3..47]`
  = shN organised channel-major as `[r(15) || g(15) || b(15)]`.
  Forward applies the same `+0.5 + clamp[0,1]` activation as the
  legacy `sh_eval_kernel`; backward zeroes the gradient on saturated
  pixels (matching `sh_backward_kernel`, unlike the simple-RGB path
  which ignores clamp).
- `AppearanceField.view_dep` flag, set automatically when
  `D_out == 48`. `init_random` zeros W2 rows 3..47 (shN) so the field
  starts as pure DC and the optimiser grows higher-order terms on
  demand — matches the 3DGS convention.
- Orchestrator now takes `(d_means, d_normalised_means, cam_pos)`:
  the un-normalised means + cam_pos feed the SH eval, the normalised
  means feed the hash grid query.
- 11 gradient checks now pass (`make run-gradcheck`):
  `run_nexel_color_view_dep_gradcheck` exercises the packed SH path
  end-to-end through hash → MLP → SH eval, FD-checks 8 grid entries
  with abs_err ~1e-5.
- 500-iter smoke (`./build/train --use-nexel --nexel-view-dep`)
  converges normally: loss 0.31 → 0.20, PSNR climbs through 20 dB by
  iter 400; preview at iter 500 visibly sharper than the non-view-dep
  preview at the same iter. iter/s = 5.4 (vs 8.9 for non-view-dep)
  — the SH eval kernels are the overhead.

### Hash-grid backward — warp-dedup variant — landed 2026-05-20

`hash_grid_backward_warpdedup_kernel` in `kernels/hash_grid.cu` is now
the default backward. Uses `cooperative_groups::labeled_partition` to
group same-slot lanes within a warp, reduces their per-corner
contributions with `cg::reduce`, then a single leader lane does the
`atomicAdd`. At coarse levels where many positions hit the same voxel,
this collapses 32 contended atomics into one.

The naïve atomic kernel is preserved for A/B comparison via
`hash_grid_bwd_use_warp_dedup()` (process-wide toggle, exposed at the
CLI as `--hash-bwd-naive`).

A/B numbers (garden/images_4, T=2^17, fresh-init N≈140k, 50 measured
iters, non-view-dep, single-process):

| variant     | `nexel_color_bwd` ms/iter |
|-------------|------------------------:|
| warp-dedup  | 20.06                   |
| naive       | 21.83                   |

8% speedup on the composite bucket (which also includes the MLP +
activation backward). Isolating the hash backward alone would show a
larger relative win; at production `T=2^19` with more coarse-level
collisions the gap should widen further.

### Quality comparison: SH vs Nexels (RGB) vs Nexels (view-dep) — landed 2026-05-20

Same garden/images_4, identical seed + densification schedule
(`densify-start=250 every=100 stop=3750`), 5000 iters each. Driven by
`scripts/compare_sh_vs_nexel.sh` (now 3-way). Output:
`logs/quality/{per_iter,summary}.csv`, plus side-by-side previews
`previews/quality_{sh,nexel,nexel_vd}_final.png`.

| | wall (s) | final loss | final PSNR | best PSNR | best @ iter |
|---|---:|---:|---:|---:|---:|
| SH               |  482 | 0.094 | **25.08** | 25.60 | 4650 |
| Nexels (RGB)     | 1042 | 0.133 | **21.78** | 22.77 | 4650 |
| Nexels (view-dep)| 3314 | 0.103 | **24.18** | **24.98** | 4650 |

**Δ headline:** Nexels-RGB lags SH by ~3.3 dB; the view-dep SH-basis
upgrade closes that gap to ~0.9 dB (and the *best* PSNR gap to
0.62 dB) at the cost of a 6.9× wall-time penalty vs SH. The view-dep
render is visually nearly indistinguishable from SH — same sharp
detail in the table grain, brick wall, and vase. The RGB render
(non-view-dep) is the blurry one with chromatic speckle in
undertrained patches.

Wall-time root-cause analysis (view-dep): MLP backward dominates the
6.9× hit because `D_out=48` makes W2 16× larger than the
non-view-dep case (3 outputs), and the block-shared gradient
accumulator for W2 in `mlp_backward_kernel` does
`D_hid * D_out = 64 * 48 = 3072` atomicAdds per thread into shared
memory. The hash-grid backward and MLP forward both stay cheap. MLP
backward is the next obvious optimisation target — see Phase 3
"future work" in the final-report scaffold.

Remaining factors not yet exercised in this comparison:
- Per-pixel-hit Nexels (paper-faithful) — moves the hash query into
  the rasterizer's composite loop. Would close the remaining 0.9 dB.
  Significant rewrite of `rasterize_fwd/bwd`.
- 30k-iter runs — SH plateaus around iter 5k; both Nexels variants
  are still climbing at iter 5k. Speculative but probably worth
  another dB on view-dep.
- Error-tensor densification — drop grad-magnitude for
  `|render − gt|`-weighted sampling per the paper.

### Nexels MPI scaling sweep (NEXEL=1) — landed 2026-05-20

Same garden/images_4 setup, hash table T=2^17, host-stage allreduce,
oversubscribed L4. Bench scripts re-runnable as
`NEXEL=1 bash scripts/bench_strong_scaling.sh` and
`NEXEL=1 bash scripts/bench_weak_scaling.sh`.

*Strong scaling — fixed total camera-iters = 200:*

| np | wall_sec | it/s | vs SH-only wall | parallel efficiency (wall) |
|---:|---------:|-----:|----------------:|---------------------------:|
| 1  | 33       | 6.06 | +14%            | 100%                       |
| 2  | 53       | 1.89 | +18%            | ~16%                       |
| 4  | 77       | 0.65 | +26%            | ~3%                        |

*Weak scaling — fixed 50 iters/rank:*

| np | wall_sec | it/s | vs SH-only wall | parallel efficiency (wall) |
|---:|---------:|-----:|----------------:|---------------------------:|
| 1  | 16       | 3.13 | +14%            | 100%                       |
| 2  | 38       | 1.32 | +19%            | ~21%                       |
| 4  | 72       | 0.69 | +18%            | ~6%                        |

Two effects from enabling Nexels:
- `nexel_color_fwd + bwd + adam` adds ~40 ms/iter compute per rank
  (light-green + red slabs in the bar chart).
- `mpi_allreduce` grows ~30-40% larger because the hash-grid gradient
  (16 MB at T=2^17, 64 MB at the production T=2^19) tacks onto the
  existing per-Gaussian gradient payload.

The scaling efficiency penalty compared to SH-only is larger at
higher np because (a) MPI payload grew and (b) the compute side did
not (Nexels is parallel-friendly per-rank), so the comm/compute ratio
worsens. On CUDA-aware MPI hardware the picture should flip — payload
grows but stays on-device. PNGs in `logs/perf/{strong,weak}_scaling_nexel_*.png`.
  Header-include caveat: `nexel_color.cu` guards its dependency
  includes with `#ifndef INCLUDED_AS_HEADER` so its `#undef` does not
  clobber the outer flag when included from `gradient_checks.cu` (or
  the trainer translation unit). When including from a TU that already
  defines `INCLUDED_AS_HEADER` (like `train.cu`), the caller must
  include `hash_grid.cu` and `mlp.cu` explicitly first.

### Nexels train.cu integration — landed 2026-05-20

- ✓ CLI: `--use-nexel`, `--hash-levels`, `--hash-log-table-size`,
  `--hash-features`, `--hash-min-res`/`-max-res`, `--mlp-hidden`,
  `--lr-grid`/`-mlp`, `--nexel-init-seed`. Default off → trainer
  unchanged.
- ✓ `train()` at startup, gated on `cfg.use_nexel`, builds
  `AppearanceField` (random init), `AppearanceWorkspace`,
  `AppearanceGrads`, `AppearanceAdam`, computes a bbox from current
  Gaussian means (5% margin), uploads as `(origin, inv_extent)` for
  position normalisation.
- ✓ Iter loop swaps `sh_eval` for `launch_nexel_normalize_positions →
  nexel_color_forward` and `sh_bwd` for `nexel_color_backward` when
  the flag is on. SH params stay allocated, receive no gradient.
- ✓ `nexel_adam_step` runs after the splat `adam_step`. Workspace
  resizes automatically on densification via `ensure()`.
- ✓ End-to-end smokes:
  - 500-iter run (no densify): loss 0.31 → 0.18, ~9 it/s vs SH's
    12 it/s; iter-1 preview is the expected uniform grey, iter-500
    preview shows the garden scene with emerging colour (greens,
    browns, the central bowl visible).
  - 600-iter run with densify-every=100 in [100, 500]: N grows
    138k → 200k across 5 rounds, loss continues to drop, workspace
    resizes cleanly, no NaN.
- ✓ Default (SH) path bit-identical to pre-nexel trainer
  (smoke shows iter 1 loss = 0.23823 matching M4 baseline).

Error-tensor densification (`kernels/densify.cuh` rewrite — replace
grad-magnitude with `|render − gt|`-weighted pixel sampling +
unprojection) follows the appearance pipeline; only the *selection
criterion* changes, prune/clone/split scaffolding stays.

### 7. Billboard / Textured Gaussian support
See CLAUDE.md §Primitive type design. Add `PrimitiveType` enum and `if constexpr`
branches in the rasterizer for `Gaussian` / `Billboard` / `TexturedGaussian`.

### 8. OptiX BVH ray tracing (IRGS-style)
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
- [x] Median-depth output + `--depth-ratio` mix into normal-consistency surface depth (median is stop-gradient; default `depth_ratio=0` is bit-identical to previous behavior, gradcheck 28/28)
- [x] Viewer coordinate-convention fix: convert viser OpenGL c2w → COLMAP/OpenCV before sending to the renderer subprocess (was causing the funky-orbit / mismatched-preview viewer behavior)
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
