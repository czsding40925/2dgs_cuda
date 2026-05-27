// hash_grid.cu
//
// Multi-resolution spatial hash grid à la Müller et al. "Instant Neural
// Graphics Primitives" (SIGGRAPH 2022). This is the appearance backbone
// for the Phase B (Nexels) work: each Gaussian queries a single shared
// hash grid at its mean position, the resulting L * F feature vector is
// fed to a small MLP, and the MLP output replaces the per-Gaussian SH
// colour. See TODO.md §"Phase 3 — Nexels I-NGP appearance" and
// docs/M2_design.md §4.
//
// What this file provides
//   * launch_hash_grid_fwd: per-Gaussian forward query → feature vector
//   * launch_hash_grid_bwd: scatter gradient back to the L*T*F grid table
//                            via atomicAdd to the 8 voxel corners per level
//
// What it intentionally does NOT do (yet)
//   * Gradient flow back into query positions. We will need this for the
//     full "couple geometry to appearance" loss path; for the MVP it is
//     omitted so the kernel is smaller and easier to validate. The
//     Gaussian means still receive a geometric gradient from the
//     rasterizer; the hash grid simply re-learns colours at whatever
//     positions the means drift to.
//   * Half-precision storage or TensorCore GEMM. fp32 throughout, MLP
//     stays in fp32; revisit only if profiling motivates it.
//   * Warp-level reduce-before-atomic. First pass uses plain atomicAdd;
//     M2_design.md §3 flags this as the headline backward bottleneck, to
//     be revisited once we have a measured number.
//
// Data layout
//   grid[L][T][F] flattened row-major:
//     grid_index = ((level * table_size) + slot) * F + feat
//   features[N][L][F] flattened row-major:
//     feat_index = ((n * L) + level) * F + feat
//
// L (levels), T (table_size, must be power of two), F (features) are
// runtime parameters. Per-level voxel resolution is supplied as a host
// float array via launch_*.

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifndef HASH_GRID_CUDA_CHECK
#define HASH_GRID_CUDA_CHECK(call)                                              \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "[hash_grid] CUDA error at %s:%d — %s\n",                 \
              __FILE__, __LINE__, cudaGetErrorString(_e));                      \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)
#endif

// Müller-style spatial-hash primes. p0 = 1 keeps the x-axis untouched,
// which gives slightly better behaviour in common usage patterns where
// queries vary mostly along one axis.
__device__ __forceinline__ uint32_t hash_grid_spatial_hash(
    int ix, int iy, int iz, uint32_t table_mask
) {
    constexpr uint32_t P1 = 2654435761u;   // golden-ratio prime
    constexpr uint32_t P2 = 805459861u;
    uint32_t h = (uint32_t)ix
               ^ ((uint32_t)iy * P1)
               ^ ((uint32_t)iz * P2);
    return h & table_mask;                  // table_mask = T - 1, T = 2^k
}

// ─── Forward kernel ─────────────────────────────────────────────────────────
//
// One thread per (Gaussian, level). With L=16 levels and N≈400k Gaussians
// that's 6.4M threads → fine. We could instead do one thread per
// (n, level, feature) for slightly more parallelism, but the F=2 inner
// loop is cheap and gives better register usage per thread.

__global__ void hash_grid_forward_kernel(
    const float* __restrict__ positions,    // [N, 3]
    const float* __restrict__ grid,         // [L, T, F]
    const float* __restrict__ level_scales, // [L] = voxel resolution per level
    float*       __restrict__ features,     // [N, L, F]   (written, not accumulated)
    int N, int L, int F, uint32_t table_size
) {
    int n     = blockIdx.x * blockDim.x + threadIdx.x;
    int level = blockIdx.y;
    if (n >= N || level >= L) return;

    const uint32_t table_mask = table_size - 1u;
    const float scale = level_scales[level];

    float px = positions[n * 3 + 0] * scale;
    float py = positions[n * 3 + 1] * scale;
    float pz = positions[n * 3 + 2] * scale;

    int ix0 = (int)floorf(px);
    int iy0 = (int)floorf(py);
    int iz0 = (int)floorf(pz);
    float fx = px - (float)ix0;
    float fy = py - (float)iy0;
    float fz = pz - (float)iz0;

    const float* grid_level = grid + (size_t)level * (size_t)table_size * (size_t)F;
    float* feat_out         = features + ((size_t)n * (size_t)L + (size_t)level) * (size_t)F;

    // Accumulate the 8-corner trilinear blend into a register vector of
    // length F (F is typically 2). We zero by hand for predictable
    // behaviour even when F > 4.
    constexpr int F_MAX = 8;
    float acc[F_MAX];
    for (int f = 0; f < F && f < F_MAX; f++) acc[f] = 0.f;

    #pragma unroll
    for (int o = 0; o < 8; o++) {
        int ox = (o >> 0) & 1;
        int oy = (o >> 1) & 1;
        int oz = (o >> 2) & 1;
        float wx = ox ? fx : (1.f - fx);
        float wy = oy ? fy : (1.f - fy);
        float wz = oz ? fz : (1.f - fz);
        float w  = wx * wy * wz;

        uint32_t slot = hash_grid_spatial_hash(ix0 + ox, iy0 + oy, iz0 + oz, table_mask);
        const float* g = grid_level + (size_t)slot * (size_t)F;
        for (int f = 0; f < F && f < F_MAX; f++) {
            acc[f] += w * g[f];
        }
    }

    for (int f = 0; f < F && f < F_MAX; f++) feat_out[f] = acc[f];
}

// ─── Backward kernel ────────────────────────────────────────────────────────
//
// Same thread layout as forward. Each thread fetches grad_features for its
// (n, level), recomputes the trilinear weights, and scatters
// `w * grad_features[f]` into the relevant grid corner via atomicAdd.
//
// Atomic-contention behaviour: collisions are highest at the LOW levels
// (coarse resolution → many positions hit the same corner). At the
// finest level each Gaussian's corner is essentially unique. Total
// atomic-traffic per iter is 8 * L * N * F adds; warp-reduce-before-atomic
// is the standard mitigation if profiling motivates it.

// "Naïve" backward kernel: every thread does its own atomicAdd for each
// of the 8 corners × F features. Simple and correct, but on coarse
// levels many threads in a warp hit the same slot → contention.
__global__ void hash_grid_backward_kernel(
    const float* __restrict__ positions,       // [N, 3]
    const float* __restrict__ level_scales,    // [L]
    const float* __restrict__ grad_features,   // [N, L, F]
    float*       __restrict__ grad_grid,       // [L, T, F]    (accumulated)
    int N, int L, int F, uint32_t table_size
) {
    int n     = blockIdx.x * blockDim.x + threadIdx.x;
    int level = blockIdx.y;
    if (n >= N || level >= L) return;

    const uint32_t table_mask = table_size - 1u;
    const float scale = level_scales[level];

    float px = positions[n * 3 + 0] * scale;
    float py = positions[n * 3 + 1] * scale;
    float pz = positions[n * 3 + 2] * scale;

    int ix0 = (int)floorf(px);
    int iy0 = (int)floorf(py);
    int iz0 = (int)floorf(pz);
    float fx = px - (float)ix0;
    float fy = py - (float)iy0;
    float fz = pz - (float)iz0;

    const float* gf = grad_features + ((size_t)n * (size_t)L + (size_t)level) * (size_t)F;
    float* grad_grid_level = grad_grid + (size_t)level * (size_t)table_size * (size_t)F;

    // Pre-load grad_features into a small register vector. F is small
    // (typically 2) so this stays in registers.
    constexpr int F_MAX = 8;
    float gv[F_MAX];
    for (int f = 0; f < F && f < F_MAX; f++) gv[f] = gf[f];

    #pragma unroll
    for (int o = 0; o < 8; o++) {
        int ox = (o >> 0) & 1;
        int oy = (o >> 1) & 1;
        int oz = (o >> 2) & 1;
        float wx = ox ? fx : (1.f - fx);
        float wy = oy ? fy : (1.f - fy);
        float wz = oz ? fz : (1.f - fz);
        float w  = wx * wy * wz;

        uint32_t slot = hash_grid_spatial_hash(ix0 + ox, iy0 + oy, iz0 + oz, table_mask);
        float* g = grad_grid_level + (size_t)slot * (size_t)F;
        for (int f = 0; f < F && f < F_MAX; f++) {
            atomicAdd(&g[f], w * gv[f]);
        }
    }
}

// Warp-deduped backward kernel: threads in a warp that hit the same
// hash slot for the same corner pre-sum their contributions via
// `cooperative_groups::labeled_partition` + `reduce`, then a single
// "leader" lane per same-slot group does the atomicAdd. At coarse
// levels (low voxel resolution) the dedup ratio is large; at fine
// levels (near-unique slots) it degenerates to one atomic per thread
// plus a few cheap shuffles, which is still cheaper than the SOL of
// the naïve kernel's atomic-contention.
//
// Uses cooperative_groups (CUDA 11+), so the kernel cannot be used on
// pre-CC-7.0 hardware. L4 is sm_89, fine.
__global__ void hash_grid_backward_warpdedup_kernel(
    const float* __restrict__ positions,       // [N, 3]
    const float* __restrict__ level_scales,    // [L]
    const float* __restrict__ grad_features,   // [N, L, F]
    float*       __restrict__ grad_grid,       // [L, T, F]    (accumulated)
    int N, int L, int F, uint32_t table_size
) {
    namespace cg = cooperative_groups;

    int n     = blockIdx.x * blockDim.x + threadIdx.x;
    int level = blockIdx.y;
    bool active = (n < N && level < L);

    // coalesced_threads() picks up only the active subset of the warp
    // (i.e. those that didn't early-return). We early-return only on
    // out-of-range, which is rare in practice; the typical warp has
    // 32 active threads.
    auto warp = cg::coalesced_threads();

    if (!active) return;

    const uint32_t table_mask = table_size - 1u;
    const float scale = level_scales[level];

    float px = positions[n * 3 + 0] * scale;
    float py = positions[n * 3 + 1] * scale;
    float pz = positions[n * 3 + 2] * scale;

    int ix0 = (int)floorf(px);
    int iy0 = (int)floorf(py);
    int iz0 = (int)floorf(pz);
    float fx = px - (float)ix0;
    float fy = py - (float)iy0;
    float fz = pz - (float)iz0;

    const float* gf = grad_features + ((size_t)n * (size_t)L + (size_t)level) * (size_t)F;
    float* grad_grid_level = grad_grid + (size_t)level * (size_t)table_size * (size_t)F;

    constexpr int F_MAX = 8;
    float gv[F_MAX];
    for (int f = 0; f < F && f < F_MAX; f++) gv[f] = gf[f];

    #pragma unroll
    for (int o = 0; o < 8; o++) {
        int ox = (o >> 0) & 1;
        int oy = (o >> 1) & 1;
        int oz = (o >> 2) & 1;
        float wx = ox ? fx : (1.f - fx);
        float wy = oy ? fy : (1.f - fy);
        float wz = oz ? fz : (1.f - fz);
        float w  = wx * wy * wz;

        uint32_t slot = hash_grid_spatial_hash(ix0 + ox, iy0 + oy, iz0 + oz, table_mask);

        // Partition the active warp by slot. Threads with the same slot
        // form a group that we'll reduce within.
        auto group = cg::labeled_partition(warp, (int)slot);

        for (int f = 0; f < F && f < F_MAX; f++) {
            float contribution = w * gv[f];
            float total = cg::reduce(group, contribution, cg::plus<float>());
            if (group.thread_rank() == 0) {
                atomicAdd(&grad_grid_level[(size_t)slot * (size_t)F + f], total);
            }
        }
    }
}

// ─── Host-side launch helpers ────────────────────────────────────────────────

inline void launch_hash_grid_fwd(
    const float* d_positions,     // [N, 3]
    const float* d_grid,          // [L, T, F]
    const float* d_level_scales,  // [L]
    float*       d_features,      // [N, L, F]
    int N, int L, int F, uint32_t table_size
) {
    if (N <= 0 || L <= 0 || F <= 0) return;
    const int threads = 128;
    dim3 blocks((N + threads - 1) / threads, L);
    hash_grid_forward_kernel<<<blocks, threads>>>(
        d_positions, d_grid, d_level_scales, d_features,
        N, L, F, table_size
    );
    HASH_GRID_CUDA_CHECK(cudaGetLastError());
}

// Process-wide toggle: when true, launch_hash_grid_bwd uses the
// cooperative-groups warp-dedup kernel; otherwise the plain atomicAdd
// version. Default true on CUDA 11+ hardware where the dedup pays for
// itself. Flip to false for A/B benchmarking or pre-Volta fallback.
inline bool& hash_grid_bwd_use_warp_dedup() {
    static bool flag = true;
    return flag;
}

inline void launch_hash_grid_bwd(
    const float* d_positions,
    const float* d_level_scales,
    const float* d_grad_features,
    float*       d_grad_grid,        // caller must zero-init before the call
    int N, int L, int F, uint32_t table_size
) {
    if (N <= 0 || L <= 0 || F <= 0) return;
    const int threads = 128;
    dim3 blocks((N + threads - 1) / threads, L);
    if (hash_grid_bwd_use_warp_dedup()) {
        hash_grid_backward_warpdedup_kernel<<<blocks, threads>>>(
            d_positions, d_level_scales, d_grad_features,
            d_grad_grid,
            N, L, F, table_size
        );
    } else {
        hash_grid_backward_kernel<<<blocks, threads>>>(
            d_positions, d_level_scales, d_grad_features,
            d_grad_grid,
            N, L, F, table_size
        );
    }
    HASH_GRID_CUDA_CHECK(cudaGetLastError());
}

// Geometric progression of voxel resolutions matching Instant-NGP:
//   res[l] = floor(min_res * (max_res / min_res)^(l / (L-1)))
// Used as the per-level scale factor that multiplies normalized positions
// before the integer corner / fractional split.
inline std::vector<float> hash_grid_build_level_scales(
    int L, float min_res, float max_res
) {
    std::vector<float> scales((size_t)L);
    if (L == 1) {
        scales[0] = min_res;
        return scales;
    }
    const double b = std::log((double)max_res / (double)min_res) / (double)(L - 1);
    for (int l = 0; l < L; l++) {
        scales[(size_t)l] = (float)((double)min_res * std::exp(b * l));
    }
    return scales;
}

// ─── Standalone smoke test (built only when not used as a header) ────────────

#ifndef INCLUDED_AS_HEADER
#include <random>
#include <vector>
#include <cstdlib>

int main() {
    // Tiny configuration to keep the test cheap.
    constexpr int N = 4;
    constexpr int L = 2;
    constexpr int F = 2;
    constexpr uint32_t T = 64;  // power of two

    std::vector<float> h_pos = {
        0.10f, 0.20f, 0.30f,
        0.50f, 0.50f, 0.50f,
        0.90f, 0.10f, 0.80f,
        0.25f, 0.75f, 0.40f,
    };
    std::vector<float> level_scales = hash_grid_build_level_scales(L, 4.f, 16.f);

    std::vector<float> h_grid((size_t)L * T * F);
    // Deterministic but non-trivial grid contents so the forward output
    // is interpretable: g[level][slot][feat] = sin(level + 0.13*slot + 0.7*feat)
    for (int l = 0; l < L; l++) {
        for (uint32_t s = 0; s < T; s++) {
            for (int f = 0; f < F; f++) {
                h_grid[((size_t)l * T + s) * F + f] =
                    std::sin((float)l + 0.13f * (float)s + 0.7f * (float)f);
            }
        }
    }

    float *d_pos, *d_grid, *d_scales, *d_feat;
    HASH_GRID_CUDA_CHECK(cudaMalloc(&d_pos,    (size_t)N * 3 * sizeof(float)));
    HASH_GRID_CUDA_CHECK(cudaMalloc(&d_grid,   (size_t)L * T * F * sizeof(float)));
    HASH_GRID_CUDA_CHECK(cudaMalloc(&d_scales, (size_t)L * sizeof(float)));
    HASH_GRID_CUDA_CHECK(cudaMalloc(&d_feat,   (size_t)N * L * F * sizeof(float)));
    HASH_GRID_CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(),
                                    (size_t)N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    HASH_GRID_CUDA_CHECK(cudaMemcpy(d_grid, h_grid.data(),
                                    (size_t)L * T * F * sizeof(float), cudaMemcpyHostToDevice));
    HASH_GRID_CUDA_CHECK(cudaMemcpy(d_scales, level_scales.data(),
                                    (size_t)L * sizeof(float), cudaMemcpyHostToDevice));

    launch_hash_grid_fwd(d_pos, d_grid, d_scales, d_feat, N, L, F, T);
    HASH_GRID_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_feat((size_t)N * L * F);
    HASH_GRID_CUDA_CHECK(cudaMemcpy(h_feat.data(), d_feat,
                                    (size_t)N * L * F * sizeof(float), cudaMemcpyDeviceToHost));

    printf("hash_grid forward smoke test (N=%d, L=%d, F=%d, T=%u):\n", N, L, F, T);
    for (int n = 0; n < N; n++) {
        printf("  n=%d  pos=(%.2f, %.2f, %.2f)", n,
               h_pos[n*3+0], h_pos[n*3+1], h_pos[n*3+2]);
        for (int l = 0; l < L; l++) {
            printf("  L%d=(", l);
            for (int f = 0; f < F; f++) {
                printf("%s%.4f", f ? ", " : "",
                       h_feat[((size_t)n * L + l) * F + f]);
            }
            printf(")");
        }
        printf("\n");
    }

    // Backward smoke: synthetic grad_features = all-ones, expect the
    // accumulated grad_grid sum-per-Gaussian-per-level to equal F (since
    // the 8 corner weights sum to 1 by construction of trilinear interp).
    std::vector<float> h_grad_feat((size_t)N * L * F, 1.f);
    float *d_grad_feat, *d_grad_grid;
    HASH_GRID_CUDA_CHECK(cudaMalloc(&d_grad_feat, h_grad_feat.size() * sizeof(float)));
    HASH_GRID_CUDA_CHECK(cudaMalloc(&d_grad_grid, (size_t)L * T * F * sizeof(float)));
    HASH_GRID_CUDA_CHECK(cudaMemcpy(d_grad_feat, h_grad_feat.data(),
                                    h_grad_feat.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
    HASH_GRID_CUDA_CHECK(cudaMemset(d_grad_grid, 0, (size_t)L * T * F * sizeof(float)));

    launch_hash_grid_bwd(d_pos, d_scales, d_grad_feat, d_grad_grid, N, L, F, T);
    HASH_GRID_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_grad_grid((size_t)L * T * F);
    HASH_GRID_CUDA_CHECK(cudaMemcpy(h_grad_grid.data(), d_grad_grid,
                                    (size_t)L * T * F * sizeof(float),
                                    cudaMemcpyDeviceToHost));
    double tot = 0.0;
    for (float v : h_grad_grid) tot += (double)v;
    // Expected: sum of all grad_grid entries = N * L * F * (sum of 8 weights = 1)
    //         = N * L * F
    double expected = (double)N * L * F;
    printf("backward smoke: sum(grad_grid) = %.6f  (expected %.6f, diff %.3e)\n",
           tot, expected, std::abs(tot - expected));

    cudaFree(d_pos); cudaFree(d_grid); cudaFree(d_scales); cudaFree(d_feat);
    cudaFree(d_grad_feat); cudaFree(d_grad_grid);

    bool ok = std::abs(tot - expected) < 1e-3 * expected;
    printf("%s\n", ok ? "hash_grid smoke OK" : "hash_grid smoke FAILED");
    return ok ? 0 : 1;
}
#endif
