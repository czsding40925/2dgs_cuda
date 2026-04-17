// intersect_tile.cu
//
// Tile intersection for 2DGS rasterization.  Given the projected 2D
// Gaussians (means2d, radii) and their depths, produces the sorted list
// of (tile, Gaussian) pairs needed by the rasterizer.
//
// Pipeline (four steps, three kernel launches):
//
//   1. COUNT  — tiles_per_gauss[i]  = number of tiles Gaussian i's AABB overlaps
//   2. PREFIX — cum_tiles_per_gauss  = exclusive prefix sum  (CUB)
//               n_isects             = cum_tiles_per_gauss[N-1] + tiles_per_gauss[N-1]
//   3. FILL   — isect_ids[k]   = packed 64-bit key: upper 32 = tile_id, lower 32 = depth bits
//               flatten_ids[k] = index into [0, N) identifying the Gaussian
//   4. SORT   — radix sort on isect_ids, carrying flatten_ids along (CUB)
//   5. OFFSETS — tile_offsets[t] = start index in sorted list for tile t  (one kernel)
//
// After this, the rasterizer knows for each tile exactly which Gaussians
// to draw and in what front-to-back order.
//
// Encoding of isect_id (64 bits):
//   bits [63:32]  tile_id  (row * tile_width + col)
//   bits [31: 0]  depth as raw uint32 bit-pattern of float
//
// Because IEEE 754 floats sort the same as uint32 for positive values,
// a single radix sort on the 64-bit key simultaneously sorts by tile then
// by depth (front-to-back) within each tile.
//
// Memory:
//   means2d        [N, 2]            float  — pixel-space projected center
//   radii          [N, 2]            int32  — AABB half-extents in pixels (rx, ry)
//   depths         [N]               float  — camera-space z
//   tiles_per_gauss [N]              int32  — output of count pass
//   cum_tiles_per_gauss [N]          int32  — exclusive prefix sum
//   isect_ids      [n_isects]        int64  — (tile_id << 32 | depth_bits), unsorted then sorted
//   flatten_ids    [n_isects]        int32  — Gaussian index, sorted alongside isect_ids
//   tile_offsets   [tile_h*tile_w]   int32  — start offset in sorted list for each tile

#include "splat_data.cuh"

#include <cstdint>
#include <vector>
#include <cstdio>
#include <cmath>

#include <cub/cub.cuh>          // DeviceScan, DeviceRadixSort
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: count tiles per Gaussian
// ─────────────────────────────────────────────────────────────────────────────
//
// Each thread handles one Gaussian.  Clips its AABB [mean ± radius] to the
// tile grid and writes the number of tiles covered.  Gaussians outside the
// image or with zero/negative radius write 0.

__global__ void count_tiles_per_gauss_kernel(
    const float*   __restrict__ means2d,         // [N, 2]  pixel-space x,y
    const int32_t* __restrict__ radii,            // [N, 2]  int pixel radii rx,ry
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,     // e.g. 16
    const uint32_t tile_width,    // ceil(image_width  / tile_size)
    const uint32_t tile_height,   // ceil(image_height / tile_size)
    int32_t* __restrict__ tiles_per_gauss,  // [N] output
    const int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int32_t rx = radii[i * 2 + 0];
    const int32_t ry = radii[i * 2 + 1];

    if (rx <= 0 || ry <= 0) {
        tiles_per_gauss[i] = 0;
        return;
    }

    const float cx = means2d[i * 2 + 0];
    const float cy = means2d[i * 2 + 1];

    // Cull entirely off-screen Gaussians
    if (cx + rx <= 0.f || cx - rx >= (float)image_width ||
        cy + ry <= 0.f || cy - ry >= (float)image_height) {
        tiles_per_gauss[i] = 0;
        return;
    }

    // Tile-space AABB — inclusive min, exclusive max
    const float ts = (float)tile_size;
    uint32_t tx_min = (uint32_t)max(0,  (int)floorf((cx - rx) / ts));
    uint32_t ty_min = (uint32_t)max(0,  (int)floorf((cy - ry) / ts));
    uint32_t tx_max = (uint32_t)min((int)tile_width,  (int)ceilf((cx + rx) / ts));
    uint32_t ty_max = (uint32_t)min((int)tile_height, (int)ceilf((cy + ry) / ts));

    tiles_per_gauss[i] = (int32_t)((ty_max - ty_min) * (tx_max - tx_min));
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: fill isect_ids and flatten_ids
// ─────────────────────────────────────────────────────────────────────────────
//
// Called after prefix sum.  Each thread writes the (tile_id, depth) entries
// for its Gaussian, starting at cum_tiles_per_gauss[i].

__global__ void fill_isect_ids_kernel(
    const float*   __restrict__ means2d,             // [N, 2]
    const int32_t* __restrict__ radii,               // [N, 2]
    const float*   __restrict__ depths,              // [N]
    const int32_t* __restrict__ cum_tiles_per_gauss, // [N] exclusive prefix sum
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    int64_t* __restrict__ isect_ids,   // [n_isects]  output
    int32_t* __restrict__ flatten_ids, // [n_isects]  output
    const int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int32_t rx = radii[i * 2 + 0];
    const int32_t ry = radii[i * 2 + 1];
    if (rx <= 0 || ry <= 0) return;

    const float cx = means2d[i * 2 + 0];
    const float cy = means2d[i * 2 + 1];

    const float ts = (float)tile_size;
    uint32_t tx_min = (uint32_t)max(0,  (int)floorf((cx - rx) / ts));
    uint32_t ty_min = (uint32_t)max(0,  (int)floorf((cy - ry) / ts));
    uint32_t tx_max = (uint32_t)min((int)tile_width,  (int)ceilf((cx + rx) / ts));
    uint32_t ty_max = (uint32_t)min((int)tile_height, (int)ceilf((cy + ry) / ts));

    // Pack depth as raw bits so integer radix sort preserves float order
    // (valid for positive depths — camera-space z should be > 0)
    uint32_t depth_bits = __float_as_uint(depths[i]);

    // Exclusive prefix sum: cum[i] = number of intersections for all Gaussians
    // with index < i.  So cum[i] is exactly the start offset for Gaussian i.
    int64_t cur = (int64_t)cum_tiles_per_gauss[i];
    for (uint32_t ty = ty_min; ty < ty_max; ++ty) {
        for (uint32_t tx = tx_min; tx < tx_max; ++tx) {
            int64_t tile_id = (int64_t)(ty * tile_width + tx);
            // Upper 32 bits = tile_id, lower 32 bits = depth
            isect_ids[cur]   = (tile_id << 32) | (int64_t)depth_bits;
            flatten_ids[cur] = i;
            ++cur;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: compute tile_offsets from sorted isect_ids
// ─────────────────────────────────────────────────────────────────────────────
//
// After sorting, isect_ids are grouped by tile.  This kernel finds the
// boundary between tiles and writes the start offset for each tile.
//
// tile_offsets[t] = first index in sorted isect_ids that belongs to tile t.
// Tiles with no Gaussians get the same value as the next non-empty tile
// (or n_isects for the last tile).

__global__ void compute_tile_offsets_kernel(
    const int64_t* __restrict__ isect_ids_sorted, // [n_isects]
    const uint32_t n_tiles,   // tile_width * tile_height
    const uint32_t n_isects,
    int32_t* __restrict__ tile_offsets  // [n_tiles]
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_isects) return;

    // Extract tile_id from upper 32 bits
    uint32_t tile_curr = (uint32_t)(isect_ids_sorted[idx] >> 32);

    if (idx == 0) {
        // Fill offsets for all tiles before (and including) the first tile
        for (uint32_t t = 0; t <= tile_curr; ++t)
            tile_offsets[t] = 0;
    }
    if (idx == n_isects - 1) {
        // Fill offsets for all remaining tiles (they are empty)
        for (uint32_t t = tile_curr + 1; t < n_tiles; ++t)
            tile_offsets[t] = (int32_t)n_isects;
    }

    if (idx > 0) {
        uint32_t tile_prev = (uint32_t)(isect_ids_sorted[idx - 1] >> 32);
        if (tile_prev != tile_curr) {
            // Tile boundary: write offset for all tiles between prev and curr
            for (uint32_t t = tile_prev + 1; t <= tile_curr; ++t)
                tile_offsets[t] = (int32_t)idx;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side orchestration
// ─────────────────────────────────────────────────────────────────────────────
//
// Runs all four steps and returns pointers to the sorted arrays and offsets.
// Caller is responsible for freeing the allocations listed at the bottom.

struct TileIntersectBuffers {
    int32_t* tiles_per_gauss;       // [N]
    int32_t* cum_tiles_per_gauss;   // [N]  exclusive prefix sum
    int64_t* isect_ids;             // [n_isects]  sorted
    int32_t* flatten_ids;           // [n_isects]  sorted
    int32_t* tile_offsets;          // [tile_h * tile_w]
    int32_t  n_isects;
};

// Forward declaration so main() can call it
TileIntersectBuffers launch_tile_intersect(
    const float*   d_means2d,    // [N, 2]
    const int32_t* d_radii,      // [N, 2]
    const float*   d_depths,     // [N]
    int N,
    uint32_t image_width,
    uint32_t image_height,
    uint32_t tile_size
) {
    const uint32_t tile_width  = (image_width  + tile_size - 1) / tile_size;
    const uint32_t tile_height = (image_height + tile_size - 1) / tile_size;
    const uint32_t n_tiles = tile_width * tile_height;

    dim3 threads(256);
    dim3 grid((N + 255) / 256);

    // ── Step 1: count ─────────────────────────────────────────────────────────
    int32_t* d_tiles_per_gauss;
    CUDA_CHECK(cudaMalloc(&d_tiles_per_gauss, N * sizeof(int32_t)));

    count_tiles_per_gauss_kernel<<<grid, threads>>>(
        d_means2d, d_radii,
        image_width, image_height,
        tile_size, tile_width, tile_height,
        d_tiles_per_gauss, N
    );
    CUDA_CHECK(cudaGetLastError());

    // ── Step 2: prefix sum (CUB ExclusiveSum) ─────────────────────────────────
    int32_t* d_cum_tiles;
    CUDA_CHECK(cudaMalloc(&d_cum_tiles, N * sizeof(int32_t)));

    // Query temp storage size
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes,
                                  d_tiles_per_gauss, d_cum_tiles, N);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes,
                                  d_tiles_per_gauss, d_cum_tiles, N);
    CUDA_CHECK(cudaFree(d_temp));

    // Total intersections = last element of exclusive sum + last tiles_per_gauss
    int32_t last_cum, last_count;
    CUDA_CHECK(cudaMemcpy(&last_cum,   d_cum_tiles         + N - 1, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_count, d_tiles_per_gauss   + N - 1, sizeof(int32_t), cudaMemcpyDeviceToHost));
    int32_t n_isects = last_cum + last_count;
    if (n_isects == 0) {
        // Nothing to rasterize — return empty buffers
        TileIntersectBuffers out{};
        out.tiles_per_gauss     = d_tiles_per_gauss;
        out.cum_tiles_per_gauss = d_cum_tiles;
        out.n_isects            = 0;
        int32_t* d_offsets;
        CUDA_CHECK(cudaMalloc(&d_offsets, n_tiles * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_offsets, 0, n_tiles * sizeof(int32_t)));
        out.tile_offsets = d_offsets;
        return out;
    }

    // ── Step 3: fill ──────────────────────────────────────────────────────────
    int64_t* d_isect_ids;
    int32_t* d_flatten_ids;
    CUDA_CHECK(cudaMalloc(&d_isect_ids,   n_isects * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_flatten_ids, n_isects * sizeof(int32_t)));

    fill_isect_ids_kernel<<<grid, threads>>>(
        d_means2d, d_radii, d_depths, d_cum_tiles,
        tile_size, tile_width, tile_height,
        d_isect_ids, d_flatten_ids, N
    );
    CUDA_CHECK(cudaGetLastError());

    // ── Step 4: radix sort ────────────────────────────────────────────────────
    //
    // Sort (isect_ids, flatten_ids) pairs by isect_id key.
    // Upper 32 bits = tile_id, lower 32 bits = depth — one sort handles both.
    int64_t* d_isect_ids_sorted;
    int32_t* d_flatten_ids_sorted;
    CUDA_CHECK(cudaMalloc(&d_isect_ids_sorted,   n_isects * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_flatten_ids_sorted, n_isects * sizeof(int32_t)));

    d_temp = nullptr; temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_isect_ids,   d_isect_ids_sorted,
        d_flatten_ids, d_flatten_ids_sorted,
        n_isects);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_isect_ids,   d_isect_ids_sorted,
        d_flatten_ids, d_flatten_ids_sorted,
        n_isects);
    CUDA_CHECK(cudaFree(d_temp));

    cudaFree(d_isect_ids);
    cudaFree(d_flatten_ids);

    // ── Step 5: tile offsets ──────────────────────────────────────────────────
    int32_t* d_tile_offsets;
    CUDA_CHECK(cudaMalloc(&d_tile_offsets, n_tiles * sizeof(int32_t)));
    // Initialize to n_isects so empty tiles point past the end
    cudaMemset(d_tile_offsets, 0, n_tiles * sizeof(int32_t));

    dim3 grid_isects((n_isects + 255) / 256);
    compute_tile_offsets_kernel<<<grid_isects, threads>>>(
        d_isect_ids_sorted, n_tiles, (uint32_t)n_isects, d_tile_offsets
    );
    CUDA_CHECK(cudaGetLastError());

    TileIntersectBuffers out;
    out.tiles_per_gauss     = d_tiles_per_gauss;
    out.cum_tiles_per_gauss = d_cum_tiles;
    out.isect_ids           = d_isect_ids_sorted;
    out.flatten_ids         = d_flatten_ids_sorted;
    out.tile_offsets        = d_tile_offsets;
    out.n_isects            = n_isects;
    return out;
}

void free_tile_intersect_buffers(TileIntersectBuffers& b) {
    cudaFree(b.tiles_per_gauss);
    cudaFree(b.cum_tiles_per_gauss);
    cudaFree(b.isect_ids);
    cudaFree(b.flatten_ids);
    cudaFree(b.tile_offsets);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test / main
// ─────────────────────────────────────────────────────────────────────────────
//
// Synthetic scene: a 64x64 image with tile_size=16 → 4x4=16 tiles.
// Three Gaussians placed at known positions with known radii.

#ifndef INCLUDED_AS_HEADER
int main() {
    printf("=== intersect_tile test ===\n\n");

    const uint32_t W = 64, H = 64, TS = 16;
    const uint32_t TW = W / TS, TH = H / TS;  // 4 × 4 tile grid
    printf("Image: %ux%u  tile_size: %u  tile grid: %ux%u\n\n", W, H, TS, TW, TH);

    // ── Synthetic Gaussians ───────────────────────────────────────────────────
    //
    // Gaussian 0: center=(8, 8),   radius=(8,  8)  → spans tile (0,0) only
    // Gaussian 1: center=(32, 32), radius=(20, 20) → spans tiles around center (3x3 = 9)
    // Gaussian 2: center=(56, 8),  radius=(4,  4)  → spans tile (3,0) only
    // Gaussian 3: center=(-5, -5), radius=(3,  3)  → entirely off-screen
    const int N = 4;

    float h_means2d[N * 2] = {
         8.f,  8.f,   // G0
        32.f, 32.f,   // G1
        56.f,  8.f,   // G2
        -5.f, -5.f,   // G3 (off-screen)
    };
    int32_t h_radii[N * 2] = {
         8,  8,   // G0
        20, 20,   // G1
         4,  4,   // G2
         3,  3,   // G3
    };
    float h_depths[N] = { 1.5f, 3.0f, 2.0f, 0.5f };

    // ── Upload to GPU ─────────────────────────────────────────────────────────
    float*   d_means2d; int32_t* d_radii; float* d_depths;
    CUDA_CHECK(cudaMalloc(&d_means2d, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii,   N * 2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_depths,  N     * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_means2d, h_means2d, N*2*sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radii,   h_radii,   N*2*sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depths,  h_depths,  N*sizeof(float),     cudaMemcpyHostToDevice));

    // ── Run tile intersection ─────────────────────────────────────────────────
    TileIntersectBuffers bufs = launch_tile_intersect(
        d_means2d, d_radii, d_depths, N, W, H, TS
    );

    printf("n_isects = %d\n", bufs.n_isects);
    printf("Expected: G0→1 tile, G1→16 tiles (radius 20 spans full 64px grid), G2→1 tile, G3→0 = 18 total\n\n");

    // ── Verify tiles_per_gauss ────────────────────────────────────────────────
    {
        int32_t h_tpg[N];
        CUDA_CHECK(cudaMemcpy(h_tpg, bufs.tiles_per_gauss, N*sizeof(int32_t), cudaMemcpyDeviceToHost));
        printf("tiles_per_gauss: ");
        for (int i = 0; i < N; i++) printf("G%d=%d  ", i, h_tpg[i]);
        printf("\n");
    }

    // ── Print sorted (tile_id, depth, gaussian_id) triples ───────────────────
    if (bufs.n_isects > 0) {
        int32_t n = bufs.n_isects;
        std::vector<int64_t> h_ids(n);
        std::vector<int32_t> h_flat(n);
        CUDA_CHECK(cudaMemcpy(h_ids.data(),  bufs.isect_ids,   n*sizeof(int64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_flat.data(), bufs.flatten_ids, n*sizeof(int32_t), cudaMemcpyDeviceToHost));
        printf("\nSorted intersections (tile_id, depth, gaussian_id):\n");
        for (int k = 0; k < n; k++) {
            uint32_t tile_id   = (uint32_t)(h_ids[k] >> 32);
            uint32_t depth_raw = (uint32_t)(h_ids[k] & 0xFFFFFFFF);
            float    depth     = *reinterpret_cast<float*>(&depth_raw);
            uint32_t tx = tile_id % TW, ty = tile_id / TW;
            printf("  [%2d]  tile(%u,%u)=%2u  depth=%.2f  gauss=%d\n",
                   k, tx, ty, tile_id, depth, h_flat[k]);
        }
    }

    // ── Print tile_offsets ────────────────────────────────────────────────────
    {
        uint32_t n_tiles = TW * TH;
        std::vector<int32_t> h_off(n_tiles);
        CUDA_CHECK(cudaMemcpy(h_off.data(), bufs.tile_offsets, n_tiles*sizeof(int32_t), cudaMemcpyDeviceToHost));
        printf("\ntile_offsets [tile_y][tile_x]:\n");
        for (uint32_t ty = 0; ty < TH; ty++) {
            printf("  row %u: ", ty);
            for (uint32_t tx = 0; tx < TW; tx++)
                printf("%3d ", h_off[ty*TW + tx]);
            printf("\n");
        }
        printf("  (n_isects=%d = sentinel for empty tiles past the end)\n", bufs.n_isects);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    free_tile_intersect_buffers(bufs);
    cudaFree(d_means2d); cudaFree(d_radii); cudaFree(d_depths);

    printf("\nDone.\n");
    return 0;
}
#endif // INCLUDED_AS_HEADER
