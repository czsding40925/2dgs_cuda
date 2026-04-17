// rasterize_fwd.cu
//
// Forward rasterizer for 2D Gaussian Splatting.
//
// One CUDA block per tile, one thread per pixel.
// Gaussians within each tile are processed front-to-back (pre-sorted by depth).
//
// Grid / block layout:
//   grid  = { tile_height, tile_width }
//   block = { tile_size,   tile_size  }   (e.g. 16×16 = 256 threads per block)
//
// Shared memory per block (reloaded each batch):
//   id_batch        [block_size]  int32   — Gaussian index into [0,N)
//   xy_opacity_batch[block_size]  float3  — (mean2d.x, mean2d.y, opacity)
//   u_M_batch       [block_size]  float3  — row 0 of ray_transforms
//   v_M_batch       [block_size]  float3  — row 1 of ray_transforms
//   w_M_batch       [block_size]  float3  — row 2 of ray_transforms
//
// Per-pixel inner loop:
//   h_u  = px * w_M - u_M              homogeneous plane equations
//   h_v  = py * w_M - v_M
//   zeta = cross(h_u, h_v)             ray-splat intersection (homogeneous)
//   (u,v)= (zeta.x/zeta.z, zeta.y/zeta.z)  splat-local UV
//
//   gauss_3d = u² + v²                 3D Gaussian kernel in UV space
//   gauss_2d = 2 * |pixel - mean2d|²   2D screen falloff (anti-aliasing)
//   sigma    = 0.5 * min(gauss_3d, gauss_2d)
//   alpha    = min(0.999, opacity * exp(-sigma))
//
//   vis  = alpha * T                   volume rendering weight
//   color += vis * c                   accumulate
//   T    *= (1 - alpha)                update transmittance
//
// Outputs:
//   render_colors [H, W, 3]  — accumulated RGB
//   render_alphas [H, W]     — 1 - T_final (total opacity)
//   last_ids      [H, W]     — index in sorted list of last contributing Gaussian
//                               (needed by backward pass to replay in reverse)

#include "splat_data.cuh"

#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace cg = cooperative_groups;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr uint32_t TILE_SIZE       = 16;
static constexpr float    ALPHA_THRESHOLD = 1.f / 255.f;
// 2D screen-space falloff weight — 1/(2*0.3²) = 1/0.18 ≈ 5.56, gsplat uses 2.0
static constexpr float    FILTER_INV_SQ   = 2.0f;

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

struct float3_s { float x, y, z; };

__device__ inline float3_s cross3(float3_s a, float3_s b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward rasterization kernel
// ─────────────────────────────────────────────────────────────────────────────

__global__ void rasterize_fwd_kernel(
    // geometry
    const float*   __restrict__ means2d,       // [N, 2]   pixel-space centers
    const float*   __restrict__ ray_transforms, // [N, 9]   u_M, v_M, w_M rows
    const float*   __restrict__ opacities,     // [N]      raw logit, sigmoid applied here
    // appearance
    const float*   __restrict__ colors,        // [N, 3]   pre-evaluated RGB in [0,1]
    // tile data (from intersect_tile)
    const int32_t* __restrict__ tile_offsets,  // [tile_h * tile_w]
    const int32_t* __restrict__ flatten_ids,   // [n_isects]  sorted Gaussian indices
    const int32_t  n_isects,
    // image dimensions
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    float*   __restrict__ render_colors,  // [H, W, 3]
    float*   __restrict__ render_alphas,  // [H, W]
    int32_t* __restrict__ last_ids        // [H, W]
) {
    auto block = cg::this_thread_block();

    // ── Block / thread → tile / pixel mapping ─────────────────────────────────
    uint32_t tile_y = block.group_index().x;
    uint32_t tile_x = block.group_index().y;
    uint32_t tile_id = tile_y * tile_width + tile_x;

    uint32_t py = tile_y * TILE_SIZE + block.thread_index().x;  // pixel row
    uint32_t px_i = tile_x * TILE_SIZE + block.thread_index().y; // pixel col

    float px = (float)px_i + 0.5f;
    float py_f = (float)py + 0.5f;
    int32_t pix_id = py * image_width + px_i;

    bool inside = (py < image_height && px_i < image_width);
    bool done   = !inside;

    // ── Tile range in sorted list ─────────────────────────────────────────────
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end   = (tile_id == tile_width * tile_height - 1)
                          ? n_isects
                          : tile_offsets[tile_id + 1];

    const uint32_t block_size = block.size();  // TILE_SIZE²
    uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // ── Shared memory ─────────────────────────────────────────────────────────
    extern __shared__ int smem[];
    int32_t*  id_batch         = (int32_t*)smem;
    float3_s* xy_opacity_batch = (float3_s*)&id_batch[block_size];
    float3_s* u_M_batch        = (float3_s*)&xy_opacity_batch[block_size];
    float3_s* v_M_batch        = (float3_s*)&u_M_batch[block_size];
    float3_s* w_M_batch        = (float3_s*)&v_M_batch[block_size];

    // ── Per-pixel state ───────────────────────────────────────────────────────
    float T        = 1.0f;
    float pix_r    = 0.f, pix_g = 0.f, pix_b = 0.f;
    int32_t cur_idx = -1;
    uint32_t tr    = block.thread_rank();

    // ── Main loop — process Gaussians in batches ──────────────────────────────
    for (uint32_t b = 0; b < num_batches; ++b) {

        // Early exit if every thread in the tile is done
        if (__syncthreads_count(done) >= block_size)
            break;

        // Cooperative load: each thread fetches one Gaussian into shared memory
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < (uint32_t)range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            xy_opacity_batch[tr] = {
                means2d[g*2],
                means2d[g*2+1],
                SplatData::sigmoid(opacities[g])
            };
            u_M_batch[tr] = { ray_transforms[g*9],   ray_transforms[g*9+1], ray_transforms[g*9+2] };
            v_M_batch[tr] = { ray_transforms[g*9+3], ray_transforms[g*9+4], ray_transforms[g*9+5] };
            w_M_batch[tr] = { ray_transforms[g*9+6], ray_transforms[g*9+7], ray_transforms[g*9+8] };
        }
        block.sync();

        // Each thread processes the whole batch for its own pixel
        uint32_t batch_size = min(block_size, (uint32_t)(range_end - batch_start));
        for (uint32_t t = 0; t < batch_size && !done; ++t) {
            const float3_s xyo = xy_opacity_batch[t];
            const float3_s u_M = u_M_batch[t];
            const float3_s v_M = v_M_batch[t];
            const float3_s w_M = w_M_batch[t];

            // Ray-splat intersection
            float3_s h_u = { px * w_M.x - u_M.x,
                             px * w_M.y - u_M.y,
                             px * w_M.z - u_M.z };
            float3_s h_v = { py_f * w_M.x - v_M.x,
                             py_f * w_M.y - v_M.y,
                             py_f * w_M.z - v_M.z };
            float3_s zeta = cross3(h_u, h_v);
            if (zeta.z == 0.f) continue;

            float u = zeta.x / zeta.z;
            float v = zeta.y / zeta.z;

            // Merge 3D Gaussian kernel with 2D screen falloff (anti-aliasing)
            float gauss_3d = u*u + v*v;
            float dx = xyo.x - px, dy = xyo.y - py_f;
            float gauss_2d = FILTER_INV_SQ * (dx*dx + dy*dy);
            float sigma = 0.5f * fminf(gauss_3d, gauss_2d);

            if (sigma < 0.f) continue;
            float alpha = fminf(0.999f, xyo.z * __expf(-sigma));
            if (alpha < ALPHA_THRESHOLD) continue;

            float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) { done = true; break; }

            // Accumulate color
            float vis = alpha * T;
            int32_t g  = id_batch[t];
            pix_r += colors[g*3+0] * vis;
            pix_g += colors[g*3+1] * vis;
            pix_b += colors[g*3+2] * vis;

            cur_idx = batch_start + t;
            T = next_T;
        }
        block.sync();
    }

    // ── Write outputs ─────────────────────────────────────────────────────────
    if (inside) {
        render_colors[pix_id*3+0] = pix_r;
        render_colors[pix_id*3+1] = pix_g;
        render_colors[pix_id*3+2] = pix_b;
        render_alphas[pix_id]     = 1.f - T;
        last_ids[pix_id]          = cur_idx;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launch wrapper
// ─────────────────────────────────────────────────────────────────────────────

void launch_rasterize_fwd(
    const float*   d_means2d,
    const float*   d_ray_transforms,
    const float*   d_opacities,
    const float*   d_colors,
    const int32_t* d_tile_offsets,
    const int32_t* d_flatten_ids,
    int32_t        n_isects,
    uint32_t       image_width,
    uint32_t       image_height,
    // outputs
    float*   d_render_colors,
    float*   d_render_alphas,
    int32_t* d_last_ids
) {
    uint32_t tile_width  = (image_width  + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t tile_height = (image_height + TILE_SIZE - 1) / TILE_SIZE;

    // Grid: one block per tile. Threads: TILE_SIZE × TILE_SIZE per block.
    dim3 grid(tile_height, tile_width);
    dim3 block(TILE_SIZE, TILE_SIZE);

    size_t smem = TILE_SIZE * TILE_SIZE * (
        sizeof(int32_t) +   // id_batch
        sizeof(float3_s) +  // xy_opacity_batch
        sizeof(float3_s) +  // u_M_batch
        sizeof(float3_s) +  // v_M_batch
        sizeof(float3_s)    // w_M_batch
    );

    rasterize_fwd_kernel<<<grid, block, smem>>>(
        d_means2d, d_ray_transforms, d_opacities, d_colors,
        d_tile_offsets, d_flatten_ids, n_isects,
        image_width, image_height, tile_width, tile_height,
        d_render_colors, d_render_alphas, d_last_ids
    );
    CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// End-to-end test
// ─────────────────────────────────────────────────────────────────────────────
//
// Chains all three kernels: projection → tile intersection → rasterizer.
// Scene: 64×64 image, 2 Gaussians with known positions and colors.
//   G0: red,   center (16, 16), depth 1.0
//   G1: blue,  center (48, 48), depth 2.0
//
// Both Gaussians should appear in the rendered image at the correct positions.

// Include kernel implementations from the other pipeline stages.
// INCLUDED_AS_HEADER suppresses their standalone main() functions.
// Only define/undef here if the caller didn't already set it.
#ifndef INCLUDED_AS_HEADER
#  define RASTERIZE_FWD_SET_HEADER_GUARD
#  define INCLUDED_AS_HEADER
#endif
#include "projection_2dgs.cu"
#include "intersect_tile.cu"
#ifdef RASTERIZE_FWD_SET_HEADER_GUARD
#  undef INCLUDED_AS_HEADER
#  undef RASTERIZE_FWD_SET_HEADER_GUARD
#endif

#include <cub/cub.cuh>

#ifndef INCLUDED_AS_HEADER
int main() {
    printf("=== rasterize_fwd end-to-end test ===\n\n");

    const uint32_t W = 64, H = 64, TS = TILE_SIZE;
    const uint32_t TW = (W + TS - 1) / TS;   // 4
    const uint32_t TH = (H + TS - 1) / TS;   // 4
    const int N = 2;

    // ── Camera: 64×64, fx=fy=64 (90° fov), principal at center ──────────────
    float fx=64.f, fy=64.f, cx=32.f, cy=32.f, near=0.1f;

    // ── Gaussians ─────────────────────────────────────────────────────────────
    // Place each Gaussian so it projects to the desired pixel position.
    // With identity viewmat and K: projected_x = p.x/p.z * fx + cx
    // For G0 → pixel (16,16): p = ((16-32)/64 * 1.0, (16-32)/64 * 1.0, 1.0) = (-0.25,-0.25,1)
    // For G1 → pixel (48,48): p = ((48-32)/64 * 2.0, (48-32)/64 * 2.0, 2.0) = (0.5, 0.5, 2)
    float h_means[N*3]    = { -0.25f, -0.25f, 1.0f,   // G0
                               0.5f,   0.5f,  2.0f };  // G1
    float h_rotation[N*4] = { 1,0,0,0, 1,0,0,0 };     // identity quats
    float h_scaling[N*3]  = { -2.f,-2.f,-2.f,          // G0: log(e^-2)≈0.14 px scale
                               -2.f,-2.f,-2.f };        // G1: same
    float h_colors[N*3]   = { 1.f, 0.f, 0.f,           // G0: red
                               0.f, 0.f, 1.f };         // G1: blue
    float h_opacities[N]  = { 5.f, 5.f };              // logit(high) ≈ opaque

    float h_viewmat[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }; // identity

    // ── GPU allocations ───────────────────────────────────────────────────────
    float   *d_means, *d_rotation, *d_scaling, *d_viewmat;
    float   *d_colors, *d_opacities;
    float   *d_ray_transforms, *d_means2d, *d_depths, *d_normals;
    int32_t *d_radii;
    CUDA_CHECK(cudaMalloc(&d_means,          N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rotation,       N*4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scaling,        N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_viewmat,        16*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colors,         N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opacities,      N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ray_transforms, N*9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means2d,        N*2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii,          N*2*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_depths,         N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normals,        N*3*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_means,    h_means,    N*3*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rotation, h_rotation, N*4*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scaling,  h_scaling,  N*3*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_viewmat,  h_viewmat,  16*sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colors,   h_colors,   N*3*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacities,h_opacities,N*sizeof(float),    cudaMemcpyHostToDevice));

    // ── Step 1: Projection ────────────────────────────────────────────────────
    printf("Step 1: projection_2dgs...\n");
    projection_2dgs_kernel<<<1, N>>>(
        d_means, d_rotation, d_scaling, d_viewmat,
        fx, fy, cx, cy, near, (int)W, (int)H,
        d_ray_transforms, d_means2d, d_radii, d_depths, d_normals, N
    );
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    float h_m2d[N*2]; int32_t h_radii[N*2];
    CUDA_CHECK(cudaMemcpy(h_m2d,  d_means2d, N*2*sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_radii,d_radii,   N*2*sizeof(int32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
        printf("  G%d: means2d=(%.1f, %.1f)  radii=(%d, %d)\n",
               i, h_m2d[i*2], h_m2d[i*2+1], h_radii[i*2], h_radii[i*2+1]);

    // ── Step 2: Tile intersection ─────────────────────────────────────────────
    printf("Step 2: tile intersection...\n");
    int32_t *d_tiles_per_gauss, *d_cum_tiles;
    CUDA_CHECK(cudaMalloc(&d_tiles_per_gauss, N*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_cum_tiles,       N*sizeof(int32_t)));

    count_tiles_per_gauss_kernel<<<1, N>>>(
        d_means2d, d_radii, W, H, TS, TW, TH, d_tiles_per_gauss, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_tiles_per_gauss, d_cum_tiles, N);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_tiles_per_gauss, d_cum_tiles, N);
    CUDA_CHECK(cudaFree(d_tmp));

    int32_t last_cum, last_cnt;
    CUDA_CHECK(cudaMemcpy(&last_cum, d_cum_tiles+N-1,          sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_cnt, d_tiles_per_gauss+N-1,    sizeof(int32_t), cudaMemcpyDeviceToHost));
    int32_t n_isects = last_cum + last_cnt;
    printf("  n_isects = %d\n", n_isects);

    int64_t *d_isect_ids, *d_isect_ids_sorted;
    int32_t *d_flatten_ids, *d_flatten_ids_sorted, *d_tile_offsets;
    CUDA_CHECK(cudaMalloc(&d_isect_ids,         n_isects*sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_isect_ids_sorted,  n_isects*sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_flatten_ids,       n_isects*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_flatten_ids_sorted,n_isects*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_tile_offsets,      TW*TH*sizeof(int32_t)));

    fill_isect_ids_kernel<<<1, N>>>(
        d_means2d, d_radii, d_depths, d_cum_tiles,
        TS, TW, TH, d_isect_ids, d_flatten_ids, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    d_tmp = nullptr; tmp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes,
        d_isect_ids, d_isect_ids_sorted, d_flatten_ids, d_flatten_ids_sorted, n_isects);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes,
        d_isect_ids, d_isect_ids_sorted, d_flatten_ids, d_flatten_ids_sorted, n_isects);
    CUDA_CHECK(cudaFree(d_tmp));

    CUDA_CHECK(cudaMemset(d_tile_offsets, 0, TW*TH*sizeof(int32_t)));
    compute_tile_offsets_kernel<<<(n_isects+255)/256, 256>>>(
        d_isect_ids_sorted, TW*TH, (uint32_t)n_isects, d_tile_offsets);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    // ── Step 3: Rasterize ─────────────────────────────────────────────────────
    printf("Step 3: rasterize_fwd...\n");
    float   *d_render_colors; float   *d_render_alphas; int32_t *d_last_ids;
    CUDA_CHECK(cudaMalloc(&d_render_colors, H*W*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_alphas, H*W*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_last_ids,      H*W*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d_render_colors, 0, H*W*3*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_render_alphas, 0, H*W*sizeof(float)));

    launch_rasterize_fwd(
        d_means2d, d_ray_transforms, d_opacities, d_colors,
        d_tile_offsets, d_flatten_ids_sorted, n_isects,
        W, H, d_render_colors, d_render_alphas, d_last_ids
    );

    // ── Sample pixels at expected Gaussian centers ────────────────────────────
    printf("\nSampled rendered pixels:\n");
    std::vector<float>   h_colors_out(H*W*3);
    std::vector<float>   h_alphas_out(H*W);
    CUDA_CHECK(cudaMemcpy(h_colors_out.data(), d_render_colors, H*W*3*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_alphas_out.data(), d_render_alphas, H*W*sizeof(float),   cudaMemcpyDeviceToHost));

    auto sample = [&](const char* label, int px, int py) {
        int id = py * W + px;
        printf("  %-12s px=(%2d,%2d)  rgb=(%.3f, %.3f, %.3f)  alpha=%.3f\n",
               label, px, py,
               h_colors_out[id*3], h_colors_out[id*3+1], h_colors_out[id*3+2],
               h_alphas_out[id]);
    };

    sample("G0 center",  16, 16);  // expect red
    sample("G1 center",  48, 48);  // expect blue
    sample("background",  0,  0);  // expect black

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_means); cudaFree(d_rotation); cudaFree(d_scaling); cudaFree(d_viewmat);
    cudaFree(d_colors); cudaFree(d_opacities);
    cudaFree(d_ray_transforms); cudaFree(d_means2d); cudaFree(d_radii);
    cudaFree(d_depths); cudaFree(d_normals);
    cudaFree(d_tiles_per_gauss); cudaFree(d_cum_tiles);
    cudaFree(d_isect_ids); cudaFree(d_isect_ids_sorted);
    cudaFree(d_flatten_ids); cudaFree(d_flatten_ids_sorted);
    cudaFree(d_tile_offsets);
    cudaFree(d_render_colors); cudaFree(d_render_alphas); cudaFree(d_last_ids);

    printf("\nDone.\n");
    return 0;
}
#endif // INCLUDED_AS_HEADER
