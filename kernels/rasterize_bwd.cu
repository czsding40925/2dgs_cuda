// rasterize_bwd.cu
//
// Backward pass for the 2DGS rasterizer.
// Mirrors rasterize_fwd.cu but iterates back-to-front, recovering transmittance
// T going backward from T_final = 1 - render_alpha[pix].
//
// Per-pixel inner-loop math (going back-to-front, Gaussian i):
//
//   Forward replay:
//     alpha_i = min(0.999, sigmoid(raw_opacity_i) * exp(-sigma_i))
//     T_i  recovered as T_{i+1} / (1 - alpha_i)
//     fac  = alpha_i * T_i
//
//   Color gradient:
//     v_colors[g] += fac * v_render_colors[pix]
//
//   Alpha gradient (via "buffer" = sum of later contributions):
//     v_alpha = sum_c (c_i[c]*T_i - buffer[c]*ra) * v_render_c[c]
//     (where ra = 1/(1-alpha_i))
//     buffer[c] += c_i[c] * fac   // updated AFTER gradient
//
//   Case 1 — 3D Gaussian kernel active (gauss_3d <= gauss_2d):
//     v_G = opac * v_alpha        (d(alpha)/d(vis) = opac)
//     v_s = -vis * v_G * [u, v]   (d(vis)/d(u²+v²) → chain through sigma)
//     → v_ray_transforms via cross-product chain rule
//
//   Case 2 — 2D screen falloff active (gauss_3d > gauss_2d):
//     v_means2d[g] += -vis * FILTER_INV_SQ * d * v_G
//
//   Opacity gradient (raw logit):
//     v_opacities[g] += vis * v_alpha * opac * (1 - opac)  (sigmoid deriv)
//
// All output gradients accumulated via atomicAdd (multiple pixels → same Gaussian).
// Shared memory layout identical to forward pass.

#include "splat_data.cuh"
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

static constexpr float BWD_FILTER_INV_SQ = 2.0f;
static constexpr float BWD_ALPHA_THRESHOLD = 1.f / 255.f;
static constexpr float BWD_RASTER_NEAR_PLANE = 0.2f;

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

struct bwd_f3 { float x, y, z; };

__device__ inline bwd_f3 cross3_b(bwd_f3 a, bwd_f3 b) {
    return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward kernel
// ─────────────────────────────────────────────────────────────────────────────

__global__ void rasterize_bwd_kernel(
    // fwd inputs (for replaying forward)
    const float*   __restrict__ means2d,        // [N, 2]
    const float*   __restrict__ ray_transforms, // [N, 9]
    const float*   __restrict__ opacities,      // [N]  raw logit
    const float*   __restrict__ colors,         // [N, 3]
    const float*   __restrict__ normals,        // [N, 3]
    const int32_t* __restrict__ tile_offsets,   // [tile_h * tile_w]
    const int32_t* __restrict__ flatten_ids,    // [n_isects]
    const int32_t  n_isects,
    // fwd outputs
    const float*   __restrict__ render_alphas,  // [H, W]   1 - T_final
    const float*   __restrict__ render_depth_accum, // [H, W] optional
    const int32_t* __restrict__ last_ids,       // [H, W]   index of last contributor
    // loss gradient into the rasterizer
    const float*   __restrict__ v_render_colors, // [H*W, 3]
    const float*   __restrict__ v_render_alphas, // [H*W] or nullptr
    const float*   __restrict__ v_render_normals, // [H*W, 3] or nullptr
    const float*   __restrict__ v_render_depth_accum, // [H*W] or nullptr
    const float*   __restrict__ v_render_distort, // [H*W] or nullptr
    // image dims
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs — zeroed before launch, accumulated via atomicAdd
    float*   __restrict__ v_ray_transforms, // [N, 9]
    float*   __restrict__ v_opacities,      // [N]
    float*   __restrict__ v_colors,         // [N, 3]
    float*   __restrict__ v_normals,        // [N, 3]
    float*   __restrict__ v_means2d,        // [N, 2]
    float*   __restrict__ v_means2d_abs     // [N, 2]  for densification
) {
    auto block = cg::this_thread_block();

    // ── Block / thread → tile / pixel ────────────────────────────────────────
    uint32_t tile_y = block.group_index().x;
    uint32_t tile_x = block.group_index().y;
    uint32_t tile_id = tile_y * tile_width + tile_x;

    uint32_t py   = tile_y * 16u + block.thread_index().x;
    uint32_t px_i = tile_x * 16u + block.thread_index().y;
    float    px   = (float)px_i + 0.5f;
    float    py_f = (float)py    + 0.5f;
    int32_t  pix_id = (int32_t)(py * image_width + px_i);

    bool inside = (py < image_height && px_i < image_width);

    // ── Tile range in sorted list ─────────────────────────────────────────────
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end   = (tile_id == tile_width * tile_height - 1u)
                          ? n_isects : tile_offsets[tile_id + 1];

    const uint32_t block_size = block.size();
    const int32_t  num_batches = (int32_t)((range_end - range_start + block_size - 1) / block_size);

    // ── Shared memory (same layout as forward) ───────────────────────────────
    extern __shared__ int bsmem[];
    int32_t* id_batch         = (int32_t*)bsmem;
    bwd_f3*  xy_opacity_batch = (bwd_f3*)&id_batch[block_size];
    bwd_f3*  u_M_batch        = (bwd_f3*)&xy_opacity_batch[block_size];
    bwd_f3*  v_M_batch        = (bwd_f3*)&u_M_batch[block_size];
    bwd_f3*  w_M_batch        = (bwd_f3*)&v_M_batch[block_size];
    float*   rgb_batch        = (float* )&w_M_batch[block_size]; // [block_size * 3]
    float*   normal_batch     = rgb_batch + block_size * 3;      // [block_size * 3]

    // ── Per-pixel state ───────────────────────────────────────────────────────
    float T_final  = inside ? 1.f - render_alphas[pix_id] : 1.f;
    float T        = T_final;
    int32_t bin_final = inside ? last_ids[pix_id] : -1;

    // Accumulated "later" contributions: buffer[c] = sum_{j > cur} c_j*alpha_j*T_j
    float buf_r = 0.f, buf_g = 0.f, buf_b = 0.f;
    float buf_nx = 0.f, buf_ny = 0.f, buf_nz = 0.f;
    float buf_depth = 0.f;

    // Per-pixel loss gradient
    float v_r = inside ? v_render_colors[pix_id*3+0] : 0.f;
    float v_g = inside ? v_render_colors[pix_id*3+1] : 0.f;
    float v_b = inside ? v_render_colors[pix_id*3+2] : 0.f;
    float v_a = (inside && v_render_alphas != nullptr) ? v_render_alphas[pix_id] : 0.f;
    float v_nx = (inside && v_render_normals != nullptr) ? v_render_normals[pix_id*3+0] : 0.f;
    float v_ny = (inside && v_render_normals != nullptr) ? v_render_normals[pix_id*3+1] : 0.f;
    float v_nz = (inside && v_render_normals != nullptr) ? v_render_normals[pix_id*3+2] : 0.f;
    float v_depth_acc = (inside && v_render_depth_accum != nullptr) ? v_render_depth_accum[pix_id] : 0.f;
    float v_distort = (inside && v_render_distort != nullptr) ? v_render_distort[pix_id] : 0.f;
    float accum_d = 0.f, accum_w = 0.f;
    float accum_d_buffer = 0.f, accum_w_buffer = 0.f;
    float distort_buffer = 0.f;
    if (inside && v_render_distort != nullptr && render_depth_accum != nullptr) {
        accum_d = render_depth_accum[pix_id];
        accum_w = render_alphas[pix_id];
        accum_d_buffer = accum_d;
        accum_w_buffer = accum_w;
    }

    uint32_t tr = block.thread_rank();

    // ── Main loop — batches back-to-front ─────────────────────────────────────
    for (int32_t b = 0; b < num_batches; ++b) {

        block.sync();

        // Thread tr loads element at batch_end - tr (back-to-front assignment)
        int32_t batch_end  = range_end - 1 - (int32_t)(block_size * b);
        int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);

        int32_t load_idx = batch_end - (int32_t)tr;
        if (load_idx >= range_start) {
            int32_t g = flatten_ids[load_idx];
            id_batch[tr]         = g;
            float opac = SplatData::sigmoid(opacities[g]);
            xy_opacity_batch[tr] = { means2d[g*2], means2d[g*2+1], opac };
            u_M_batch[tr]        = { ray_transforms[g*9],   ray_transforms[g*9+1], ray_transforms[g*9+2] };
            v_M_batch[tr]        = { ray_transforms[g*9+3], ray_transforms[g*9+4], ray_transforms[g*9+5] };
            w_M_batch[tr]        = { ray_transforms[g*9+6], ray_transforms[g*9+7], ray_transforms[g*9+8] };
            rgb_batch[tr*3+0]    = colors[g*3+0];
            rgb_batch[tr*3+1]    = colors[g*3+1];
            rgb_batch[tr*3+2]    = colors[g*3+2];
            normal_batch[tr*3+0] = normals[g*3+0];
            normal_batch[tr*3+1] = normals[g*3+1];
            normal_batch[tr*3+2] = normals[g*3+2];
        }

        block.sync();

        // Inner loop: t=0 = furthest-back element (batch_end), back-to-front
        for (int32_t t = 0; t < batch_size; ++t) {

            // Only process pixels where this Gaussian was in range
            bool valid = inside && (batch_end - t <= bin_final);
            if (!valid) continue;

            const bwd_f3 xyo = xy_opacity_batch[t];
            const bwd_f3 uM  = u_M_batch[t];
            const bwd_f3 vM  = v_M_batch[t];
            const bwd_f3 wM  = w_M_batch[t];
            float opac = xyo.z;

            // ── Replay forward ──────────────────────────────────────────────
            bwd_f3 h_u = { px * wM.x - uM.x, px * wM.y - uM.y, px * wM.z - uM.z };
            bwd_f3 h_v = { py_f * wM.x - vM.x, py_f * wM.y - vM.y, py_f * wM.z - vM.z };
            bwd_f3 zeta = cross3_b(h_u, h_v);
            if (zeta.z == 0.f) continue;

            float u = zeta.x / zeta.z;
            float v = zeta.y / zeta.z;
            float depth = u * wM.x + v * wM.y + wM.z;
            if (depth < BWD_RASTER_NEAR_PLANE) continue;

            float gauss_3d = u*u + v*v;
            float dx = xyo.x - px, dy = xyo.y - py_f;
            float gauss_2d = BWD_FILTER_INV_SQ * (dx*dx + dy*dy);
            float sigma = 0.5f * fminf(gauss_3d, gauss_2d);
            if (sigma < 0.f) continue;

            float vis   = __expf(-sigma);
            float alpha = fminf(0.999f, opac * vis);
            if (alpha < BWD_ALPHA_THRESHOLD) continue;

            // ── Recover T_i ─────────────────────────────────────────────────
            float ra = 1.f / (1.f - alpha);
            T *= ra;          // T_i = T_{i+1} / (1 - alpha_i)
            float fac = alpha * T;

            // ── Color gradient ───────────────────────────────────────────────
            float c_r = rgb_batch[t*3+0];
            float c_g = rgb_batch[t*3+1];
            float c_b = rgb_batch[t*3+2];
            float n_r = normal_batch[t*3+0];
            float n_g = normal_batch[t*3+1];
            float n_b = normal_batch[t*3+2];

            float v_c_r = fac * v_r;
            float v_c_g = fac * v_g;
            float v_c_b = fac * v_b;
            float v_n_r = fac * v_nx;
            float v_n_g = fac * v_ny;
            float v_n_b = fac * v_nz;

            // ── Alpha gradient ───────────────────────────────────────────────
            float v_alpha =
                (c_r * T - buf_r * ra) * v_r +
                (c_g * T - buf_g * ra) * v_g +
                (c_b * T - buf_b * ra) * v_b;
            if (v_render_normals != nullptr) {
                v_alpha +=
                    (n_r * T - buf_nx * ra) * v_nx +
                    (n_g * T - buf_ny * ra) * v_ny +
                    (n_b * T - buf_nz * ra) * v_nz;
            }
            if (v_render_depth_accum != nullptr) {
                v_alpha += (depth * T - buf_depth * ra) * v_depth_acc;
            }
            if (v_render_alphas != nullptr) {
                v_alpha += T_final * ra * v_a;
            }

            // Update buffer AFTER computing gradient
            buf_r += c_r * fac;
            buf_g += c_g * fac;
            buf_b += c_b * fac;
            if (v_render_normals != nullptr) {
                buf_nx += n_r * fac;
                buf_ny += n_g * fac;
                buf_nz += n_b * fac;
            }
            if (v_render_depth_accum != nullptr) {
                buf_depth += depth * fac;
            }

            float v_depth = fac * v_depth_acc;
            if (v_render_distort != nullptr) {
                float dl_dw =
                    2.f * (2.f * (depth * accum_w_buffer - accum_d_buffer) +
                           (accum_d - depth * accum_w));
                v_alpha += (dl_dw * T - distort_buffer * ra) * v_distort;
                accum_d_buffer -= fac * depth;
                accum_w_buffer -= fac;
                distort_buffer += dl_dw * fac;
                v_depth += 2.f * fac * (2.f - 2.f * T - accum_w + fac) * v_distort;
            }

            // ── Grad through alpha = opac * vis ──────────────────────────────
            float v_G    = opac * v_alpha;       // d(alpha)/d(vis) = opac
            float v_opac = vis  * v_alpha;       // d(alpha)/d(opac) = vis
            // Through sigmoid: d(opac)/d(raw) = opac*(1-opac)
            float v_raw_opacity = v_opac * opac * (1.f - opac);

            // ── Geometry gradient ────────────────────────────────────────────
            float v_uM0=0, v_uM1=0, v_uM2=0;
            float v_vM0=0, v_vM1=0, v_vM2=0;
            float v_wM0=0, v_wM1=0, v_wM2=0;
            float v_xy0=0, v_xy1=0;
            float v_xyabs0=0, v_xyabs1=0;
            float v_s_x = v_depth * wM.x;
            float v_s_y = v_depth * wM.y;
            float v_w_extra0 = v_depth * u;
            float v_w_extra1 = v_depth * v;
            float v_w_extra2 = v_depth;

            if (opac * vis <= 0.999f) {
                if (gauss_3d <= gauss_2d) {
                    // Case 1: 3D kernel — backprop through ray-splat intersection
                    // vis = exp(-0.5*(u²+v²))   d(vis)/d(u) = -vis*u
                    v_s_x += v_G * (-vis) * u;
                    v_s_y += v_G * (-vis) * v;
                } else {
                    // Case 2: 2D screen falloff — gradient goes to means2d
                    float v_G_dx = -vis * BWD_FILTER_INV_SQ * dx;
                    float v_G_dy = -vis * BWD_FILTER_INV_SQ * dy;
                    v_xy0 = v_G * v_G_dx;
                    v_xy1 = v_G * v_G_dy;
                    v_xyabs0 = fabsf(v_xy0);
                    v_xyabs1 = fabsf(v_xy1);
                }
            }

            if (v_s_x != 0.f || v_s_y != 0.f || v_w_extra0 != 0.f || v_w_extra1 != 0.f || v_w_extra2 != 0.f) {
                float v_sx_pz = v_s_x / zeta.z;
                float v_sy_pz = v_s_y / zeta.z;
                bwd_f3 v_zeta = {
                    v_sx_pz,
                    v_sy_pz,
                    -(v_sx_pz * u + v_sy_pz * v)
                };

                bwd_f3 v_h_u = cross3_b(h_v, v_zeta);
                bwd_f3 v_h_v = cross3_b(v_zeta, h_u);

                v_uM0 += -v_h_u.x;   v_uM1 += -v_h_u.y;   v_uM2 += -v_h_u.z;
                v_vM0 += -v_h_v.x;   v_vM1 += -v_h_v.y;   v_vM2 += -v_h_v.z;
                v_wM0 += px * v_h_u.x + py_f * v_h_v.x + v_w_extra0;
                v_wM1 += px * v_h_u.y + py_f * v_h_v.y + v_w_extra1;
                v_wM2 += px * v_h_u.z + py_f * v_h_v.z + v_w_extra2;
            }

            // ── Accumulate into global memory ────────────────────────────────
            int32_t g = id_batch[t];
            atomicAdd(&v_ray_transforms[g*9+0], v_uM0);
            atomicAdd(&v_ray_transforms[g*9+1], v_uM1);
            atomicAdd(&v_ray_transforms[g*9+2], v_uM2);
            atomicAdd(&v_ray_transforms[g*9+3], v_vM0);
            atomicAdd(&v_ray_transforms[g*9+4], v_vM1);
            atomicAdd(&v_ray_transforms[g*9+5], v_vM2);
            atomicAdd(&v_ray_transforms[g*9+6], v_wM0);
            atomicAdd(&v_ray_transforms[g*9+7], v_wM1);
            atomicAdd(&v_ray_transforms[g*9+8], v_wM2);
            atomicAdd(&v_opacities[g],          v_raw_opacity);
            atomicAdd(&v_colors[g*3+0],         v_c_r);
            atomicAdd(&v_colors[g*3+1],         v_c_g);
            atomicAdd(&v_colors[g*3+2],         v_c_b);
            atomicAdd(&v_normals[g*3+0],        v_n_r);
            atomicAdd(&v_normals[g*3+1],        v_n_g);
            atomicAdd(&v_normals[g*3+2],        v_n_b);
            atomicAdd(&v_means2d[g*2+0],        v_xy0);
            atomicAdd(&v_means2d[g*2+1],        v_xy1);
            atomicAdd(&v_means2d_abs[g*2+0],    v_xyabs0);
            atomicAdd(&v_means2d_abs[g*2+1],    v_xyabs1);
        }

        block.sync();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launch wrapper
// ─────────────────────────────────────────────────────────────────────────────

void launch_rasterize_bwd(
    const float*   d_means2d,
    const float*   d_ray_transforms,
    const float*   d_opacities,
    const float*   d_colors,
    const float*   d_normals,
    const int32_t* d_tile_offsets,
    const int32_t* d_flatten_ids,
    int32_t        n_isects,
    const float*   d_render_alphas,
    const float*   d_render_depth_accum,
    const int32_t* d_last_ids,
    const float*   d_v_render_colors,
    const float*   d_v_render_alphas,
    const float*   d_v_render_normals,
    const float*   d_v_render_depth_accum,
    const float*   d_v_render_distort,
    uint32_t image_width,
    uint32_t image_height,
    float*   d_v_ray_transforms,
    float*   d_v_opacities,
    float*   d_v_colors,
    float*   d_v_normals,
    float*   d_v_means2d,
    float*   d_v_means2d_abs
) {
    uint32_t tile_width  = (image_width  + 15u) / 16u;
    uint32_t tile_height = (image_height + 15u) / 16u;

    dim3 grid(tile_height, tile_width);
    dim3 block(16u, 16u);

    size_t smem = 256u * (sizeof(int32_t) + sizeof(bwd_f3)*4u + sizeof(float)*6u);

    rasterize_bwd_kernel<<<grid, block, smem>>>(
        d_means2d, d_ray_transforms, d_opacities, d_colors, d_normals,
        d_tile_offsets, d_flatten_ids, n_isects,
        d_render_alphas, d_render_depth_accum, d_last_ids,
        d_v_render_colors, d_v_render_alphas, d_v_render_normals,
        d_v_render_depth_accum, d_v_render_distort,
        image_width, image_height, tile_width, tile_height,
        d_v_ray_transforms, d_v_opacities, d_v_colors, d_v_normals,
        d_v_means2d, d_v_means2d_abs
    );
    CUDA_CHECK(cudaGetLastError());
}
