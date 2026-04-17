// train.cu — 2DGS training entry point
//
// Usage:
//   ./build/train --data path/to/dataset [--iters 30000] [--sh-degree 3]
//   ./build/train --data path/to/dataset --ply splats/foo.ply --render out.png [--cam 0]
//
// What this does right now:
//   1. Load COLMAP scene (cameras + point cloud or --ply checkpoint)
//   2. Initialize SplatData from the point cloud (or PLY file)
//   3. Forward pass each iteration:
//        projection_2dgs → tile_intersect → rasterize_fwd
//      Colors: DC SH term (sh0 * C0 + 0.5), clamped to [0,1]
//   4. Adam optimizer state is allocated and ready for backward gradients
//   5. Loss / backward / basic prune/clone/split densification
//
// --ply <file>       Load a trained checkpoint instead of COLMAP point cloud
// --render <file>    Render camera --cam N and save PNG, then exit
// --cam <N>          Which COLMAP camera to render (default 0)
// --orbit <N>        Render N-frame orbit sequence and exit
// --orbit-out <pfx>  Output filename prefix for orbit (default "orbit" → orbit_0000.png …)
//
// Dataset layout expected (MipNeRF-360, Tanks & Temples):
//   dataset/
//     sparse/0/cameras.bin  images.bin  points3D.bin
//     images/

// Pull in all three forward kernels as a single compilation unit.
// INCLUDED_AS_HEADER suppresses standalone main() in each file.
#define INCLUDED_AS_HEADER
#include "kernels/rasterize_fwd.cu"   // also includes projection_2dgs.cu + intersect_tile.cu
#include "kernels/rasterize_bwd.cu"
#include "kernels/projection_2dgs_bwd.cu"
#include "kernels/adam.cu"
#include "kernels/loss.cu"
#undef INCLUDED_AS_HEADER

#include "kernels/colmap_reader.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "../LichtFeld-Studio/external/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../LichtFeld-Studio/external/stb_image_write.h"

// ---------------------------------------------------------------------------
// CLI parsing  →  train_config.hpp
// ---------------------------------------------------------------------------

#include "train_config.hpp"


static void print_training_progress(
    int iter,
    int iters,
    int sh_active,
    int sh_max,
    int n_isects,
    const LossResult& loss,
    double elapsed_sec,
    bool final_line
) {
    constexpr int BAR_W = 28;
    float frac = (iters > 0) ? (float)iter / (float)iters : 1.f;
    int filled = (int)(frac * BAR_W + 0.5f);
    if (filled < 0) filled = 0;
    if (filled > BAR_W) filled = BAR_W;

    char bar[BAR_W + 1];
    for (int i = 0; i < BAR_W; i++)
        bar[i] = (i < filled) ? '=' : ' ';
    bar[BAR_W] = '\0';

    double iter_per_sec = elapsed_sec > 0.0 ? iter / elapsed_sec : 0.0;
    double eta_sec = (iter_per_sec > 0.0) ? (iters - iter) / iter_per_sec : 0.0;

    int eta_m = (int)(eta_sec / 60.0);
    int eta_s = (int)eta_sec % 60;

    printf("\r[%s] %5d/%d  sh=%d/%d  loss=%.5f  l1=%.5f  dssim=%.5f  isects=%d  %.2fit/s  eta=%02d:%02d",
           bar, iter, iters,
           sh_active, sh_max,
           loss.loss, loss.loss_l1, loss.loss_dssim,
           n_isects, iter_per_sec, eta_m, eta_s);
    if (final_line) printf("\n");
    fflush(stdout);
}

static bool save_device_rgb_png(
    const float* d_render_colors,
    uint32_t W,
    uint32_t H,
    const std::string& path
) {
    std::filesystem::path p(path);
    if (!p.parent_path().empty())
        std::filesystem::create_directories(p.parent_path());

    std::vector<float> h_render((size_t)W * H * 3);
    CUDA_CHECK(cudaMemcpy(h_render.data(), d_render_colors,
                          W * H * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<uint8_t> pixels((size_t)W * H * 3);
    for (size_t i = 0; i < pixels.size(); i++)
        pixels[i] = (uint8_t)(std::max(0.f, std::min(1.f, h_render[i])) * 255.f + 0.5f);

    return stbi_write_png(path.c_str(), (int)W, (int)H, 3, pixels.data(), (int)W * 3) != 0;
}

static bool save_ply_checkpoint_atomic(const SplatData& splats, const std::string& path) {
    try {
        std::filesystem::path out(path);
        if (!out.parent_path().empty())
            std::filesystem::create_directories(out.parent_path());

        const std::string tmp_path = path + ".tmp";
        splats.save_to_ply(tmp_path);

        std::error_code ec;
        std::filesystem::rename(tmp_path, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(tmp_path, path, ec);
            if (ec)
                throw std::runtime_error("rename failed: " + ec.message());
        }
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error saving PLY checkpoint to %s: %s\n", path.c_str(), e.what());
        return false;
    }
}

static void maybe_save_ply_checkpoint(
    const Config& cfg,
    int iter,
    bool final_checkpoint,
    const SplatData& splats
) {
    if (cfg.save_ply.empty()) return;

    const bool periodic_checkpoint =
        (cfg.save_ply_every > 0) && (iter % cfg.save_ply_every == 0);
    if (!final_checkpoint && !periodic_checkpoint) return;

    bool ok = save_ply_checkpoint_atomic(splats, cfg.save_ply);
    printf("\ncheckpoint: iter=%d → %s%s\n",
           iter, cfg.save_ply.c_str(), ok ? "" : "  [WRITE FAILED]");
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

struct TrainingImageEntry {
    uint32_t W = 0;
    uint32_t H = 0;
    std::vector<unsigned char> rgb_u8;
};

struct TrainingImageSet {
    std::vector<TrainingImageEntry> entries;
    size_t total_bytes = 0;
    bool cached = false;
};

static std::string format_bytes(size_t bytes) {
    char buf[64];
    const double kib = 1024.0;
    const double mib = kib * 1024.0;
    const double gib = mib * 1024.0;
    if (bytes >= (size_t)gib)
        std::snprintf(buf, sizeof(buf), "%.2f GiB", bytes / gib);
    else if (bytes >= (size_t)mib)
        std::snprintf(buf, sizeof(buf), "%.1f MiB", bytes / mib);
    else if (bytes >= (size_t)kib)
        std::snprintf(buf, sizeof(buf), "%.1f KiB", bytes / kib);
    else
        std::snprintf(buf, sizeof(buf), "%zu B", bytes);
    return std::string(buf);
}

static unsigned char* load_rgb_image_u8(const std::string& path, int expected_W, int expected_H) {
    int W = 0, H = 0, C = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &W, &H, &C, 3);
    if (!pixels) {
        throw std::runtime_error("failed to load image '" + path + "': " + stbi_failure_reason());
    }
    if (W != expected_W || H != expected_H) {
        stbi_image_free(pixels);
        throw std::runtime_error(
            "image dimensions for '" + path + "' are " + std::to_string(W) + "x" + std::to_string(H) +
            ", expected " + std::to_string(expected_W) + "x" + std::to_string(expected_H) +
            ". Use --images with the folder matching COLMAP intrinsics.");
    }
    return pixels;
}

static void rescale_intrinsics(Intrinsics& K, uint32_t new_W, uint32_t new_H) {
    float sx = (K.width  > 0) ? (float)new_W / (float)K.width  : 1.f;
    float sy = (K.height > 0) ? (float)new_H / (float)K.height : 1.f;
    K.fx *= sx;
    K.fy *= sy;
    K.cx *= sx;
    K.cy *= sy;
    K.width = new_W;
    K.height = new_H;
}

static TrainingImageSet prepare_training_images(
    ColmapScene& scene,
    const std::string& images_subdir,
    bool cache_pixels
) {
    static constexpr size_t kImageCacheMaxBytes = 2ull * 1024ull * 1024ull * 1024ull;

    TrainingImageSet out{};
    out.entries.resize(scene.cameras.size());

    bool intrinsics_rescaled = false;
    float min_sx = std::numeric_limits<float>::max();
    float max_sx = 0.f;
    float min_sy = std::numeric_limits<float>::max();
    float max_sy = 0.f;

    for (size_t i = 0; i < scene.cameras.size(); i++) {
        CameraInfo& cam = scene.cameras[i];
        TrainingImageEntry& entry = out.entries[i];

        int W = 0, H = 0, C = 0;
        if (!stbi_info(cam.image_path.c_str(), &W, &H, &C)) {
            throw std::runtime_error(
                "failed to inspect image '" + cam.image_path + "': " + stbi_failure_reason());
        }

        entry.W = (uint32_t)W;
        entry.H = (uint32_t)H;
        out.total_bytes += (size_t)W * (size_t)H * 3ull;

        if ((uint32_t)W != cam.K.width || (uint32_t)H != cam.K.height) {
            float sx = (cam.K.width  > 0) ? (float)W / (float)cam.K.width  : 1.f;
            float sy = (cam.K.height > 0) ? (float)H / (float)cam.K.height : 1.f;
            min_sx = std::min(min_sx, sx);
            max_sx = std::max(max_sx, sx);
            min_sy = std::min(min_sy, sy);
            max_sy = std::max(max_sy, sy);
            rescale_intrinsics(cam.K, (uint32_t)W, (uint32_t)H);
            intrinsics_rescaled = true;
        }
    }

    out.cached = cache_pixels && out.total_bytes <= kImageCacheMaxBytes;
    if (out.cached) {
        for (size_t i = 0; i < scene.cameras.size(); i++) {
            const CameraInfo& cam = scene.cameras[i];
            TrainingImageEntry& entry = out.entries[i];
            unsigned char* pixels = load_rgb_image_u8(cam.image_path, (int)entry.W, (int)entry.H);
            entry.rgb_u8.assign(pixels, pixels + (size_t)entry.W * entry.H * 3ull);
            stbi_image_free(pixels);
        }
    }

    printf("Image set   : %zu views  total=%s  cache=%s\n",
           scene.cameras.size(),
           format_bytes(out.total_bytes).c_str(),
           out.cached ? "enabled" : "disabled");
    if (intrinsics_rescaled) {
        printf("Intrinsics  : matched '%s' image folder  scale_x=[%.3f, %.3f]  scale_y=[%.3f, %.3f]\n",
               images_subdir.c_str(), min_sx, max_sx, min_sy, max_sy);
    }
    if (cache_pixels && !out.cached) {
        printf("Image cache : skipped because dataset exceeds %s\n",
               format_bytes(kImageCacheMaxBytes).c_str());
    }

    return out;
}

// ---------------------------------------------------------------------------
// Dataset summary
// ---------------------------------------------------------------------------

static void print_scene_summary(const ColmapScene& scene) {
    printf("Cameras : %zu\n", scene.cameras.size());
    printf("Points3D: %zu\n", scene.points3D.size());

    if (!scene.cameras.empty()) {
        const auto& K = scene.cameras[0].K;
        printf("First camera: %ux%u  fx=%.1f fy=%.1f  cx=%.1f cy=%.1f\n",
               K.width, K.height, K.fx, K.fy, K.cx, K.cy);
    }

    float xmin=1e9, xmax=-1e9, ymin=1e9, ymax=-1e9, zmin=1e9, zmax=-1e9;
    for (const auto& p : scene.points3D) {
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);
    }
    printf("Scene bbox: x[%.2f, %.2f]  y[%.2f, %.2f]  z[%.2f, %.2f]\n",
           xmin, xmax, ymin, ymax, zmin, zmax);
}

static float compute_scene_scale(const ColmapScene& scene) {
    if (scene.cameras.empty()) return 1.f;

    float cx = 0.f, cy = 0.f, cz = 0.f;
    for (const auto& cam : scene.cameras) {
        cx += cam.camtoworld[0][3];
        cy += cam.camtoworld[1][3];
        cz += cam.camtoworld[2][3];
    }
    float inv_n = 1.f / (float)scene.cameras.size();
    cx *= inv_n;
    cy *= inv_n;
    cz *= inv_n;

    float diagonal = 0.f;
    for (const auto& cam : scene.cameras) {
        float dx = cam.camtoworld[0][3] - cx;
        float dy = cam.camtoworld[1][3] - cy;
        float dz = cam.camtoworld[2][3] - cz;
        diagonal = std::max(diagonal, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return std::max(1e-4f, diagonal * 1.1f);
}

// ---------------------------------------------------------------------------
// Spherical harmonics evaluation (degrees 0–3)
//
// Coefficients match gsplat/cuda/csrc/utils.cuh and the original
// 3DGS implementation.
//
// SH layout in SplatData:
//   sh0  [N, 3]   — degree-0 coefficient per channel (DC term)
//   shN  [N, 45]  — degrees 1–3, packed as [N, 3 channels × 15 coefficients]
//                   shN[i*45 + c*15 + k]  = coefficient k, channel c, Gaussian i
//
// View direction convention: d = normalize(mean - cam_pos)
// Color: clamp(SH_sum + 0.5, 0, 1)
// ---------------------------------------------------------------------------

static constexpr float SH_C0   =  0.28209479177f;   // 1/(2√π)

// Degree 1
static constexpr float SH_C1   =  0.48860251190f;   // √(3/(4π))

// Degree 2  (Y_{2,m} normalization factors)
static constexpr float SH_C2_0 =  1.09254843059f;
static constexpr float SH_C2_1 = -1.09254843059f;
static constexpr float SH_C2_2 =  0.31539156525f;
static constexpr float SH_C2_3 = -1.09254843059f;
static constexpr float SH_C2_4 =  0.54627421529f;

// Degree 3
static constexpr float SH_C3_0 = -0.59004358992f;
static constexpr float SH_C3_1 =  2.89061144264f;
static constexpr float SH_C3_2 = -0.45704579946f;
static constexpr float SH_C3_3 =  0.37317633259f;
static constexpr float SH_C3_4 = -0.45704579946f;
static constexpr float SH_C3_5 =  1.44530572132f;
static constexpr float SH_C3_6 = -0.59004358992f;

// Evaluate SH color for all N Gaussians given the camera world position.
// sh_active  — number of active SH degrees (0 = DC only, up to 3)
// shN may be nullptr when sh_active == 0.
__global__ void sh_eval_kernel(
    const float* __restrict__ means,   // [N, 3]  world positions
    const float* __restrict__ sh0,     // [N, 3]  degree-0 coefficients
    const float* __restrict__ shN,     // [N, 45] degrees 1–3 coefficients
    float cam_x, float cam_y, float cam_z,
    float* __restrict__ colors,        // [N, 3]  output, clamped [0,1]
    int sh_active, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // View direction: Gaussian → camera, normalized
    float dx = means[i*3+0] - cam_x;
    float dy = means[i*3+1] - cam_y;
    float dz = means[i*3+2] - cam_z;
    float len_inv = rsqrtf(dx*dx + dy*dy + dz*dz + 1e-12f);
    float x = dx * len_inv;
    float y = dy * len_inv;
    float z = dz * len_inv;

    for (int c = 0; c < 3; c++) {
        float v = SH_C0 * sh0[i*3 + c];

        if (sh_active >= 1) {
            // shN layout: [N, channel, 15 coefficients]
            const float* sh = shN + i*45 + c*15;
            // Degree 1: Y_{1,-1}=-C1*y,  Y_{1,0}=C1*z,  Y_{1,1}=-C1*x
            v += SH_C1 * (-sh[0]*y + sh[1]*z - sh[2]*x);

            if (sh_active >= 2) {
                // Degree 2
                float xx = x*x, yy = y*y, zz = z*z;
                v += SH_C2_0 * sh[3] * (x*y);
                v += SH_C2_1 * sh[4] * (y*z);
                v += SH_C2_2 * sh[5] * (2.f*zz - xx - yy);
                v += SH_C2_3 * sh[6] * (x*z);
                v += SH_C2_4 * sh[7] * (xx - yy);

                if (sh_active >= 3) {
                    // Degree 3
                    v += SH_C3_0 * sh[8]  * y * (3.f*xx - yy);
                    v += SH_C3_1 * sh[9]  * x * y * z;
                    v += SH_C3_2 * sh[10] * y * (4.f*zz - xx - yy);
                    v += SH_C3_3 * sh[11] * z * (2.f*zz - 3.f*xx - 3.f*yy);
                    v += SH_C3_4 * sh[12] * x * (4.f*zz - xx - yy);
                    v += SH_C3_5 * sh[13] * z * (xx - yy);
                    v += SH_C3_6 * sh[14] * x * (xx - 3.f*yy);
                }
            }
        }

        colors[i*3 + c] = fmaxf(0.f, fminf(1.f, v + 0.5f));
    }
}

// Backward for SH evaluation.
//
// This propagates dL/d(color) to SH coefficients only. We intentionally drop
// gradients through the view direction d = normalize(mean - cam_pos), which
// would couple SH color back into means. That is a common simplification for
// Gaussian splatting trainers and keeps the first backward path focused on the
// main geometric gradients coming from rasterization/projection.
__global__ void sh_backward_kernel(
    const float* __restrict__ means,      // [N, 3]
    const float* __restrict__ sh0,        // [N, 3]
    const float* __restrict__ shN,        // [N, 45]
    float cam_x, float cam_y, float cam_z,
    const float* __restrict__ v_colors,   // [N, 3]
    float* __restrict__ v_sh0,            // [N, 3]
    float* __restrict__ v_shN,            // [N, 45]
    int sh_active, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float dx = means[i*3+0] - cam_x;
    float dy = means[i*3+1] - cam_y;
    float dz = means[i*3+2] - cam_z;
    float len_inv = rsqrtf(dx*dx + dy*dy + dz*dz + 1e-12f);
    float x = dx * len_inv;
    float y = dy * len_inv;
    float z = dz * len_inv;

    float xx = x*x, yy = y*y, zz = z*z;

    for (int c = 0; c < 3; c++) {
        float v = SH_C0 * sh0[i*3 + c];
        const float* sh = (shN != nullptr) ? (shN + i*45 + c*15) : nullptr;

        if (sh_active >= 1 && sh != nullptr) {
            v += SH_C1 * (-sh[0]*y + sh[1]*z - sh[2]*x);

            if (sh_active >= 2) {
                v += SH_C2_0 * sh[3] * (x*y);
                v += SH_C2_1 * sh[4] * (y*z);
                v += SH_C2_2 * sh[5] * (2.f*zz - xx - yy);
                v += SH_C2_3 * sh[6] * (x*z);
                v += SH_C2_4 * sh[7] * (xx - yy);

                if (sh_active >= 3) {
                    v += SH_C3_0 * sh[8]  * y * (3.f*xx - yy);
                    v += SH_C3_1 * sh[9]  * x * y * z;
                    v += SH_C3_2 * sh[10] * y * (4.f*zz - xx - yy);
                    v += SH_C3_3 * sh[11] * z * (2.f*zz - 3.f*xx - 3.f*yy);
                    v += SH_C3_4 * sh[12] * x * (4.f*zz - xx - yy);
                    v += SH_C3_5 * sh[13] * z * (xx - yy);
                    v += SH_C3_6 * sh[14] * x * (xx - 3.f*yy);
                }
            }
        }

        const float unclamped = v + 0.5f;
        if (unclamped <= 0.f || unclamped >= 1.f) {
            v_sh0[i*3 + c] = 0.f;
            if (v_shN != nullptr) {
                float* out = v_shN + i*45 + c*15;
                for (int k = 0; k < 15; k++) out[k] = 0.f;
            }
            continue;
        }

        const float grad = v_colors[i*3 + c];
        v_sh0[i*3 + c] = grad * SH_C0;

        if (v_shN != nullptr) {
            float* out = v_shN + i*45 + c*15;
            for (int k = 0; k < 15; k++) out[k] = 0.f;

            if (sh_active >= 1) {
                out[0] = grad * (SH_C1 * (-y));
                out[1] = grad * (SH_C1 * z);
                out[2] = grad * (SH_C1 * (-x));
            }
            if (sh_active >= 2) {
                out[3] = grad * (SH_C2_0 * (x*y));
                out[4] = grad * (SH_C2_1 * (y*z));
                out[5] = grad * (SH_C2_2 * (2.f*zz - xx - yy));
                out[6] = grad * (SH_C2_3 * (x*z));
                out[7] = grad * (SH_C2_4 * (xx - yy));
            }
            if (sh_active >= 3) {
                out[8]  = grad * (SH_C3_0 * y * (3.f*xx - yy));
                out[9]  = grad * (SH_C3_1 * x * y * z);
                out[10] = grad * (SH_C3_2 * y * (4.f*zz - xx - yy));
                out[11] = grad * (SH_C3_3 * z * (2.f*zz - 3.f*xx - 3.f*yy));
                out[12] = grad * (SH_C3_4 * x * (4.f*zz - xx - yy));
                out[13] = grad * (SH_C3_5 * z * (xx - yy));
                out[14] = grad * (SH_C3_6 * x * (xx - 3.f*yy));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Camera helpers
// ---------------------------------------------------------------------------

// Build a COLMAP-convention camera-to-world matrix from eye, target, world_up.
//
// COLMAP camera axes: X = right, Y = down, Z = forward (into scene).
// c2w columns: [right | down | fwd | eye]
//
// world_up should point away from the ground in world space.
// Tip: derive it from existing cameras as average of -(c2w column 1).
static Mat4 look_at(const float eye[3], const float target[3], const float world_up[3]) {
    // Camera +Z = forward = normalize(target - eye)
    float fwd[3] = { target[0]-eye[0], target[1]-eye[1], target[2]-eye[2] };
    float len = sqrtf(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]);
    fwd[0] /= len;  fwd[1] /= len;  fwd[2] /= len;

    // Camera +X = right = normalize(fwd × world_up)
    float right[3] = {
        fwd[1]*world_up[2] - fwd[2]*world_up[1],
        fwd[2]*world_up[0] - fwd[0]*world_up[2],
        fwd[0]*world_up[1] - fwd[1]*world_up[0]
    };
    len = sqrtf(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    right[0] /= len;  right[1] /= len;  right[2] /= len;

    // Camera +Y = down = fwd × right  (already unit length)
    float down[3] = {
        fwd[1]*right[2] - fwd[2]*right[1],
        fwd[2]*right[0] - fwd[0]*right[2],
        fwd[0]*right[1] - fwd[1]*right[0]
    };

    // Row-major c2w: columns are [right, down, fwd, eye]
    Mat4 c2w{};
    c2w[0][0]=right[0]; c2w[0][1]=down[0]; c2w[0][2]=fwd[0]; c2w[0][3]=eye[0];
    c2w[1][0]=right[1]; c2w[1][1]=down[1]; c2w[1][2]=fwd[1]; c2w[1][3]=eye[1];
    c2w[2][0]=right[2]; c2w[2][1]=down[2]; c2w[2][2]=fwd[2]; c2w[2][3]=eye[2];
    c2w[3][3] = 1.f;
    return c2w;
}

// Build world-to-camera (row-major 4x4) from the camera-to-world matrix.
//
// projection_2dgs.cu uses row-vector convention:
//   p_cam[j] = p_world[0]*viewmat[0*4+j]
//            + p_world[1]*viewmat[1*4+j]
//            + p_world[2]*viewmat[2*4+j]
//            + viewmat[3*4+j]
//
// For a conventional column-vector c2w matrix p_world = R_c2w*p_cam + C,
// the equivalent row-vector w2c upload is:
//   upper-left = R_w2c^T = R_c2w
//   bottom row = t_w2c^T = (-R_c2w^T * C)^T
static void c2w_to_w2c(const Mat4& c2w, float out[16]) {
    float C[3] = { c2w[0][3], c2w[1][3], c2w[2][3] };
    float t[3] = {0.f, 0.f, 0.f};
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
            t[j] -= c2w[i][j] * C[i];  // -R_c2w^T * C

    out[0]  = c2w[0][0]; out[1]  = c2w[0][1]; out[2]  = c2w[0][2]; out[3]  = 0.f;
    out[4]  = c2w[1][0]; out[5]  = c2w[1][1]; out[6]  = c2w[1][2]; out[7]  = 0.f;
    out[8]  = c2w[2][0]; out[9]  = c2w[2][1]; out[10] = c2w[2][2]; out[11] = 0.f;
    out[12] = t[0];       out[13] = t[1];       out[14] = t[2];       out[15] = 1.f;
}

struct RenderCamera {
    uint32_t W, H;
    float fx, fy;
    float cx, cy;
};

static RenderCamera scaled_camera(const Intrinsics& K, float scale) {
    RenderCamera out{};
    out.W  = std::max(1u, (uint32_t)std::lround(K.width  * scale));
    out.H  = std::max(1u, (uint32_t)std::lround(K.height * scale));
    out.fx = K.fx * scale;
    out.fy = K.fy * scale;
    out.cx = K.cx * scale;
    out.cy = K.cy * scale;
    return out;
}

static bool load_render_camera_file(
    const std::string& path,
    RenderCamera& rc,
    Mat4& c2w
) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Error: could not open camera spec file %s\n", path.c_str());
        return false;
    }

    double vals[22];
    for (double& v : vals) {
        if (!(f >> v)) {
            fprintf(stderr, "Error: camera spec %s must contain 22 numeric values\n", path.c_str());
            return false;
        }
    }

    rc.W  = std::max(1u, (uint32_t)std::lround(vals[0]));
    rc.H  = std::max(1u, (uint32_t)std::lround(vals[1]));
    rc.fx = (float)vals[2];
    rc.fy = (float)vals[3];
    rc.cx = (float)vals[4];
    rc.cy = (float)vals[5];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            c2w[r][c] = (float)vals[6 + r * 4 + c];
    return true;
}

static bool parse_render_request_line(
    const std::string& line,
    RenderCamera& rc,
    Mat4& c2w
) {
    std::istringstream ss(line);
    double vals[22];
    for (double& v : vals) {
        if (!(ss >> v))
            return false;
    }
    rc.W  = std::max(1u, (uint32_t)std::lround(vals[0]));
    rc.H  = std::max(1u, (uint32_t)std::lround(vals[1]));
    rc.fx = (float)vals[2];
    rc.fy = (float)vals[3];
    rc.cx = (float)vals[4];
    rc.cy = (float)vals[5];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            c2w[r][c] = (float)vals[6 + r * 4 + c];
    return true;
}

static bool estimate_focus_point_from_camera(const ColmapScene& scene, int cam_idx, float focus[3]) {
    if (cam_idx < 0 || cam_idx >= (int)scene.cameras.size() || scene.points3D.empty())
        return false;

    const CameraInfo& cam = scene.cameras[cam_idx];
    const float px = cam.camtoworld[0][3];
    const float py = cam.camtoworld[1][3];
    const float pz = cam.camtoworld[2][3];

    float dx = cam.camtoworld[0][2];
    float dy = cam.camtoworld[1][2];
    float dz = cam.camtoworld[2][2];
    float dl = sqrtf(dx*dx + dy*dy + dz*dz + 1e-12f);
    dx /= dl;  dy /= dl;  dz /= dl;

    bool found = false;
    float best_score = 1e30f;
    for (const auto& p : scene.points3D) {
        const float vx = p.x - px;
        const float vy = p.y - py;
        const float vz = p.z - pz;
        const float t = vx*dx + vy*dy + vz*dz;
        if (t <= 0.2f) continue;

        const float dist2 = vx*vx + vy*vy + vz*vz;
        const float perp2 = std::max(0.f, dist2 - t*t);
        const float score = perp2 + 0.01f * t * t;
        if (score < best_score) {
            best_score = score;
            focus[0] = p.x;
            focus[1] = p.y;
            focus[2] = p.z;
            found = true;
        }
    }
    return found;
}

// ---------------------------------------------------------------------------
// GPU buffer bundle for forward pass outputs
// ---------------------------------------------------------------------------

struct ForwardBuffers {
    // Projection outputs (one per Gaussian)
    float*   ray_transforms;  // [N, 9]
    float*   means2d;         // [N, 2]
    int32_t* radii;           // [N, 2]
    float*   depths;          // [N]
    float*   normals;         // [N, 3]
    float*   colors;          // [N, 3]  evaluated from sh0

    // Rasterizer outputs (one per pixel at max resolution)
    float*   render_colors;   // [H*W, 3]
    float*   render_alphas;   // [H*W]
    int32_t* last_ids;        // [H*W]
    unsigned char* target_rgb_u8; // [H*W, 3]  staging upload buffer
    float*   target_colors;   // [H*W, 3]  ground-truth RGB
    float*   grad_render;     // [H*W, 3]  dL/d(render_colors)

    // Camera pose (uploaded each frame)
    float*   viewmat;         // [16]  world→camera, row-major

    // Backward buffers (one per Gaussian)
    float*   grad_ray_transforms; // [N, 9]
    float*   grad_opacity;        // [N]
    float*   grad_colors;         // [N, 3]
    float*   grad_means2d;        // [N, 2]
    float*   grad_means2d_abs;    // [N, 2]
    float*   grad_means;          // [N, 3]
    float*   grad_rotation;       // [N, 4]
    float*   grad_scaling;        // [N, 3]
    float*   grad_sh0;            // [N, 3]
    float*   grad_shN;            // [N, K]

    int N_cap;    // current N capacity
    int pix_cap;  // current pixel capacity
    int shN_numel;
};

static ForwardBuffers alloc_forward_buffers(int N, int max_pixels, int max_sh_degree) {
    ForwardBuffers b{};
    int shn_dim = sh_coeffs_per_channel(max_sh_degree) * 3;
    CUDA_CHECK(cudaMalloc(&b.ray_transforms, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.means2d,        N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.radii,          N * 2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&b.depths,         N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.normals,        N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.colors,         N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.render_colors,  max_pixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.render_alphas,  max_pixels     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.last_ids,       max_pixels     * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&b.target_rgb_u8,  max_pixels * 3 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&b.target_colors,  max_pixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.grad_render,    max_pixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.viewmat,        16             * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.grad_ray_transforms, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.grad_opacity,        N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.grad_colors,         N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.grad_means2d,        N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.grad_means2d_abs,    N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.grad_means,          N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.grad_rotation,       N * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.grad_scaling,        N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.grad_sh0,            N * 3 * sizeof(float)));
    if (shn_dim > 0)
        CUDA_CHECK(cudaMalloc(&b.grad_shN, N * shn_dim * sizeof(float)));
    b.N_cap   = N;
    b.pix_cap = max_pixels;
    b.shN_numel = N * shn_dim;
    return b;
}

static void free_forward_buffers(ForwardBuffers& b) {
    cudaFree(b.ray_transforms); cudaFree(b.means2d); cudaFree(b.radii);
    cudaFree(b.depths);         cudaFree(b.normals);  cudaFree(b.colors);
    cudaFree(b.render_colors);  cudaFree(b.render_alphas); cudaFree(b.last_ids);
    cudaFree(b.target_rgb_u8);  cudaFree(b.target_colors);  cudaFree(b.grad_render);
    cudaFree(b.viewmat);
    cudaFree(b.grad_ray_transforms); cudaFree(b.grad_opacity); cudaFree(b.grad_colors);
    cudaFree(b.grad_means2d); cudaFree(b.grad_means2d_abs);
    cudaFree(b.grad_means); cudaFree(b.grad_rotation); cudaFree(b.grad_scaling);
    cudaFree(b.grad_sh0); cudaFree(b.grad_shN);
    b = {};
}

static void ensure_forward_buffers(ForwardBuffers& b, int N, int max_pixels, int max_sh_degree) {
    int shn_dim = sh_coeffs_per_channel(max_sh_degree) * 3;
    bool sh_changed = (b.N_cap > 0) && (b.shN_numel != b.N_cap * shn_dim);
    if (b.N_cap >= N && b.pix_cap >= max_pixels && !sh_changed) return;
    free_forward_buffers(b);
    b = alloc_forward_buffers(N, max_pixels, max_sh_degree);
}

__global__ void u8_to_float_kernel(
    const unsigned char* __restrict__ in,
    float* __restrict__ out,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * (1.0f / 255.0f);
}

static void upload_training_image(
    const TrainingImageSet& images,
    int cam_idx,
    const CameraInfo& cam,
    ForwardBuffers& fwd
) {
    const int W = (int)cam.K.width;
    const int H = (int)cam.K.height;
    const int numel = W * H * 3;
    const size_t num_bytes = (size_t)numel * sizeof(unsigned char);

    const unsigned char* src = nullptr;
    unsigned char* loaded = nullptr;
    if (images.cached) {
        const TrainingImageEntry& entry = images.entries[cam_idx];
        src = entry.rgb_u8.data();
    } else {
        loaded = load_rgb_image_u8(cam.image_path, W, H);
        src = loaded;
    }

    CUDA_CHECK(cudaMemcpy(fwd.target_rgb_u8, src, num_bytes, cudaMemcpyHostToDevice));
    if (loaded) stbi_image_free(loaded);

    int blocks = (numel + 255) / 256;
    u8_to_float_kernel<<<blocks, 256>>>(fwd.target_rgb_u8, fwd.target_colors, numel);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Densification  →  kernels/densify.cuh
// ---------------------------------------------------------------------------

#include "kernels/densify.cuh"

// ---------------------------------------------------------------------------
// Single-frame render + save
// ---------------------------------------------------------------------------
//
// Renders one camera view into a PNG file and returns.
// Used by --render to inspect a trained PLY checkpoint, and by the viewer bridge.

static bool render_camera_to_file(
    const std::string& out_path,
    SplatData& splats,
    const RenderCamera& rc,
    const Mat4& c2w,
    bool verbose
) {
    const uint32_t W = rc.W;
    const uint32_t H = rc.H;
    const int N = splats.N();

    if (verbose) {
        printf("\nRendering custom camera (%ux%u, fx=%.1f fy=%.1f)  N=%d Gaussians ...\n",
               W, H, rc.fx, rc.fy, N);
    }

    ForwardBuffers fwd = alloc_forward_buffers(N, W * H, splats.max_sh_degree());

    float h_w2c[16];
    c2w_to_w2c(c2w, h_w2c);
    CUDA_CHECK(cudaMemcpy(fwd.viewmat, h_w2c, 16*sizeof(float), cudaMemcpyHostToDevice));

    float cam_px = c2w[0][3];
    float cam_py = c2w[1][3];
    float cam_pz = c2w[2][3];

    {
        int blocks = (N + 255) / 256;
        sh_eval_kernel<<<blocks, 256>>>(
            splats.means(), splats.sh0(), splats.shN(),
            cam_px, cam_py, cam_pz,
            fwd.colors, splats.active_sh_degree(), N);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        int blocks = (N + 255) / 256;
        projection_2dgs_kernel<<<blocks, 256>>>(
            splats.means(), splats.rotation(), splats.scaling(), fwd.viewmat,
            rc.fx, rc.fy, rc.cx, rc.cy,
            /*near_plane=*/0.2f, (int)W, (int)H,
            fwd.ray_transforms, fwd.means2d, fwd.radii,
            fwd.depths, fwd.normals, N
        );
        CUDA_CHECK(cudaGetLastError());
    }

    if (verbose) {
        std::vector<int32_t> h_radii(N * 2);
        CUDA_CHECK(cudaMemcpy(h_radii.data(), fwd.radii,
                              N * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost));
        int visible = 0;
        long long sum_r = 0;
        for (int i = 0; i < N; i++) {
            int rx = h_radii[i * 2], ry = h_radii[i * 2 + 1];
            if (rx > 0 && ry > 0) {
                visible++;
                sum_r += rx + ry;
            }
        }
        printf("  visible=%d/%d  avg_radius=%.1f px\n",
               visible, N, visible ? (double)sum_r / (2 * visible) : 0.0);
    }

    TileIntersectBuffers tile_buf = launch_tile_intersect(
        fwd.means2d, fwd.radii, fwd.depths, N, W, H, TILE_SIZE);
    if (verbose)
        printf("  n_isects=%d\n", tile_buf.n_isects);

    CUDA_CHECK(cudaMemset(fwd.render_colors, 0, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(fwd.render_alphas, 0, W * H * sizeof(float)));
    launch_rasterize_fwd(
        fwd.means2d, fwd.ray_transforms, splats.opacity(), fwd.colors,
        tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
        W, H,
        fwd.render_colors, fwd.render_alphas, fwd.last_ids
    );

    bool ok = save_device_rgb_png(fwd.render_colors, W, H, out_path);

    free_tile_intersect_buffers(tile_buf);
    free_forward_buffers(fwd);

    if (verbose) {
        if (ok)
            printf("Saved %s  (%ux%u)\n", out_path.c_str(), W, H);
        else
            fprintf(stderr, "Error: could not write PNG to %s\n", out_path.c_str());
    }
    return ok;
}

static void render_to_file(const Config& cfg, SplatData& splats,
                           const ColmapScene& scene) {
    if (!cfg.camera_file.empty()) {
        RenderCamera rc{};
        Mat4 c2w{};
        if (!load_render_camera_file(cfg.camera_file, rc, c2w))
            return;
        render_camera_to_file(cfg.render_out, splats, rc, c2w, /*verbose=*/true);
        return;
    }

    if ((int)scene.cameras.size() <= cfg.render_cam) {
        fprintf(stderr, "Error: --cam %d out of range (scene has %zu cameras)\n",
                cfg.render_cam, scene.cameras.size());
        return;
    }

    const CameraInfo& cam = scene.cameras[cfg.render_cam];
    const RenderCamera rc = scaled_camera(cam.K, cfg.render_scale);
    render_camera_to_file(cfg.render_out, splats, rc, cam.camtoworld, /*verbose=*/true);
}

static void serve_render_loop(const Config& cfg, SplatData& splats) {
    std::filesystem::file_time_type ply_mtime{};
    bool have_mtime = false;
    if (!cfg.ply_path.empty()) {
        std::error_code ec;
        ply_mtime = std::filesystem::last_write_time(cfg.ply_path, ec);
        have_mtime = !ec;
    }

    printf("READY\n");
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit")
            break;
        if (line.empty())
            continue;

        RenderCamera rc{};
        Mat4 c2w{};
        if (!parse_render_request_line(line, rc, c2w)) {
            printf("ERR parse\n");
            continue;
        }

        if (!cfg.ply_path.empty()) {
            std::error_code ec;
            auto new_mtime = std::filesystem::last_write_time(cfg.ply_path, ec);
            if (!ec && (!have_mtime || new_mtime != ply_mtime)) {
                try {
                    splats.init_from_ply(cfg.ply_path, cfg.sh_degree);
                    ply_mtime = new_mtime;
                    have_mtime = true;
                } catch (const std::exception& e) {
                    fprintf(stderr, "Error reloading PLY in serve mode: %s\n", e.what());
                    printf("ERR reload\n");
                    continue;
                }
            }
        }

        bool ok = render_camera_to_file(cfg.serve_output, splats, rc, c2w, /*verbose=*/false);
        printf(ok ? "OK\n" : "ERR render\n");
    }
}

// ---------------------------------------------------------------------------
// Camera orbit image sequence
// ---------------------------------------------------------------------------
//
// Orbits the camera around the scene center computed from the COLMAP cameras.
// The orbit plane is derived automatically:
//   - center    = mean of all camera positions
//   - world_up  = mean of all camera "up" directions (= -c2w column 1)
//   - radius    = mean camera-to-center distance
//
// Frames are saved as  <out_prefix>_0000.png, <out_prefix>_0001.png, …
// The intrinsics of the first camera are used for all frames.

// Solve a 3x3 linear system A*x = b using Gaussian elimination with partial
// pivoting.  A is stored row-major (A[row*3+col]).  Returns false if singular.
static bool solve3x3(float A[9], float b[3], float x[3]) {
    // Augmented matrix [A | b]
    float M[3][4];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) M[r][c] = A[r*3+c];
        M[r][3] = b[r];
    }
    for (int col = 0; col < 3; col++) {
        // Partial pivot
        int pivot = col;
        for (int r = col+1; r < 3; r++)
            if (fabsf(M[r][col]) > fabsf(M[pivot][col])) pivot = r;
        if (fabsf(M[pivot][col]) < 1e-10f) return false;
        for (int c = 0; c < 4; c++) std::swap(M[col][c], M[pivot][c]);
        float inv = 1.f / M[col][col];
        for (int r = col+1; r < 3; r++) {
            float f = M[r][col] * inv;
            for (int c = col; c < 4; c++) M[r][c] -= f * M[col][c];
        }
    }
    // Back-substitution
    for (int r = 2; r >= 0; r--) {
        x[r] = M[r][3];
        for (int c = r+1; c < 3; c++) x[r] -= M[r][c] * x[c];
        x[r] /= M[r][r];
    }
    return true;
}

static void render_orbit(const Config& cfg, SplatData& splats,
                         const ColmapScene& scene) {
    const int N_frames = cfg.orbit_frames;
    const int N = splats.N();
    const int nc = (int)scene.cameras.size();

    if (nc == 0) { fprintf(stderr, "No cameras in scene.\n"); return; }

    // ── Scene center: least-squares convergence point of camera forward rays ─
    //
    // For each camera ray (position p_i, unit forward d_i), the squared
    // distance from point x to the ray is ||(I - d_i d_i^T)(x - p_i)||².
    // Minimizing over x gives:  A x = b
    //   A = Σ (I - d_i d_i^T)
    //   b = Σ (I - d_i d_i^T) p_i
    //
    // This is better than the mean camera position when cameras aren't
    // arranged symmetrically — it finds what they're all looking at.
    float A[9] = {};
    float b[3] = {};
    float upx = 0, upy = 0, upz = 0;

    for (const auto& cam : scene.cameras) {
        float px = cam.camtoworld[0][3];
        float py = cam.camtoworld[1][3];
        float pz = cam.camtoworld[2][3];

        // Forward direction = column 2 of c2w
        float dx = cam.camtoworld[0][2];
        float dy = cam.camtoworld[1][2];
        float dz = cam.camtoworld[2][2];
        float dl = sqrtf(dx*dx + dy*dy + dz*dz + 1e-12f);
        dx /= dl;  dy /= dl;  dz /= dl;

        // M = I - d d^T  (row-major)
        float M[9] = {
            1.f-dx*dx, -dx*dy,    -dx*dz,
            -dy*dx,    1.f-dy*dy, -dy*dz,
            -dz*dx,    -dz*dy,    1.f-dz*dz
        };
        for (int j = 0; j < 9; j++) A[j] += M[j];
        b[0] += M[0]*px + M[1]*py + M[2]*pz;
        b[1] += M[3]*px + M[4]*py + M[5]*pz;
        b[2] += M[6]*px + M[7]*py + M[8]*pz;

        // World up = -(c2w column 1)
        upx += -cam.camtoworld[0][1];
        upy += -cam.camtoworld[1][1];
        upz += -cam.camtoworld[2][1];
    }

    float scene_center[3];
    if (!solve3x3(A, b, scene_center)) {
        // Degenerate: fall back to mean camera position
        for (int j = 0; j < 9; j++) A[j] /= nc;
        scene_center[0] = b[0]/nc;
        scene_center[1] = b[1]/nc;
        scene_center[2] = b[2]/nc;
    }

    float cx = scene_center[0], cy = scene_center[1], cz = scene_center[2];

    float up_len = sqrtf(upx*upx + upy*upy + upz*upz);
    upx /= up_len;  upy /= up_len;  upz /= up_len;

    int orbit_start_cam = cfg.orbit_focus_cam;
    if (orbit_start_cam < 0 || orbit_start_cam >= nc)
        orbit_start_cam = -1;

    bool focus_overridden = false;
    if (orbit_start_cam >= 0) {
        float focus[3];
        if (estimate_focus_point_from_camera(scene, orbit_start_cam, focus)) {
            cx = focus[0];
            cy = focus[1];
            cz = focus[2];
            focus_overridden = true;
        } else {
            orbit_start_cam = -1;
        }
    }

    // ── Orbit radius and height offset (along world-up) ─────────────────────
    // Decompose each camera-to-center vector into axial (world_up) and radial
    // components.  The orbit matches the real cameras' average standoff.
    float orbit_r      = 0;  // mean radial distance (in the orbit plane)
    float height_off   = 0;  // mean offset along world_up

    for (const auto& cam : scene.cameras) {
        float dx = cam.camtoworld[0][3] - cx;
        float dy = cam.camtoworld[1][3] - cy;
        float dz = cam.camtoworld[2][3] - cz;
        float axial  = dx*upx + dy*upy + dz*upz;
        float rad_sq = dx*dx + dy*dy + dz*dz - axial*axial;
        orbit_r    += sqrtf(rad_sq > 0.f ? rad_sq : 0.f);
        height_off += axial;
    }
    orbit_r    /= nc;
    height_off /= nc;
    orbit_r    *= cfg.orbit_radius_scale;
    height_off *= cfg.orbit_height_scale;

    // ── Two orthonormal vectors spanning the orbit plane ────────────────────
    float ax = (fabsf(upx) < 0.9f) ? 1.f : 0.f;
    float ay = (fabsf(upx) < 0.9f) ? 0.f : 1.f;
    float az = 0.f;
    float dot = ax*upx + ay*upy + az*upz;
    float a1x = ax - dot*upx,  a1y = ay - dot*upy,  a1z = az - dot*upz;
    float a1l = sqrtf(a1x*a1x + a1y*a1y + a1z*a1z);
    a1x /= a1l;  a1y /= a1l;  a1z /= a1l;
    // axis2 = world_up × axis1
    float a2x = upy*a1z - upz*a1y;
    float a2y = upz*a1x - upx*a1z;
    float a2z = upx*a1y - upy*a1x;

    // ── Use first camera's intrinsics for all frames ─────────────────────────
    const CameraInfo& ref = scene.cameras[0];
    const RenderCamera rc = scaled_camera(ref.K, cfg.render_scale);
    const uint32_t W = rc.W;
    const uint32_t H = rc.H;

    printf("\n--- Orbit render ---\n");
    if (focus_overridden) {
        printf("Orbit focus : camera %d ray-picked point (%.3f, %.3f, %.3f)\n",
               orbit_start_cam, cx, cy, cz);
    } else {
        printf("Scene center (ray convergence): (%.3f, %.3f, %.3f)\n", cx, cy, cz);
    }
    printf("World up    : (%.3f, %.3f, %.3f)\n", upx, upy, upz);
    printf("Orbit radius: %.3f   height offset: %.3f\n", orbit_r, height_off);
    printf("Orbit scale : radius x %.3f   height x %.3f\n",
           cfg.orbit_radius_scale, cfg.orbit_height_scale);
    printf("Frames      : %d   Resolution: %ux%u (scale=%.3f)\n\n", N_frames, W, H, cfg.render_scale);

    ForwardBuffers fwd = alloc_forward_buffers(N, W * H, splats.max_sh_degree());

    float theta0 = 0.f;
    if (orbit_start_cam >= 0) {
        const CameraInfo& start_cam = scene.cameras[orbit_start_cam];
        const float vx = start_cam.camtoworld[0][3] - (cx + height_off*upx);
        const float vy = start_cam.camtoworld[1][3] - (cy + height_off*upy);
        const float vz = start_cam.camtoworld[2][3] - (cz + height_off*upz);
        theta0 = atan2f(vx*a2x + vy*a2y + vz*a2z,
                        vx*a1x + vy*a1y + vz*a1z);
        printf("Orbit start : aligned to camera %d (theta0=%.3f rad)\n\n",
               orbit_start_cam, theta0);
    }

    for (int frame = 0; frame < N_frames; frame++) {
        float theta  = theta0 + 2.f * 3.14159265358979f * frame / N_frames;
        float cos_t  = cosf(theta);
        float sin_t  = sinf(theta);

        float eye[3] = {
            cx + orbit_r * (cos_t*a1x + sin_t*a2x) + height_off*upx,
            cy + orbit_r * (cos_t*a1y + sin_t*a2y) + height_off*upy,
            cz + orbit_r * (cos_t*a1z + sin_t*a2z) + height_off*upz
        };
        float target[3] = { cx, cy, cz };
        float world_up[3] = { upx, upy, upz };

        Mat4 c2w = look_at(eye, target, world_up);

        // Upload viewmat
        float h_w2c[16];
        c2w_to_w2c(c2w, h_w2c);
        CUDA_CHECK(cudaMemcpy(fwd.viewmat, h_w2c, 16*sizeof(float), cudaMemcpyHostToDevice));

        // Full SH evaluation with camera position from c2w
        {
            int blocks = (N + 255) / 256;
            sh_eval_kernel<<<blocks, 256>>>(
                splats.means(), splats.sh0(), splats.shN(),
                c2w[0][3], c2w[1][3], c2w[2][3],
                fwd.colors, splats.active_sh_degree(), N);
            CUDA_CHECK(cudaGetLastError());
        }

        // Projection
        {
            int blocks = (N + 255) / 256;
            projection_2dgs_kernel<<<blocks, 256>>>(
                splats.means(), splats.rotation(), splats.scaling(), fwd.viewmat,
                rc.fx, rc.fy, rc.cx, rc.cy,
                /*near_plane=*/0.2f, (int)W, (int)H,
                fwd.ray_transforms, fwd.means2d, fwd.radii,
                fwd.depths, fwd.normals, N);
            CUDA_CHECK(cudaGetLastError());
        }

        // Tile intersect + rasterize
        TileIntersectBuffers tile_buf = launch_tile_intersect(
            fwd.means2d, fwd.radii, fwd.depths, N, W, H, TILE_SIZE);

        CUDA_CHECK(cudaMemset(fwd.render_colors, 0, W*H*3*sizeof(float)));
        CUDA_CHECK(cudaMemset(fwd.render_alphas, 0, W*H*sizeof(float)));
        launch_rasterize_fwd(
            fwd.means2d, fwd.ray_transforms, splats.opacity(), fwd.colors,
            tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
            W, H, fwd.render_colors, fwd.render_alphas, fwd.last_ids);
        CUDA_CHECK(cudaDeviceSynchronize());

        free_tile_intersect_buffers(tile_buf);

        // Download and save
        std::vector<float> h_render(W * H * 3);
        CUDA_CHECK(cudaMemcpy(h_render.data(), fwd.render_colors,
                              W*H*3*sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<uint8_t> pixels(W * H * 3);
        for (size_t k = 0; k < pixels.size(); k++)
            pixels[k] = (uint8_t)(std::max(0.f, std::min(1.f, h_render[k])) * 255.f + 0.5f);

        char fname[512];
        snprintf(fname, sizeof(fname), "%s_%04d.png", cfg.orbit_out.c_str(), frame);
        int ok = stbi_write_png(fname, (int)W, (int)H, 3, pixels.data(), (int)W*3);
        printf("  frame %3d/%d  n_isects=%-8d  → %s%s\n",
               frame+1, N_frames, tile_buf.n_isects, fname,
               ok ? "" : "  [WRITE FAILED]");
    }

    free_forward_buffers(fwd);
    printf("\nOrbit sequence done.\n");
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

static void train(
    const Config& cfg,
    SplatData& splats,
    const ColmapScene& scene,
    const TrainingImageSet& training_images
) {
    const int sh_increase_every = 1000;
    const float scene_scale = compute_scene_scale(scene);

    // Determine max image size across all cameras
    uint32_t max_W = 0, max_H = 0;
    for (const auto& cam : scene.cameras) {
        max_W = std::max(max_W, cam.K.width);
        max_H = std::max(max_H, cam.K.height);
    }

    printf("\n--- Training ---\n");
    printf("Iterations  : %d\n", cfg.iters);
    printf("SH degree   : 0 → %d  (one step every %d iters)\n",
           cfg.sh_degree, sh_increase_every);
    printf("Gaussians   : %d\n", splats.N());
    printf("Scene scale : %.4f\n", scene_scale);
    printf("Max image   : %ux%u\n", max_W, max_H);
    printf("Image cache : %s\n", training_images.cached ? "enabled" : "disabled");
    printf("Log every   : %d\n", cfg.log_every);
    if (cfg.preview_every > 0)
        printf("Previews    : every %d iters → %s_XXXXXX.png\n",
               cfg.preview_every, cfg.preview_out.c_str());
    if (!cfg.save_ply.empty()) {
        if (cfg.save_ply_every > 0)
            printf("Checkpoint  : every %d iters + final → %s\n",
                   cfg.save_ply_every, cfg.save_ply.c_str());
        else
            printf("Checkpoint  : final only → %s\n", cfg.save_ply.c_str());
    }
    if (cfg.densify_every > 0)
        printf("Densify     : every %d iters in [%d, %d]\n",
               cfg.densify_every, cfg.densify_start, cfg.densify_stop);
    if (cfg.opacity_reset_every > 0)
        printf("Opacity rst : every %d iters  prune_alpha=%.3f  grow_scale=%.4f*scene  prune_scale=%.4f*scene  grad_thresh=%g\n",
               cfg.opacity_reset_every, cfg.densify_prune_alpha,
               cfg.densify_grow_scale3d, cfg.densify_prune_scale3d,
               cfg.densify_grad_thresh);

    ForwardBuffers fwd = alloc_forward_buffers(splats.N(), max_W * max_H, splats.max_sh_degree());
    LossWorkspace loss_ws{};
    ensure_loss_workspace(loss_ws, (int)max_W * (int)max_H * 3);
    DensifyState densify{};
    alloc_densify_state(densify, splats.N());
    SplatAdam optimizer(splats);
    AdamConfig opt_cfg;
    printf("Optimizer   : Adam  lr_xyz=%.2g  lr_opacity=%.2g  lr_sh0=%.2g\n\n",
           opt_cfg.lr_means, opt_cfg.lr_opacity, opt_cfg.lr_sh0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> cam_dist(0, (int)scene.cameras.size() - 1);

    auto train_start = std::chrono::steady_clock::now();

    for (int iter = 1; iter <= cfg.iters; iter++) {

        // ── SH curriculum ────────────────────────────────────────────────────
        if (iter % sh_increase_every == 0)
            splats.increment_sh_degree();

        // ── Pick a training camera ────────────────────────────────────────────
        const int N = splats.N();
        int cam_idx = cam_dist(rng);
        const CameraInfo& cam = scene.cameras[cam_idx];
        const uint32_t W = cam.K.width;
        const uint32_t H = cam.K.height;

        upload_training_image(training_images, cam_idx, cam, fwd);

        // ── Build world-to-camera and upload ─────────────────────────────────
        float h_w2c[16];
        c2w_to_w2c(cam.camtoworld, h_w2c);
        CUDA_CHECK(cudaMemcpy(fwd.viewmat, h_w2c, 16*sizeof(float), cudaMemcpyHostToDevice));

        // ── Full SH evaluation: view-dependent color per Gaussian ─────────────
        {
            float cam_px = cam.camtoworld[0][3];
            float cam_py = cam.camtoworld[1][3];
            float cam_pz = cam.camtoworld[2][3];
            int blocks = (N + 255) / 256;
            sh_eval_kernel<<<blocks, 256>>>(
                splats.means(), splats.sh0(), splats.shN(),
                cam_px, cam_py, cam_pz,
                fwd.colors, splats.active_sh_degree(), N);
            CUDA_CHECK(cudaGetLastError());
        }

        // ── Step 1: Projection ────────────────────────────────────────────────
        {
            int blocks = (N + 255) / 256;
            projection_2dgs_kernel<<<blocks, 256>>>(
                splats.means(), splats.rotation(), splats.scaling(), fwd.viewmat,
                cam.K.fx, cam.K.fy, cam.K.cx, cam.K.cy,
                /*near_plane=*/0.2f, (int)W, (int)H,
                fwd.ray_transforms, fwd.means2d, fwd.radii,
                fwd.depths, fwd.normals, N
            );
            CUDA_CHECK(cudaGetLastError());
        }

        if (iter == 1) {
            std::vector<int32_t> h_radii(N * 2);
            CUDA_CHECK(cudaMemcpy(h_radii.data(), fwd.radii,
                                  N * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost));
            int visible = 0;
            int max_rx = 0, max_ry = 0;
            long long sum_rx = 0, sum_ry = 0;
            for (int i = 0; i < N; i++) {
                int rx = h_radii[i*2], ry = h_radii[i*2+1];
                if (rx <= 0 || ry <= 0) continue;
                visible++;
                max_rx = std::max(max_rx, rx);
                max_ry = std::max(max_ry, ry);
                sum_rx += rx;
                sum_ry += ry;
            }
            printf("projection: visible=%d/%d avg_radius=(%.1f, %.1f) max_radius=(%d, %d)\n",
                   visible, N,
                   visible ? (double)sum_rx / visible : 0.0,
                   visible ? (double)sum_ry / visible : 0.0,
                   max_rx, max_ry);
        }

        // ── Step 2: Tile intersection ─────────────────────────────────────────
        TileIntersectBuffers tile_buf = launch_tile_intersect(
            fwd.means2d, fwd.radii, fwd.depths, N, W, H, TILE_SIZE);

        // ── Step 3: Rasterize ─────────────────────────────────────────────────
        CUDA_CHECK(cudaMemset(fwd.render_colors, 0, W*H*3*sizeof(float)));
        CUDA_CHECK(cudaMemset(fwd.render_alphas, 0, W*H*sizeof(float)));

        launch_rasterize_fwd(
            fwd.means2d, fwd.ray_transforms, splats.opacity(), fwd.colors,
            tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
            W, H,
            fwd.render_colors, fwd.render_alphas, fwd.last_ids
        );

        LossResult loss = photometric_loss(
            fwd.render_colors, fwd.target_colors, fwd.grad_render, loss_ws,
            (int)H, (int)W, 0.2f, 3);

        // ── Backward pass ────────────────────────────────────────────────────
        CUDA_CHECK(cudaMemset(fwd.grad_ray_transforms, 0, N*9*sizeof(float)));
        CUDA_CHECK(cudaMemset(fwd.grad_opacity,        0, N*sizeof(float)));
        CUDA_CHECK(cudaMemset(fwd.grad_colors,         0, N*3*sizeof(float)));
        CUDA_CHECK(cudaMemset(fwd.grad_means2d,        0, N*2*sizeof(float)));
        CUDA_CHECK(cudaMemset(fwd.grad_means2d_abs,    0, N*2*sizeof(float)));
        CUDA_CHECK(cudaMemset(fwd.grad_means,          0, N*3*sizeof(float)));
        CUDA_CHECK(cudaMemset(fwd.grad_rotation,       0, N*4*sizeof(float)));
        CUDA_CHECK(cudaMemset(fwd.grad_scaling,        0, N*3*sizeof(float)));
        CUDA_CHECK(cudaMemset(fwd.grad_sh0,            0, N*3*sizeof(float)));
        if (fwd.shN_numel > 0)
            CUDA_CHECK(cudaMemset(fwd.grad_shN, 0, fwd.shN_numel*sizeof(float)));

        launch_rasterize_bwd(
            fwd.means2d, fwd.ray_transforms, splats.opacity(), fwd.colors,
            tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
            fwd.render_alphas, fwd.last_ids, fwd.grad_render,
            W, H,
            fwd.grad_ray_transforms, fwd.grad_opacity, fwd.grad_colors,
            fwd.grad_means2d, fwd.grad_means2d_abs
        );

        launch_projection_2dgs_bwd(
            splats.means(), splats.rotation(), splats.scaling(), fwd.viewmat,
            cam.K.fx, cam.K.fy, cam.K.cx, cam.K.cy,
            fwd.ray_transforms, fwd.radii,
            fwd.grad_ray_transforms, fwd.grad_means2d,
            /*d_v_depths=*/nullptr, /*d_v_normals=*/nullptr,
            fwd.grad_means, fwd.grad_rotation, fwd.grad_scaling,
            N
        );

        {
            float cam_px = cam.camtoworld[0][3];
            float cam_py = cam.camtoworld[1][3];
            float cam_pz = cam.camtoworld[2][3];
            int blocks = (N + 255) / 256;
            sh_backward_kernel<<<blocks, 256>>>(
                splats.means(), splats.sh0(), splats.shN(),
                cam_px, cam_py, cam_pz,
                fwd.grad_colors, fwd.grad_sh0, fwd.grad_shN,
                splats.active_sh_degree(), N
            );
            CUDA_CHECK(cudaGetLastError());
        }

        {
            int blocks = (N + 255) / 256;
            accumulate_grad_means2d_abs_kernel<<<blocks, 256>>>(
                fwd.grad_means2d_abs, densify.grad_accum, N);
            accumulate_visibility_count_kernel<<<blocks, 256>>>(
                fwd.radii, densify.count, N);
            CUDA_CHECK(cudaGetLastError());
            densify.accum_steps++;
        }

        free_tile_intersect_buffers(tile_buf);

        // ── Optimizer step ───────────────────────────────────────────────────
        // Exponential decay for position LR: 1.6e-4 → 1.6e-6 over all iters.
        // Matches gsplat's schedule: large early steps, fine-grained at convergence.
        {
            constexpr float lr_init  = 1.6e-4f;
            constexpr float lr_final = 1.6e-6f;
            float t = (float)(iter - 1) / (float)std::max(cfg.iters - 1, 1);
            opt_cfg.lr_means = lr_init * std::pow(lr_final / lr_init, t);
        }

        SplatGradients grads{
            fwd.grad_means,
            fwd.grad_rotation,
            fwd.grad_scaling,
            fwd.grad_opacity,
            fwd.grad_sh0,
            fwd.grad_shN
        };
        optimizer.step(splats, grads, opt_cfg);

        // ── Optional preview snapshot ────────────────────────────────────────
        if (cfg.preview_every > 0 &&
            (iter % cfg.preview_every == 0 || iter == 1 || iter == cfg.iters)) {
            char fname[1024];
            snprintf(fname, sizeof(fname), "%s_%06d.png", cfg.preview_out.c_str(), iter);
            bool ok = save_device_rgb_png(fwd.render_colors, W, H, fname);
            printf("\npreview: iter=%d cam=%d loss=%.6f → %s%s\n",
                   iter, cam_idx, loss.loss, fname, ok ? "" : "  [WRITE FAILED]");
        }

        // ── Basic densification ──────────────────────────────────────────────
        maybe_densify(cfg, iter, scene_scale, splats, optimizer, fwd, densify, max_W * max_H);
        maybe_reset_opacity(cfg, iter, splats, optimizer);
        maybe_save_ply_checkpoint(cfg, iter, iter == cfg.iters, splats);

        // ── Progress log ─────────────────────────────────────────────────────
        if (iter % cfg.log_every == 0 || iter == 1 || iter == cfg.iters) {
            double elapsed_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - train_start).count();
            print_training_progress(
                iter, cfg.iters,
                splats.active_sh_degree(), splats.max_sh_degree(),
                tile_buf.n_isects,
                loss,
                elapsed_sec,
                iter == cfg.iters
            );
        }
    }

    if (cfg.iters % cfg.log_every != 0) printf("\n");

    free_forward_buffers(fwd);
    free_loss_workspace(loss_ws);
    free_densify_state(densify);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered stdout for debugger

    Config cfg = parse_args(argc, argv);

    ColmapScene scene;
    TrainingImageSet training_images;
    const bool render_custom_camera = !cfg.render_out.empty() && !cfg.camera_file.empty();
    const bool needs_scene =
        (!cfg.render_out.empty() && cfg.camera_file.empty()) ||
        (cfg.orbit_frames > 0) ||
        (cfg.render_out.empty() && cfg.orbit_frames == 0 && !cfg.serve_render);
    const bool needs_training_images =
        cfg.render_out.empty() && cfg.orbit_frames == 0 && !cfg.serve_render;

    if (needs_scene) {
        printf("Loading COLMAP scene from: %s\n", cfg.data_dir.c_str());
        try {
            scene = load_colmap(cfg.data_dir, cfg.images);
        } catch (const std::exception& e) {
            fprintf(stderr, "Error loading scene: %s\n", e.what());
            return 1;
        }
        print_scene_summary(scene);
    }

    if (needs_training_images) {
        const bool cache_training_images = true;
        try {
            training_images = prepare_training_images(scene, cfg.images, cache_training_images);
        } catch (const std::exception& e) {
            fprintf(stderr, "Error preparing images: %s\n", e.what());
            return 1;
        }
    }

    SplatData splats(0, cfg.sh_degree);

    if (!cfg.ply_path.empty()) {
        // ── Load trained checkpoint ───────────────────────────────────────────
        printf("\nLoading PLY checkpoint: %s\n", cfg.ply_path.c_str());
        try {
            splats.init_from_ply(cfg.ply_path, cfg.sh_degree);
        } catch (const std::exception& e) {
            fprintf(stderr, "Error loading PLY: %s\n", e.what());
            return 1;
        }
        splats.print_summary();
    } else {
        // ── Initialize from COLMAP point cloud ────────────────────────────────
        if (!needs_scene) {
            fprintf(stderr, "Error: rendering without --ply requires a loaded scene\n");
            return 1;
        }
        const int M = (int)scene.points3D.size();
        if (M == 0) {
            fprintf(stderr, "Error: no 3D points in scene\n");
            return 1;
        }
        std::vector<float> xyz(M * 3), rgb(M * 3);
        for (int i = 0; i < M; i++) {
            xyz[i*3+0] = scene.points3D[i].x;
            xyz[i*3+1] = scene.points3D[i].y;
            xyz[i*3+2] = scene.points3D[i].z;
            rgb[i*3+0] = scene.point_colors[i].x;
            rgb[i*3+1] = scene.point_colors[i].y;
            rgb[i*3+2] = scene.point_colors[i].z;
        }
        printf("\nInitializing %d Gaussians...\n", M);
        splats.init_from_points(xyz.data(), rgb.data(), M);
        splats.print_summary();
    }

    if (cfg.serve_render) {
        serve_render_loop(cfg, splats);
        return 0;
    }

    if (!cfg.render_out.empty()) {
        if (render_custom_camera) {
            RenderCamera rc{};
            Mat4 c2w{};
            if (!load_render_camera_file(cfg.camera_file, rc, c2w))
                return 1;
            if (!render_camera_to_file(cfg.render_out, splats, rc, c2w, /*verbose=*/true))
                return 1;
        } else {
            render_to_file(cfg, splats, scene);
        }
        return 0;
    }

    if (cfg.orbit_frames > 0) {
        render_orbit(cfg, splats, scene);
        return 0;
    }

    train(cfg, splats, scene, training_images);

    printf("\nDone.\n");
    return 0;
}
