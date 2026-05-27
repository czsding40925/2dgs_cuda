// nexel_color.cu
//
// Per-Gaussian appearance pipeline for the Phase B (Nexels) work. Chains
// the hash-grid query and the MLP forward/backward kernels into a single
// "given means, write RGB colours" entry point, matching the role of the
// existing `sh_eval_kernel` in train.cu so that swapping in nexel-style
// appearance only requires changing two call-sites in the trainer.
//
//   forward:
//      means[N,3]
//          ─ hash_grid_fwd ─→ features[N, D_in=L*F]
//          ─ mlp_fwd       ─→ hidden[N,D_hid], mlp_out[N,D_out=3]
//          ─ +0.5 clamp[0,1] ─→ colors[N,3]
//
//   backward:
//      grad_colors[N,3]
//          ─ ignore clamp at backward (matches sh_backward; gradient
//             flows through saturated outputs as if the clamp were
//             absent — small bias, simple code path)
//          ─ mlp_bwd       ─→ grad_features[N, D_in], grad_W1/W2/b1/b2
//          ─ hash_grid_bwd ─→ grad_grid[L,T,F]
//
// Grad-back-to-means is NOT yet wired (matches the hash_grid.cu MVP
// decision). The Gaussian means continue to receive a geometric
// gradient from the rasterizer; the hash grid + MLP just re-learn at
// whatever positions the means land. We will revisit if PSNR
// significantly trails the reference Nexels numbers.

#pragma once

// In a standalone build (`nvcc kernels/nexel_color.cu`), pull in the
// dependency kernels as headers. When included from another translation
// unit that already provides them (e.g. `gradient_checks.cu`),
// INCLUDED_AS_HEADER is already defined and we skip — both because the
// kernels are already in scope and because doing a local `#undef`
// would clobber the outer scope's flag and accidentally enable our
// own `main()` at the bottom of this file.
#ifndef INCLUDED_AS_HEADER
  #define INCLUDED_AS_HEADER
  #include "hash_grid.cu"
  #include "mlp.cu"
  #undef INCLUDED_AS_HEADER
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#ifndef NEXEL_CUDA_CHECK
#define NEXEL_CUDA_CHECK(call)                                                  \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "[nexel] CUDA error at %s:%d — %s\n",                     \
              __FILE__, __LINE__, cudaGetErrorString(_e));                      \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)
#endif

// ─── AppearanceField: persistent parameters (grid + MLP) ─────────────────────
//
// Owned for the lifetime of training; resized only on initialisation. The
// per-iter workspaces and gradient buffers live in separate structs so
// that densification (which changes N) does not force a re-allocation of
// the parameter tensors.

struct AppearanceField {
    int L      = 0;     // hash levels
    int T      = 0;     // table size per level (power of 2)
    int F      = 0;     // features per level
    int D_in   = 0;     // = L * F
    int D_hid  = 0;
    int D_out  = 3;     // RGB by default; 48 when view_dep (degree-3 SH coefficients)
    float min_res = 0.f;
    float max_res = 0.f;

    // When true, the MLP output is interpreted as packed degree-3 SH
    // coefficients (3 DC + 45 directional) and run through
    // nexel_sh_eval_packed_fwd before being handed to the rasterizer.
    // Closes most of the SH-vs-Nexels quality gap without rasterizer
    // surgery. Set by AppearanceField::allocate via D_out_ == 48.
    bool view_dep = false;

    float* grid          = nullptr;   // [L * T * F]
    float* W1            = nullptr;   // [D_hid, D_in]
    float* b1            = nullptr;   // [D_hid]
    float* W2            = nullptr;   // [D_out, D_hid]
    float* b2            = nullptr;   // [D_out]
    float* level_scales  = nullptr;   // [L]

    // ── Construction ─────────────────────────────────────────────────────
    void allocate(int L_, int T_, int F_, int D_hid_, int D_out_,
                  float min_res_, float max_res_) {
        free();
        L = L_; T = T_; F = F_;
        D_in = L * F;
        D_hid = D_hid_;
        D_out = D_out_;
        view_dep = (D_out_ == 48);
        min_res = min_res_;
        max_res = max_res_;

        NEXEL_CUDA_CHECK(cudaMalloc(&grid,         (size_t)L * T * F * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&W1,           (size_t)D_hid * D_in * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&b1,           (size_t)D_hid * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&W2,           (size_t)D_out * D_hid * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&b2,           (size_t)D_out * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&level_scales, (size_t)L * sizeof(float)));

        // Pre-compute and upload level scales — never change once set.
        std::vector<float> h_scales = hash_grid_build_level_scales(L, min_res, max_res);
        NEXEL_CUDA_CHECK(cudaMemcpy(level_scales, h_scales.data(),
                                    (size_t)L * sizeof(float), cudaMemcpyHostToDevice));
    }

    // ── Random initialisation (matches Nexels defaults) ──────────────────
    //
    // grid:    small constant 0.01 — Nexels initialises the table this way
    //          so the appearance field starts roughly featureless.
    // W1, W2:  uniform(-1, +1) / sqrt(fan_in)  (Xavier-ish, weight_factor=1)
    // b1, b2:  zero  (Nexels keeps biases at zero)
    void init_random(uint64_t seed = 42) {
        if (grid == nullptr) {
            fprintf(stderr, "AppearanceField::init_random called before allocate()\n");
            std::exit(1);
        }

        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> unif(-1.f, 1.f);

        // grid ← 0.01 (constant)
        std::vector<float> h_grid((size_t)L * T * F, 0.01f);
        NEXEL_CUDA_CHECK(cudaMemcpy(grid, h_grid.data(),
                                    h_grid.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));

        // W1 ← uniform / sqrt(D_in)
        std::vector<float> h_W1((size_t)D_hid * D_in);
        const float w1_scale = 1.f / std::sqrt((float)D_in);
        for (auto& w : h_W1) w = unif(rng) * w1_scale;
        NEXEL_CUDA_CHECK(cudaMemcpy(W1, h_W1.data(),
                                    h_W1.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));

        // W2 ← uniform / sqrt(D_hid).
        // In view_dep mode the MLP outputs SH coefficients [sh0(3) || shN(45)].
        // The shN rows (3..D_out-1) are initialised to zero so the field
        // starts as pure DC (matching the 3DGS convention of `shN ← 0`),
        // and the optimiser is free to grow higher-order terms as needed.
        std::vector<float> h_W2((size_t)D_out * D_hid);
        const float w2_scale = 1.f / std::sqrt((float)D_hid);
        for (int r = 0; r < D_out; r++) {
            const bool zero_row = view_dep && r >= 3;   // shN rows
            for (int k = 0; k < D_hid; k++) {
                h_W2[(size_t)r * D_hid + k] =
                    zero_row ? 0.f : unif(rng) * w2_scale;
            }
        }
        NEXEL_CUDA_CHECK(cudaMemcpy(W2, h_W2.data(),
                                    h_W2.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));

        NEXEL_CUDA_CHECK(cudaMemset(b1, 0, (size_t)D_hid * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMemset(b2, 0, (size_t)D_out * sizeof(float)));
    }

    size_t total_param_floats() const {
        return (size_t)L * T * F
             + (size_t)D_hid * D_in
             + (size_t)D_hid
             + (size_t)D_out * D_hid
             + (size_t)D_out;
    }

    void free() {
        if (grid)         cudaFree(grid);
        if (W1)           cudaFree(W1);
        if (b1)           cudaFree(b1);
        if (W2)           cudaFree(W2);
        if (b2)           cudaFree(b2);
        if (level_scales) cudaFree(level_scales);
        grid = W1 = b1 = W2 = b2 = level_scales = nullptr;
    }

    AppearanceField()  = default;
    ~AppearanceField() { free(); }
    AppearanceField(const AppearanceField&) = delete;
    AppearanceField& operator=(const AppearanceField&) = delete;
};

// ─── AppearanceWorkspace: per-N transient buffers ────────────────────────────
//
// Resized on densification when N grows. We over-allocate to the current
// `capacity` and only realloc when N exceeds it (mirrors ForwardBuffers'
// `N_cap` strategy in train.cu).

struct AppearanceWorkspace {
    int N_cap  = 0;
    int D_in   = 0;
    int D_hid  = 0;
    int D_out  = 0;

    float* features         = nullptr;   // [N, D_in]
    float* hidden           = nullptr;   // [N, D_hid]    post-ReLU cache for backward
    float* mlp_out          = nullptr;   // [N, D_out]    pre-clamp output
    float* normalized_means = nullptr;   // [N, 3]        bbox-normalised hash query positions

    void ensure(int N, int D_in_, int D_hid_, int D_out_) {
        if (N <= N_cap && D_in == D_in_ && D_hid == D_hid_ && D_out == D_out_)
            return;
        free();
        N_cap = N;
        D_in  = D_in_;
        D_hid = D_hid_;
        D_out = D_out_;
        NEXEL_CUDA_CHECK(cudaMalloc(&features,         (size_t)N_cap * D_in  * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&hidden,           (size_t)N_cap * D_hid * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&mlp_out,          (size_t)N_cap * D_out * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&normalized_means, (size_t)N_cap * 3     * sizeof(float)));
    }

    void free() {
        if (features)         cudaFree(features);
        if (hidden)           cudaFree(hidden);
        if (mlp_out)          cudaFree(mlp_out);
        if (normalized_means) cudaFree(normalized_means);
        features = hidden = mlp_out = normalized_means = nullptr;
        N_cap = 0;
    }

    AppearanceWorkspace()  = default;
    ~AppearanceWorkspace() { free(); }
    AppearanceWorkspace(const AppearanceWorkspace&) = delete;
    AppearanceWorkspace& operator=(const AppearanceWorkspace&) = delete;
};

// ─── AppearanceGrads: gradient buffers for the parameter set ─────────────────

struct AppearanceGrads {
    int L     = 0, T = 0, F = 0;
    int D_in  = 0, D_hid = 0, D_out = 0;
    float* grad_grid = nullptr;  // [L, T, F]
    float* grad_W1   = nullptr;
    float* grad_b1   = nullptr;
    float* grad_W2   = nullptr;
    float* grad_b2   = nullptr;
    float* grad_features = nullptr;   // [N_cap, D_in] — workspace for hash_grid_bwd

    int N_cap_features = 0;

    void allocate(const AppearanceField& af) {
        free();
        L = af.L; T = af.T; F = af.F;
        D_in = af.D_in; D_hid = af.D_hid; D_out = af.D_out;
        NEXEL_CUDA_CHECK(cudaMalloc(&grad_grid, (size_t)L * T * F * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&grad_W1,   (size_t)D_hid * D_in * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&grad_b1,   (size_t)D_hid * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&grad_W2,   (size_t)D_out * D_hid * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMalloc(&grad_b2,   (size_t)D_out * sizeof(float)));
    }

    void ensure_features(int N) {
        if (N <= N_cap_features) return;
        if (grad_features) cudaFree(grad_features);
        NEXEL_CUDA_CHECK(cudaMalloc(&grad_features, (size_t)N * D_in * sizeof(float)));
        N_cap_features = N;
    }

    void zero_params() {
        NEXEL_CUDA_CHECK(cudaMemset(grad_grid, 0, (size_t)L * T * F * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMemset(grad_W1,   0, (size_t)D_hid * D_in * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMemset(grad_b1,   0, (size_t)D_hid * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMemset(grad_W2,   0, (size_t)D_out * D_hid * sizeof(float)));
        NEXEL_CUDA_CHECK(cudaMemset(grad_b2,   0, (size_t)D_out * sizeof(float)));
    }

    void free() {
        if (grad_grid)     cudaFree(grad_grid);
        if (grad_W1)       cudaFree(grad_W1);
        if (grad_b1)       cudaFree(grad_b1);
        if (grad_W2)       cudaFree(grad_W2);
        if (grad_b2)       cudaFree(grad_b2);
        if (grad_features) cudaFree(grad_features);
        grad_grid = grad_W1 = grad_b1 = grad_W2 = grad_b2 = grad_features = nullptr;
        N_cap_features = 0;
    }

    AppearanceGrads()  = default;
    ~AppearanceGrads() { free(); }
    AppearanceGrads(const AppearanceGrads&) = delete;
    AppearanceGrads& operator=(const AppearanceGrads&) = delete;
};

// ─── Activation kernel: mlp_out → colors with +0.5 & clamp[0,1] ──────────────
//
// Matches sh_eval_kernel's `fmaxf(0, fminf(1, v + 0.5f))` convention so
// that the rasterizer composite path is bit-identical between the SH and
// nexel-colour code paths (other than the source of `colors`).

__global__ void nexel_apply_color_activation_kernel(
    const float* __restrict__ mlp_out,  // [N, D_out]
    float*       __restrict__ colors,   // [N, 3]
    int N, int D_out
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    // Only the first 3 channels of mlp_out are interpreted as RGB. With
    // D_out > 3 the extras are ignored at the colour boundary; reserved
    // for future use (view-dependent SH-style bases).
    int limit = D_out < 3 ? D_out : 3;
    for (int c = 0; c < limit; c++) {
        float v = mlp_out[(size_t)n * D_out + c] + 0.5f;
        if (v < 0.f) v = 0.f;
        if (v > 1.f) v = 1.f;
        colors[(size_t)n * 3 + c] = v;
    }
    // Pad with 0 if D_out < 3 (shouldn't happen with our config).
    for (int c = limit; c < 3; c++) colors[(size_t)n * 3 + c] = 0.f;
}

// Backward: pass-through. Saturated entries should technically zero out
// the gradient but we ignore that here to match the legacy SH path.
__global__ void nexel_apply_color_activation_bwd_kernel(
    const float* __restrict__ grad_colors,  // [N, 3]
    float*       __restrict__ grad_mlp_out, // [N, D_out]   (written)
    int N, int D_out
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int limit = D_out < 3 ? D_out : 3;
    for (int c = 0; c < limit; c++)
        grad_mlp_out[(size_t)n * D_out + c] = grad_colors[(size_t)n * 3 + c];
    for (int c = limit; c < D_out; c++)
        grad_mlp_out[(size_t)n * D_out + c] = 0.f;
}

// Forward declarations — the packed SH eval launch helpers are defined
// after the orchestrators (in the SH-evaluation section below) but the
// view-dep branch needs them visible here.
inline void launch_nexel_sh_eval_packed_fwd(
    const float*, const float*, float, float, float, float*, int);
inline void launch_nexel_sh_eval_packed_bwd(
    const float*, const float*, float, float, float,
    const float*, float*, int);

// ─── Forward orchestrator ────────────────────────────────────────────────────

inline void nexel_color_forward(
    const float* d_means,
    const AppearanceField& af,
    AppearanceWorkspace& ws,
    const float* d_normalised_means,    // [N, 3]   hash query positions
    float* d_colors,                    // [N, 3]   output (clamped [0,1])
    int N,
    // World-space camera position. Only used in view_dep mode; pass any
    // value (e.g. 0,0,0) when af.view_dep is false.
    float cam_x = 0.f, float cam_y = 0.f, float cam_z = 0.f
) {
    if (N <= 0) return;
    ws.ensure(N, af.D_in, af.D_hid, af.D_out);

    launch_hash_grid_fwd(d_normalised_means, af.grid, af.level_scales,
                         ws.features,
                         N, af.L, af.F, (uint32_t)af.T);

    launch_mlp_fwd(ws.features,
                   af.W1, af.b1, af.W2, af.b2,
                   ws.hidden, ws.mlp_out,
                   N, af.D_in, af.D_hid, af.D_out);

    if (af.view_dep) {
        // mlp_out is interpreted as [N, 48] packed SH coefficients.
        // We use the *un-normalised* means and the camera position to
        // compute the per-Gaussian view direction.
        launch_nexel_sh_eval_packed_fwd(
            d_means, ws.mlp_out, cam_x, cam_y, cam_z, d_colors, N);
    } else {
        const int threads = 256;
        const int blocks  = (N + threads - 1) / threads;
        nexel_apply_color_activation_kernel<<<blocks, threads>>>(
            ws.mlp_out, d_colors, N, af.D_out);
        NEXEL_CUDA_CHECK(cudaGetLastError());
    }
}

// ─── Backward orchestrator ───────────────────────────────────────────────────
//
// Reuses `ws.mlp_out` as scratch for `grad_mlp_out` after forward has
// finished — the forward output is no longer needed once backward has
// the upstream grad_colors. This saves one [N, D_out] allocation.
//
// Param gradients (grad_grid, grad_W1, grad_b1, grad_W2, grad_b2) are
// accumulated. Caller is responsible for zeroing them (e.g. via
// grads.zero_params()) before each iter.

inline void nexel_color_backward(
    const float* d_means,
    const AppearanceField& af,
    AppearanceWorkspace& ws,
    const float* d_normalised_means,    // [N, 3]   hash query positions used in forward
    const float* d_grad_colors,
    AppearanceGrads& grads,
    int N,
    float cam_x = 0.f, float cam_y = 0.f, float cam_z = 0.f
) {
    if (N <= 0) return;
    grads.ensure_features(N);

    // Stage 1: grad_colors → grad_mlp_out.
    // In view_dep mode this is the SH-eval backward; otherwise the
    // pass-through activation backward (matches the legacy non-view-dep
    // path, ignoring clamp).
    if (af.view_dep) {
        // ws.mlp_out still holds the forward SH coefficients (we haven't
        // overwritten them yet) — needed by the bwd kernel to evaluate
        // unclamped vs clamped state. After this call we re-use the
        // buffer to store grad_mlp_out, but the bwd kernel reads then
        // writes the same memory atomically per-Gaussian so the in-place
        // overwrite is safe (each thread reads its own coeffs[i] before
        // writing v_coeffs[i]).
        launch_nexel_sh_eval_packed_bwd(
            d_means, ws.mlp_out, cam_x, cam_y, cam_z,
            d_grad_colors, ws.mlp_out, N);
    } else {
        const int threads = 256;
        const int blocks  = (N + threads - 1) / threads;
        nexel_apply_color_activation_bwd_kernel<<<blocks, threads>>>(
            d_grad_colors, ws.mlp_out, N, af.D_out);
        NEXEL_CUDA_CHECK(cudaGetLastError());
    }

    // Stage 2: MLP backward.
    launch_mlp_bwd(ws.features, ws.hidden,
                   af.W1, af.W2,
                   ws.mlp_out,                // upstream grad
                   grads.grad_features,
                   grads.grad_W1, grads.grad_b1,
                   grads.grad_W2, grads.grad_b2,
                   N, af.D_in, af.D_hid, af.D_out);

    // Stage 3: hash grid backward.
    launch_hash_grid_bwd(d_normalised_means, af.level_scales,
                         grads.grad_features,
                         grads.grad_grid,
                         N, af.L, af.F, (uint32_t)af.T);
}

// ─── Packed-coeff SH evaluation (for view-dependent appearance) ─────────────
//
// When AppearanceField.view_dep is true, the MLP outputs D_out=48
// coefficients per Gaussian: 3 DC + 45 (15 per channel for degrees 1–3).
// Layout matches sh_eval_kernel's split (sh0[N,3] || shN[N,45] with
// shN organised as [r_coeffs(15) || g_coeffs(15) || b_coeffs(15)]),
// but stored contiguously per Gaussian:
//
//   coeffs[i*48 + 0..2]  = sh0[i, 0..2]
//   coeffs[i*48 + 3..17]  = shN[i, r_coeffs]
//   coeffs[i*48 + 18..32] = shN[i, g_coeffs]
//   coeffs[i*48 + 33..47] = shN[i, b_coeffs]
//
// Forward applies the same +0.5 + clamp[0,1] activation as sh_eval_kernel
// so the rasterizer composite path is bit-identical with the legacy SH
// trainer. Backward zeroes the gradient on saturated pixels (matching
// sh_backward_kernel, unlike the simple-RGB path which ignores clamp).

namespace nx_sh {
__device__ static constexpr float C0   =  0.28209479177f;
__device__ static constexpr float C1   =  0.48860251190f;
__device__ static constexpr float C2_0 =  1.09254843059f;
__device__ static constexpr float C2_1 = -1.09254843059f;
__device__ static constexpr float C2_2 =  0.31539156525f;
__device__ static constexpr float C2_3 = -1.09254843059f;
__device__ static constexpr float C2_4 =  0.54627421529f;
__device__ static constexpr float C3_0 = -0.59004358992f;
__device__ static constexpr float C3_1 =  2.89061144264f;
__device__ static constexpr float C3_2 = -0.45704579946f;
__device__ static constexpr float C3_3 =  0.37317633259f;
__device__ static constexpr float C3_4 = -0.45704579946f;
__device__ static constexpr float C3_5 =  1.44530572132f;
__device__ static constexpr float C3_6 = -0.59004358992f;
}  // namespace nx_sh

__global__ void nexel_sh_eval_packed_fwd_kernel(
    const float* __restrict__ means,    // [N, 3]
    const float* __restrict__ coeffs,   // [N, 48]
    float cam_x, float cam_y, float cam_z,
    float*       __restrict__ colors,   // [N, 3]   (written, clamped [0,1])
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float dx = means[i*3+0] - cam_x;
    float dy = means[i*3+1] - cam_y;
    float dz = means[i*3+2] - cam_z;
    float inv = rsqrtf(dx*dx + dy*dy + dz*dz + 1e-12f);
    float x = dx * inv, y = dy * inv, z = dz * inv;
    float xx = x*x, yy = y*y, zz = z*z;

    const float* c0 = coeffs + (size_t)i * 48;          // sh0[r, g, b]
    const float* cN = coeffs + (size_t)i * 48 + 3;      // shN: r(15) || g(15) || b(15)

    for (int c = 0; c < 3; c++) {
        const float* sh = cN + c * 15;
        float v = nx_sh::C0 * c0[c];

        // Degree 1
        v += nx_sh::C1 * (-sh[0]*y + sh[1]*z - sh[2]*x);
        // Degree 2
        v += nx_sh::C2_0 * sh[3] * (x*y);
        v += nx_sh::C2_1 * sh[4] * (y*z);
        v += nx_sh::C2_2 * sh[5] * (2.f*zz - xx - yy);
        v += nx_sh::C2_3 * sh[6] * (x*z);
        v += nx_sh::C2_4 * sh[7] * (xx - yy);
        // Degree 3
        v += nx_sh::C3_0 * sh[8]  * y * (3.f*xx - yy);
        v += nx_sh::C3_1 * sh[9]  * x * y * z;
        v += nx_sh::C3_2 * sh[10] * y * (4.f*zz - xx - yy);
        v += nx_sh::C3_3 * sh[11] * z * (2.f*zz - 3.f*xx - 3.f*yy);
        v += nx_sh::C3_4 * sh[12] * x * (4.f*zz - xx - yy);
        v += nx_sh::C3_5 * sh[13] * z * (xx - yy);
        v += nx_sh::C3_6 * sh[14] * x * (xx - 3.f*yy);

        float clamped = v + 0.5f;
        if (clamped < 0.f) clamped = 0.f;
        if (clamped > 1.f) clamped = 1.f;
        colors[(size_t)i*3 + c] = clamped;
    }
}

__global__ void nexel_sh_eval_packed_bwd_kernel(
    const float* __restrict__ means,
    const float* __restrict__ coeffs,        // [N, 48]
    float cam_x, float cam_y, float cam_z,
    const float* __restrict__ v_colors,      // [N, 3]
    float*       __restrict__ v_coeffs,      // [N, 48]   (written, not accumulated)
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float dx = means[i*3+0] - cam_x;
    float dy = means[i*3+1] - cam_y;
    float dz = means[i*3+2] - cam_z;
    float inv = rsqrtf(dx*dx + dy*dy + dz*dz + 1e-12f);
    float x = dx * inv, y = dy * inv, z = dz * inv;
    float xx = x*x, yy = y*y, zz = z*z;

    const float* c0 = coeffs + (size_t)i * 48;
    const float* cN = coeffs + (size_t)i * 48 + 3;

    for (int c = 0; c < 3; c++) {
        const float* sh = cN + c * 15;
        // Re-evaluate forward to check clamp.
        float v = nx_sh::C0 * c0[c];
        v += nx_sh::C1 * (-sh[0]*y + sh[1]*z - sh[2]*x);
        v += nx_sh::C2_0 * sh[3] * (x*y);
        v += nx_sh::C2_1 * sh[4] * (y*z);
        v += nx_sh::C2_2 * sh[5] * (2.f*zz - xx - yy);
        v += nx_sh::C2_3 * sh[6] * (x*z);
        v += nx_sh::C2_4 * sh[7] * (xx - yy);
        v += nx_sh::C3_0 * sh[8]  * y * (3.f*xx - yy);
        v += nx_sh::C3_1 * sh[9]  * x * y * z;
        v += nx_sh::C3_2 * sh[10] * y * (4.f*zz - xx - yy);
        v += nx_sh::C3_3 * sh[11] * z * (2.f*zz - 3.f*xx - 3.f*yy);
        v += nx_sh::C3_4 * sh[12] * x * (4.f*zz - xx - yy);
        v += nx_sh::C3_5 * sh[13] * z * (xx - yy);
        v += nx_sh::C3_6 * sh[14] * x * (xx - 3.f*yy);
        float unclamped = v + 0.5f;

        float* g0  = v_coeffs + (size_t)i * 48;
        float* gN  = g0 + 3 + c * 15;

        if (unclamped <= 0.f || unclamped >= 1.f) {
            // Clamp saturated → gradient zero through this channel.
            g0[c] = 0.f;
            for (int k = 0; k < 15; k++) gN[k] = 0.f;
            continue;
        }

        const float grad = v_colors[(size_t)i*3 + c];
        g0[c]  = grad * nx_sh::C0;
        gN[0]  = grad * (nx_sh::C1 * (-y));
        gN[1]  = grad * (nx_sh::C1 * z);
        gN[2]  = grad * (nx_sh::C1 * (-x));
        gN[3]  = grad * (nx_sh::C2_0 * (x*y));
        gN[4]  = grad * (nx_sh::C2_1 * (y*z));
        gN[5]  = grad * (nx_sh::C2_2 * (2.f*zz - xx - yy));
        gN[6]  = grad * (nx_sh::C2_3 * (x*z));
        gN[7]  = grad * (nx_sh::C2_4 * (xx - yy));
        gN[8]  = grad * (nx_sh::C3_0 * y * (3.f*xx - yy));
        gN[9]  = grad * (nx_sh::C3_1 * x * y * z);
        gN[10] = grad * (nx_sh::C3_2 * y * (4.f*zz - xx - yy));
        gN[11] = grad * (nx_sh::C3_3 * z * (2.f*zz - 3.f*xx - 3.f*yy));
        gN[12] = grad * (nx_sh::C3_4 * x * (4.f*zz - xx - yy));
        gN[13] = grad * (nx_sh::C3_5 * z * (xx - yy));
        gN[14] = grad * (nx_sh::C3_6 * x * (xx - 3.f*yy));
    }
}

inline void launch_nexel_sh_eval_packed_fwd(
    const float* d_means, const float* d_coeffs,
    float cam_x, float cam_y, float cam_z,
    float* d_colors, int N
) {
    if (N <= 0) return;
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    nexel_sh_eval_packed_fwd_kernel<<<blocks, threads>>>(
        d_means, d_coeffs, cam_x, cam_y, cam_z, d_colors, N);
    NEXEL_CUDA_CHECK(cudaGetLastError());
}

inline void launch_nexel_sh_eval_packed_bwd(
    const float* d_means, const float* d_coeffs,
    float cam_x, float cam_y, float cam_z,
    const float* d_v_colors, float* d_v_coeffs, int N
) {
    if (N <= 0) return;
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    nexel_sh_eval_packed_bwd_kernel<<<blocks, threads>>>(
        d_means, d_coeffs, cam_x, cam_y, cam_z,
        d_v_colors, d_v_coeffs, N);
    NEXEL_CUDA_CHECK(cudaGetLastError());
}

// ─── Position normalisation kernel ─────────────────────────────────────────
//
// Hash-grid scale factors assume queries roughly in [0,1]^3. Scene-space
// Gaussian means span the COLMAP bounding box, which may be tens or
// hundreds of units across — without normalisation the coarse hash
// levels collapse to a single near-uniform table lookup.
//
// `origin` = bbox_min, `inv_extent` = 1 / max(bbox_extent_per_axis, eps).
// Caller passes both as float3 (3 floats packed). Out positions are
// clamped to [0, 1] so points exactly on the max corner still hash into
// the table interior.

__global__ void nexel_normalize_positions_kernel(
    const float* __restrict__ positions_in,    // [N, 3] world-space
    float3 origin,
    float3 inv_extent,
    float* __restrict__ positions_out,         // [N, 3] in roughly [0, 1]
    int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float x = (positions_in[n*3+0] - origin.x) * inv_extent.x;
    float y = (positions_in[n*3+1] - origin.y) * inv_extent.y;
    float z = (positions_in[n*3+2] - origin.z) * inv_extent.z;
    // Soft clamp so points outside the original bbox (e.g. drifted means
    // during training) hash into the same range. Clamp doesn't affect
    // gradient flow because we don't backprop through positions yet.
    if (x < 0.f) x = 0.f; else if (x > 1.f) x = 1.f;
    if (y < 0.f) y = 0.f; else if (y > 1.f) y = 1.f;
    if (z < 0.f) z = 0.f; else if (z > 1.f) z = 1.f;
    positions_out[n*3+0] = x;
    positions_out[n*3+1] = y;
    positions_out[n*3+2] = z;
}

inline void launch_nexel_normalize_positions(
    const float* d_pos_in, float3 origin, float3 inv_extent,
    float* d_pos_out, int N
) {
    if (N <= 0) return;
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    nexel_normalize_positions_kernel<<<blocks, threads>>>(
        d_pos_in, origin, inv_extent, d_pos_out, N);
    NEXEL_CUDA_CHECK(cudaGetLastError());
}

// ─── Standalone smoke test ──────────────────────────────────────────────────

#ifndef INCLUDED_AS_HEADER
int main() {
    // Tiny scene to keep allocations cheap.
    constexpr int N    = 8;
    constexpr int L    = 4;
    constexpr int T    = 64;
    constexpr int F    = 2;
    constexpr int D_hid= 16;
    constexpr int D_out= 3;
    constexpr float MIN_RES = 4.f;
    constexpr float MAX_RES = 16.f;

    AppearanceField af;
    af.allocate(L, T, F, D_hid, D_out, MIN_RES, MAX_RES);
    af.init_random(/*seed=*/123);

    AppearanceWorkspace ws;
    AppearanceGrads grads;
    grads.allocate(af);

    // Random means in [0, 1]^3.
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<float> h_means((size_t)N * 3);
    for (auto& v : h_means) v = u(rng);

    float* d_means;
    NEXEL_CUDA_CHECK(cudaMalloc(&d_means, h_means.size() * sizeof(float)));
    NEXEL_CUDA_CHECK(cudaMemcpy(d_means, h_means.data(),
                                h_means.size() * sizeof(float),
                                cudaMemcpyHostToDevice));

    float* d_colors;
    NEXEL_CUDA_CHECK(cudaMalloc(&d_colors, (size_t)N * 3 * sizeof(float)));

    // Smoke uses non-view-dep mode; cam_pos irrelevant. Same buffer passed
    // for raw + normalised means (no normalisation step in the smoke).
    nexel_color_forward(d_means, af, ws, d_means, d_colors, N);
    NEXEL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_colors((size_t)N * 3);
    NEXEL_CUDA_CHECK(cudaMemcpy(h_colors.data(), d_colors,
                                h_colors.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
    printf("nexel_color forward smoke (N=%d, L=%d, T=%d, F=%d, D_hid=%d):\n",
           N, L, T, F, D_hid);
    bool all_in_range = true;
    for (int n = 0; n < N; n++) {
        printf("  n=%d  rgb=(%.4f, %.4f, %.4f)\n", n,
               h_colors[n*3+0], h_colors[n*3+1], h_colors[n*3+2]);
        for (int c = 0; c < 3; c++) {
            float v = h_colors[n*3+c];
            if (v < 0.f || v > 1.f) all_in_range = false;
        }
    }

    // Backward smoke: random upstream grad_colors, check gradients
    // accumulate to plausible magnitudes (non-zero, finite).
    std::vector<float> h_grad_colors((size_t)N * 3);
    std::uniform_real_distribution<float> gd(-0.1f, 0.1f);
    for (auto& v : h_grad_colors) v = gd(rng);
    float* d_grad_colors;
    NEXEL_CUDA_CHECK(cudaMalloc(&d_grad_colors, h_grad_colors.size() * sizeof(float)));
    NEXEL_CUDA_CHECK(cudaMemcpy(d_grad_colors, h_grad_colors.data(),
                                h_grad_colors.size() * sizeof(float),
                                cudaMemcpyHostToDevice));

    grads.zero_params();
    nexel_color_backward(d_means, af, ws, d_means, d_grad_colors, grads, N);
    NEXEL_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_grad_grid((size_t)L * T * F);
    NEXEL_CUDA_CHECK(cudaMemcpy(h_grad_grid.data(), grads.grad_grid,
                                h_grad_grid.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
    int nz_grid = 0;
    double sum_grid = 0.0;
    for (float v : h_grad_grid) { if (std::abs(v) > 1e-9) nz_grid++; sum_grid += v; }

    std::vector<float> h_grad_W1((size_t)D_hid * af.D_in);
    NEXEL_CUDA_CHECK(cudaMemcpy(h_grad_W1.data(), grads.grad_W1,
                                h_grad_W1.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
    int nz_W1 = 0;
    for (float v : h_grad_W1) if (std::abs(v) > 1e-9) nz_W1++;

    printf("backward smoke: nonzero grad_grid entries=%d/%zu  sum=%.4e   nonzero grad_W1=%d/%zu\n",
           nz_grid, h_grad_grid.size(), sum_grid, nz_W1, h_grad_W1.size());

    cudaFree(d_means); cudaFree(d_colors); cudaFree(d_grad_colors);
    af.free(); ws.free(); grads.free();

    bool ok = all_in_range && nz_grid > 0 && nz_W1 > 0;
    printf("%s\n", ok ? "nexel_color smoke OK" : "nexel_color smoke FAILED");
    return ok ? 0 : 1;
}
#endif
