// loss.cu
//
// Photometric loss for 2DGS training:
//   L = (1 − λ) · L1(render, gt) + λ · (1 − SSIM(render, gt))
//
// Image format: float* [H * W * C], row-major HWC layout
//   pixel (y, x) channel c → img[y*W*C + x*C + c]
//
// API:
//   LossResult photometric_loss(render, gt, grad_out, H, W, lambda=0.2f, C=3)
//   Returns scalar loss (host). Writes dL/d(render) into grad_out (GPU).
//
// References:
//   LichtFeld-Studio ssim.cu  — two-pass separable conv, combined dm_dmu1 trick
//   gsplat/utils.py           — L = 0.8*L1 + 0.2*(1−SSIM), lambda_dssim = 0.2
//   Wang et al. 2004          — SSIM paper

#include "splat_data.cuh"   // CUDA_CHECK, std::vector
#include <cstdio>
#include <cmath>
#include <vector>

// ─── Constants ───────────────────────────────────────────────────────────────

// 11-tap separable Gaussian, σ=1.5 (same as PyTorch / LichtFeld-Studio SSIM)
__constant__ float cG[11] = {
    0.001028380123898387f,  0.0075987582094967365f, 0.036000773310661316f,
    0.10936068743467331f,   0.21300552785396576f,   0.26601171493530273f,
    0.21300552785396576f,   0.10936068743467331f,   0.036000773310661316f,
    0.0075987582094967365f, 0.001028380123898387f,
};

static constexpr float SSIM_C1 = 0.01f * 0.01f;   // (K1·L)²  K1=0.01, L=1
static constexpr float SSIM_C2 = 0.03f * 0.03f;   // (K2·L)²  K2=0.03
static constexpr int   SSIM_R  = 5;                // 11×11 window

// ─── Device helpers ──────────────────────────────────────────────────────────

__device__ __forceinline__ float get_pix(
    const float* img, int y, int x, int c, int H, int W, int C)
{
    if (y < 0 || y >= H || x < 0 || x >= W) return 0.f;
    return img[y * W * C + x * C + c];
}

// ─── Utility kernels ─────────────────────────────────────────────────────────

// Fill array with a constant value
__global__ void fill_kernel(float* arr, float val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) arr[i] = val;
}

// L1 per-element: l1_vals[i] = |a[i]−b[i]|/N,  grad[i] = sign(a[i]−b[i])/N
__global__ void l1_fwd_bwd_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ grad,
    float*       __restrict__ l1_vals,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float d    = a[i] - b[i];
    l1_vals[i] = fabsf(d) / (float)N;
    grad[i]    = (d > 0.f ? 1.f : (d < 0.f ? -1.f : 0.f)) / (float)N;
}

// Parallel sum reduction via shared memory + atomicAdd
__global__ void reduce_sum_kernel(const float* in, float* out, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < N) ? in[i] : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// out[i] = sa*a[i] + sb*b[i]
__global__ void scale_add_kernel(
    const float* __restrict__ a, float sa,
    const float* __restrict__ b, float sb,
    float* __restrict__ out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = sa * a[i] + sb * b[i];
}

// ─── SSIM forward ────────────────────────────────────────────────────────────
//
// One thread per output pixel. For each channel c, computes 11×11 Gaussian-
// weighted statistics and the SSIM value. Saves three derivative maps needed
// for the backward pass.
//
// Combined dm_dmu1 trick (LichtFeld-Studio):
//   Naive ∂SSIM/∂mu1 uses 2 terms. We fold the (−2mu1·∂SSIM/∂σ1²) and
//   (−mu2·∂SSIM/∂σ12) correction terms into dm_dmu1 so the backward kernel
//   only needs raw img1[p] and img2[p], not (img1[p]−mu1) or (img2[p]−mu2).
//   This eliminates saving the local means.
//
//   Combined dm_dmu1 = 2mu2·D/(AB) − 2mu2·C/(AB) − 2mu1·CD/(A²B) + 2mu1·CD/(AB²)
//     where A=mu1²+mu2²+C1, B=σ1²+σ2²+C2, C=2mu1·mu2+C1, D=2σ12+C2

__global__ void ssim_fwd_kernel(
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    float* __restrict__ ssim_map,
    float* __restrict__ dm_dmu1,
    float* __restrict__ dm_dsigma1_sq,
    float* __restrict__ dm_dsigma12,
    int H, int W, int C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    for (int c = 0; c < C; c++) {
        // Gaussian-weighted window statistics
        float sx = 0, sy = 0, sx2 = 0, sy2 = 0, sxy = 0;
        for (int dy = -SSIM_R; dy <= SSIM_R; dy++) {
            float wy = cG[dy + SSIM_R];
            for (int dx = -SSIM_R; dx <= SSIM_R; dx++) {
                float w  = wy * cG[dx + SSIM_R];
                float px = get_pix(img1, y+dy, x+dx, c, H, W, C);
                float py = get_pix(img2, y+dy, x+dx, c, H, W, C);
                sx  += w*px;    sy  += w*py;
                sx2 += w*px*px; sy2 += w*py*py;
                sxy += w*px*py;
            }
        }

        float mu1 = sx, mu2 = sy;
        float s1sq = sx2 - mu1*mu1;
        float s2sq = sy2 - mu2*mu2;
        float s12  = sxy - mu1*mu2;

        float A  = mu1*mu1 + mu2*mu2 + SSIM_C1;
        float B  = s1sq + s2sq + SSIM_C2;
        float Cv = 2.f*mu1*mu2 + SSIM_C1;   // numerator term (luminance+contrast)
        float D  = 2.f*s12     + SSIM_C2;   // numerator term (structure)
        float AB = A * B;

        int idx = y*W*C + x*C + c;
        ssim_map[idx] = (AB > 1e-8f) ? Cv*D/AB : 1.f;

        if (AB > 1e-8f) {
            dm_dsigma12[idx]   =  2.f*Cv / AB;
            dm_dsigma1_sq[idx] = -Cv*D   / (A*B*B);
            dm_dmu1[idx]       =  2.f*mu2*D  / AB         // combined partial (see above)
                               -  2.f*mu2*Cv / AB
                               -  2.f*mu1*Cv*D / (A*A*B)
                               +  2.f*mu1*Cv*D / (A*B*B);
        } else {
            dm_dmu1[idx] = dm_dsigma1_sq[idx] = dm_dsigma12[idx] = 0.f;
        }
    }
}

// ─── SSIM backward ───────────────────────────────────────────────────────────
//
// One thread per pixel. Accumulates gradient from every 11×11 window containing
// this pixel. The combined dm_dmu1 from the forward lets us write:
//
//   dL/d(img1[y,x,c]) =
//     Σ_{w: (y,x)∈window(w)} g[y−wy, x−wx] · dL_dmap[w,c] ·
//       ( dm_dmu1[w,c]
//       + 2·img1[y,x,c] · dm_σ1²[w,c]    ← raw img1[p], NOT img1[p]−mu1
//       +   img2[y,x,c] · dm_σ12[w,c] )  ← raw img2[p], NOT img2[p]−mu2
//
// Equivalently: three Gaussian convolution passes on (dm_dmu1·dL), (dm_σ1²·dL),
// (dm_σ12·dL), then weighted by 1, 2·img1, img2 respectively.

__global__ void ssim_bwd_kernel(
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    const float* __restrict__ dL_dmap,
    const float* __restrict__ dm_dmu1,
    const float* __restrict__ dm_dsigma1_sq,
    const float* __restrict__ dm_dsigma12,
    float*       __restrict__ grad_img1,
    int H, int W, int C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    for (int c = 0; c < C; c++) {
        float p1 = img1[y*W*C + x*C + c];
        float p2 = img2[y*W*C + x*C + c];

        // Accumulate from all windows whose center (cy,cx) has (y,x) in their 11×11 halo.
        // Pixel (y,x) is at offset (dy, dx) = (y−cy, x−cx) from window center (cy,cx).
        float acc_mu1 = 0.f, acc_s1sq = 0.f, acc_s12 = 0.f;
        for (int dy = -SSIM_R; dy <= SSIM_R; dy++) {
            int cy = y - dy;
            if (cy < 0 || cy >= H) continue;
            float wy = cG[dy + SSIM_R];
            for (int dx = -SSIM_R; dx <= SSIM_R; dx++) {
                int cx = x - dx;
                if (cx < 0 || cx >= W) continue;
                float g     = wy * cG[dx + SSIM_R];
                int   wi    = cy*W*C + cx*C + c;
                float chain = dL_dmap[wi];
                acc_mu1  += g * chain * dm_dmu1[wi];
                acc_s1sq += g * chain * dm_dsigma1_sq[wi];
                acc_s12  += g * chain * dm_dsigma12[wi];
            }
        }
        grad_img1[y*W*C + x*C + c] = acc_mu1 + 2.f*p1*acc_s1sq + p2*acc_s12;
    }
}

// ─── Host entry point ─────────────────────────────────────────────────────────

struct LossResult {
    float loss;
    float loss_l1;
    float loss_dssim;   // = 1 − mean(SSIM)
};

struct LossWorkspace {
    float* d_l1_vals = nullptr;
    float* d_grad_l1 = nullptr;
    float* d_dL_dmap = nullptr;
    float* d_ssim_map = nullptr;
    float* d_dm_dmu1 = nullptr;
    float* d_dm_dsigma1_sq = nullptr;
    float* d_dm_dsigma12 = nullptr;
    float* d_grad_ssim = nullptr;
    float* d_scalar = nullptr;
    int n_cap = 0;
};

static void free_loss_workspace(LossWorkspace& ws) {
    cudaFree(ws.d_l1_vals);       ws.d_l1_vals = nullptr;
    cudaFree(ws.d_grad_l1);       ws.d_grad_l1 = nullptr;
    cudaFree(ws.d_dL_dmap);       ws.d_dL_dmap = nullptr;
    cudaFree(ws.d_ssim_map);      ws.d_ssim_map = nullptr;
    cudaFree(ws.d_dm_dmu1);       ws.d_dm_dmu1 = nullptr;
    cudaFree(ws.d_dm_dsigma1_sq); ws.d_dm_dsigma1_sq = nullptr;
    cudaFree(ws.d_dm_dsigma12);   ws.d_dm_dsigma12 = nullptr;
    cudaFree(ws.d_grad_ssim);     ws.d_grad_ssim = nullptr;
    cudaFree(ws.d_scalar);        ws.d_scalar = nullptr;
    ws.n_cap = 0;
}

static void alloc_loss_workspace(LossWorkspace& ws, int N) {
    CUDA_CHECK(cudaMalloc(&ws.d_l1_vals,       N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_grad_l1,       N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_dL_dmap,       N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_ssim_map,      N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_dm_dmu1,       N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_dm_dsigma1_sq, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_dm_dsigma12,   N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_grad_ssim,     N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws.d_scalar,        sizeof(float)));
    ws.n_cap = N;
}

static void ensure_loss_workspace(LossWorkspace& ws, int N) {
    if (ws.n_cap >= N) return;
    free_loss_workspace(ws);
    alloc_loss_workspace(ws, N);
}

// render, gt, grad_out are GPU device pointers.
// grad_out receives dL/d(render) [H*W*C].
LossResult photometric_loss(
    const float* render,
    const float* gt,
    float*       grad_out,
    LossWorkspace& ws,
    int H, int W,
    float lambda = 0.2f,
    int C = 3)
{
    const int N   = H * W * C;
    const int T   = 256;
    const int B1D = (N + T - 1) / T;
    dim3 blk2d(16, 16);
    dim3 grd2d((W+15)/16, (H+15)/16);

    ensure_loss_workspace(ws, N);
    float* d_l1_vals = ws.d_l1_vals;
    float* d_grad_l1 = ws.d_grad_l1;
    float* d_dL_dmap = ws.d_dL_dmap;
    float* d_ssim_map = ws.d_ssim_map;
    float* d_dm_dmu1 = ws.d_dm_dmu1;
    float* d_dm_dsigma1_sq = ws.d_dm_dsigma1_sq;
    float* d_dm_dsigma12 = ws.d_dm_dsigma12;
    float* d_grad_ssim = ws.d_grad_ssim;
    float* d_scalar = ws.d_scalar;

    // ── L1 forward + backward ────────────────────────────────────────────────
    l1_fwd_bwd_kernel<<<B1D, T>>>(render, gt, d_grad_l1, d_l1_vals, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_scalar, 0, sizeof(float)));
    reduce_sum_kernel<<<B1D, T, T*sizeof(float)>>>(d_l1_vals, d_scalar, N);
    CUDA_CHECK(cudaGetLastError());
    float loss_l1;
    CUDA_CHECK(cudaMemcpy(&loss_l1, d_scalar, sizeof(float), cudaMemcpyDeviceToHost));

    // ── SSIM forward ─────────────────────────────────────────────────────────
    ssim_fwd_kernel<<<grd2d, blk2d>>>(
        render, gt, d_ssim_map,
        d_dm_dmu1, d_dm_dsigma1_sq, d_dm_dsigma12,
        H, W, C);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_scalar, 0, sizeof(float)));
    reduce_sum_kernel<<<B1D, T, T*sizeof(float)>>>(d_ssim_map, d_scalar, N);
    CUDA_CHECK(cudaGetLastError());
    float ssim_sum;
    CUDA_CHECK(cudaMemcpy(&ssim_sum, d_scalar, sizeof(float), cudaMemcpyDeviceToHost));
    float mean_ssim  = ssim_sum / (float)N;
    float loss_dssim = 1.f - mean_ssim;

    // ── SSIM backward ────────────────────────────────────────────────────────
    // loss_ssim = 1 − mean(ssim_map)  →  dL/d(ssim_map[p]) = −1/N  (uniform)
    fill_kernel<<<B1D, T>>>(d_dL_dmap, -1.f / (float)N, N);
    CUDA_CHECK(cudaGetLastError());

    ssim_bwd_kernel<<<grd2d, blk2d>>>(
        render, gt, d_dL_dmap,
        d_dm_dmu1, d_dm_dsigma1_sq, d_dm_dsigma12,
        d_grad_ssim,
        H, W, C);
    CUDA_CHECK(cudaGetLastError());

    // ── Combine: dL/d(render) = (1−λ)·grad_l1 + λ·grad_ssim ────────────────
    // grad_ssim already includes the −1/N factor from dL_dmap,
    // so it equals d(1−SSIM)/d(render) · (1/something)?
    //
    // Let's trace: dL_dmap[p] = −1/N, so the backward computes
    //   grad_ssim[p] = Σ_w g·(−1/N)·(dm_dmu1 + 2p1·dm_s1sq + p2·dm_s12)
    //                = −(1/N) · [Gauss★dm_dmu1 + 2p1·Gauss★dm_s1sq + p2·Gauss★dm_s12]
    //                = d(1−SSIM)/d(render[p])
    //
    // Combined loss gradient:
    //   dL/d(render) = (1−λ)·grad_l1 + λ·grad_ssim
    scale_add_kernel<<<B1D, T>>>(
        d_grad_l1,  1.f - lambda,
        d_grad_ssim, lambda,
        grad_out, N);
    CUDA_CHECK(cudaGetLastError());

    return {(1.f-lambda)*loss_l1 + lambda*loss_dssim, loss_l1, loss_dssim};
}

LossResult photometric_loss(
    const float* render,
    const float* gt,
    float*       grad_out,
    int H, int W,
    float lambda = 0.2f,
    int C = 3)
{
    LossWorkspace ws{};
    ensure_loss_workspace(ws, H * W * C);
    LossResult out = photometric_loss(render, gt, grad_out, ws, H, W, lambda, C);
    free_loss_workspace(ws);
    return out;
}

#ifndef INCLUDED_AS_HEADER
// ─── main — tests ─────────────────────────────────────────────────────────────
int main() {
    const int H = 64, W = 64, C = 3;
    const int N = H * W * C;

    std::vector<float> h_img(N);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < C; c++)
                h_img[(y*W+x)*C+c] = ((x/8 + y/8) % 2 == 0) ? 0.8f : 0.2f;

    float *d_render, *d_gt, *d_grad;
    CUDA_CHECK(cudaMalloc(&d_render, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gt,     N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad,   N*sizeof(float)));

    // Test 1: identical images → loss = 0
    CUDA_CHECK(cudaMemcpy(d_render, h_img.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gt,     h_img.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    auto r1 = photometric_loss(d_render, d_gt, d_grad, H, W);
    printf("Test 1 — identical images:\n");
    printf("  L1     = %.6f  (expected ~0.0)\n", r1.loss_l1);
    printf("  D-SSIM = %.6f  (expected ~0.0)\n", r1.loss_dssim);
    printf("  Loss   = %.6f  (expected ~0.0)\n\n", r1.loss);

    // Test 2: render=0.5, gt=0.0
    std::vector<float> h_render(N, 0.5f), h_gt(N, 0.0f);
    CUDA_CHECK(cudaMemcpy(d_render, h_render.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gt,     h_gt.data(),     N*sizeof(float), cudaMemcpyHostToDevice));
    auto r2 = photometric_loss(d_render, d_gt, d_grad, H, W);
    printf("Test 2 — render=0.5, gt=0.0:\n");
    printf("  L1     = %.6f  (expected 0.5)\n", r2.loss_l1);
    printf("  D-SSIM = %.6f\n",                  r2.loss_dssim);
    printf("  Loss   = %.6f  (expected %.4f)\n\n", r2.loss, 0.8f*0.5f + 0.2f*r2.loss_dssim);

    // Test 3: numerical gradient check at pixel 0
    std::vector<float> h_r3 = h_img, h_g3 = h_img;
    h_g3[0] += 0.1f;  // gt slightly different at first element
    CUDA_CHECK(cudaMemcpy(d_render, h_r3.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gt,     h_g3.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    photometric_loss(d_render, d_gt, d_grad, H, W);
    float h_grad0;
    CUDA_CHECK(cudaMemcpy(&h_grad0, d_grad, sizeof(float), cudaMemcpyDeviceToHost));

    const float eps = 1e-3f;
    auto perturb = [&](float delta) {
        std::vector<float> tmp = h_r3; tmp[0] += delta;
        CUDA_CHECK(cudaMemcpy(d_render, tmp.data(), N*sizeof(float), cudaMemcpyHostToDevice));
        return photometric_loss(d_render, d_gt, d_grad, H, W).loss;
    };
    float num_grad = (perturb(eps) - perturb(-eps)) / (2.f * eps);
    printf("Test 3 — numerical gradient check at pixel 0:\n");
    printf("  Analytic  = % .6f\n", h_grad0);
    printf("  Numerical = % .6f\n", num_grad);
    printf("  Rel error = %.2f%%\n\n",
           fabsf(h_grad0 - num_grad) / (fabsf(num_grad) + 1e-8f) * 100.f);

    cudaFree(d_render); cudaFree(d_gt); cudaFree(d_grad);
    printf("Done.\n");
    return 0;
}
#endif
