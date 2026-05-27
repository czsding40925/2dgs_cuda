// mlp.cu
//
// Fused 2-layer MLP for the Phase B (Nexels) appearance pipeline:
//
//   h = ReLU(W1 · x + b1)                     // W1: [D_hid, D_in], b1: [D_hid]
//   y =      W2 · h + b2                      // W2: [D_out, D_hid], b2: [D_out]
//
// One Gaussian per thread. Weights and biases are shared across all
// Gaussians, so each block stages W1/W2/b1/b2 into shared memory once and
// every thread in the block reads them from there. The MLP runs after
// the hash-grid query: x = hash_grid_features (D_in = L * F = 32 in the
// default Nexels config).
//
// Backward
//   v_y = grad_output                                              // [N, D_out]
//   v_b2 += sum_n v_y_n                                            // [D_out]
//   v_W2 += sum_n outer(v_y_n, h_n)                                // [D_out, D_hid]
//   v_h_n = (W2^T · v_y_n) * 1[pre_relu_n > 0]                     // [D_hid]
//   v_b1 += sum_n v_h_n                                            // [D_hid]
//   v_W1 += sum_n outer(v_h_n, x_n)                                // [D_hid, D_in]
//   v_x_n = W1^T · v_h_n                                           // [D_in]
//
// To bound the global-atomic count on the parameter gradients, each block
// accumulates v_W1/v_W2/v_b1/v_b2 in shared memory across all of its
// Gaussians, then a single sync + grid-strided atomicAdd flushes the
// block-local accumulators to the global gradient buffers. For a block
// of 128 threads this is a ~128× reduction in global atomic traffic
// versus per-thread atomicAdd.
//
// Conventions
//   * post-ReLU `hidden_cache` written by forward, consumed by backward.
//   * Bias gradients are concatenated nowhere; we keep them as separate
//     [D_hid] / [D_out] tensors.
//   * D_in / D_hid / D_out are template parameters so the kernel can
//     specialise. We expose the default Nexels config (32, 64, 3) plus
//     a flexible runtime path used by the gradcheck.

#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef MLP_CUDA_CHECK
#define MLP_CUDA_CHECK(call)                                                    \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "[mlp] CUDA error at %s:%d — %s\n",                       \
              __FILE__, __LINE__, cudaGetErrorString(_e));                      \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)
#endif

// ─── Forward kernel ───────────────────────────────────────────────────────────

__global__ void mlp_forward_kernel(
    const float* __restrict__ features,     // [N, D_in]
    const float* __restrict__ W1,           // [D_hid, D_in]
    const float* __restrict__ b1,           // [D_hid]
    const float* __restrict__ W2,           // [D_out, D_hid]
    const float* __restrict__ b2,           // [D_out]
    float*       __restrict__ hidden_cache, // [N, D_hid]   post-ReLU
    float*       __restrict__ output,       // [N, D_out]
    int N, int D_in, int D_hid, int D_out
) {
    extern __shared__ float mlp_smem[];
    float* sW1 = mlp_smem;                                            // D_hid * D_in
    float* sb1 = sW1 + (size_t)D_hid * D_in;                          // D_hid
    float* sW2 = sb1 + D_hid;                                         // D_out * D_hid
    float* sb2 = sW2 + (size_t)D_out * D_hid;                         // D_out

    // Cooperative load of weights into shared mem.
    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    for (int i = tid; i < D_hid * D_in; i += block_threads) sW1[i] = W1[i];
    for (int i = tid; i < D_hid;          i += block_threads) sb1[i] = b1[i];
    for (int i = tid; i < D_out * D_hid;  i += block_threads) sW2[i] = W2[i];
    for (int i = tid; i < D_out;          i += block_threads) sb2[i] = b2[i];
    __syncthreads();

    int n = blockIdx.x * block_threads + tid;
    if (n >= N) return;

    const float* x = features + (size_t)n * D_in;
    float* h_out   = hidden_cache + (size_t)n * D_hid;
    float* y_out   = output       + (size_t)n * D_out;

    // h = ReLU(W1 · x + b1)
    // Tight per-thread loops; D_hid is small (<= 128 in practice) so we
    // don't bother with register tiling here.
    for (int j = 0; j < D_hid; j++) {
        float a = sb1[j];
        const float* w_row = sW1 + (size_t)j * D_in;
        for (int i = 0; i < D_in; i++) a += w_row[i] * x[i];
        h_out[j] = a > 0.f ? a : 0.f;
    }

    // y = W2 · h + b2
    for (int k = 0; k < D_out; k++) {
        float a = sb2[k];
        const float* w_row = sW2 + (size_t)k * D_hid;
        for (int j = 0; j < D_hid; j++) a += w_row[j] * h_out[j];
        y_out[k] = a;
    }
}

// ─── Backward kernel ─────────────────────────────────────────────────────────

__global__ void mlp_backward_kernel(
    const float* __restrict__ features,     // [N, D_in]
    const float* __restrict__ hidden_cache, // [N, D_hid]  post-ReLU from forward
    const float* __restrict__ W1,           // [D_hid, D_in]
    const float* __restrict__ W2,           // [D_out, D_hid]
    const float* __restrict__ grad_output,  // [N, D_out]
    float*       __restrict__ grad_features,// [N, D_in]   (written, not accumulated)
    float*       __restrict__ grad_W1,      // [D_hid, D_in]  (accumulated globally)
    float*       __restrict__ grad_b1,      // [D_hid]        (accumulated globally)
    float*       __restrict__ grad_W2,      // [D_out, D_hid] (accumulated globally)
    float*       __restrict__ grad_b2,      // [D_out]        (accumulated globally)
    int N, int D_in, int D_hid, int D_out
) {
    // Shared memory layout:
    //   sW1   : D_hid * D_in    (read-only weight, loaded once)
    //   sW2   : D_out * D_hid   (read-only weight, loaded once)
    //   sg_W1 : D_hid * D_in    (block-local accumulator → flushed at end)
    //   sg_W2 : D_out * D_hid   (block-local accumulator → flushed at end)
    //   sg_b1 : D_hid           (block-local accumulator)
    //   sg_b2 : D_out           (block-local accumulator)
    extern __shared__ float mlp_smem[];
    float* sW1   = mlp_smem;
    float* sW2   = sW1   + (size_t)D_hid * D_in;
    float* sg_W1 = sW2   + (size_t)D_out * D_hid;
    float* sg_W2 = sg_W1 + (size_t)D_hid * D_in;
    float* sg_b1 = sg_W2 + (size_t)D_out * D_hid;
    float* sg_b2 = sg_b1 + D_hid;

    int tid = threadIdx.x;
    int block_threads = blockDim.x;

    // Load read-only weights.
    for (int i = tid; i < D_hid * D_in;  i += block_threads) sW1[i] = W1[i];
    for (int i = tid; i < D_out * D_hid; i += block_threads) sW2[i] = W2[i];
    // Zero block-local gradient accumulators.
    for (int i = tid; i < D_hid * D_in;  i += block_threads) sg_W1[i] = 0.f;
    for (int i = tid; i < D_out * D_hid; i += block_threads) sg_W2[i] = 0.f;
    for (int i = tid; i < D_hid;         i += block_threads) sg_b1[i] = 0.f;
    for (int i = tid; i < D_out;         i += block_threads) sg_b2[i] = 0.f;
    __syncthreads();

    int n = blockIdx.x * block_threads + tid;
    bool active = (n < N);

    // Local D_hid-vector for v_h. Live in registers if D_hid is small; we
    // fall back to a fixed-size array. 128 is enough for our planned 64.
    constexpr int D_HID_MAX = 128;
    float v_h[D_HID_MAX];

    if (active) {
        const float* x   = features     + (size_t)n * D_in;
        const float* h   = hidden_cache + (size_t)n * D_hid;
        const float* v_y = grad_output  + (size_t)n * D_out;

        // v_b2 and v_W2 accumulate from this thread's contribution.
        // We accumulate into shared via atomicAdd because multiple threads
        // in the block touch the same parameter cells. Atomic on shared
        // memory is cheap relative to global atomic.
        for (int k = 0; k < D_out; k++) {
            atomicAdd(&sg_b2[k], v_y[k]);
            float vy = v_y[k];
            float* w_row = sg_W2 + (size_t)k * D_hid;
            for (int j = 0; j < D_hid; j++) {
                atomicAdd(&w_row[j], vy * h[j]);
            }
        }

        // v_h = W2^T · v_y, gated by ReLU mask (h > 0).
        for (int j = 0; j < D_hid; j++) {
            float a = 0.f;
            for (int k = 0; k < D_out; k++) {
                a += sW2[(size_t)k * D_hid + j] * v_y[k];
            }
            float relu_grad = h[j] > 0.f ? 1.f : 0.f;
            v_h[j] = a * relu_grad;
        }

        // v_b1 and v_W1.
        for (int j = 0; j < D_hid; j++) {
            atomicAdd(&sg_b1[j], v_h[j]);
            float vh = v_h[j];
            float* w_row = sg_W1 + (size_t)j * D_in;
            for (int i = 0; i < D_in; i++) {
                atomicAdd(&w_row[i], vh * x[i]);
            }
        }

        // v_x = W1^T · v_h. Per-Gaussian so a plain write is fine.
        float* v_x = grad_features + (size_t)n * D_in;
        for (int i = 0; i < D_in; i++) {
            float a = 0.f;
            for (int j = 0; j < D_hid; j++) {
                a += sW1[(size_t)j * D_in + i] * v_h[j];
            }
            v_x[i] = a;
        }
    }

    __syncthreads();

    // Flush block-local accumulators to global gradient buffers. One
    // global atomicAdd per parameter per block instead of per (parameter,
    // Gaussian).
    for (int i = tid; i < D_hid * D_in;  i += block_threads)
        if (sg_W1[i] != 0.f) atomicAdd(&grad_W1[i], sg_W1[i]);
    for (int i = tid; i < D_out * D_hid; i += block_threads)
        if (sg_W2[i] != 0.f) atomicAdd(&grad_W2[i], sg_W2[i]);
    for (int i = tid; i < D_hid;         i += block_threads)
        if (sg_b1[i] != 0.f) atomicAdd(&grad_b1[i], sg_b1[i]);
    for (int i = tid; i < D_out;         i += block_threads)
        if (sg_b2[i] != 0.f) atomicAdd(&grad_b2[i], sg_b2[i]);
}

// ─── Host launch helpers ─────────────────────────────────────────────────────

inline size_t mlp_forward_smem_bytes(int D_in, int D_hid, int D_out) {
    return (size_t)((D_hid * D_in) + D_hid + (D_out * D_hid) + D_out) * sizeof(float);
}

inline size_t mlp_backward_smem_bytes(int D_in, int D_hid, int D_out) {
    // 2 × (W1 + W2) for read-only + accumulator, plus b1 + b2 accumulators.
    return (size_t)((D_hid * D_in) * 2 + (D_out * D_hid) * 2 + D_hid + D_out) * sizeof(float);
}

inline void launch_mlp_fwd(
    const float* d_features,
    const float* d_W1, const float* d_b1,
    const float* d_W2, const float* d_b2,
    float* d_hidden_cache,
    float* d_output,
    int N, int D_in, int D_hid, int D_out,
    int threads = 128
) {
    if (N <= 0) return;
    int blocks = (N + threads - 1) / threads;
    size_t smem = mlp_forward_smem_bytes(D_in, D_hid, D_out);
    mlp_forward_kernel<<<blocks, threads, smem>>>(
        d_features, d_W1, d_b1, d_W2, d_b2,
        d_hidden_cache, d_output,
        N, D_in, D_hid, D_out
    );
    MLP_CUDA_CHECK(cudaGetLastError());
}

inline void launch_mlp_bwd(
    const float* d_features,
    const float* d_hidden_cache,
    const float* d_W1, const float* d_W2,
    const float* d_grad_output,
    float* d_grad_features,
    float* d_grad_W1, float* d_grad_b1,
    float* d_grad_W2, float* d_grad_b2,
    int N, int D_in, int D_hid, int D_out,
    int threads = 128
) {
    if (N <= 0) return;
    int blocks = (N + threads - 1) / threads;
    size_t smem = mlp_backward_smem_bytes(D_in, D_hid, D_out);
    mlp_backward_kernel<<<blocks, threads, smem>>>(
        d_features, d_hidden_cache, d_W1, d_W2, d_grad_output,
        d_grad_features, d_grad_W1, d_grad_b1, d_grad_W2, d_grad_b2,
        N, D_in, D_hid, D_out
    );
    MLP_CUDA_CHECK(cudaGetLastError());
}

// ─── Standalone smoke test ────────────────────────────────────────────────────

#ifndef INCLUDED_AS_HEADER
#include <vector>
#include <cmath>

int main() {
    constexpr int N = 5;
    constexpr int D_in = 8;
    constexpr int D_hid = 16;
    constexpr int D_out = 3;

    std::vector<float> h_x(N * D_in), h_W1(D_hid * D_in), h_b1(D_hid),
                       h_W2(D_out * D_hid), h_b2(D_out);
    for (int i = 0; i < N * D_in;   i++) h_x[i]  = 0.1f * std::sin(0.3f * i + 0.4f);
    for (int i = 0; i < D_hid*D_in; i++) h_W1[i] = 0.2f * std::sin(1.3f * i + 0.1f);
    for (int i = 0; i < D_hid;      i++) h_b1[i] = 0.05f * std::cos(0.7f * i);
    for (int i = 0; i < D_out*D_hid;i++) h_W2[i] = 0.2f * std::cos(0.9f * i + 0.2f);
    for (int i = 0; i < D_out;      i++) h_b2[i] = 0.02f * (i + 1);

    float *d_x, *d_W1, *d_b1, *d_W2, *d_b2, *d_h, *d_y;
    MLP_CUDA_CHECK(cudaMalloc(&d_x,  N * D_in   * sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_W1, D_hid*D_in * sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_b1, D_hid      * sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_W2, D_out*D_hid* sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_b2, D_out      * sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_h,  N * D_hid  * sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_y,  N * D_out  * sizeof(float)));

    MLP_CUDA_CHECK(cudaMemcpy(d_x,  h_x.data(),  h_x.size() * sizeof(float),  cudaMemcpyHostToDevice));
    MLP_CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(), h_W1.size()* sizeof(float), cudaMemcpyHostToDevice));
    MLP_CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size()* sizeof(float), cudaMemcpyHostToDevice));
    MLP_CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(), h_W2.size()* sizeof(float), cudaMemcpyHostToDevice));
    MLP_CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size()* sizeof(float), cudaMemcpyHostToDevice));

    launch_mlp_fwd(d_x, d_W1, d_b1, d_W2, d_b2, d_h, d_y, N, D_in, D_hid, D_out);
    MLP_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_y(N * D_out);
    MLP_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    printf("mlp forward smoke (N=%d, D_in=%d, D_hid=%d, D_out=%d):\n",
           N, D_in, D_hid, D_out);
    for (int n = 0; n < N; n++) {
        printf("  n=%d  y=(", n);
        for (int k = 0; k < D_out; k++)
            printf("%s%.4f", k ? ", " : "", h_y[n*D_out+k]);
        printf(")\n");
    }

    // Backward smoke: zero gradients, run, check that something accumulates.
    float *d_grad_y, *d_grad_x, *d_grad_W1, *d_grad_b1, *d_grad_W2, *d_grad_b2;
    MLP_CUDA_CHECK(cudaMalloc(&d_grad_y,  N * D_out * sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_grad_x,  N * D_in  * sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_grad_W1, D_hid*D_in * sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_grad_b1, D_hid     * sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_grad_W2, D_out*D_hid* sizeof(float)));
    MLP_CUDA_CHECK(cudaMalloc(&d_grad_b2, D_out     * sizeof(float)));

    std::vector<float> h_grad_y(N * D_out, 1.f);   // upstream grad of 1 everywhere
    MLP_CUDA_CHECK(cudaMemcpy(d_grad_y, h_grad_y.data(), h_grad_y.size()*sizeof(float), cudaMemcpyHostToDevice));
    MLP_CUDA_CHECK(cudaMemset(d_grad_W1, 0, D_hid*D_in * sizeof(float)));
    MLP_CUDA_CHECK(cudaMemset(d_grad_b1, 0, D_hid     * sizeof(float)));
    MLP_CUDA_CHECK(cudaMemset(d_grad_W2, 0, D_out*D_hid* sizeof(float)));
    MLP_CUDA_CHECK(cudaMemset(d_grad_b2, 0, D_out     * sizeof(float)));

    launch_mlp_bwd(d_x, d_h, d_W1, d_W2, d_grad_y,
                   d_grad_x, d_grad_W1, d_grad_b1, d_grad_W2, d_grad_b2,
                   N, D_in, D_hid, D_out);
    MLP_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_grad_b2(D_out);
    MLP_CUDA_CHECK(cudaMemcpy(h_grad_b2.data(), d_grad_b2, D_out*sizeof(float), cudaMemcpyDeviceToHost));
    printf("backward smoke: grad_b2 = (");
    for (int k = 0; k < D_out; k++) printf("%s%.3f", k ? ", " : "", h_grad_b2[k]);
    printf(")  (expected each = N = %d)\n", N);

    bool ok = true;
    for (int k = 0; k < D_out; k++) {
        if (std::abs(h_grad_b2[k] - (float)N) > 1e-3f) ok = false;
    }
    cudaFree(d_x); cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_h); cudaFree(d_y);
    cudaFree(d_grad_y); cudaFree(d_grad_x);
    cudaFree(d_grad_W1); cudaFree(d_grad_b1); cudaFree(d_grad_W2); cudaFree(d_grad_b2);

    printf("%s\n", ok ? "mlp smoke OK" : "mlp smoke FAILED");
    return ok ? 0 : 1;
}
#endif
