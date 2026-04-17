// adam.cu
//
// CUDA Adam optimizer for raw SplatData arrays.
//
// This mirrors gsplat's simple per-element Adam kernel, adapted for this
// C++/CUDA-first project:
//   - parameters and gradients are plain float* device buffers
//   - optional valid mask skips whole Gaussians, useful after pruning
//   - moment buffers are owned by small RAII structs
//   - SplatAdam groups SplatData fields with separate learning rates

#include "splat_data.cuh"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Device kernel
// ─────────────────────────────────────────────────────────────────────────────

__global__ void adam_step_kernel(
    const int      N,              // number of Gaussians / rows
    const int      D,              // values per Gaussian / row
    float*         __restrict__ param,
    const float*   __restrict__ grad,
    float*         __restrict__ exp_avg,
    float*         __restrict__ exp_avg_sq,
    const bool*    __restrict__ valid,  // optional [N]
    const float    lr,
    const float    beta1,
    const float    beta2,
    const float    eps
) {
    int p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = N * D;
    if (p_idx >= numel) return;

    int g_idx = p_idx / D;
    if (valid != nullptr && !valid[g_idx]) return;

    float g = grad[p_idx];
    float m = exp_avg[p_idx];
    float v = exp_avg_sq[p_idx];

    m = beta1 * m + (1.f - beta1) * g;
    v = beta2 * v + (1.f - beta2) * g * g;

    param[p_idx] -= lr * m / (sqrtf(v) + eps);

    exp_avg[p_idx] = m;
    exp_avg_sq[p_idx] = v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launch helpers
// ─────────────────────────────────────────────────────────────────────────────

struct AdamConfig {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-15f;  // matches gsplat/PyTorch 3DGS convention; 1e-8 over-damps updates
    bool  bias_correction = true;

    // Per-parameter-group learning rates. These match the usual 3DGS ordering:
    // means move slowly, opacity/scales can move faster, higher-order SH slower.
    float lr_means    = 1.6e-4f;
    float lr_rotation = 1.0e-3f;
    float lr_scaling  = 5.0e-3f;
    float lr_opacity  = 5.0e-2f;
    float lr_sh0      = 2.5e-3f;
    float lr_shN      = 1.25e-4f;
};

static inline float adam_effective_lr(float lr, int step, const AdamConfig& cfg) {
    if (!cfg.bias_correction || step <= 0) return lr;
    float bc1 = 1.f - std::pow(cfg.beta1, (float)step);
    float bc2 = 1.f - std::pow(cfg.beta2, (float)step);
    return lr * std::sqrt(bc2) / bc1;
}

void launch_adam_step(
    float*       d_param,
    const float* d_grad,
    float*       d_exp_avg,
    float*       d_exp_avg_sq,
    int          N,
    int          D,
    float        lr,
    const AdamConfig& cfg,
    int          step,
    const bool*  d_valid = nullptr
) {
    if (N <= 0 || D <= 0 || d_param == nullptr || d_grad == nullptr) return;

    int numel = N * D;
    dim3 threads(256);
    dim3 grid((numel + threads.x - 1) / threads.x);
    float lr_t = adam_effective_lr(lr, step, cfg);

    adam_step_kernel<<<grid, threads>>>(
        N, D, d_param, d_grad, d_exp_avg, d_exp_avg_sq, d_valid,
        lr_t, cfg.beta1, cfg.beta2, cfg.eps
    );
    CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// Moment-buffer ownership
// ─────────────────────────────────────────────────────────────────────────────

struct AdamMomentBuffer {
    float* exp_avg = nullptr;
    float* exp_avg_sq = nullptr;
    int    numel = 0;

    AdamMomentBuffer() = default;
    explicit AdamMomentBuffer(int n) { allocate(n); }
    ~AdamMomentBuffer() { free(); }

    AdamMomentBuffer(const AdamMomentBuffer&) = delete;
    AdamMomentBuffer& operator=(const AdamMomentBuffer&) = delete;

    AdamMomentBuffer(AdamMomentBuffer&& o) noexcept { steal(o); }
    AdamMomentBuffer& operator=(AdamMomentBuffer&& o) noexcept {
        if (this != &o) { free(); steal(o); }
        return *this;
    }

    void allocate(int n) {
        free();
        numel = n;
        if (numel <= 0) return;
        CUDA_CHECK(cudaMalloc(&exp_avg,    numel * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&exp_avg_sq, numel * sizeof(float)));
        zero();
    }

    void zero() {
        if (numel <= 0) return;
        CUDA_CHECK(cudaMemset(exp_avg,    0, numel * sizeof(float)));
        CUDA_CHECK(cudaMemset(exp_avg_sq, 0, numel * sizeof(float)));
    }

    void free() {
        cudaFree(exp_avg);    exp_avg = nullptr;
        cudaFree(exp_avg_sq); exp_avg_sq = nullptr;
        numel = 0;
    }

    void upload(const std::vector<float>& host_exp_avg,
                const std::vector<float>& host_exp_avg_sq) {
        if (host_exp_avg.size() != host_exp_avg_sq.size())
            throw std::runtime_error("AdamMomentBuffer::upload size mismatch");
        allocate((int)host_exp_avg.size());
        if (numel <= 0) return;
        CUDA_CHECK(cudaMemcpy(exp_avg, host_exp_avg.data(),
                              numel * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(exp_avg_sq, host_exp_avg_sq.data(),
                              numel * sizeof(float), cudaMemcpyHostToDevice));
    }

    void download(std::vector<float>& host_exp_avg,
                  std::vector<float>& host_exp_avg_sq) const {
        host_exp_avg.assign(numel, 0.f);
        host_exp_avg_sq.assign(numel, 0.f);
        if (numel <= 0) return;
        CUDA_CHECK(cudaMemcpy(host_exp_avg.data(), exp_avg,
                              numel * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_exp_avg_sq.data(), exp_avg_sq,
                              numel * sizeof(float), cudaMemcpyDeviceToHost));
    }

private:
    void steal(AdamMomentBuffer& o) {
        exp_avg = o.exp_avg;       o.exp_avg = nullptr;
        exp_avg_sq = o.exp_avg_sq; o.exp_avg_sq = nullptr;
        numel = o.numel;           o.numel = 0;
    }
};

struct SplatGradients {
    float* means = nullptr;     // [N, 3]
    float* rotation = nullptr;  // [N, 4]
    float* scaling = nullptr;   // [N, 3]
    float* opacity = nullptr;   // [N]
    float* sh0 = nullptr;       // [N, 3]
    float* shN = nullptr;       // [N, K*3], optional
};

class SplatAdam {
public:
    SplatAdam() = default;
    explicit SplatAdam(const SplatData& splats) { allocate(splats.N(), splats.max_sh_degree()); }

    void allocate(int N, int max_sh_degree) {
        _N = N;
        _max_sh_degree = max_sh_degree;
        _step = 0;
        means.allocate(N * 3);
        rotation.allocate(N * 4);
        scaling.allocate(N * 3);
        opacity.allocate(N);
        sh0.allocate(N * 3);

        int shn_dim = sh_coeffs_per_channel(max_sh_degree) * 3;
        shN.allocate(N * shn_dim);
    }

    int step_count() const { return _step; }
    int N() const { return _N; }

    void step(SplatData& splats,
              const SplatGradients& grads,
              const AdamConfig& cfg,
              const bool* d_valid = nullptr) {
        ++_step;
        step_group(splats.means(),    grads.means,    means,    _N, 3, cfg.lr_means,    cfg, d_valid);
        step_group(splats.rotation(), grads.rotation, rotation, _N, 4, cfg.lr_rotation, cfg, d_valid);
        step_group(splats.scaling(),  grads.scaling,  scaling,  _N, 3, cfg.lr_scaling,  cfg, d_valid);
        step_group(splats.opacity(),  grads.opacity,  opacity,  _N, 1, cfg.lr_opacity,  cfg, d_valid);
        step_group(splats.sh0(),      grads.sh0,      sh0,      _N, 3, cfg.lr_sh0,      cfg, d_valid);

        int shn_dim = sh_coeffs_per_channel(_max_sh_degree) * 3;
        if (shn_dim > 0) {
            step_group(splats.shN(), grads.shN, shN, _N, shn_dim, cfg.lr_shN, cfg, d_valid);
        }
    }

    void remap_rows(const std::vector<int>& src_rows) {
        remap_group(means, src_rows, 3);
        remap_group(rotation, src_rows, 4);
        remap_group(scaling, src_rows, 3);
        remap_group(opacity, src_rows, 1);
        remap_group(sh0, src_rows, 3);

        int shn_dim = sh_coeffs_per_channel(_max_sh_degree) * 3;
        if (shn_dim > 0)
            remap_group(shN, src_rows, shn_dim);

        _N = (int)src_rows.size();
    }

    AdamMomentBuffer means, rotation, scaling, opacity, sh0, shN;

private:
    int _N = 0;
    int _max_sh_degree = 0;
    int _step = 0;

    void step_group(float* param,
                    const float* grad,
                    AdamMomentBuffer& moment,
                    int N,
                    int D,
                    float lr,
                    const AdamConfig& cfg,
                    const bool* d_valid) {
        if (param == nullptr || grad == nullptr || moment.numel == 0) return;
        launch_adam_step(param, grad, moment.exp_avg, moment.exp_avg_sq,
                         N, D, lr, cfg, _step, d_valid);
    }

    void remap_group(AdamMomentBuffer& moment,
                     const std::vector<int>& src_rows,
                     int D) {
        std::vector<float> old_exp_avg;
        std::vector<float> old_exp_avg_sq;
        moment.download(old_exp_avg, old_exp_avg_sq);

        std::vector<float> new_exp_avg(src_rows.size() * D, 0.f);
        std::vector<float> new_exp_avg_sq(src_rows.size() * D, 0.f);
        for (size_t row = 0; row < src_rows.size(); row++) {
            int src = src_rows[row];
            if (src < 0 || src >= _N) continue;
            for (int d = 0; d < D; d++) {
                new_exp_avg[row * D + d] = old_exp_avg[src * D + d];
                new_exp_avg_sq[row * D + d] = old_exp_avg_sq[src * D + d];
            }
        }

        moment.upload(new_exp_avg, new_exp_avg_sq);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Standalone test
// ─────────────────────────────────────────────────────────────────────────────

#ifndef INCLUDED_AS_HEADER
int main() {
    printf("=== adam optimizer test ===\n\n");

    const int N = 2;
    const int D = 2;
    const int numel = N * D;

    float h_param[numel] = {1.f, 2.f, 3.f, 4.f};
    float h_grad[numel]  = {1.f, -2.f, 10.f, 10.f};
    bool  h_valid[N]     = {true, false};

    float *d_param, *d_grad;
    bool* d_valid;
    CUDA_CHECK(cudaMalloc(&d_param, numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad,  numel * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, N * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_param, h_param, numel * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad,  h_grad,  numel * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valid, h_valid, N * sizeof(bool), cudaMemcpyHostToDevice));

    AdamMomentBuffer moment(numel);
    AdamConfig cfg;
    cfg.beta1 = 0.f;
    cfg.beta2 = 0.f;
    cfg.eps = 1e-6f;
    cfg.bias_correction = false;

    launch_adam_step(d_param, d_grad, moment.exp_avg, moment.exp_avg_sq,
                     N, D, /*lr=*/0.1f, cfg, /*step=*/1, d_valid);
    CUDA_CHECK(cudaDeviceSynchronize());

    float out[numel];
    CUDA_CHECK(cudaMemcpy(out, d_param, numel * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Updated params:\n");
    for (int i = 0; i < numel; i++) {
        printf("  p[%d] = %.6f\n", i, out[i]);
    }

    bool ok = true;
    ok &= fabsf(out[0] - 0.900000f) < 1e-4f;  // valid row, positive grad
    ok &= fabsf(out[1] - 2.100000f) < 1e-4f;  // valid row, negative grad
    ok &= fabsf(out[2] - 3.000000f) < 1e-6f;  // invalid row unchanged
    ok &= fabsf(out[3] - 4.000000f) < 1e-6f;

    cudaFree(d_param);
    cudaFree(d_grad);
    cudaFree(d_valid);

    printf("\n%s\n", ok ? "All tests passed." : "SOME TESTS FAILED.");
    return ok ? 0 : 1;
}
#endif // INCLUDED_AS_HEADER
