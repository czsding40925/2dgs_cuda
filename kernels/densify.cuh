#pragma once
// Densification and opacity-reset logic.
//
// Included directly into train.cu after Config, SplatData, SplatAdam,
// ForwardBuffers, and sh_coeffs_per_channel are in scope.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

struct DensifyState {
    float* grad_accum = nullptr;   // [N]  accumulated |d_loss/d_x2d| + |d_loss/d_y2d|
    int*   count      = nullptr;   // [N]  visibility count (frames where radii > 0)
    int    N_cap = 0;
    int    accum_steps = 0;        // total forward passes since last reset (guard: skip if 0)
};

static void alloc_densify_state(DensifyState& s, int N) {
    s.N_cap = N;
    s.accum_steps = 0;
    if (N > 0) {
        CUDA_CHECK(cudaMalloc(&s.grad_accum, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(s.grad_accum, 0, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.count, N * sizeof(int)));
        CUDA_CHECK(cudaMemset(s.count, 0, N * sizeof(int)));
    }
}

static void free_densify_state(DensifyState& s) {
    cudaFree(s.grad_accum);
    cudaFree(s.count);
    s = {};
}

__global__ static void accumulate_grad_means2d_abs_kernel(
    const float* __restrict__ grad_means2d_abs,
    float* __restrict__ grad_accum,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float gx = grad_means2d_abs[idx * 2 + 0];
    float gy = grad_means2d_abs[idx * 2 + 1];
    grad_accum[idx] += gx + gy;
}

// Increments count[i] for each Gaussian that was visible (both radii > 0).
// Used for per-Gaussian gradient averaging (mirrors gsplat's per-Gaussian count).
__global__ static void accumulate_visibility_count_kernel(
    const int32_t* __restrict__ radii,   // [N, 2]  (rx, ry per Gaussian)
    int* __restrict__ count,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (radii[idx * 2 + 0] > 0 && radii[idx * 2 + 1] > 0)
        count[idx]++;
}

__global__ static void clamp_opacity_logits_kernel(
    float* __restrict__ opacity_logits,
    float max_logit,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    opacity_logits[idx] = fminf(opacity_logits[idx], max_logit);
}

struct HostSplatState {
    std::vector<float> means;
    std::vector<float> rotation;
    std::vector<float> scaling;
    std::vector<float> opacity;
    std::vector<float> sh0;
    std::vector<float> shN;
};

static HostSplatState download_splats(const SplatData& splats) {
    HostSplatState host;
    const int N = splats.N();
    const int shn_dim = sh_coeffs_per_channel(splats.max_sh_degree()) * 3;
    host.means.resize(N * 3);
    host.rotation.resize(N * 4);
    host.scaling.resize(N * 3);
    host.opacity.resize(N);
    host.sh0.resize(N * 3);
    host.shN.resize(N * shn_dim);

    if (N > 0) {
        CUDA_CHECK(cudaMemcpy(host.means.data(), splats.means(),
                              N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.rotation.data(), splats.rotation(),
                              N * 4 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.scaling.data(), splats.scaling(),
                              N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.opacity.data(), splats.opacity(),
                              N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.sh0.data(), splats.sh0(),
                              N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        if (shn_dim > 0) {
            CUDA_CHECK(cudaMemcpy(host.shN.data(), splats.shN(),
                                  N * shn_dim * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    return host;
}

static void upload_splats(SplatData& splats, const HostSplatState& host) {
    const int N = (int)host.opacity.size();
    const int shn_dim = sh_coeffs_per_channel(splats.max_sh_degree()) * 3;
    splats.resize(N);
    if (N <= 0) return;

    CUDA_CHECK(cudaMemcpy(splats.means(), host.means.data(),
                          N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(splats.rotation(), host.rotation.data(),
                          N * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(splats.scaling(), host.scaling.data(),
                          N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(splats.opacity(), host.opacity.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(splats.sh0(), host.sh0.data(),
                          N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    if (shn_dim > 0 && !host.shN.empty()) {
        CUDA_CHECK(cudaMemcpy(splats.shN(), host.shN.data(),
                              N * shn_dim * sizeof(float), cudaMemcpyHostToDevice));
    }
}

static void resize_densify_state(DensifyState& s, int N) {
    free_densify_state(s);
    alloc_densify_state(s, N);
}

static float logistic(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static float inverse_logistic(float y) {
    y = std::max(1e-6f, std::min(1.f - 1e-6f, y));
    return std::log(y / (1.f - y));
}

static void quat_to_rotmat_host(const float* q_raw, float R[9]) {
    float w = q_raw[0];
    float x = q_raw[1];
    float y = q_raw[2];
    float z = q_raw[3];
    float norm = std::sqrt(w*w + x*x + y*y + z*z);
    if (norm <= 1e-12f) {
        R[0] = 1.f; R[1] = 0.f; R[2] = 0.f;
        R[3] = 0.f; R[4] = 1.f; R[5] = 0.f;
        R[6] = 0.f; R[7] = 0.f; R[8] = 1.f;
        return;
    }
    w /= norm; x /= norm; y /= norm; z /= norm;

    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    R[0] = 1.f - 2.f * (y2 + z2);
    R[1] = 2.f * (xy - wz);
    R[2] = 2.f * (xz + wy);
    R[3] = 2.f * (xy + wz);
    R[4] = 1.f - 2.f * (x2 + z2);
    R[5] = 2.f * (yz - wx);
    R[6] = 2.f * (xz - wy);
    R[7] = 2.f * (yz + wx);
    R[8] = 1.f - 2.f * (x2 + y2);
}

static bool maybe_densify(
    const Config& cfg,
    int iter,
    float scene_scale,
    SplatData& splats,
    SplatAdam& optimizer,
    ForwardBuffers& fwd,
    DensifyState& densify,
    int max_pixels
) {
    if (cfg.densify_every <= 0) return false;
    if (iter < cfg.densify_start) return false;
    if (cfg.densify_stop > 0 && iter > cfg.densify_stop) return false;
    if (iter % cfg.densify_every != 0) return false;
    if (densify.accum_steps <= 0) return false;

    const int N = splats.N();
    if (N <= 0) return false;
    if (cfg.max_gaussians > 0 && N >= cfg.max_gaussians) return false;

    HostSplatState host = download_splats(splats);
    std::vector<float> grad_accum(N, 0.f);
    std::vector<int>   vis_count(N, 0);
    CUDA_CHECK(cudaMemcpy(grad_accum.data(), densify.grad_accum,
                          N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vis_count.data(), densify.count,
                          N * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<float> grad_avg(N, 0.f);
    std::vector<float> tangent_max_scale(N, 0.f);
    std::vector<float> full_max_scale(N, 0.f);
    std::vector<int> keep_rows;
    std::vector<int> clone_candidates;
    std::vector<int> split_candidates;
    keep_rows.reserve(N);

    double grad_sum = 0.0;
    std::vector<float> kept_scales;
    kept_scales.reserve(N);
    float max_grad_avg = 0.f;
    for (int i = 0; i < N; i++) {
        float alpha = logistic(host.opacity[i]);
        // Divide by per-Gaussian visibility count so Gaussians visible in
        // only a fraction of steps don't get underweighted (mirrors gsplat).
        grad_avg[i] = (vis_count[i] > 0)
            ? grad_accum[i] / (float)vis_count[i]
            : 0.f;
        max_grad_avg = std::max(max_grad_avg, grad_avg[i]);

        float sx = std::exp(host.scaling[i * 3 + 0]);
        float sy = std::exp(host.scaling[i * 3 + 1]);
        float sz = std::exp(host.scaling[i * 3 + 2]);
        tangent_max_scale[i] = std::max(sx, sy);
        full_max_scale[i] = std::max(tangent_max_scale[i], sz);

        bool prune_alpha = alpha < cfg.densify_prune_alpha;
        bool prune_big = (iter > cfg.opacity_reset_every && cfg.densify_prune_scale3d > 0.f &&
                          full_max_scale[i] > cfg.densify_prune_scale3d * scene_scale);
        if (prune_alpha || prune_big) continue;

        keep_rows.push_back(i);
        grad_sum += grad_avg[i];
        kept_scales.push_back(tangent_max_scale[i]);
    }

    if (keep_rows.empty()) {
        keep_rows.push_back(0);
        grad_avg[0] = (vis_count[0] > 0) ? grad_accum[0] / (float)vis_count[0] : 0.f;
        float sx = std::exp(host.scaling[0]);
        float sy = std::exp(host.scaling[1]);
        tangent_max_scale[0] = std::max(sx, sy);
        kept_scales.push_back(tangent_max_scale[0]);
        grad_sum += grad_avg[0];
    }

    float mean_grad = (float)(grad_sum / keep_rows.size());
    // Threshold = mean_grad * densify_grad_mult (default 1.5).
    // Only Gaussians in the top fraction by gradient trigger densification.
    float grad_threshold = (cfg.densify_grad_thresh > 0.f)
        ? cfg.densify_grad_thresh
        : (mean_grad * cfg.densify_grad_mult);
    std::nth_element(kept_scales.begin(),
                     kept_scales.begin() + kept_scales.size() / 2,
                     kept_scales.end());
    float median_scale = kept_scales[kept_scales.size() / 2];
    float grow_scale_limit = cfg.densify_grow_scale3d * scene_scale;

    for (int src : keep_rows) {
        if (grad_avg[src] <= grad_threshold)
            continue;
        if (tangent_max_scale[src] <= grow_scale_limit)
            clone_candidates.push_back(src);
        else
            split_candidates.push_back(src);
    }

    // Respect the max_gaussians ceiling: trim candidates so new_N stays at or below it.
    if (cfg.max_gaussians > 0) {
        int n_keep = (int)keep_rows.size();
        int budget = std::max(0, cfg.max_gaussians - n_keep);
        // Allocate budget proportionally between clones and splits.
        int n_clone = std::min((int)clone_candidates.size(), budget / 2 + budget % 2);
        int n_split = std::min((int)split_candidates.size(), budget - n_clone);
        clone_candidates.resize(n_clone);
        split_candidates.resize(n_split);
    }

    int pruned_count = N - (int)keep_rows.size();
    int new_N = (int)keep_rows.size() + (int)clone_candidates.size() + (int)split_candidates.size();
    if (new_N == N) {
        CUDA_CHECK(cudaMemset(densify.grad_accum, 0, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(densify.count,      0, N * sizeof(int)));
        densify.accum_steps = 0;
        return false;
    }

    const int shn_dim = sh_coeffs_per_channel(splats.max_sh_degree()) * 3;
    HostSplatState next;
    next.means.resize(new_N * 3);
    next.rotation.resize(new_N * 4);
    next.scaling.resize(new_N * 3);
    next.opacity.resize(new_N);
    next.sh0.resize(new_N * 3);
    next.shN.resize(new_N * shn_dim);

    std::vector<int> moment_src_rows;
    moment_src_rows.reserve(new_N);
    std::vector<char> is_split_parent(N, 0);
    for (int src : split_candidates)
        is_split_parent[src] = 1;

    auto copy_row = [&](int dst_row, int src_row) {
        std::copy_n(host.means.data() + src_row * 3, 3, next.means.data() + dst_row * 3);
        std::copy_n(host.rotation.data() + src_row * 4, 4, next.rotation.data() + dst_row * 4);
        std::copy_n(host.scaling.data() + src_row * 3, 3, next.scaling.data() + dst_row * 3);
        next.opacity[dst_row] = host.opacity[src_row];
        std::copy_n(host.sh0.data() + src_row * 3, 3, next.sh0.data() + dst_row * 3);
        if (shn_dim > 0)
            std::copy_n(host.shN.data() + src_row * shn_dim, shn_dim,
                        next.shN.data() + dst_row * shn_dim);
    };

    int dst = 0;
    for (int src : keep_rows) {
        if (is_split_parent[src]) continue;
        copy_row(dst, src);
        moment_src_rows.push_back(src);
        dst++;
    }
    for (int src : clone_candidates) {
        copy_row(dst, src);
        moment_src_rows.push_back(-1);
        dst++;
    }
    for (int src : split_candidates) {
        float R[9];
        quat_to_rotmat_host(host.rotation.data() + src * 4, R);

        float sx = std::exp(host.scaling[src * 3 + 0]);
        float sy = std::exp(host.scaling[src * 3 + 1]);
        int split_axis = (sx >= sy) ? 0 : 1;
        int other_axis = 1 - split_axis;

        float axis_vec[3] = {
            R[split_axis + 0],
            R[split_axis + 3],
            R[split_axis + 6]
        };
        float axis_scale = std::exp(host.scaling[src * 3 + split_axis]);
        float offset_mag = 0.5f * axis_scale;

        float new_scaling[3] = {
            host.scaling[src * 3 + 0],
            host.scaling[src * 3 + 1],
            host.scaling[src * 3 + 2]
        };
        new_scaling[split_axis] += std::log(0.5f);
        new_scaling[other_axis] += std::log(0.85f);

        float new_alpha = logistic(host.opacity[src]) * 0.6f;
        float new_opacity = inverse_logistic(new_alpha);

        for (int child = 0; child < 2; child++) {
            copy_row(dst, src);
            float sign = (child == 0) ? 1.f : -1.f;
            next.means[dst * 3 + 0] += sign * axis_vec[0] * offset_mag;
            next.means[dst * 3 + 1] += sign * axis_vec[1] * offset_mag;
            next.means[dst * 3 + 2] += sign * axis_vec[2] * offset_mag;
            next.scaling[dst * 3 + 0] = new_scaling[0];
            next.scaling[dst * 3 + 1] = new_scaling[1];
            next.scaling[dst * 3 + 2] = new_scaling[2];
            next.opacity[dst] = new_opacity;
            moment_src_rows.push_back(-1);
            dst++;
        }
    }

    if (dst != new_N)
        throw std::runtime_error("densify: output row count mismatch");

    splats.reserve(new_N);
    upload_splats(splats, next);
    optimizer.remap_rows(moment_src_rows);
    resize_densify_state(densify, new_N);
    ensure_forward_buffers(fwd, new_N, max_pixels, splats.max_sh_degree());

    densify.accum_steps = 0;

    printf("\ndensify: iter=%d  old_N=%d  pruned=%d  cloned=%zu  split=%zu  new_N=%d  mean_grad=%.6g  max_grad=%.6g  grad_thresh=%.6g  grow_scale=%.6g  median_scale=%.6g\n",
           iter, N, pruned_count, clone_candidates.size(), split_candidates.size(),
           new_N, mean_grad, max_grad_avg, grad_threshold, grow_scale_limit, median_scale);
    return true;
}

static void maybe_reset_opacity(
    const Config& cfg,
    int iter,
    SplatData& splats,
    SplatAdam& optimizer
) {
    if (cfg.opacity_reset_every <= 0) return;
    if (iter % cfg.opacity_reset_every != 0) return;

    float reset_alpha = std::min(0.999f, cfg.densify_prune_alpha * 2.f);
    float max_logit = inverse_logistic(reset_alpha);
    int N = splats.N();
    int blocks = (N + 255) / 256;
    clamp_opacity_logits_kernel<<<blocks, 256>>>(splats.opacity(), max_logit, N);
    CUDA_CHECK(cudaGetLastError());
    optimizer.opacity.zero();

    printf("\nopacity-reset: iter=%d  max_alpha=%.4f\n", iter, reset_alpha);
}
