// gradient_checks.cu
//
// Finite-difference validation for the active 2DGS backward path:
//   1. rasterize_bwd.cu
//   2. projection_2dgs_bwd.cu
//   3. SH coefficient backward used in train.cu
//
// The checks are intentionally tiny and deterministic so failures are easy to
// inspect. Each test constructs a scalar objective
//
//   f(x) = <forward_output(x), upstream_gradient>
//
// compares the analytical gradient from the backward kernel against centered
// finite differences, and prints both values.

#define INCLUDED_AS_HEADER
#include "rasterize_fwd.cu"
#include "rasterize_bwd.cu"
#include "projection_2dgs_bwd.cu"
#undef INCLUDED_AS_HEADER

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static constexpr float SH_C0_GC   =  0.28209479177f;
static constexpr float SH_C1_GC   =  0.48860251190f;
static constexpr float SH_C2_0_GC =  1.09254843059f;
static constexpr float SH_C2_1_GC = -1.09254843059f;
static constexpr float SH_C2_2_GC =  0.31539156525f;
static constexpr float SH_C2_3_GC = -1.09254843059f;
static constexpr float SH_C2_4_GC =  0.54627421529f;
static constexpr float SH_C3_0_GC = -0.59004358992f;
static constexpr float SH_C3_1_GC =  2.89061144264f;
static constexpr float SH_C3_2_GC = -0.45704579946f;
static constexpr float SH_C3_3_GC =  0.37317633259f;
static constexpr float SH_C3_4_GC = -0.45704579946f;
static constexpr float SH_C3_5_GC =  1.44530572132f;
static constexpr float SH_C3_6_GC = -0.59004358992f;

__global__ void sh_eval_gradcheck_kernel(
    const float* __restrict__ means,
    const float* __restrict__ sh0,
    const float* __restrict__ shN,
    float cam_x, float cam_y, float cam_z,
    float* __restrict__ colors,
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

    for (int c = 0; c < 3; c++) {
        float v = SH_C0_GC * sh0[i*3 + c];

        if (sh_active >= 1) {
            const float* sh = shN + i*45 + c*15;
            v += SH_C1_GC * (-sh[0]*y + sh[1]*z - sh[2]*x);
            if (sh_active >= 2) {
                float xx = x*x, yy = y*y, zz = z*z;
                v += SH_C2_0_GC * sh[3] * (x*y);
                v += SH_C2_1_GC * sh[4] * (y*z);
                v += SH_C2_2_GC * sh[5] * (2.f*zz - xx - yy);
                v += SH_C2_3_GC * sh[6] * (x*z);
                v += SH_C2_4_GC * sh[7] * (xx - yy);
                if (sh_active >= 3) {
                    v += SH_C3_0_GC * sh[8]  * y * (3.f*xx - yy);
                    v += SH_C3_1_GC * sh[9]  * x * y * z;
                    v += SH_C3_2_GC * sh[10] * y * (4.f*zz - xx - yy);
                    v += SH_C3_3_GC * sh[11] * z * (2.f*zz - 3.f*xx - 3.f*yy);
                    v += SH_C3_4_GC * sh[12] * x * (4.f*zz - xx - yy);
                    v += SH_C3_5_GC * sh[13] * z * (xx - yy);
                    v += SH_C3_6_GC * sh[14] * x * (xx - 3.f*yy);
                }
            }
        }
        colors[i*3 + c] = fmaxf(0.f, fminf(1.f, v + 0.5f));
    }
}

__global__ void sh_backward_gradcheck_kernel(
    const float* __restrict__ means,
    const float* __restrict__ sh0,
    const float* __restrict__ shN,
    float cam_x, float cam_y, float cam_z,
    const float* __restrict__ v_colors,
    float* __restrict__ v_sh0,
    float* __restrict__ v_shN,
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
        float v = SH_C0_GC * sh0[i*3 + c];
        const float* sh = shN + i*45 + c*15;
        if (sh_active >= 1) {
            v += SH_C1_GC * (-sh[0]*y + sh[1]*z - sh[2]*x);
            if (sh_active >= 2) {
                v += SH_C2_0_GC * sh[3] * (x*y);
                v += SH_C2_1_GC * sh[4] * (y*z);
                v += SH_C2_2_GC * sh[5] * (2.f*zz - xx - yy);
                v += SH_C2_3_GC * sh[6] * (x*z);
                v += SH_C2_4_GC * sh[7] * (xx - yy);
                if (sh_active >= 3) {
                    v += SH_C3_0_GC * sh[8]  * y * (3.f*xx - yy);
                    v += SH_C3_1_GC * sh[9]  * x * y * z;
                    v += SH_C3_2_GC * sh[10] * y * (4.f*zz - xx - yy);
                    v += SH_C3_3_GC * sh[11] * z * (2.f*zz - 3.f*xx - 3.f*yy);
                    v += SH_C3_4_GC * sh[12] * x * (4.f*zz - xx - yy);
                    v += SH_C3_5_GC * sh[13] * z * (xx - yy);
                    v += SH_C3_6_GC * sh[14] * x * (xx - 3.f*yy);
                }
            }
        }

        float unclamped = v + 0.5f;
        float* out_sh = v_shN + i*45 + c*15;
        for (int k = 0; k < 15; k++) out_sh[k] = 0.f;

        if (unclamped <= 0.f || unclamped >= 1.f) {
            v_sh0[i*3 + c] = 0.f;
            continue;
        }

        float grad = v_colors[i*3 + c];
        v_sh0[i*3 + c] = grad * SH_C0_GC;

        if (sh_active >= 1) {
            out_sh[0] = grad * (SH_C1_GC * (-y));
            out_sh[1] = grad * (SH_C1_GC * z);
            out_sh[2] = grad * (SH_C1_GC * (-x));
        }
        if (sh_active >= 2) {
            out_sh[3] = grad * (SH_C2_0_GC * (x*y));
            out_sh[4] = grad * (SH_C2_1_GC * (y*z));
            out_sh[5] = grad * (SH_C2_2_GC * (2.f*zz - xx - yy));
            out_sh[6] = grad * (SH_C2_3_GC * (x*z));
            out_sh[7] = grad * (SH_C2_4_GC * (xx - yy));
        }
        if (sh_active >= 3) {
            out_sh[8]  = grad * (SH_C3_0_GC * y * (3.f*xx - yy));
            out_sh[9]  = grad * (SH_C3_1_GC * x * y * z);
            out_sh[10] = grad * (SH_C3_2_GC * y * (4.f*zz - xx - yy));
            out_sh[11] = grad * (SH_C3_3_GC * z * (2.f*zz - 3.f*xx - 3.f*yy));
            out_sh[12] = grad * (SH_C3_4_GC * x * (4.f*zz - xx - yy));
            out_sh[13] = grad * (SH_C3_5_GC * z * (xx - yy));
            out_sh[14] = grad * (SH_C3_6_GC * x * (xx - 3.f*yy));
        }
    }
}

static float dot_host(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.f;
    for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}

static bool check_close(
    const char* label,
    float analytic,
    float finite_diff,
    float abs_tol,
    float rel_tol
) {
    float abs_err = fabsf(analytic - finite_diff);
    float rel_err = abs_err / std::max(std::max(fabsf(analytic), fabsf(finite_diff)), 1e-6f);
    bool ok = abs_err <= abs_tol || rel_err <= rel_tol;
    printf("  %-24s analytic=% .6e  finite_diff=% .6e  abs_err=% .3e  rel_err=% .3e  %s\n",
           label, analytic, finite_diff, abs_err, rel_err, ok ? "OK" : "FAIL");
    return ok;
}

template <typename EvalFn>
static float centered_diff(std::vector<float>& params, int idx, float eps, const EvalFn& eval) {
    float old = params[idx];
    params[idx] = old + eps;
    float fp = eval();
    params[idx] = old - eps;
    float fm = eval();
    params[idx] = old;
    return (fp - fm) / (2.f * eps);
}

static bool run_projection_gradcheck() {
    printf("=== Projection Gradient Check ===\n");

    const int N = 1;
    const int W = 64, H = 64;
    const float fx = 48.f, fy = 46.f, cx = 31.f, cy = 33.f;

    std::vector<float> h_means    = {0.17f, -0.11f, 2.3f};
    std::vector<float> h_rotation = {0.95f, 0.10f, -0.04f, 0.02f};
    std::vector<float> h_scaling  = {-1.15f, -0.82f, 0.0f};
    std::vector<float> h_viewmat  = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };
    std::vector<float> h_v_means2d = {0.31f, -0.27f};
    std::vector<float> h_v_depths  = {0.12f};
    std::vector<float> h_vT = {
         0.07f, -0.03f,  0.11f,
        -0.05f,  0.09f, -0.04f,
         0.02f, -0.08f,  0.06f
    };

    float *d_means, *d_rotation, *d_scaling, *d_viewmat;
    float *d_T, *d_means2d, *d_depths, *d_normals;
    int32_t* d_radii;
    float *d_vT, *d_v_means2d, *d_v_depths;
    float *d_g_means, *d_g_rotation, *d_g_scaling;

    CUDA_CHECK(cudaMalloc(&d_means, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rotation, 4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scaling, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_viewmat, 16*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means2d, 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depths, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normals, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii, 2*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_vT, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_means2d, 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_depths, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_means, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_rotation, 4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_scaling, 3*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_viewmat, h_viewmat.data(), 16*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vT, h_vT.data(), 9*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_means2d, h_v_means2d.data(), 2*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_depths, h_v_depths.data(), sizeof(float), cudaMemcpyHostToDevice));

    auto eval = [&]() -> float {
        CUDA_CHECK(cudaMemcpy(d_means, h_means.data(), 3*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rotation, h_rotation.data(), 4*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scaling, h_scaling.data(), 3*sizeof(float), cudaMemcpyHostToDevice));
        projection_2dgs_kernel<<<1, 1>>>(
            d_means, d_rotation, d_scaling, d_viewmat,
            fx, fy, cx, cy,
            0.2f, W, H,
            d_T, d_means2d, d_radii, d_depths, d_normals, N
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> out_T(9), out_means2d(2), out_depths(1);
        CUDA_CHECK(cudaMemcpy(out_T.data(), d_T, 9*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_means2d.data(), d_means2d, 2*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_depths.data(), d_depths, sizeof(float), cudaMemcpyDeviceToHost));
        return dot_host(out_T, h_vT) + dot_host(out_means2d, h_v_means2d) + dot_host(out_depths, h_v_depths);
    };

    (void)eval();
    launch_projection_2dgs_bwd(
        d_means, d_rotation, d_scaling, d_viewmat,
        fx, fy, cx, cy,
        d_T, d_radii,
        d_vT, d_v_means2d, d_v_depths, nullptr,
        d_g_means, d_g_rotation, d_g_scaling, N
    );

    std::vector<float> g_means(3), g_rotation(4), g_scaling(3);
    CUDA_CHECK(cudaMemcpy(g_means.data(), d_g_means, 3*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_rotation.data(), d_g_rotation, 4*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_scaling.data(), d_g_scaling, 3*sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    const float eps = 1e-3f;
    const float rot_eps = 5e-4f;
    ok &= check_close("means.x",    g_means[0],    centered_diff(h_means,    0, eps, eval),     5e-3f, 5e-2f);
    ok &= check_close("means.y",    g_means[1],    centered_diff(h_means,    1, eps, eval),     5e-3f, 5e-2f);
    ok &= check_close("means.z",    g_means[2],    centered_diff(h_means,    2, eps, eval),     5e-3f, 5e-2f);
    ok &= check_close("rotation.x", g_rotation[1], centered_diff(h_rotation, 1, rot_eps, eval), 5e-3f, 7e-2f);
    ok &= check_close("rotation.y", g_rotation[2], centered_diff(h_rotation, 2, rot_eps, eval), 5e-3f, 7e-2f);
    ok &= check_close("scaling.x",  g_scaling[0],  centered_diff(h_scaling,  0, eps, eval),     5e-3f, 5e-2f);
    ok &= check_close("scaling.y",  g_scaling[1],  centered_diff(h_scaling,  1, eps, eval),     5e-3f, 5e-2f);

    cudaFree(d_means); cudaFree(d_rotation); cudaFree(d_scaling); cudaFree(d_viewmat);
    cudaFree(d_T); cudaFree(d_means2d); cudaFree(d_depths); cudaFree(d_normals);
    cudaFree(d_radii); cudaFree(d_vT); cudaFree(d_v_means2d); cudaFree(d_v_depths);
    cudaFree(d_g_means); cudaFree(d_g_rotation); cudaFree(d_g_scaling);

    printf("%s\n\n", ok ? "Projection gradients passed." : "Projection gradients FAILED.");
    return ok;
}

static bool run_sh_gradcheck() {
    printf("=== SH Gradient Check ===\n");

    const int N = 1;
    const int sh_active = 3;
    const float cam_x = 0.f, cam_y = 0.f, cam_z = 0.f;

    std::vector<float> h_means = {0.2f, -0.15f, 2.0f};
    std::vector<float> h_sh0   = {0.05f, -0.08f, 0.04f};
    std::vector<float> h_shN(45, 0.f);
    for (int i = 0; i < 45; i++)
        h_shN[i] = 0.01f * ((i % 7) - 3);
    std::vector<float> h_v_colors = {0.4f, -0.25f, 0.3f};

    float *d_means, *d_sh0, *d_shN, *d_colors, *d_v_colors, *d_g_sh0, *d_g_shN;
    CUDA_CHECK(cudaMalloc(&d_means, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sh0, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_shN, 45*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colors, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_colors, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_sh0, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_shN, 45*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_means, h_means.data(), 3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_colors, h_v_colors.data(), 3*sizeof(float), cudaMemcpyHostToDevice));

    auto eval = [&]() -> float {
        CUDA_CHECK(cudaMemcpy(d_sh0, h_sh0.data(), 3*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_shN, h_shN.data(), 45*sizeof(float), cudaMemcpyHostToDevice));
        sh_eval_gradcheck_kernel<<<1, 1>>>(
            d_means, d_sh0, d_shN,
            cam_x, cam_y, cam_z,
            d_colors, sh_active, N
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out_colors(3);
        CUDA_CHECK(cudaMemcpy(out_colors.data(), d_colors, 3*sizeof(float), cudaMemcpyDeviceToHost));
        return dot_host(out_colors, h_v_colors);
    };

    (void)eval();
    sh_backward_gradcheck_kernel<<<1, 1>>>(
        d_means, d_sh0, d_shN,
        cam_x, cam_y, cam_z,
        d_v_colors, d_g_sh0, d_g_shN,
        sh_active, N
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> g_sh0(3), g_shN(45);
    CUDA_CHECK(cudaMemcpy(g_sh0.data(), d_g_sh0, 3*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_shN.data(), d_g_shN, 45*sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    const float eps = 1e-4f;
    ok &= check_close("sh0.r",         g_sh0[0], centered_diff(h_sh0, 0, eps, eval), 2e-4f, 2e-3f);
    ok &= check_close("sh0.g",         g_sh0[1], centered_diff(h_sh0, 1, eps, eval), 2e-4f, 2e-3f);
    ok &= check_close("shN[c0,k0]",    g_shN[0], centered_diff(h_shN, 0, eps, eval), 2e-4f, 2e-3f);
    ok &= check_close("shN[c1,k5]",    g_shN[20], centered_diff(h_shN, 20, eps, eval), 2e-4f, 2e-3f);
    ok &= check_close("shN[c2,k14]",   g_shN[44], centered_diff(h_shN, 44, eps, eval), 2e-4f, 2e-3f);

    cudaFree(d_means); cudaFree(d_sh0); cudaFree(d_shN);
    cudaFree(d_colors); cudaFree(d_v_colors); cudaFree(d_g_sh0); cudaFree(d_g_shN);

    printf("%s\n\n", ok ? "SH gradients passed." : "SH gradients FAILED.");
    return ok;
}

static bool run_rasterize_gradcheck() {
    printf("=== Rasterizer Gradient Check ===\n");

    const int N = 1;
    const uint32_t W = 16, H = 16;
    const float fx = 32.f, fy = 32.f, cx = 8.f, cy = 8.f;

    std::vector<float> h_proj_means    = {0.f, 0.f, 2.f};
    std::vector<float> h_proj_rotation = {1.f, 0.f, 0.f, 0.f};
    std::vector<float> h_proj_scaling  = {-2.4f, -2.4f, 0.f};
    std::vector<float> h_viewmat = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    float *d_proj_means, *d_proj_rotation, *d_proj_scaling, *d_viewmat;
    float *d_means2d, *d_T, *d_depths, *d_normals;
    int32_t* d_radii;
    CUDA_CHECK(cudaMalloc(&d_proj_means, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj_rotation, 4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj_scaling, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_viewmat, 16*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means2d, 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depths, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normals, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii, 2*sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_proj_means, h_proj_means.data(), 3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_proj_rotation, h_proj_rotation.data(), 4*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_proj_scaling, h_proj_scaling.data(), 3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_viewmat, h_viewmat.data(), 16*sizeof(float), cudaMemcpyHostToDevice));
    projection_2dgs_kernel<<<1,1>>>(
        d_proj_means, d_proj_rotation, d_proj_scaling, d_viewmat,
        fx, fy, cx, cy,
        0.2f, (int)W, (int)H,
        d_T, d_means2d, d_radii, d_depths, d_normals, N
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_means2d(2), h_T(9), h_depths(1);
    std::vector<int32_t> h_radii(2);
    CUDA_CHECK(cudaMemcpy(h_means2d.data(), d_means2d, 2*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_T.data(), d_T, 9*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_depths.data(), d_depths, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_radii.data(), d_radii, 2*sizeof(int32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_proj_means); cudaFree(d_proj_rotation); cudaFree(d_proj_scaling); cudaFree(d_viewmat);
    cudaFree(d_depths); cudaFree(d_normals); cudaFree(d_radii);

    std::vector<float> h_opacity = {0.25f};
    std::vector<float> h_colors  = {0.75f, 0.2f, 0.1f};
    std::vector<float> h_v_render(W * H * 3, 0.f);
    for (uint32_t y = 5; y <= 10; y++) {
        for (uint32_t x = 5; x <= 10; x++) {
            size_t idx = (y * W + x) * 3;
            h_v_render[idx + 0] = 0.01f * ((int)x - 7);
            h_v_render[idx + 1] = -0.015f * ((int)y - 7);
            h_v_render[idx + 2] = 0.005f * ((int)x + (int)y - 14);
        }
    }

    float *d_rast_means2d, *d_rast_T, *d_rast_opacity, *d_rast_colors, *d_rast_depths;
    int32_t* d_rast_radii;
    float *d_render_colors, *d_render_alphas, *d_v_render;
    int32_t* d_last_ids;
    float *d_g_T, *d_g_opacity, *d_g_colors, *d_g_means2d, *d_g_means2d_abs;

    CUDA_CHECK(cudaMalloc(&d_rast_means2d, 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_T, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_opacity, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_colors, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_depths, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_radii, 2*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_render_colors, W*H*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_alphas, W*H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_last_ids, W*H*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_v_render, W*H*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_T, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_opacity, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_colors, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_means2d, 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_means2d_abs, 2*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rast_depths, h_depths.data(), sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rast_radii, h_radii.data(), 2*sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_render, h_v_render.data(), W*H*3*sizeof(float), cudaMemcpyHostToDevice));

    auto eval = [&]() -> float {
        CUDA_CHECK(cudaMemcpy(d_rast_means2d, h_means2d.data(), 2*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rast_T, h_T.data(), 9*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rast_opacity, h_opacity.data(), sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rast_colors, h_colors.data(), 3*sizeof(float), cudaMemcpyHostToDevice));

        TileIntersectBuffers tile_buf = launch_tile_intersect(
            d_rast_means2d, d_rast_radii, d_rast_depths, N, W, H, TILE_SIZE
        );

        CUDA_CHECK(cudaMemset(d_render_colors, 0, W*H*3*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_render_alphas, 0, W*H*sizeof(float)));
        launch_rasterize_fwd(
            d_rast_means2d, d_rast_T, d_rast_opacity, d_rast_colors,
            tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
            W, H,
            d_render_colors, d_render_alphas, d_last_ids
        );

        std::vector<float> h_render(W * H * 3);
        CUDA_CHECK(cudaMemcpy(h_render.data(), d_render_colors, W*H*3*sizeof(float), cudaMemcpyDeviceToHost));
        free_tile_intersect_buffers(tile_buf);
        return dot_host(h_render, h_v_render);
    };

    (void)eval();

    CUDA_CHECK(cudaMemcpy(d_rast_means2d, h_means2d.data(), 2*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rast_T, h_T.data(), 9*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rast_opacity, h_opacity.data(), sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rast_colors, h_colors.data(), 3*sizeof(float), cudaMemcpyHostToDevice));
    TileIntersectBuffers tile_buf = launch_tile_intersect(
        d_rast_means2d, d_rast_radii, d_rast_depths, N, W, H, TILE_SIZE
    );
    CUDA_CHECK(cudaMemset(d_render_colors, 0, W*H*3*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_render_alphas, 0, W*H*sizeof(float)));
    launch_rasterize_fwd(
        d_rast_means2d, d_rast_T, d_rast_opacity, d_rast_colors,
        tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
        W, H,
        d_render_colors, d_render_alphas, d_last_ids
    );
    CUDA_CHECK(cudaMemset(d_g_T, 0, 9*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_opacity, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_colors, 0, 3*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_means2d, 0, 2*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_means2d_abs, 0, 2*sizeof(float)));
    launch_rasterize_bwd(
        d_rast_means2d, d_rast_T, d_rast_opacity, d_rast_colors,
        tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
        d_render_alphas, d_last_ids, d_v_render,
        W, H,
        d_g_T, d_g_opacity, d_g_colors, d_g_means2d, d_g_means2d_abs
    );

    std::vector<float> g_T(9), g_colors(3), g_means2d(2), g_opacity(1);
    CUDA_CHECK(cudaMemcpy(g_T.data(), d_g_T, 9*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_colors.data(), d_g_colors, 3*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_means2d.data(), d_g_means2d, 2*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_opacity.data(), d_g_opacity, sizeof(float), cudaMemcpyDeviceToHost));
    free_tile_intersect_buffers(tile_buf);

    bool ok = true;
    const float eps = 1e-3f;
    ok &= check_close("means2d.x",   g_means2d[0], centered_diff(h_means2d, 0, eps, eval), 7e-3f, 8e-2f);
    ok &= check_close("rayT[0]",     g_T[0],       centered_diff(h_T,       0, eps, eval), 7e-3f, 8e-2f);
    ok &= check_close("rayT[8]",     g_T[8],       centered_diff(h_T,       8, eps, eval), 7e-3f, 8e-2f);
    ok &= check_close("opacity",     g_opacity[0], centered_diff(h_opacity, 0, eps, eval), 7e-3f, 8e-2f);
    ok &= check_close("color.r",     g_colors[0],  centered_diff(h_colors,  0, eps, eval), 7e-3f, 8e-2f);

    cudaFree(d_means2d); cudaFree(d_T);
    cudaFree(d_rast_means2d); cudaFree(d_rast_T); cudaFree(d_rast_opacity);
    cudaFree(d_rast_colors); cudaFree(d_rast_depths); cudaFree(d_rast_radii);
    cudaFree(d_render_colors); cudaFree(d_render_alphas); cudaFree(d_last_ids);
    cudaFree(d_v_render); cudaFree(d_g_T); cudaFree(d_g_opacity); cudaFree(d_g_colors);
    cudaFree(d_g_means2d); cudaFree(d_g_means2d_abs);

    printf("%s\n\n", ok ? "Rasterizer gradients passed." : "Rasterizer gradients FAILED.");
    return ok;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    bool ok = true;
    ok &= run_projection_gradcheck();
    ok &= run_sh_gradcheck();
    ok &= run_rasterize_gradcheck();

    printf("%s\n", ok ? "All gradient checks passed." : "SOME GRADIENT CHECKS FAILED.");
    return ok ? 0 : 1;
}
