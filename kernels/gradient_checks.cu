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
#include "loss.cu"
#include "hash_grid.cu"
#include "mlp.cu"
#include "nexel_color.cu"
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

struct RasterSceneData {
    std::vector<float> means2d;
    std::vector<float> T;
    std::vector<float> depths;
    std::vector<float> normals;
    std::vector<int32_t> radii;
};

static RasterSceneData make_overlapping_raster_scene(
    uint32_t W, uint32_t H,
    float fx, float fy, float cx, float cy)
{
    const int N = 2;
    std::vector<float> h_means = {
         0.00f,  0.00f, 2.00f,
         0.05f, -0.03f, 2.25f
    };
    std::vector<float> h_rotation = {
        1.f, 0.f, 0.f, 0.f,
        1.f, 0.f, 0.f, 0.f
    };
    std::vector<float> h_scaling = {
        -1.95f, -1.95f, 0.f,
        -2.05f, -2.05f, 0.f
    };
    std::vector<float> h_viewmat = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    float *d_means, *d_rotation, *d_scaling, *d_viewmat;
    float *d_T, *d_means2d, *d_depths, *d_normals;
    int32_t* d_radii;
    CUDA_CHECK(cudaMalloc(&d_means, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rotation, N * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scaling, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_viewmat, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means2d, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depths, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normals, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii, N * 2 * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_means, h_means.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rotation, h_rotation.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scaling, h_scaling.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_viewmat, h_viewmat.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));

    projection_2dgs_kernel<<<1, N>>>(
        d_means, d_rotation, d_scaling, d_viewmat,
        fx, fy, cx, cy,
        0.2f, (int)W, (int)H,
        d_T, d_means2d, d_radii, d_depths, d_normals, N
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    RasterSceneData scene;
    scene.means2d.resize(N * 2);
    scene.T.resize(N * 9);
    scene.depths.resize(N);
    scene.normals.resize(N * 3);
    scene.radii.resize(N * 2);
    CUDA_CHECK(cudaMemcpy(scene.means2d.data(), d_means2d, N * 2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scene.T.data(), d_T, N * 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scene.depths.data(), d_depths, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scene.normals.data(), d_normals, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scene.radii.data(), d_radii, N * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_means); cudaFree(d_rotation); cudaFree(d_scaling); cudaFree(d_viewmat);
    cudaFree(d_T); cudaFree(d_means2d); cudaFree(d_depths); cudaFree(d_normals); cudaFree(d_radii);
    return scene;
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
    std::vector<float> h_v_normals = {0.09f, -0.07f, 0.05f};

    float *d_means, *d_rotation, *d_scaling, *d_viewmat;
    float *d_T, *d_means2d, *d_depths, *d_normals;
    int32_t* d_radii;
    float *d_vT, *d_v_means2d, *d_v_depths, *d_v_normals;
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
    CUDA_CHECK(cudaMalloc(&d_v_normals, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_means, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_rotation, 4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_scaling, 3*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_viewmat, h_viewmat.data(), 16*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vT, h_vT.data(), 9*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_means2d, h_v_means2d.data(), 2*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_depths, h_v_depths.data(), sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_normals, h_v_normals.data(), 3*sizeof(float), cudaMemcpyHostToDevice));

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

        std::vector<float> out_T(9), out_means2d(2), out_depths(1), out_normals(3);
        CUDA_CHECK(cudaMemcpy(out_T.data(), d_T, 9*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_means2d.data(), d_means2d, 2*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_depths.data(), d_depths, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_normals.data(), d_normals, 3*sizeof(float), cudaMemcpyDeviceToHost));
        return dot_host(out_T, h_vT) +
               dot_host(out_means2d, h_v_means2d) +
               dot_host(out_depths, h_v_depths) +
               dot_host(out_normals, h_v_normals);
    };

    (void)eval();
    launch_projection_2dgs_bwd(
        d_means, d_rotation, d_scaling, d_viewmat,
        fx, fy, cx, cy,
        d_T, d_radii,
        d_vT, d_v_means2d, d_v_depths, d_v_normals,
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
    ok &= check_close("rotation.z", g_rotation[3], centered_diff(h_rotation, 3, rot_eps, eval), 5e-3f, 7e-2f);
    ok &= check_close("scaling.x",  g_scaling[0],  centered_diff(h_scaling,  0, eps, eval),     5e-3f, 5e-2f);
    ok &= check_close("scaling.y",  g_scaling[1],  centered_diff(h_scaling,  1, eps, eval),     5e-3f, 5e-2f);

    cudaFree(d_means); cudaFree(d_rotation); cudaFree(d_scaling); cudaFree(d_viewmat);
    cudaFree(d_T); cudaFree(d_means2d); cudaFree(d_depths); cudaFree(d_normals);
    cudaFree(d_radii); cudaFree(d_vT); cudaFree(d_v_means2d); cudaFree(d_v_depths); cudaFree(d_v_normals);
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

    float *d_rast_means2d, *d_rast_T, *d_rast_opacity, *d_rast_gamma, *d_rast_colors, *d_rast_depths, *d_rast_normals;
    int32_t* d_rast_radii;
    float *d_render_colors, *d_render_alphas, *d_v_render;
    int32_t* d_last_ids;
    float *d_g_T, *d_g_opacity, *d_g_gamma, *d_g_colors, *d_g_normals, *d_g_means2d, *d_g_means2d_abs;

    CUDA_CHECK(cudaMalloc(&d_rast_means2d, 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_T, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_opacity, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_gamma, 2*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_rast_gamma, 0, 2*sizeof(float)));  // γ=2 parity
    CUDA_CHECK(cudaMalloc(&d_rast_colors, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_depths, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_normals, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rast_radii, 2*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_render_colors, W*H*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_alphas, W*H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_last_ids, W*H*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_v_render, W*H*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_T, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_opacity, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_gamma, 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_colors, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_normals, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_means2d, 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_means2d_abs, 2*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rast_depths, h_depths.data(), sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_rast_normals, 0, 3*sizeof(float)));
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
            d_rast_means2d, d_rast_T, d_rast_opacity, d_rast_gamma, d_rast_colors, d_rast_normals,
            tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
            W, H,
            d_render_colors, d_render_alphas,
            nullptr, nullptr, nullptr, nullptr, d_last_ids
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
        d_rast_means2d, d_rast_T, d_rast_opacity, d_rast_gamma, d_rast_colors, d_rast_normals,
        tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
        W, H,
        d_render_colors, d_render_alphas,
        nullptr, nullptr, nullptr, nullptr, d_last_ids
    );
    CUDA_CHECK(cudaMemset(d_g_T, 0, 9*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_opacity, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_gamma, 0, 2*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_colors, 0, 3*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_normals, 0, 3*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_means2d, 0, 2*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_means2d_abs, 0, 2*sizeof(float)));
    launch_rasterize_bwd(
        d_rast_means2d, d_rast_T, d_rast_opacity, d_rast_gamma, d_rast_colors, d_rast_normals,
        tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
        d_render_alphas, nullptr, d_last_ids,
        d_v_render, nullptr, nullptr, nullptr, nullptr,
        W, H,
        d_g_T, d_g_opacity, d_g_gamma, d_g_colors, d_g_normals, d_g_means2d, d_g_means2d_abs
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
    cudaFree(d_rast_means2d); cudaFree(d_rast_T); cudaFree(d_rast_opacity); cudaFree(d_rast_gamma);
    cudaFree(d_rast_colors); cudaFree(d_rast_depths); cudaFree(d_rast_normals); cudaFree(d_rast_radii);
    cudaFree(d_render_colors); cudaFree(d_render_alphas); cudaFree(d_last_ids);
    cudaFree(d_v_render); cudaFree(d_g_T); cudaFree(d_g_opacity); cudaFree(d_g_gamma);
    cudaFree(d_g_colors); cudaFree(d_g_normals);
    cudaFree(d_g_means2d); cudaFree(d_g_means2d_abs);

    printf("%s\n\n", ok ? "Rasterizer gradients passed." : "Rasterizer gradients FAILED.");
    return ok;
}

static bool run_rasterize_aux_gradcheck() {
    printf("=== Rasterizer Aux Gradient Check ===\n");

    const int N = 2;
    const uint32_t W = 24, H = 24;
    const float fx = 44.f, fy = 44.f, cx = 12.f, cy = 12.f;

    RasterSceneData scene = make_overlapping_raster_scene(W, H, fx, fy, cx, cy);
    std::vector<float> h_means2d = scene.means2d;
    std::vector<float> h_T = scene.T;
    std::vector<float> h_depths = scene.depths;
    std::vector<int32_t> h_radii = scene.radii;
    std::vector<float> h_normals = {
         0.20f,  0.10f, 0.97f,
        -0.12f,  0.18f, 0.94f
    };
    std::vector<float> h_opacity = {0.35f, 0.05f};
    std::vector<float> h_colors = {
        0.75f, 0.20f, 0.10f,
        0.10f, 0.55f, 0.80f
    };

    std::vector<float> h_v_render_colors(W * H * 3, 0.f);
    std::vector<float> h_v_render_normals(W * H * 3, 0.f);
    std::vector<float> h_v_render_depth(W * H, 0.f);
    std::vector<float> h_v_render_distort(W * H, 0.f);
    for (uint32_t y = 7; y <= 16; ++y) {
        for (uint32_t x = 7; x <= 16; ++x) {
            size_t pix = y * W + x;
            h_v_render_normals[pix * 3 + 0] = 0.02f * ((int)x - 11);
            h_v_render_normals[pix * 3 + 1] = -0.015f * ((int)y - 11);
            h_v_render_normals[pix * 3 + 2] = 0.01f;
            h_v_render_depth[pix] = 0.004f * ((int)x + (int)y - 22);
            h_v_render_distort[pix] = 0.03f;
        }
    }

    float *d_means2d, *d_T, *d_opacity, *d_gamma, *d_colors, *d_depths, *d_normals;
    int32_t* d_radii;
    float *d_render_colors, *d_render_alphas, *d_render_normals, *d_render_depth, *d_render_distort;
    int32_t* d_last_ids;
    float *d_v_render_colors, *d_v_render_normals, *d_v_render_depth, *d_v_render_distort;
    float *d_g_T, *d_g_opacity, *d_g_gamma, *d_g_colors, *d_g_normals, *d_g_means2d, *d_g_means2d_abs;

    CUDA_CHECK(cudaMalloc(&d_means2d, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opacity, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gamma, 0, N * 2 * sizeof(float)));  // γ=2 parity
    CUDA_CHECK(cudaMalloc(&d_colors, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depths, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normals, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii, N * 2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_render_colors, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_alphas, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_normals, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_depth, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_distort, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_last_ids, W * H * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_v_render_colors, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_render_normals, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_render_depth, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_render_distort, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_T, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_opacity, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_gamma, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_colors, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_normals, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_means2d, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_means2d_abs, N * 2 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_depths, h_depths.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radii, h_radii.data(), N * 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_render_colors, h_v_render_colors.data(), W * H * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_render_normals, h_v_render_normals.data(), W * H * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_render_depth, h_v_render_depth.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_render_distort, h_v_render_distort.data(), W * H * sizeof(float), cudaMemcpyHostToDevice));

    auto eval = [&]() -> float {
        CUDA_CHECK(cudaMemcpy(d_means2d, h_means2d.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_T, h_T.data(), N * 9 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_opacity, h_opacity.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colors, h_colors.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_normals, h_normals.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));

        TileIntersectBuffers tile_buf = launch_tile_intersect(
            d_means2d, d_radii, d_depths, N, W, H, TILE_SIZE
        );
        CUDA_CHECK(cudaMemset(d_render_colors, 0, W * H * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_render_alphas, 0, W * H * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_render_normals, 0, W * H * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_render_depth, 0, W * H * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_render_distort, 0, W * H * sizeof(float)));
        launch_rasterize_fwd(
            d_means2d, d_T, d_opacity, d_gamma, d_colors, d_normals,
            tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
            W, H,
            d_render_colors, d_render_alphas, d_render_normals,
            d_render_depth, d_render_distort, nullptr, d_last_ids
        );

        std::vector<float> out_normals(W * H * 3), out_depth(W * H), out_distort(W * H);
        CUDA_CHECK(cudaMemcpy(out_normals.data(), d_render_normals, W * H * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_depth.data(), d_render_depth, W * H * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(out_distort.data(), d_render_distort, W * H * sizeof(float), cudaMemcpyDeviceToHost));
        free_tile_intersect_buffers(tile_buf);
        return dot_host(out_normals, h_v_render_normals) +
               dot_host(out_depth, h_v_render_depth) +
               dot_host(out_distort, h_v_render_distort);
    };

    (void)eval();

    CUDA_CHECK(cudaMemcpy(d_means2d, h_means2d.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T, h_T.data(), N * 9 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacity, h_opacity.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colors, h_colors.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_normals, h_normals.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    TileIntersectBuffers tile_buf = launch_tile_intersect(
        d_means2d, d_radii, d_depths, N, W, H, TILE_SIZE
    );
    CUDA_CHECK(cudaMemset(d_render_colors, 0, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_render_alphas, 0, W * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_render_normals, 0, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_render_depth, 0, W * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_render_distort, 0, W * H * sizeof(float)));
    launch_rasterize_fwd(
        d_means2d, d_T, d_opacity, d_gamma, d_colors, d_normals,
        tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
        W, H,
        d_render_colors, d_render_alphas, d_render_normals,
        d_render_depth, d_render_distort, nullptr, d_last_ids
    );
    CUDA_CHECK(cudaMemset(d_g_T, 0, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_opacity, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_gamma, 0, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_colors, 0, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_normals, 0, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_means2d, 0, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_means2d_abs, 0, N * 2 * sizeof(float)));
    launch_rasterize_bwd(
        d_means2d, d_T, d_opacity, d_gamma, d_colors, d_normals,
        tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
        d_render_alphas, d_render_depth, d_last_ids,
        d_v_render_colors, nullptr, d_v_render_normals, d_v_render_depth, d_v_render_distort,
        W, H,
        d_g_T, d_g_opacity, d_g_gamma, d_g_colors, d_g_normals, d_g_means2d, d_g_means2d_abs
    );

    std::vector<float> g_T(N * 9), g_opacity(N), g_normals(N * 3);
    CUDA_CHECK(cudaMemcpy(g_T.data(), d_g_T, N * 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_opacity.data(), d_g_opacity, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_normals.data(), d_g_normals, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    free_tile_intersect_buffers(tile_buf);

    bool ok = true;
    const float eps = 1e-3f;
    ok &= check_close("normal0.x", g_normals[0], centered_diff(h_normals, 0, eps, eval), 8e-3f, 9e-2f);
    ok &= check_close("rayT0[8]",  g_T[8],      centered_diff(h_T,       8, eps, eval), 8e-3f, 9e-2f);
    ok &= check_close("opacity0",  g_opacity[0], centered_diff(h_opacity, 0, eps, eval), 8e-3f, 9e-2f);

    cudaFree(d_means2d); cudaFree(d_T); cudaFree(d_opacity); cudaFree(d_gamma);
    cudaFree(d_colors); cudaFree(d_depths); cudaFree(d_normals);
    cudaFree(d_radii); cudaFree(d_render_colors); cudaFree(d_render_alphas); cudaFree(d_render_normals);
    cudaFree(d_render_depth); cudaFree(d_render_distort); cudaFree(d_last_ids);
    cudaFree(d_v_render_colors); cudaFree(d_v_render_normals); cudaFree(d_v_render_depth); cudaFree(d_v_render_distort);
    cudaFree(d_g_T); cudaFree(d_g_opacity); cudaFree(d_g_gamma);
    cudaFree(d_g_colors); cudaFree(d_g_normals); cudaFree(d_g_means2d); cudaFree(d_g_means2d_abs);

    printf("%s\n\n", ok ? "Rasterizer aux gradients passed." : "Rasterizer aux gradients FAILED.");
    return ok;
}

static bool run_geometry_loss_gradcheck() {
    printf("=== Geometry Loss Gradient Check ===\n");

    const int N = 2;
    const uint32_t W = 24, H = 24;
    const float fx = 44.f, fy = 44.f, cx = 12.f, cy = 12.f;
    const float normal_lambda = 5e-2f;
    const float dist_lambda = 1e-2f;

    RasterSceneData scene = make_overlapping_raster_scene(W, H, fx, fy, cx, cy);
    std::vector<float> h_means2d = scene.means2d;
    std::vector<float> h_T = scene.T;
    std::vector<float> h_depths = scene.depths;
    std::vector<int32_t> h_radii = scene.radii;
    std::vector<float> h_normals = {
         0.24f,  0.08f, 0.95f,
        -0.15f,  0.20f, 0.91f
    };
    std::vector<float> h_opacity = {0.35f, 0.05f};
    std::vector<float> h_colors = {
        0.75f, 0.20f, 0.10f,
        0.10f, 0.55f, 0.80f
    };

    float *d_means2d, *d_T, *d_opacity, *d_gamma, *d_colors, *d_depths, *d_normals;
    int32_t* d_radii;
    float *d_render_colors, *d_render_alphas, *d_render_normals, *d_render_depth, *d_render_distort;
    float *d_grad_render_colors, *d_grad_render_alphas, *d_grad_render_normals, *d_grad_render_depth, *d_grad_render_distort;
    int32_t* d_last_ids;
    float *d_g_T, *d_g_opacity, *d_g_gamma, *d_g_colors, *d_g_normals, *d_g_means2d, *d_g_means2d_abs;
    GeometryLossWorkspace geom_ws{};

    CUDA_CHECK(cudaMalloc(&d_means2d, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_opacity, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gamma, 0, N * 2 * sizeof(float)));  // γ=2 parity
    CUDA_CHECK(cudaMalloc(&d_colors, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_depths, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normals, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii, N * 2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_render_colors, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_alphas, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_normals, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_depth, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_render_distort, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_render_colors, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_render_alphas, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_render_normals, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_render_depth, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_render_distort, W * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_last_ids, W * H * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_g_T, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_opacity, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_gamma, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_colors, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_normals, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_means2d, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_means2d_abs, N * 2 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_depths, h_depths.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radii, h_radii.data(), N * 2 * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_grad_render_colors, 0, W * H * 3 * sizeof(float)));
    ensure_geometry_loss_workspace(geom_ws, W * H);

    auto eval = [&]() -> float {
        CUDA_CHECK(cudaMemcpy(d_means2d, h_means2d.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_T, h_T.data(), N * 9 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_opacity, h_opacity.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_colors, h_colors.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_normals, h_normals.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));

        TileIntersectBuffers tile_buf = launch_tile_intersect(
            d_means2d, d_radii, d_depths, N, W, H, TILE_SIZE
        );
        CUDA_CHECK(cudaMemset(d_render_colors, 0, W * H * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_render_alphas, 0, W * H * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_render_normals, 0, W * H * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_render_depth, 0, W * H * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_render_distort, 0, W * H * sizeof(float)));
        launch_rasterize_fwd(
            d_means2d, d_T, d_opacity, d_gamma, d_colors, d_normals,
            tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
            W, H,
            d_render_colors, d_render_alphas, d_render_normals,
            d_render_depth, d_render_distort, nullptr, d_last_ids
        );
        CUDA_CHECK(cudaMemset(d_grad_render_alphas, 0, W * H * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_render_normals, 0, W * H * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_render_depth, 0, W * H * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_grad_render_distort, 0, W * H * sizeof(float)));
        GeometryLossResult geom = geometry_loss_2dgs(
            d_render_depth, d_render_alphas, d_render_normals, d_render_distort,
            /*render_median=*/nullptr,
            d_grad_render_depth, d_grad_render_alphas, d_grad_render_normals, d_grad_render_distort,
            geom_ws, H, W, fx, fy, cx, cy, normal_lambda, dist_lambda,
            /*depth_ratio=*/0.f
        );
        free_tile_intersect_buffers(tile_buf);
        return geom.loss_total;
    };

    (void)eval();

    CUDA_CHECK(cudaMemcpy(d_means2d, h_means2d.data(), N * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T, h_T.data(), N * 9 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_opacity, h_opacity.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colors, h_colors.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_normals, h_normals.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    TileIntersectBuffers tile_buf = launch_tile_intersect(
        d_means2d, d_radii, d_depths, N, W, H, TILE_SIZE
    );
    CUDA_CHECK(cudaMemset(d_render_colors, 0, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_render_alphas, 0, W * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_render_normals, 0, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_render_depth, 0, W * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_render_distort, 0, W * H * sizeof(float)));
    launch_rasterize_fwd(
        d_means2d, d_T, d_opacity, d_gamma, d_colors, d_normals,
        tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
        W, H,
        d_render_colors, d_render_alphas, d_render_normals,
        d_render_depth, d_render_distort, nullptr, d_last_ids
    );
    CUDA_CHECK(cudaMemset(d_grad_render_alphas, 0, W * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_render_normals, 0, W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_render_depth, 0, W * H * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_render_distort, 0, W * H * sizeof(float)));
    (void)geometry_loss_2dgs(
        d_render_depth, d_render_alphas, d_render_normals, d_render_distort,
        /*render_median=*/nullptr,
        d_grad_render_depth, d_grad_render_alphas, d_grad_render_normals, d_grad_render_distort,
        geom_ws, H, W, fx, fy, cx, cy, normal_lambda, dist_lambda,
        /*depth_ratio=*/0.f
    );
    CUDA_CHECK(cudaMemset(d_g_T, 0, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_opacity, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_gamma, 0, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_colors, 0, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_normals, 0, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_means2d, 0, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_means2d_abs, 0, N * 2 * sizeof(float)));
    launch_rasterize_bwd(
        d_means2d, d_T, d_opacity, d_gamma, d_colors, d_normals,
        tile_buf.tile_offsets, tile_buf.flatten_ids, tile_buf.n_isects,
        d_render_alphas, d_render_depth, d_last_ids,
        d_grad_render_colors, d_grad_render_alphas, d_grad_render_normals,
        d_grad_render_depth, d_grad_render_distort,
        W, H,
        d_g_T, d_g_opacity, d_g_gamma, d_g_colors, d_g_normals, d_g_means2d, d_g_means2d_abs
    );

    std::vector<float> g_T(N * 9), g_opacity(N), g_normals(N * 3);
    CUDA_CHECK(cudaMemcpy(g_T.data(), d_g_T, N * 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_opacity.data(), d_g_opacity, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_normals.data(), d_g_normals, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    free_tile_intersect_buffers(tile_buf);

    bool ok = true;
    const float eps = 1e-3f;
    ok &= check_close("geom normal0.x", g_normals[0], centered_diff(h_normals, 0, eps, eval), 1e-2f, 1.2e-1f);
    ok &= check_close("geom rayT0[8]",  g_T[8],      centered_diff(h_T,       8, eps, eval), 1e-2f, 1.2e-1f);
    ok &= check_close("geom opacity0",  g_opacity[0], centered_diff(h_opacity, 0, eps, eval), 1e-2f, 1.2e-1f);

    free_geometry_loss_workspace(geom_ws);
    cudaFree(d_means2d); cudaFree(d_T); cudaFree(d_opacity); cudaFree(d_gamma);
    cudaFree(d_colors); cudaFree(d_depths); cudaFree(d_normals);
    cudaFree(d_radii); cudaFree(d_render_colors); cudaFree(d_render_alphas); cudaFree(d_render_normals);
    cudaFree(d_render_depth); cudaFree(d_render_distort); cudaFree(d_grad_render_colors); cudaFree(d_grad_render_alphas);
    cudaFree(d_grad_render_normals); cudaFree(d_grad_render_depth); cudaFree(d_grad_render_distort); cudaFree(d_last_ids);
    cudaFree(d_g_T); cudaFree(d_g_opacity); cudaFree(d_g_gamma);
    cudaFree(d_g_colors); cudaFree(d_g_normals); cudaFree(d_g_means2d); cudaFree(d_g_means2d_abs);

    printf("%s\n\n", ok ? "Geometry loss gradients passed." : "Geometry loss gradients FAILED.");
    return ok;
}

// ─── M4 linearity check ───────────────────────────────────────────────────────
//
// The MPI data-parallel trainer relies on a single invariant: the analytic
// gradient of `sum_c loss_c` over a set of cameras C equals `sum_c
// (analytic_gradient of loss_c)`. The allreduce-SUM after backward is
// literally the right-hand side; if the trainer's backward is correct then
// this identity holds by calculus. This test confirms the identity numerically
// using the same fwd+bwd codepath the trainer uses, *as if* two MPI ranks had
// processed two different cameras with shared parameters.
//
// We construct two independent "camera observations" of one Gaussian — same
// means2d/T/depth, but different colors and different target renders, simulating
// two views with shared scene geometry. The shared parameter under test is
// opacity[0]:
//
//   analytic = grad_opacity_cam0 + grad_opacity_cam1
//   fd       = centered_diff(opacity, (loss_cam0 + loss_cam1))
//
// Passing this check means the trainer's per-iter post-`sh_bwd` allreduce
// would produce the same gradient as a single rank had processed both cameras
// sequentially with summed losses.
static bool run_split_batch_linearity_check() {
    printf("=== MPI split-batch linearity check ===\n");

    const int N = 1;
    const uint32_t W = 16, H = 16;
    const float fx = 32.f, fy = 32.f, cx = 8.f, cy = 8.f;

    // ── Project the Gaussian once to get a common (T, means2d, depth, radii).
    // We then synthesise two "cameras" by shifting means2d slightly — a quick
    // proxy for two cameras whose projections produce different screen-space
    // footprints, without paying the full price of running projection twice.
    std::vector<float> h_proj_means    = {0.f, 0.f, 2.f};
    std::vector<float> h_proj_rotation = {1.f, 0.f, 0.f, 0.f};
    std::vector<float> h_proj_scaling  = {-2.4f, -2.4f, 0.f};
    std::vector<float> h_viewmat = {
        1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1
    };
    float *d_pm, *d_pr, *d_ps, *d_vm;
    float *d_T0, *d_m2d0, *d_dep0, *d_nm0;
    int32_t* d_rad0;
    CUDA_CHECK(cudaMalloc(&d_pm, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pr, 4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ps, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vm, 16*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T0, 9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m2d0, 2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dep0, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nm0, 3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rad0, 2*sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_pm, h_proj_means.data(), 3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pr, h_proj_rotation.data(), 4*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ps, h_proj_scaling.data(), 3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vm, h_viewmat.data(), 16*sizeof(float), cudaMemcpyHostToDevice));
    projection_2dgs_kernel<<<1,1>>>(
        d_pm, d_pr, d_ps, d_vm, fx, fy, cx, cy,
        0.2f, (int)W, (int)H,
        d_T0, d_m2d0, d_rad0, d_dep0, d_nm0, N
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_T(9), h_m2d_base(2), h_dep(1);
    std::vector<int32_t> h_rad(2);
    CUDA_CHECK(cudaMemcpy(h_T.data(),       d_T0,   9*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m2d_base.data(),d_m2d0, 2*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dep.data(),     d_dep0, sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rad.data(),     d_rad0, 2*sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_pm); cudaFree(d_pr); cudaFree(d_ps); cudaFree(d_vm);
    cudaFree(d_T0); cudaFree(d_m2d0); cudaFree(d_dep0); cudaFree(d_nm0); cudaFree(d_rad0);

    // Two synthetic cameras: same scene parameters, slightly different
    // screen-space positions + different per-camera color & target render.
    std::vector<std::vector<float>> h_m2d_cam = {
        { h_m2d_base[0] + 0.1f, h_m2d_base[1] - 0.05f },
        { h_m2d_base[0] - 0.2f, h_m2d_base[1] + 0.15f },
    };
    std::vector<std::vector<float>> h_colors_cam = {
        {0.75f, 0.20f, 0.10f},
        {0.20f, 0.60f, 0.85f},
    };
    std::vector<std::vector<float>> h_v_render(2,
        std::vector<float>(W * H * 3, 0.f));
    for (int c = 0; c < 2; c++) {
        for (uint32_t y = 5; y <= 10; y++) {
            for (uint32_t x = 5; x <= 10; x++) {
                size_t idx = (y * W + x) * 3;
                float scale = (c == 0) ? 1.f : -0.7f;
                h_v_render[c][idx + 0] = scale * 0.012f * ((int)x - 7);
                h_v_render[c][idx + 1] = scale * -0.017f * ((int)y - 7);
                h_v_render[c][idx + 2] = scale * 0.008f * ((int)x + (int)y - 14);
            }
        }
    }

    // Shared parameter under test: opacity. Same Gaussian in both cameras.
    std::vector<float> h_opacity = {0.3f};

    // Per-camera device buffers + a single shared opacity buffer.
    struct CamBuffers {
        float *means2d, *T, *colors, *depths, *normals, *gamma;
        int32_t* radii;
        float *render_colors, *render_alphas, *v_render;
        int32_t* last_ids;
        float *g_T, *g_opacity, *g_gamma, *g_colors, *g_normals, *g_m2d, *g_m2d_abs;
    };
    CamBuffers cb[2]{};
    float *d_opacity_shared = nullptr;
    CUDA_CHECK(cudaMalloc(&d_opacity_shared, sizeof(float)));
    for (int c = 0; c < 2; c++) {
        CUDA_CHECK(cudaMalloc(&cb[c].means2d, 2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].T,       9*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].colors,  3*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].depths,  sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].normals, 3*sizeof(float)));
        CUDA_CHECK(cudaMemset(cb[c].normals, 0, 3*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].gamma,   2*sizeof(float)));
        CUDA_CHECK(cudaMemset(cb[c].gamma, 0, 2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].radii,   2*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&cb[c].render_colors, W*H*3*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].render_alphas, W*H*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].last_ids,      W*H*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&cb[c].v_render,      W*H*3*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].g_T,       9*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].g_opacity, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].g_gamma,   2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].g_colors,  3*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].g_normals, 3*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].g_m2d,     2*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&cb[c].g_m2d_abs, 2*sizeof(float)));

        CUDA_CHECK(cudaMemcpy(cb[c].depths, h_dep.data(), sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cb[c].radii,  h_rad.data(), 2*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cb[c].T,      h_T.data(),   9*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cb[c].means2d,h_m2d_cam[c].data(),    2*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cb[c].colors, h_colors_cam[c].data(), 3*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cb[c].v_render, h_v_render[c].data(), W*H*3*sizeof(float), cudaMemcpyHostToDevice));
    }

    auto fwd_one = [&](int c) -> float {
        // T, m2d, colors, normals, gamma, radii, depths already on device.
        // Only opacity changes between FD evaluations, so refresh it each call.
        CUDA_CHECK(cudaMemcpy(d_opacity_shared, h_opacity.data(), sizeof(float), cudaMemcpyHostToDevice));
        TileIntersectBuffers tile = launch_tile_intersect(
            cb[c].means2d, cb[c].radii, cb[c].depths, N, W, H, TILE_SIZE
        );
        CUDA_CHECK(cudaMemset(cb[c].render_colors, 0, W*H*3*sizeof(float)));
        CUDA_CHECK(cudaMemset(cb[c].render_alphas, 0, W*H*sizeof(float)));
        launch_rasterize_fwd(
            cb[c].means2d, cb[c].T, d_opacity_shared, cb[c].gamma,
            cb[c].colors, cb[c].normals,
            tile.tile_offsets, tile.flatten_ids, tile.n_isects,
            W, H,
            cb[c].render_colors, cb[c].render_alphas,
            nullptr, nullptr, nullptr, nullptr, cb[c].last_ids
        );
        std::vector<float> out(W*H*3);
        CUDA_CHECK(cudaMemcpy(out.data(), cb[c].render_colors, W*H*3*sizeof(float), cudaMemcpyDeviceToHost));
        free_tile_intersect_buffers(tile);
        return dot_host(out, h_v_render[c]);
    };

    auto eval_summed = [&]() -> float { return fwd_one(0) + fwd_one(1); };

    (void)eval_summed();   // sanity warmup

    // Analytic: per-camera backward, accumulate g_opacity across cameras.
    float g_opacity_sum = 0.f;
    for (int c = 0; c < 2; c++) {
        CUDA_CHECK(cudaMemcpy(d_opacity_shared, h_opacity.data(), sizeof(float), cudaMemcpyHostToDevice));
        TileIntersectBuffers tile = launch_tile_intersect(
            cb[c].means2d, cb[c].radii, cb[c].depths, N, W, H, TILE_SIZE
        );
        CUDA_CHECK(cudaMemset(cb[c].render_colors, 0, W*H*3*sizeof(float)));
        CUDA_CHECK(cudaMemset(cb[c].render_alphas, 0, W*H*sizeof(float)));
        launch_rasterize_fwd(
            cb[c].means2d, cb[c].T, d_opacity_shared, cb[c].gamma,
            cb[c].colors, cb[c].normals,
            tile.tile_offsets, tile.flatten_ids, tile.n_isects,
            W, H,
            cb[c].render_colors, cb[c].render_alphas,
            nullptr, nullptr, nullptr, nullptr, cb[c].last_ids
        );
        CUDA_CHECK(cudaMemset(cb[c].g_T, 0, 9*sizeof(float)));
        CUDA_CHECK(cudaMemset(cb[c].g_opacity, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(cb[c].g_gamma, 0, 2*sizeof(float)));
        CUDA_CHECK(cudaMemset(cb[c].g_colors, 0, 3*sizeof(float)));
        CUDA_CHECK(cudaMemset(cb[c].g_normals, 0, 3*sizeof(float)));
        CUDA_CHECK(cudaMemset(cb[c].g_m2d, 0, 2*sizeof(float)));
        CUDA_CHECK(cudaMemset(cb[c].g_m2d_abs, 0, 2*sizeof(float)));
        launch_rasterize_bwd(
            cb[c].means2d, cb[c].T, d_opacity_shared, cb[c].gamma,
            cb[c].colors, cb[c].normals,
            tile.tile_offsets, tile.flatten_ids, tile.n_isects,
            cb[c].render_alphas, nullptr, cb[c].last_ids,
            cb[c].v_render, nullptr, nullptr, nullptr, nullptr,
            W, H,
            cb[c].g_T, cb[c].g_opacity, cb[c].g_gamma,
            cb[c].g_colors, cb[c].g_normals, cb[c].g_m2d, cb[c].g_m2d_abs
        );
        float g_op_c = 0.f;
        CUDA_CHECK(cudaMemcpy(&g_op_c, cb[c].g_opacity, sizeof(float), cudaMemcpyDeviceToHost));
        g_opacity_sum += g_op_c;
        free_tile_intersect_buffers(tile);
    }

    const float eps = 1e-3f;
    float fd = centered_diff(h_opacity, 0, eps, eval_summed);
    bool ok = check_close("opacity (sum-of-grads ≡ grad-of-sum)",
                          g_opacity_sum, fd, 7e-3f, 8e-2f);

    cudaFree(d_opacity_shared);
    for (int c = 0; c < 2; c++) {
        cudaFree(cb[c].means2d); cudaFree(cb[c].T);
        cudaFree(cb[c].colors); cudaFree(cb[c].depths); cudaFree(cb[c].normals); cudaFree(cb[c].gamma);
        cudaFree(cb[c].radii); cudaFree(cb[c].render_colors); cudaFree(cb[c].render_alphas);
        cudaFree(cb[c].last_ids); cudaFree(cb[c].v_render);
        cudaFree(cb[c].g_T); cudaFree(cb[c].g_opacity); cudaFree(cb[c].g_gamma);
        cudaFree(cb[c].g_colors); cudaFree(cb[c].g_normals); cudaFree(cb[c].g_m2d); cudaFree(cb[c].g_m2d_abs);
    }

    printf("%s\n\n", ok ? "Split-batch linearity passed."
                        : "Split-batch linearity FAILED.");
    return ok;
}

// ─── M5 hash grid gradcheck ───────────────────────────────────────────────────
//
// Backward-by-finite-difference for kernels/hash_grid.cu. Strategy:
//
//   1. Tiny config (N=3 positions, L=2 levels, F=2 features, T=16 slots).
//   2. Forward: features[N, L, F] = hash_grid_fwd(positions, grid).
//   3. Define scalar objective L = <features, v_features>, where v_features
//      is a fixed pseudo-random upstream gradient.
//   4. Run hash_grid_bwd → grad_grid.
//   5. For each grid entry that received a non-zero gradient (i.e. was
//      actually touched by one of the 24 corner reads), compare the
//      analytic grad_grid entry against the centered finite difference of
//      L w.r.t. that entry.
//
// Positions are picked so that several Gaussians land in overlapping
// hash slots at the coarse level, which validates that the atomic-add
// scatter accumulates correctly.
static bool run_hash_grid_gradcheck() {
    printf("=== Hash Grid Gradient Check ===\n");

    constexpr int N = 3;
    constexpr int L = 2;
    constexpr int F = 2;
    constexpr uint32_t T = 16;          // power of two

    std::vector<float> h_positions = {
        0.10f, 0.20f, 0.30f,
        0.55f, 0.40f, 0.65f,
        0.85f, 0.75f, 0.15f,
    };
    std::vector<float> h_scales = hash_grid_build_level_scales(L, 4.f, 8.f);

    // Deterministic non-trivial grid.
    std::vector<float> h_grid((size_t)L * T * F);
    for (int l = 0; l < L; l++)
        for (uint32_t s = 0; s < T; s++)
            for (int f = 0; f < F; f++)
                h_grid[((size_t)l * T + s) * F + f] =
                    0.3f * std::sin((float)(l * 31 + (int)s * 7 + f * 11));

    std::vector<float> h_v_feat((size_t)N * L * F);
    for (size_t i = 0; i < h_v_feat.size(); i++)
        h_v_feat[i] = 0.7f * std::cos(0.31f * (float)i + 0.05f);

    float *d_pos, *d_grid, *d_scales, *d_feat;
    float *d_v_feat, *d_grad_grid;
    CUDA_CHECK(cudaMalloc(&d_pos,    (size_t)N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid,   (size_t)L * T * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, (size_t)L * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_feat,   (size_t)N * L * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_feat, (size_t)N * L * F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_grid, (size_t)L * T * F * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_pos,    h_positions.data(), (size_t)N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(),    (size_t)L * sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_feat, h_v_feat.data(),    (size_t)N * L * F * sizeof(float), cudaMemcpyHostToDevice));

    auto eval = [&]() -> float {
        CUDA_CHECK(cudaMemcpy(d_grid, h_grid.data(),
                              (size_t)L * T * F * sizeof(float), cudaMemcpyHostToDevice));
        launch_hash_grid_fwd(d_pos, d_grid, d_scales, d_feat, N, L, F, T);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out_feat((size_t)N * L * F);
        CUDA_CHECK(cudaMemcpy(out_feat.data(), d_feat,
                              (size_t)N * L * F * sizeof(float), cudaMemcpyDeviceToHost));
        return dot_host(out_feat, h_v_feat);
    };

    // Analytic gradient.
    (void)eval();   // also leaves d_grid populated with current h_grid
    CUDA_CHECK(cudaMemset(d_grad_grid, 0, (size_t)L * T * F * sizeof(float)));
    launch_hash_grid_bwd(d_pos, d_scales, d_v_feat, d_grad_grid, N, L, F, T);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_grad_grid((size_t)L * T * F);
    CUDA_CHECK(cudaMemcpy(h_grad_grid.data(), d_grad_grid,
                          (size_t)L * T * F * sizeof(float), cudaMemcpyDeviceToHost));

    // FD-check every entry that received a non-zero analytic gradient. With
    // 24 corner reads (N*8) per level there are at most 48 touched entries
    // across both levels in this scene; usually fewer because of slot
    // collisions, which is the interesting case for atomicAdd correctness.
    const float eps = 1e-3f;
    bool ok = true;
    int checked = 0;
    for (int l = 0; l < L; l++) {
        for (uint32_t s = 0; s < T; s++) {
            for (int f = 0; f < F; f++) {
                size_t idx = ((size_t)l * T + s) * F + f;
                float g_an = h_grad_grid[idx];
                if (std::fabs(g_an) < 1e-7f) continue;
                float g_fd = centered_diff(h_grid, (int)idx, eps, eval);
                char label[64];
                std::snprintf(label, sizeof(label), "L=%d s=%2u f=%d", l, s, f);
                ok &= check_close(label, g_an, g_fd, 1e-4f, 5e-3f);
                checked++;
            }
        }
    }
    printf("  (checked %d grid entries with non-zero analytic gradient)\n", checked);

    cudaFree(d_pos); cudaFree(d_grid); cudaFree(d_scales); cudaFree(d_feat);
    cudaFree(d_v_feat); cudaFree(d_grad_grid);

    printf("%s\n\n", ok ? "Hash grid gradients passed." : "Hash grid gradients FAILED.");
    return ok;
}

// ─── M5 MLP gradcheck ─────────────────────────────────────────────────────────
//
// Two-layer fused MLP from kernels/mlp.cu:
//   h = ReLU(W1 · x + b1)
//   y =      W2 · h + b2
//
// Test plan
//   * Pick a tiny config (N=4 Gaussians, D_in=8, D_hid=6, D_out=3) with
//     biased inputs so the ReLU mask has a mix of fired / dead units.
//   * Scalar objective: L = <y, v_y>.
//   * Compare analytic backward (grad_W1/W2/b1/b2/features) vs centered
//     finite differences for one entry of each parameter group, plus a
//     few `features` entries.
//
// The bias on inputs and `b1` is set so ~half the hidden units fire,
// exercising both branches of the ReLU mask.
static bool run_mlp_gradcheck() {
    printf("=== MLP Gradient Check ===\n");

    constexpr int N     = 4;
    constexpr int D_in  = 8;
    constexpr int D_hid = 6;
    constexpr int D_out = 3;

    std::vector<float> h_x   ((size_t)N * D_in);
    std::vector<float> h_W1  ((size_t)D_hid * D_in);
    std::vector<float> h_b1  ((size_t)D_hid);
    std::vector<float> h_W2  ((size_t)D_out * D_hid);
    std::vector<float> h_b2  ((size_t)D_out);
    std::vector<float> h_v_y ((size_t)N * D_out);

    for (size_t i = 0; i < h_x.size();   i++) h_x[i]  = 0.4f * std::sin(0.31f * (float)i + 0.6f) + 0.1f;
    for (size_t i = 0; i < h_W1.size();  i++) h_W1[i] = 0.5f * std::sin(1.13f * (float)i + 0.1f);
    for (size_t i = 0; i < h_b1.size();  i++) h_b1[i] = 0.2f * std::cos(0.7f * (float)i) - 0.05f;  // mix of pos/neg
    for (size_t i = 0; i < h_W2.size();  i++) h_W2[i] = 0.4f * std::cos(0.9f  * (float)i + 0.2f);
    for (size_t i = 0; i < h_b2.size();  i++) h_b2[i] = 0.02f * (float)(i + 1);
    for (size_t i = 0; i < h_v_y.size(); i++) h_v_y[i] = 0.6f * std::sin(0.7f * (float)i + 0.3f);

    float *d_x, *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_h, *d_y, *d_v_y;
    float *d_g_x, *d_g_W1, *d_g_b1, *d_g_W2, *d_g_b2;

    CUDA_CHECK(cudaMalloc(&d_x,   h_x.size()  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1,  h_W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1,  h_b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2,  h_W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2,  h_b2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h,   (size_t)N * D_hid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,   (size_t)N * D_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_y, h_v_y.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_x, (size_t)N * D_in  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_W1, h_W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_b1, h_b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_W2, h_W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_g_b2, h_b2.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_v_y, h_v_y.data(), h_v_y.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto eval = [&]() -> float {
        CUDA_CHECK(cudaMemcpy(d_x,  h_x.data(),  h_x.size()  * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
        launch_mlp_fwd(d_x, d_W1, d_b1, d_W2, d_b2, d_h, d_y,
                       N, D_in, D_hid, D_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out((size_t)N * D_out);
        CUDA_CHECK(cudaMemcpy(out.data(), d_y, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return dot_host(out, h_v_y);
    };

    // Analytic gradient (cache hidden via the forward call already done).
    (void)eval();
    CUDA_CHECK(cudaMemset(d_g_x,  0, (size_t)N * D_in  * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_W1, 0, h_W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_b1, 0, h_b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_W2, 0, h_W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g_b2, 0, h_b2.size() * sizeof(float)));
    launch_mlp_bwd(d_x, d_h, d_W1, d_W2, d_v_y,
                   d_g_x, d_g_W1, d_g_b1, d_g_W2, d_g_b2,
                   N, D_in, D_hid, D_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> g_x((size_t)N * D_in),
                       g_W1(h_W1.size()), g_b1(h_b1.size()),
                       g_W2(h_W2.size()), g_b2(h_b2.size());
    CUDA_CHECK(cudaMemcpy(g_x.data(),  d_g_x,  g_x.size()  * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_W1.data(), d_g_W1, g_W1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_b1.data(), d_g_b1, g_b1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_W2.data(), d_g_W2, g_W2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g_b2.data(), d_g_b2, g_b2.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const float eps     = 1e-3f;
    const float abs_tol = 5e-4f;
    const float rel_tol = 5e-3f;
    bool ok = true;

    // Spot-check entries from each parameter group + features. We do not
    // FD-check *every* entry — the test would be N * (D_in + ...) forward
    // calls. A representative sample is sufficient: if any entry of a
    // group is correct, the kernel implements that gradient correctly.
    ok &= check_close("g_W1[0,0]", g_W1[0],
                      centered_diff(h_W1, 0, eps, eval), abs_tol, rel_tol);
    ok &= check_close("g_W1[3,5]", g_W1[3 * D_in + 5],
                      centered_diff(h_W1, 3 * D_in + 5, eps, eval), abs_tol, rel_tol);
    ok &= check_close("g_W2[1,4]", g_W2[1 * D_hid + 4],
                      centered_diff(h_W2, 1 * D_hid + 4, eps, eval), abs_tol, rel_tol);
    ok &= check_close("g_b1[2]",   g_b1[2],
                      centered_diff(h_b1, 2, eps, eval), abs_tol, rel_tol);
    ok &= check_close("g_b1[5]",   g_b1[5],
                      centered_diff(h_b1, 5, eps, eval), abs_tol, rel_tol);
    ok &= check_close("g_b2[0]",   g_b2[0],
                      centered_diff(h_b2, 0, eps, eval), abs_tol, rel_tol);
    ok &= check_close("g_x[0,3]",  g_x[0 * D_in + 3],
                      centered_diff(h_x, 0 * D_in + 3, eps, eval), abs_tol, rel_tol);
    ok &= check_close("g_x[2,7]",  g_x[2 * D_in + 7],
                      centered_diff(h_x, 2 * D_in + 7, eps, eval), abs_tol, rel_tol);

    cudaFree(d_x); cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_h); cudaFree(d_y); cudaFree(d_v_y);
    cudaFree(d_g_x); cudaFree(d_g_W1); cudaFree(d_g_b1);
    cudaFree(d_g_W2); cudaFree(d_g_b2);

    printf("%s\n\n", ok ? "MLP gradients passed." : "MLP gradients FAILED.");
    return ok;
}

// ─── M5 nexel-color end-to-end gradcheck ──────────────────────────────────────
//
// Chains hash_grid_fwd → mlp_fwd → +0.5 clamp via the nexel_color
// orchestrator and back through the corresponding backward calls.
// Validates that:
//   (a) the +0.5 clamp + pass-through backward kernel agrees with FD;
//   (b) the orchestrator's hand-off of `mlp_out` as backward scratch
//       does not stomp on values needed downstream;
//   (c) the AppearanceField / Grads lifecycle types behave correctly.
//
// We deliberately seed the grid with small noise (not the production
// constant-0.01 init) so the FD perturbation is non-degenerate.
static bool run_nexel_color_gradcheck() {
    printf("=== Nexel Color End-to-End Gradient Check ===\n");

    constexpr int N     = 3;
    constexpr int L     = 2;
    constexpr int T     = 16;
    constexpr int F     = 2;
    constexpr int D_hid = 4;
    constexpr int D_out = 3;

    AppearanceField af;
    af.allocate(L, T, F, D_hid, D_out, 4.f, 8.f);

    // Seed with non-trivial noise so the FD signal is non-zero.
    std::vector<float> h_grid((size_t)L * T * F);
    for (size_t i = 0; i < h_grid.size(); i++)
        h_grid[i] = 0.2f * std::sin(0.31f * (float)i + 0.6f);
    std::vector<float> h_W1((size_t)D_hid * af.D_in);
    for (size_t i = 0; i < h_W1.size(); i++)
        h_W1[i] = 0.5f * std::sin(1.13f * (float)i);
    std::vector<float> h_b1((size_t)D_hid);
    for (size_t i = 0; i < h_b1.size(); i++)
        h_b1[i] = 0.1f * std::cos(0.7f * (float)i) - 0.04f;
    std::vector<float> h_W2((size_t)D_out * D_hid);
    for (size_t i = 0; i < h_W2.size(); i++)
        h_W2[i] = 0.4f * std::cos(0.9f * (float)i);
    std::vector<float> h_b2((size_t)D_out);
    for (size_t i = 0; i < h_b2.size(); i++)
        h_b2[i] = 0.05f * (float)i;
    CUDA_CHECK(cudaMemcpy(af.grid, h_grid.data(), h_grid.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(af.W1,   h_W1.data(),   h_W1.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(af.b1,   h_b1.data(),   h_b1.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(af.W2,   h_W2.data(),   h_W2.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(af.b2,   h_b2.data(),   h_b2.size()   * sizeof(float), cudaMemcpyHostToDevice));

    AppearanceWorkspace ws;
    AppearanceGrads grads;
    grads.allocate(af);

    std::vector<float> h_means = {
        0.13f, 0.27f, 0.41f,
        0.62f, 0.34f, 0.58f,
        0.85f, 0.71f, 0.19f,
    };
    std::vector<float> h_v_colors((size_t)N * 3);
    for (size_t i = 0; i < h_v_colors.size(); i++)
        h_v_colors[i] = 0.5f * std::sin(0.7f * (float)i + 0.4f);

    float *d_means, *d_colors, *d_v_colors;
    CUDA_CHECK(cudaMalloc(&d_means,    h_means.size()    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colors,   h_v_colors.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_colors, h_v_colors.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_means,    h_means.data(),    h_means.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_colors, h_v_colors.data(), h_v_colors.size()* sizeof(float), cudaMemcpyHostToDevice));

    // Push current host-side grid into device before each FD eval (the
    // forward only consumes from device, so the FD perturbation must
    // round-trip via cudaMemcpy).
    auto eval = [&]() -> float {
        CUDA_CHECK(cudaMemcpy(af.grid, h_grid.data(),
                              h_grid.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        // Non-view-dep mode in this check; reuse d_means for both raw +
        // normalised position arguments (no bbox normalisation here —
        // means are already in [0,1] for this test).
        nexel_color_forward(d_means, af, ws, d_means, d_colors, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out((size_t)N * 3);
        CUDA_CHECK(cudaMemcpy(out.data(), d_colors,
                              out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return dot_host(out, h_v_colors);
    };

    // Analytic: forward once, then backward.
    (void)eval();
    grads.zero_params();
    nexel_color_backward(d_means, af, ws, d_means, d_v_colors, grads, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> g_grid((size_t)L * T * F);
    CUDA_CHECK(cudaMemcpy(g_grid.data(), grads.grad_grid,
                          g_grid.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // FD-check the grid entries that received a non-zero analytic gradient.
    const float eps = 1e-3f;
    bool ok = true;
    int checked = 0;
    for (size_t idx = 0; idx < g_grid.size(); idx++) {
        if (std::fabs(g_grid[idx]) < 1e-7f) continue;
        float g_fd = centered_diff(h_grid, (int)idx, eps, eval);
        char label[64];
        std::snprintf(label, sizeof(label), "grid[%zu]", idx);
        ok &= check_close(label, g_grid[idx], g_fd, 5e-4f, 5e-3f);
        checked++;
        if (checked >= 8) break;   // 8 entries is plenty for a regression test
    }
    printf("  (FD-checked %d non-zero grid entries)\n", checked);

    cudaFree(d_means); cudaFree(d_colors); cudaFree(d_v_colors);
    af.free(); ws.free(); grads.free();

    printf("%s\n\n", ok ? "Nexel-color end-to-end passed."
                        : "Nexel-color end-to-end FAILED.");
    return ok;
}

// ─── M5 view-dep nexel color gradcheck ───────────────────────────────────────
//
// Same machinery as run_nexel_color_gradcheck but with view_dep=true:
// MLP outputs D_out=48 packed SH coefficients, the orchestrator runs
// nexel_sh_eval_packed_fwd which mixes them with a per-Gaussian view
// direction to produce RGB. Verifies that the new SH-basis kernels
// integrate correctly with the MLP and hash-grid backward chain.
static bool run_nexel_color_view_dep_gradcheck() {
    printf("=== Nexel Color View-Dependent (SH) Gradient Check ===\n");

    constexpr int N     = 3;
    constexpr int L     = 2;
    constexpr int T     = 16;
    constexpr int F     = 2;
    constexpr int D_hid = 4;
    constexpr int D_out = 48;     // 3 DC + 45 directional

    AppearanceField af;
    af.allocate(L, T, F, D_hid, D_out, 4.f, 8.f);

    // Seed grid + W1 + W2 with non-trivial noise. For the SH path it is
    // important that the upper rows of W2 (shN, rows 3+) are NOT zero —
    // otherwise the directional terms produce zero gradient through the
    // FD perturbations and the check becomes degenerate.
    std::vector<float> h_grid((size_t)L * T * F);
    for (size_t i = 0; i < h_grid.size(); i++)
        h_grid[i] = 0.15f * std::sin(0.31f * (float)i + 0.6f);
    std::vector<float> h_W1((size_t)D_hid * af.D_in);
    for (size_t i = 0; i < h_W1.size(); i++)
        h_W1[i] = 0.5f * std::sin(1.13f * (float)i);
    std::vector<float> h_b1((size_t)D_hid);
    for (size_t i = 0; i < h_b1.size(); i++)
        h_b1[i] = 0.1f * std::cos(0.7f * (float)i) - 0.04f;
    std::vector<float> h_W2((size_t)D_out * D_hid);
    for (size_t i = 0; i < h_W2.size(); i++)
        h_W2[i] = 0.05f * std::cos(0.9f * (float)i);    // small + non-zero
    std::vector<float> h_b2((size_t)D_out, 0.f);
    CUDA_CHECK(cudaMemcpy(af.grid, h_grid.data(), h_grid.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(af.W1,   h_W1.data(),   h_W1.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(af.b1,   h_b1.data(),   h_b1.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(af.W2,   h_W2.data(),   h_W2.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(af.b2,   h_b2.data(),   h_b2.size()   * sizeof(float), cudaMemcpyHostToDevice));

    AppearanceWorkspace ws;
    AppearanceGrads grads;
    grads.allocate(af);

    // Place means in world-space coords (well away from clamp) and use
    // a non-trivial camera so the view direction is distinct per
    // Gaussian.
    std::vector<float> h_means = {
        0.3f, 0.2f, 0.4f,
        0.6f, 0.5f, 0.7f,
        0.85f, 0.7f, 0.2f,
    };
    const float cam_x = -0.5f, cam_y = -0.3f, cam_z = 0.1f;
    std::vector<float> h_v_colors((size_t)N * 3);
    for (size_t i = 0; i < h_v_colors.size(); i++)
        h_v_colors[i] = 0.5f * std::sin(0.7f * (float)i + 0.4f);

    float *d_means, *d_colors, *d_v_colors;
    CUDA_CHECK(cudaMalloc(&d_means,    h_means.size()    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colors,   h_v_colors.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_colors, h_v_colors.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_means,    h_means.data(),    h_means.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_colors, h_v_colors.data(), h_v_colors.size()* sizeof(float), cudaMemcpyHostToDevice));

    auto eval = [&]() -> float {
        // Refresh the grid from the (potentially perturbed) host copy.
        CUDA_CHECK(cudaMemcpy(af.grid, h_grid.data(),
                              h_grid.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        nexel_color_forward(d_means, af, ws, d_means, d_colors, N,
                            cam_x, cam_y, cam_z);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out((size_t)N * 3);
        CUDA_CHECK(cudaMemcpy(out.data(), d_colors,
                              out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return dot_host(out, h_v_colors);
    };

    (void)eval();
    grads.zero_params();
    nexel_color_backward(d_means, af, ws, d_means, d_v_colors, grads, N,
                         cam_x, cam_y, cam_z);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> g_grid((size_t)L * T * F);
    CUDA_CHECK(cudaMemcpy(g_grid.data(), grads.grad_grid,
                          g_grid.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const float eps = 1e-3f;
    bool ok = true;
    int checked = 0;
    for (size_t idx = 0; idx < g_grid.size(); idx++) {
        if (std::fabs(g_grid[idx]) < 1e-7f) continue;
        float g_fd = centered_diff(h_grid, (int)idx, eps, eval);
        char label[64];
        std::snprintf(label, sizeof(label), "grid[%zu]", idx);
        ok &= check_close(label, g_grid[idx], g_fd, 5e-4f, 1e-2f);
        checked++;
        if (checked >= 8) break;
    }
    printf("  (FD-checked %d non-zero grid entries through SH path)\n", checked);

    cudaFree(d_means); cudaFree(d_colors); cudaFree(d_v_colors);
    af.free(); ws.free(); grads.free();

    printf("%s\n\n", ok ? "Nexel-color view-dep passed."
                        : "Nexel-color view-dep FAILED.");
    return ok;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    bool ok = true;
    ok &= run_projection_gradcheck();
    ok &= run_sh_gradcheck();
    ok &= run_rasterize_gradcheck();
    ok &= run_rasterize_aux_gradcheck();
    ok &= run_geometry_loss_gradcheck();
    ok &= run_split_batch_linearity_check();
    ok &= run_hash_grid_gradcheck();
    ok &= run_mlp_gradcheck();
    ok &= run_nexel_color_gradcheck();
    ok &= run_nexel_color_view_dep_gradcheck();

    printf("%s\n", ok ? "All gradient checks passed." : "SOME GRADIENT CHECKS FAILED.");
    return ok ? 0 : 1;
}
