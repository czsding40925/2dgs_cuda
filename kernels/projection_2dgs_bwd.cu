// projection_2dgs_bwd.cu
//
// Backward pass for the 2DGS projection stage.
//
// Inputs:
//   - forward parameters: means, rotation (raw quat), scaling (raw log-scale)
//   - camera intrinsics / viewmat
//   - forward outputs: ray_transforms, radii
//   - upstream gradients: v_ray_transforms, v_means2d, v_depths, v_normals
//
// Outputs:
//   - v_means    [N, 3]
//   - v_rotation [N, 4]   raw quaternion gradient
//   - v_scaling  [N, 3]   raw log-scale gradient
//
// This matches the project forward path in projection_2dgs.cu:
//   raw quat  --normalize--> R
//   raw scale --exp-------> (sx, sy)
//   (R, s, mean) --view/intrinsics--> ray_transforms, means2d, depths, normals
//
// The `means2d -> ray_transforms` VJP follows the same closed-form derivative
// used by gsplat's Projection2DGS helper. The rest of the chain is written
// directly against this project's row-major / row-vector conventions.

#include "splat_data.cuh"

#include <cmath>
#include <cstdio>

namespace {

struct proj_bwd_f3 {
    float x, y, z;
};

struct proj_bwd_f4 {
    float w, x, y, z;
};

__device__ inline float dot3_pb(proj_bwd_f3 a, proj_bwd_f3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline void quat_to_rotmat_with_cache(
    const float* q_raw,
    float* R,
    proj_bwd_f4& q_norm,
    float& inv_norm
) {
    float w = q_raw[0];
    float x = q_raw[1];
    float y = q_raw[2];
    float z = q_raw[3];

    inv_norm = rsqrtf(w*w + x*x + y*y + z*z);
    w *= inv_norm;
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    q_norm = {w, x, y, z};

    float x2 = x*x, y2 = y*y, z2 = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;

    // Row-major.
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

__device__ inline void quat_to_rotmat_vjp_raw(
    const proj_bwd_f4 q_norm,
    float inv_norm,
    const float* v_R,
    float* v_q_raw
) {
    const float w = q_norm.w;
    const float x = q_norm.x;
    const float y = q_norm.y;
    const float z = q_norm.z;

    const float vR00 = v_R[0], vR01 = v_R[1], vR02 = v_R[2];
    const float vR10 = v_R[3], vR11 = v_R[4], vR12 = v_R[5];
    const float vR20 = v_R[6], vR21 = v_R[7], vR22 = v_R[8];

    proj_bwd_f4 v_qn{
        2.f * (x * (vR21 - vR12) +
               y * (vR02 - vR20) +
               z * (vR10 - vR01)),
        2.f * (-2.f * x * (vR11 + vR22) +
               y * (vR01 + vR10) +
               z * (vR02 + vR20) +
               w * (vR21 - vR12)),
        2.f * (x * (vR01 + vR10) -
               2.f * y * (vR00 + vR22) +
               z * (vR12 + vR21) +
               w * (vR02 - vR20)),
        2.f * (x * (vR02 + vR20) +
               y * (vR12 + vR21) -
               2.f * z * (vR00 + vR11) +
               w * (vR10 - vR01))
    };

    const float dot_vq_q = v_qn.w * w + v_qn.x * x + v_qn.y * y + v_qn.z * z;
    v_q_raw[0] = (v_qn.w - dot_vq_q * w) * inv_norm;
    v_q_raw[1] = (v_qn.x - dot_vq_q * x) * inv_norm;
    v_q_raw[2] = (v_qn.y - dot_vq_q * y) * inv_norm;
    v_q_raw[3] = (v_qn.z - dot_vq_q * z) * inv_norm;
}

} // namespace

__global__ void projection_2dgs_bwd_kernel(
    const float* __restrict__ means,          // [N, 3]
    const float* __restrict__ rotation,       // [N, 4] raw quat
    const float* __restrict__ scaling,        // [N, 3] raw log-scale
    const float* __restrict__ viewmat,        // [4, 4]
    float fx, float fy, float cx, float cy,
    const float* __restrict__ ray_transforms, // [N, 9]
    const int32_t* __restrict__ radii,        // [N, 2]
    const float* __restrict__ v_ray_transforms, // [N, 9]
    const float* __restrict__ v_means2d,      // [N, 2], optional
    const float* __restrict__ v_depths,       // [N], optional
    const float* __restrict__ v_normals,      // [N, 3], optional
    float* __restrict__ v_means,              // [N, 3]
    float* __restrict__ v_rotation,           // [N, 4]
    float* __restrict__ v_scaling,            // [N, 3]
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (radii[i*2] <= 0 || radii[i*2 + 1] <= 0) {
        v_means[i*3 + 0] = 0.f;
        v_means[i*3 + 1] = 0.f;
        v_means[i*3 + 2] = 0.f;
        v_rotation[i*4 + 0] = 0.f;
        v_rotation[i*4 + 1] = 0.f;
        v_rotation[i*4 + 2] = 0.f;
        v_rotation[i*4 + 3] = 0.f;
        v_scaling[i*3 + 0] = 0.f;
        v_scaling[i*3 + 1] = 0.f;
        v_scaling[i*3 + 2] = 0.f;
        return;
    }

    const float* p_raw = means + i * 3;
    const float* q_raw = rotation + i * 4;
    const float* s_raw = scaling + i * 3;
    const float* T = ray_transforms + i * 9;

    proj_bwd_f4 q_norm{};
    float inv_q_norm = 0.f;
    float Rq[9];
    quat_to_rotmat_with_cache(q_raw, Rq, q_norm, inv_q_norm);

    const float sx = expf(s_raw[0]);
    const float sy = expf(s_raw[1]);

    proj_bwd_f3 col0{Rq[0], Rq[3], Rq[6]};
    proj_bwd_f3 col1{Rq[1], Rq[4], Rq[7]};
    proj_bwd_f3 col2{Rq[2], Rq[5], Rq[8]};

    proj_bwd_f3 u_world{col0.x * sx, col0.y * sx, col0.z * sx};
    proj_bwd_f3 v_world{col1.x * sy, col1.y * sy, col1.z * sy};
    proj_bwd_f3 n_world{col2.x, col2.y, col2.z};

    proj_bwd_f3 u_cam{}, v_cam{}, p_cam{}, n_cam{};
    for (int j = 0; j < 3; ++j) {
        u_cam.x += (&u_world.x)[j] * viewmat[j * 4 + 0];
        u_cam.y += (&u_world.x)[j] * viewmat[j * 4 + 1];
        u_cam.z += (&u_world.x)[j] * viewmat[j * 4 + 2];

        v_cam.x += (&v_world.x)[j] * viewmat[j * 4 + 0];
        v_cam.y += (&v_world.x)[j] * viewmat[j * 4 + 1];
        v_cam.z += (&v_world.x)[j] * viewmat[j * 4 + 2];

        p_cam.x += p_raw[j] * viewmat[j * 4 + 0];
        p_cam.y += p_raw[j] * viewmat[j * 4 + 1];
        p_cam.z += p_raw[j] * viewmat[j * 4 + 2];

        n_cam.x += (&n_world.x)[j] * viewmat[j * 4 + 0];
        n_cam.y += (&n_world.x)[j] * viewmat[j * 4 + 1];
        n_cam.z += (&n_world.x)[j] * viewmat[j * 4 + 2];
    }
    p_cam.x += viewmat[12];
    p_cam.y += viewmat[13];
    p_cam.z += viewmat[14];

    float vT[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k)
        vT[k] = v_ray_transforms[i * 9 + k];

    if (v_depths != nullptr)
        vT[8] += v_depths[i];

    if (v_means2d != nullptr) {
        const float vmx = v_means2d[i * 2 + 0];
        const float vmy = v_means2d[i * 2 + 1];
        if (vmx != 0.f || vmy != 0.f) {
            const float distance = T[6] * T[6] + T[7] * T[7] - T[8] * T[8];
            if (distance != 0.f) {
                const float inv_dist = 1.f / distance;
                const float mx = (T[0] * T[6] + T[1] * T[7] - T[2] * T[8]) * inv_dist;
                const float my = (T[3] * T[6] + T[4] * T[7] - T[5] * T[8]) * inv_dist;

                vT[0] += vmx * (T[6] * inv_dist);
                vT[1] += vmx * (T[7] * inv_dist);
                vT[2] += vmx * (-T[8] * inv_dist);
                vT[3] += vmy * (T[6] * inv_dist);
                vT[4] += vmy * (T[7] * inv_dist);
                vT[5] += vmy * (-T[8] * inv_dist);

                vT[6] += vmx * ((T[0] - 2.f * mx * T[6]) * inv_dist) +
                         vmy * ((T[3] - 2.f * my * T[6]) * inv_dist);
                vT[7] += vmx * ((T[1] - 2.f * mx * T[7]) * inv_dist) +
                         vmy * ((T[4] - 2.f * my * T[7]) * inv_dist);
                vT[8] += vmx * ((-T[2] + 2.f * mx * T[8]) * inv_dist) +
                         vmy * ((-T[5] + 2.f * my * T[8]) * inv_dist);
            }
        }
    }

    proj_bwd_f3 v_u_cam{
        fx * vT[0],
        fy * vT[3],
        cx * vT[0] + cy * vT[3] + vT[6]
    };
    proj_bwd_f3 v_v_cam{
        fx * vT[1],
        fy * vT[4],
        cx * vT[1] + cy * vT[4] + vT[7]
    };
    proj_bwd_f3 v_p_cam{
        fx * vT[2],
        fy * vT[5],
        cx * vT[2] + cy * vT[5] + vT[8]
    };
    proj_bwd_f3 v_n_cam{0.f, 0.f, 0.f};

    if (v_normals != nullptr) {
        const float flip =
            (-(n_cam.x * p_cam.x + n_cam.y * p_cam.y + n_cam.z * p_cam.z) > 0.f)
                ? 1.f
                : -1.f;
        v_n_cam.x += flip * v_normals[i * 3 + 0];
        v_n_cam.y += flip * v_normals[i * 3 + 1];
        v_n_cam.z += flip * v_normals[i * 3 + 2];
    }

    proj_bwd_f3 v_u_world{0.f, 0.f, 0.f};
    proj_bwd_f3 v_v_world{0.f, 0.f, 0.f};
    proj_bwd_f3 v_n_world{0.f, 0.f, 0.f};
    proj_bwd_f3 v_mean{0.f, 0.f, 0.f};

    for (int k = 0; k < 3; ++k) {
        (&v_u_world.x)[k] =
            v_u_cam.x * viewmat[k * 4 + 0] +
            v_u_cam.y * viewmat[k * 4 + 1] +
            v_u_cam.z * viewmat[k * 4 + 2];
        (&v_v_world.x)[k] =
            v_v_cam.x * viewmat[k * 4 + 0] +
            v_v_cam.y * viewmat[k * 4 + 1] +
            v_v_cam.z * viewmat[k * 4 + 2];
        (&v_n_world.x)[k] =
            v_n_cam.x * viewmat[k * 4 + 0] +
            v_n_cam.y * viewmat[k * 4 + 1] +
            v_n_cam.z * viewmat[k * 4 + 2];
        (&v_mean.x)[k] =
            v_p_cam.x * viewmat[k * 4 + 0] +
            v_p_cam.y * viewmat[k * 4 + 1] +
            v_p_cam.z * viewmat[k * 4 + 2];
    }

    float v_Rq[9] = {0.f};
    v_Rq[0] += v_u_world.x * sx;
    v_Rq[3] += v_u_world.y * sx;
    v_Rq[6] += v_u_world.z * sx;
    v_Rq[1] += v_v_world.x * sy;
    v_Rq[4] += v_v_world.y * sy;
    v_Rq[7] += v_v_world.z * sy;
    v_Rq[2] += v_n_world.x;
    v_Rq[5] += v_n_world.y;
    v_Rq[8] += v_n_world.z;

    const float v_sx = dot3_pb(v_u_world, col0) * sx;
    const float v_sy = dot3_pb(v_v_world, col1) * sy;

    float v_q_raw[4];
    quat_to_rotmat_vjp_raw(q_norm, inv_q_norm, v_Rq, v_q_raw);

    v_means[i*3 + 0] = v_mean.x;
    v_means[i*3 + 1] = v_mean.y;
    v_means[i*3 + 2] = v_mean.z;
    v_rotation[i*4 + 0] = v_q_raw[0];
    v_rotation[i*4 + 1] = v_q_raw[1];
    v_rotation[i*4 + 2] = v_q_raw[2];
    v_rotation[i*4 + 3] = v_q_raw[3];
    v_scaling[i*3 + 0] = v_sx;
    v_scaling[i*3 + 1] = v_sy;
    v_scaling[i*3 + 2] = 0.f;
}

void launch_projection_2dgs_bwd(
    const float* d_means,
    const float* d_rotation,
    const float* d_scaling,
    const float* d_viewmat,
    float fx, float fy, float cx, float cy,
    const float* d_ray_transforms,
    const int32_t* d_radii,
    const float* d_v_ray_transforms,
    const float* d_v_means2d,
    const float* d_v_depths,
    const float* d_v_normals,
    float* d_v_means,
    float* d_v_rotation,
    float* d_v_scaling,
    int N
) {
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    projection_2dgs_bwd_kernel<<<grid, block>>>(
        d_means, d_rotation, d_scaling, d_viewmat,
        fx, fy, cx, cy,
        d_ray_transforms, d_radii,
        d_v_ray_transforms, d_v_means2d, d_v_depths, d_v_normals,
        d_v_means, d_v_rotation, d_v_scaling, N
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

#ifndef INCLUDED_AS_HEADER
#define PROJECTION_2DGS_BWD_SET_HEADER_GUARD
#define INCLUDED_AS_HEADER
#include "projection_2dgs.cu"
#undef INCLUDED_AS_HEADER
#undef PROJECTION_2DGS_BWD_SET_HEADER_GUARD

int main() {
    printf("=== projection_2dgs_bwd smoke test ===\n\n");

    const int N = 1;
    const int W = 512, H = 512;

    float h_means[N * 3] = {0.f, 0.f, 2.f};
    float h_rotation[N * 4] = {1.f, 0.f, 0.f, 0.f};
    float h_scaling[N * 3] = {0.f, 0.f, 0.f};
    float h_viewmat[16] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    float *d_means, *d_rotation, *d_scaling, *d_viewmat;
    float *d_ray_transforms, *d_means2d, *d_depths, *d_normals;
    int32_t* d_radii;
    float *d_v_ray_transforms, *d_v_means2d, *d_v_depths, *d_v_normals;
    float *d_v_means, *d_v_rotation, *d_v_scaling;

    CUDA_CHECK(cudaMalloc(&d_means, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rotation, N * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scaling, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_viewmat, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ray_transforms, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means2d, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii, N * 2 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_depths, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normals, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_ray_transforms, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_means2d, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_depths, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_normals, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_means, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_rotation, N * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_scaling, N * 3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_means, h_means, sizeof(h_means), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rotation, h_rotation, sizeof(h_rotation), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scaling, h_scaling, sizeof(h_scaling), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_viewmat, h_viewmat, sizeof(h_viewmat), cudaMemcpyHostToDevice));

    projection_2dgs_kernel<<<1, 1>>>(
        d_means, d_rotation, d_scaling, d_viewmat,
        512.f, 512.f, 256.f, 256.f,
        0.1f, W, H,
        d_ray_transforms, d_means2d, d_radii, d_depths, d_normals, N
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_v_ray_transforms, 0, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_means2d, 0, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_normals, 0, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_means, 0, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_rotation, 0, N * 4 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_scaling, 0, N * 3 * sizeof(float)));

    float h_v_depths[N] = {1.f};
    CUDA_CHECK(cudaMemcpy(d_v_depths, h_v_depths, sizeof(h_v_depths), cudaMemcpyHostToDevice));

    launch_projection_2dgs_bwd(
        d_means, d_rotation, d_scaling, d_viewmat,
        512.f, 512.f, 256.f, 256.f,
        d_ray_transforms, d_radii,
        d_v_ray_transforms, d_v_means2d, d_v_depths, d_v_normals,
        d_v_means, d_v_rotation, d_v_scaling, N
    );

    float h_out_means[3];
    float h_out_rotation[4];
    float h_out_scaling[3];
    CUDA_CHECK(cudaMemcpy(h_out_means, d_v_means, sizeof(h_out_means), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_rotation, d_v_rotation, sizeof(h_out_rotation), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_scaling, d_v_scaling, sizeof(h_out_scaling), cudaMemcpyDeviceToHost));

    printf("v_means    = (%.6f, %.6f, %.6f)\n", h_out_means[0], h_out_means[1], h_out_means[2]);
    printf("v_rotation = (%.6f, %.6f, %.6f, %.6f)\n",
           h_out_rotation[0], h_out_rotation[1], h_out_rotation[2], h_out_rotation[3]);
    printf("v_scaling  = (%.6f, %.6f, %.6f)\n",
           h_out_scaling[0], h_out_scaling[1], h_out_scaling[2]);

    bool ok = true;
    ok &= fabsf(h_out_means[0]) < 1e-5f;
    ok &= fabsf(h_out_means[1]) < 1e-5f;
    ok &= fabsf(h_out_means[2] - 1.f) < 1e-5f;
    ok &= fabsf(h_out_rotation[0]) < 1e-5f;
    ok &= fabsf(h_out_rotation[1]) < 1e-5f;
    ok &= fabsf(h_out_rotation[2]) < 1e-5f;
    ok &= fabsf(h_out_rotation[3]) < 1e-5f;
    ok &= fabsf(h_out_scaling[0]) < 1e-5f;
    ok &= fabsf(h_out_scaling[1]) < 1e-5f;
    ok &= fabsf(h_out_scaling[2]) < 1e-5f;

    cudaFree(d_means);
    cudaFree(d_rotation);
    cudaFree(d_scaling);
    cudaFree(d_viewmat);
    cudaFree(d_ray_transforms);
    cudaFree(d_means2d);
    cudaFree(d_radii);
    cudaFree(d_depths);
    cudaFree(d_normals);
    cudaFree(d_v_ray_transforms);
    cudaFree(d_v_means2d);
    cudaFree(d_v_depths);
    cudaFree(d_v_normals);
    cudaFree(d_v_means);
    cudaFree(d_v_rotation);
    cudaFree(d_v_scaling);

    printf("\n%s\n", ok ? "All tests passed." : "SOME TESTS FAILED.");
    return ok ? 0 : 1;
}
#endif
