// splat_data_test.cu — basic compile + runtime check for SplatData

#include "splat_data.cuh"
#include <cstdio>

// Kernel that reads raw values through the activation helpers
__global__ void check_activations_kernel(
    const float* opacity_raw,
    const float* scaling_raw,
    const float* rotation_raw,
    float* out_opacity,
    float* out_scale0,
    float* out_rot_norm,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    out_opacity[i]  = SplatData::sigmoid(opacity_raw[i]);
    float3 s        = SplatData::exp3(scaling_raw + i * 3);
    out_scale0[i]   = s.x;
    float4 q        = SplatData::normalize4(rotation_raw + i * 4);
    out_rot_norm[i] = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;  // should be ~1
}

int main() {
    const int N = 4;

    // ── Allocation test ───────────────────────────────────────────────────────
    SplatData splats(N, /*max_sh_degree=*/3);
    splats.print_summary();

    // sh_coeffs: degree 3 → (3+1)^2 - 1 = 15 per channel, 45 total floats per Gaussian
    printf("sh_coeffs_per_channel(3) = %d  (expected 15)\n", sh_coeffs_per_channel(3));
    printf("sh_coeffs_per_channel(0) = %d  (expected 0)\n",  sh_coeffs_per_channel(0));

    // ── Fill with known host values, copy to GPU ──────────────────────────────
    float h_opacity[N]       = { 0.f,  2.f, -2.f,  1.f };   // logits
    float h_scaling[N*3]     = { 0.f,0.f,0.f,  1.f,1.f,1.f,  -1.f,-1.f,-1.f,  0.5f,0.5f,0.5f };
    float h_rotation[N*4]    = {
        1,0,0,0,         // identity
        2,0,0,0,         // unnormalized — should normalise to identity
        0,1,0,0,         // 90° around X
        1,1,0,0,         // 45° around X (unnormalized)
    };

    CUDA_CHECK(cudaMemcpy(splats.opacity(),  h_opacity,   N   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(splats.scaling(),  h_scaling,   N*3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(splats.rotation(), h_rotation,  N*4 * sizeof(float), cudaMemcpyHostToDevice));

    // ── Run activation kernel ─────────────────────────────────────────────────
    float *d_out_opacity, *d_out_scale0, *d_out_rot_norm;
    CUDA_CHECK(cudaMalloc(&d_out_opacity,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_scale0,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_rot_norm, N * sizeof(float)));

    check_activations_kernel<<<1, N>>>(
        splats.opacity(), splats.scaling(), splats.rotation(),
        d_out_opacity, d_out_scale0, d_out_rot_norm, N
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float out_opacity[N], out_scale0[N], out_rot_norm[N];
    CUDA_CHECK(cudaMemcpy(out_opacity,  d_out_opacity,  N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_scale0,   d_out_scale0,   N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_rot_norm, d_out_rot_norm, N*sizeof(float), cudaMemcpyDeviceToHost));

    // ── Validate ──────────────────────────────────────────────────────────────
    bool ok = true;
    printf("\n%-4s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
           "i", "logit", "sigmoid", "log_s0", "exp_s0", "||q||²");

    float expected_opacity[]  = { 0.5f, 0.8808f, 0.1192f, 0.7311f };
    float expected_scale0[]   = { 1.f,  2.7183f, 0.3679f, 1.6487f };

    for (int i = 0; i < N; i++) {
        bool op_ok  = fabsf(out_opacity[i]  - expected_opacity[i]) < 1e-3f;
        bool sc_ok  = fabsf(out_scale0[i]   - expected_scale0[i])  < 1e-3f;
        bool rot_ok = fabsf(out_rot_norm[i] - 1.f)                 < 1e-5f;
        bool row_ok = op_ok && sc_ok && rot_ok;
        ok &= row_ok;
        printf("%-4d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.6f  %s\n",
               i, h_opacity[i], out_opacity[i],
               h_scaling[i*3], out_scale0[i],
               out_rot_norm[i], row_ok ? "OK" : "FAIL");
    }

    // ── Reserve / resize preserves existing rows ────────────────────────────
    splats.reserve(8);
    splats.resize(6);
    float preserved_opacity[N] = {};
    CUDA_CHECK(cudaMemcpy(preserved_opacity, splats.opacity(),
                          N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        bool reserve_ok = fabsf(preserved_opacity[i] - h_opacity[i]) < 1e-6f;
        ok &= reserve_ok;
    }
    printf("\nreserve: N=%d capacity=%d\n", splats.N(), splats.capacity());

    // ── Move semantics ────────────────────────────────────────────────────────
    SplatData splats2 = std::move(splats);
    printf("\nAfter move — original: "); splats.print_summary();
    printf("After move — new:      "); splats2.print_summary();

    cudaFree(d_out_opacity); cudaFree(d_out_scale0); cudaFree(d_out_rot_norm);

    printf("\n%s\n\n", ok ? "All tests passed." : "SOME TESTS FAILED.");
    return ok ? 0 : 1;
}
