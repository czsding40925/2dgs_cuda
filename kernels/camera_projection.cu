// camera_projection.cu
//
// Pinhole camera: projects N 3D points (in camera space) to 2D pixel coords.
//
// Math:
//   u = fx * (X / Z) + cx
//   v = fy * (Y / Z) + cy
//
// (fx, fy) are focal lengths in pixels; (cx, cy) is the principal point.
// This is the inner-most operation in any 3D rendering pipeline — used
// in 3DGS/2DGS to determine which tile/pixel each Gaussian contributes to.

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d  —  %s\n",                         \
              __FILE__, __LINE__, cudaGetErrorString(err));                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

struct Intrinsics {
    float fx, fy;   // focal length (pixels)
    float cx, cy;   // principal point (pixels)
};

// ---------------------------------------------------------------------------
// CUDA kernel
// ---------------------------------------------------------------------------
//
// Thread layout: 1D, one thread per point.
//   blockIdx.x  — which block of 256 points we're in
//   threadIdx.x — which point within that block
//
// __restrict__ tells the compiler the input/output arrays don't alias,
// enabling better memory access optimisations.

__global__ void project_points_kernel(
    const float3* __restrict__ points,   // [N]  (X, Y, Z) in camera space
    float2*       __restrict__ pixels,   // [N]  (u, v)    in pixel coords
    Intrinsics K,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;  // guard: last block may be partially filled

    float X = points[i].x;
    float Y = points[i].y;
    float Z = points[i].z;

    pixels[i] = make_float2(
        K.fx * (X / Z) + K.cx,
        K.fy * (Y / Z) + K.cy
    );
}

// ---------------------------------------------------------------------------
// CPU reference  (same math, used to validate the kernel)
// ---------------------------------------------------------------------------

void project_points_cpu(
    const float3* points, float2* pixels, Intrinsics K, int N
) {
    for (int i = 0; i < N; i++) {
        pixels[i].x = K.fx * (points[i].x / points[i].z) + K.cx;
        pixels[i].y = K.fy * (points[i].y / points[i].z) + K.cy;
    }
}

// ---------------------------------------------------------------------------
// main — launches kernel, compares to CPU reference, checks analytic cases
// ---------------------------------------------------------------------------

int main() {
    // Typical 640×480 pinhole camera
    Intrinsics K = { 500.0f, 500.0f, 320.0f, 240.0f };

    // Test points (X, Y, Z) in camera space.  Z > 0 = in front of camera.
    float3 h_points[] = {
        {  0.0f,  0.0f, 1.0f },  // on optical axis → (cx, cy)
        {  1.0f,  0.0f, 1.0f },  // 1 unit right at depth 1 → (fx+cx, cy)
        {  0.0f,  1.0f, 1.0f },  // 1 unit down  at depth 1 → (cx, fy+cy)
        {  1.0f,  1.0f, 2.0f },  // halved by depth 2       → (fx/2+cx, fy/2+cy)
        { -0.5f,  0.3f, 5.0f },  // arbitrary depth
        {  0.0f,  0.0f, 10.0f }, // far point on axis       → still (cx, cy)
    };
    const int N = sizeof(h_points) / sizeof(float3);

    // Analytic expectations for the first 4 points
    float2 expected[4] = {
        { K.cx,               K.cy              },
        { K.fx     + K.cx,    K.cy              },
        { K.cx,               K.fy     + K.cy   },
        { K.fx/2.f + K.cx,    K.fy/2.f + K.cy  },
    };

    // --- Allocate and copy to device ---
    float3* d_points;
    float2* d_pixels;
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_pixels, N * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(d_points, h_points, N * sizeof(float3), cudaMemcpyHostToDevice));

    // --- Launch kernel ---
    // 256 threads per block is a common default; covers all N points in one block here.
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    project_points_kernel<<<BLOCKS, THREADS>>>(d_points, d_pixels, K, N);
    CUDA_CHECK(cudaGetLastError());     // catches config errors (bad block size, etc.)
    CUDA_CHECK(cudaDeviceSynchronize()); // wait for kernel to finish

    // --- Copy results back ---
    float2 gpu_pixels[N];
    CUDA_CHECK(cudaMemcpy(gpu_pixels, d_pixels, N * sizeof(float2), cudaMemcpyDeviceToHost));

    // --- CPU reference ---
    float2 cpu_pixels[N];
    project_points_cpu(h_points, cpu_pixels, K, N);

    // --- Print & validate ---
    printf("\nIntrinsics: fx=%.0f  fy=%.0f  cx=%.0f  cy=%.0f\n\n",
           K.fx, K.fy, K.cx, K.cy);
    printf("  %-3s  %-22s  %-18s  %-18s  %s\n",
           "#", "point (X, Y, Z)", "gpu  (u, v)", "cpu  (u, v)", "status");
    printf("  %-3s  %-22s  %-18s  %-18s  %s\n",
           "---", "----------------------", "------------------",
           "------------------", "------");

    bool all_ok = true;
    const float EPS = 1e-3f;

    for (int i = 0; i < N; i++) {
        bool gpu_cpu_match =
            fabsf(gpu_pixels[i].x - cpu_pixels[i].x) < EPS &&
            fabsf(gpu_pixels[i].y - cpu_pixels[i].y) < EPS;

        bool analytic_ok = true;
        if (i < 4) {
            analytic_ok =
                fabsf(gpu_pixels[i].x - expected[i].x) < EPS &&
                fabsf(gpu_pixels[i].y - expected[i].y) < EPS;
        }

        bool ok = gpu_cpu_match && analytic_ok;
        all_ok &= ok;

        printf("  %-3d  (%+6.2f, %+6.2f, %5.2f)  (%7.2f, %7.2f)  (%7.2f, %7.2f)  %s\n",
               i,
               h_points[i].x, h_points[i].y, h_points[i].z,
               gpu_pixels[i].x, gpu_pixels[i].y,
               cpu_pixels[i].x, cpu_pixels[i].y,
               ok ? "OK" : "FAIL");

        if (i < 4 && !analytic_ok)
            printf("       expected  (%7.2f, %7.2f)\n",
                   expected[i].x, expected[i].y);
    }

    printf("\n%s\n\n", all_ok ? "All tests passed." : "SOME TESTS FAILED.");

    cudaFree(d_points);
    cudaFree(d_pixels);
    return all_ok ? 0 : 1;
}
