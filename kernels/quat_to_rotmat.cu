// quat_to_rotmat.cu
//
// Converts N unit quaternions to 3x3 rotation matrices.
// This is the first step in computing a Gaussian's shape: every Gaussian
// stores its orientation as a quaternion q = (w, x, y, z).
//
// Formula (standard, matches gsplat Utils.cuh):
//   R = [ 1-2(y²+z²)   2(xy-wz)   2(xz+wy) ]
//       [   2(xy+wz)  1-2(x²+z²)  2(yz-wx) ]
//       [   2(xz-wy)   2(yz+wx)  1-2(x²+y²)]
//
// Memory layout (row-major, flat arrays):
//   quats:   [w0,x0,y0,z0,  w1,x1,y1,z1,  ...]   (N*4 floats)
//   rotmats: [r00,r01,r02, r10,...,r22,    ...]   (N*9 floats)
//
// New concepts vs camera_projection.cu:
//   - __device__ helper functions (callable from kernels, not from host)
//   - CUDA events for timing
//   - Realistic N (1M) — multiple blocks of 256 threads each

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
// __device__ helper — runs on GPU, callable only from kernels
//
// Separating the math into a __device__ function keeps the kernel readable
// and lets the compiler inline it (no function-call overhead on GPU).
// This is the pattern gsplat uses throughout Utils.cuh.
// ---------------------------------------------------------------------------

__device__ void quat_to_rotmat_device(
    float w, float x, float y, float z,
    float* R   // output: 9 floats, row-major
) {
    // Normalise (quats should already be unit, but guard against drift)
    float inv_norm = rsqrtf(w*w + x*x + y*y + z*z);  // rsqrtf = 1/sqrt, fast GPU intrinsic
    w *= inv_norm;  x *= inv_norm;  y *= inv_norm;  z *= inv_norm;

    float x2 = x*x, y2 = y*y, z2 = z*z;
    float xy = x*y, xz = x*z, yz = y*z;
    float wx = w*x, wy = w*y, wz = w*z;

    R[0] = 1.f - 2.f*(y2 + z2);   R[1] = 2.f*(xy - wz);          R[2] = 2.f*(xz + wy);
    R[3] = 2.f*(xy + wz);          R[4] = 1.f - 2.f*(x2 + z2);   R[5] = 2.f*(yz - wx);
    R[6] = 2.f*(xz - wy);          R[7] = 2.f*(yz + wx);          R[8] = 1.f - 2.f*(x2 + y2);
}

// ---------------------------------------------------------------------------
// Kernel — one thread per Gaussian
// ---------------------------------------------------------------------------

__global__ void quat_to_rotmat_kernel(
    const float* __restrict__ quats,    // [N*4]
    float*       __restrict__ rotmats,  // [N*9]
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Pointer arithmetic to stride into the flat arrays for this Gaussian.
    // This pattern — "shift the pointer, then index locally" — is everywhere
    // in gsplat kernels (see QuatScaleToCovarCUDA.cu).
    const float* q = quats   + i * 4;
    float*       R = rotmats + i * 9;

    quat_to_rotmat_device(q[0], q[1], q[2], q[3], R);
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

void quat_to_rotmat_cpu(const float* quats, float* rotmats, int N) {
    for (int i = 0; i < N; i++) {
        const float* q = quats   + i * 4;
        float*       R = rotmats + i * 9;
        float w=q[0], x=q[1], y=q[2], z=q[3];
        float inv = 1.f / sqrtf(w*w + x*x + y*y + z*z);
        w*=inv; x*=inv; y*=inv; z*=inv;
        float x2=x*x, y2=y*y, z2=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;
        R[0]=1-2*(y2+z2); R[1]=2*(xy-wz);  R[2]=2*(xz+wy);
        R[3]=2*(xy+wz);   R[4]=1-2*(x2+z2); R[5]=2*(yz-wx);
        R[6]=2*(xz-wy);   R[7]=2*(yz+wx);  R[8]=1-2*(x2+y2);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void print_mat3(const char* label, const float* R) {
    printf("%s\n", label);
    for (int r = 0; r < 3; r++)
        printf("  [ %+.4f  %+.4f  %+.4f ]\n", R[r*3], R[r*3+1], R[r*3+2]);
}

bool mat3_close(const float* A, const float* B, float eps = 1e-5f) {
    for (int k = 0; k < 9; k++)
        if (fabsf(A[k] - B[k]) > eps) return false;
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    // ---- Analytic test cases ------------------------------------------------
    // Each row: [w, x, y, z], expected R (row-major)
    struct Case { float q[4]; float R[9]; const char* label; };
    Case cases[] = {
        {
            {1, 0, 0, 0},                                        // identity quat
            {1,0,0, 0,1,0, 0,0,1},                              // identity matrix
            "identity"
        },
        {
            // 90° CCW around Z: w=cos(45°), z=sin(45°)
            {0.7071068f, 0, 0, 0.7071068f},
            {0,-1,0,  1,0,0,  0,0,1},
            "90 deg around Z"
        },
        {
            // 90° CCW around X: w=cos(45°), x=sin(45°)
            {0.7071068f, 0.7071068f, 0, 0},
            {1,0,0,  0,0,-1,  0,1,0},
            "90 deg around X"
        },
        {
            // 180° around Y: w=0, y=1
            {0, 0, 1, 0},
            {-1,0,0,  0,1,0,  0,0,-1},
            "180 deg around Y"
        },
    };
    const int N_CASES = sizeof(cases) / sizeof(Case);

    printf("=== Analytic tests ===\n\n");
    bool all_analytic_ok = true;
    float gpu_R[9];

    for (int c = 0; c < N_CASES; c++) {
        // Run on GPU (tiny N=1 kernel launch)
        float* d_q; float* d_R;
        CUDA_CHECK(cudaMalloc(&d_q, 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_R, 9 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_q, cases[c].q, 4*sizeof(float), cudaMemcpyHostToDevice));
        quat_to_rotmat_kernel<<<1, 1>>>(d_q, d_R, 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(gpu_R, d_R, 9*sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_q); cudaFree(d_R);

        bool ok = mat3_close(gpu_R, cases[c].R);
        all_analytic_ok &= ok;
        printf("[%s] %s\n", ok ? "OK  " : "FAIL", cases[c].label);
        if (!ok) {
            print_mat3("  got:     ", gpu_R);
            print_mat3("  expected:", cases[c].R);
        }
    }

    // ---- Large N: GPU vs CPU + timing ----------------------------------------
    const int N = 1 << 20;  // 1,048,576 — realistic Gaussian scene size
    printf("\n=== Large N = %d (%.1fM) ===\n\n", N, N / 1e6f);

    // Generate random-ish quats on host (normalised)
    float* h_quats   = new float[N * 4];
    float* h_rotmats = new float[N * 9];
    float* cpu_rotmats = new float[N * 9];

    for (int i = 0; i < N; i++) {
        // Spread of orientations via a simple pattern; not truly random but varied
        float t = (float)i / N * 6.2831853f;
        float w = cosf(t * 0.3f), x = sinf(t * 0.7f), y = cosf(t * 1.1f), z = sinf(t * 0.5f);
        float inv = 1.f / sqrtf(w*w + x*x + y*y + z*z);
        h_quats[i*4+0] = w*inv;  h_quats[i*4+1] = x*inv;
        h_quats[i*4+2] = y*inv;  h_quats[i*4+3] = z*inv;
    }

    // GPU
    float *d_quats, *d_rotmats;
    CUDA_CHECK(cudaMalloc(&d_quats,   N * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rotmats, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_quats, h_quats, N*4*sizeof(float), cudaMemcpyHostToDevice));

    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;

    // Warm-up launch (first launch has JIT overhead; don't measure it)
    quat_to_rotmat_kernel<<<BLOCKS, THREADS>>>(d_quats, d_rotmats, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed launch using CUDA events
    // Events live on the GPU timeline — more accurate than host-side timers
    // for measuring kernel duration.
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    quat_to_rotmat_kernel<<<BLOCKS, THREADS>>>(d_quats, d_rotmats, N);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    printf("GPU kernel:  %.3f ms  (%d blocks x %d threads)\n", ms, BLOCKS, THREADS);

    CUDA_CHECK(cudaMemcpy(h_rotmats, d_rotmats, N*9*sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference (just first 10k to keep it fast)
    quat_to_rotmat_cpu(h_quats, cpu_rotmats, 10000);

    // Validate first 10k against CPU
    bool all_match = true;
    for (int i = 0; i < 10000; i++) {
        if (!mat3_close(h_rotmats + i*9, cpu_rotmats + i*9)) {
            printf("MISMATCH at i=%d\n", i);
            all_match = false;
            break;
        }
    }
    printf("GPU vs CPU (first 10k): %s\n", all_match ? "OK" : "FAIL");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_quats);
    cudaFree(d_rotmats);
    delete[] h_quats;
    delete[] h_rotmats;
    delete[] cpu_rotmats;

    printf("\n%s\n\n", (all_analytic_ok && all_match) ? "All tests passed." : "SOME TESTS FAILED.");
    return (all_analytic_ok && all_match) ? 0 : 1;
}
