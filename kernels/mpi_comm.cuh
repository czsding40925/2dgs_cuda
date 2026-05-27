#pragma once
// MPI helpers for the M4 distributed trainer.
//
// Build modes:
//   * Plain (no -DUSE_MPI)   — every function is a no-op stub; rank()=0,
//     world_size()=1. The existing single-GPU `train` binary uses this.
//   * MPI (`-DUSE_MPI`)      — backs onto OpenMPI / MPICH. Built into
//     the separate `train_mpi` binary via `nvcc -ccbin mpicxx`.
//
// On the MPI path we assume CUDA-aware MPI by default: device pointers
// are passed straight to MPI_Allreduce. If the build cannot guarantee
// that, define -DMPI_FORCE_HOST_STAGE and the helpers below copy
// through pinned host memory.
//
// Determinism note: MPI_Allreduce is *consistent across ranks* (every
// rank receives byte-identical output) but the reduction order is
// implementation-defined. We rely on consistency, not bitwise
// reproducibility across runs.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#ifdef USE_MPI
  #include <mpi.h>
#endif

namespace mpicomm {

// ─── Init / finalize ──────────────────────────────────────────────────────────

class MpiContext {
public:
    MpiContext(int* argc, char*** argv) {
#ifdef USE_MPI
        int provided = 0;
        int rc = MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
        if (rc != MPI_SUCCESS)
            throw std::runtime_error("MPI_Init_thread failed");
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_);
        initialised_ = true;
#else
        (void)argc; (void)argv;
        rank_ = 0;
        world_ = 1;
#endif
    }

    ~MpiContext() {
#ifdef USE_MPI
        if (initialised_) MPI_Finalize();
#endif
    }

    MpiContext(const MpiContext&)            = delete;
    MpiContext& operator=(const MpiContext&) = delete;

    int rank()       const { return rank_; }
    int world_size() const { return world_; }
    bool enabled()   const { return world_ > 1; }
    bool is_root()   const { return rank_ == 0; }

    void barrier() const {
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    void abort(int code) const {
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, code);
#else
        std::exit(code);
#endif
    }

private:
    int  rank_ = 0;
    int  world_ = 1;
    bool initialised_ = false;
};

// ─── Collectives ──────────────────────────────────────────────────────────────

#ifdef USE_MPI
inline void check(int rc, const char* what) {
    if (rc != MPI_SUCCESS) {
        std::fprintf(stderr, "[mpi] %s failed (rc=%d)\n", what, rc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#endif

// In-place SUM allreduce of `n` floats living in device memory.
// `host_stage`: if true (or if MPI_FORCE_HOST_STAGE is defined at compile
// time), copy through pinned host memory instead of trusting CUDA-aware
// MPI. Useful as a portability fallback and as a benchmarking knob to
// measure the cost of host staging vs CUDA-aware paths.
inline void allreduce_sum_inplace_device_float(
    float* d_buf, std::size_t n, bool host_stage = false
) {
    if (n == 0) return;
#ifndef USE_MPI
    (void)d_buf; (void)host_stage;
    return;  // no-op in single-process build
#else
  #ifdef MPI_FORCE_HOST_STAGE
    host_stage = true;
  #endif
    if (!host_stage) {
        check(MPI_Allreduce(MPI_IN_PLACE, d_buf, (int)n, MPI_FLOAT, MPI_SUM,
                            MPI_COMM_WORLD),
              "MPI_Allreduce(d_float)");
        return;
    }
    // Host-staged fallback: D2H → Allreduce → H2D using pinned memory.
    float* h_buf = nullptr;
    cudaError_t ce = cudaMallocHost(&h_buf, n * sizeof(float));
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "[mpi] cudaMallocHost failed: %s\n",
                     cudaGetErrorString(ce));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaMemcpy(h_buf, d_buf, n * sizeof(float), cudaMemcpyDeviceToHost);
    check(MPI_Allreduce(MPI_IN_PLACE, h_buf, (int)n, MPI_FLOAT, MPI_SUM,
                        MPI_COMM_WORLD),
          "MPI_Allreduce(h_float)");
    cudaMemcpy(d_buf, h_buf, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaFreeHost(h_buf);
#endif
}

// Same, but for int32 (used for densification visibility counts).
inline void allreduce_sum_inplace_device_int(
    int* d_buf, std::size_t n, bool host_stage = false
) {
    if (n == 0) return;
#ifndef USE_MPI
    (void)d_buf; (void)host_stage;
    return;
#else
  #ifdef MPI_FORCE_HOST_STAGE
    host_stage = true;
  #endif
    if (!host_stage) {
        check(MPI_Allreduce(MPI_IN_PLACE, d_buf, (int)n, MPI_INT, MPI_SUM,
                            MPI_COMM_WORLD),
              "MPI_Allreduce(d_int)");
        return;
    }
    int* h_buf = nullptr;
    cudaError_t ce = cudaMallocHost(&h_buf, n * sizeof(int));
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "[mpi] cudaMallocHost failed: %s\n",
                     cudaGetErrorString(ce));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaMemcpy(h_buf, d_buf, n * sizeof(int), cudaMemcpyDeviceToHost);
    check(MPI_Allreduce(MPI_IN_PLACE, h_buf, (int)n, MPI_INT, MPI_SUM,
                        MPI_COMM_WORLD),
          "MPI_Allreduce(h_int)");
    cudaMemcpy(d_buf, h_buf, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaFreeHost(h_buf);
#endif
}

// Host-side helpers used for control-plane state (RNG seeds, camera
// indices, densification decisions). Cheap; always host-side.

inline void broadcast_u64(uint64_t* val, int root = 0) {
#ifdef USE_MPI
    check(MPI_Bcast(val, 1, MPI_UINT64_T, root, MPI_COMM_WORLD),
          "MPI_Bcast(u64)");
#else
    (void)val; (void)root;
#endif
}

inline void broadcast_int_array(int* buf, std::size_t n, int root = 0) {
#ifdef USE_MPI
    if (n == 0) return;
    check(MPI_Bcast(buf, (int)n, MPI_INT, root, MPI_COMM_WORLD),
          "MPI_Bcast(int_array)");
#else
    (void)buf; (void)n; (void)root;
#endif
}

// All ranks must agree on the per-iteration camera batch. The simplest
// reproducible scheme is "shared RNG seed; every rank draws the same K
// indices in the same order; rank `r` consumes slot[r]". No broadcast
// needed at iter time once the seed is shared.
inline int rank_local_cam_slot(int rank, int slot_per_rank) {
    return rank * slot_per_rank;
}

// ─── In-place scale (SUM → MEAN after allreduce) ─────────────────────────────

namespace detail {
__global__ static void scale_inplace_kernel(
    float* __restrict__ buf, std::size_t n, float scale
) {
    std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] *= scale;
}
} // namespace detail

inline void scale_inplace_device_float(float* d_buf, std::size_t n, float scale) {
    if (n == 0 || scale == 1.f) return;
    int threads = 256;
    int blocks  = (int)((n + (std::size_t)threads - 1) / (std::size_t)threads);
    detail::scale_inplace_kernel<<<blocks, threads>>>(d_buf, n, scale);
}

// ─── Device selection ─────────────────────────────────────────────────────────

// Picks one CUDA device per rank. With fewer devices than ranks
// (e.g. oversubscribed single-GPU test), ranks wrap modulo device count.
inline int select_device_for_rank(int rank) {
    int n_devices = 0;
    cudaError_t ce = cudaGetDeviceCount(&n_devices);
    if (ce != cudaSuccess || n_devices == 0) {
        std::fprintf(stderr, "[mpi] no CUDA devices visible\n");
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 3);
#else
        std::exit(3);
#endif
    }
    int dev = rank % n_devices;
    cudaSetDevice(dev);
    return dev;
}

} // namespace mpicomm
