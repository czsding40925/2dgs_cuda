// simple_knn.cuh
//
// Standalone CUDA KNN helper for Gaussian scale initialization.
// Mirrors the GraphDECO simple-knn idea without Torch: Morton-sort points,
// build coarse boxes, then compute the mean squared distance to the 3 nearest
// neighbors for each input point.

#pragma once

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error at %s:%d — %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_e));                      \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)
#endif

namespace simple_knn {

namespace cg = cooperative_groups;

static constexpr int KNN_BOX_SIZE = 1024;

struct MinMax {
    float3 minn;
    float3 maxx;
};

struct CustomMin {
    __device__ __forceinline__ float3 operator()(const float3& a, const float3& b) const {
        return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
    }
};

struct CustomMax {
    __device__ __forceinline__ float3 operator()(const float3& a, const float3& b) const {
        return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
    }
};

__host__ __device__ inline uint32_t prep_morton(uint32_t x) {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    return x;
}

__host__ __device__ inline uint32_t coord_to_morton(float3 p, float3 minn, float3 maxx) {
    float dx = fmaxf(maxx.x - minn.x, 1e-12f);
    float dy = fmaxf(maxx.y - minn.y, 1e-12f);
    float dz = fmaxf(maxx.z - minn.z, 1e-12f);
    uint32_t x = prep_morton((uint32_t)(fminf(fmaxf((p.x - minn.x) / dx, 0.f), 1.f) * ((1u << 10) - 1u)));
    uint32_t y = prep_morton((uint32_t)(fminf(fmaxf((p.y - minn.y) / dy, 0.f), 1.f) * ((1u << 10) - 1u)));
    uint32_t z = prep_morton((uint32_t)(fminf(fmaxf((p.z - minn.z) / dz, 0.f), 1.f) * ((1u << 10) - 1u)));
    return x | (y << 1) | (z << 2);
}

__global__ void coord_to_morton_kernel(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes) {
    int idx = cg::this_grid().thread_rank();
    if (idx >= P) return;
    codes[idx] = coord_to_morton(points[idx], minn, maxx);
}

__global__ void box_minmax_kernel(uint32_t P, const float3* points, const uint32_t* indices, MinMax* boxes) {
    uint32_t idx = cg::this_grid().thread_rank();

    MinMax me;
    if (idx < P) {
        me.minn = points[indices[idx]];
        me.maxx = points[indices[idx]];
    } else {
        me.minn = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        me.maxx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }

    __shared__ MinMax reduced[KNN_BOX_SIZE];
    for (int off = KNN_BOX_SIZE / 2; off >= 1; off /= 2) {
        if (threadIdx.x < 2 * off) reduced[threadIdx.x] = me;
        __syncthreads();

        if (threadIdx.x < off) {
            MinMax other = reduced[threadIdx.x + off];
            me.minn.x = fminf(me.minn.x, other.minn.x);
            me.minn.y = fminf(me.minn.y, other.minn.y);
            me.minn.z = fminf(me.minn.z, other.minn.z);
            me.maxx.x = fmaxf(me.maxx.x, other.maxx.x);
            me.maxx.y = fmaxf(me.maxx.y, other.maxx.y);
            me.maxx.z = fmaxf(me.maxx.z, other.maxx.z);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) boxes[blockIdx.x] = me;
}

__host__ __device__ inline float dist_box_point(const MinMax& box, const float3& p) {
    float3 diff = make_float3(0.f, 0.f, 0.f);
    if (p.x < box.minn.x || p.x > box.maxx.x)
        diff.x = fminf(fabsf(p.x - box.minn.x), fabsf(p.x - box.maxx.x));
    if (p.y < box.minn.y || p.y > box.maxx.y)
        diff.y = fminf(fabsf(p.y - box.minn.y), fabsf(p.y - box.maxx.y));
    if (p.z < box.minn.z || p.z > box.maxx.z)
        diff.z = fminf(fabsf(p.z - box.minn.z), fabsf(p.z - box.maxx.z));
    return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template <int K>
__device__ inline void update_k_best(const float3& ref, const float3& point, float* best) {
    float3 d = make_float3(point.x - ref.x, point.y - ref.y, point.z - ref.z);
    float dist = d.x * d.x + d.y * d.y + d.z * d.z;
    if (dist <= 1e-20f) return;
    for (int j = 0; j < K; j++) {
        if (best[j] > dist) {
            float tmp = best[j];
            best[j] = dist;
            dist = tmp;
        }
    }
}

__global__ void box_mean_dist_kernel(
    uint32_t P,
    const float3* points,
    const uint32_t* indices,
    const MinMax* boxes,
    float* mean_dists2
) {
    int idx = cg::this_grid().thread_rank();
    if (idx >= (int)P) return;

    float3 point = points[indices[idx]];
    float best[3] = {FLT_MAX, FLT_MAX, FLT_MAX};

    for (int i = max(0, idx - 3); i <= min((int)P - 1, idx + 3); i++) {
        update_k_best<3>(point, points[indices[i]], best);
    }

    float reject = best[2];
    best[0] = FLT_MAX;
    best[1] = FLT_MAX;
    best[2] = FLT_MAX;

    int n_boxes = (P + KNN_BOX_SIZE - 1) / KNN_BOX_SIZE;
    for (int b = 0; b < n_boxes; b++) {
        float box_dist = dist_box_point(boxes[b], point);
        if (box_dist > reject || box_dist > best[2]) continue;

        int lo = b * KNN_BOX_SIZE;
        int hi = min((int)P, (b + 1) * KNN_BOX_SIZE);
        for (int i = lo; i < hi; i++) {
            update_k_best<3>(point, points[indices[i]], best);
        }
    }

    float sum = 0.f;
    int count = 0;
    for (int k = 0; k < 3; k++) {
        if (best[k] < FLT_MAX) {
            sum += best[k];
            count++;
        }
    }
    mean_dists2[indices[idx]] = count > 0 ? sum / (float)count : 1e-4f;
}

inline void dist_cuda2(int P, const float3* d_points, float* d_mean_dists2) {
    if (P <= 0) return;
    if (P == 1) {
        CUDA_CHECK(cudaMemset(d_mean_dists2, 0, sizeof(float)));
        return;
    }

    float3* d_reduce = nullptr;
    CUDA_CHECK(cudaMalloc(&d_reduce, sizeof(float3)));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    float3 init_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 init_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    float3 minn, maxx;

    cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_points, d_reduce, P, CustomMin(), init_min);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_points, d_reduce, P, CustomMin(), init_min);
    CUDA_CHECK(cudaMemcpy(&minn, d_reduce, sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_temp));

    d_temp = nullptr;
    temp_bytes = 0;
    cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_points, d_reduce, P, CustomMax(), init_max);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_points, d_reduce, P, CustomMax(), init_max);
    CUDA_CHECK(cudaMemcpy(&maxx, d_reduce, sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_reduce));

    thrust::device_vector<uint32_t> morton(P);
    thrust::device_vector<uint32_t> morton_sorted(P);
    coord_to_morton_kernel<<<(P + 255) / 256, 256>>>(P, d_points, minn, maxx, thrust::raw_pointer_cast(morton.data()));
    CUDA_CHECK(cudaGetLastError());

    thrust::device_vector<uint32_t> indices(P);
    thrust::sequence(indices.begin(), indices.end());
    thrust::device_vector<uint32_t> indices_sorted(P);

    d_temp = nullptr;
    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        thrust::raw_pointer_cast(morton.data()),
        thrust::raw_pointer_cast(morton_sorted.data()),
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(indices_sorted.data()),
        P);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        thrust::raw_pointer_cast(morton.data()),
        thrust::raw_pointer_cast(morton_sorted.data()),
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(indices_sorted.data()),
        P);
    CUDA_CHECK(cudaFree(d_temp));

    uint32_t n_boxes = (P + KNN_BOX_SIZE - 1) / KNN_BOX_SIZE;
    thrust::device_vector<MinMax> boxes(n_boxes);
    box_minmax_kernel<<<n_boxes, KNN_BOX_SIZE>>>(
        P, d_points, thrust::raw_pointer_cast(indices_sorted.data()), thrust::raw_pointer_cast(boxes.data()));
    CUDA_CHECK(cudaGetLastError());

    box_mean_dist_kernel<<<n_boxes, KNN_BOX_SIZE>>>(
        P, d_points, thrust::raw_pointer_cast(indices_sorted.data()), thrust::raw_pointer_cast(boxes.data()), d_mean_dists2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

inline std::vector<float> dist_cuda2_host(const float* h_xyz, int P) {
    std::vector<float> mean_dists2(P);
    if (P <= 0) return mean_dists2;

    float3* d_points = nullptr;
    float* d_mean_dists2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, P * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_mean_dists2, P * sizeof(float)));

    std::vector<float3> h_points(P);
    for (int i = 0; i < P; i++) {
        h_points[i] = make_float3(h_xyz[i*3], h_xyz[i*3+1], h_xyz[i*3+2]);
    }
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), P * sizeof(float3), cudaMemcpyHostToDevice));

    dist_cuda2(P, d_points, d_mean_dists2);

    CUDA_CHECK(cudaMemcpy(mean_dists2.data(), d_mean_dists2, P * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_mean_dists2));
    return mean_dists2;
}

} // namespace simple_knn
