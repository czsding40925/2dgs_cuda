// bindings.cpp
//
// Option 1: pybind11 wrapper that exposes CUDA kernels to Python.
// Python passes numpy arrays (or torch tensors) → we extract raw pointers
// → call the CUDA kernel → return results as numpy arrays.
//
// The kernels themselves are unchanged. Only this thin boundary layer is new.
//
// Build (requires pybind11 and Python dev headers):
//   nvcc -O2 -std=c++17 -arch=native \
//        $(python3 -m pybind11 --includes) \
//        -shared -fPIC \
//        -o gs_kernels$(python3-config --extension-suffix) \
//        bindings.cpp ../kernels/camera_projection.cu ../kernels/quat_to_rotmat.cu

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace py = pybind11;

// ─── Forward-declare the kernels (defined in their own .cu files) ───────────

__global__ void project_points_kernel(const float3*, float2*, struct Intrinsics, int);
__global__ void quat_to_rotmat_kernel(const float*, float*, int);

// ─── Helper: check numpy array is float32 and C-contiguous ──────────────────

static void check_array(const py::array_t<float>& arr, const char* name) {
    if (!(arr.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) + " must be C-contiguous");
    if (arr.itemsize() != sizeof(float))
        throw std::runtime_error(std::string(name) + " must be float32");
}

// ─── project_points binding ─────────────────────────────────────────────────
//
// Python call:
//   pixels = gs_kernels.project_points(points, fx, fy, cx, cy)
//   # points: np.ndarray [N, 3] float32
//   # pixels: np.ndarray [N, 2] float32

py::array_t<float> py_project_points(
    py::array_t<float> points,   // [N, 3]
    float fx, float fy, float cx, float cy
) {
    check_array(points, "points");
    if (points.ndim() != 2 || points.shape(1) != 3)
        throw std::runtime_error("points must be shape [N, 3]");

    int N = points.shape(0);
    auto pixels = py::array_t<float>({N, 2});   // allocate output

    // Copy input to GPU
    float3* d_points;
    float2* d_pixels;
    cudaMalloc(&d_points, N * sizeof(float3));
    cudaMalloc(&d_pixels, N * sizeof(float2));
    cudaMemcpy(d_points, points.data(), N * sizeof(float3), cudaMemcpyHostToDevice);

    // Inline Intrinsics struct (mirrors the one in camera_projection.cu)
    struct { float fx, fy, cx, cy; } K = {fx, fy, cx, cy};

    int threads = 256, blocks = (N + threads - 1) / threads;
    project_points_kernel<<<blocks, threads>>>(
        d_points, d_pixels,
        *reinterpret_cast<struct Intrinsics*>(&K),  // reinterpret to match kernel type
        N
    );
    cudaDeviceSynchronize();

    cudaMemcpy(pixels.mutable_data(), d_pixels, N * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaFree(d_points);
    cudaFree(d_pixels);
    return pixels;
}

// ─── quat_to_rotmat binding ─────────────────────────────────────────────────
//
// Python call:
//   rotmats = gs_kernels.quat_to_rotmat(quats)
//   # quats:   np.ndarray [N, 4] float32  (w, x, y, z)
//   # rotmats: np.ndarray [N, 3, 3] float32

py::array_t<float> py_quat_to_rotmat(py::array_t<float> quats) {
    check_array(quats, "quats");
    if (quats.ndim() != 2 || quats.shape(1) != 4)
        throw std::runtime_error("quats must be shape [N, 4]");

    int N = quats.shape(0);
    auto rotmats = py::array_t<float>({N, 3, 3});

    float* d_quats;
    float* d_rotmats;
    cudaMalloc(&d_quats,   N * 4 * sizeof(float));
    cudaMalloc(&d_rotmats, N * 9 * sizeof(float));
    cudaMemcpy(d_quats, quats.data(), N * 4 * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256, blocks = (N + threads - 1) / threads;
    quat_to_rotmat_kernel<<<blocks, threads>>>(d_quats, d_rotmats, N);
    cudaDeviceSynchronize();

    cudaMemcpy(rotmats.mutable_data(), d_rotmats, N * 9 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_quats);
    cudaFree(d_rotmats);
    return rotmats;
}

// ─── Module definition ──────────────────────────────────────────────────────

PYBIND11_MODULE(gs_kernels, m) {
    m.doc() = "CUDA kernels for 2DGS — wrapped for Python via pybind11";

    m.def("project_points", &py_project_points,
          py::arg("points"), py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          "Project [N,3] camera-space points to [N,2] pixel coords");

    m.def("quat_to_rotmat", &py_quat_to_rotmat,
          py::arg("quats"),
          "Convert [N,4] quaternions (w,x,y,z) to [N,3,3] rotation matrices");
}
