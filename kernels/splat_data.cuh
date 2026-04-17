// splat_data.cuh
//
// Central data structure holding all N Gaussian parameters on the GPU.
// Mirrors LichtFeld-Studio's SplatData / PyTorch's GaussianModel, but
// uses plain float* GPU pointers — no Tensor class, no Python runtime.
//
// Storage convention (same as gsplat + LichtFeld-Studio):
//   Parameters are stored in their *raw* (pre-activation) form.
//   Activations are applied inline inside kernels at read time:
//
//     opacity  → sigmoid(opacity_raw)     keeps values in (0, 1)
//     scaling  → exp(scaling_raw)         keeps scales positive
//     rotation → normalize(rotation_raw)  keeps quaternion on unit sphere
//
//   This means the optimizer always works in unconstrained space, which
//   is numerically better and matches how PyTorch GaussianModel works.
//
// Memory layout (struct-of-arrays, row-major):
//   means       [N, 3]   float32   world-space positions
//   rotation    [N, 4]   float32   quaternions (w, x, y, z), raw unnormalized
//   scaling     [N, 3]   float32   log-scale per axis
//   opacity     [N]      float32   logit opacity
//   sh0         [N, 3]   float32   degree-0 SH (base RGB color)
//   shN         [N, K]   float32   degree 1-3 SH, K = 3*(sh_degree^2+2*sh_degree)
//
// Struct-of-arrays (SoA) is preferred over array-of-structs (AoS) for GPU
// because threads in a warp access the same field of consecutive Gaussians,
// giving coalesced memory reads (all threads in a warp read adjacent floats).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

// ─── Error checking ───────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error at %s:%d — %s\n",                            \
              __FILE__, __LINE__, cudaGetErrorString(_e));                      \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

#include "simple_knn.cuh"

// ─── SH coefficient count ─────────────────────────────────────────────────────

// Degrees 1–3 add (3*1+1)*3=12, (3*4+1)*3=... actually:
// degree d has (d+1)^2 total coefficients; degree 0 is stored separately in sh0.
// shN holds degrees 1..max_sh_degree: coeffs = (max+1)^2 - 1
inline int sh_coeffs_per_channel(int max_sh_degree) {
    return (max_sh_degree + 1) * (max_sh_degree + 1) - 1;
}

// ─── SplatData ────────────────────────────────────────────────────────────────

class SplatData {
public:
    // ── Construction / destruction ────────────────────────────────────────────

    SplatData() = default;

    // Allocate GPU arrays for N Gaussians with given SH degree (0–3).
    explicit SplatData(int N, int max_sh_degree = 3) {
        allocate(N, max_sh_degree);
    }

    ~SplatData() { free(); }

    // No copy — GPU allocations are owned exclusively.
    SplatData(const SplatData&)            = delete;
    SplatData& operator=(const SplatData&) = delete;

    // Move is fine.
    SplatData(SplatData&& o) noexcept { steal(o); }
    SplatData& operator=(SplatData&& o) noexcept {
        if (this != &o) { free(); steal(o); }
        return *this;
    }

    // ── Size / metadata ───────────────────────────────────────────────────────

    int  N()              const { return _N; }
    int  capacity()       const { return _capacity; }
    int  max_sh_degree()  const { return _max_sh_degree; }
    int  active_sh_degree() const { return _active_sh_degree; }
    void set_active_sh_degree(int d) { _active_sh_degree = d; }
    void increment_sh_degree() {
        if (_active_sh_degree < _max_sh_degree) _active_sh_degree++;
    }

    // ── Raw GPU pointers (pass directly to kernels) ───────────────────────────
    //
    // Kernels receive raw pointers and apply activations inline, e.g.:
    //   float opacity = sigmoid(opacity_raw[i]);
    //   float3 scale  = exp3(scaling_raw + i*3);
    //   float4 rot    = normalize4(rotation_raw + i*4);

    float* means()    { return _means; }    // [N, 3]
    float* rotation() { return _rotation; } // [N, 4]  raw, unnormalized
    float* scaling()  { return _scaling; }  // [N, 3]  log-scale
    float* opacity()  { return _opacity; }  // [N]     logit
    float* sh0()      { return _sh0; }      // [N, 3]  DC color
    float* shN()      { return _shN; }      // [N, K]  higher-order SH (nullptr if degree 0)

    const float* means()    const { return _means; }
    const float* rotation() const { return _rotation; }
    const float* scaling()  const { return _scaling; }
    const float* opacity()  const { return _opacity; }
    const float* sh0()      const { return _sh0; }
    const float* shN()      const { return _shN; }

    // ── Activation helpers (inline device functions) ──────────────────────────
    //
    // These are called inside kernels, not on the host.
    // Usage:  float  alpha = SplatData::sigmoid(opacity_raw[i]);
    //         float3 s     = SplatData::exp3(scaling_raw + i*3);

    __device__ static float sigmoid(float x) {
        return 1.f / (1.f + expf(-x));
    }

    __device__ static float3 exp3(const float* v) {
        return make_float3(expf(v[0]), expf(v[1]), expf(v[2]));
    }

    __device__ static float4 normalize4(const float* q) {
        float inv = rsqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
        return make_float4(q[0]*inv, q[1]*inv, q[2]*inv, q[3]*inv);
    }

    // ── Capacity management (for densification) ───────────────────────────────

    // TODO(phase1): reallocate all arrays to new_capacity, preserving existing data.
    // Called by the densification step when cloning/splitting Gaussians.
    void reserve(int new_capacity);

    void resize(int new_size) {
        if (new_size < 0)
            throw std::runtime_error("SplatData::resize got negative size");
        if (new_size > _capacity)
            reserve(new_size);
        _N = new_size;
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    // Fill GPU arrays from a host-side point cloud.
    // Matches the standard GaussianModel / LichtFeld-Studio init:
    //   means    ← point positions
    //   sh0      ← point colors converted to degree-0 SH coefficient
    //   scaling  ← log(nearest_neighbor_dist) * ones(3)  (isotropic init)
    //   rotation ← identity quaternion (1, 0, 0, 0)
    //   opacity  ← logit(0.1)   (small initial opacity)
    //   shN      ← zero (higher-order SH starts at zero, curriculum learning)
    void init_from_points(const float* xyz,   // [M, 3] host pointer
                          const float* rgb,   // [M, 3] host pointer, values in [0, 1]
                          int M) {
        // ── Nearest-neighbor distances for isotropic scale init ──────────────
        // Match the GraphDECO/simple-knn convention: compute the mean squared
        // distance to the 3 nearest neighbors, then initialize scale as
        // log(sqrt(mean_dist2)).
        std::vector<float> mean_dist2 = simple_knn::dist_cuda2_host(xyz, M);
        std::vector<float> nn_dist(M);
        for (int i = 0; i < M; i++)
            nn_dist[i] = std::sqrt(std::max(mean_dist2[i], 1e-8f));

        // Isolated COLMAP points can otherwise initialize as huge splats and
        // explode the tile-intersection list. Cap at the robust p99 distance.
        std::vector<float> sorted_dist = nn_dist;
        int p99_idx = std::min(M - 1, std::max(0, (int)(0.99f * (M - 1))));
        std::nth_element(sorted_dist.begin(), sorted_dist.begin() + p99_idx, sorted_dist.end());
        float max_init_dist = std::max(sorted_dist[p99_idx], 1e-4f);
        for (float& d : nn_dist)
            d = std::min(d, max_init_dist);

        // ── Build host arrays ────────────────────────────────────────────────
        std::vector<float> h_means(M*3), h_rotation(M*4), h_scaling(M*3),
                           h_opacity(M),  h_sh0(M*3);

        // Degree-0 SH coefficient = (rgb - 0.5) / C0
        // C0 = 1 / (2 * sqrt(pi)) ≈ 0.28209479
        const float C0        = 0.28209479177387814f;
        const float logit_01  = std::log(0.1f / 0.9f);  // logit(0.1) ≈ -2.197

        for (int i = 0; i < M; i++) {
            h_means[i*3+0] = xyz[i*3+0];
            h_means[i*3+1] = xyz[i*3+1];
            h_means[i*3+2] = xyz[i*3+2];

            h_rotation[i*4+0] = 1.f;  // identity quaternion (w,x,y,z)
            h_rotation[i*4+1] = 0.f;
            h_rotation[i*4+2] = 0.f;
            h_rotation[i*4+3] = 0.f;

            float log_d = std::log(nn_dist[i]);
            h_scaling[i*3+0] = log_d;  // isotropic: same scale on all axes
            h_scaling[i*3+1] = log_d;
            h_scaling[i*3+2] = log_d;

            h_opacity[i] = logit_01;

            h_sh0[i*3+0] = (rgb[i*3+0] - 0.5f) / C0;
            h_sh0[i*3+1] = (rgb[i*3+1] - 0.5f) / C0;
            h_sh0[i*3+2] = (rgb[i*3+2] - 0.5f) / C0;
        }

        // ── Reallocate GPU arrays and upload ─────────────────────────────────
        free();
        allocate(M, _max_sh_degree);

        CUDA_CHECK(cudaMemcpy(_means,    h_means.data(),    M*3*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_rotation, h_rotation.data(), M*4*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_scaling,  h_scaling.data(),  M*3*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_opacity,  h_opacity.data(),  M  *sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_sh0,      h_sh0.data(),      M*3*sizeof(float), cudaMemcpyHostToDevice));
        // shN starts at zero — SH curriculum increases active_sh_degree during training
        if (_shN) {
            int K_sh = sh_coeffs_per_channel(_max_sh_degree) * 3;
            CUDA_CHECK(cudaMemset(_shN, 0, M * K_sh * sizeof(float)));
        }
    }

    // ── PLY loading ───────────────────────────────────────────────────────────
    //
    // Loads a trained 2DGS/3DGS checkpoint from a PLY file with the standard
    // gaussian-splatting / gsplat property layout:
    //
    //   x, y, z            — world positions → _means
    //   nx, ny, nz         — normals (ignored)
    //   f_dc_0..2          — degree-0 SH coefficients → _sh0   (raw logit stored as-is)
    //   f_rest_0..44       — degrees 1-3 SH (45 floats per vertex) → _shN
    //   opacity            — logit opacity → _opacity           (raw, no activation applied)
    //   scale_0..2         — log-scale per axis → _scaling      (raw, no activation applied)
    //   rot_0..3           — quaternion (w,x,y,z) → _rotation   (raw, no normalization)
    //
    // All values are stored raw (pre-activation), matching our SplatData convention.

    void init_from_ply(const std::string& path, int max_sh_degree = 3) {
        // ── Parse header ──────────────────────────────────────────────────────
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("cannot open PLY file: " + path);

        std::string line;
        int n_vertices = 0;
        // Map: property name → byte offset within one vertex record
        std::unordered_map<std::string, int> prop_offset;
        int vertex_stride = 0;  // total bytes per vertex

        bool in_vertex_element = false;
        bool header_done = false;
        while (std::getline(f, line)) {
            if (line.back() == '\r') line.pop_back();  // CRLF
            if (line == "end_header") { header_done = true; break; }

            std::istringstream ss(line);
            std::string tok;
            ss >> tok;
            if (tok == "element") {
                std::string name; int count;
                ss >> name >> count;
                in_vertex_element = (name == "vertex");
                if (in_vertex_element) n_vertices = count;
                vertex_stride = 0;  // reset stride for new element
            } else if (tok == "property" && in_vertex_element) {
                std::string type, name;
                ss >> type >> name;
                int sz = (type == "float" || type == "int") ? 4 :
                         (type == "double" || type == "uint64") ? 8 : 4;
                prop_offset[name] = vertex_stride;
                vertex_stride += sz;
            }
        }
        if (!header_done || n_vertices == 0)
            throw std::runtime_error("invalid PLY header in: " + path);

        // ── Required properties ───────────────────────────────────────────────
        auto require = [&](const std::string& name) -> int {
            auto it = prop_offset.find(name);
            if (it == prop_offset.end())
                throw std::runtime_error("PLY missing property: " + name);
            return it->second;
        };

        int off_x       = require("x");
        int off_y       = require("y");
        int off_z       = require("z");
        int off_fdc0    = require("f_dc_0");
        int off_fdc1    = require("f_dc_1");
        int off_fdc2    = require("f_dc_2");
        int off_opacity = require("opacity");
        int off_scale0  = require("scale_0");
        int off_scale1  = require("scale_1");
        int off_scale2  = require("scale_2");
        int off_rot0    = require("rot_0");
        int off_rot1    = require("rot_1");
        int off_rot2    = require("rot_2");
        int off_rot3    = require("rot_3");

        // f_rest offsets (optional — may not all exist for lower SH degrees)
        int n_rest = sh_coeffs_per_channel(max_sh_degree) * 3;
        std::vector<int> off_frest(n_rest, -1);
        for (int k = 0; k < n_rest; k++) {
            auto it = prop_offset.find("f_rest_" + std::to_string(k));
            if (it != prop_offset.end()) off_frest[k] = it->second;
        }

        // ── Read all vertex data ──────────────────────────────────────────────
        std::vector<uint8_t> buf((size_t)n_vertices * vertex_stride);
        f.read(reinterpret_cast<char*>(buf.data()), buf.size());
        if (!f) throw std::runtime_error("PLY read error (truncated?): " + path);
        f.close();

        // ── Unpack into host arrays ───────────────────────────────────────────
        const int N = n_vertices;
        std::vector<float> h_means(N*3), h_rotation(N*4), h_scaling(N*3),
                           h_opacity(N), h_sh0(N*3);

        int K_shN = sh_coeffs_per_channel(max_sh_degree) * 3;
        std::vector<float> h_shN(N * K_shN, 0.f);

        auto rd = [&](const uint8_t* base, int offset) -> float {
            float v; std::memcpy(&v, base + offset, 4); return v;
        };

        for (int i = 0; i < N; i++) {
            const uint8_t* v = buf.data() + (size_t)i * vertex_stride;
            h_means[i*3+0] = rd(v, off_x);
            h_means[i*3+1] = rd(v, off_y);
            h_means[i*3+2] = rd(v, off_z);

            h_sh0[i*3+0]   = rd(v, off_fdc0);
            h_sh0[i*3+1]   = rd(v, off_fdc1);
            h_sh0[i*3+2]   = rd(v, off_fdc2);

            h_opacity[i]   = rd(v, off_opacity);

            h_scaling[i*3+0] = rd(v, off_scale0);
            h_scaling[i*3+1] = rd(v, off_scale1);
            h_scaling[i*3+2] = rd(v, off_scale2);

            // rot_0=w, rot_1=x, rot_2=y, rot_3=z — same (w,x,y,z) order as SplatData
            h_rotation[i*4+0] = rd(v, off_rot0);
            h_rotation[i*4+1] = rd(v, off_rot1);
            h_rotation[i*4+2] = rd(v, off_rot2);
            h_rotation[i*4+3] = rd(v, off_rot3);

            for (int k = 0; k < K_shN; k++)
                if (off_frest[k] >= 0)
                    h_shN[i * K_shN + k] = rd(v, off_frest[k]);
        }

        // ── Upload to GPU ─────────────────────────────────────────────────────
        free();
        allocate(N, max_sh_degree);

        CUDA_CHECK(cudaMemcpy(_means,    h_means.data(),    N*3*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_rotation, h_rotation.data(), N*4*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_scaling,  h_scaling.data(),  N*3*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_opacity,  h_opacity.data(),  N  *sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_sh0,      h_sh0.data(),      N*3*sizeof(float), cudaMemcpyHostToDevice));
        if (_shN && K_shN > 0)
            CUDA_CHECK(cudaMemcpy(_shN, h_shN.data(), N*K_shN*sizeof(float), cudaMemcpyHostToDevice));

        // Loaded model has all SH degrees trained — activate them all
        _active_sh_degree = max_sh_degree;

        printf("Loaded PLY: N=%d  max_sh_degree=%d  stride=%d bytes\n",
               N, max_sh_degree, vertex_stride);
    }

    void save_to_ply(const std::string& path) const {
        if (_N <= 0)
            throw std::runtime_error("SplatData::save_to_ply called with no splats");

        std::filesystem::path out_path(path);
        if (out_path.has_parent_path())
            std::filesystem::create_directories(out_path.parent_path());

        const int N = _N;
        const int K_shN = sh_coeffs_per_channel(_max_sh_degree) * 3;

        std::vector<float> h_means(N * 3), h_rotation(N * 4), h_scaling(N * 3),
                           h_opacity(N), h_sh0(N * 3);
        std::vector<float> h_shN;
        if (K_shN > 0)
            h_shN.resize((size_t)N * K_shN);

        CUDA_CHECK(cudaMemcpy(h_means.data(), _means, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rotation.data(), _rotation, N * 4 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_scaling.data(), _scaling, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_opacity.data(), _opacity, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sh0.data(), _sh0, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
        if (K_shN > 0 && _shN)
            CUDA_CHECK(cudaMemcpy(h_shN.data(), _shN, (size_t)N * K_shN * sizeof(float), cudaMemcpyDeviceToHost));

        std::ofstream f(path, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("cannot open output PLY: " + path);

        f << "ply\n";
        f << "format binary_little_endian 1.0\n";
        f << "element vertex " << N << "\n";
        f << "property float x\n";
        f << "property float y\n";
        f << "property float z\n";
        f << "property float nx\n";
        f << "property float ny\n";
        f << "property float nz\n";
        f << "property float f_dc_0\n";
        f << "property float f_dc_1\n";
        f << "property float f_dc_2\n";
        for (int k = 0; k < K_shN; k++)
            f << "property float f_rest_" << k << "\n";
        f << "property float opacity\n";
        f << "property float scale_0\n";
        f << "property float scale_1\n";
        f << "property float scale_2\n";
        f << "property float rot_0\n";
        f << "property float rot_1\n";
        f << "property float rot_2\n";
        f << "property float rot_3\n";
        f << "end_header\n";

        auto wr = [&](float v) {
            f.write(reinterpret_cast<const char*>(&v), sizeof(float));
        };

        for (int i = 0; i < N; i++) {
            wr(h_means[i * 3 + 0]);
            wr(h_means[i * 3 + 1]);
            wr(h_means[i * 3 + 2]);

            wr(0.f);
            wr(0.f);
            wr(0.f);

            wr(h_sh0[i * 3 + 0]);
            wr(h_sh0[i * 3 + 1]);
            wr(h_sh0[i * 3 + 2]);

            for (int k = 0; k < K_shN; k++)
                wr(h_shN[(size_t)i * K_shN + k]);

            wr(h_opacity[i]);

            wr(h_scaling[i * 3 + 0]);
            wr(h_scaling[i * 3 + 1]);
            wr(h_scaling[i * 3 + 2]);

            wr(h_rotation[i * 4 + 0]);
            wr(h_rotation[i * 4 + 1]);
            wr(h_rotation[i * 4 + 2]);
            wr(h_rotation[i * 4 + 3]);
        }

        if (!f.good())
            throw std::runtime_error("PLY write failed: " + path);
    }

    // ── Debug ─────────────────────────────────────────────────────────────────

    void print_summary() const {
        printf("SplatData: N=%d  sh_degree=%d/%d  arrays: %s\n",
               _N, _active_sh_degree, _max_sh_degree,
               _means ? "allocated" : "empty");
    }

private:
    // ── GPU arrays (device pointers) ──────────────────────────────────────────

    float* _means    = nullptr;
    float* _rotation = nullptr;
    float* _scaling  = nullptr;
    float* _opacity  = nullptr;
    float* _sh0      = nullptr;
    float* _shN      = nullptr;   // nullptr when max_sh_degree == 0

    int _N              = 0;
    int _capacity       = 0;
    int _max_sh_degree  = 3;
    int _active_sh_degree = 0;    // increased during training (SH curriculum)

    // ── Helpers ───────────────────────────────────────────────────────────────

    void allocate(int N, int max_sh_degree) {
        _N             = N;
        _capacity      = N;
        _max_sh_degree = max_sh_degree;

        if (N <= 0) return;

        CUDA_CHECK(cudaMalloc(&_means,    N * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&_rotation, N * 4 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&_scaling,  N * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&_opacity,  N     * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&_sh0,      N * 3 * sizeof(float)));

        int K = sh_coeffs_per_channel(max_sh_degree);
        if (K > 0)
            CUDA_CHECK(cudaMalloc(&_shN, N * K * 3 * sizeof(float)));
    }

    void free() {
        cudaFree(_means);    _means    = nullptr;
        cudaFree(_rotation); _rotation = nullptr;
        cudaFree(_scaling);  _scaling  = nullptr;
        cudaFree(_opacity);  _opacity  = nullptr;
        cudaFree(_sh0);      _sh0      = nullptr;
        cudaFree(_shN);      _shN      = nullptr;
        _N = 0;
        _capacity = 0;
    }

    void steal(SplatData& o) {
        _means    = o._means;    o._means    = nullptr;
        _rotation = o._rotation; o._rotation = nullptr;
        _scaling  = o._scaling;  o._scaling  = nullptr;
        _opacity  = o._opacity;  o._opacity  = nullptr;
        _sh0      = o._sh0;      o._sh0      = nullptr;
        _shN      = o._shN;      o._shN      = nullptr;
        _N              = o._N;              o._N              = 0;
        _capacity       = o._capacity;       o._capacity       = 0;
        _max_sh_degree  = o._max_sh_degree;
        _active_sh_degree = o._active_sh_degree;
    }
};

inline void SplatData::reserve(int new_capacity) {
    if (new_capacity <= _capacity) return;

    float* new_means = nullptr;
    float* new_rotation = nullptr;
    float* new_scaling = nullptr;
    float* new_opacity = nullptr;
    float* new_sh0 = nullptr;
    float* new_shN = nullptr;

    CUDA_CHECK(cudaMalloc(&new_means,    new_capacity * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_rotation, new_capacity * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_scaling,  new_capacity * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_opacity,  new_capacity     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&new_sh0,      new_capacity * 3 * sizeof(float)));

    int K = sh_coeffs_per_channel(_max_sh_degree);
    if (K > 0)
        CUDA_CHECK(cudaMalloc(&new_shN, new_capacity * K * 3 * sizeof(float)));

    if (_N > 0) {
        CUDA_CHECK(cudaMemcpy(new_means, _means, _N * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_rotation, _rotation, _N * 4 * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_scaling, _scaling, _N * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_opacity, _opacity, _N * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_sh0, _sh0, _N * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
        if (new_shN && _shN)
            CUDA_CHECK(cudaMemcpy(new_shN, _shN, _N * K * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    cudaFree(_means);
    cudaFree(_rotation);
    cudaFree(_scaling);
    cudaFree(_opacity);
    cudaFree(_sh0);
    cudaFree(_shN);

    _means = new_means;
    _rotation = new_rotation;
    _scaling = new_scaling;
    _opacity = new_opacity;
    _sh0 = new_sh0;
    _shN = new_shN;
    _capacity = new_capacity;
}
