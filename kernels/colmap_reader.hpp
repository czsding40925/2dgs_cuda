// colmap_reader.hpp
//
// Option 2: thin header-only C++ COLMAP binary reader.
// No external dependencies beyond the C++ standard library.
//
// Usage:
//   ColmapScene scene = load_colmap("path/to/dataset");
//   // scene.cameras  — vector of CameraInfo (intrinsics + extrinsics)
//   // scene.points3D — vector of float3 (initial point cloud)

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// float3 / float4 stubs when compiling without nvcc
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
#endif

// ─── Data structures ─────────────────────────────────────────────────────────

struct Intrinsics {
    float fx, fy;
    float cx, cy;
    uint32_t width, height;
};

// Row-major 4x4 matrix (same layout as a flat float[16])
struct Mat4 {
    float m[4][4] = {};
    float* operator[](int r) { return m[r]; }
    const float* operator[](int r) const { return m[r]; }
};

struct CameraInfo {
    Intrinsics  K;
    Mat4        camtoworld;   // camera-to-world (c2w), row-major
    std::string image_path;
};

struct ColmapScene {
    std::vector<CameraInfo> cameras;
    std::vector<float3>     points3D;      // XYZ positions
    std::vector<float3>     point_colors;  // RGB in [0, 1], parallel to points3D
};

// ─── Internal helpers ────────────────────────────────────────────────────────

namespace colmap_detail {

// Number of intrinsic params per camera model
static int num_params(int model_id) {
    switch (model_id) {
        case 0: return 3;   // SIMPLE_PINHOLE: f, cx, cy
        case 1: return 4;   // PINHOLE:        fx, fy, cx, cy
        case 2: return 4;   // SIMPLE_RADIAL:  f, cx, cy, k1
        case 3: return 5;   // RADIAL:         f, cx, cy, k1, k2
        case 4: return 8;   // OPENCV:         fx, fy, cx, cy, k1-k2, p1-p2
        default: return 4;
    }
}

template <typename T>
static T read(std::ifstream& f) {
    T val;
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

// COLMAP quaternion (w,x,y,z) → row-major 3x3 rotation matrix
static void quat_to_R(double w, double x, double y, double z, float R[3][3]) {
    double n = std::sqrt(w*w + x*x + y*y + z*z);
    w /= n;  x /= n;  y /= n;  z /= n;
    R[0][0] = float(1 - 2*(y*y + z*z));  R[0][1] = float(2*(x*y - w*z));  R[0][2] = float(2*(x*z + w*y));
    R[1][0] = float(2*(x*y + w*z));      R[1][1] = float(1 - 2*(x*x + z*z)); R[1][2] = float(2*(y*z - w*x));
    R[2][0] = float(2*(x*z - w*y));      R[2][1] = float(2*(y*z + w*x));  R[2][2] = float(1 - 2*(x*x + y*y));
}

// Invert a 4x4 rigid-body matrix (R | t) without a full matrix inverse
static Mat4 invert_rigid(const Mat4& m) {
    Mat4 inv = {};
    // Transpose the rotation block
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            inv[i][j] = m[j][i];
    // New translation = -R^T * t
    for (int i = 0; i < 3; i++) {
        inv[i][3] = 0;
        for (int j = 0; j < 3; j++)
            inv[i][3] -= inv[i][j] * m[j][3];
    }
    inv[3][3] = 1.0f;
    return inv;
}

struct RawCamera {
    int model_id;
    uint64_t width, height;
    std::vector<double> params;
};

static std::unordered_map<uint32_t, RawCamera> read_cameras(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    std::unordered_map<uint32_t, RawCamera> cameras;
    uint64_t num = read<uint64_t>(f);
    for (uint64_t i = 0; i < num; i++) {
        uint32_t cam_id  = read<uint32_t>(f);
        int32_t  model   = read<int32_t>(f);
        uint64_t width   = read<uint64_t>(f);
        uint64_t height  = read<uint64_t>(f);
        int np = num_params(model);
        std::vector<double> params(np);
        f.read(reinterpret_cast<char*>(params.data()), np * sizeof(double));
        cameras[cam_id] = {model, width, height, std::move(params)};
    }
    return cameras;
}

static Intrinsics parse_K(const RawCamera& cam) {
    const auto& p = cam.params;
    float fx, fy, cx, cy;
    if (cam.model_id == 1) {                   // PINHOLE
        fx = float(p[0]);  fy = float(p[1]);
        cx = float(p[2]);  cy = float(p[3]);
    } else {                                   // SIMPLE_PINHOLE, SIMPLE_RADIAL, …
        fx = fy = float(p[0]);
        cx = float(p[1]);  cy = float(p[2]);
    }
    return {fx, fy, cx, cy,
            static_cast<uint32_t>(cam.width),
            static_cast<uint32_t>(cam.height)};
}

} // namespace colmap_detail

// ─── Public API ──────────────────────────────────────────────────────────────

inline ColmapScene load_colmap(
    const std::string& data_dir,
    const std::string& images_subdir = "images"
) {
    using namespace colmap_detail;

    // Find sparse directory
    std::string sparse;
    for (const char* s : {"sparse/0", "sparse", "."}) {
        std::ifstream test(data_dir + "/" + s + "/cameras.bin", std::ios::binary);
        if (test) { sparse = data_dir + "/" + s; break; }
    }
    if (sparse.empty())
        throw std::runtime_error("No cameras.bin found under " + data_dir);

    // ── cameras.bin ──────────────────────────────────────────────────────────
    auto raw_cameras = read_cameras(sparse + "/cameras.bin");

    // ── images.bin ───────────────────────────────────────────────────────────
    std::ifstream f(sparse + "/images.bin", std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open images.bin");

    ColmapScene scene;
    uint64_t num_images = read<uint64_t>(f);
    scene.cameras.reserve(num_images);

    for (uint64_t i = 0; i < num_images; i++) {
        uint32_t image_id = read<uint32_t>(f);
        double qw = read<double>(f), qx = read<double>(f),
               qy = read<double>(f), qz = read<double>(f);
        double tx = read<double>(f), ty = read<double>(f), tz = read<double>(f);
        uint32_t cam_id = read<uint32_t>(f);

        // Read null-terminated image name
        std::string name;
        char c;
        while (f.get(c) && c != '\0') name += c;

        // Skip 2D point observations  (x:d, y:d, point3D_id:i64 = 24 bytes each)
        uint64_t num_pts2d = read<uint64_t>(f);
        f.seekg(num_pts2d * 24, std::ios::cur);

        // Build world-to-camera matrix
        float R[3][3];
        quat_to_R(qw, qx, qy, qz, R);

        Mat4 w2c = {};
        for (int r = 0; r < 3; r++)
            for (int c2 = 0; c2 < 3; c2++)
                w2c[r][c2] = R[r][c2];
        w2c[0][3] = float(tx);  w2c[1][3] = float(ty);  w2c[2][3] = float(tz);
        w2c[3][3] = 1.f;

        CameraInfo cam;
        cam.K          = parse_K(raw_cameras.at(cam_id));
        cam.camtoworld = invert_rigid(w2c);
        cam.image_path = data_dir + "/" + images_subdir + "/" + name;
        scene.cameras.push_back(std::move(cam));
    }

    // ── points3D.bin ─────────────────────────────────────────────────────────
    std::ifstream fp(sparse + "/points3D.bin", std::ios::binary);
    if (fp) {
        uint64_t num_pts = read<uint64_t>(fp);
        scene.points3D.reserve(num_pts);
        scene.point_colors.reserve(num_pts);
        for (uint64_t i = 0; i < num_pts; i++) {
            fp.seekg(8, std::ios::cur);                   // skip point3D_id (uint64)
            double x = read<double>(fp), y = read<double>(fp), z = read<double>(fp);
            uint8_t r = read<uint8_t>(fp), g = read<uint8_t>(fp), b = read<uint8_t>(fp);
            fp.seekg(8, std::ios::cur);                   // skip error (double)
            uint64_t track_len = read<uint64_t>(fp);
            fp.seekg(track_len * 8, std::ios::cur);       // skip track
            scene.points3D.push_back({float(x), float(y), float(z)});
            scene.point_colors.push_back({r / 255.f, g / 255.f, b / 255.f});
        }
    }

    return scene;
}
