// colmap_reader_test.cu
//
// Tests colmap_reader.hpp by writing a minimal synthetic cameras.bin +
// images.bin + points3D.bin, then loading them back and verifying the result.
// No real dataset needed.

#include "colmap_reader.hpp"
#include <cstdio>
#include <cstring>
#include <cassert>

// ─── Helpers to write COLMAP binary files ─────────────────────────────────────

template <typename T>
static void wr(std::ofstream& f, T val) {
    f.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

static void write_synthetic_scene(const std::string& dir) {
    // cameras.bin: 1 PINHOLE camera, 640x480, fx=500, fy=500, cx=320, cy=240
    {
        std::ofstream f(dir + "/cameras.bin", std::ios::binary);
        wr<uint64_t>(f, 1);          // num_cameras
        wr<uint32_t>(f, 1);          // camera_id = 1
        wr<int32_t>(f,  1);          // model = PINHOLE
        wr<uint64_t>(f, 640);        // width
        wr<uint64_t>(f, 480);        // height
        wr<double>(f, 500.0);        // fx
        wr<double>(f, 500.0);        // fy
        wr<double>(f, 320.0);        // cx
        wr<double>(f, 240.0);        // cy
    }

    // images.bin: 2 images with known poses
    // Image 1: identity rotation, translation (0,0,5) — camera 5m above origin looking down -Z
    // Image 2: 90° rotation around Y, translation (3,0,0)
    {
        std::ofstream f(dir + "/images.bin", std::ios::binary);
        wr<uint64_t>(f, 2);           // num_images

        // Image 1 — identity quat (w=1,x=0,y=0,z=0), t=(0,0,5)
        wr<uint32_t>(f, 1);                                    // image_id
        wr<double>(f, 1.0); wr<double>(f, 0.0);               // qw, qx
        wr<double>(f, 0.0); wr<double>(f, 0.0);               // qy, qz
        wr<double>(f, 0.0); wr<double>(f, 0.0); wr<double>(f, 5.0); // tx, ty, tz
        wr<uint32_t>(f, 1);                                    // camera_id
        f.write("frame_001.png\0", 14);                        // name + null
        wr<uint64_t>(f, 0);                                    // num_points2D

        // Image 2 — 90° around Y: quat (w=cos45°, x=0, y=sin45°, z=0)
        double c = 0.7071068, s = 0.7071068;
        wr<uint32_t>(f, 2);
        wr<double>(f, c);  wr<double>(f, 0.0);
        wr<double>(f, s);  wr<double>(f, 0.0);
        wr<double>(f, 3.0); wr<double>(f, 0.0); wr<double>(f, 0.0);
        wr<uint32_t>(f, 1);
        f.write("frame_002.png\0", 14);
        wr<uint64_t>(f, 0);
    }

    // points3D.bin: 3 points at known positions
    {
        std::ofstream f(dir + "/points3D.bin", std::ios::binary);
        wr<uint64_t>(f, 3);
        double pts[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
        for (auto& p : pts) {
            wr<uint64_t>(f, 0);                     // point3D_id
            wr<double>(f, p[0]); wr<double>(f, p[1]); wr<double>(f, p[2]);
            f.write("\x00\x00\x00", 3);             // rgb
            wr<double>(f, 0.0);                     // error
            wr<uint64_t>(f, 0);                     // track_len
        }
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    const std::string dir = "/tmp/synthetic_colmap/sparse/0";
    system(("mkdir -p " + dir).c_str());
    write_synthetic_scene(dir);

    ColmapScene scene = load_colmap("/tmp/synthetic_colmap");

    bool ok = true;

    // Check camera count
    ok &= (scene.cameras.size() == 2);
    printf("[%s] num cameras = %zu (expected 2)\n",
           scene.cameras.size() == 2 ? "OK" : "FAIL", scene.cameras.size());

    // Check intrinsics of first camera
    const Intrinsics& K = scene.cameras[0].K;
    ok &= (K.fx == 500.f && K.fy == 500.f && K.cx == 320.f && K.cy == 240.f);
    printf("[%s] intrinsics: fx=%.0f fy=%.0f cx=%.0f cy=%.0f (expected 500 500 320 240)\n",
           (K.fx==500&&K.fy==500&&K.cx==320&&K.cy==240) ? "OK" : "FAIL",
           K.fx, K.fy, K.cx, K.cy);

    // Camera 1: identity rotation → camtoworld should also be identity rotation
    // w2c has R=I, t=(0,0,5), so c2w has R=I, t=(0,0,-5)
    const Mat4& c2w = scene.cameras[0].camtoworld;
    bool rot_ok = (fabsf(c2w[0][0]-1)<1e-5 && fabsf(c2w[1][1]-1)<1e-5 && fabsf(c2w[2][2]-1)<1e-5);
    bool t_ok   = (fabsf(c2w[0][3])<1e-5 && fabsf(c2w[1][3])<1e-5 && fabsf(c2w[2][3]+5)<1e-5);
    ok &= (rot_ok && t_ok);
    printf("[%s] cam0 camtoworld: R=I=%s  t=(%.1f,%.1f,%.1f) (expected 0,0,-5)\n",
           (rot_ok&&t_ok) ? "OK" : "FAIL", rot_ok?"yes":"no",
           c2w[0][3], c2w[1][3], c2w[2][3]);

    // Point cloud
    ok &= (scene.points3D.size() == 3);
    printf("[%s] num points3D = %zu (expected 3)\n",
           scene.points3D.size()==3 ? "OK" : "FAIL", scene.points3D.size());

    printf("\n%s\n\n", ok ? "All tests passed." : "SOME TESTS FAILED.");
    return ok ? 0 : 1;
}
