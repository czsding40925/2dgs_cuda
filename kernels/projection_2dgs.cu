// projection_2dgs.cu
//
// For each 2D Gaussian, computes:
//   - ray_transforms T [N, 3, 3]  — pixel-space matrix used by the rasterizer
//   - means2d         [N, 2]      — projected 2D center in pixel space
//   - radii           [N, 2]      — AABB half-extents in pixels (int32, 0 = culled)
//   - depths          [N]         — camera-space z (for front-to-back sort)
//   - normals         [N, 3]      — splat z-axis in camera space (dual-sided)
//
// T matrix (3×3, pixel space)
// ───────────────────────────
// Built as T = WH^T * K^T, where:
//   WH   = [u_cam | v_cam | p_cam]  (columns: camera-space tangent axes + center)
//   K    = pinhole intrinsics (fx,fy,cx,cy)
//
// T rows in closed form:
//   T[0] = (u_cam.x*fx,  u_cam.y*fy,  u_cam.x*cx + u_cam.y*cy + u_cam.z)
//   T[1] = (v_cam.x*fx,  v_cam.y*fy,  v_cam.x*cx + v_cam.y*cy + v_cam.z)
//   T[2] = (p_cam.x*fx,  p_cam.y*fy,  p_cam.x*cx + p_cam.y*cy + p_cam.z)
//
// At rasterize time, for pixel (px, py):
//   h_u = px * T[2] - T[0]        (homogeneous plane for u direction)
//   h_v = py * T[2] - T[1]        (homogeneous plane for v direction)
//   zeta = cross(h_u, h_v)        (ray-splat intersection, homogeneous)
//   (u, v) = (zeta.x/zeta.z, zeta.y/zeta.z)
//   alpha  = sigmoid(opacity) * exp(-0.5 * (u² + v²))
//
// means2d and radii
// ─────────────────
// Derived analytically from T without a separate projection step:
//   distance    = T2.x² + T2.y² - T2.z²
//   mean2d.x    = (T0·T2 with z-sign) / distance
//   mean2d.y    = (T1·T2 with z-sign) / distance
//   half_ext.x  = mean2d.x² - (T0·T0 with z-sign) / distance
//   radius_x    = ceil(3.33 * sqrt(max(1e-4, half_ext.x)))
//
// Culling: radii set to 0 (and Gaussian skipped) if:
//   - depth < near_plane (behind camera)
//   - AABB entirely outside image
//   - distance == 0 (degenerate projection)
//
// Viewmat convention (row-vector, same as PyTorch/gsplat):
//   p_cam[j] = p_world[0]*viewmat[0*4+j] + p_world[1]*viewmat[1*4+j]
//            + p_world[2]*viewmat[2*4+j] + viewmat[3*4+j]
//   Matrix layout: [R^T | 0; t | 1]  (translation in bottom row)

#include "splat_data.cuh"
#include <cstdio>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

__device__ void quat_to_rotmat_device(float w, float x, float y, float z, float* R) {
    float inv = rsqrtf(w*w + x*x + y*y + z*z);
    w *= inv; x *= inv; y *= inv; z *= inv;
    float x2=x*x, y2=y*y, z2=z*z;
    float xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;
    R[0]=1.f-2.f*(y2+z2); R[1]=2.f*(xy-wz);    R[2]=2.f*(xz+wy);
    R[3]=2.f*(xy+wz);     R[4]=1.f-2.f*(x2+z2); R[5]=2.f*(yz-wx);
    R[6]=2.f*(xz-wy);     R[7]=2.f*(yz+wx);     R[8]=1.f-2.f*(x2+y2);
}

// dot product with sign flip on z: a.x*b.x + a.y*b.y - a.z*b.z
__device__ float dot_zflip(float ax, float ay, float az,
                            float bx, float by, float bz) {
    return ax*bx + ay*by - az*bz;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel — one thread per Gaussian
// ─────────────────────────────────────────────────────────────────────────────

__global__ void projection_2dgs_kernel(
    const float* __restrict__ means,           // [N, 3]
    const float* __restrict__ rotation,        // [N, 4]  raw (w,x,y,z), normalized here
    const float* __restrict__ scaling,         // [N, 3]  log-scale, exp'd here
    const float* __restrict__ viewmat,         // [4, 4]  world→camera (row-vector conv.)
    float fx, float fy, float cx, float cy,    // pinhole intrinsics
    float near_plane,
    int   image_width,
    int   image_height,
    float*   __restrict__ ray_transforms,      // [N, 9]   output T (3×3 row-major)
    float*   __restrict__ means2d,             // [N, 2]   output pixel-space center
    int32_t* __restrict__ radii,               // [N, 2]   output AABB half-extents (0=culled)
    float*   __restrict__ depths,              // [N]      output camera-z
    float*   __restrict__ normals,             // [N, 3]   output splat normal in cam space
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // ── 1. Load + activate ───────────────────────────────────────────────────
    const float* p_raw = means    + i * 3;
    const float* q_raw = rotation + i * 4;
    const float* s_raw = scaling  + i * 3;

    float4 q = SplatData::normalize4(q_raw);    // (w,x,y,z)
    float3 s = SplatData::exp3(s_raw);          // (sx, sy, sz)

    // ── 2. Rotation matrix from quaternion ───────────────────────────────────
    float Rq[9];
    quat_to_rotmat_device(q.x, q.y, q.z, q.w, Rq);

    // ── 3. Scaled tangent axes in world space ────────────────────────────────
    // u_world = col0(Rq) * s.x,  v_world = col1(Rq) * s.y
    // z-scale is 1 for 2DGS (splat is flat — no s.z used)
    // col0(Rq) = (Rq[0], Rq[3], Rq[6])  (row-major storage)
    // col1(Rq) = (Rq[1], Rq[4], Rq[7])
    // col2(Rq) = (Rq[2], Rq[5], Rq[8])  (for the normal)
    float u_world[3] = { Rq[0]*s.x, Rq[3]*s.x, Rq[6]*s.x };
    float v_world[3] = { Rq[1]*s.y, Rq[4]*s.y, Rq[7]*s.y };
    float n_world[3] = { Rq[2],     Rq[5],     Rq[8]     };  // unit z-axis

    // ── 4. Transform to camera space (row-vector convention) ─────────────────
    // out[j] = in[0]*viewmat[0*4+j] + in[1]*viewmat[1*4+j]
    //        + in[2]*viewmat[2*4+j] (+ viewmat[3*4+j] for points)
    float u_cam[3], v_cam[3], p_cam[3], n_cam[3];
    for (int j = 0; j < 3; j++) {
        u_cam[j] = u_world[0]*viewmat[0*4+j]
                 + u_world[1]*viewmat[1*4+j]
                 + u_world[2]*viewmat[2*4+j];
        v_cam[j] = v_world[0]*viewmat[0*4+j]
                 + v_world[1]*viewmat[1*4+j]
                 + v_world[2]*viewmat[2*4+j];
        p_cam[j] = p_raw[0]*viewmat[0*4+j]
                 + p_raw[1]*viewmat[1*4+j]
                 + p_raw[2]*viewmat[2*4+j]
                 + viewmat[3*4+j];
        n_cam[j] = n_world[0]*viewmat[0*4+j]
                 + n_world[1]*viewmat[1*4+j]
                 + n_world[2]*viewmat[2*4+j];
    }

    // ── 5. Depth + near-plane cull ───────────────────────────────────────────
    depths[i] = p_cam[2];
    if (p_cam[2] < near_plane) {
        radii[i*2] = 0; radii[i*2+1] = 0;
        return;
    }

    // ── 6. Build T = M columns  (T = K * WH, stored as [u_M | v_M | w_M]) ────
    //
    // M = K * WH where WH = [u_cam | v_cam | p_cam].
    // We store M as three column vectors u_M, v_M, w_M (9 floats total).
    //
    // M column k = K * WH[:,k], i.e. K applied to each of u_cam, v_cam, p_cam.
    //
    // With K = [[fx,0,cx],[0,fy,cy],[0,0,1]]:
    //
    //   u_M = K * u_cam = (fx*u.x + cx*u.z,  fy*u.y + cy*u.z,  u.z)
    //   v_M = K * v_cam = (fx*v.x + cx*v.z,  fy*v.y + cy*v.z,  v.z)
    //   w_M = K * p_cam = (fx*p.x + cx*p.z,  fy*p.y + cy*p.z,  p.z)
    //
    // The rasterizer reads these as rows of ray_transforms[9]:
    //   ray_transforms[0..2] = u_M
    //   ray_transforms[3..5] = v_M
    //   ray_transforms[6..8] = w_M
    // Each u_M[j] = K_row0 · WH_col_j, where WH cols are u_cam, v_cam, p_cam.
    // K_row0=(fx,0,cx), K_row1=(0,fy,cy), K_row2=(0,0,1)
    float u_M[3] = {
        fx*u_cam[0] + cx*u_cam[2],  // K_row0 · u_cam
        fx*v_cam[0] + cx*v_cam[2],  // K_row0 · v_cam
        fx*p_cam[0] + cx*p_cam[2],  // K_row0 · p_cam
    };
    float v_M[3] = {
        fy*u_cam[1] + cy*u_cam[2],  // K_row1 · u_cam
        fy*v_cam[1] + cy*v_cam[2],  // K_row1 · v_cam
        fy*p_cam[1] + cy*p_cam[2],  // K_row1 · p_cam
    };
    float w_M[3] = {
        u_cam[2],  // K_row2 · u_cam = u.z
        v_cam[2],  // K_row2 · v_cam = v.z
        p_cam[2],  // K_row2 · p_cam = p.z
    };

    // ── 7. Compute means2d and radii from u_M, v_M, w_M ─────────────────────
    // All formulas operate on w_M (the projected center column):
    //   distance = w_M.x² + w_M.y² - w_M.z²   (always negative for depth > 0)
    //   mean2d.x = dot_zflip(u_M, w_M) / distance
    //   mean2d.y = dot_zflip(v_M, w_M) / distance
    //   half_ext.x = mean2d.x² - dot_zflip(u_M, u_M) / distance
    float dist = dot_zflip(w_M[0], w_M[1], w_M[2], w_M[0], w_M[1], w_M[2]);
    if (dist == 0.f) {
        radii[i*2] = 0; radii[i*2+1] = 0;
        return;
    }
    float inv_dist = 1.f / dist;

    float mx = dot_zflip(u_M[0], u_M[1], u_M[2], w_M[0], w_M[1], w_M[2]) * inv_dist;
    float my = dot_zflip(v_M[0], v_M[1], v_M[2], w_M[0], w_M[1], w_M[2]) * inv_dist;

    float hx = mx*mx - dot_zflip(u_M[0], u_M[1], u_M[2], u_M[0], u_M[1], u_M[2]) * inv_dist;
    float hy = my*my - dot_zflip(v_M[0], v_M[1], v_M[2], v_M[0], v_M[1], v_M[2]) * inv_dist;

    // Cap radius to 1/3 of the image shortest side to avoid tile explosion.
    // Gaussians that are too large relative to the image dominate the tile
    // budget without contributing meaningful detail.
    const float max_radius = fmaxf(4.f, (float)min(image_width, image_height) / 3.f);

    float rx = fminf(max_radius, ceilf(3.33f * sqrtf(fmaxf(1e-4f, hx))));
    float ry = fminf(max_radius, ceilf(3.33f * sqrtf(fmaxf(1e-4f, hy))));

    // Off-screen cull
    if (mx + rx <= 0.f || mx - rx >= (float)image_width ||
        my + ry <= 0.f || my - ry >= (float)image_height) {
        radii[i*2] = 0; radii[i*2+1] = 0;
        return;
    }

    // ── 8. Dual-sided normal ─────────────────────────────────────────────────
    // Flip so normal faces toward the camera (dot(-n_cam, p_cam) > 0)
    float flip = (-(n_cam[0]*p_cam[0] + n_cam[1]*p_cam[1] + n_cam[2]*p_cam[2]) > 0.f)
                 ? 1.f : -1.f;

    // ── 9. Write outputs ─────────────────────────────────────────────────────
    means2d[i*2]   = mx;
    means2d[i*2+1] = my;
    radii[i*2]     = (int32_t)rx;
    radii[i*2+1]   = (int32_t)ry;

    float* T_out = ray_transforms + i * 9;
    T_out[0]=u_M[0]; T_out[1]=u_M[1]; T_out[2]=u_M[2];
    T_out[3]=v_M[0]; T_out[4]=v_M[1]; T_out[5]=v_M[2];
    T_out[6]=w_M[0]; T_out[7]=w_M[1]; T_out[8]=w_M[2];

    float* n_out = normals + i * 3;
    n_out[0]=n_cam[0]*flip; n_out[1]=n_cam[1]*flip; n_out[2]=n_cam[2]*flip;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

void print_mat3(const char* label, const float* T) {
    printf("%s\n", label);
    for (int r = 0; r < 3; r++)
        printf("  [ %+8.4f  %+8.4f  %+8.4f ]\n", T[r*3], T[r*3+1], T[r*3+2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// main — sanity test
// ─────────────────────────────────────────────────────────────────────────────
//
// Scene: 1 Gaussian at (0, 0, 2), identity orientation, unit scale.
// Camera at origin looking along +Z.
// 512×512 image, fx=fy=512 (90° fov), cx=cy=256 (principal point at center).
//
// Expected:
//   depth   = 2.0
//   means2d = (256, 256)  — projects to image center
//   T[row2] encodes p_cam projected through K

#ifndef INCLUDED_AS_HEADER
int main() {
    printf("=== projection_2dgs test ===\n\n");

    const int N = 1;

    float h_means[3]    = { 0.f, 0.f, 2.f };
    float h_rotation[4] = { 1.f, 0.f, 0.f, 0.f };  // identity (w,x,y,z)
    float h_scaling[3]  = { 0.f, 0.f, 0.f };         // log(1)=0 → scale=1

    // Identity viewmat (camera at origin, looking along +Z)
    float h_viewmat[16] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1,
    };

    // Pinhole intrinsics: 512×512 image, 90° fov → fx=fy=512, cx=cy=256
    float fx=512.f, fy=512.f, cx=256.f, cy=256.f;
    int W=512, H=512;
    float near=0.1f;

    // ── GPU allocation ────────────────────────────────────────────────────────
    float   *d_means, *d_rotation, *d_scaling, *d_viewmat;
    float   *d_ray_transforms, *d_means2d, *d_depths, *d_normals;
    int32_t *d_radii;

    CUDA_CHECK(cudaMalloc(&d_means,          N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rotation,       N*4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scaling,        N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_viewmat,        16*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ray_transforms, N*9*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means2d,        N*2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radii,          N*2*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_depths,         N  *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normals,        N*3*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_means,    h_means,    N*3*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rotation, h_rotation, N*4*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scaling,  h_scaling,  N*3*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_viewmat,  h_viewmat,  16*sizeof(float),   cudaMemcpyHostToDevice));

    // ── Launch ────────────────────────────────────────────────────────────────
    projection_2dgs_kernel<<<1, 1>>>(
        d_means, d_rotation, d_scaling, d_viewmat,
        fx, fy, cx, cy, near, W, H,
        d_ray_transforms, d_means2d, d_radii, d_depths, d_normals, N
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Read back ─────────────────────────────────────────────────────────────
    float   h_T[9], h_m2d[2], h_depth, h_normal[3];
    int32_t h_radii[2];
    CUDA_CHECK(cudaMemcpy(h_T,      d_ray_transforms, 9*sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m2d,    d_means2d,        2*sizeof(float),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_radii,  d_radii,          2*sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_depth, d_depths,         sizeof(float),     cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_normal, d_normals,        3*sizeof(float),   cudaMemcpyDeviceToHost));

    // ── Print ─────────────────────────────────────────────────────────────────
    printf("ray_transforms [u_M | v_M | w_M] (pixel space):\n");
    printf("  u_M = (%+8.3f, %+8.3f, %+8.3f)\n", h_T[0], h_T[1], h_T[2]);
    printf("  v_M = (%+8.3f, %+8.3f, %+8.3f)\n", h_T[3], h_T[4], h_T[5]);
    printf("  w_M = (%+8.3f, %+8.3f, %+8.3f)\n", h_T[6], h_T[7], h_T[8]);
    printf("depth:   %.4f  (expected 2.0)\n", h_depth);
    printf("means2d: (%.1f, %.1f)  (expected 256.0, 256.0)\n", h_m2d[0], h_m2d[1]);
    printf("radii:   (%d, %d)\n", h_radii[0], h_radii[1]);
    printf("normal:  (%.3f, %.3f, %.3f)\n", h_normal[0], h_normal[1], h_normal[2]);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_means); cudaFree(d_rotation); cudaFree(d_scaling);
    cudaFree(d_viewmat); cudaFree(d_ray_transforms); cudaFree(d_means2d);
    cudaFree(d_radii); cudaFree(d_depths); cudaFree(d_normals);

    printf("\nDone.\n");
    return 0;
}
#endif // INCLUDED_AS_HEADER
