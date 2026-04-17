# example.py
#
# Full Option 1 workflow:
#   load COLMAP data in Python → send to CUDA kernels via pybind11
#
# (Assumes gs_kernels.so has been built — see build command in bindings.cpp)

import numpy as np
import sys
sys.path.insert(0, ".")          # find gs_kernels.so in current dir
import gs_kernels                # our pybind11 module

from colmap_loader import load_colmap

# ─── 1. Load scene ────────────────────────────────────────────────────────────

scene = load_colmap("../../data/garden")   # or any COLMAP dataset
print(f"Loaded {len(scene.image_paths)} cameras, {len(scene.points3D)} points")

# ─── 2. Project point cloud with first camera ─────────────────────────────────

# Get world-to-camera transform for camera 0
c2w = scene.camtoworlds[0]                 # [4, 4]
w2c = np.linalg.inv(c2w)
R, t = w2c[:3, :3], w2c[:3, 3]            # rotation, translation

# Transform world points → camera space
pts_world = scene.points3D                 # [M, 3]
pts_cam   = (R @ pts_world.T).T + t       # [M, 3]

# Keep only points in front of camera
mask     = pts_cam[:, 2] > 0
pts_cam  = np.ascontiguousarray(pts_cam[mask].astype(np.float32))

# ─── 3. Call CUDA kernel from Python ─────────────────────────────────────────

K  = scene.Ks[0]
fx, fy = float(K[0, 0]), float(K[1, 1])
cx, cy = float(K[0, 2]), float(K[1, 2])

pixels = gs_kernels.project_points(pts_cam, fx, fy, cx, cy)   # [M, 2]
print(f"Projected {len(pixels)} points — first few: {pixels[:3]}")

# ─── 4. Run quat_to_rotmat on all cameras at once ────────────────────────────

# Extract quaternions from camtoworld rotation matrices
# (real pipeline: Gaussians have stored quats; here we derive them from cameras)
from scipy.spatial.transform import Rotation
c2w_R = scene.camtoworlds[:, :3, :3]         # [N, 3, 3]
quats = Rotation.from_matrix(c2w_R).as_quat()  # scipy gives (x,y,z,w)
quats_wxyz = quats[:, [3, 0, 1, 2]].astype(np.float32)  # reorder to (w,x,y,z)

rotmats = gs_kernels.quat_to_rotmat(quats_wxyz)   # [N, 3, 3]
print(f"Rotation matrices: {rotmats.shape}")

# Sanity: reconstructed R should match original
err = np.max(np.abs(rotmats - c2w_R.astype(np.float32)))
print(f"Max roundtrip error (quat→R vs original R): {err:.6f}")
