# colmap_loader.py
#
# Option 1 data loader: pure Python, no pycolmap dependency.
# Reads COLMAP binary format and returns plain numpy arrays ready to pass
# into C++ kernels via pybind11 bindings.
#
# Output arrays (all float32, contiguous):
#   camtoworlds  [N, 4, 4]  camera-to-world transforms
#   Ks           [N, 3, 3]  intrinsic matrices
#   image_paths  [N]        str paths to images
#   points3D     [M, 3]     initial point cloud (from COLMAP)

import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Number of intrinsic params per COLMAP camera model
_MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 9}

# ─── Binary readers ────────────────────────────────────────────────────────────

def _read_cameras(path: Path) -> Dict[int, dict]:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id   = struct.unpack("<I", f.read(4))[0]   # uint32
            model_id = struct.unpack("<i", f.read(4))[0]   # int32
            width, height = struct.unpack("<QQ", f.read(16))
            n_params = _MODEL_NUM_PARAMS.get(model_id, 4)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {
                "model_id": model_id,
                "width": width, "height": height,
                "params": params,
            }
    return cameras


def _parse_intrinsics(cam: dict) -> np.ndarray:
    """Extract K [3,3] from COLMAP camera params."""
    p = cam["params"]
    model = cam["model_id"]
    if model == 0:              # SIMPLE_PINHOLE: f, cx, cy
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif model == 1:            # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:                       # SIMPLE_RADIAL, RADIAL, OPENCV, …
        fx = fy = p[0]; cx, cy = p[1], p[2]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w,x,y,z) → 3x3 rotation matrix."""
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def _read_images(path: Path) -> List[dict]:
    images = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<I", f.read(4))[0]
            # null-terminated name
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            # skip 2D points (not needed for training setup)
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)  # (x:d, y:d, point3D_id:q) = 24 bytes each
            images.append({
                "id": image_id, "qvec": qvec, "tvec": tvec,
                "camera_id": camera_id, "name": name.decode(),
            })
    return images


def _read_points3d(path: Path) -> np.ndarray:
    points = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            f.read(8)                          # point3D_id
            xyz = struct.unpack("<3d", f.read(24))
            f.read(3)                          # rgb (3 bytes)
            f.read(8)                          # reprojection error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)              # track (image_id, point2D_idx)
            points.append(xyz)
    return np.array(points, dtype=np.float32)

# ─── Public API ────────────────────────────────────────────────────────────────

@dataclass
class ColmapScene:
    camtoworlds: np.ndarray   # [N, 4, 4]  float32, camera-to-world
    Ks:          np.ndarray   # [N, 3, 3]  float32, intrinsics
    image_paths: List[str]    # [N]        absolute paths
    points3D:    np.ndarray   # [M, 3]     float32, initial point cloud
    width:  int
    height: int


def load_colmap(data_dir: str, images_subdir: str = "images") -> ColmapScene:
    root = Path(data_dir)
    # Try sparse/0/ first, then sparse/, then root
    for sparse in ["sparse/0", "sparse", "."]:
        p = root / sparse
        if (p / "cameras.bin").exists():
            sparse_dir = p
            break
    else:
        raise FileNotFoundError(f"No cameras.bin found under {root}")

    cameras = _read_cameras(sparse_dir / "cameras.bin")
    images  = _read_images(sparse_dir  / "images.bin")
    points  = _read_points3d(sparse_dir / "points3D.bin")

    # Sort by name for reproducibility
    images.sort(key=lambda im: im["name"])

    camtoworlds, Ks, paths = [], [], []
    for im in images:
        R   = _qvec_to_rotmat(im["qvec"])          # world-to-cam rotation
        t   = im["tvec"].astype(np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R;  w2c[:3, 3] = t
        camtoworlds.append(np.linalg.inv(w2c))     # cam-to-world

        cam = cameras[im["camera_id"]]
        Ks.append(_parse_intrinsics(cam))
        paths.append(str(root / images_subdir / im["name"]))

    first_cam = cameras[images[0]["camera_id"]]
    return ColmapScene(
        camtoworlds = np.stack(camtoworlds).astype(np.float32),
        Ks          = np.stack(Ks).astype(np.float32),
        image_paths = paths,
        points3D    = points,
        width       = first_cam["width"],
        height      = first_cam["height"],
    )


# ─── Example ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/garden"
    scene = load_colmap(data_dir)

    print(f"Loaded {len(scene.image_paths)} cameras,  {len(scene.points3D)} 3D points")
    print(f"Image size: {scene.width} x {scene.height}")
    print(f"camtoworlds: {scene.camtoworlds.shape}  {scene.camtoworlds.dtype}")
    print(f"Ks:          {scene.Ks.shape}  {scene.Ks.dtype}")
    print(f"\nFirst camera K:\n{scene.Ks[0]}")
    print(f"\nFirst camtoworld:\n{scene.camtoworlds[0]}")
