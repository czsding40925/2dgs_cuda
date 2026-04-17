#!/usr/bin/env python3
"""Live point-cloud viewer for exported 2DGS checkpoints.

This is the baseline interactive viewer path for the C++/CUDA trainer:
- `train.cu` periodically writes a standard Gaussian-splatting PLY checkpoint
- this script loads the checkpoint and shows Gaussian centers in `viser`
- when the file changes, the point cloud updates in-place

It does not render surface splats yet. It visualizes the learned geometry and
base color distribution, which is still useful for live inspection on AWS.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


C0 = np.float32(0.28209479177387814)

PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_binary_vertex_ply(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        first = f.readline().decode("ascii").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")

        vertex_count = None
        vertex_props: List[Tuple[str, str]] = []
        in_vertex = False

        while True:
            line_b = f.readline()
            if not line_b:
                raise ValueError(f"{path} has no end_header")
            line = line_b.decode("ascii").strip()
            if not line:
                continue
            if line.startswith("format "):
                if "binary_little_endian" not in line:
                    raise ValueError(f"{path} is not binary_little_endian")
            elif line.startswith("element "):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"bad PLY element line: {line}")
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                    vertex_props = []
            elif line.startswith("property ") and in_vertex:
                parts = line.split()
                if len(parts) == 5 and parts[1] == "list":
                    raise ValueError("list properties are not supported")
                if len(parts) != 3:
                    raise ValueError(f"bad PLY property line: {line}")
                dtype = PLY_DTYPE_MAP.get(parts[1])
                if dtype is None:
                    raise ValueError(f"unsupported PLY dtype: {parts[1]}")
                vertex_props.append((parts[2], dtype))
            elif line == "end_header":
                break

        if vertex_count is None or vertex_count <= 0:
            raise ValueError(f"{path} has no vertex element")

        dtype = np.dtype(vertex_props)
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
        if data.shape[0] != vertex_count:
            raise ValueError(f"{path} ended early while reading vertices")
        return data


def decode_point_cloud(vertices: np.ndarray, min_alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.stack(
        [
            np.asarray(vertices["x"], dtype=np.float32),
            np.asarray(vertices["y"], dtype=np.float32),
            np.asarray(vertices["z"], dtype=np.float32),
        ],
        axis=1,
    )
    colors = np.stack(
        [
            np.asarray(vertices["f_dc_0"], dtype=np.float32),
            np.asarray(vertices["f_dc_1"], dtype=np.float32),
            np.asarray(vertices["f_dc_2"], dtype=np.float32),
        ],
        axis=1,
    )
    colors = np.clip(colors * C0 + 0.5, 0.0, 1.0)

    alpha = sigmoid(np.asarray(vertices["opacity"], dtype=np.float32))
    keep = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1) & (alpha >= min_alpha)

    # Slight alpha tint keeps nearly-pruned points visually quieter.
    alpha_vis = alpha[keep, None]
    colors = np.clip(colors[keep] * (0.25 + 0.75 * alpha_vis), 0.0, 1.0)
    return points[keep], colors, alpha[keep]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View exported 2DGS checkpoints in Viser.")
    parser.add_argument("--ply", type=Path, required=True, help="checkpoint PLY to load and watch")
    parser.add_argument("--port", type=int, default=8080, help="viser server port")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="checkpoint refresh interval")
    parser.add_argument("--point-size", type=float, default=0.0025, help="rendered point size in world units")
    parser.add_argument("--min-alpha", type=float, default=0.02, help="drop very transparent splats")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import viser
    except ImportError:
        print("Missing dependency: install `viser` first, e.g. `pip install -r python/viewer_requirements.txt`.", file=sys.stderr)
        return 1

    server = viser.ViserServer(port=args.port, verbose=False)
    point_cloud_handle = None
    last_signature = None
    waiting_reported = False

    print(f"Viewer listening on http://localhost:{args.port}")
    print(f"Watching checkpoint: {args.ply}")

    try:
        while True:
            if not args.ply.exists():
                if not waiting_reported:
                    print(f"Waiting for checkpoint: {args.ply}")
                    waiting_reported = True
                time.sleep(args.poll_seconds)
                continue

            waiting_reported = False
            stat = args.ply.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != last_signature:
                vertices = load_binary_vertex_ply(args.ply)
                points, colors, alpha = decode_point_cloud(vertices, args.min_alpha)
                if point_cloud_handle is None:
                    point_cloud_handle = server.scene.add_point_cloud(
                        "/splats",
                        points=points,
                        colors=colors,
                        point_size=args.point_size,
                    )
                else:
                    point_cloud_handle.points = points
                    point_cloud_handle.colors = colors
                    point_cloud_handle.point_size = args.point_size

                last_signature = signature
                kept = points.shape[0]
                total = vertices.shape[0]
                alpha_min = float(alpha.min()) if alpha.size > 0 else 0.0
                alpha_max = float(alpha.max()) if alpha.size > 0 else 0.0
                print(
                    f"Reloaded {args.ply.name}: kept {kept:,}/{total:,} points  "
                    f"(min_alpha={args.min_alpha:.3f}, alpha_range=[{alpha_min:.3f}, {alpha_max:.3f}])"
                )

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nViewer stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
