#!/usr/bin/env python3
"""Interactive viewer for Gaussian-splatting PLY checkpoints via gsplat + nerfview.

Loads a PLY, computes scene bounds, and launches an in-process viser viewer with
the initial camera placed outside the scene looking at its centroid.
Optionally saves a converted .pt checkpoint with --ckpt-out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from .viser_point_cloud_viewer import load_binary_vertex_ply
except ImportError:
    from viser_point_cloud_viewer import load_binary_vertex_ply


def ply_to_gsplat_state(vertices: np.ndarray) -> dict[str, torch.Tensor]:
    names = set(vertices.dtype.names or [])
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"PLY missing required properties: {missing}")

    means = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    quats = np.stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]], axis=1).astype(np.float32)
    scales = np.stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]], axis=1).astype(np.float32)
    opacities = np.asarray(vertices["opacity"], dtype=np.float32)
    sh0 = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)

    rest_names = sorted(
        [name for name in names if name.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if len(rest_names) % 3 != 0:
        raise ValueError(f"Expected f_rest_* property count divisible by 3, got {len(rest_names)}")

    if rest_names:
        flat_rest = np.stack([vertices[name] for name in rest_names], axis=1).astype(np.float32)
        sh_coeffs = len(rest_names) // 3
        shN = flat_rest.reshape(len(vertices), 3, sh_coeffs).transpose(0, 2, 1).copy()
    else:
        shN = np.zeros((len(vertices), 0, 3), dtype=np.float32)

    return {
        "means": torch.from_numpy(means),
        "quats": torch.from_numpy(quats),
        "scales": torch.from_numpy(scales),
        "opacities": torch.from_numpy(opacities),
        "sh0": torch.from_numpy(sh0[:, None, :].copy()),
        "shN": torch.from_numpy(shN),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch gsplat's simple viewer from a PLY checkpoint.")
    parser.add_argument("--ply", type=Path, required=True, help="input PLY checkpoint")
    parser.add_argument("--ckpt-out", type=Path, default=None, help="output gsplat .pt checkpoint path")
    parser.add_argument("--gsplat-root", type=Path, default=Path("/home/ubuntu/repos/gsplat"), help="local gsplat repo root")
    parser.add_argument("--port", type=int, default=8081, help="viewer port")
    parser.add_argument("--convert-only", action="store_true", help="only write the converted checkpoint")
    return parser.parse_args()


def _run_viewer_inline(
    state: dict,
    gsplat_root: Path,
    port: int,
    scene_center: "np.ndarray",
    scene_radius: float,
) -> int:
    """Launch the viewer in-process so we can set the initial camera from scene bounds."""
    import math
    import time

    if str(gsplat_root) not in sys.path:
        sys.path.insert(0, str(gsplat_root))

    try:
        import nerfview
        import viser
        import torch.nn.functional as F
        from gsplat.rendering import rasterization
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splats = state["splats"]
    means     = splats["means"].to(device)
    quats     = F.normalize(splats["quats"], p=2, dim=-1).to(device)
    scales    = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    colors    = torch.cat([splats["sh0"], splats["shN"]], dim=-2).to(device)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    print(f"Gaussians: {len(means)}  SH degree: {sh_degree}")

    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh):
        W, H = img_wh
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K   = torch.from_numpy(camera_state.get_K(img_wh)).float().to(device)
        render_colors, _, _ = rasterization(
            means, quats, scales, opacities, colors,
            c2w.inverse()[None], K[None], W, H,
            sh_degree=sh_degree, render_mode="RGB", radius_clip=3,
        )
        return (render_colors[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()

    server = viser.ViserServer(port=port, verbose=False)

    # Place the camera outside the scene, looking at its centroid.
    cx, cy, cz = float(scene_center[0]), float(scene_center[1]), float(scene_center[2])
    r = scene_radius
    server.initial_camera.look_at = (cx, cy, cz)
    server.initial_camera.position = (cx + r * 0.5, cy - r * 0.5, cz + r * 0.5)

    _ = nerfview.Viewer(server=server, render_fn=viewer_render_fn, mode="rendering")
    print(f"Viewer running at http://localhost:{port}  (Ctrl+C to exit)")
    print(f"  scene center=({cx:.2f}, {cy:.2f}, {cz:.2f})  radius={r:.2f}")
    try:
        time.sleep(100000)
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    args = parse_args()

    if not args.ply.exists():
        print(f"PLY not found: {args.ply}", file=sys.stderr)
        return 1

    vertices = load_binary_vertex_ply(args.ply)
    state = {"splats": ply_to_gsplat_state(vertices)}

    # Compute scene bounds for camera placement.
    means_np = state["splats"]["means"].numpy()
    scene_center = means_np.mean(axis=0)
    extents = means_np.max(axis=0) - means_np.min(axis=0)
    scene_radius = float(extents.max()) / 2.0

    if args.ckpt_out is not None:
        ckpt_path = args.ckpt_out
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt_path)
        print(f"Converted checkpoint: {ckpt_path}")

    if args.convert_only:
        return 0

    return _run_viewer_inline(state, args.gsplat_root, args.port, scene_center, scene_radius)


if __name__ == "__main__":
    raise SystemExit(main())
