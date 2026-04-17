#!/usr/bin/env python3
"""Interactive rendered viewer for exported 2DGS checkpoints.

This uses `viser` as the browser frontend and the CUDA trainer binary as a
persistent subprocess renderer. The subprocess keeps the checkpoint loaded,
accepts camera requests over stdin, and writes PNG frames to a temporary file.

Compared to `viser_point_cloud_viewer.py`, this shows actual splat renders
instead of only Gaussian centers.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

try:
    from .viser_point_cloud_viewer import decode_point_cloud, load_binary_vertex_ply
except ImportError:
    from viser_point_cloud_viewer import decode_point_cloud, load_binary_vertex_ply


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def camera_to_c2w(camera) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = quat_to_rotmat_wxyz(np.asarray(camera.wxyz, dtype=np.float32))
    c2w[:3, 3] = np.asarray(camera.position, dtype=np.float32)
    return c2w


def make_camera_request(camera, render_width: int) -> str:
    width = max(64, int(render_width))
    aspect = max(float(camera.aspect), 1e-4)
    height = max(64, int(round(width / aspect)))
    fy = 0.5 * height / math.tan(0.5 * float(camera.fov))
    fx = fy
    cx = 0.5 * width
    cy = 0.5 * height
    c2w = camera_to_c2w(camera).reshape(-1)
    vals = [width, height, fx, fy, cx, cy, *c2w.tolist()]
    parts = []
    for i, v in enumerate(vals):
        if i < 2:
            parts.append(str(int(v)))
        else:
            parts.append(f"{float(v):.9g}")
    return " ".join(parts) + "\n"


class RenderProcess:
    def __init__(self, renderer: Path, ply: Path, output_png: Path, sh_degree: int) -> None:
        cmd = [
            str(renderer),
            "--ply",
            str(ply),
            "--sh-degree",
            str(sh_degree),
            "--serve-render",
            "--serve-output",
            str(output_png),
        ]
        self.output_png = output_png
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._wait_for_ready()

    def _read_protocol_line(self) -> str:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("renderer subprocess exited unexpectedly")
            line = line.strip()
            if not line:
                continue
            if line == "READY" or line.startswith("OK") or line.startswith("ERR"):
                return line
            print(f"[renderer] {line}", file=sys.stderr)

    def _wait_for_ready(self) -> None:
        line = self._read_protocol_line()
        if line != "READY":
            raise RuntimeError(f"renderer failed to start: {line}")

    def render(self, request_line: str) -> bool:
        assert self.proc.stdin is not None
        self.proc.stdin.write(request_line)
        self.proc.stdin.flush()
        line = self._read_protocol_line()
        return line.startswith("OK")

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive splat viewer for exported 2DGS checkpoints.")
    parser.add_argument("--ply", type=Path, required=True, help="checkpoint PLY to render")
    parser.add_argument(
        "--renderer",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build" / "train",
        help="path to the CUDA renderer binary",
    )
    parser.add_argument("--port", type=int, default=8080, help="viser server port")
    parser.add_argument("--resolution", type=int, default=1024, help="default render width")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="background JPEG quality")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="checkpoint poll interval")
    parser.add_argument("--sh-degree", type=int, default=3, help="max SH degree used when loading the checkpoint")
    parser.add_argument("--point-size", type=float, default=0.0025, help="point-cloud overlay size")
    parser.add_argument("--min-alpha", type=float, default=0.02, help="minimum alpha for point-cloud overlay")
    parser.add_argument("--show-point-cloud", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flip-vertical", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import viser
    except ImportError:
        print("Missing dependency: install `viser` first, e.g. `pip install -r python/viewer_requirements.txt`.", file=sys.stderr)
        return 1

    if not args.renderer.exists():
        print(f"Renderer binary not found: {args.renderer}", file=sys.stderr)
        return 1

    temp_png = Path(tempfile.gettempdir()) / f"project_viewer_render_{int(time.time())}.png"
    renderer = RenderProcess(args.renderer, args.ply, temp_png, args.sh_degree)

    server = viser.ViserServer(port=args.port, verbose=False)
    render_width_slider = server.gui.add_slider("Render Width", min=256, max=1920, step=16, initial_value=args.resolution)
    auto_render_checkbox = server.gui.add_checkbox("Auto Render", initial_value=True)
    show_pc_checkbox = server.gui.add_checkbox("Show Point Cloud", initial_value=args.show_point_cloud)
    render_button = server.gui.add_button("Render Now")
    status_text = server.gui.add_text("Status", initial_value="Starting", disabled=True)
    latency_text = server.gui.add_text("Last Render", initial_value="-", disabled=True)

    point_cloud_handle = None
    last_checkpoint_sig = None
    client_dirty: Dict[int, bool] = {}

    def sync_point_cloud(force: bool = False) -> None:
        nonlocal point_cloud_handle, last_checkpoint_sig
        if not args.ply.exists():
            return
        stat = args.ply.stat()
        sig = (stat.st_mtime_ns, stat.st_size)
        if not force and sig == last_checkpoint_sig:
            return
        changed = sig != last_checkpoint_sig
        last_checkpoint_sig = sig
        if not show_pc_checkbox.value:
            if point_cloud_handle is not None:
                point_cloud_handle.remove()
                point_cloud_handle = None
            if changed:
                mark_all_dirty()
            return
        vertices = load_binary_vertex_ply(args.ply)
        points, colors, _alpha = decode_point_cloud(vertices, args.min_alpha)
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
        if changed:
            mark_all_dirty()

    def mark_all_dirty() -> None:
        for client in server.get_clients().values():
            client_dirty[client.client_id] = True

    @render_width_slider.on_update
    def _(_event) -> None:
        mark_all_dirty()

    @show_pc_checkbox.on_update
    def _(_event) -> None:
        sync_point_cloud(force=True)
        mark_all_dirty()

    @render_button.on_click
    def _(_event) -> None:
        mark_all_dirty()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client_dirty[client.client_id] = True

        @client.camera.on_update
        def _(_cam) -> None:
            if auto_render_checkbox.value:
                client_dirty[client.client_id] = True

    status_text.value = f"Viewer listening on http://localhost:{args.port}"

    try:
        while True:
            sync_point_cloud()

            for client in list(server.get_clients().values()):
                if not client_dirty.get(client.client_id, True):
                    continue
                if not args.ply.exists():
                    status_text.value = f"Waiting for checkpoint: {args.ply}"
                    continue

                request_line = make_camera_request(client.camera, render_width_slider.value)
                t0 = time.perf_counter()
                ok = renderer.render(request_line)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if not ok:
                    status_text.value = "Render failed"
                    continue

                img = np.asarray(Image.open(temp_png).convert("RGB"))
                if args.flip_vertical:
                    img = np.flipud(img).copy()
                client.scene.set_background_image(
                    img,
                    format="jpeg",
                    jpeg_quality=args.jpeg_quality,
                )
                client_dirty[client.client_id] = False
                status_text.value = f"Rendered {img.shape[1]}x{img.shape[0]}"
                latency_text.value = f"{dt_ms:.1f} ms"

            time.sleep(args.poll_seconds if not server.get_clients() else 0.05)
    except KeyboardInterrupt:
        print("\nViewer stopped.")
        return 0
    finally:
        renderer.close()


if __name__ == "__main__":
    raise SystemExit(main())
