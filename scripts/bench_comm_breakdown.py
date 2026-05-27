#!/usr/bin/env python3
"""scripts/bench_comm_breakdown.py

Plot the comm-vs-compute breakdown + parallel-efficiency curve from a
sweep CSV produced by scripts/bench_strong_scaling.sh or
scripts/bench_weak_scaling.sh.

Inputs:
  - <breakdown.csv>    columns: tag,bucket,mean_ms,std_ms,calls,iters,
                                N_start,N_end,W,H
                       (produced by the M3 profile harness; one row per
                       (run, bucket) pair).
  - <breakdown>_summary.csv  optional sibling with end-to-end wall time:
                             np,...,wall_sec,iters_per_sec,host_stage

Outputs (next to the input CSV):
  - <basename>_bars.png         stacked bar of bucket means across np
  - <basename>_efficiency.png   per-iter wall + parallel efficiency

Run:
  python3 scripts/bench_comm_breakdown.py logs/perf/strong_scaling.csv
"""
from __future__ import annotations

import os
import sys
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Bucket → (display-order, colour-family). Order is the stack order from
# bottom up; mpi_allreduce is split out and coloured red so it pops out of
# the stack visually. Anything not in this list falls through to a grey
# "other" bucket and the script warns.
BUCKET_ORDER = [
    "projection_fwd",
    "tile_intersect",
    "rasterize_fwd",
    "rasterize_bwd",
    "projection_bwd",
    "sh_eval",
    "sh_bwd",
    "nexel_color_fwd",
    "nexel_color_bwd",
    "clear_fwd",
    "clear_bwd",
    "photometric_loss",
    "geometry_loss",
    "densify_accum",
    "adam_step",
    "nexel_adam_step",
    "mpi_allreduce",
    "mpi_allreduce_densify",
]
COMM_BUCKETS = {"mpi_allreduce", "mpi_allreduce_densify"}


def parse_np_from_tag(tag: str) -> int:
    m = re.match(r"np(\d+)", str(tag))
    if not m:
        raise ValueError(f"cannot parse np from tag '{tag}'")
    return int(m.group(1))


def load_breakdown(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tag", "bucket", "mean_ms"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"breakdown CSV missing columns: {missing}")
    df["np"] = df["tag"].map(parse_np_from_tag)
    return df


def load_summary(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_stacked_bars(df: pd.DataFrame, out_png: Path) -> None:
    nps = sorted(df["np"].unique())
    buckets_present = sorted(df["bucket"].unique(),
                             key=lambda b: BUCKET_ORDER.index(b)
                                           if b in BUCKET_ORDER else 1e6)
    unknown = [b for b in buckets_present if b not in BUCKET_ORDER]
    if unknown:
        print(f"[warn] unknown buckets, treated as 'other': {unknown}",
              file=sys.stderr)

    # Pivot: rows = np, cols = bucket, cells = mean_ms (sum if duplicates).
    pivot = (df.groupby(["np", "bucket"])["mean_ms"].sum()
               .unstack(fill_value=0.0)
               .reindex(index=nps, columns=buckets_present, fill_value=0.0))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(nps))
    width = 0.55
    bottoms = np.zeros(len(nps))

    cmap = plt.colormaps.get_cmap("tab20")
    for i, b in enumerate(buckets_present):
        vals = pivot[b].to_numpy()
        if b in COMM_BUCKETS:
            colour = "#d62728"   # red
            edgec = "black"
        else:
            colour = cmap(i % cmap.N)
            edgec = "none"
        ax.bar(x, vals, width=width, bottom=bottoms, label=b,
               color=colour, edgecolor=edgec, linewidth=0.6)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"np={n}" for n in nps])
    ax.set_ylabel("mean ms / iter")
    ax.set_title(f"kernel-bucket breakdown across rank counts\n({out_png.parent.name}/{out_png.name})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_efficiency(df: pd.DataFrame, summary: pd.DataFrame | None,
                    out_png: Path) -> None:
    nps = sorted(df["np"].unique())
    iter_ms = (df.groupby("np")["mean_ms"].sum()
                 .reindex(nps).to_numpy())  # mean ms per iter from buckets
    iter_per_sec_kernel = np.where(iter_ms > 0, 1000.0 / iter_ms, 0.0)

    if summary is not None and "iters_per_sec" in summary.columns:
        ips_wall = (summary.set_index("np")["iters_per_sec"]
                            .reindex(nps).to_numpy(dtype=float))
    else:
        ips_wall = None

    base = iter_per_sec_kernel[0] if iter_per_sec_kernel[0] > 0 else 1.0
    speedup_kernel = iter_per_sec_kernel / base
    eff_kernel = speedup_kernel / np.array(nps, dtype=float)

    fig, (ax_iter, ax_eff) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: per-iter wall time
    ax_iter.plot(nps, iter_ms, marker="o", label="sum-of-buckets")
    if ips_wall is not None:
        wall_ms_per_iter = np.where(ips_wall > 0, 1000.0 / ips_wall, 0.0)
        ax_iter.plot(nps, wall_ms_per_iter, marker="s", linestyle="--",
                     label="end-to-end wall")
    ax_iter.set_xlabel("rank count (np)")
    ax_iter.set_ylabel("ms / iter")
    ax_iter.set_title("per-iter wall time vs np")
    ax_iter.set_xticks(nps)
    ax_iter.grid(True, alpha=0.3)
    ax_iter.legend(fontsize=9)

    # Right: parallel efficiency
    ax_eff.plot(nps, eff_kernel, marker="o", label="efficiency (kernel-only)")
    if ips_wall is not None:
        speedup_wall = ips_wall / (ips_wall[0] if ips_wall[0] > 0 else 1.0)
        eff_wall = speedup_wall / np.array(nps, dtype=float)
        ax_eff.plot(nps, eff_wall, marker="s", linestyle="--",
                    label="efficiency (wall)")
    ax_eff.axhline(1.0, color="gray", linestyle=":", linewidth=1.0,
                   label="ideal")
    ax_eff.set_xlabel("rank count (np)")
    ax_eff.set_ylabel("parallel efficiency")
    ax_eff.set_title("parallel efficiency vs np")
    ax_eff.set_xticks(nps)
    ax_eff.set_ylim(0, max(1.05, eff_kernel.max() * 1.15))
    ax_eff.grid(True, alpha=0.3)
    ax_eff.legend(fontsize=9)

    fig.suptitle(f"{out_png.name}  (host_stage = "
                 f"{summary['host_stage'].iloc[0] if summary is not None and 'host_stage' in summary.columns else 'unknown'})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    in_csv = Path(sys.argv[1]).resolve()
    if not in_csv.exists():
        print(f"no such file: {in_csv}", file=sys.stderr)
        return 1

    df = load_breakdown(in_csv)
    if df.empty:
        print(f"no rows in {in_csv}", file=sys.stderr)
        return 1

    # Sibling summary lookup. Try two naming conventions:
    #   strong_scaling.csv          → strong_summary.csv
    #   strong_scaling_nexel.csv    → strong_summary_nexel.csv  (preferred)
    #                                or strong_nexel_summary.csv (legacy)
    stem = in_csv.stem
    candidates = []
    base = stem.replace("_scaling", "_summary")
    candidates.append(in_csv.parent / f"{base}.csv")
    base2 = stem.replace("_scaling", "")
    candidates.append(in_csv.parent / f"{base2}_summary.csv")

    summary = None
    summary_path = None
    for c in candidates:
        s = load_summary(c)
        if s is not None:
            summary = s
            summary_path = c
            break
    if summary is not None:
        print(f"using wall-time summary: {summary_path}")
    else:
        print(f"(no summary CSV — tried {[str(c) for c in candidates]}; "
              "falling back to kernel-only timing)")

    out_bars = in_csv.with_name(f"{stem}_bars.png")
    out_eff  = in_csv.with_name(f"{stem}_efficiency.png")

    plot_stacked_bars(df, out_bars)
    plot_efficiency(df, summary, out_eff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
