#!/usr/bin/env python3
"""Heatmap: refactored (dist::) vs published baseline ms, per (kernel, shape).

Same visual style as overall_grid_efa.png. Cell value = % change in
wall-clock vs the published baseline. Negative (faster) = green, positive
(slower) = red.

Inputs:
  release/bench/results/{kernel}_efa.json        — current refactored
  release/bench/source_of_truth/{kernel}.json    — published baseline
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

HERE    = Path(__file__).resolve().parent
RELEASE = HERE.parent
RESULTS = RELEASE / "bench" / "results"
SOT     = RELEASE / "bench" / "source_of_truth"

ROW_ORDER = ["ag_gemm", "gemm_ar", "dispatch_gemm", "ring_attention", "gemm_rs"]
ROW_LABELS = {
    "ag_gemm":        "AllGather + GEMM",
    "gemm_ar":        "GEMM + AllReduce",
    "dispatch_gemm":  "MoE Dispatch + GEMM",
    "ring_attention": "Ring Attention",
    "gemm_rs":        "GEMM + ReduceScatter",
}

# Color scale: clip at ±20% so the small-shape noise doesn't wash everything out.
PCT_CLIP = 20.0
ALPHA    = 0.65


def shape_label(s):
    if isinstance(s, str):
        return s.replace("M=", "").replace("tokens=", "tk=").replace("seq=", "s=")
    return str(s)


def main():
    n_rows = len(ROW_ORDER)
    n_cols = 5  # take the first 5 shapes; some kernels (gemm_rs) have 6, drop M=65536 to fit

    pct_grid = np.full((n_rows, n_cols), np.nan)
    cell_text = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    shape_lbl = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    abs_text  = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    for r, kernel in enumerate(ROW_ORDER):
        cur = json.load(open(RESULTS / f"{kernel}_efa.json"))
        ref = json.load(open(SOT / f"{kernel}.json"))
        cur_sizes = [str(s) for s in cur["sizes"]]
        ref_sizes = [str(s) for s in ref["sizes"]]
        cur_ms = cur["fused_ms"]
        ref_ms = ref["fused_ms"]

        # Iterate the first n_cols shapes from the refactored series
        for c in range(min(n_cols, len(cur_sizes))):
            sz = cur_sizes[c]
            shape_lbl[r][c] = shape_label(sz)
            if sz in ref_sizes:
                base = ref_ms[ref_sizes.index(sz)]
                ours = cur_ms[c]
                if base > 0 and math.isfinite(base) and math.isfinite(ours):
                    pct = (ours - base) / base * 100.0
                    pct_grid[r, c] = pct
                    cell_text[r][c] = f"{pct:+.1f}%"
                    abs_text[r][c]  = f"{ours:.2f} / {base:.2f} ms"
                else:
                    cell_text[r][c] = "n/a"
            else:
                cell_text[r][c] = "n/a"

    # Color map: red = regression, green = win (faster).
    cmap = LinearSegmentedColormap.from_list(
        "regression_red_white_green",
        ["#2E7D32", "#FFFFFF", "#C62828"],   # green (negative=faster) → white → red (positive=slower)
    )
    norm = TwoSlopeNorm(vmin=-PCT_CLIP, vcenter=0.0, vmax=PCT_CLIP)
    rgba = np.ones((n_rows, n_cols, 4), dtype=float)
    for r in range(n_rows):
        for c in range(n_cols):
            v = pct_grid[r, c]
            if not math.isfinite(v):
                rgba[r, c] = (0.88, 0.88, 0.88, ALPHA)
            else:
                rgba[r, c] = cmap(norm(float(np.clip(v, -PCT_CLIP, PCT_CLIP))))
                rgba[r, c, 3] = ALPHA

    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([ROW_LABELS[k] for k in ROW_ORDER], fontsize=12)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([f"Config {i+1}" for i in range(n_cols)], fontsize=11)
    ax.tick_params(axis="x", which="both", length=0)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for r in range(n_rows):
        for c in range(n_cols):
            v = pct_grid[r, c]
            tc = "black"
            if math.isfinite(v) and abs(v) >= 0.6 * PCT_CLIP:
                tc = "white"
            ax.text(c, r - 0.20, cell_text[r][c],
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color=tc)
            ax.text(c, r + 0.10, abs_text[r][c],
                    ha="center", va="center",
                    fontsize=8, color=tc)
            ax.text(c, r + 0.32, shape_lbl[r][c],
                    ha="center", va="center",
                    fontsize=8, color=tc, style="italic")

    ax.set_xlabel("Shape (small → large, per-kernel labels in cells)", fontsize=12)
    ax.set_title("Refactored (dist::) vs published baseline — wall-clock Δ%\n"
                 "green = refactored faster, red = refactored slower  (cell text: refactored / baseline ms)",
                 fontsize=11)

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("wall-clock Δ vs baseline (%)", fontsize=11)
    cbar.set_ticks([-PCT_CLIP, -10, 0, 10, PCT_CLIP])
    cbar.set_ticklabels([f"≤-{PCT_CLIP:.0f}", "-10", "0", "+10", f"≥+{PCT_CLIP:.0f}"])

    fig.tight_layout()
    out = HERE / "refactored_vs_baseline_efa.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
