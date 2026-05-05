#!/usr/bin/env python3
"""Render an overall grid plot: kernel x shape, cell = % vs best baseline.

For each (kernel, shape) cell:
  - best_baseline = max over available baseline TFLOPS (NCCL, plus cx7 where present)
  - delta_pct     = (ours - best_baseline) / best_baseline * 100
  - cell color    = green (alpha 0.6) if delta_pct >= 0 else red (alpha 0.6),
                    saturation scaled by |delta_pct|
  - cell text     = signed percentage (e.g. "+42%" or "-13%"), with the per-kernel
                    shape label below it so the column meaning stays readable.

Reads the same JSONs as plot_tflops_efa.py and writes
release/plots/overall_grid_efa.png.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

# Reuse formulas + metadata + loader from the existing plot script.
from plot_tflops_efa import KERNELS, load_series, _load_external_q4, _load_triton_dist

HERE = Path(__file__).resolve().parent

# Row order (top → bottom) for the grid. Same order as the 5-panel chart.
ROW_ORDER = ["ag_gemm", "gemm_ar", "dispatch_gemm", "ring_attention", "gemm_rs"]

# Pretty row labels.
ROW_LABELS = {
    "ag_gemm": "AllGather + GEMM",
    "gemm_ar": "GEMM + AllReduce",
    "dispatch_gemm": "MoE Dispatch + GEMM",
    "ring_attention": "Ring Attention",
    "gemm_rs": "GEMM + ReduceScatter",
}

# Cap the color scale at this absolute percentage so a single huge win does
# not wash out the rest of the grid.
PCT_CLIP = 60.0
ALPHA = 0.6


def best_baseline(nccl_tf, cx7_tf, i, *extras, include_cx7=False):
    """Return the max baseline TFLOPS at shape index i, or NaN if none available.
    Extra positional args are additional per-shape baseline lists (e.g. Mercury,
    MagiAttention for the ring_attention row); each is included in the max if
    finite at index i.

    cx7 is included only when `include_cx7=True` (used for dispatch_gemm,
    where DeepEP+DeepGEMM is the canonical reference). For other rows, cx7
    is a separate-cluster reference, not apples-to-apples on EFA, so it is
    excluded from the WIN/LOSS calculation by default."""
    candidates = []
    if nccl_tf is not None and i < len(nccl_tf) and math.isfinite(nccl_tf[i]):
        candidates.append(nccl_tf[i])
    if include_cx7 and cx7_tf is not None and i < len(cx7_tf) \
            and math.isfinite(cx7_tf[i]):
        candidates.append(cx7_tf[i])
    for extra in extras:
        if extra is not None and i < len(extra) and math.isfinite(extra[i]):
            candidates.append(extra[i])
    return max(candidates) if candidates else float("nan")


def main():
    # Decide the column count: every kernel has 5 shapes that line up with a
    # baseline in the published charts, so use 5 columns.
    n_cols = 5
    n_rows = len(ROW_ORDER)

    pct_grid = np.full((n_rows, n_cols), np.nan)
    shape_labels = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    pct_text = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    for r, kernel in enumerate(ROW_ORDER):
        shapes, fused_tf, nccl_tf, cx7_tf = load_series(kernel)
        # Ring Attention has two extra published baselines (Mercury, MagiAttention)
        # — include them as candidates for best_baseline so the % win reflects
        # the actually-strongest competitor at each shape rather than just NCCL.
        extra_baselines = ()
        if kernel == "ring_attention":
            # Apples-to-apples = same head-sharding assumption (full H per rank).
            # We exclude TE-CP a2a+p2p (Ulysses+ring) because it shards heads
            # H/8 per rank — different parallelization. All others (Mercury,
            # MagiAttention, FA2+NCCL, TE-CP-ring, ring-flash-attn) keep full-H.
            mercury_tf, magi_tf, te_p2p_tf, _te_a2ap2p_tf, rfa_tf = \
                _load_external_q4(shapes)
            extra_baselines = (mercury_tf, magi_tf, te_p2p_tf, rfa_tf)
        # Triton-distributed: applicable to ag_gemm (ag_gemm), gemm_rs (gemm_rs),
        # ring_attention (ring_attention). Same EFA cluster, apples-to-apples.
        td_tf = _load_triton_dist(kernel, shapes)
        if td_tf is not None:
            extra_baselines = extra_baselines + (td_tf,)
        # dispatch_gemm: include cx7 (DeepEP+DeepGEMM reference) as a baseline
        # candidate. For all other kernels, cx7 is excluded as it was measured
        # on a different fabric.
        include_cx7 = (kernel == "dispatch_gemm")
        label_fn = KERNELS[kernel]["label_fn"]
        # Take the first n_cols shapes (small → large, JSON order is already sorted).
        for c in range(min(n_cols, len(shapes))):
            base = best_baseline(nccl_tf, cx7_tf, c, *extra_baselines,
                                 include_cx7=include_cx7)
            ours = fused_tf[c] if c < len(fused_tf) else float("nan")
            shape_labels[r][c] = label_fn(shapes[c])
            if math.isfinite(base) and math.isfinite(ours) and base > 0:
                pct = (ours - base) / base * 100.0
                pct_grid[r, c] = pct
                pct_text[r][c] = f"{pct:+.0f}%"
            else:
                pct_text[r][c] = "n/a"

    # Build per-cell RGBA colors. Values are clipped only for color intensity;
    # the cell text still reports the true signed percentage.
    cmap = LinearSegmentedColormap.from_list(
        "release_red_white_green",
        ["#C62828", "#FFFFFF", "#2E7D32"],
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

    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.imshow(rgba, aspect="auto", interpolation="nearest")

    # Row labels on the y-axis.
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([ROW_LABELS[k] for k in ROW_ORDER], fontsize=13)

    # Hide the global x ticks; we draw per-cell shape labels instead.
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([f"Config {i+1}" for i in range(n_cols)], fontsize=11)
    ax.tick_params(axis="x", which="both", length=0)

    # Light grid between cells.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # Annotate each cell with the percentage (large) and the shape label (small).
    for r in range(n_rows):
        for c in range(n_cols):
            v = pct_grid[r, c]
            text_color = "black"
            if math.isfinite(v) and abs(v) >= 0.55 * PCT_CLIP:
                text_color = "white"
            ax.text(c, r - 0.12, pct_text[r][c],
                    ha="center", va="center",
                    fontsize=15, fontweight="bold", color=text_color)
            ax.text(c, r + 0.28, shape_labels[r][c],
                    ha="center", va="center",
                    fontsize=9, color=text_color)

    ax.set_xlabel("Shape (small → large, per-kernel labels in cells)", fontsize=13)
    ax.set_title("TFLOPS vs best baseline "
                 "(NCCL on EFA; +DeepEP+DeepGEMM/cx7 for MoE Dispatch+GEMM; "
                 "+Mercury/MagiAttention/TE-CP-ring for Ring Attention)\n"
                 "green = Ours > baseline, red = Ours < baseline (alpha 0.6)",
                 fontsize=11)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Improvement vs best baseline (%)", fontsize=11)
    cbar.set_ticks([-PCT_CLIP, -30, 0, 30, PCT_CLIP])
    cbar.set_ticklabels([f"≤-{PCT_CLIP:.0f}", "-30", "0", "+30", f"≥+{PCT_CLIP:.0f}"])

    fig.tight_layout()
    out = HERE / "overall_grid_efa.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
