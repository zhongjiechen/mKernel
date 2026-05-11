#!/usr/bin/env python3
"""Render L20x 3-node per-kernel TFLOPS plots."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "bench" / "results"
WORLD = 24


def _load(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _shape(v) -> int:
    if isinstance(v, int):
        return v
    m = re.search(r"(\d+)$", str(v))
    if not m:
        raise ValueError(f"cannot parse shape {v!r}")
    return int(m.group(1))


def _series(data: dict, key: str) -> tuple[list[int], list[float]]:
    return [_shape(s) for s in data["sizes"]], [float(x) for x in data[key]]


def _safe(ms: float) -> bool:
    return math.isfinite(ms) and ms > 0


def ag_gemm_tflops(m: int, ms: float) -> float:
    return (2.0 * m * m * (m // 16)) / (ms * 1e-3) / 1e12 if _safe(ms) else float("nan")


def gemm_ar_tflops(m: int, ms: float) -> float:
    k = m // WORLD
    return (2.0 * m * k * m) / (ms * 1e-3) / 1e12 if _safe(ms) else float("nan")


def dispatch_tflops(tokens: int, ms: float) -> float:
    h, i, topk = 7168, 2048, 8
    return (2.0 * tokens * topk * h * i / WORLD) / (ms * 1e-3) / 1e12 if _safe(ms) else float("nan")


def ring_tflops(seq: int, ms: float) -> float:
    b, h, d = 4, 16, 128
    # Match the 2-node plot convention: ring_attention chart shapes are
    # total_seq, and TFLOPS are reported per GPU.
    return (4.0 * b * h * seq * seq * d / WORLD) / (ms * 1e-3) / 1e12 if _safe(ms) else float("nan")


def gemm_rs_tflops(m: int, ms: float) -> float:
    k = m // WORLD
    return (2.0 * m * k * m) / (ms * 1e-3) / 1e12 if _safe(ms) else float("nan")


KERNELS = {
    "ag_gemm": ("AllGather + GEMM on L20x3", "Shape M", ag_gemm_tflops),
    "gemm_ar": ("GEMM + AllReduce on L20x3", "Shape M", gemm_ar_tflops),
    "dispatch_gemm": ("MoE Dispatch + GEMM on L20x3", "Tokens", dispatch_tflops),
    "ring_attention": ("Ring Attention on L20x3", "Total sequence length", ring_tflops),
    "gemm_rs": ("GEMM + ReduceScatter on L20x3", "Shape M", gemm_rs_tflops),
}


def plot_kernel(kernel: str) -> None:
    title, xlabel, tflops_fn = KERNELS[kernel]
    ours = _load(RESULTS / f"{kernel}_l20x3.json")
    nccl = _load(RESULTS / f"{kernel}_nccl_l20x3.json")
    shapes, ours_ms = _series(ours, "fused_ms")
    base_shapes, base_ms = _series(nccl, "nccl_ms")
    if shapes != base_shapes:
        ours_by_shape = dict(zip(shapes, ours_ms))
        base_by_shape = dict(zip(base_shapes, base_ms))
        common = [s for s in shapes if s in base_by_shape]
        if not common:
            raise ValueError(f"no common shapes for {kernel}: {shapes} vs {base_shapes}")
        missing_ours = [s for s in base_shapes if s not in ours_by_shape]
        missing_base = [s for s in shapes if s not in base_by_shape]
        print(
            f"{kernel}: plotting common shapes {common}; "
            f"missing ours={missing_ours} missing nccl={missing_base}"
        )
        shapes = common
        ours_ms = [ours_by_shape[s] for s in common]
        base_ms = [base_by_shape[s] for s in common]

    plot_shapes = [s * WORLD for s in shapes] if kernel == "ring_attention" else shapes
    ours_t = [tflops_fn(s, ms) for s, ms in zip(plot_shapes, ours_ms)]
    base_t = [tflops_fn(s, ms) for s, ms in zip(plot_shapes, base_ms)]

    x = np.arange(len(shapes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
    bars_base = ax.bar(x - width / 2, base_t, width, label="CuBLAS+NCCL", color="#8CA0B3")
    bars_ours = ax.bar(x + width / 2, ours_t, width, label="mKernel", color="#D95F02")

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel)
    ylabel = "Algorithmic attention TFLOPS per GPU" if kernel == "ring_attention" else "TFLOPS per GPU"
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in plot_shapes], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)

    top = max([v for v in base_t + ours_t if math.isfinite(v)] + [1.0])
    ax.set_ylim(0, top * 1.22)
    for bars in (bars_base, bars_ours):
        for b in bars:
            h = b.get_height()
            if math.isfinite(h) and h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + top * 0.025,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = HERE / f"{kernel}_l20x3.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    kernels = sys.argv[1:] or list(KERNELS)
    unknown = [kernel for kernel in kernels if kernel not in KERNELS]
    if unknown:
        raise ValueError(f"unknown kernels: {unknown}; valid={list(KERNELS)}")
    for kernel in kernels:
        plot_kernel(kernel)


if __name__ == "__main__":
    main()
