#!/usr/bin/env python3
"""Check if both plot goals are met (zero red cells).

Goal 1: refactored_vs_baseline_efa.png — every fused_ms <= source_of_truth fused_ms.
Goal 2: overall_grid_efa.png            — every fused TFLOPS > best plotted baseline.

Exit 0 if both green, 1 otherwise.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "plots"))
from plot_tflops_efa import load_series, _load_external_q4, _load_triton_dist  # noqa: E402
from plot_overall_grid import best_baseline  # noqa: E402

KERNELS = ["ag_gemm", "gemm_ar", "dispatch_gemm", "ring_attention", "gemm_rs"]


def goal1_reds():
    """Return list of (kernel, shape, pct) reds vs source_of_truth."""
    reds = []
    for k in KERNELS:
        cur = json.load(open(HERE / "bench" / "results" / f"{k}_efa.json"))
        ref = json.load(open(HERE / "bench" / "source_of_truth" / f"{k}.json"))
        rs = [str(s) for s in ref["sizes"]]
        for s, o in zip(cur["sizes"], cur["fused_ms"]):
            s = str(s)
            if s not in rs:
                continue
            b = ref["fused_ms"][rs.index(s)]
            p = (o - b) / b * 100
            if p > 0:
                reds.append((k, s, p))
    return reds


def goal2_reds():
    """Return list of (kernel, shape, pct) reds vs best plotted baseline."""
    reds = []
    for k in KERNELS:
        shapes, fused, nccl, cx7 = load_series(k)
        if k == "ring_attention":
            mercury, magi, te_p2p, _te_a2ap2p, rfa = _load_external_q4(shapes)
            # Exclude TE a2a+p2p (Ulysses+ring): shards heads H/8 per rank,
            # not apples-to-apples vs full-H baselines. Matches the same
            # filtering in plot_overall_grid.py.
            extras = (mercury, magi, te_p2p, rfa)
        else:
            extras = ()
        td = _load_triton_dist(k, shapes)
        if td is not None:
            extras = extras + (td,)
        inc_cx7 = (k == "dispatch_gemm")
        for i, s in enumerate(shapes[:5]):
            b = best_baseline(nccl, cx7, i, *extras, include_cx7=inc_cx7)
            f = fused[i]
            if b <= 0:
                continue
            p = (f - b) / b * 100
            if p < 0:
                reds.append((k, s, p))
    return reds


def main():
    g1 = goal1_reds()
    g2 = goal2_reds()

    print(f"Goal 1 (vs source_of_truth) — refactored_vs_baseline_efa.png")
    if g1:
        for k, s, p in g1:
            print(f"  RED  {k:14s} {s:>10s}  {p:+.2f}%")
    else:
        print("  ALL GREEN")

    print(f"\nGoal 2 (vs best baseline) — overall_grid_efa.png")
    if g2:
        for k, s, p in g2:
            print(f"  RED  {k:14s} shape={s:<10}  {p:+.2f}%")
    else:
        print("  ALL GREEN")

    total = len(g1) + len(g2)
    print(f"\nTotal RED cells: {total} (Goal 1: {len(g1)}, Goal 2: {len(g2)})")
    if total == 0:
        print("DONE — both goals met.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
