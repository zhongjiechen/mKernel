"""Shared bench utilities for the release kernels.

Each <kernel>_bench.py reuses these primitives so the per-kernel script
stays focused on alloc + run_once + correctness reference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.distributed as dist

# Make the python loader importable.
HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
sys.path.insert(0, str(RELEASE / "python"))
import load_module  # noqa: E402


# ----------------------------------------------------------------------
# Distributed init / launcher helpers
# ----------------------------------------------------------------------

_PEER_IP_DEFAULTS = {0: "172.31.1.237", 1: "172.31.11.6"}


def get_peer_ips(node_idx: int, num_nodes: int) -> list[str]:
    """Resolve the list of peer IPs (length num_nodes - 1) for this node.

    Reads NODE0_IP, NODE1_IP, ..., NODE{N-1}_IP from the environment, falling
    back to the 2-node testbed defaults when the env var is unset (only sane
    for N <= 2). Returns the list of OTHER nodes' IPs in slot order, skipping
    self.
    """
    all_ips = []
    for i in range(num_nodes):
        ip = os.environ.get(f"NODE{i}_IP")
        if not ip:
            ip = _PEER_IP_DEFAULTS.get(i, "")
        all_ips.append(ip)
    return [ip for i, ip in enumerate(all_ips) if i != node_idx]


def get_num_nodes() -> int:
    """Resolve number of nodes from the NUM_NODES env var (default: 2).

    Bench scripts and the launcher consult this to size the global world
    (`WORLD_SIZE = NUM_NODES * LOCAL_WORLD_SIZE`) and to drive future
    multi-peer session setup. For now, only NUM_NODES=2 is fully wired
    through the kernels and session layer; >2 prints a WIP warning so we
    don't silently produce wrong results.
    """
    n = int(os.environ.get("NUM_NODES", "2"))
    if n > 2 and int(os.environ.get("LOCAL_RANK", "0")) == 0 \
            and int(os.environ.get("NODE_IDX", "0")) == 0:
        print(f"[bench] NUM_NODES={n} > 2: WIP — kernel/session layer "
              f"still assumes 2 nodes; results for N>2 are not yet "
              f"validated. Tracking: see README §Backends.", flush=True)
    return n


def init_dist():
    """Init torch.distributed from torchrun env. Returns (local_rank, world_size, node_idx)."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "16"))
    node_idx = int(os.environ.get("NODE_IDX", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return local_rank, world_size, node_idx


def host_barrier():
    """Cheap collective barrier on a 1-element CPU tensor (gloo backend)."""
    t = torch.zeros(1, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)


def max_across_ranks(value: float) -> float:
    """Return max(value) across the torchrun world using the CPU gloo group."""
    t = torch.tensor([float(value)], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def maybe_print_rank_values(label: str, value: float) -> None:
    """Optionally print one scalar per rank for timing diagnostics."""
    want_print = torch.tensor(
        [1 if os.environ.get("RELEASE_BENCH_PRINT_RANK_TIMES", "0") == "1" else 0],
        dtype=torch.int32,
    )
    dist.all_reduce(want_print, op=dist.ReduceOp.MAX)
    if int(want_print.item()) == 0:
        return
    t = torch.tensor([float(value)], dtype=torch.float64)
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    if dist.get_rank() == 0:
        rank_ms = [round(float(x.item()), 3) for x in gathered]
        print(f"[rank-times] {label}: {rank_ms}", flush=True)


def avg_then_max(samples: list[float], label: str = "") -> float:
    """Experiment-compatible timing: local average, then max over all ranks."""
    local = sum(samples) / len(samples) if samples else float("nan")
    if label:
        maybe_print_rank_values(label, local)
    return max_across_ranks(local)


def median_then_max(samples: list[float], label: str = "") -> float:
    """Experiment-compatible timing: local median, then max over all ranks."""
    if not samples:
        return max_across_ranks(float("nan"))
    sorted_samples = sorted(samples)
    local = sorted_samples[len(sorted_samples) // 2]
    if label:
        maybe_print_rank_values(label, local)
    return max_across_ranks(local)


# ----------------------------------------------------------------------
# Bench timing wrapper
# ----------------------------------------------------------------------

def run_bench_loop(
    run_once: Callable[[], None],
    iters: int,
    warmup: int,
    reset_state: Callable[[], None] = None,
) -> list[float]:
    """Standard timed loop. Returns per-iter wall (ms).

    Pattern (load-bearing — see commit 23d15b4):
      for warmup_iter:
          reset_state(); run_once(); cuda.synchronize()
      for iter:
          reset_state()
          host_barrier(); time.sleep(0.05)   # proxy CQE drain
          start.record(); run_once(); end.record(); cuda.synchronize()
          samples.append(start.elapsed_time(end))
    """
    for _ in range(warmup):
        if reset_state is not None:
            reset_state()
        run_once()
        torch.cuda.synchronize()
        host_barrier()

    samples = []
    for _ in range(iters):
        if reset_state is not None:
            reset_state()
        host_barrier()
        time.sleep(0.05)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_once()
        e.record()
        torch.cuda.synchronize()
        host_barrier()
        samples.append(s.elapsed_time(e))
    return samples


# ----------------------------------------------------------------------
# Reproduction check
# ----------------------------------------------------------------------

def compare_to_reference(
    observed_ms: list[float],
    reference_json: Path,
    tolerance: float = 0.05,
) -> bool:
    """Compare observed wall-times to the source-of-truth JSON.

    Returns True if every shape passes (observed ≤ reference × (1 + tolerance)).
    Prints PASS/FAIL per shape. Caller should sys.exit(non-zero) on failure.
    """
    if not reference_json.exists():
        print(f"[reproduce-check] reference JSON not found: {reference_json}",
              file=sys.stderr)
        return False
    with open(reference_json) as f:
        ref = json.load(f)
    ref_ms = ref["fused_ms"]
    ref_sizes = ref["sizes"]
    if len(observed_ms) != len(ref_ms):
        print(f"[reproduce-check] shape count mismatch: "
              f"observed={len(observed_ms)} reference={len(ref_ms)}", file=sys.stderr)
        return False
    all_pass = True
    print(f"[reproduce-check] vs {reference_json.name}, tolerance ±{tolerance*100:.0f}%")
    for i, (obs, ref_v, sz) in enumerate(zip(observed_ms, ref_ms, ref_sizes)):
        upper = ref_v * (1.0 + tolerance)
        ok = obs <= upper
        mark = "PASS" if ok else "FAIL"
        delta_pct = (obs - ref_v) / ref_v * 100
        print(f"  shape {sz}: observed {obs:.3f} ms vs ref {ref_v:.3f} ms "
              f"({delta_pct:+.1f}%)  [{mark}]")
        all_pass = all_pass and ok
    return all_pass


def compare_named_results(
    kernel: str,
    result_sizes: list[str],
    result_fused: list[float],
    reference_json: str | Path,
    tolerance: float = 0.05,
) -> bool:
    """Compare result rows by shape label, allowing targeted shape runs."""
    reference_json = Path(reference_json)
    with open(reference_json) as f:
        ref = json.load(f)
    ref_by_size = dict(zip(ref["sizes"], ref["fused_ms"]))
    ok = True
    print(f"[{kernel}] vs {reference_json} (tol ±{tolerance * 100:.0f}%)", flush=True)
    for sz, obs in zip(result_sizes, result_fused):
        if sz not in ref_by_size:
            print(f"  {sz}: no reference row [FAIL]", flush=True)
            ok = False
            continue
        refv = ref_by_size[sz]
        delta = (obs - refv) / refv * 100
        mark = "PASS" if obs <= refv * (1.0 + tolerance) else "FAIL"
        if obs > refv * (1.0 + tolerance):
            ok = False
        print(f"  {sz}: obs={obs:.3f} ref={refv:.3f} ({delta:+.1f}%) {mark}", flush=True)
    return ok


# ----------------------------------------------------------------------
# Result JSON writer
# ----------------------------------------------------------------------

def write_results_json(
    out_path: Path,
    kernel: str,
    sizes: Iterable,
    fused_ms: Iterable[float],
    note: str = "",
):
    """Write a release-format JSON. Schema matches experiments/multinode/sweep/results/efa/.

    Crucial: this function MERGES with the existing JSON at `out_path` rather
    than replacing it wholesale. A bench run with `--shapes 4096` shouldn't
    erase results for the other 4 shapes that the chart needs. New entries
    overwrite same-shape entries; absent shapes are preserved.
    """
    new_sizes = list(sizes)
    new_ms = list(fused_ms)
    out_path = Path(out_path)
    existing_sizes: list = []
    existing_ms: list = []
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.load(open(out_path))
            existing_sizes = list(existing.get("sizes", []))
            existing_ms = list(existing.get("fused_ms", []))
        except Exception:
            existing_sizes, existing_ms = [], []

    # Build a merged map: existing shapes first, new ones overwrite.
    # If OSGC_BENCH_BEST_OF_N=1, between-sweep best-of: take min when same
    # shape is rewritten. Used by run.sh's BEST_OF_N=k loop to absorb
    # cluster-noise variance. Reset by deleting the JSON before the loop.
    best_of = os.environ.get("OSGC_BENCH_BEST_OF_N", "0") == "1"
    merged: dict = {}
    for s, ms in zip(existing_sizes, existing_ms):
        merged[s] = ms
    for s, ms in zip(new_sizes, new_ms):
        if best_of and s in merged:
            merged[s] = min(merged[s], ms)
        else:
            merged[s] = ms

    # Sort by numeric size (handle "M=4096", "tokens=4096", or bare int).
    def _numkey(s):
        if isinstance(s, int):
            return s
        s = str(s)
        if "=" in s:
            try:
                return int(s.split("=", 1)[1])
            except ValueError:
                return 0
        try:
            return int(s)
        except ValueError:
            return 0
    sorted_sizes = sorted(merged.keys(), key=_numkey)
    sorted_ms = [merged[s] for s in sorted_sizes]

    data = dict(existing)  # preserve nccl_ms, _note, etc. on existing JSON
    data.update({
        "kernel": kernel,
        "sizes": sorted_sizes,
        "fused_ms": sorted_ms,
        "world_size": existing.get("world_size", 8),
        "num_nodes": existing.get("num_nodes", 2),
        "_note": note or existing.get("_note", f"release {kernel} bench (EFA backend)"),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)


# ----------------------------------------------------------------------
# CLI parser shared across all bench scripts
# ----------------------------------------------------------------------

def make_argparser(default_shapes: list[int], kernel_name: str):
    p = argparse.ArgumentParser(description=f"{kernel_name} release bench")
    p.add_argument("--mode", choices=["check", "bench"], default="bench")
    p.add_argument("--shapes", type=str, default=",".join(str(s) for s in default_shapes))
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None,
                   help="Path to reference JSON; non-zero exit on >5% regression.")
    return p


def parse_shapes(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]
