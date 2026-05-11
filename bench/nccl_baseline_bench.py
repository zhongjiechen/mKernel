#!/usr/bin/env python3
"""CuBLAS/NCCL baselines for the release kernels."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

BATCH = 4
HEADS = 16
D_HEAD = 128
HIDDEN = 7168
INTERMEDIATE = 2048
TOP_K = 8
_BARRIER_GROUP = None

DEFAULT_SHAPES = {
    "ag_gemm": [4096, 8192, 16384, 24576, 32768],
    "gemm_ar": [2048, 4096, 8192, 16384, 32768],
    "dispatch_gemm": [8192, 16384, 32768, 65536, 131072],
    "ring_attention": [768, 1536, 3072, 6144, 12288],
    "gemm_rs": [2048, 4096, 8192, 16384, 32768],
}

if int(os.environ.get("NUM_NODES", "2")) == 3:
    DEFAULT_SHAPES["ag_gemm"] = [6144, 12288, 24576, 36864, 49152]
    DEFAULT_SHAPES["gemm_ar"] = [3072, 6144, 12288, 24576, 49152]
    DEFAULT_SHAPES["dispatch_gemm"] = [12288, 24576, 49152, 98304, 196608]
    DEFAULT_SHAPES["gemm_rs"] = [3072, 6144, 12288, 24576, 49152]

if int(os.environ.get("NUM_NODES", "2")) == 4:
    DEFAULT_SHAPES["ag_gemm"] = [8192, 16384, 32768, 49152]
    DEFAULT_SHAPES["gemm_ar"] = [8192, 12288, 16384, 20480, 22528]
    DEFAULT_SHAPES["gemm_rs"] = [4096, 8192, 16384, 32768, 65536]


def parse_args():
    parser = argparse.ArgumentParser(description="CuBLAS/NCCL baseline bench")
    parser.add_argument("kernel", choices=[*DEFAULT_SHAPES.keys(), "all"])
    parser.add_argument("--shapes", default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--save-dir", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--tag", default="nccl")
    return parser.parse_args()


def init_dist():
    global _BARRIER_GROUP
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if os.environ.get("BASELINE_INIT_DEVICE_ID", "0") == "1":
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    else:
        dist.init_process_group("nccl")
    if os.environ.get("BASELINE_BARRIER_BACKEND", "gloo") == "gloo":
        _BARRIER_GROUP = dist.new_group(backend="gloo")
    return dist.get_rank(), local_rank, dist.get_world_size()


def sync_barrier():
    if _BARRIER_GROUP is None:
        dist.barrier()
    else:
        dist.barrier(group=_BARRIER_GROUP)


def max_across_ranks(value):
    t = torch.tensor([float(value)], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def timed_cuda(run_once, warmup, iters):
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    sync_barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once()
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / max(iters, 1)
    sync_barrier()
    return max_across_ranks(avg_ms)


def randn(shape, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16)


def bench_ag_gemm(shape, rank, world, warmup, iters):
    m = shape
    k = m
    n = m // world
    m_local = m // world
    a_local = randn((m_local, k), 42 + rank) / (k ** 0.25)
    b = randn((k, n), 100) / (k ** 0.25)
    a_full = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)

    def run_once():
        dist.all_gather_into_tensor(a_full, a_local)
        torch.matmul(a_full, b)

    return timed_cuda(run_once, warmup, iters)


def bench_gemm_ar(shape, rank, world, warmup, iters):
    m = shape
    k = m // world
    a = randn((m, k), 42 + rank) / (k ** 0.25)
    b = randn((k, m), 100 + rank) / (k ** 0.25)

    def run_once():
        c = torch.matmul(a, b)
        dist.all_reduce(c)

    return timed_cuda(run_once, warmup, iters)


def bench_gemm_rs(shape, rank, world, warmup, iters):
    m = shape
    k = m // world
    a = randn((m, k), 42 + rank) / (k ** 0.25)
    b = randn((k, m), 100 + rank) / (k ** 0.25)
    out = torch.empty((m // world, m), device="cuda", dtype=torch.bfloat16)

    def run_once():
        c = torch.matmul(a, b).contiguous()
        dist.reduce_scatter_tensor(out, c)

    return timed_cuda(run_once, warmup, iters)


def bench_dispatch_gemm(tokens, rank, world, warmup, iters):
    assert tokens % world == 0
    local_tokens = tokens // world
    rows = local_tokens * TOP_K
    send = randn((rows, HIDDEN), 42 + rank) / (HIDDEN ** 0.25)
    base, rem = divmod(rows, world)
    input_splits = [base + (peer < rem) for peer in range(world)]
    output_splits = [input_splits[rank]] * world
    recv = torch.empty((sum(output_splits), HIDDEN), device="cuda", dtype=torch.bfloat16)
    weight = randn((HIDDEN, INTERMEDIATE), 100 + rank) / (HIDDEN ** 0.25)

    def run_once():
        dist.all_to_all_single(
            recv, send,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
        torch.matmul(recv, weight)

    return timed_cuda(run_once, warmup, iters)


def bench_ring_attention(seq_per_dev, rank, world, warmup, iters):
    scale = D_HEAD ** -0.25
    q = randn((BATCH, HEADS, seq_per_dev, D_HEAD), 42 + rank) * scale
    k_local = randn((BATCH, HEADS, seq_per_dev, D_HEAD), 100 + rank) * scale
    v_local = randn((BATCH, HEADS, seq_per_dev, D_HEAD), 200 + rank) * scale
    impl = os.environ.get("BASELINE_RING_ATTN_IMPL", "flash_all_gather")

    if impl in ("all_gather", "torch_all_gather", "flash_all_gather"):
        # Best-light PyTorch+NCCL baseline: use optimized NCCL all-gather for
        # K/V, then one fused attention call over the full global KV. The gathered
        # buffers are sequence-major so NCCL concatenates ranks directly into
        # total_seq order.
        k_seq = k_local.permute(2, 0, 1, 3).contiguous()
        v_seq = v_local.permute(2, 0, 1, 3).contiguous()
        k_full_seq = torch.empty(
            (world * seq_per_dev, BATCH, HEADS, D_HEAD),
            device="cuda", dtype=torch.bfloat16)
        v_full_seq = torch.empty_like(k_full_seq)

        use_flash = impl == "flash_all_gather"
        if use_flash and flash_attn_func is None:
            use_flash = False

        if use_flash:
            q_flash = q.permute(0, 2, 1, 3).contiguous()

            def run_once():
                dist.all_gather_into_tensor(k_full_seq, k_seq)
                dist.all_gather_into_tensor(v_full_seq, v_seq)
                k_flash = k_full_seq.permute(1, 0, 2, 3).contiguous()
                v_flash = v_full_seq.permute(1, 0, 2, 3).contiguous()
                flash_attn_func(q_flash, k_flash, v_flash, causal=False)

            return timed_cuda(run_once, warmup, iters)

        k_full = k_full_seq.permute(1, 2, 0, 3)
        v_full = v_full_seq.permute(1, 2, 0, 3)

        def run_once():
            dist.all_gather_into_tensor(k_full_seq, k_seq)
            dist.all_gather_into_tensor(v_full_seq, v_seq)
            F.scaled_dot_product_attention(q, k_full, v_full)

        return timed_cuda(run_once, warmup, iters)

    if impl != "p2p_ring":
        raise ValueError(f"unknown BASELINE_RING_ATTN_IMPL={impl!r}")

    k_recv = torch.empty_like(k_local)
    v_recv = torch.empty_like(v_local)
    prev_rank = (rank - 1 + world) % world
    next_rank = (rank + 1) % world

    def run_once():
        k_cur = k_local
        v_cur = v_local
        recv_k = k_recv
        recv_v = v_recv
        for step in range(world):
            F.scaled_dot_product_attention(q, k_cur, v_cur)
            if step + 1 == world:
                break
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, k_cur, next_rank),
                dist.P2POp(dist.irecv, recv_k, prev_rank),
                dist.P2POp(dist.isend, v_cur, next_rank),
                dist.P2POp(dist.irecv, recv_v, prev_rank),
            ])
            for req in reqs:
                req.wait()
            k_cur, recv_k = recv_k, k_cur
            v_cur, recv_v = recv_v, v_cur

    ring_iters = min(iters, 3 if seq_per_dev >= 6144 else iters)
    return timed_cuda(run_once, warmup, ring_iters)


BENCH = {
    "ag_gemm": bench_ag_gemm,
    "gemm_ar": bench_gemm_ar,
    "dispatch_gemm": bench_dispatch_gemm,
    "ring_attention": bench_ring_attention,
    "gemm_rs": bench_gemm_rs,
}


def shape_label(kernel, shape):
    if kernel in {"ag_gemm", "gemm_ar", "gemm_rs"}:
        return f"M={shape}"
    if kernel == "dispatch_gemm":
        return f"tokens={shape}"
    return shape


def baseline_config(args):
    net = os.environ.get("BASELINE_NET", "unknown")
    smoke = net == "socket" and args.warmup == 0 and args.iters == 1
    return {
        "warmup": args.warmup,
        "iters": args.iters,
        "baseline_net": net,
        "baseline_quality": "socket_smoke" if smoke else f"{net}_benchmark",
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
        "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
        "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
        "nccl_ib_gid_index": os.environ.get("NCCL_IB_GID_INDEX"),
        "gloo_socket_ifname": os.environ.get("GLOO_SOCKET_IFNAME"),
        "ring_attention_impl": os.environ.get("BASELINE_RING_ATTN_IMPL", "all_gather"),
    }


def write_json(path, kernel, sizes, ms, world, args):
    merged = {}
    run_configs = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            for size, value in zip(existing.get("sizes", []), existing.get("nccl_ms", [])):
                merged[size] = value
            run_configs.update(existing.get("shape_run_configs", {}))
        except Exception:
            merged = {}
            run_configs = {}
    config = baseline_config(args)
    for size, value in zip(sizes, ms):
        merged[size] = value
        run_configs[size] = config

    data = {
        "kernel": kernel,
        "sizes": list(merged.keys()),
        "nccl_ms": list(merged.values()),
        "world_size": world,
        "num_nodes": int(os.environ.get("NUM_NODES", "2")),
        "run_config": config,
        "shape_run_configs": run_configs,
        "_note": f"CuBLAS/NCCL baseline bench (world={world})",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n")


def main():
    args = parse_args()
    rank, _local_rank, world = init_dist()
    kernels = list(DEFAULT_SHAPES) if args.kernel == "all" else [args.kernel]
    if args.shapes and len(kernels) != 1:
        raise ValueError("--shapes can only be used with one kernel")

    for kernel in kernels:
        shapes = [int(x) for x in args.shapes.split(",")] if args.shapes else DEFAULT_SHAPES[kernel]
        result_sizes = []
        result_ms = []
        if rank == 0:
            print(f"[nccl] {kernel} shapes={shapes} world={world}", flush=True)
        for shape in shapes:
            sync_barrier()
            try:
                ms = BENCH[kernel](shape, rank, world, args.warmup, args.iters)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                ms = float("nan")
            if rank == 0:
                print(f"[nccl] {kernel} {shape}: {ms:.3f} ms", flush=True)
            result_sizes.append(shape_label(kernel, shape))
            result_ms.append(ms)
            torch.cuda.empty_cache()

        if rank == 0:
            out = Path(args.save_dir) / f"{kernel}_{args.tag}.json"
            write_json(out, kernel, result_sizes, result_ms, world, args)
            print(f"[nccl] wrote {out}", flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
