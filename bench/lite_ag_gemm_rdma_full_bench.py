#!/usr/bin/env python3
"""Basic TP inference AllGather(A)+GEMM benchmark for L4.

Semantics:
    A_full = all_gather(A_local)
    C_i    = A_full @ B_i

Each rank owns a different column-parallel B_i shard.  The fused path does not
use NCCL or cuBLAS: it uses mKernel's GPU FIFO -> CPU RDMA proxy to exchange the
paired remote A shard, CUDA IPC/P2P copies to materialize A_full within the
node, and a hand-written TF32 WMMA GEMM for A_full @ B_i.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lite import load_ag_gemm_rdma_extension  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Basic TP no-NCCL/no-cuBLAS AG+GEMM benchmark")
    p.add_argument("--mode", choices=["check", "bench"], default="bench")
    p.add_argument("--shapes", default="512,1024,4096,8192")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=7)
    p.add_argument("--chunk-rows", type=int, default=64)
    p.add_argument("--tcp-port", type=int, default=30000)
    p.add_argument("--fifo-capacity", type=int, default=4096)
    p.add_argument("--num-qps", type=int, default=1)
    p.add_argument("--num-proxy-threads", type=int, default=1)
    p.add_argument("--max-inflight", type=int, default=512)
    p.add_argument("--save-json", default=None)
    p.add_argument("--check-correctness", action="store_true")
    p.add_argument("--check-atol", type=float, default=1e-2)
    p.add_argument("--check-rtol", type=float, default=1e-2)
    p.add_argument("--fast-epoch", action="store_true")
    p.add_argument("--constant-epoch", action="store_true")
    return p.parse_args()


def init_dist() -> tuple[int, int, int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world = dist.get_world_size()
    node_rank = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", rank // local_world)))
    return rank, local_rank, local_world, node_rank, world


def peer_ip(node_rank: int) -> str:
    return os.environ.get(
        "NODE1_IP" if node_rank == 0 else "NODE0_IP",
        "10.10.55.2" if node_rank == 0 else "10.10.55.1",
    )


def randn(shape: tuple[int, int], seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return torch.randn(shape, device="cuda", dtype=torch.float32) / (shape[1] ** 0.25)


def timed_cuda(run_once, prepare_once, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        prepare_once()
        dist.barrier()
        run_once()
        torch.cuda.synchronize()
    dist.barrier()

    samples: list[float] = []
    for _ in range(iters):
        prepare_once()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_once()
        end.record()
        torch.cuda.synchronize()
        dist.barrier()
        samples.append(start.elapsed_time(end))

    samples.sort()
    local_median = samples[len(samples) // 2]
    t = torch.tensor([float(local_median)], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def tflops_per_gpu(m: int, k: int, n: int, ms: float) -> float:
    return (2.0 * float(m) * float(k) * float(n)) / (ms * 1.0e-3) / 1.0e12


def check_close(name: str, observed: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> bool:
    diff = (observed - expected).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = float((diff / expected.abs().clamp_min(1e-6)).max().item()) if diff.numel() else 0.0
    tolerance = atol + rtol * expected.abs()
    violations = torch.count_nonzero(diff > tolerance).to(torch.int64)
    nonfinite = torch.count_nonzero(~torch.isfinite(observed) | ~torch.isfinite(expected)).to(torch.int64)
    stats = torch.tensor([max_abs, max_rel], dtype=torch.float64, device="cuda")
    counts = torch.stack([violations, nonfinite]).to(device="cuda")
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    ok = bool(counts[0].item() == 0 and counts[1].item() == 0)
    if dist.get_rank() == 0:
        print(
            f"[tp-ag-gemm-check] {name}: max_abs={stats[0].item():.6f} "
            f"max_rel={stats[1].item():.6f} violations={counts[0].item()} "
            f"nonfinite={counts[1].item()} {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    return ok


def main() -> int:
    args = parse_args()
    rank, local_rank, local_world, node_rank, world = init_dist()
    if world != 2 * local_world:
        raise ValueError(f"expected two nodes, got world={world} local_world={local_world}")
    if args.chunk_rows <= 0 or args.chunk_rows % 32 != 0:
        raise ValueError("--chunk-rows must be a positive multiple of 32")

    torch.backends.cuda.matmul.allow_tf32 = True
    ext = load_ag_gemm_rdma_extension(verbose=(rank == 0))
    shapes = [int(x) for x in args.shapes.split(",") if x.strip()]
    if rank == 0:
        print(f"[tp-ag-gemm] world={world} local_world={local_world} shapes={shapes}", flush=True)

    sizes: list[str] = []
    fused_ms_values: list[float] = []
    baseline_ms_values: list[float] = []
    fused_tflops_values: list[float] = []
    baseline_tflops_values: list[float] = []
    speedups: list[float] = []
    ok_all = True

    for shape_index, m in enumerate(shapes):
        if m % world != 0:
            raise ValueError(f"M={m} must be divisible by world={world}")
        k = m
        n = m // world
        local_m = m // world
        if local_m % 32 != 0 or n % 16 != 0 or k % 8 != 0:
            raise ValueError("shape is incompatible with the TF32 WMMA tile shape")

        chunk_rows = min(args.chunk_rows, local_m)
        num_chunks = (local_m + chunk_rows - 1) // chunk_rows

        a_local = randn((local_m, k), 42 + rank)
        # Rank-dependent B_i is required for column-parallel TP correctness.
        b_local = randn((k, n), 100 + rank)
        recv_pair = torch.empty_like(a_local)
        a_full_baseline = torch.empty((m, k), device="cuda", dtype=torch.float32)
        a_full_buf = ext.DistBuffer((m, k), torch.float32, local_rank, local_world, False)
        ready_flags = ext.DistBuffer((2 * local_world,), torch.int32, local_rank, local_world, False)
        out_baseline = torch.empty((m, n), device="cuda", dtype=torch.float32)
        out_fused = torch.empty((m, n), device="cuda", dtype=torch.float32)
        ready_flags.data_.zero_()
        torch.cuda.synchronize()
        ext.local_barrier(ready_flags)

        if rank == 0:
            print(
                f"\n[tp-ag-gemm] M={m} K={k} N={n} local_M={local_m} chunk_rows={chunk_rows}",
                flush=True,
            )

        ext.create_session(
            node_rank,
            peer_ip(node_rank),
            args.tcp_port + shape_index * local_world + local_rank,
            a_local,
            recv_pair,
            num_chunks,
            args.fifo_capacity,
            local_rank,
            args.num_qps,
            args.num_proxy_threads,
            args.max_inflight,
            "",
        )
        ext.reset_arrival_flags()
        torch.cuda.synchronize()
        dist.barrier()
        epoch = 1 + shape_index * 100000

        def prepare_fused() -> None:
            nonlocal epoch
            torch.cuda.synchronize()
            if args.constant_epoch:
                epoch = 1
                ext.reset_arrival_flags()
            else:
                epoch += 1
                if args.fast_epoch:
                    ext.fast_set_epoch(epoch)
                else:
                    ext.set_epoch(epoch)
            ready_flags.data_.zero_()
            torch.cuda.synchronize()

        def run_fused() -> None:
            ext.push_full_tf32_gemm(
                a_local,
                recv_pair,
                a_full_buf,
                ready_flags,
                b_local,
                out_fused,
                node_rank,
                local_rank,
                local_world,
                local_m,
                k,
                chunk_rows,
                num_chunks,
                epoch,
            )

        def prepare_baseline() -> None:
            return None

        def run_baseline() -> None:
            dist.all_gather_into_tensor(a_full_baseline, a_local)
            torch.mm(a_full_baseline, b_local, out=out_baseline)

        if args.mode == "check" or args.check_correctness:
            run_baseline()
            torch.cuda.synchronize()
            prepare_fused()
            dist.barrier()
            run_fused()
            torch.cuda.synchronize()
            ok_all = check_close(f"M={m}", out_fused, out_baseline, args.check_atol, args.check_rtol) and ok_all

        baseline_ms = timed_cuda(run_baseline, prepare_baseline, args.warmup, args.iters)
        fused_ms = timed_cuda(run_fused, prepare_fused, args.warmup, args.iters)
        speedup = baseline_ms / fused_ms
        fused_tflops = tflops_per_gpu(m, k, n, fused_ms)
        baseline_tflops = tflops_per_gpu(m, k, n, baseline_ms)
        if rank == 0:
            print(
                f"[tp-ag-gemm] M={m}: fused={fused_ms:.3f} ms "
                f"({fused_tflops:.2f} TFLOPS/GPU) "
                f"nccl+cublas={baseline_ms:.3f} ms "
                f"({baseline_tflops:.2f} TFLOPS/GPU) "
                f"speedup={speedup:.3f}x",
                flush=True,
            )

        sizes.append(f"M={m}")
        fused_ms_values.append(fused_ms)
        baseline_ms_values.append(baseline_ms)
        fused_tflops_values.append(fused_tflops)
        baseline_tflops_values.append(baseline_tflops)
        speedups.append(speedup)
        ext.destroy_session()
        del a_full_buf, ready_flags
        torch.cuda.empty_cache()
        time.sleep(0.2)

    if rank == 0 and args.save_json:
        payload = {
            "kernel": "lite_tp_ag_gemm_push_wmma_basic",
            "dtype": "float32_tf32",
            "baseline": "NCCL+cuBLAS",
            "sizes": sizes,
            "fused_ms": fused_ms_values,
            "nccl_cublas_ms": baseline_ms_values,
            "fused_tflops_per_gpu": fused_tflops_values,
            "nccl_cublas_tflops_per_gpu": baseline_tflops_values,
            "speedup_vs_nccl_cublas": speedups,
            "chunk_rows": [args.chunk_rows for _ in sizes],
            "world_size": world,
            "local_world_size": local_world,
            "note": "TP-correct basic AG+GEMM: rank-local B_i, no NCCL/cuBLAS fused path, RDMA pair exchange + IPC A_full staging + WMMA GEMM.",
        }
        Path(args.save_json).write_text(json.dumps(payload, indent=4) + "\n")
        print(f"[tp-ag-gemm] wrote {args.save_json}", flush=True)

    dist.destroy_process_group()
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
