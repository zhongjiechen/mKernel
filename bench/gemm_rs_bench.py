"""gemm_rs GEMM + Reduce-Scatter bench (release version).

Reproduces fused_q5_efa.json (5 shapes M∈{2048,4096,8192,16384,32768}, K=M/16).
Bakes the on-by-default gemm_rs env vars into Python:
  GEMM_RS_FUSE_COMPUTE_INTRA=1, GEMM_RS_INTRA_RS_DIRECT_STAGING=1, GEMM_RS_SEND_READY_BITMAP=1.
Per-shape SM split tuned to match the chunk512 numbers (see SM_SPLIT below).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

# Keep the release default, but allow parity tests against the experiment
# harness, which leaves this unset.
os.environ.setdefault("OSGC_BIND_RETAINED_HANDLE", "1")

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import compare_named_results  # noqa: E402

KERNEL_NAME = "gemm_rs"
NUM_NODES = 2
ROW_BLOCK = 128
COL_BLOCK = 256

# Sweep matches the bar chart (5 shapes; M=N=base_n, K=base_n/world_size = base_n/16).
DEFAULT_SHAPES = [2048, 4096, 8192, 16384, 32768]
# Note: M=65536 is in published bar chart but requires GEMM_RS_SEND_READY_BITMAP+FUSE_COMPUTE_INTRA
# which hangs on this hardware. Excluded from default sweep.

# 3-role split with FUSE_COMPUTE_INTRA=1 (SEND_FIRST_READY, no BITMAP — BITMAP+FCI hangs on this HW).
SM_SPLIT = {
    2048:  (112, 0, 10, 10, 2),
    4096:  (116, 0, 8,  8,  4),
    8192:  (118, 0, 6, 8, 4),  # canonical (best historical: 0.7426 ms at 26556fc)
    16384: (120, 0, 4,  8,  4),
    32768: (120, 0, 4,  8,  8),
    # M=65536: per-shape tune from experiments-side pick_role_split + pick_chunk_tiles
    # (CX7 sweep SLURM 11587 / 11578-83). 124 compute + 2 send + 6 reduce + CT=32
    # was the canonical config used to produce fused_q5_efa.json[M=65536]=68.47ms.
    65536: (124, 0, 2,  6,  32),
}


def poll_tuning(m: int) -> tuple[int, int]:
    # Match the experiment/source-of-truth path: acquire polling only helped
    # at 4K, while a short reducer sleep avoided flag-line thrash elsewhere.
    return (1 if m == 4096 else 0, 100)


def median_then_max_cuda(samples):
    median = sorted(float(x) for x in samples)[len(samples) // 2]
    t = torch.tensor([median], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "bench"], default="bench")
    p.add_argument("--shapes", type=str,
                   default=",".join(str(s) for s in DEFAULT_SHAPES))
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=7)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None)
    p.add_argument("--node-idx", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    node_idx = args.node_idx if args.node_idx is not None else int(os.environ.get("NODE_IDX", "0"))
    is_chief = (local_rank == 0 and node_idx == 0)

    peer_ip = os.environ.get("PEER_IP")
    if not peer_ip:
        peer_ip = os.environ.get("NODE1_IP", "172.31.11.6") if node_idx == 0 \
                  else os.environ.get("NODE0_IP", "172.31.1.237")
    tcp_port = int(os.environ.get("TCP_PORT", "19790")) + local_rank

    mod = load_module.load(KERNEL_NAME)

    if is_chief:
        print(f"[gemm_rs] world={world_size*NUM_NODES} per_node={world_size} "
              f"shapes={args.shapes}", flush=True)

    shapes = [int(x) for x in args.shapes.split(",") if x.strip()]
    global_gpu_idx = node_idx * world_size + local_rank

    result_sizes, result_fused = [], []

    for m in shapes:
        n = m
        k = m // (NUM_NODES * world_size)
        m_local = m // world_size  # each rank owns M/8 rows after intra-RS

        n_comp, n_intra, n_send, n_reduce, chunk_tiles = SM_SPLIT.get(
            m, (118, 0, 6, 8, 4))

        if is_chief:
            print(f"\n[gemm_rs] M={m} K={k} N={n} m_local={m_local} "
                  f"split=({n_comp},{n_intra},{n_send},{n_reduce}) ct={chunk_tiles}",
                  flush=True)
        print(f"[gemm_rs] node{node_idx}/lr{local_rank} alloc start", flush=True)

        torch.manual_seed(42 + global_gpu_idx)
        torch.cuda.manual_seed(42 + global_gpu_idx)
        A = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / (k ** 0.25)
        B = torch.randn((k, n), device="cuda", dtype=torch.bfloat16) / (k ** 0.25)

        workspace = mod.DistBuffer(
            (m, n), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        workspace.data_.zero_()

        output = mod.DistBuffer(
            (m_local, n), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        output.data_.zero_()

        barrier = mod.DistBuffer(
            (2, 1024, 1024), dtype=torch.int,
            local_rank=local_rank, local_world_size=world_size, multicast=True,
        )
        barrier.data_.zero_()

        total_compute_tiles = (m // ROW_BLOCK) * (n // COL_BLOCK)
        ready = torch.zeros(total_compute_tiles, device="cuda", dtype=torch.int32)

        def ready_entries(m_):
            ct_ = SM_SPLIT.get(m_, (0,0,0,0,4))[4]
            return (m_ // ROW_BLOCK) * (m_ // COL_BLOCK // max(ct_, 1))
        ready_max = max(ready_entries(m_) for m_ in shapes)
        ready_cap = 4096
        while ready_cap < ready_max * 2:
            ready_cap *= 2
        ready_chunk = mod.DistBuffer(
            (ready_cap,), dtype=torch.int32,
            local_rank=local_rank, local_world_size=world_size, multicast=True,
        )
        ready_chunk.data_.zero_()

        # GEMM_RS_INTRA_RS_DIRECT_STAGING=1: staging is a DistBuffer.
        staging_pgl = mod.DistBuffer(
            (m_local, n), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        staging_pgl.data_.zero_()
        staging_buf = staging_pgl.data_

        os.environ["GEMM_RS_RDMA_CHUNK_TILES_RT"] = str(chunk_tiles)

        staging_bytes = m_local * n * 2  # bf16
        recv_bytes = staging_bytes
        total_inter_tiles = (m_local // ROW_BLOCK) * (n // COL_BLOCK)
        fifo_cap = 2048
        while fifo_cap < total_inter_tiles * 2:
            fifo_cap *= 2

        dist.barrier()
        print(f"[gemm_rs] node{node_idx}/lr{local_rank} pre create_session peer={peer_ip}:{tcp_port}", flush=True)
        mod.create_session(
            node_idx, peer_ip, tcp_port,
            staging_buf.data_ptr(), staging_bytes,
            recv_bytes, total_inter_tiles, fifo_cap, local_rank,
        )
        print(f"[gemm_rs] node{node_idx}/lr{local_rank} post create_session", flush=True)
        fifo = mod.get_fifo_handles()
        arrival_ptr = mod.get_arrival_flags_ptr()
        recv_ptr = mod.get_recv_buf_ptr()

        epoch = 1
        mod.set_epoch(epoch)
        dist.barrier(); time.sleep(0.5)

        use_acquire_poll, reduce_poll_sleep_ns = poll_tuning(m)

        def reset_state():
            workspace.data_.zero_(); output.data_.zero_(); ready.zero_()
            barrier.data_.zero_(); ready_chunk.data_.zero_()
            staging_pgl.data_.zero_()

        def run_once():
            mod.gemm_rs_fused(
                A, B, workspace, output, barrier, ready,
                recv_ptr, staging_buf.data_ptr(),
                fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                arrival_ptr, epoch, node_idx,
                n_comp, n_intra, n_send, n_reduce,
                use_acquire_poll, reduce_poll_sleep_ns,
                ready_chunk,
                staging_pgl,
            )

        for wi in range(args.warmup):
            reset_state(); epoch += 1; mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.1)
            print(f"[gemm_rs] node{node_idx}/lr{local_rank} warmup{wi} pre-launch", flush=True)
            run_once(); torch.cuda.synchronize()
            print(f"[gemm_rs] node{node_idx}/lr{local_rank} warmup{wi} done", flush=True)
            dist.barrier()

        samples = []
        for _ in range(args.iters):
            reset_state(); epoch += 1; mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.05)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); run_once(); e.record(); torch.cuda.synchronize()
            samples.append(s.elapsed_time(e))
            dist.barrier()

        wall_ms = median_then_max_cuda(samples)
        if is_chief:
            print(f"[gemm_rs] M={m} wall={wall_ms:.3f} ms", flush=True)
        result_sizes.append(f"M={m}")
        result_fused.append(wall_ms)

    if is_chief and args.save_json:
        # Merge with existing JSON so a single-shape bench doesn't erase others.
        from common import write_results_json
        write_results_json(Path(args.save_json), "gemm_rs",
                           result_sizes, result_fused,
                           note=f"release gemm_rs bench (world={world_size*NUM_NODES})")
        print(f"[gemm_rs] wrote {args.save_json}", flush=True)

    if is_chief and args.compare_to:
        ok = compare_named_results("gemm_rs", result_sizes, result_fused, args.compare_to)
        dist.destroy_process_group()
        if not ok: return 1
        return 0

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
