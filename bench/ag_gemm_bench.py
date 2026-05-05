"""ag_gemm All-Gather + GEMM bench (release version).

Reproduces fused_q1_efa_fresh.json (5 shapes M∈{4096,8192,16384,24576,32768}).
SM split: --num-comm-sms 64 (50/50 split intra/inter inside the kernel).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ["OSGC_BIND_RETAINED_HANDLE"] = "1"

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import compare_named_results, get_peer_ips, get_peer_ports  # noqa: E402

KERNEL_NAME = "ag_gemm"
from common import get_num_nodes  # noqa: E402
NUM_NODES = get_num_nodes()
ROW_BLOCK = 128
COL_BLOCK = 256
RED_BLOCK = 64
CHUNK_BYTES = 64 * 1024  # baked from AG_CHUNK_BYTES=65536

DEFAULT_SHAPES = [4096, 8192, 16384, 24576, 32768]

# Per-shape num_comm_sms override. Smaller values reduce coordination overhead
# at small M (NCCL has minimal launch overhead and beats the fused path there
# unless we cut the comm-CTA budget). Defaults found via sweep on this cluster.
SMS_PER_SHAPE = {4096: 8, 8192: 8, 24576: 8}


def avg_then_max_cuda(samples):
    # Median-then-max for robustness against outlier iters (matches gemm_rs).
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
    p.add_argument("--num-comm-sms", type=int, default=64)
    p.add_argument("--num-intra-comm-sms", type=int, default=0)
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
        print(f"[ag_gemm] world={world_size*NUM_NODES} shapes={args.shapes}", flush=True)

    shapes = [int(x) for x in args.shapes.split(",") if x.strip()]
    global_world = NUM_NODES * world_size
    global_gpu_idx = node_idx * world_size + local_rank

    result_sizes, result_fused = [], []

    # Per-shape intra override that bypasses the max(4) floor in the kernel
    # by going through num_intra_comm_override path (line 940 of src/ag_gemm.cu).
    INTRA_OVERRIDE = {4096: 2}
    for base_n in shapes:
        # Per-shape num_comm_sms override (small-M overhead reduction).
        if base_n in SMS_PER_SHAPE:
            args.num_comm_sms = SMS_PER_SHAPE[base_n]
            if is_chief:
                print(f"[ag_gemm] M={base_n}: per-shape num_comm_sms={args.num_comm_sms}",
                      flush=True)
        if base_n in INTRA_OVERRIDE:
            args.num_intra_comm_sms = INTRA_OVERRIDE[base_n]
            if is_chief:
                print(f"[ag_gemm] M={base_n}: per-shape num_intra_comm_sms={args.num_intra_comm_sms}",
                      flush=True)
        else:
            args.num_intra_comm_sms = 0
        # ag_gemm TP-column-parallel: M=K=base_n, N=base_n/global_world.
        M, K, N = base_n, base_n, base_n // global_world
        M_half = M // 2
        M_local = M // (2 * world_size)
        assert M % ROW_BLOCK == 0 and K % RED_BLOCK == 0 and N % COL_BLOCK == 0

        if is_chief:
            print(f"\n[ag_gemm] M={M} K={K} N={N} M_half={M_half} M_local={M_local}", flush=True)
        print(f"[ag_gemm] node{node_idx}/lr{local_rank} start alloc", flush=True)

        torch.manual_seed(42 + global_gpu_idx); torch.cuda.manual_seed(42 + global_gpu_idx)
        A_local = torch.randn((M_local, K), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
        torch.manual_seed(100); torch.cuda.manual_seed(100)
        B = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) / (K ** 0.25)
        print(f"[ag_gemm] node{node_idx}/lr{local_rank} A,B done", flush=True)

        a_tk = mod.DistBuffer((M_half, K), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        print(f"[ag_gemm] node{node_idx}/lr{local_rank} a_tk done", flush=True)
        start_row = local_rank * M_local
        a_tk.data_[start_row:start_row + M_local].copy_(A_local)

        a_recv_tk = mod.DistBuffer((M_half, K), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        print(f"[ag_gemm] node{node_idx}/lr{local_rank} a_recv_tk done", flush=True)
        a_recv_tk.data_.zero_()

        barrier = mod.DistBuffer((3, 1024, 1024), dtype=torch.int,
            local_rank=local_rank, local_world_size=world_size, multicast=True)
        print(f"[ag_gemm] node{node_idx}/lr{local_rank} barrier done", flush=True)
        barrier.data_.zero_()

        C = torch.zeros((M, N), device="cuda", dtype=torch.bfloat16)

        a_half_bytes = M_half * K * 2
        total_chunks = (a_half_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES

        dist.barrier()
        fifo_cap = 2048
        while fifo_cap < total_chunks * 2: fifo_cap *= 2
        a_tk_ptr = int(a_tk.data_.data_ptr())
        print(f"[ag_gemm] node{node_idx}/lr{local_rank} pre create_session peer={peer_ip}:{tcp_port}", flush=True)
        # Both MR0 (local_gpu_buf, src_view=0) and MR1 (clocal_gpu_buf, src_view=1)
        # point at the same DMA-BUF-registered VMM tensor — kernel only reads
        # via src_view=1, so MR0 is just a structural placeholder.
        peer_ips = get_peer_ips(node_idx, NUM_NODES)
        mod.create_session(
            node_idx, peer_ip, tcp_port,
            a_tk_ptr, a_half_bytes, a_half_bytes,
            total_chunks, fifo_cap, local_rank,
            clocal_buf_ptr=a_tk_ptr, clocal_buf_size=a_half_bytes,
            peer_ips=peer_ips,
            peer_tcp_ports=get_peer_ports(node_idx, NUM_NODES, tcp_port),
        )
        print(f"[ag_gemm] node{node_idx}/lr{local_rank} post create_session", flush=True)
        fifo = mod.get_fifo_handles()
        arrival_ptr = mod.get_arrival_flags_ptr()
        recv_ptr = mod.get_recv_buf_ptr()

        epoch = 1
        mod.set_epoch(epoch)
        dist.barrier(); time.sleep(0.5)
        print(f"[ag_gemm] node{node_idx}/lr{local_rank} epoch=1 settled, starting warmup", flush=True)

        def reset_state():
            barrier.data_.zero_()
            a_tk.data_[start_row:start_row + M_local].copy_(A_local)
            C.zero_()

        def run_once():
            mod.ag_gemm_multinode(
                a_tk, B, C, barrier,
                recv_ptr,
                int(fifo[0]), int(fifo[1]), int(fifo[2]), int(fifo[3]), int(fifo[4]),
                arrival_ptr, epoch, node_idx, args.num_comm_sms, a_half_bytes,
                a_recv_tk, 132, args.num_intra_comm_sms, NUM_NODES,
            )

        for wi in range(args.warmup):
            reset_state(); epoch += 1; mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.1)
            print(f"[ag_gemm] node{node_idx}/lr{local_rank} warmup{wi} pre-launch epoch={epoch}", flush=True)
            run_once(); torch.cuda.synchronize()
            print(f"[ag_gemm] node{node_idx}/lr{local_rank} warmup{wi} done", flush=True)
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

        wall_ms = avg_then_max_cuda(samples)
        if is_chief:
            print(f"[ag_gemm] M={M} wall={wall_ms:.3f} ms", flush=True)
        result_sizes.append(f"M={M}")
        result_fused.append(wall_ms)

    if is_chief and args.save_json:
        # MERGE with existing JSON so a single-shape bench doesn't erase the
        # other shapes the chart needs.
        from common import write_results_json
        write_results_json(Path(args.save_json), "ag_gemm",
                           result_sizes, result_fused,
                           note=f"release ag_gemm bench (world={world_size*NUM_NODES})")
        print(f"[ag_gemm] wrote {args.save_json}", flush=True)

    if is_chief and args.compare_to:
        ok = compare_named_results("ag_gemm", result_sizes, result_fused, args.compare_to)
        dist.destroy_process_group()
        if not ok: return 1
        return 0
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
