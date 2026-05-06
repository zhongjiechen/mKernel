"""dispatch_gemm MoE Dispatch + Group GEMM bench (release version).

Runs the prebuilt release/build/libdispatch_gemm.so (no JIT) with the default
dispatch_gemm configuration:

    DISPATCH_LOCAL_FIRST=1, DISPATCH_ZERO_COPY=1, DISPATCH_DISPATCH_PIPELINE=1, fused exec mode.
    CHUNK_BYTES=512KB (baked in src/dispatch_gemm.cu).
    SM split: send=4, copy=4, comm=64 at large shapes (131k).

Reproduction target (from experiments/multinode/sweep/results/efa/fused_q3_efa_chunk512.json):
    8k: 0.631 ms, 16k: 1.057 ms, 32k: 1.538 ms, 65k: 2.669 ms, 131k: 5.16 ms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Required for multicast bind on this hardware/setup. Must be set BEFORE
# importing the prebuilt module since the bind logic reads getenv at C++ time.
os.environ["MKERNEL_BIND_RETAINED_HANDLE"] = "1"

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import compare_named_results  # noqa: E402

KERNEL_NAME = "dispatch_gemm"

# Constants matching the build (-DTK_MOE_*).
H = 7168
I = 2048
NUM_EXPERTS = 256
TOP_K = 8
from common import get_num_nodes  # noqa: E402
NUM_NODES = get_num_nodes()
ROW_BLOCK = 128
CHUNK_BYTES = 512 * 1024  # baked in source

# Default sweep matches the bar chart x-axis.
DEFAULT_SHAPES = [8192, 16384, 32768, 65536, 131072]


def avg_then_max_cuda(samples):
    avg = sum(float(x) for x in samples) / len(samples)
    t = torch.tensor([avg], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "bench"], default="bench")
    p.add_argument("--shapes", type=str,
                   default=",".join(str(s) for s in DEFAULT_SHAPES))
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=7)
    # SM split (per-shape tuned per INSTRUCTION.md §2).
    p.add_argument("--num-comm-sms", type=int, default=None)
    p.add_argument("--num-send-sms", type=int, default=None)
    p.add_argument("--num-copy-sms", type=int, default=None)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None)
    p.add_argument("--node-idx", type=int, default=None)
    return p.parse_args()


def build_routing(num_tokens_global, world_size_per_node):
    """Build deterministic uniform routing across all 16 GPUs."""
    total_gpus = NUM_NODES * world_size_per_node
    num_local_tokens = num_tokens_global // total_gpus
    assert num_tokens_global % total_gpus == 0
    expert_to_tokens = [[] for _ in range(NUM_EXPERTS)]
    for t in range(num_tokens_global):
        for k in range(TOP_K):
            expert_id = (t * TOP_K + k) % NUM_EXPERTS
            global_gpu = t // num_local_tokens
            src_node = global_gpu // world_size_per_node
            src_dev = global_gpu % world_size_per_node
            local_tok = t % num_local_tokens
            expert_to_tokens[expert_id].append((src_node, src_dev, local_tok))
    padded_list = []
    for e in range(NUM_EXPERTS):
        actual = len(expert_to_tokens[e])
        padded = ((actual + ROW_BLOCK - 1) // ROW_BLOCK) * ROW_BLOCK
        padded_list.append(padded)
    return padded_list, expert_to_tokens


def build_pull_indices(global_gpu_idx, num_experts_per_dev, padded_list,
                       expert_to_tokens, world_size_per_node):
    """LOCAL_FIRST=1: emit local-source tokens before peer-source within each expert."""
    expert_start = global_gpu_idx * num_experts_per_dev
    expert_end = expert_start + num_experts_per_dev
    this_node = global_gpu_idx // world_size_per_node
    rows = []
    for e in range(expert_start, expert_end):
        padded = padded_list[e]
        toks = expert_to_tokens[e]
        locals_ = [t for t in toks if t[0] == this_node]
        peers_ = [t for t in toks if t[0] != this_node]
        ordered = locals_ + peers_
        for i in range(padded):
            if i < len(ordered):
                rows.append(ordered[i])
            else:
                rows.append((-1, -1, -1))
    arr = torch.tensor(rows, dtype=torch.int32, device="cuda")
    return arr, arr.shape[0]


def per_shape_sm_split(num_local_tokens):
    """Return (n_send, n_copy, n_comm) tuned per local-token count.

    Per INSTRUCTION.md §2:
      ≤1024 (small global): send=8 copy=8 comm=44
      ≥2048 (large global): send=4 copy=4 comm=64
    """
    # Iter-4 tweak: M=16384 (1024 local) and Iter-5 attempt: M=8192 (512 local)
    # also into the compute-heavy bucket. Freed 8 SMs go to GEMM.
    if num_local_tokens < 512:
        return 8, 8, 44
    return 4, 4, 64


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    node_idx = args.node_idx if args.node_idx is not None else int(os.environ.get("NODE_IDX", "0"))
    is_chief = (local_rank == 0 and node_idx == 0)

    # Peer IP / TCP port for session bootstrap (matches experiment harness).
    peer_ip = os.environ.get("PEER_IP")
    if not peer_ip:
        # Default for the 2-node setup: each node connects to the OTHER's private IP.
        peer_ip = os.environ.get("NODE1_IP", "172.31.11.6") if node_idx == 0 \
                  else os.environ.get("NODE0_IP", "172.31.1.237")
    tcp_port = int(os.environ.get("TCP_PORT", "19790")) + local_rank

    mod = load_module.load(KERNEL_NAME)
    print(f"[dispatch_gemm] node{node_idx}/lr{local_rank} loaded mod, peer_ip={peer_ip} "
          f"tcp_port={tcp_port} MKERNEL_BIND={os.environ.get('MKERNEL_BIND_RETAINED_HANDLE','-')}", flush=True)

    if is_chief:
        print(f"[dispatch_gemm] world={world_size*NUM_NODES} per_node={world_size} "
              f"shapes={args.shapes}", flush=True)

    shapes = [int(x) for x in args.shapes.split(",") if x.strip()]
    total_gpus = NUM_NODES * world_size
    num_experts_per_dev = NUM_EXPERTS // total_gpus
    global_gpu_idx = node_idx * world_size + local_rank

    result_sizes = []
    result_fused = []

    for num_tokens_global in shapes:
        num_local_tokens = num_tokens_global // total_gpus
        assert num_tokens_global % total_gpus == 0

        n_send, n_copy, n_comm = per_shape_sm_split(num_local_tokens)
        # CLI override (lets the launcher tune without rebuilding).
        if args.num_send_sms is not None: n_send = args.num_send_sms
        if args.num_copy_sms is not None: n_copy = args.num_copy_sms
        if args.num_comm_sms is not None: n_comm = args.num_comm_sms

        if is_chief:
            print(f"\n[dispatch_gemm] tokens={num_tokens_global} "
                  f"local_tokens={num_local_tokens} "
                  f"sm(send,copy,comm)=({n_send},{n_copy},{n_comm})", flush=True)

        padded_list, expert_to_tokens = build_routing(num_tokens_global, world_size)
        padded_ppe = torch.tensor(padded_list, dtype=torch.int32, device="cuda")
        pull_idx, num_padded_local = build_pull_indices(
            global_gpu_idx, num_experts_per_dev, padded_list,
            expert_to_tokens, world_size)

        # Per-expert pure-local row_block count (LOCAL_FIRST baked on).
        this_node = global_gpu_idx // world_size
        expert_start = global_gpu_idx * num_experts_per_dev
        local_rb_list = []
        for e in range(expert_start, expert_start + num_experts_per_dev):
            local_count = sum(1 for t in expert_to_tokens[e] if t[0] == this_node)
            local_rb_list.append(local_count // ROW_BLOCK)
        local_rb_per_expert = torch.tensor(local_rb_list, dtype=torch.int32, device="cuda")

        # Per-rank deterministic data.
        torch.manual_seed(42 + global_gpu_idx)
        torch.cuda.manual_seed(42 + global_gpu_idx)
        pre_tokens_data = (
            torch.randn((num_local_tokens, H), device="cuda", dtype=torch.bfloat16)
            / (H ** 0.25)
        )
        torch.manual_seed(100 + global_gpu_idx)
        torch.cuda.manual_seed(100 + global_gpu_idx)
        weights = (
            torch.randn((num_experts_per_dev, H, I), device="cuda", dtype=torch.bfloat16)
            / (H ** 0.25)
        )

        pre_tokens = mod.DistBuffer(
            (num_local_tokens, H), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        pre_tokens.data_.copy_(pre_tokens_data)

        peer_tokens = mod.DistBuffer(
            (num_local_tokens, H), dtype=torch.bfloat16,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        peer_tokens.data_.zero_()

        num_row_blocks = max(1, (num_padded_local + ROW_BLOCK - 1) // ROW_BLOCK)
        barrier = mod.DistBuffer(
            (world_size, num_row_blocks, 1), dtype=torch.int32,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        barrier.data_.zero_()

        sync_barrier = mod.DistBuffer(
            (1, 1, 2), dtype=torch.int32,
            local_rank=local_rank, local_world_size=world_size, multicast=True,
        )
        sync_barrier.data_.zero_()

        post_tokens = torch.zeros((num_padded_local, H),
                                   device="cuda", dtype=torch.bfloat16)
        outputs = torch.zeros((num_padded_local, I),
                               device="cuda", dtype=torch.bfloat16)

        pre_tokens_bytes = num_local_tokens * H * 2
        total_chunks = (pre_tokens_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES
        copy_ready = mod.DistBuffer(
            (world_size, total_chunks, 1), dtype=torch.int32,
            local_rank=local_rank, local_world_size=world_size, multicast=False,
        )
        copy_ready.data_.zero_()
        send_buf = torch.empty((num_local_tokens, H),
                                device="cuda", dtype=torch.bfloat16)

        dist.barrier()
        fifo_cap = 2048
        while fifo_cap < total_chunks * 2:
            fifo_cap *= 2

        # ZERO_COPY baked on: peer_tokens IS the RDMA destination.
        external_recv_buf_ptr = int(peer_tokens.data_.data_ptr())
        print(f"[dispatch_gemm] node{node_idx}/lr{local_rank} alloc done; "
              f"creating session peer={peer_ip}:{tcp_port}", flush=True)
        # Zero-copy send: register pre_tokens as the proxy's data MR. Kernel
        # reads straight from pre_tokens — no send_buf pack required.
        mod.create_session(
            node_idx, peer_ip, tcp_port,
            send_buf.data_ptr(), pre_tokens_bytes,
            pre_tokens_bytes, total_chunks, fifo_cap, local_rank,
            external_recv_buf_ptr,
            int(pre_tokens.data_.data_ptr()),
            pre_tokens_bytes,
        )
        print(f"[dispatch_gemm] node{node_idx}/lr{local_rank} session created", flush=True)
        fifo = mod.get_fifo_handles()
        arrival_ptr = mod.get_arrival_flags_ptr()
        recv_ptr = mod.get_recv_buf_ptr()

        epoch = 1
        mod.set_epoch(epoch)
        dist.barrier(); time.sleep(0.5)

        # send_buf no longer used by the proxy (zero-copy reads pre_tokens
        # directly via DMA-BUF MR), but the allocation is kept for the
        # session-config arg until the signature is cleaned up.

        def run_once():
            mod.moe_dispatch_gemm_fused(
                pre_tokens, peer_tokens, copy_ready,
                post_tokens, weights, outputs, padded_ppe, pull_idx,
                local_rb_per_expert, barrier, sync_barrier,
                recv_ptr,
                fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                arrival_ptr, epoch,
                node_idx, num_local_tokens, num_padded_local,
                n_send, n_copy, n_comm,
            )

        def reset_state():
            barrier.data_.zero_()
            sync_barrier.data_.zero_()
            copy_ready.data_.zero_()

        # Prime: first epoch is a known stale-state warmup on EFA dispatch_gemm.
        reset_state(); dist.barrier(); time.sleep(0.05)
        run_once(); torch.cuda.synchronize(); dist.barrier()

        epoch += 1; mod.set_epoch(epoch); reset_state()
        dist.barrier(); time.sleep(0.05)

        # Warmup
        for _ in range(args.warmup):
            reset_state(); epoch += 1; mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.05)
            run_once(); torch.cuda.synchronize()
            dist.barrier()

        # Timed iters
        samples = []
        for _ in range(args.iters):
            reset_state(); epoch += 1; mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.05)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); run_once(); e.record(); torch.cuda.synchronize()
            dist.barrier()
            samples.append(s.elapsed_time(e))

        wall_ms = avg_then_max_cuda(samples)
        if is_chief:
            print(f"[dispatch_gemm] tokens={num_tokens_global} wall={wall_ms:.3f} ms", flush=True)
        result_sizes.append(f"tokens={num_tokens_global}")
        result_fused.append(wall_ms)
        # Don't call destroy_session — re-creating per shape is fine and
        # destroy_session has caused state issues. Process exit cleans up.

    if is_chief and args.save_json:
        # Merge with existing JSON so a single-shape bench doesn't erase others.
        from common import write_results_json
        write_results_json(Path(args.save_json), "dispatch_gemm",
                           result_sizes, result_fused,
                           note=f"release dispatch_gemm bench (world={world_size*NUM_NODES})")
        print(f"[dispatch_gemm] wrote {args.save_json}", flush=True)

    if is_chief and args.compare_to:
        ok = compare_named_results("dispatch_gemm", result_sizes, result_fused,
                                   args.compare_to)
        dist.destroy_process_group()
        if not ok:
            return 1
        return 0

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
