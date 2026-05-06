"""ring_attention Ring Attention bench (release version).

Reproduces fused_q4_efa_per_stage.json (5 shapes seq_per_dev∈{768,1536,3072,6144,12288}).
Uses --per-stage build (RING_ATTN_PER_STAGE_KERNEL=1 baked in Makefile).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ["MKERNEL_BIND_RETAINED_HANDLE"] = "1"

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import compare_named_results, get_peer_ips, get_peer_ports  # noqa: E402

KERNEL_NAME = "ring_attention"
from common import get_num_nodes  # noqa: E402
NUM_NODES = get_num_nodes()
BATCH = 4
HEADS = 16
D = 128
QO_BLOCK = 64
KV_BLOCK = 128
CONSUMER_WARPGROUPS = 3
CHUNK_BYTES = 256 * 1024  # baked

DEFAULT_SHAPES = [768, 1536, 3072, 6144, 12288]


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
    p.add_argument("--num-comm-sms", type=int, default=16)
    p.add_argument("--num-send-sms", type=int, default=8)
    p.add_argument("--num-copy-sms", type=int, default=8)
    p.add_argument("--save-json", type=str, default=None)
    p.add_argument("--compare-to", type=str, default=None)
    p.add_argument("--node-idx", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))

    node_idx = args.node_idx if args.node_idx is not None else int(os.environ.get("NODE_IDX", "0"))
    is_chief = (local_rank == 0 and node_idx == 0)
    peer_ip = os.environ.get("PEER_IP")
    if not peer_ip:
        peer_ip = os.environ.get("NODE1_IP", "172.31.11.6") if node_idx == 0 \
                  else os.environ.get("NODE0_IP", "172.31.1.237")
    tcp_port = int(os.environ.get("TCP_PORT", "18560")) + local_rank

    mod = load_module.load(KERNEL_NAME)
    if is_chief:
        print(f"[ring_attn] world={world_size} per_node={local_world_size} "
              f"shapes={args.shapes}", flush=True)

    seqs = [int(x) for x in args.shapes.split(",") if x.strip()]
    global_gpu = node_idx * local_world_size + local_rank

    result_sizes, result_fused = [], []

    # Per-shape SM split (comm, send, copy). Larger seqs benefit from
    # smaller comm pool (more compute SMs); small seqs need comm coverage.
    SM_SPLIT = {
        1536:  (8, 12, 12),
        3072:  (12, 8, 8),
        6144:  (12, 8, 8),
        12288: (8, 12, 12),
    }
    for seq_per_dev in seqs:
        if seq_per_dev in SM_SPLIT:
            c, s, cp = SM_SPLIT[seq_per_dev]
            args.num_comm_sms = c; args.num_send_sms = s; args.num_copy_sms = cp
        assert seq_per_dev % (CONSUMER_WARPGROUPS * QO_BLOCK) == 0
        assert seq_per_dev % KV_BLOCK == 0

        if is_chief:
            print(f"\n[ring_attn] seq_per_dev={seq_per_dev}", flush=True)

        torch.manual_seed(42 + global_gpu); torch.cuda.manual_seed(42 + global_gpu)
        scale = D ** -0.25
        Q = torch.randn(BATCH, HEADS, seq_per_dev, D, device="cuda",
                        dtype=torch.bfloat16) * scale
        K_local = torch.randn(BATCH, HEADS, seq_per_dev, D, device="cuda",
                              dtype=torch.bfloat16) * scale
        V_local = torch.randn(BATCH, HEADS, seq_per_dev, D, device="cuda",
                              dtype=torch.bfloat16) * scale

        def dbuf(shape):
            return mod.DistBuffer(
                shape, dtype=torch.bfloat16,
                local_rank=local_rank, local_world_size=local_world_size,
                multicast=False)

        K0 = dbuf((BATCH, HEADS, seq_per_dev, D))
        K1 = dbuf((BATCH, HEADS, seq_per_dev, D))
        V0 = dbuf((BATCH, HEADS, seq_per_dev, D))
        V1 = dbuf((BATCH, HEADS, seq_per_dev, D))

        L = torch.zeros(BATCH, HEADS, seq_per_dev, device="cuda", dtype=torch.float32)
        L_block = torch.zeros_like(L)
        O = torch.zeros_like(Q)
        O_block = torch.zeros_like(Q)

        barrier = mod.DistBuffer(
            (2, 1024, 1024), dtype=torch.int32,
            local_rank=local_rank, local_world_size=local_world_size,
            multicast=True)

        k_bytes = K_local.numel() * 2
        v_bytes = V_local.numel() * 2
        kv_buf_bytes = k_bytes + v_bytes
        send_buf = torch.empty(kv_buf_bytes // 2, device="cuda", dtype=torch.bfloat16)
        send_buf_ptr = send_buf.data_ptr()
        total_chunks = (kv_buf_bytes + CHUNK_BYTES - 1) // CHUNK_BYTES

        # Per-peer sizing (1× at N == 2, identical to legacy).
        n_peers = NUM_NODES - 1
        recv_buf_chunks = n_peers * total_chunks
        recv_buf_bytes = n_peers * kv_buf_bytes

        dist.barrier()
        fifo_cap = 2048
        while fifo_cap < recv_buf_chunks * 2: fifo_cap *= 2

        # Zero-copy send: K0 and V0 are registered as DMA-BUF MRs. Proxy
        # posts single-SGE WRs straight from the VMM tensors — no pack.
        peer_ips = get_peer_ips(node_idx, NUM_NODES)
        mod.create_session(
            node_idx, peer_ip, tcp_port,
            send_buf_ptr, kv_buf_bytes, recv_buf_bytes,
            recv_buf_chunks, fifo_cap, local_rank,
            int(K0.data_.data_ptr()), k_bytes,
            int(V0.data_.data_ptr()), v_bytes,
            peer_ips=peer_ips,
            peer_tcp_ports=get_peer_ports(node_idx, NUM_NODES, tcp_port),
        )
        fifo = mod.get_fifo_handles()
        arrival_ptr = mod.get_arrival_flags_ptr()
        recv_ptr = mod.get_recv_buf_ptr()

        K0.data_.copy_(K_local); V0.data_.copy_(V_local)
        K1.data_.zero_(); V1.data_.zero_()
        barrier.data_.zero_()

        epoch = 1
        mod.set_epoch(epoch)
        dist.barrier(); time.sleep(0.5)

        def reset_and_run(ep):
            K0.data_.copy_(K_local); V0.data_.copy_(V_local)
            K1.data_.zero_(); V1.data_.zero_()
            barrier.data_.zero_()
            L.zero_(); L_block.zero_()
            O.zero_(); O_block.zero_()
            mod.set_epoch(ep)
            dist.barrier(); time.sleep(0.05)
            mod.ring_attn_multinode(
                Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier,
                send_buf_ptr, recv_ptr,
                fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                arrival_ptr, ep,
                node_idx, args.num_comm_sms,
                args.num_send_sms, args.num_copy_sms, NUM_NODES,
            )

        for _ in range(args.warmup + 1):
            epoch += 1; reset_and_run(epoch); torch.cuda.synchronize()
            dist.barrier()

        samples = []
        for _ in range(args.iters):
            epoch += 1
            K0.data_.copy_(K_local); V0.data_.copy_(V_local)
            K1.data_.zero_(); V1.data_.zero_()
            barrier.data_.zero_()
            L.zero_(); L_block.zero_(); O.zero_(); O_block.zero_()
            mod.set_epoch(epoch)
            dist.barrier(); time.sleep(0.05)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            mod.ring_attn_multinode(
                Q, K0, K1, V0, V1, L, L_block, O, O_block, barrier,
                send_buf_ptr, recv_ptr,
                fifo[0], fifo[1], fifo[2], fifo[3], fifo[4],
                arrival_ptr, epoch,
                node_idx, args.num_comm_sms,
                args.num_send_sms, args.num_copy_sms, NUM_NODES,
            )
            e.record(); torch.cuda.synchronize()
            samples.append(s.elapsed_time(e))
            dist.barrier()

        wall_ms = median_then_max_cuda(samples)
        if is_chief:
            print(f"[ring_attn] seq={seq_per_dev} wall={wall_ms:.3f} ms", flush=True)
        # Store as total_seq to match NCCL reference + published chart x-axis.
        result_sizes.append(seq_per_dev)
        result_fused.append(wall_ms)

    if is_chief and args.save_json:
        # Merge with existing JSON so a single-shape bench doesn't erase others.
        from common import write_results_json
        write_results_json(Path(args.save_json), "ring_attention",
                           result_sizes, result_fused,
                           note=f"release ring_attention bench (world={world_size})")
        print(f"[ring_attn] wrote {args.save_json}", flush=True)

    if is_chief and args.compare_to:
        ok = compare_named_results("ring_attn", result_sizes, result_fused, args.compare_to)
        dist.destroy_process_group()
        if not ok: return 1
        return 0
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
