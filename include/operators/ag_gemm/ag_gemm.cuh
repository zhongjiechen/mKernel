#pragma once

/**
 * @file ag_gemm_multinode.cu
 * @brief Multi-node All-Gather + GEMM — truly fused single-kernel (2 nodes, 8 GPUs).
 *
 * Single kernel launch. Two CTA groups run concurrently:
 *
 *   Intra-comm CTAs [0, num_intra_comm):
 *     IPC multicast gather of A shards within the node.
 *     Also post zero-copy RDMA WRs for this rank's M_local slice at kernel
 *     entry (src_view=1 → DMA-BUF MR aliasing A.data_), so the NIC pipelines
 *     the inter-node send concurrently with intra-gather + compute.
 *     Signals barrier[row_block] when each row block is fully gathered.
 *
 *   Comp CTAs [num_intra_comm, 132):
 *     GEMM on all M rows. Atomically claim tiles:
 *       - Local-half tile: wait on intra-node barrier, TMA load from A_distributed_tensor
 *       - Remote-half tile: wait on arrival_flags, TMA load from A_recv_local_tensor
 *
 * The distributed A buffer is DMA-BUF-registered for RDMA (no staging copy).
 */

#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "common/cuda_checks.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "comm/comm.cuh"
#include "comm/internode/d2h_fifo.cuh"
#include "comm/internode/arrival.cuh"
#include "comm/internode/types.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <vector>

using namespace kittens;

#ifndef INTRA_NUM_DEVICES
#define INTRA_NUM_DEVICES 8
#endif

namespace ag_gemm_multinode {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 132;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;
};

// 64 KB chunks — tuned via EFA sweep. Larger than DeepEP's per-token
// strategy because EFA SRD per-WR cost is high enough that larger chunks
// amortize it well (see also dispatch_gemm.cu's 512 KB choice).
static constexpr int CHUNK_BYTES = 64 * 1024;

struct comp_task {
    int rb;
    int col_idx;
    bool is_remote;
};


// ============================================================================
// Fused globals
// ============================================================================

struct globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int NUM_NODES = 2;
    // Three stages keep the producer/consumer pipeline deep enough while
    // leaving more shared memory headroom than a four-stage pipeline.
    static constexpr int PIPELINE_STAGES = 3;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using A_comm_tile = st_bf<ROW_BLOCK * 2, RED_BLOCK * 2>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    static constexpr int NUM_COMM_CHUNKS = config::DYNAMIC_SHARED_MEMORY / sizeof(A_comm_tile);

    // Intra-node IPC
    using A_distributed_tensor = dist::distributed_tensor<dist::local_tensor<bf16, 1, 1, -1, -1, A_tile, A_comm_tile>, NUM_DEVICES, true, 0, 1, A_comm_tile>;
    using barrier_distributed_tensor = dist::barrier_distributed_tensor<NUM_DEVICES>;

    A_distributed_tensor A;
    barrier_distributed_tensor barrier;

    // Inter-node RDMA + compute. A_local_tensor carries both A_tile (compute loads) and
    // A_comm_tile (phase-2 receive fan-out under #8) so TMA descriptors for both
    // are available on recv_buf / A_local without creating separate GL types.
    using A_local_tensor = dist::local_tensor<bf16, 1, 1, -1, -1, A_tile, A_comm_tile>;
    using B_local_tensor = dist::local_tensor<bf16, 1, 1, -1, -1, B_tile>;
    using C_local_tensor = dist::local_tensor<bf16, 1, 1, -1, -1, C_tile>;

    A_local_tensor A_local;
    A_local_tensor A_recv_local_tensor;      // per-rank unicast view of recv_buf (RDMA landing zone)
    // Plan federated-weaving-ocean #8: multicast-backed A_recv buffer. Each
    // rank r on node N publishes the 1/NUM_DEVICES slice of peer A_half it
    // received via RDMA into this dbuf, so all ranks on node N see the full
    // peer A_half after phase-2. Compute remote-tile loads read from this.
    A_distributed_tensor A_recv;
    B_local_tensor B;
    C_local_tensor C;

    internode::D2HFifoDeviceBundle d2h_fifos;
    volatile uint32_t*       arrival_flags;
    uint32_t                 epoch;
    int                      total_chunks;
    int                      a_half_bytes;

    const int dev_idx;
    const int node_idx;
    // Total node count (>= 2). For N == 2 the kernel's inter-node fan-out
    // loop runs once with peer rank == 1 - node_idx — bit-identical to the
    // legacy 2-node code path. For N > 2 the loop runs (N - 1) times,
    // writing each peer's dst_rank in turn. The buffer / arrival-flag
    // sizing required for true N > 2 operation is NOT yet in place; this
    // field is scaffolding for kernel-side multi-peer iteration only.
    const int num_nodes;
    const int num_intra_comm;  // CTAs for intra-node IPC gather + RDMA push
    const int num_comp_sms;    // CTAs for GEMM compute


    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
};

// Wait for arrival_flags[chunk_id] == G.epoch. Single helper used at every
// consumer-CTA poll site so PTX scope and inline RX-CQ poll integration stay
// in one place.
//
// Under proxy: just spin on a volatile load (host-mapped memory or HBM
// remote-NIC writes). Under inline-poll backends: also poll the local RX
// CQ each iteration; on RX-IMM CQE the helper publishes the imm into
// arrival_flags via st.release.gpu.global. Reader uses .gpu scope (writer
// is a local CTA on same GPU L2).
__device__ __forceinline__ void ag_gemm_wait_arrival_one(const globals& G, int slot_ck) {
    uint32_t v;
    do {
        v = comm::atomic_u32::volatile_load(&G.arrival_flags[slot_ck]);
        if (v == G.epoch) break;
        __nanosleep(100);
    } while (true);
}

// Wait for chunk `ck` to arrive from EVERY peer slot. The arrival_flags array
// is laid out [peer_slot * total_chunks + ck]; at N == 2 the loop runs once
// with slot == 0, indexing arrival_flags[ck] exactly as the legacy single-
// peer code did. At N > 2 it waits on (N-1) flags before returning.
//
// The caller is still responsible for actually consuming each peer's data —
// today it reads only A_recv (slot 0). Per-peer data merging is the
// per-kernel testbed-side step that follows from this wait being multi-
// peer-correct.
__device__ __forceinline__ void ag_gemm_wait_arrival(const globals& G, int ck) {
    const int n_peers = G.num_nodes - 1;
    for (int slot = 0; slot < n_peers; ++slot) {
        ag_gemm_wait_arrival_one(G, slot * G.total_chunks + ck);
    }
}


// ============================================================================
// Intra-comm SM: IPC multicast gather + signal intra_done per row block
// ============================================================================

// Split large row-block sends into roughly 2 MB sub-WRs, capped by `ceiling`.
__device__ __forceinline__ int ag1_compute_wr_split(int rb_bytes, int ceiling) {
    // Default knee at 2 MB. AG1_WR_SPLIT_KNEE_BYTES could override at compile
    // time if tuning is needed.
    constexpr int KNEE_BYTES = 2 * 1024 * 1024;
    int s = rb_bytes / KNEE_BYTES;
    if (s < 1) s = 1;
    if (s > ceiling) s = ceiling;
    // chunks_per_rb must be divisible by s for clean flag splitting; caller
    // guarantees chunks_per_rb is power-of-2 at k16 shapes so s in {1,2,4}
    // always divides. If a weird shape breaks this, caller falls back to 1.
    return s;
}

// Merge: post both inter rbs' DMA-BUF WRs for a single intra row that this
// rank owns (intra 256-row unit = 2 inter 128-row rbs).
//
// Under AG1_MERGE_WR_SPLIT (ceiling N>1), each inter-rb is split into
// effective_split sub-WRs sized rb_bytes/split, striped across QPs by
// (rb*split + sw) key. effective_split computed from rb_bytes — only kicks
// in at large shapes where sub-WR stays >= 2 MB.
__device__ inline void post_merge_wrs_for_intra_row(
    const globals& G, int global_row_idx, int chunks_per_rb) {
    constexpr int WR_SPLIT_CEILING = 1;
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        int rb = 2 * global_row_idx + sub;
        int first_chunk = rb * chunks_per_rb;
        int last_chunk = min(first_chunk + chunks_per_rb, G.total_chunks);
        uint32_t base_offset = (uint32_t)(first_chunk * CHUNK_BYTES);
        uint32_t end_offset = (last_chunk * CHUNK_BYTES > G.a_half_bytes)
            ? (uint32_t)G.a_half_bytes
            : (uint32_t)(last_chunk * CHUNK_BYTES);
        uint32_t rb_bytes = end_offset - base_offset;

        // Decide runtime split: knee-based + divisibility check on chunks_per_rb.
        int split = ag1_compute_wr_split((int)rb_bytes, WR_SPLIT_CEILING);
        if (chunks_per_rb < split || (chunks_per_rb % split) != 0) split = 1;
        int chunks_per_sub = chunks_per_rb / split;
        uint32_t bytes_per_sub = (uint32_t)(chunks_per_sub * CHUNK_BYTES);

        // Fan out one cmd per peer (num_nodes - 1 peers). The buffer and
        // arrival-flag layouts haven't been generalized for N > 2 yet, so
        // for N > 2 multiple peers would race on the same recv slot —
        // tracked under "kernel-side multi-peer scaffolding". For N == 2
        // (the validated configuration) the loop runs once with the
        // legacy peer rank.
        const int n_peers = G.num_nodes - 1;
        // Per-peer recv_buf / arrival-flag layout: peer p's data lands at
        //   recv_buf  + slot_at_peer * G.a_half_bytes
        //   arrival   + slot_at_peer * G.total_chunks
        // For N == 2, slot_at_peer is always 0, so the multiplications below
        // contribute zero and behavior is bit-identical. For N > 2 these
        // place each sender's data into its own slot at the receiver. The
        // bench must allocate (N-1) × the legacy size for recv_buf and
        // num_tiles; consumer-side wait loops over (N-1) peer slots — the
        // bench/consumer-side counterpart is the next testbed-side step.
        if (split == 1) {
            // Fast-path unchanged behavior: single WR per rb, keyed by rb.
            for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                const int peer_rank = internode::peer_rank_for_slot(
                    G.node_idx, G.num_nodes, peer_slot);
                const int sap = internode::slot_at_peer(G.node_idx, peer_rank);
                internode::TransferCmd cmd{};
                cmd.cmd_type = internode::CmdType::WRITE;
                cmd.dst_rank = (uint8_t)peer_rank;
                cmd.tile_id = (uint16_t)(sap * G.total_chunks + first_chunk);
                cmd.bytes = rb_bytes;
                cmd.local_offset = base_offset;
                cmd.remote_offset = (uint32_t)sap * (uint32_t)G.a_half_bytes + base_offset;
                cmd.src_view = 1;
                cmd.lane_id = (uint16_t)rb;
                internode::D2HFifoDevice fifo =
                    internode::gemm_ar_select_fifo_for_lane(G.d2h_fifos, (uint32_t)rb);
                fifo.push(cmd);
            }
        } else {
            for (int sw = 0; sw < split; ++sw) {
                int sub_first_chunk = first_chunk + sw * chunks_per_sub;
                uint32_t sub_base = (uint32_t)(sub_first_chunk * CHUNK_BYTES);
                uint32_t sub_end = sub_base + bytes_per_sub;
                if (sw == split - 1 && sub_end > end_offset) sub_end = end_offset;
                uint32_t sub_bytes = sub_end - sub_base;
                for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                    const int peer_rank = internode::peer_rank_for_slot(
                        G.node_idx, G.num_nodes, peer_slot);
                    const int sap = internode::slot_at_peer(G.node_idx, peer_rank);
                    internode::TransferCmd cmd{};
                    cmd.cmd_type = internode::CmdType::WRITE;
                    cmd.dst_rank = (uint8_t)peer_rank;
                    cmd.tile_id = (uint16_t)(sap * G.total_chunks + sub_first_chunk);
                    cmd.bytes = sub_bytes;
                    cmd.local_offset = sub_base;
                    cmd.remote_offset = (uint32_t)sap * (uint32_t)G.a_half_bytes + sub_base;
                    cmd.src_view = 1;
                    cmd.lane_id = (uint16_t)(rb * split + sw);
                    internode::D2HFifoDevice fifo =
                        internode::gemm_ar_select_fifo_for_lane(
                            G.d2h_fifos, (uint32_t)(rb * split + sw));
                    fifo.push(cmd);
                }
            }
        }
    }
}

// ============================================================================
// Host entrypoint
// ============================================================================

// Forward declaration: kernel body and raw CUDA launch live in src/ag_gemm.cu.
void launch_fused_ag_gemm(const globals& G, unsigned int active_sms);

void entrypoint(
    dist::ParallelBuffer& A,
    const at::Tensor& B,
    at::Tensor& C,
    dist::ParallelBuffer& barrier,
    int64_t recv_buf_ptr,
    int64_t fifo_triggers, int64_t fifo_head,
    int64_t fifo_tail, int64_t fifo_tail_cache, int fifo_capacity,
    int64_t arrival_flags_ptr,
    int epoch,
    int node_idx,
    int num_comm_sms,
    int a_half_bytes,
    dist::ParallelBuffer& A_recv,  // #8: multicast-backed peer A_half
    const int active_sms = config::NUM_BLOCKS,
    int num_intra_comm_override = 0,
    int num_nodes = 2  // total node count (>= 2). N == 2 reproduces the
                       // legacy 2-node behavior bit-for-bit. N > 2 routes
                       // each cmd to the right peer rank but receive
                       // buffers / arrival flags are still single-peer-
                       // sized — finishing N > 2 needs a 4-node testbed.
) {
    TORCH_CHECK(B.is_cuda() && B.is_contiguous(), "B must be contiguous CUDA");
    TORCH_CHECK(C.is_cuda() && C.is_contiguous(), "C must be contiguous CUDA");
    TORCH_CHECK(B.dtype() == at::ScalarType::BFloat16, "B must be bf16");

    const int dev_idx = A.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    const int M_half = A.data_.size(0);
    const int K = A.data_.size(1);
    const int N = B.size(1);
    const int M = M_half * 2;

    TORCH_CHECK(M % globals::ROW_BLOCK == 0);
    TORCH_CHECK(K % globals::RED_BLOCK == 0);
    TORCH_CHECK(N % globals::COL_BLOCK == 0);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N);
    // Intra-gather geometry needs M_half/(ROW_BLOCK*2) divisible by NUM_DEVICES
    // and >= NUM_DEVICES, i.e. M_half must be a multiple of NUM_DEVICES*ROW_BLOCK*2.
    // For 8 GPUs: M_half % 2048 == 0, so M % 4096 == 0.
    TORCH_CHECK(M_half >= globals::NUM_DEVICES * globals::ROW_BLOCK * 2,
                "M must be >= ", 4 * globals::NUM_DEVICES * globals::ROW_BLOCK,
                " (got M=", M, ")");
    TORCH_CHECK(M_half % (globals::NUM_DEVICES * globals::ROW_BLOCK * 2) == 0,
                "M must be a multiple of ", 4 * globals::NUM_DEVICES * globals::ROW_BLOCK,
                " (got M=", M, ")");

    int total_chunks = (a_half_bytes + CHUNK_BYTES - 1) / CHUNK_BYTES;

    // Split CTAs between intra-comm and compute. Intra-gather CTAs also post
    // zero-copy inter-node RDMA WRs, so there is no separate inter-comm pool.
    int adaptive_comm_sms = num_comm_sms;
    if (std::getenv("AG1_ADAPTIVE_COMM_SMS") == nullptr ||
        std::atoi(std::getenv("AG1_ADAPTIVE_COMM_SMS")) != 0) {
        if (M >= 32768) adaptive_comm_sms = std::min(num_comm_sms, 8);
        else if (M >= 16384) adaptive_comm_sms = std::min(num_comm_sms, 32);
        // Small-M: at M<=4K intra-gather has only local_row_blocks=2 rows per
        // rank with col_blocks=2 (K=256/128), so 4 total intra tasks. Under
        // MERGE+EARLY_SEND extra intra CTAs sit idle but steal from compute.
        else if (M <= 4096) adaptive_comm_sms = std::min(num_comm_sms, 8);
        // At M<=8K, keep enough intra CTAs to cover the small number of
        // gather tasks while returning idle CTAs to compute.
        else if (M <= 8192) adaptive_comm_sms = std::min(num_comm_sms, 32);
    }
    const int num_intra_comm = (num_intra_comm_override > 0)
        ? num_intra_comm_override
        : std::max(4, adaptive_comm_sms / 2);
    int num_comp_sms = active_sms - num_intra_comm;
    TORCH_CHECK(num_comp_sms > 0, "num_comp_sms must be > 0, got ", num_comp_sms,
                " (active_sms=", active_sms, " num_intra_comm=", num_intra_comm, ")");

    auto A_local = ::dist::make_local_tensor<globals::A_local_tensor>(
        (uint64_t)A.data_.data_ptr(), 1, 1, M_half, K);
    auto A_recv_local_tensor = ::dist::make_local_tensor<globals::A_local_tensor>(
        (uint64_t)recv_buf_ptr, 1, 1, M_half, K);

    auto fifo_bundle = internode::resolve_fifo_bundle(
        fifo_triggers, fifo_head, fifo_tail, fifo_tail_cache, fifo_capacity,
        16);

    globals G{
        .A = ::dist::distributed_tensor_from_buffer<globals::A_distributed_tensor>(A),
        .barrier = ::dist::distributed_tensor_from_buffer<globals::barrier_distributed_tensor>(barrier),
        .A_local = A_local,
        .A_recv_local_tensor = A_recv_local_tensor,
        .A_recv = ::dist::distributed_tensor_from_buffer<globals::A_distributed_tensor>(A_recv),
        .B = ::dist::local_tensor_from_tensor<globals::B_local_tensor>(B),
        .C = ::dist::local_tensor_from_tensor<globals::C_local_tensor>(C),
        .d2h_fifos = fifo_bundle,
        .arrival_flags = reinterpret_cast<volatile uint32_t*>(arrival_flags_ptr),
        .epoch = (uint32_t)epoch,
        .total_chunks = total_chunks,
        .a_half_bytes = a_half_bytes,
        .dev_idx = dev_idx,
        .node_idx = node_idx,
        .num_nodes = num_nodes,
        .num_intra_comm = num_intra_comm,
        .num_comp_sms = num_comp_sms,
    };


    // Fast-path #1: barrier_reset is inlined into fused_kernel's exit (see
    // fused_kernel in src/ag_gemm.cu). Saves one cudaLaunchKernel +
    // persistent-CTA-startup per iter — measurable at small M where total
    // time is sub-millisecond.
    launch_fused_ag_gemm(G, (unsigned int)active_sms);

}

}  // namespace ag_gemm_multinode
