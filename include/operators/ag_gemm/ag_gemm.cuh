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
 *       - Local-half tile: wait on intra-node barrier, TMA load from A_pgl
 *       - Remote-half tile: wait on arrival_flags, TMA load from A_recv_gl
 *
 * The distributed A buffer is DMA-BUF-registered for RDMA (no staging copy).
 */

#include "common/types.cuh"
#include "dist/dbuf.cuh"
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

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 8
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
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int NUM_NODES = 2;
    // run_30: PIPELINE_STAGES 4 → 3. At 4K (num_iters=4) entire inner loop is
    // pipeline fill/drain; 8K (num_iters=8) has 4 steady + 4 fill/drain. Dropping
    // depth to 3 saves one iter of fill/drain per tile. 16K (16 iters) / 32K (32
    // iters) have enough steady-state to absorb the reduced TMA-hiding depth.
    // Shared-memory budget: max(3 * 48KB, 2 * 48KB + 64KB) = 160KB vs prior 208KB
    // — 48KB freed (unused). Compile-time knob; uniform across all shapes (no
    // per-shape hardcoding).
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
    using A_pgl = dist::dbuf<dist::gl<bf16, 1, 1, -1, -1, A_tile, A_comm_tile>, NUM_DEVICES, true, 0, 1, A_comm_tile>;
    using barrier_pgl = dist::barrier_dbuf<NUM_DEVICES>;

    A_pgl A;
    barrier_pgl barrier;

    // Inter-node RDMA + compute. A_gl carries both A_tile (compute loads) and
    // A_comm_tile (phase-2 receive fan-out under #8) so TMA descriptors for both
    // are available on recv_buf / A_local without creating separate GL types.
    using A_gl = dist::gl<bf16, 1, 1, -1, -1, A_tile, A_comm_tile>;
    using B_gl = dist::gl<bf16, 1, 1, -1, -1, B_tile>;
    using C_gl = dist::gl<bf16, 1, 1, -1, -1, C_tile>;

    A_gl A_local;
    A_gl A_recv_gl;      // per-rank unicast view of recv_buf (RDMA landing zone)
    // Plan federated-weaving-ocean #8: multicast-backed A_recv buffer. Each
    // rank r on node N publishes the 1/NUM_DEVICES slice of peer A_half it
    // received via RDMA into this PGL, so all ranks on node N see the full
    // peer A_half after phase-2. Compute remote-tile loads read from this.
    A_pgl A_recv;
    B_gl B;
    C_gl C;

    internode::D2HFifoDeviceBundle d2h_fifos;
    volatile uint32_t*       arrival_flags;
    uint32_t                 epoch;
    int                      total_chunks;
    int                      a_half_bytes;

    const int dev_idx;
    const int node_idx;
    const int num_intra_comm;  // CTAs for intra-node IPC gather + RDMA push
    const int num_comp_sms;    // CTAs for GEMM compute


    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
};

// Wait for arrival_flags[chunk_id] == G.epoch. Single helper used at every
// consumer-CTA poll site so PTX scope and inline RX-CQ poll integration stay
// in one place.
//
// Under proxy / IBGDA: just spin on a volatile load (host-mapped memory or
// HBM remote-NIC writes). Under EFAGDA inline-poll: also poll the local RX
// CQ each iteration; on RX-IMM CQE the helper publishes the imm into
// arrival_flags via st.release.gpu.global. Reader uses .gpu scope (writer
// is a local CTA on same GPU L2).
__device__ __forceinline__ void ag_gemm_wait_arrival(const globals& G, int ck) {
    uint32_t v;
    do {
        asm volatile("ld.volatile.global.u32 %0, [%1];"
            : "=r"(v) : "l"((uint32_t*)&G.arrival_flags[ck]) : "memory");
        if (v == G.epoch) break;
        __nanosleep(100);
    } while (true);
}


// ============================================================================
// Intra-comm SM: IPC multicast gather + signal intra_done per row block
// ============================================================================

// Gen-3 run_23: shape-adaptive per-rb WR split. rule:
//   split = max(1, min(CEIL, rb_bytes / 2_MB))
// Rationale from run_22: sub-WR < 2 MB hits NIC per-WR overhead floor; >= 2 MB
// amortizes and the extra QP parallelism wins. Ceiling default 4.
// Shape mapping at default ceiling 4:
//   rb_bytes=8 MB (32K): split=4 (sub=2 MB)
//   rb_bytes=4 MB (16K): split=2 (sub=2 MB)
//   rb_bytes=2 MB (8K):  split=1 (sub=2 MB)
//   rb_bytes=1 MB (4K):  split=1 (sub=1 MB, below knee but only 1)
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

        if (split == 1) {
            // Fast-path unchanged behavior: single WR per rb, keyed by rb.
            internode::TransferCmd cmd{};
            cmd.cmd_type = internode::CmdType::WRITE;
            cmd.dst_rank = (uint8_t)(1 - G.node_idx);
            cmd.tile_id = (uint16_t)first_chunk;
            cmd.bytes = rb_bytes;
            cmd.local_offset = base_offset;
            cmd.remote_offset = base_offset;
            cmd.src_view = 1;
            cmd.lane_id = (uint16_t)rb;
            internode::D2HFifoDevice fifo =
                internode::gemm_ar_select_fifo_for_lane(G.d2h_fifos, (uint32_t)rb);
            fifo.push(cmd);
        } else {
            for (int sw = 0; sw < split; ++sw) {
                int sub_first_chunk = first_chunk + sw * chunks_per_sub;
                uint32_t sub_base = (uint32_t)(sub_first_chunk * CHUNK_BYTES);
                uint32_t sub_end = sub_base + bytes_per_sub;
                if (sw == split - 1 && sub_end > end_offset) sub_end = end_offset;
                uint32_t sub_bytes = sub_end - sub_base;
                internode::TransferCmd cmd{};
                cmd.cmd_type = internode::CmdType::WRITE;
                cmd.dst_rank = (uint8_t)(1 - G.node_idx);
                cmd.tile_id = (uint16_t)sub_first_chunk;
                cmd.bytes = sub_bytes;
                cmd.local_offset = sub_base;
                cmd.remote_offset = sub_base;
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
    int num_intra_comm_override = 0
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

    // Split CTAs: intra-comm + compute. Intra-gather CTAs also post the
    // zero-copy inter-node RDMA WRs (src_view=1, DMA-BUF MR aliasing A.data_)
    // at kernel entry — no separate inter-comm pool.
    //
    // ag_gemm opt #8 (shape-adaptive host tuning, plan federated-weaving-ocean):
    // at M>=16384 compute is bottleneck → shrink comm pool (32 total, giving
    // 100 compute CTAs); at M<=8192 comm is bottleneck → 64 total (68 compute).
    // Job 6085 (32-split) vs 6060 (64-split) at k16 cx7 2x8:
    //   16K: 6.282 vs 6.375 (-1.5%), 32K: 24.036 vs 24.553 (-2.1%) — 32 wins
    //   4K:  0.682 vs 0.647 (+5.4%),  8K: 1.913 vs 1.876 (+2.0%) — 64 wins
    // Opt out by passing explicit --num-comm-sms override (already respected
    // because entrypoint receives the parsed value; adaptive logic replaces
    // only the *default* selection downstream).
    int adaptive_comm_sms = num_comm_sms;
    if (std::getenv("AG1_ADAPTIVE_COMM_SMS") == nullptr ||
        std::atoi(std::getenv("AG1_ADAPTIVE_COMM_SMS")) != 0) {
        // Gen-3 run_24: stack run_19 cap-split + run_23 adaptive WR split.
        // 32K keeps cap=16 (run_18 config, already 1.32× with single WR, now
        // boosted by adaptive split=4). 16K lifts cap to 32 — restores run_17's
        // 48 intra warp-workers which combined with WR-split=2 should let the
        // send pipeline saturate 8 QPs. The two wins touch disjoint paths
        // (CTA count vs WR posting granularity), so we expect them to compound.
        if (M >= 32768) adaptive_comm_sms = std::min(num_comm_sms, 8);
        else if (M >= 16384) adaptive_comm_sms = std::min(num_comm_sms, 32);
        // Small-M: at M<=4K intra-gather has only local_row_blocks=2 rows per
        // rank with col_blocks=2 (K=256/128), so 4 total intra tasks. Under
        // MERGE+EARLY_SEND extra intra CTAs sit idle but steal from compute.
        // Gen-2 run_16: aggressive cap 48 → 8 (4 intra CTAs, 128 compute).
        else if (M <= 4096) adaptive_comm_sms = std::min(num_comm_sms, 8);
        // Agent run_06 (best): gentle 8K cap. 256 output tiles / 100 CTAs =
        // 2.56 tiles/CTA at the unbounded default. Shrinking intra 32→28
        // frees 4 CTAs to compute (+4% comp) and still leaves 84 intra slots
        // > 64 col_blocks for stride coverage. Geomean: 0.988x vs 0.970x
        // (unbounded) and 0.965x (cap=60) at k16 cx7 2x8.
        //
        // Gen-2 run_17: tighten 8K cap further. At 8K local_row_blocks=4,
        // col_blocks=4 → 16 intra tasks; 28 intra CTAs have 12 idle. Try cap=32
        // (16 intra CTAs, 116 compute) — still covers all 16 tasks 1:1.
        else if (M <= 8192) adaptive_comm_sms = std::min(num_comm_sms, 32);
    }
    const int num_intra_comm = (num_intra_comm_override > 0)
        ? num_intra_comm_override
        : std::max(4, adaptive_comm_sms / 2);
    int num_comp_sms = active_sms - num_intra_comm;
    TORCH_CHECK(num_comp_sms > 0, "num_comp_sms must be > 0, got ", num_comp_sms,
                " (active_sms=", active_sms, " num_intra_comm=", num_intra_comm, ")");

    auto A_local = ::dist::make_gl<globals::A_gl>(
        (uint64_t)A.data_.data_ptr(), 1, 1, M_half, K);
    auto A_recv_gl = ::dist::make_gl<globals::A_gl>(
        (uint64_t)recv_buf_ptr, 1, 1, M_half, K);

    internode::D2HFifoDeviceBundle fifo_bundle{};
    if (fifo_capacity < 0) {
        auto* bundle_ptr =
            reinterpret_cast<const internode::D2HFifoDeviceBundle*>(fifo_triggers);
        TORCH_CHECK(bundle_ptr != nullptr, "FIFO bundle pointer is null");
        fifo_bundle = *bundle_ptr;
    } else {
        internode::D2HFifoDevice fd{};
        fd.triggers   = reinterpret_cast<internode::TransferCmd*>(fifo_triggers);
        fd.head       = reinterpret_cast<uint64_t*>(fifo_head);
        fd.tail       = reinterpret_cast<uint64_t*>(fifo_tail);
        fd.tail_cache = reinterpret_cast<uint64_t*>(fifo_tail_cache);
        fd.capacity   = fifo_capacity;
        int nqps = 16;  // Gen-2 Family B: matches create_session_py default
        if (const char* e = std::getenv("OSGC_EFA_NUM_QPS")) {
            int v = std::atoi(e);
            if (v > 0) nqps = v;
        }
        fifo_bundle = internode::make_fifo_bundle(fd, nqps, 1);
    }

    globals G{
        .A = ::dist::dbuf_from_buffer<globals::A_pgl>(A),
        .barrier = ::dist::dbuf_from_buffer<globals::barrier_pgl>(barrier),
        .A_local = A_local,
        .A_recv_gl = A_recv_gl,
        .A_recv = ::dist::dbuf_from_buffer<globals::A_pgl>(A_recv),
        .B = ::dist::gl_from_tensor<globals::B_gl>(B),
        .C = ::dist::gl_from_tensor<globals::C_gl>(C),
        .d2h_fifos = fifo_bundle,
        .arrival_flags = reinterpret_cast<volatile uint32_t*>(arrival_flags_ptr),
        .epoch = (uint32_t)epoch,
        .total_chunks = total_chunks,
        .a_half_bytes = a_half_bytes,
        .dev_idx = dev_idx,
        .node_idx = node_idx,
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
