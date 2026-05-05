#pragma once

/**
 * @file gemm_rs_multinode.cu
 * @brief Proper 2-node × 8-GPU GEMM + Reduce-Scatter.
 *
 * Borrows the intra-node 8-GPU pattern from
 *   experiments/dynamic_sm_allocation/question5_gemm_rs/gemm_rs.cu
 * (output_pgl + tma::store_add_async + per-task ready flags + atomic
 * task claiming) and bolts on an inter-node phase that exchanges each
 * GPU's owned M/8 rows with its same-index peer on the other node.
 *
 * Data flow per GPU (g ∈ [0, 8) within node n ∈ [0, 2)):
 *   Initial: A[g] (M, K), B[g] (K, N)            ← partial inputs (different per GPU)
 *   Phase A: local workspace = A[g] @ B[g]
 *           tma::store_add_async to output_pgl[owner_in_node]
 *           After A: output[g] has M/8 rows summed over 8 LOCAL GPUs.
 *   Sync 1:  intra-node barrier_all (all 8 local GPUs done with Phase A)
 *   Phase B: copy output[g] → staging_buf (contiguous), push RDMA to peer
 *           node's GPU g, which receives into recv_buf. Set sender_done[tile].
 *   Phase C: poll arrival_flags + sender_done, add recv_buf to output[g]
 *           in-place. Final: output[g] has M/8 rows summed over all 16 GPUs.
 *
 * Output shape per GPU: (M/8, N) — true 16-way RS divided 1/8 per node.
 *
 * Current in-tree hot path is a single CTA-specialized kernel:
 *   compute CTAs -> intranode RS CTAs -> inter send CTAs -> inter reduce CTAs
 * with fine-grained per-tile handoff between each phase.
 */

#include "common/types.cuh"
#include "dist/dbuf.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "common/cuda_checks.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "comm/comm.cuh"
#include "comm/atomic_u32.cuh"
#include "comm/internode/d2h_fifo.cuh"
#include "comm/internode/arrival.cuh"
#include "comm/internode/types.h"

#include <ATen/ATen.h>
#include <assert.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace kittens;

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 8
#endif

namespace gemm_rs_multinode {

// ============================================================================
// Config (matches intranode gemm_rs.cu)
// ============================================================================

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

// ============================================================================
// Acquire/release memory helpers (match gemm_ar's gemm_ar_acquire_load_u32 /
// gemm_ar_release_store_u32 but namespaced for gemm_rs).
// ============================================================================

// PTX wrappers come from include/comm/atomic_u32.cuh (shared with gemm_ar
// and any future kernel). Kept as gemm_rs_-prefixed inline aliases so existing
// call sites read unchanged.
template <typename PtrT>
__device__ inline uint32_t gemm_rs_acquire_load_u32(PtrT* ptr) {
    return osgc::atomic_u32::acquire_load_gpu(ptr);
}
template <typename PtrT>
__device__ inline void gemm_rs_release_store_u32(PtrT* ptr, uint32_t val) {
    osgc::atomic_u32::release_store_gpu(ptr, val);
}

// GPU-scope arrival-flag polling primitives. Under EFAGDA inline-poll,
// arrival_flags lives in GPU HBM, is written by a local CQ-poller CTA
// (st.release.gpu.global), and read here by the reducer CTA. Both same GPU,
// same L2 — .gpu scope is sufficient (.sys adds an HBM-flush fence we don't
// need). Under proxy-EFA, the CPU proxy still writes the flag via PCIe to
// host-mapped memory and the reader needed .sys; with EFAGDA's inline poll
// + HBM allocation that path is gone.
// - acquire.gpu — best when a single chunk is the critical-path gate.
// - relaxed.gpu — better throughput when many chunks polled in parallel.
// The caller chooses at runtime via Rt.use_acquire_poll.
__device__ __forceinline__ uint32_t gemm_rs_poll_arrival_acquire(volatile uint32_t* p) {
    return osgc::atomic_u32::acquire_load_gpu(const_cast<uint32_t*>(p));
}
__device__ __forceinline__ uint32_t gemm_rs_poll_arrival_relaxed(volatile uint32_t* p) {
    return osgc::atomic_u32::relaxed_load_gpu(const_cast<uint32_t*>(p));
}

// ============================================================================
// Kernel 1: intra-node compute + 8-GPU reduce-scatter via PGL atomic-add
// (Direct port of dynamic_sm_allocation/question5_gemm_rs/gemm_rs.cu)
// ============================================================================

struct intra_globals {
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    using A_gl = dist::gl<bf16, 1, 1, -1, -1, A_tile>;
    using B_gl = dist::gl<bf16, 1, 1, -1, -1, B_tile>;
    // Workspace is a multicast PGL under GEMM_RS_MULTIMEM_RS: each GPU has its own
    // replica, and multimem.ld_reduce via workspace.mc_ptr_at pulls the 8-way
    // sum in one HW op (matching gemm_ar's gemm_ar_pipelined_rs_tile). The
    // compute/intra-RS TMA paths index workspace[dev_idx] for the local gl,
    // which doesn't need a multicast binding — so when GEMM_RS_MULTIMEM_RS is off
    // we gate the multicast flag to match the Python-side DistBuffer.
    // This is required at M=65536 where the 8 GiB workspace exceeds the
    // cuMulticastBindMem single-binding granularity (CUDA_ERROR_ILLEGAL_STATE).
    using workspace_pgl = dist::dbuf<dist::gl<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, false>;
    using output_pgl = dist::dbuf<dist::gl<bf16, 1, 1, -1, -1, C_tile>, NUM_DEVICES, false>;
    using barrier_pgl = dist::barrier_dbuf<NUM_DEVICES>;
    // GEMM_RS_READY_VIA_MULTIMEM: per-chunk "this GPU's compute done" flag. Each
    // peer writes `epoch` to its own replica on the last tile of a chunk.
    // Owner polls multimem.ld_reduce.min across all 8 replicas — returns
    // `epoch` iff every peer has stamped this chunk. Replaces the 8-way PGL
    // barrier (8 sys atomics per chunk) with 8 local stores + 1 HW op.
    using ready_chunk_pgl = dist::dbuf<dist::gl<int, 1, 1, -1, -1>, NUM_DEVICES, true>;

    A_gl A;
    B_gl B;
    workspace_pgl workspace;
    output_pgl output;
    // GEMM_RS_INTRA_RS_DIRECT_STAGING (Option A): cross-GPU TMA atomic-add target
    // for intra-RS. Points to per-owner staging_buf (chunk-major tight). Built
    // from a Python-side DistBuffer whose shape is (m_local, n) but we
    // view it as a gl with (total_inter_tiles*128, 256) rows × cols so tile
    // coord (2*global_tile_idx + i, 0) with C_tile=(64,256) lands on the
    // correct chunk-major byte offset.
    output_pgl staging;
    barrier_pgl barrier;
    int *ready;
    const int dev_idx;
    const int num_comm_sms;
    const int num_comp_sms;
    unsigned int *next_compute;
    unsigned int *next_comm;
    unsigned int *kernel_done;

    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
};

struct fused_globals {
    static constexpr int NUM_DEVICES = intra_globals::NUM_DEVICES;
    static constexpr int PIPELINE_STAGES = intra_globals::PIPELINE_STAGES;
    static constexpr int SUPER_M = intra_globals::SUPER_M;
    static constexpr int ROW_BLOCK = intra_globals::ROW_BLOCK;
    static constexpr int COL_BLOCK = intra_globals::COL_BLOCK;
    static constexpr int RED_BLOCK = intra_globals::RED_BLOCK;
    // Inter-node send/reduce granularity: N column-tiles per chunk (matches gemm_ar).
    // RDMA fires per row-block (coalesced) after all chunks in row complete.
    // Override via -DQ5_RDMA_CHUNK_TILES=N (mirrors GEMM_AR_RDMA_CHUNK_TILES).
    // gemm_ar cx7 run showed M=2048/4096 win with CHUNK_TILES=2.
    static constexpr int CHUNK_TILES = 4;

    using A_tile = intra_globals::A_tile;
    using B_tile = intra_globals::B_tile;
    using C_tile = intra_globals::C_tile;

    using A_gl = intra_globals::A_gl;
    using B_gl = intra_globals::B_gl;
    using workspace_pgl = intra_globals::workspace_pgl;
    using output_pgl = intra_globals::output_pgl;
    using barrier_pgl = intra_globals::barrier_pgl;

    struct runtime_state {
        int *inter_ready;
        bf16 *output_local;
        bf16 *recv_buf;
        bf16 *staging_buf;
        int M_local;
        int N;
        int node_idx;
        int num_nodes;  // total node count (>= 2). N == 2 reproduces the
                        // legacy 2-node code path bit-for-bit. Scaffolding
                        // for kernel-side multi-peer iteration only;
                        // recv_buf / staging are still single-peer-sized.
        unsigned int *sender_done;
        internode::D2HFifoDeviceBundle d2h_fifos;
        volatile uint32_t *arrival_flags;
        uint32_t epoch;
        int total_tiles;
        unsigned int *next_send;       // work-stealing cursor for per-chunk send
        unsigned int *next_reduce;     // work-stealing cursor for per-chunk reduce
        unsigned int *chunks_processed; // EFAGDA stay-and-poll: incremented per chunk
                                        // done; reducer pollers exit stay-and-poll
                                        // when *chunks_processed == total_chunks.
        uint32_t *chunk_tiles_done;    // per-chunk atomic counter: how many tiles finished intra-RS

        // Chunk barrier: per-chunk counter for batching PGL signals (matches gemm_ar)
        uint32_t *comp_chunk_tiles_done;   // size: row_blocks * chunks_per_row (global row indexing)
        // GEMM_RS_MULTIMEM_RS: compute-side chunk-done counter (compute CTAs
        // increment this; the last to hit tiles_this_chunk signals the PGL
        // barrier so the owner's intra-RS can begin its multimem pull).
        uint32_t *comp_gemm_done;
        // GEMM_RS_MULTIMEM_RS: per-chunk "ready to send" flag set by the owner's
        // intra-RS after multimem.ld_reduce + local store for all tiles of
        // the chunk. SFR sender polls this (instead of the PGL barrier) to
        // avoid aliasing the compute-side 8-way barrier signal.
        uint32_t *chunk_sendable;
        // GEMM_RS_HELP_SEND (iter 5 axis): per-chunk claim bitmap used to arbitrate
        // between dedicated SFR senders and intra-RS donor CTAs that help post
        // chunks after their primary intra-RS drain completes. Size =
        // total_global_chunks (same indexing as chunk_sendable). Each slot is
        // atomicCAS(0->1) before a CTA posts that chunk. Whichever CTA wins
        // the CAS posts once; the loser skips. Memset to 0 per launch.
        uint32_t *chunk_send_claimed;
        // GEMM_RS_READY_VIA_MULTIMEM: multicast u32 per-chunk "my compute done"
        // flag (per GPU stamps `epoch` on last-tile-in-chunk). Non-nullptr
        // when the flag path is compiled in.
        intra_globals::ready_chunk_pgl *ready_chunk;
        // Arrival queue: per-chunk flag published by reducer after RDMA arrival detected
        uint32_t *remote_arrived_flag;     // size: total_chunks; 1 = arrived
        // Cached layout values for device-side use
        int chunks_per_row;
        int chunk_tiles_val;    // min(CHUNK_TILES, col_blocks)
        int col_blocks_val;
        int row_blocks_per_slice;  // M_local / ROW_BLOCK
        // Arrival poll tuning (T2a/T2b). Selected host-side based on shape:
        //   use_acquire_poll=1 -> ld.acquire.sys  (lower latency at small/mid M)
        //   use_acquire_poll=0 -> ld.relaxed.sys  (better throughput at large M)
        // reduce_poll_sleep_ns: backoff between arrival polls. At M>=16K the RDMA
        // round-trip is many ms; 1000ns reduces cache thrash without visibly
        // adding wake latency. Default 100ns preserves prior behavior.
        uint8_t use_acquire_poll;
        // Iter24 EXPLORATION (LAYOUT_STAGING + PRODUCER_STORE axes refinement):
        // Runtime gate for the intra-RS dual-write path (iter23 / run_023 CW).
        // When 1: gemm_rs_pipelined_rs_tile writes tmps[u] to BOTH output_local AND
        // staging_buf (eliminating the sender pack). When 0: write only to
        // output_local (run_017 byte-identical path) and the sender CTAs do the
        // explicit pack. Host-side sets this based on total_chunks threshold:
        // small-M (total_chunks < GEMM_RS_INTRA_RS_DUAL_WRITE_GATE, default 4)
        // reverts to run_017's path to reclaim the LSU store-queue-pressure
        // cost at M=2K that was observed in run_023.
        //
        // This is a PURE RUNTIME gate to avoid the iter-15 FALSIFIED pathology
        // where a new compile-time branch emission itself slowed M=2K by 11%.
        uint8_t use_intra_rs_dual_write;
        uint8_t _pad0[2];
        uint32_t reduce_poll_sleep_ns;
    };

    intra_globals intra;
    int num_send_sms;
    int num_reduce_sms;
    runtime_state *rt;


};

// ============================================================================
// Tile visit order decode functions (matches gemm_ar gemm_ar_multinode_common.cuh).
// Four variants selectable via compile-time #define.
// ============================================================================

// Default: all tiles of slice 0, then slice 1, etc. Within each slice: row-major.
__device__ inline void gemm_rs_slice_row_major_decode(
    int task_id, int row_blocks_per_slice, int col_blocks, int& row_idx, int& col_idx
) {
    const int tiles_per_slice = row_blocks_per_slice * col_blocks;
    const int slice_idx = task_id / tiles_per_slice;
    const int local_idx = task_id - slice_idx * tiles_per_slice;
    const int rb_in_slice = local_idx / col_blocks;
    col_idx = local_idx - rb_in_slice * col_blocks;
    row_idx = slice_idx * row_blocks_per_slice + rb_in_slice;
}

// Round-robin across GPU slices by row-block index. Balances barrier progress.
__device__ inline void gemm_rs_slice_interleaved_decode(
    int task_id, int row_blocks_per_slice, int col_blocks, int num_slices,
    int& row_idx, int& col_idx
) {
    const int tiles_per_round = num_slices * col_blocks;
    const int round_idx = task_id / tiles_per_round;
    const int within_round = task_id - round_idx * tiles_per_round;
    const int slice_idx = within_round / col_blocks;
    col_idx = within_round - slice_idx * col_blocks;
    const int rb_in_slice = round_idx;
    row_idx = slice_idx * row_blocks_per_slice + rb_in_slice;
}

// Device-relative interleaving: each GPU starts its round with its own slice.
// Front-loads RDMA-relevant tiles so inter-node sends start earlier.
__device__ inline void gemm_rs_slice_interleaved_devrel_decode(
    int task_id, int row_blocks_per_slice, int col_blocks, int num_slices,
    int dev_idx, int& row_idx, int& col_idx
) {
    const int tiles_per_round = num_slices * col_blocks;
    const int round_idx = task_id / tiles_per_round;
    const int within_round = task_id - round_idx * tiles_per_round;
    const int raw_slice = within_round / col_blocks;
    const int slice_idx = (raw_slice + dev_idx) % num_slices;
    col_idx = within_round - raw_slice * col_blocks;
    const int rb_in_slice = round_idx;
    row_idx = slice_idx * row_blocks_per_slice + rb_in_slice;
}

// Super-M interleaving within each slice (existing gemm_rs pattern, matches gemm_ar variant).
__device__ inline void gemm_rs_slice_super_m_decode(
    int task_id, int row_blocks_per_slice, int col_blocks, int super_m,
    int& row_idx, int& col_idx
) {
    const int tiles_per_slice = row_blocks_per_slice * col_blocks;
    const int slice_idx = task_id / tiles_per_slice;
    const int local_idx = task_id - slice_idx * tiles_per_slice;
    const int super_rows = (row_blocks_per_slice / super_m) * super_m;
    const int super_blocks = super_m * col_blocks;
    if (local_idx < super_rows * col_blocks) {
        const int band = local_idx / super_blocks;
        const int in_band = local_idx - band * super_blocks;
        const int rb_in_band = in_band % super_m;
        col_idx = in_band / super_m;
        row_idx = slice_idx * row_blocks_per_slice + band * super_m + rb_in_band;
        return;
    }
    const int tail_rows = row_blocks_per_slice - super_rows;
    const int tail_idx = local_idx - super_rows * col_blocks;
    const int rb_in_tail = tail_idx % tail_rows;
    col_idx = tail_idx / tail_rows;
    row_idx = slice_idx * row_blocks_per_slice + super_rows + rb_in_tail;
}

// Dispatcher: compile-time selection of tile visit order (matches gemm_ar's gemm_ar_decode_comp_task).
template <typename G>
__device__ inline void gemm_rs_decode_comp_task(
    int task_id, int row_blocks_per_slice, int col_blocks, int dev_idx,
    int& row_idx, int& col_idx
) {
    gemm_rs_slice_interleaved_devrel_decode(task_id, row_blocks_per_slice, col_blocks,
                                       G::NUM_DEVICES, dev_idx, row_idx, col_idx);
}

// Compute -> intra-RS ready signalling. Default is per-tile (batch=1): each
// compute tile writes 1 to ready[task_id]. With GEMM_RS_COMPUTE_SIGNAL_BATCH=N>1
// tiles are grouped in consecutive N-ID runs; each tile atomic-adds 1 to the
// group's first-tile slot, and wait_ready spins until that slot reaches N.
// This reduces release-store count by N and amortizes the ld.acquire loop but
// serializes intra-RS start on the SLOWEST tile within a group, so batch=2/4
// is the practical sweet range.
#define GEMM_RS_COMPUTE_SIGNAL_BATCH 1

__device__ inline void signal_ready(int *ready, int task_id) {
    asm volatile("{st.release.gpu.global.u32 [%0], %1;}"
                 :: "l"(ready + task_id), "r"(1u) : "memory");
}

// gemm_rs-local variant of TK's signal(): performs a cross-GPU atomic-add release
// on the peer's barrier slot at GPU scope rather than system scope. Cross-GPU
// PGL memory on Hopper is hardware-coherent over NVLink, so release.gpu
// should deliver the increment to peer-GPU readers (the `wait()` in the send
// path uses ld.relaxed.sys, whose acquire is for ordering not visibility).
// Used in fused_comm_tile_impl to avoid the sys-scope atomic cost on the
// intra-RS → send barrier path.
__device__ inline void gemm_rs_signal_barrier_gpu(int *barrier_slot_ptr, int val) {
    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}"
                 :: "l"(barrier_slot_ptr), "r"(val) : "memory");
}

__device__ inline void wait_ready(const int *ready, int task_id) {
    unsigned int v;
    do {
        asm volatile("{ld.acquire.gpu.global.u32 %0, [%1];}"
                     : "=r"(v) : "l"(ready + task_id) : "memory");
        if (v != 1u) __nanosleep(16);
    } while (v != 1u);
}

__device__ __forceinline__ int gemm_rs_send_ready_bitmap_queue_capacity(
    int row_blocks_per_dev, int num_send_sms, int chunks_per_row) {
    if (num_send_sms <= 0) return 0;
    const int max_rbs_per_sender =
        (row_blocks_per_dev + num_send_sms - 1) / num_send_sms;
    return max_rbs_per_sender * chunks_per_row;
}

__device__ __forceinline__ int gemm_rs_send_ready_bitmap_words_per_queue(
    int row_blocks_per_dev, int num_send_sms, int chunks_per_row) {
    const int cap = gemm_rs_send_ready_bitmap_queue_capacity(
        row_blocks_per_dev, num_send_sms, chunks_per_row);
    return (cap + 31) >> 5;
}

__device__ __forceinline__ int gemm_rs_send_ready_bitmap_region_base(
    int row_blocks_per_dev, int chunks_per_row) {
    return row_blocks_per_dev * fused_globals::NUM_DEVICES * chunks_per_row;
}


// ============================================================================
// Host entrypoint
// ============================================================================

// Iter14 EXPLORATION (LAUNCH_SETUP axis): replace the 9 per-launch
// cudaMemsetAsync calls that zero device-side scratch buffers with ONE
// fused-zero kernel launch. Motivation: at M=2K the fused kernel's 327 µs
// tail is ~130 µs slower than NCCL's 195 µs; scout 14's diagnostic
// attributes ~20-30 µs of that to per-memset driver overhead (9 × ~2-4 µs
// CUDA launch overhead serialized on one stream). Flag-gated so the
// baseline 9-memset path is byte-identical when GEMM_RS_FUSED_RESET=0.
//
// Correctness: each region is zeroed with 32-bit writes. All target buffers
// are 4-byte aligned (cudaMalloc gives 256-byte alignment; sizes are
// multiples of sizeof(unsigned int) or sizeof(uint32_t) = 4). One block
// per region (up to GEMM_RS_FUSED_RESET_MAX_REGIONS), grid-stride over 32-bit
// words within the region. Single launch = single stream-serialization
// point, instead of 9.
#define GEMM_RS_FUSED_RESET_MAX_REGIONS 16
struct gemm_rs_zero_regions_t {
    void*    ptrs[GEMM_RS_FUSED_RESET_MAX_REGIONS];
    size_t   bytes[GEMM_RS_FUSED_RESET_MAX_REGIONS];
    int      n;
};

// Forward declarations: kernel bodies live in src/gemm_rs.cu. fused_kernel is
// launched via the thin wrapper below (template instantiation needs the body
// in scope, so it must happen in the same TU as the definition); the zero
// kernel is __global__ so a forward decl is enough.
void launch_fused_gemm_rs(const fused_globals& G, unsigned int active_sms);
__global__ void gemm_rs_fused_zero_kernel(gemm_rs_zero_regions_t regs);

static unsigned int *g_counters[intra_globals::NUM_DEVICES] = {nullptr};
// Per-device sender_done flags (one per inter-node tile). The sender CTA
// sets sender_done[tile_id]=1 after reading and staging the tile from
// output_local; the reducer CTA waits for sender_done[tile_id]==1 before
// overwriting output_local in place. This eliminates the output_final
// scratch buffer and the post-kernel cudaMemcpyAsync tail.
static unsigned int *g_sender_done[intra_globals::NUM_DEVICES] = {nullptr};
static int g_sender_done_ntiles[intra_globals::NUM_DEVICES] = {0};
static int *g_inter_ready[intra_globals::NUM_DEVICES] = {nullptr};
static int g_inter_ready_ntiles[intra_globals::NUM_DEVICES] = {0};
static fused_globals::runtime_state *g_fused_runtime[intra_globals::NUM_DEVICES] = {nullptr};
// GEMM_RS_READY_VIA_MULTIMEM: per-device storage for the ready_chunk PGL struct.
// Holds per-dev gls + mc_ptr. Runtime_state.ready_chunk points here.
static intra_globals::ready_chunk_pgl *g_ready_chunk_pgl_dev[intra_globals::NUM_DEVICES] = {nullptr};
// Per-chunk completion tracking (mirrors gemm_ar's intra_chunk_tiles_done).
static uint32_t *g_chunk_tiles_done[intra_globals::NUM_DEVICES] = {nullptr};
static int g_chunk_alloc_nchunks[intra_globals::NUM_DEVICES] = {0};
// Chunk barrier: per-chunk counter for batching PGL signals (global row indexing).
static uint32_t *g_comp_chunk_tiles_done[intra_globals::NUM_DEVICES] = {nullptr};
static int g_comp_chunk_alloc[intra_globals::NUM_DEVICES] = {0};
// GEMM_RS_MULTIMEM_RS: compute-side chunk-done counter (separate from
// comp_chunk_tiles_done, which the owner-side uses for intra-RS progress).
static uint32_t *g_comp_gemm_done[intra_globals::NUM_DEVICES] = {nullptr};
static int g_comp_gemm_done_alloc[intra_globals::NUM_DEVICES] = {0};
// GEMM_RS_MULTIMEM_RS: per-chunk "ready to send" flag set by owner's intra-RS.
static uint32_t *g_chunk_sendable[intra_globals::NUM_DEVICES] = {nullptr};
static int g_chunk_sendable_alloc[intra_globals::NUM_DEVICES] = {0};
// GEMM_RS_HELP_SEND: per-chunk atomicCAS claim bitmap arbitrating dedicated
// senders vs intra-RS donor helpers. 0=unclaimed, 1=claimed-by-someone.
static uint32_t *g_chunk_send_claimed[intra_globals::NUM_DEVICES] = {nullptr};
static int g_chunk_send_claimed_alloc[intra_globals::NUM_DEVICES] = {0};
// Arrival queue: per-chunk remote_arrived_flag (published by reducer).
static uint32_t *g_remote_arrived_flag[intra_globals::NUM_DEVICES] = {nullptr};
static int g_remote_arrived_alloc[intra_globals::NUM_DEVICES] = {0};






// Phase 1 entrypoint: compute + intranode 8-GPU reduce-scatter (no inter-node).
// Host should call dist.barrier() after this returns to ensure all 8 local GPUs
// have completed their atomic-adds before phase 2.

// True single-launch gemm_rs path: compute + intranode RS + inter send + inter reduce.
void entrypoint_fused(
    const at::Tensor &A,
    const at::Tensor &B,
    dist::ParallelBuffer &workspace,
    dist::ParallelBuffer &output,
    dist::ParallelBuffer &barrier,
    const at::Tensor &ready,
    int64_t recv_buf_ptr,
    int64_t staging_buf_ptr,
    int64_t fifo_triggers,
    int64_t fifo_head,
    int64_t fifo_tail,
    int64_t fifo_tail_cache,
    int fifo_capacity,
    int64_t arrival_flags_ptr,
    int epoch,
    int node_idx,
    int num_comp_sms,
    int num_intra_comm,
    int num_send_sms,
    int num_reduce_sms,
    int64_t use_acquire_poll,
    int64_t reduce_poll_sleep_ns,
    // GEMM_RS_READY_VIA_MULTIMEM: optional multicast tensor. Pass a dummy
    // scalar-sized DistBuffer when not in use (kernel ignores it
    // unless GEMM_RS_READY_VIA_MULTIMEM was compiled in).
    dist::ParallelBuffer &ready_chunk,
    // GEMM_RS_INTRA_RS_DIRECT_STAGING: optional staging DistBuffer. When
    // provided (not None), intra-RS atomic-add targets staging (chunk-major)
    // instead of output_local (row-major). When None, falls back to the
    // traditional output-local path — byte-identical to pre-Option-A runs.
    pybind11::object staging_obj,
    int num_nodes = 2  // Total node count (>= 2). N == 2 reproduces the
                       // legacy 2-node behavior bit-for-bit.
) {
    const int dev_idx = output.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    const int M = (int)A.size(0);
    const int N = (int)B.size(1);
    const int M_local = (int)output.data_.size(0);
    const int row_blocks = M / intra_globals::ROW_BLOCK;
    const int col_blocks = N / intra_globals::COL_BLOCK;
    const int local_row_blocks = M_local / fused_globals::ROW_BLOCK;
    const int total_inter_tiles = local_row_blocks * col_blocks;

    int num_comp = std::max(1, num_comp_sms);
    int num_intra = std::max(0, num_intra_comm);
    int num_send = std::max(0, num_send_sms);
    int num_reduce = std::max(0, num_reduce_sms);
    const bool intra_only_debug = (num_send == 0 && num_reduce == 0);
    // When compute directly performs the intra-RS peer store_add into staging,
    // keep the total CTA budget unchanged but collapse the scheduler to
    // 3 logical bands: (compute+intra), send, reduce. This avoids leaving a
    // hollow intra CTA partition in the persistent kernel.
    // Compute CTAs already issue the peer store_add into staging (fused
    // intra-RS), so dedicated intra CTAs are vestigial in both production and
    // intra_only_debug. Fold them into compute unconditionally.
    num_comp += num_intra;
    num_intra = 0;
    int total_ctas = num_comp + num_intra + num_send + num_reduce;
    if (!intra_only_debug && total_ctas > config::NUM_BLOCKS) {
        int overflow = total_ctas - config::NUM_BLOCKS;
        if (num_comp > overflow) {
            num_comp -= overflow;
        } else {
            num_comp = std::max(1, config::NUM_BLOCKS - (num_intra + num_send + num_reduce));
            int remaining = config::NUM_BLOCKS - (num_comp + num_intra + num_send + num_reduce);
            if (remaining < 0) {
                num_reduce = std::max(1, num_reduce + remaining);
            }
        }
    } else if (!intra_only_debug && total_ctas < config::NUM_BLOCKS) {
        num_comp += config::NUM_BLOCKS - total_ctas;
    }

    // Iter15 EXPLORATION (LAUNCH_SETUP axis REFINE): shape-gate the fused
    // zero-kernel. Iter14's GEMM_RS_FUSED_RESET=1 won M=2K by +10.2% but regressed
    // M=8K by -2.1% (above the ±1.1% floor). Hypothesis: the fused-zero
    // kernel has a fixed ~8-12 µs grid-dispatch cost that beats 9 serial
    // memsets when per-region byte counts are tiny (small M) but loses when
    // the driver's fast-path memset + stream coalescing is faster. Gate:
    // compute `total_chunks` up-front, then at runtime choose fused kernel
    // when `total_chunks < 16` (covers M=2K and M=4K; M=8K=16 falls to
    // memset path). Both paths are compiled in under
    // GEMM_RS_FUSED_RESET_SHAPE_GATE so the selector is a single runtime branch.
    //
    // - GEMM_RS_FUSED_RESET=0 (no gate flag): 9 memsets always, byte-identical to run_013.
    // - GEMM_RS_FUSED_RESET=1, GEMM_RS_FUSED_RESET_SHAPE_GATE unset: iter14 behavior (always fused).
    // - GEMM_RS_FUSED_RESET=1, GEMM_RS_FUSED_RESET_SHAPE_GATE=1: small M -> fused; M>=8K -> memsets.
    //
    // First compute chunk_tiles_ct / chunks_per_row / total_chunks early
    // (moved up from below so the shape gate can reference them). These are
    // used below for both allocation and memset/fused branching.
    int chunk_tiles_override = 0;
    if (const char* e = std::getenv("GEMM_RS_RDMA_CHUNK_TILES_RT")) {
        chunk_tiles_override = std::atoi(e);
    }
    const int chunk_tiles_ct = chunk_tiles_override > 0
        ? min(chunk_tiles_override, col_blocks)
        : min((int)fused_globals::CHUNK_TILES, col_blocks);
    const int chunks_per_row = (col_blocks + chunk_tiles_ct - 1) / chunk_tiles_ct;
    const int total_chunks = local_row_blocks * chunks_per_row;
    const int total_global_chunks = row_blocks * chunks_per_row;

    // Selector for the per-iter zeroing of the 9 scratch regions. Two paths
    // are compiled in (see if/else below):
    //   true  → single fused_zero_kernel launch (~8-12 µs grid-dispatch
    //           overhead); cheaper at small total_chunks where launch
    //           overhead beats memset coalescing.
    //   false → 9 cudaMemsetAsync calls (driver fast-path; coalesces well
    //           at large total_chunks).
    //
    // The iter15 hypothesis was that flipping to memset at total_chunks>=16
    // would reclaim ~17 µs at M=8K. Empirically (3-rep best-of vs 4-rep
    // best-of of the always-fused path on the current cluster: 0.813 vs
    // 0.780 ms at M=8K, both > NCCL's 0.737 ms), the gate was net 30 µs
    // worse — likely because the proxy/EFA setup path that follows blocks
    // on the same stream and the 9 separate memsets serialize behind it.
    //
    // Default to always-fused. The else-branch is kept to make this a
    // single-line knob if a future cluster characterizes differently.
    const bool use_fused_reset = true;

    // Iter24 EXPLORATION (LAYOUT_STAGING + PRODUCER_STORE axes refinement):
    // Host-side shape gate for the intra-RS dual-write (iter23 / run_023 CW).
    // Empirical chunk counts (chunk_tiles_ct set per-shape by Python caller):
    //   M=2K  -> total_chunks=2,   gate OFF (revert to run_017 pack path;
    //            run_023 at this shape was 0.614 vs run_017's 0.659 = -6.9%)
    //   M=4K  -> total_chunks=8,   gate ON (run_023 = +6.5% above 5.7% floor)
    //   M=8K  -> total_chunks=16,  gate ON
    //   M=16K -> total_chunks=32,  gate ON
    //   M=32K -> total_chunks=128, gate ON
    // Default threshold 4 fires the gate-OFF path ONLY at M=2K. Bumping via
    // GEMM_RS_INTRA_RS_DUAL_WRITE_GATE=8 would also gate OFF at M=4K (not
    // recommended — M=4K is a confirmed run_023 win).
    // CRITICAL: the dual-write path only populates staging_buf inside
    // gemm_rs_pipelined_rs_tile, which is ONLY called under GEMM_RS_MULTIMEM_RS. When
    // MMRS is off (the default build), enabling the gate here would cause
    // the sender to skip its pack (which normally populates staging from
    // output_local) while nothing else populates staging — shipping zeros
    // over RDMA. Found via --check-rs: the inter-node reducer read zeros
    // from recv_buf on both nodes, leaving output at the local 8-way sum
    // and fro_err = 70% vs the 16-way torch reference. Force the gate off
    // unconditionally when MMRS is disabled.
    const bool use_intra_rs_dual_write_rt = false;

    if (g_sender_done_ntiles[dev_idx] < total_inter_tiles) {
        if (g_sender_done[dev_idx] != nullptr) {
            cudaFree(g_sender_done[dev_idx]);
            g_sender_done[dev_idx] = nullptr;
        }
        cudaMalloc(&g_sender_done[dev_idx], total_inter_tiles * sizeof(unsigned int));
        g_sender_done_ntiles[dev_idx] = total_inter_tiles;
    }
    if (g_inter_ready_ntiles[dev_idx] < total_inter_tiles) {
        if (g_inter_ready[dev_idx] != nullptr) {
            cudaFree(g_inter_ready[dev_idx]);
            g_inter_ready[dev_idx] = nullptr;
        }
        cudaMalloc(&g_inter_ready[dev_idx], total_inter_tiles * sizeof(int));
        g_inter_ready_ntiles[dev_idx] = total_inter_tiles;
    }

    // 6 counters: next_compute, next_comm, kernel_done, next_send, next_reduce, chunks_processed
    if (g_counters[dev_idx] == nullptr) {
        cudaMalloc(&g_counters[dev_idx], 6 * sizeof(unsigned int));
    }

    // Per-chunk completion tracking: chunk_tiles_done
    // Runtime override of CHUNK_TILES: picked per-shape by the Python caller
    // via GEMM_RS_RDMA_CHUNK_TILES_RT env. Profile-driven optima on 2x8 H200 EFA:
    // M=2K/4K: CT=4 (default), M=8K: CT=8, M=16K/32K: CT=16.
    if (g_chunk_alloc_nchunks[dev_idx] < total_chunks) {
        if (g_chunk_tiles_done[dev_idx] != nullptr) cudaFree(g_chunk_tiles_done[dev_idx]);
        cudaMalloc(&g_chunk_tiles_done[dev_idx], total_chunks * sizeof(uint32_t));
        g_chunk_alloc_nchunks[dev_idx] = total_chunks;
    }

    // Chunk barrier: comp_chunk_tiles_done (global row indexing, all slices).
    if (g_comp_chunk_alloc[dev_idx] < total_global_chunks) {
        if (g_comp_chunk_tiles_done[dev_idx] != nullptr) cudaFree(g_comp_chunk_tiles_done[dev_idx]);
        cudaMalloc(&g_comp_chunk_tiles_done[dev_idx], total_global_chunks * sizeof(uint32_t));
        g_comp_chunk_alloc[dev_idx] = total_global_chunks;
    }

    if (g_comp_gemm_done_alloc[dev_idx] < total_global_chunks) {
        if (g_comp_gemm_done[dev_idx] != nullptr) cudaFree(g_comp_gemm_done[dev_idx]);
        cudaMalloc(&g_comp_gemm_done[dev_idx], total_global_chunks * sizeof(uint32_t));
        g_comp_gemm_done_alloc[dev_idx] = total_global_chunks;
    }

    if (g_chunk_sendable_alloc[dev_idx] < total_global_chunks) {
        if (g_chunk_sendable[dev_idx] != nullptr) cudaFree(g_chunk_sendable[dev_idx]);
        cudaMalloc(&g_chunk_sendable[dev_idx], total_global_chunks * sizeof(uint32_t));
        g_chunk_sendable_alloc[dev_idx] = total_global_chunks;
    }

    // GEMM_RS_HELP_SEND: per-chunk claim bitmap (SFR senders vs intra-RS donor
    // helpers arbitrate via atomicCAS here). Sized to total_global_chunks;
    // helpers only claim local chunks (indexed by global_row * chunks_per_row
    // + ci), so the allocation covers exactly what the flag path uses.
    if (g_chunk_send_claimed_alloc[dev_idx] < total_global_chunks) {
        if (g_chunk_send_claimed[dev_idx] != nullptr) cudaFree(g_chunk_send_claimed[dev_idx]);
        cudaMalloc(&g_chunk_send_claimed[dev_idx], total_global_chunks * sizeof(uint32_t));
        g_chunk_send_claimed_alloc[dev_idx] = total_global_chunks;
    }

    // Arrival queue: remote_arrived_flag per local chunk.
    if (g_remote_arrived_alloc[dev_idx] < total_chunks) {
        if (g_remote_arrived_flag[dev_idx] != nullptr) cudaFree(g_remote_arrived_flag[dev_idx]);
        cudaMalloc(&g_remote_arrived_flag[dev_idx], total_chunks * sizeof(uint32_t));
        g_remote_arrived_alloc[dev_idx] = total_chunks;
    }


    // Iter14 EXPLORATION (LAUNCH_SETUP axis): single fused-zero kernel
    // replaces the 9 per-launch cudaMemsetAsync calls above. One launch
    // serializes on the stream instead of 9, saving ~20-30 µs of per-call
    // driver overhead at every iteration. Targets small-M (M=2K/4K) where
    // the fused kernel's 327 µs tail is 130 µs slower than NCCL's 195 µs.
    // Iter15: under GEMM_RS_FUSED_RESET_SHAPE_GATE, only fire at small M.
    if (use_fused_reset) {
        gemm_rs_zero_regions_t regs{};
        regs.n = 0;
        auto add = [&](void* p, size_t b) {
            if (regs.n < GEMM_RS_FUSED_RESET_MAX_REGIONS && b > 0) {
                regs.ptrs[regs.n] = p;
                regs.bytes[regs.n] = b;
                ++regs.n;
            }
        };
        add(g_sender_done[dev_idx], (size_t)total_inter_tiles * sizeof(unsigned int));
        add(g_inter_ready[dev_idx], (size_t)total_inter_tiles * sizeof(int));
        add(g_counters[dev_idx], 6 * sizeof(unsigned int));
        add(g_chunk_tiles_done[dev_idx], (size_t)total_chunks * sizeof(uint32_t));
        add(g_comp_chunk_tiles_done[dev_idx], (size_t)total_global_chunks * sizeof(uint32_t));
        add(g_comp_gemm_done[dev_idx], (size_t)total_global_chunks * sizeof(uint32_t));
        add(g_chunk_sendable[dev_idx], (size_t)total_global_chunks * sizeof(uint32_t));
        add(g_chunk_send_claimed[dev_idx], (size_t)total_global_chunks * sizeof(uint32_t));
        add(g_remote_arrived_flag[dev_idx], (size_t)total_chunks * sizeof(uint32_t));
        // Grid = one block per region (up to 12); 128 threads/block, strided.
        if (regs.n > 0) {
            gemm_rs_fused_zero_kernel<<<dim3((unsigned)regs.n, 1, 1),
                                   dim3(128, 1, 1), 0, stream>>>(regs);
        }
    } else {
        // Memset fallback. Same regions as the fused-zero path; using the
        // driver's fast-path memset (which coalesces and overlaps stream
        // ops) is cheaper than the fused launch at large total_chunks.
        // The shape-gated selector chooses memset when total_chunks >= 16
        // (i.e. M >= 8K) per the iter15 finding documented above.
        cudaMemsetAsync(g_sender_done[dev_idx], 0,
            (size_t)total_inter_tiles * sizeof(unsigned int), stream);
        cudaMemsetAsync(g_inter_ready[dev_idx], 0,
            (size_t)total_inter_tiles * sizeof(int), stream);
        cudaMemsetAsync(g_counters[dev_idx], 0, 6 * sizeof(unsigned int), stream);
        cudaMemsetAsync(g_chunk_tiles_done[dev_idx], 0,
            (size_t)total_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_comp_chunk_tiles_done[dev_idx], 0,
            (size_t)total_global_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_comp_gemm_done[dev_idx], 0,
            (size_t)total_global_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_chunk_sendable[dev_idx], 0,
            (size_t)total_global_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_chunk_send_claimed[dev_idx], 0,
            (size_t)total_global_chunks * sizeof(uint32_t), stream);
        cudaMemsetAsync(g_remote_arrived_flag[dev_idx], 0,
            (size_t)total_chunks * sizeof(uint32_t), stream);
    }

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
        int nqps = 4;
        if (const char* e = std::getenv("OSGC_EFA_NUM_QPS")) {
            int v = std::atoi(e);
            if (v > 0) nqps = v;
        }
        fifo_bundle = internode::make_fifo_bundle(fd, nqps, 1);
    }

    // GEMM_RS_READY_VIA_MULTIMEM: build the ready_chunk PGL value from the Python
    // DistBuffer and stage it to device memory so the kernel can read
    // a stable pointer via runtime_state. Cheap to rebuild every launch (the
    // multicast ptr + per-dev ptrs come from the DistBuffer).
    intra_globals::ready_chunk_pgl *ready_chunk_dev_ptr = nullptr;
    if (!intra_only_debug) {
        if (g_ready_chunk_pgl_dev[dev_idx] == nullptr) {
            cudaMalloc(&g_ready_chunk_pgl_dev[dev_idx],
                       sizeof(intra_globals::ready_chunk_pgl));
        }
        auto rc_pgl = ::dist::dbuf_from_buffer<intra_globals::ready_chunk_pgl>(
            ready_chunk, 1, 1, 1, (int)ready_chunk.data_.numel());
        cudaMemcpyAsync(g_ready_chunk_pgl_dev[dev_idx], &rc_pgl,
                        sizeof(rc_pgl), cudaMemcpyHostToDevice, stream);
        ready_chunk_dev_ptr = g_ready_chunk_pgl_dev[dev_idx];
    }

    fused_globals::runtime_state *rt_ptr = nullptr;
    if (!intra_only_debug) {
        if (g_fused_runtime[dev_idx] == nullptr) {
            cudaMalloc(&g_fused_runtime[dev_idx], sizeof(fused_globals::runtime_state));
        }
        fused_globals::runtime_state rt{
            .inter_ready = g_inter_ready[dev_idx],
            .output_local = reinterpret_cast<bf16*>(output.data_.data_ptr()),
            .recv_buf = reinterpret_cast<bf16*>(recv_buf_ptr),
            .staging_buf = reinterpret_cast<bf16*>(staging_buf_ptr),
            .M_local = M_local,
            .N = N,
            .node_idx = node_idx,
            .num_nodes = num_nodes,
            .sender_done = g_sender_done[dev_idx],
            .d2h_fifos = fifo_bundle,
            .arrival_flags = reinterpret_cast<volatile uint32_t*>(arrival_flags_ptr),
            .epoch = (uint32_t)epoch,
            .total_tiles = total_inter_tiles,
            .next_send = &g_counters[dev_idx][3],
            .next_reduce = &g_counters[dev_idx][4],
            .chunks_processed = &g_counters[dev_idx][5],
            .chunk_tiles_done = g_chunk_tiles_done[dev_idx],
            .comp_chunk_tiles_done = g_comp_chunk_tiles_done[dev_idx],
            .comp_gemm_done = g_comp_gemm_done[dev_idx],
            .chunk_sendable = g_chunk_sendable[dev_idx],
            .chunk_send_claimed = g_chunk_send_claimed[dev_idx],
            .ready_chunk = ready_chunk_dev_ptr,
            .remote_arrived_flag = g_remote_arrived_flag[dev_idx],
            .chunks_per_row = chunks_per_row,
            .chunk_tiles_val = chunk_tiles_ct,
            .col_blocks_val = col_blocks,
            .row_blocks_per_slice = local_row_blocks,
            .use_acquire_poll = (uint8_t)(use_acquire_poll != 0 ? 1u : 0u),
            .use_intra_rs_dual_write = (uint8_t)(use_intra_rs_dual_write_rt ? 1u : 0u),
            ._pad0 = {0, 0},
            .reduce_poll_sleep_ns = (uint32_t)(reduce_poll_sleep_ns > 0 ? reduce_poll_sleep_ns : 100),
        };
        cudaMemcpyAsync(g_fused_runtime[dev_idx], &rt, sizeof(rt),
                        cudaMemcpyHostToDevice, stream);
        rt_ptr = g_fused_runtime[dev_idx];
    }

    // Build the chunk-major view of staging. Python allocates a
    // DistBuffer with shape (m_local, n); we view the bytes as
    // (total_inter_tiles*128, 256) so tma::store_add_async with C_tile
    // (64, 256) and tile coord (2*global_tile_idx+i, 0) lands at chunk-major
    // offset (global_tile_idx*128 + i*64)*256 elements from the base.
    TORCH_CHECK(!staging_obj.is_none(),
        "GEMM_RS_INTRA_RS_DIRECT_STAGING=1 requires a staging DistBuffer; "
        "Python caller passed None");
    auto &staging = staging_obj.cast<dist::ParallelBuffer &>();
    const int staging_rows = total_inter_tiles * 128;
    auto staging_pgl_built =
        ::dist::dbuf_from_buffer<intra_globals::output_pgl>(
            staging, 1, 1, staging_rows, intra_globals::COL_BLOCK);

    intra_globals intra{
        .A = ::dist::gl_from_tensor<intra_globals::A_gl>(A),
        .B = ::dist::gl_from_tensor<intra_globals::B_gl>(B),
        .workspace = ::dist::dbuf_from_buffer<intra_globals::workspace_pgl>(workspace),
        .output = ::dist::dbuf_from_buffer<intra_globals::output_pgl>(output),
        .staging = staging_pgl_built,
        .barrier = ::dist::dbuf_from_buffer<intra_globals::barrier_pgl>(barrier),
        .ready = ready.data_ptr<int>(),
        .dev_idx = dev_idx,
        .num_comm_sms = num_intra,
        .num_comp_sms = num_comp,
        .next_compute = &g_counters[dev_idx][0],
        .next_comm = &g_counters[dev_idx][1],
        .kernel_done = &g_counters[dev_idx][2],
    };

    fused_globals G{
        .intra = intra,
        .num_send_sms = num_send,
        .num_reduce_sms = num_reduce,
        .rt = rt_ptr,
    };



    (void)stream;
    if (intra_only_debug) {
        launch_fused_gemm_rs(G, (unsigned int)num_comp);
    } else {
        launch_fused_gemm_rs(G, 0);
    }


}

}  // namespace gemm_rs_multinode

