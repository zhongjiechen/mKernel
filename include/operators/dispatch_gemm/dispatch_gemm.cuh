#pragma once

/**
 * @file moe_dispatch_gemm_multinode.cu
 * @brief Proper 2-node × 8-GPU MoE Dispatch + Group GEMM.
 *
 * Borrows the intra-node 8-GPU dispatch pattern from
 *   experiments/dynamic_sm_allocation/question3_moe_dispatch_gemm_dynamic_sm/moe_dispatch_gemm.cu
 * (pre_tokens_pgl + pull-based dispatch + per-row-block barrier counter +
 * per-expert GEMM with NUM_EXPERTS_PER_DEV experts per GPU) and adds an
 * inter-node phase that exchanges pre_tokens with the peer node so each node
 * can dispatch from the FULL 16-GPU token set.
 *
 * Layout:
 *   - 16 GPUs, 8 per node, NUM_EXPERTS_PER_DEV = NUM_EXPERTS / 16
 *   - Each GPU has its own num_local_tokens tokens (in pre_tokens DistBuffer)
 *   - After inter-node exchange, each GPU has access to:
 *       (a) Local node's 8 GPU pre_tokens via PGL (pre_tokens_pgl)
 *       (b) Peer node's 8 GPU pre_tokens via a SECOND PGL (peer_tokens_pgl)
 *           — populated by RDMA writes from peer node + a local D2D copy
 *           into the IPC-shared DistBuffer.
 *
 * Three logical phases:
 *   Phase 1 (inter-node exchange):
 *     RDMA send pre_tokens to peer node's same-index GPU.
 *     Peer receives into a regular cudaMalloc'd recv_buf (RDMA-registered).
 *   Phase 2 (intra-node copy + dispatch):
 *     Each GPU copies its recv_buf into its slot of peer_tokens_pgl
 *     (DistBuffer, IPC-shared across local 8 GPUs).
 *     Then dispatch SM pulls tokens from EITHER pre_tokens_pgl OR peer_tokens_pgl
 *     based on pull_dispatch_indices (which now has 3 columns: src_node, src_dev, src_token).
 *   Phase 3 (group GEMM):
 *     Same as intranode kernel — each GPU computes GEMMs for its assigned experts.
 *
 * Hot path is a single fused kernel launch (`fused()`) that overlaps the
 * inter-node RDMA exchange, intra-node D2D copy, dispatch, and per-expert
 * GEMM in one grid.
 */

#include "common/types.cuh"
#include "dist/dbuf.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "dist/tma.cuh"
#include "comm/comm.cuh"
#include "comm/internode/d2h_fifo.cuh"
#include "comm/internode/arrival.cuh"
#include "comm/internode/types.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace kittens;


namespace moe_dispatch_gemm_multinode {

static constexpr int SM_COUNT = 132;
static constexpr int NUM_MAIN_THREADS = 384;
static constexpr int NUM_EPILOGUE_THREADS = 256;
static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - 1024;
// CHUNK_BYTES=512 KB on EFA: sweep showed +11.7% at 131k tokens vs 64 KB
// (8.39→7.50 ms, ~231→256 TFLOPS). Per-WR post→CQE cost on EFA SRD is high
// enough that larger chunks amortize it well. DeepEP's per-token (~14 KB) WR
// strategy only pays off with NVSHMEM IBGDA.
static constexpr int CHUNK_BYTES = 512 * 1024;

// ============================================================================
// Globals
// ============================================================================

struct globals {
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;       // GPUs per node
    static constexpr int NUM_NODES = 2;
    static constexpr int H = TK_MOE_H;
    static constexpr int I = TK_MOE_I;
    static constexpr int TOP_K = TK_MOE_TOP_K;
    static constexpr int NUM_EXPERTS = TK_MOE_NUM_EXPERTS;
    // Each GPU owns NUM_EXPERTS / (NUM_DEVICES * NUM_NODES) experts
    static constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / (NUM_DEVICES * NUM_NODES);

    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using token_vec = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = 16;

    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    // Two PGLs: local-node tokens, peer-node tokens (filled via RDMA + IPC copy)
    using pre_tokens_pgl  = dist::dbuf<dist::gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using peer_tokens_pgl = dist::dbuf<dist::gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using post_tokens_gl = dist::gl<bf16, 1, 1, -1, H, token_vec, A_tile>;
    using weights_gl = dist::gl<bf16, 1, NUM_EXPERTS_PER_DEV, H, I, B_tile>;
    using outputs_gl = dist::gl<bf16, 1, 1, -1, I, C_tile>;
    using padded_tokens_per_expert_gl = dist::gl<int, 1, 1, 1, NUM_EXPERTS>;
    // pull_dispatch_indices now has 3 columns: src_node, src_dev, src_token
    using pull_dispatch_indices_gl = dist::gl<int, 1, 1, -1, 3>;
    using barrier_pgl = dist::dbuf<dist::gl<int, -1, -1, -1, -1>, NUM_DEVICES, false>;

    pre_tokens_pgl  pre_tokens;
    peer_tokens_pgl peer_tokens;       // peer node's pre_tokens, IPC-shared after K1
    post_tokens_gl  post_tokens;
    weights_gl      weights;
    outputs_gl      outputs;
    padded_tokens_per_expert_gl padded_tokens_per_expert;
    pull_dispatch_indices_gl    pull_dispatch_indices;
    barrier_pgl     barrier;

    const int dev_idx;
    const int node_idx;
    const int num_local_tokens;        // tokens in this GPU's pre_tokens
    const int num_padded_local_tokens; // padded total this GPU dispatches to its experts
    const int num_comm_sms;
    const int num_comp_sms;
    unsigned int *kernel_done;

    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };
};

struct fused_globals {
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int NUM_NODES = 2;
    static constexpr int H = TK_MOE_H;
    static constexpr int I = TK_MOE_I;
    static constexpr int TOP_K = TK_MOE_TOP_K;
    static constexpr int NUM_EXPERTS = TK_MOE_NUM_EXPERTS;
    static constexpr int NUM_EXPERTS_PER_DEV = NUM_EXPERTS / (NUM_DEVICES * NUM_NODES);

    static constexpr int PIPELINE_STAGES = 4;
    static constexpr int SUPER_M = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;
    static constexpr int TOKENS_PER_BLOCK = 16;

    using token_vec = sv_bf<H>;
    using A_tile = st_bf<ROW_BLOCK / 2, RED_BLOCK>;
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK>;

    using pre_tokens_pgl  = dist::dbuf<dist::gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using peer_tokens_pgl = dist::dbuf<dist::gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using copy_ready_pgl  = dist::dbuf<dist::gl<int, -1, -1, -1, -1>, NUM_DEVICES, false>;
    using post_tokens_gl = dist::gl<bf16, 1, 1, -1, H, token_vec, A_tile>;
    using weights_gl = dist::gl<bf16, 1, NUM_EXPERTS_PER_DEV, H, I, B_tile>;
    using outputs_gl = dist::gl<bf16, 1, 1, -1, I, C_tile>;
    using padded_tokens_per_expert_gl = dist::gl<int, 1, 1, 1, NUM_EXPERTS>;
    using pull_dispatch_indices_gl = dist::gl<int, 1, 1, -1, 3>;
    // Per-GPU count of pure-local row_blocks at the head of each of this
    // rank's NUM_EXPERTS_PER_DEV owned experts. Under DISPATCH_LOCAL_FIRST routing
    // these are ready at t=0 (no RDMA dependency); DISPATCH_GEMM_LOCAL_FIRST lets
    // group_gemm_fused drain them in a first pass during RDMA.
    using local_rb_per_expert_gl = dist::gl<int, 1, 1, 1, NUM_EXPERTS_PER_DEV>;
    using barrier_pgl = dist::dbuf<dist::gl<int, -1, -1, -1, -1>, NUM_DEVICES, false>;
    using sync_barrier_pgl = dist::barrier_dbuf<TK_NUM_DEVICES>;


    pre_tokens_pgl pre_tokens;
    peer_tokens_pgl peer_tokens;
    copy_ready_pgl copy_ready;
    post_tokens_gl post_tokens;
    weights_gl weights;
    outputs_gl outputs;
    padded_tokens_per_expert_gl padded_tokens_per_expert;
    pull_dispatch_indices_gl pull_dispatch_indices;
    local_rb_per_expert_gl local_rb_per_expert;
    barrier_pgl barrier;
    sync_barrier_pgl sync_barrier;

    bf16 *recv_buf;
    bf16 *peer_tokens_local;
    int pre_tokens_bytes;
    int total_chunks;
    int node_idx;
    int num_nodes;  // Total node count (>= 2). N == 2 reproduces the
                    // legacy 2-node code path bit-for-bit. Scaffolding
                    // for N-node fan-out only; peer_tokens / arrival
                    // flag layouts not yet generalized.
    int dev_idx;
    int num_local_tokens;
    int num_padded_local_tokens;
    int num_send_sms;
    int num_copy_sms;
    int num_dispatch_sms;
    int num_comp_sms;

    internode::D2HFifoDeviceBundle d2h_fifos;
    volatile uint32_t *arrival_flags;
    uint32_t epoch;
    unsigned int *copy_phase_done;
    // Opt C: in-kernel cleanup counter. Each CTA atomicAdds at exit; the last
    // CTA zeroes per-row-block barriers in-kernel, replacing the separate
    // fused_cleanup_kernel<<<1, 256>>> launch that previously ran after
    // fused_kernel on the same stream.
    unsigned int *cleanup_done;

    struct pipeline_inputs { A_tile A[2]; B_tile B; };
    struct pipeline_outputs { C_tile C[2]; };

};

struct fused_cleanup_globals {
    using barrier_pgl = dist::dbuf<dist::gl<int, -1, -1, -1, -1>, TK_NUM_DEVICES, false>;
    barrier_pgl barrier;
    int dev_idx;
    int num_row_blocks;
};


// ============================================================================
// Host entrypoint
// ============================================================================

// Forward declaration: kernel body (template<bool Overlap>) lives in
// src/dispatch_gemm.cu. Launched via this thin wrapper so the .cuh entrypoint
// doesn't need the template body in scope.
void launch_fused_dispatch_gemm(const fused_globals& G, cudaStream_t stream);

static unsigned int *g_fused_copy_phase_done[globals::NUM_DEVICES] = {nullptr};
static unsigned int *g_fused_cleanup_done[globals::NUM_DEVICES] = {nullptr};

void fused(
    dist::ParallelBuffer &pre_tokens,
    dist::ParallelBuffer &peer_tokens,
    dist::ParallelBuffer &copy_ready,
    at::Tensor &post_tokens,
    at::Tensor &weights,
    at::Tensor &outputs,
    at::Tensor &padded_tokens_per_expert,
    at::Tensor &pull_dispatch_indices,
    at::Tensor &local_rb_per_expert,
    dist::ParallelBuffer &barrier,
    dist::ParallelBuffer &sync_barrier,
    int64_t recv_buf_ptr,
    int64_t fifo_triggers, int64_t fifo_head,
    int64_t fifo_tail, int64_t fifo_tail_cache, int fifo_capacity,
    int64_t arrival_flags_ptr,
    int epoch,
    int node_idx,
    int num_local_tokens,
    int num_padded_local_tokens,
    int num_send_sms,
    int num_copy_sms,
    int num_comm_sms_intra,
    int num_nodes = 2  // total node count (>= 2). N == 2 reproduces the
                       // legacy 2-node behavior bit-for-bit.
) {
    const int dev_idx = barrier.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    const int pre_tokens_bytes = num_local_tokens * globals::H * 2;
    const int total_chunks = (pre_tokens_bytes + CHUNK_BYTES - 1) / CHUNK_BYTES;
    int n_send = std::max(1, num_send_sms);
    int n_copy = std::max(1, num_copy_sms);
    int n_dispatch_req = num_comm_sms_intra;
    // run_09: at 131K only (num_local_tokens >= 8192), shift 16 dispatch CTAs
    // into compute by overriding num_comm_sms_intra from default 64 -> 48.
    // run_10 (Candidate 1, shape-aware mid-shape CTA split): at 32K global
    // (num_local_tokens==2048) and 65K global (num_local_tokens==4096), shift
    // 8 dispatch CTAs into compute (NC=56). Neighbor of run_03's NC=48 at 65K
    // (forbidden verbatim), preserving existing 131K NC=48 and untouched
    // overlap path at 8K/16K. Pure CTA-role split knob (no sync change);
    // all tiles are still claimed via atomicAdd(next_*, 1). Env vars
    // OSGC_Q3_NC_131K (default 48) and OSGC_Q3_NC_MID (default 56) preserve
    // reversibility.
    // run_11 (Candidate 5, overlap-path CTA shift at 8K + 16K + 16K copy boost):
    // at 8K global (num_local_tokens==512), dispatch_blocks = 512/16 = 32
    // covering 32 of 64 dispatch CTAs at default; the other 32 CTAs are idle.
    // Shift 32 CTAs into compute (n_dispatch=32, n_comp=84). At 16K global
    // (num_local_tokens==1024), dispatch_blocks = 1024/16 = 64 (exactly one
    // iter per CTA at default); shift 16 CTAs into compute (n_dispatch=48,
    // n_comp=60) and bump num_copy_sms 8->12 (n_copy=12 from n_dispatch, not
    // from n_send) to accelerate copy_ready fan-out which gates dispatch
    // IPC polling. Env vars OSGC_Q3_NC_8K, OSGC_Q3_NC_16K, OSGC_Q3_NCOPY_16K
    // preserve reversibility.
    if (num_local_tokens >= 8192) {
        // run_22 (Candidate 1 from run_22_analyst): extend NC=44 to 131K.
        // run_21 replicated the NC=48->44 signal at two mid-shapes using the
        // same `fused_kernel<false>` code path (32K +1.3%, 64K +1.1%). 131K
        // shares that path and CTA geometry; only `num_local_tokens` differs.
        // The +5.9% compute CTA bump (n_comp 68->72) trades 4 dispatch CTAs
        // for 4 compute CTAs; at 131K `dispatch_blocks=512` the +9% dispatch
        // work/CTA is amortized over many tiles per CTA, while compute
        // dominates absolute ms (project_q3_bandwidth_refuted: compute+dispatch
        // 62% of kernel at 131K). Pure host-side int initializer change;
        // `fused_kernel<false>` codegen is byte-identical (NC is a
        // `__grid_constant__` runtime arg). No sync/struct/template/walk-order
        // touched. OSGC_Q3_NC_131K preserves reversibility.
        int nc_131k = 44;
        if (const char* e = std::getenv("OSGC_Q3_NC_131K")) {
            int v = std::atoi(e);
            if (v > 0) nc_131k = v;
        }
        n_dispatch_req = nc_131k;
    } else if (num_local_tokens == 2048) {
        // run_14 (Candidate 1, 32K-only NC sweep): split 32K out of the joint
        // 32K/64K NC=56 branch. At 32K, dispatch_blocks=128 across NC CTAs;
        // lowering NC 56->48 yields +8 compute CTAs (60->68) and trades 0.4
        // iters/CTA dispatch slack for ~0.28 iters/CTA compute savings.
        // Pure host-side CTA role split; `fused_kernel<false>` binary is
        // byte-identical to run_11 and `fused_kernel<true>` (131K/8K/16K
        // paths) is unaffected by construction. OSGC_Q3_NC_32K overrides.
        // run_21 (Candidate 1 from run_21_analyst, stack NC=44 at 32K + 64K):
        // run_20 proved NC 48->44 at 32K gives +1.3% (0.696->0.705). Locking
        // that in by changing the host-side literal default from 48 to 44.
        // n_comp 68->72 (+5.9% compute CTAs) at the cost of +9% dispatch work
        // per CTA. Pure host-side int initializer change; `fused_kernel<false>`
        // codegen is byte-identical (NC is a `__grid_constant__` runtime arg).
        // run_26 (Candidate 1, 32K NC 44->40): push n_dispatch one step past
        // the proven NC=44 floor on the mid shape. dispatch_blocks=128 at 32K
        // -> NC=40 yields 3.20 iters/CTA, far below 131K's saturation knee of
        // ~11.6. n_comp 72->76 (+5.6%). Same (-4 dispatch / +4 compute) trade
        // that won 5x in a row. OSGC_Q3_NC_32K overrides.
        int nc_32k = 44;
        if (const char* e = std::getenv("OSGC_Q3_NC_32K")) {
            int v = std::atoi(e);
            if (v > 0) nc_32k = v;
        }
        n_dispatch_req = nc_32k;
    } else if (num_local_tokens == 4096) {
        // run_15: 64K-only NC sweep 56 -> 48. Ports 131K/32K/run_03-65K setpoint
        // now that run_14 severed the 32K/64K joint branch. Host-side only;
        // fused_kernel<false> codegen is byte-identical to run_14.
        // run_21 (Candidate 1, 64K extension): stack NC=44 at 64K on top of
        // the 32K win from run_20. 64K takes the same `fused_kernel<false>`
        // code path and the same CTA geometry as 32K; the +5.9% compute CTA
        // bump should dominate the ~9% dispatch slack (64K has
        // dispatch_blocks=256, already compute-bound per run_15 reasoning).
        // Pure host-side int initializer change; no codegen surface touched.
        // run_26 (Candidate 1, 64K NC 44->40): extend the NC axis on 64K too.
        // dispatch_blocks=256 at 64K -> NC=40 yields 6.40 iters/CTA, still
        // well below 131K's saturation knee (~11.6). n_comp 72->76 (+5.6%
        // compute CTAs). Matches run_21's stacked 32K+64K NC=44 precedent.
        // OSGC_Q3_NC_64K overrides.
        int nc_64k = 44;
        if (const char* e = std::getenv("OSGC_Q3_NC_64K")) {
            int v = std::atoi(e);
            if (v > 0) nc_64k = v;
        }
        n_dispatch_req = nc_64k;
    } else if (num_local_tokens == 512) {
        // 8K overlap: 32 of 64 dispatch CTAs are provably idle (dispatch_blocks=32).
        int nc_8k = 32;
        if (const char* e = std::getenv("OSGC_Q3_NC_8K")) {
            int v = std::atoi(e);
            if (v > 0) nc_8k = v;
        }
        n_dispatch_req = nc_8k;
        // run_27 (Candidate 1 from run_27_analyst, nsend_8k_shift): lower
        // n_send 8->4 at 8K overlap and give the 4 CTAs to compute. At 8K
        // pre_tokens_bytes = 7.3 MB (~115 chunks/epoch), RDMA is
        // latency-per-post bound not fan-out bound; 4 send CTAs keep the
        // ibverbs QP pipeline full. CTA split becomes
        // n_send=4, n_copy=8, n_dispatch=32, n_comp=88 (was 84). Pure
        // host-side int override on the 8K branch; `fused_kernel<true>`
        // codegen byte-identical (num_send_sms is a runtime arg). No
        // sync/struct/template/walk-order change. OSGC_Q3_NSEND_8K
        // preserves reversibility.
        int nsend_8k = 4;
        if (const char* e = std::getenv("OSGC_Q3_NSEND_8K")) {
            int v = std::atoi(e);
            if (v > 0) nsend_8k = v;
        }
        n_send = std::max(1, nsend_8k);
    } else if (num_local_tokens == 1024) {
        // 16K overlap: shift 16 dispatch CTAs -> compute; bump num_copy_sms 8->12.
        // run_25 (Candidate 1 from run_25_analyst): port NC=44 to the overlap
        // template at 16K. NC 48->44 (-4 dispatch / +4 compute) won 3x on
        // `fused_kernel<false>` (32K run_20 +1.3%, 64K run_21 +1.1%, 131K
        // run_22 +4.2%). Overlap template uses the same `num_dispatch_sms`
        // `__grid_constant__` plumbing and same atomicAdd tile claim; the
        // trade should transfer. n_comp 60->68 at 16K (with ncopy_16k=12
        // unchanged, n_send=8 default). Pure host-side int literal;
        // no codegen surface touched. OSGC_Q3_NC_16K preserves reversibility.
        int nc_16k = 44;
        if (const char* e = std::getenv("OSGC_Q3_NC_16K")) {
            int v = std::atoi(e);
            if (v > 0) nc_16k = v;
        }
        n_dispatch_req = nc_16k;
        int ncopy_16k = 12;
        if (const char* e = std::getenv("OSGC_Q3_NCOPY_16K")) {
            int v = std::atoi(e);
            if (v > 0) ncopy_16k = v;
        }
        n_copy = std::max(1, ncopy_16k);
    }
    int n_dispatch = std::max(1, n_dispatch_req);
    if (n_send + n_copy + n_dispatch >= SM_COUNT)
        n_dispatch = std::max(1, SM_COUNT - n_send - n_copy - 1);
    int n_comp = std::max(1, SM_COUNT - n_send - n_copy - n_dispatch);

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

    if (g_fused_copy_phase_done[dev_idx] == nullptr) {
        cudaMalloc(&g_fused_copy_phase_done[dev_idx], sizeof(unsigned int));
    }
    cudaMemsetAsync(g_fused_copy_phase_done[dev_idx], 0, sizeof(unsigned int), stream);
    if (g_fused_cleanup_done[dev_idx] == nullptr) {
        cudaMalloc(&g_fused_cleanup_done[dev_idx], sizeof(unsigned int));
        cudaMemsetAsync(g_fused_cleanup_done[dev_idx], 0, sizeof(unsigned int), stream);
    }
    // No per-iter memset needed: the kernel resets the counter to 0 on the
    // last-CTA path before exit (atomicExch). Saves a small launch.

    fused_globals GF{
        .pre_tokens = ::dist::dbuf_from_buffer<fused_globals::pre_tokens_pgl>(pre_tokens),
        .peer_tokens = ::dist::dbuf_from_buffer<fused_globals::peer_tokens_pgl>(peer_tokens),
        .copy_ready = ::dist::dbuf_from_buffer<fused_globals::copy_ready_pgl>(copy_ready),
        .post_tokens = ::dist::gl_from_tensor<fused_globals::post_tokens_gl>(post_tokens),
        .weights = ::dist::gl_from_tensor<fused_globals::weights_gl>(weights),
        .outputs = ::dist::gl_from_tensor<fused_globals::outputs_gl>(outputs),
        .padded_tokens_per_expert =
            ::dist::gl_from_tensor<fused_globals::padded_tokens_per_expert_gl>(padded_tokens_per_expert),
        .pull_dispatch_indices =
            ::dist::gl_from_tensor<fused_globals::pull_dispatch_indices_gl>(pull_dispatch_indices),
        .local_rb_per_expert =
            ::dist::gl_from_tensor<fused_globals::local_rb_per_expert_gl>(local_rb_per_expert),
        .barrier = ::dist::dbuf_from_buffer<fused_globals::barrier_pgl>(barrier),
        .sync_barrier =
            ::dist::dbuf_from_buffer<fused_globals::sync_barrier_pgl>(sync_barrier),
        .recv_buf = reinterpret_cast<bf16*>(recv_buf_ptr),
        .peer_tokens_local = reinterpret_cast<bf16*>(peer_tokens.data_.data_ptr()),
        .pre_tokens_bytes = pre_tokens_bytes,
        .total_chunks = total_chunks,
        .node_idx = node_idx,
        .num_nodes = num_nodes,
        .dev_idx = dev_idx,
        .num_local_tokens = num_local_tokens,
        .num_padded_local_tokens = num_padded_local_tokens,
        .num_send_sms = n_send,
        .num_copy_sms = n_copy,
        .num_dispatch_sms = n_dispatch,
        .num_comp_sms = n_comp,
        .d2h_fifos = fifo_bundle,
        .arrival_flags = reinterpret_cast<volatile uint32_t*>(arrival_flags_ptr),
        .epoch = (uint32_t)epoch,
        .copy_phase_done = g_fused_copy_phase_done[dev_idx],
        .cleanup_done = g_fused_cleanup_done[dev_idx],
    };

    // Dispatch-side local-first walk: dispatch fills local row_blocks during
    // Phase 1's RDMA wait, then peer row_blocks. Compute Pass 0 streams
    // through them while Inter-Recv is still ferrying chunks. Per-chunk
    // copy_ready polling inside dispatch_fused gates each peer token.
    launch_fused_dispatch_gemm(GF, stream);

    // Opt C: per-row-block barrier zeroing now happens inside fused_kernel's
    // last-CTA exit path (see fused_kernel above). The separate
    // fused_cleanup_kernel<<<1, 256>>>(GC) launch is removed — its launch +
    // serialization tail accounted for the 5.45 → 6.14 ms gap at 131k.

}

}  // namespace moe_dispatch_gemm_multinode
