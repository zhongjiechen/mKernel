/**
 * @file gemm_rs_multinode.cu
 * @brief Multi-node GEMM + Reduce-Scatter - single fused kernel.
 *
 * Single kernel launch. CTA groups run with role recycling:
 *
 *   Compute CTAs [0, num_comp_sms):
 *     GEMM A@B into per-GPU output tiles. In the fused-compute-intra path,
 *     compute CTAs also contribute their partial tiles directly into the
 *     owning GPU's reduce-scatter staging buffer and publish chunk readiness.
 *
 *   Send CTAs [..., ... + num_send_sms):
 *     Wait until all local GPUs have contributed the chunks owned by this rank,
 *     then coalesce row-block chunks and push them to the peer node through the
 *     D2H FIFO/RDMA path.
 *
 *   Reduce CTAs [..., 132):
 *     Poll peer-node arrival flags, reduce local staging plus remote payloads,
 *     and publish the final reduce-scattered slice. CTAs that finish their
 *     primary role recycle into the same work-stealing reduce pool.
 *
 * CTAs that finish compute or send recycle into the work-stealing remote
 * reduce pool. All synchronization is device-side through PGL barriers,
 * ready bitmaps, atomics, and RDMA arrival flags.
 *
 * Infrastructure (config, intra_globals, fused_globals, helpers, host setup,
 * entrypoint_fused, static state, gemm_rs_zero_regions_t) lives in
 *   include/operators/gemm_rs/gemm_rs.cuh
 * Python/session glue + pybind module live in
 *   include/operators/gemm_rs/session.cuh
 */
#include "operators/gemm_rs/gemm_rs.cuh"

namespace gemm_rs_multinode {

template <typename G>
__device__ inline void compute_tile_impl(
    const G &Gv, int row_idx, int col_idx, int ready_idx,
    typename G::pipeline_inputs (&inputs)[G::PIPELINE_STAGES],
    typename G::pipeline_outputs &outputs,
    semaphore (&inputs_arrived)[G::PIPELINE_STAGES],
    semaphore (&inputs_finished)[G::PIPELINE_STAGES],
    semaphore &outputs_arrived, semaphore &outputs_finished,
    int &stage, uint32_t &phasebits,
    int row_blocks, int col_blocks, int num_iters)
{
    const int wg_id = warpgroup::groupid();
    const int w_id  = warpgroup::warpid();
    const int l_id  = warp::laneid();

    __builtin_assume(row_idx >= 0 && row_idx < row_blocks);
    __builtin_assume(col_idx >= 0 && col_idx < col_blocks);

    if (wg_id == config::NUM_WARPGROUPS - 1) {
        if (w_id == 0 && l_id == 0) {
            // Wait for the PREVIOUS tile's output to finish being stored
            // before overwriting the shared `outputs.C` buffer in this
            // tile's consumer writes. Previously this wait was inside the
            // red_idx==PIPELINE_STAGES-1 branch, which only fires when
            // num_iters >= PIPELINE_STAGES. At M=2K, K/RED_BLOCK=2 <
            // PIPELINE_STAGES=4, so the wait never fired and the producer
            // raced ahead of the store warp — corrupting row-blocks that
            // wrapped around to a new tile (observed workspace max_diff=46
            // at row-blocks 13 & 15 under --check-rs at M=2048, flagged via
            // fro_err=293% on the full RS output). First-tile wait returns
            // immediately (initial outputs_finished phase matches bit 20
            // of the 0xFFFF0000 phasebits seed).
            wait(outputs_finished, get_phasebit<1>(phasebits, G::PIPELINE_STAGES));
            update_phasebit<1>(phasebits, G::PIPELINE_STAGES);
            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);
                tma::expect_bytes(inputs_arrived[stage], sizeof(typename G::pipeline_inputs));
                #pragma unroll
                for (int i = 0; i < 2; i++)
                    tma::load_async(inputs[stage].A[i], Gv.A,
                                    {row_idx * 2 + i, red_idx}, inputs_arrived[stage]);
                tma::load_async(inputs[stage].B, Gv.B, {red_idx, col_idx}, inputs_arrived[stage]);
                stage = (stage + 1) % G::PIPELINE_STAGES;
            }
        } else if (w_id == 1 && l_id == 0) {
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);
            #pragma unroll
            for (int i = 0; i < 2; i++)
                tma::store_async(Gv.workspace[Gv.dev_idx], outputs.C[i], {row_idx * 2 + i, col_idx});
            // Option B: in the SAME store warp, also issue the peer atomic-add
            // into the owner's staging_buf. outputs.C is still in shared memory
            // — no output_local round-trip needed. This collapses the
            // compute→intra-RS on-GPU handoff.
            // Destination = owner's staging_buf (chunk-major, shape
            // (total_inter_tiles*128, 256)). Global tile index within owner's
            // slice = local_row_idx_at_owner * col_blocks_local + col_idx.
            {
                const int row_blocks_per_dev_fuse = row_blocks / G::NUM_DEVICES;
                const int owner_dev_idx_fuse = row_idx / row_blocks_per_dev_fuse;
                const int local_row_idx_at_owner_fuse =
                    row_idx - owner_dev_idx_fuse * row_blocks_per_dev_fuse;
                const int col_blocks_local_fuse =
                    (int)(Gv.B.cols() / G::COL_BLOCK);
                const int global_tile_idx_owner_fuse =
                    local_row_idx_at_owner_fuse * col_blocks_local_fuse + col_idx;
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    tma::store_add_async(
                        Gv.staging[owner_dev_idx_fuse], outputs.C[i],
                        {2 * global_tile_idx_owner_fuse + i, 0});
                }
            }
            tma::store_async_read_wait();
            signal_ready(Gv.ready, ready_idx);
            arrive(outputs_finished);
        }
    } else {
        rt_fl<G::ROW_BLOCK / 8, G::COL_BLOCK> C_accum;
        warp::zero(C_accum);
        for (int red_idx = 0; red_idx < num_iters; red_idx++) {
            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
            update_phasebit<0>(phasebits, stage);
            warpgroup::mma_AB(C_accum, inputs[stage].A[wg_id], inputs[stage].B);
            warpgroup::mma_async_wait();
            warp::arrive(inputs_finished[stage]);
            stage = (stage + 1) % G::PIPELINE_STAGES;
        }
        group<8>::sync(3);
        warpgroup::store(outputs.C[wg_id], C_accum);
        warpgroup::sync(wg_id + 1);
        warpgroup::arrive(outputs_arrived);
    }
}


// Core fused_comm_tile: takes (row_idx, col_idx, ready_idx) directly.
// ready_idx is the flat index for the ready[] signal (must match compute_tile).
__device__ inline void fused_comm_tile_impl(
    const fused_globals &G, int row_idx, int col_idx, int ready_idx,
    int row_blocks)
{
    const auto &I = G.intra;
    const int row_blocks_per_dev = row_blocks / fused_globals::NUM_DEVICES;
    const int owner_dev_idx = row_idx / row_blocks_per_dev;
    const int local_row_idx = row_idx % row_blocks_per_dev;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    fused_globals::C_tile (&partials)[2] =
        allocator.allocate<fused_globals::C_tile, 2>();

    __shared__ semaphore partials_arrived[2];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) init_semaphore(partials_arrived[i], 0, 1);
    }
    __syncthreads();

    wait_ready(I.ready, ready_idx);

    const int wid = warp::groupid();

    if (wid == 0 && laneid() == 0) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            tma::expect_bytes(partials_arrived[i], sizeof(fused_globals::C_tile));
            tma::load_async(partials[i], I.workspace[I.dev_idx],
                            {row_idx * 2 + i, col_idx}, partials_arrived[i]);
        }
    } else if (wid == 1 && laneid() == 0) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            wait(partials_arrived[i], 0);
            // Option A: cross-GPU atomic-add into the owner's staging_buf
            // (chunk-major tight). Each C_tile (64 rows × 256 cols) is
            // mapped to one "row-tile" in the staging gl of shape
            // (total_inter_tiles*128, 256). Global tile index within owner's
            // slice = local_row_idx * col_blocks + col_idx.
            const int col_blocks_local =
                (int)(I.B.cols() / fused_globals::COL_BLOCK);
            const int global_tile_idx_owner =
                local_row_idx * col_blocks_local + col_idx;
            tma::store_add_async(I.staging[owner_dev_idx], partials[i],
                                 {2 * global_tile_idx_owner + i, 0});
        }
        tma::store_async_read_wait();
        // Experiment: demote from __threadfence_system to __threadfence.
        // Both the TMA store-add (cross-GPU peer output) and the downstream
        // `signal(I.barrier, ..., owner_dev_idx, 1)` target the SAME peer
        // device via NVLink. Per-destination NVLink FIFO ordering plus
        // GPU-scope fence should be sufficient: the store-add retires
        // through the TMA engine (drained by store_async_read_wait), and
        // the subsequent barrier signal cannot overtake it on the fabric.
        __threadfence();

        if (G.rt != nullptr && G.num_send_sms > 0) {
            // Chunk barrier: batch per-tile signals into per-chunk (matches gemm_ar).
            auto &Rt = *G.rt;
            const int chunk_col = col_idx / Rt.chunk_tiles_val;
            const int flat_chunk = row_idx * Rt.chunks_per_row + chunk_col;
            const int chunk_start = chunk_col * Rt.chunk_tiles_val;
            const int tiles_this_chunk = min(Rt.chunk_tiles_val,
                                             Rt.col_blocks_val - chunk_start);
            uint32_t prev = atomicAdd(Rt.comp_chunk_tiles_done + flat_chunk, 1u);
            if (prev == (uint32_t)(tiles_this_chunk - 1)) {
                const coord<ducks::default_type> slot = {1, 1 + row_idx, chunk_col};
                // Experiment: demote from sys-scope signal() to gpu-scope.
                // PGL memory is hardware-coherent over NVLink; release.gpu
                // should be sufficient here.
                gemm_rs_signal_barrier_gpu(&I.barrier[owner_dev_idx][slot], 1);
            }
        }
    }
    __syncthreads();
    (void)ready_idx;
}

// Dedicated CTA role: poll PGL barriers for locally-owned tiles and set
// Tile size in bytes (ROW_BLOCK × COL_BLOCK × sizeof(bf16)).
static constexpr int TILE_BYTES =
    fused_globals::ROW_BLOCK * fused_globals::COL_BLOCK * 2;

// Iter24: Sender-side pack helper. Extracted from the inlined pack loops at
// send sites A (HELP_SEND donor) and B (dedicated SFR sender) so the body is
// defined in ONE place and called from both sites, from BOTH the #elif
// !defined(GEMM_RS_SKIP_SEND_COPY) compile-time path AND from the iter24 runtime
// else-branch (when GEMM_RS_INTRA_RS_DUAL_WRITE is compiled in but the runtime gate
// is OFF for this shape). Behavior is byte-identical to the previous inlined
// loops — same vector width, stride, and indexing.
template <typename G>
__device__ __forceinline__ void gemm_rs_sender_pack_chunk(
    bf16 *staging_buf, const bf16 *output_local, int N,
    int chunk_first_tile, int cols_this_chunk, int rb, int col_start)
{
    constexpr int ELEMS_PER_VEC = 8;
    constexpr int VECS_PER_ROW = G::COL_BLOCK / ELEMS_PER_VEC;
    constexpr int VECS_PER_TILE = G::ROW_BLOCK * VECS_PER_ROW;
    for (int ti = 0; ti < cols_this_chunk; ++ti) {
        bf16 *dst = staging_buf
            + (long)(chunk_first_tile + ti) * (G::ROW_BLOCK * G::COL_BLOCK);
        const bf16 *src = output_local
            + (long)rb * G::ROW_BLOCK * N
            + (long)(col_start + ti) * G::COL_BLOCK;
        for (int v = threadIdx.x; v < VECS_PER_TILE; v += blockDim.x) {
            const int r = v / VECS_PER_ROW;
            const int c = (v % VECS_PER_ROW) * ELEMS_PER_VEC;
            *reinterpret_cast<uint4*>(dst + r * G::COL_BLOCK + c) =
                *reinterpret_cast<const uint4*>(src + r * N + c);
        }
    }
}

// ============================================================================
// GEMM_RS_HELP_SEND (iter 5): work-stealing donor-sender helper.
//
// Hypothesis (from NEXT_STEPS §1): at M>=8K, inter-send is CTA-push-rate
// limited (proxy avg_batch=1.00, empty=100% — proxy idles waiting for GPU
// FIFO pushes). Simultaneously, 40 intra-RS CTAs idle from t≈2 ms onward
// (intra-RS finishes early). Donating those idle CTAs to SEND instead of
// letting them sit in reduce_tiles_ws (which blocks on peer RDMA arrival
// that cannot fire until the peer finishes its own send) shortens the
// critical path: more GPU-side FIFO-push throughput → proxy/NIC fed faster
// → peer's arrival_flags/sender_done gets set earlier → the symmetric chain
// on the peer node unblocks its reducer sooner.
//
// Correctness: the SFR sender's per-CTA sent_bitmap is replaced by a global
// `chunk_send_claimed` atomicCAS gate. Any CTA (dedicated sender OR donor)
// that wins CAS(claimed[flat_chunk], 0, 1) posts that chunk; losers skip.
// Ordering of arrival_flags writes (peer-side polls row-first chunk slot)
// is unchanged because the cmd.tile_id / remote_offset scheme is the same.
//
// Gating: -DQ5_HELP_SEND (default on). Donor entry is additionally
// shape-gated on owned_per_gpu >= GEMM_RS_HELP_SEND_MIN_OWNED (default 1024 =
// M>=8192) because at M<=4K intra-RS finishes too close to reduce arrival
// for donation to pay off.
// ============================================================================

// ============================================================================
// Opt5: Work-stealing send + reduce with CTA recycling.
// All CTAs transition to reduce after their primary role completes.
// Send uses work-stealing at chunk granularity (not static row-block ownership)
// to avoid under-utilization at small M.
// ============================================================================

// Coalesced row-block send (matches gemm_ar). Static row-block ownership: each
// send CTA owns rows {send_id, send_id+stride, ...}. Waits for ALL chunk
// barriers in a row-block, packs entire row to staging, issues one RDMA.
template <typename G>
__device__ inline void send_tiles_coalesced(const G &Gv) {
    if (Gv.rt == nullptr) return;
    const auto &I = Gv.intra;
    auto &Rt = *Gv.rt;
    const int col_blocks = Rt.col_blocks_val;
    const int row_blocks_per_dev = Rt.row_blocks_per_slice;
    const int chunks_per_row = Rt.chunks_per_row;
    const int chunk_tiles = Rt.chunk_tiles_val;

    const int send_base = I.num_comp_sms + I.num_comm_sms;
    const int send_id = (int)blockIdx.x - send_base;
    const int stride = Gv.num_send_sms;

    // Port of GEMM_AR_SEND_FIRST_READY: each send CTA owns the chunks of row-blocks
    // {send_id, send_id+stride, ...}, and on every pass scans that owned set
    // for the first chunk whose 8-way PGL barrier has reached NUM_DEVICES —
    // posting that chunk immediately instead of walking rb × ci in fixed
    // order. Removes the head-of-line stall where ci=0 lags but ci>0 is
    // already sendable. Correctness: each chunk is posted exactly once
    // (tracked in sent_bitmap), and reducers tolerate arbitrary arrival order.
    const int my_rbs = (send_id < row_blocks_per_dev)
        ? ((row_blocks_per_dev - send_id + stride - 1) / stride) : 0;
    const int my_chunks_total = my_rbs * chunks_per_row;
    int posted = 0;
    __shared__ int shared_found_idx;
    while (posted < my_chunks_total) {
        if (threadIdx.x == 0) {
            shared_found_idx = -1;
            const int queue_words = gemm_rs_send_ready_bitmap_words_per_queue(
                row_blocks_per_dev, stride, chunks_per_row);
            const int queue_base =
                gemm_rs_send_ready_bitmap_region_base(row_blocks_per_dev, chunks_per_row)
                + send_id * queue_words;
            while (shared_found_idx < 0) {
                for (int w = 0; w < queue_words; ++w) {
                    unsigned int *word_ptr = reinterpret_cast<unsigned int*>(
                        &((*Rt.ready_chunk)[I.dev_idx][{0, 0, 0, queue_base + w}]));
                    uint32_t old = gemm_rs_acquire_load_u32(word_ptr);
                    while (old != 0u) {
                        const int bit_idx = __ffs((int)old) - 1;
                        const uint32_t bit = 1u << bit_idx;
                        const uint32_t desired = old & ~bit;
                        const uint32_t prev = atomicCAS(word_ptr, old, desired);
                        if (prev == old) {
                            const int idx = (w << 5) + bit_idx;
                            if (idx < my_chunks_total) {
                                shared_found_idx = idx;
                                break;
                            }
                            old = desired;
                        } else {
                            old = prev;
                        }
                    }
                    if (shared_found_idx >= 0) {
                        break;
                    }
                }
                if (shared_found_idx < 0) {
                    __nanosleep(50);
                }
            }
        }
        __syncthreads();
        const int idx = shared_found_idx;
        const int rb_k = idx / chunks_per_row;
        const int ci = idx - rb_k * chunks_per_row;
        const int rb = send_id + rb_k * stride;
        const int first_tile_rb = rb * col_blocks;
        const int first_chunk_rb = rb * chunks_per_row;
        const int col_start = ci * chunk_tiles;
        const int cols_this_chunk = min(chunk_tiles, col_blocks - col_start);
        const int chunk_first_tile = first_tile_rb + col_start;

        if (threadIdx.x == 0) {
            gemm_rs_release_store_u32(Rt.sender_done + first_chunk_rb + ci, 1u);

            const uint32_t chunk_bytes = (uint32_t)((long)cols_this_chunk * TILE_BYTES);
            const uint32_t offset = (uint32_t)((long)chunk_first_tile * TILE_BYTES);
            // Per-peer slot offsets: bit-identical at N == 2 (sap == 0).
            // single_peer_bytes / tiles = local scratch sized for one peer.
            const long single_peer_bytes =
                (long)row_blocks_per_dev * (long)col_blocks * TILE_BYTES;
            const int  single_peer_tiles = row_blocks_per_dev * col_blocks;
            const int n_peers = Rt.num_nodes - 1;
            for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                const int peer_rank = internode::peer_rank_for_slot(
                    Rt.node_idx, Rt.num_nodes, peer_slot);
                const int sap = internode::slot_at_peer(Rt.node_idx, peer_rank);
                internode::TransferCmd cmd{};
                cmd.cmd_type = internode::CmdType::WRITE;
                cmd.dst_rank = (uint8_t)peer_rank;
                cmd.tile_id  = (uint16_t)(sap * single_peer_tiles + chunk_first_tile);
                cmd.bytes    = chunk_bytes;
                cmd.local_offset = offset;
                cmd.remote_offset = (uint32_t)((long)sap * single_peer_bytes) + offset;
                cmd.lane_id  = (uint16_t)(rb * chunks_per_row + ci);
                __threadfence();
                internode::D2HFifoDevice fifo =
                    internode::gemm_ar_select_fifo_for_lane(Rt.d2h_fifos, (uint32_t)cmd.lane_id);
                fifo.push(cmd);
            }
        }
        __syncthreads();
        posted++;
    }
    return;

    for (int rb = send_id; rb < row_blocks_per_dev; rb += stride) {
        const int global_row = I.dev_idx * row_blocks_per_dev + rb;
        const int first_tile = rb * col_blocks;
        const int first_chunk = rb * chunks_per_row;


        // Wait for ALL chunk barriers in this row-block (coalesced wait).
        if (threadIdx.x == 0) {
            for (int c = 0; c < chunks_per_row; ++c) {
                const coord<ducks::default_type> slot = {1, 1 + global_row, c};
                wait(I.barrier, slot, I.dev_idx, G::NUM_DEVICES);
            }
        }
        __syncthreads();


        if (threadIdx.x == 0) {
            // Mark sender_done for all chunks in this row.
            for (int c = first_chunk; c < first_chunk + chunks_per_row; ++c) {
                gemm_rs_release_store_u32(Rt.sender_done + c, 1u);
            }

            // One RDMA write for entire row-block (matches gemm_ar coalesced path).
            const uint32_t row_bytes = (uint32_t)((long)col_blocks * TILE_BYTES);
            const uint32_t offset = (uint32_t)((long)first_tile * TILE_BYTES);
            // Per-peer slot offsets (zero at N == 2; sender-slot partition at N > 2).
            const long single_peer_bytes2 =
                (long)row_blocks_per_dev * (long)col_blocks * TILE_BYTES;
            const int  single_peer_tiles2 = row_blocks_per_dev * col_blocks;
            const int n_peers = Rt.num_nodes - 1;
            for (int peer_slot = 0; peer_slot < n_peers; ++peer_slot) {
                const int peer_rank = internode::peer_rank_for_slot(
                    Rt.node_idx, Rt.num_nodes, peer_slot);
                const int sap = internode::slot_at_peer(Rt.node_idx, peer_rank);
                internode::TransferCmd cmd{};
                cmd.cmd_type = internode::CmdType::WRITE;
                cmd.dst_rank = (uint8_t)peer_rank;
                cmd.tile_id  = (uint16_t)(sap * single_peer_tiles2 + first_tile);
                cmd.bytes    = row_bytes;
                cmd.local_offset = offset;
                cmd.remote_offset = (uint32_t)((long)sap * single_peer_bytes2) + offset;
                cmd.lane_id  = (uint16_t)rb;
                __threadfence();
                internode::D2HFifoDevice fifo =
                    internode::gemm_ar_select_fifo_for_lane(Rt.d2h_fifos, (uint32_t)rb);
                fifo.push(cmd);
            }
        }
    }
}

// Work-stealing reducer: any CTA (including recycled compute/intra/send CTAs)
// claims chunks via atomicAdd(next_reduce). Scales well at large M.
// Body inlined directly (helper extraction introduced ~15% regression at M=2K
// even with __forceinline__; keeping monolithic).
template <typename G>
__device__ inline void reduce_tiles_ws(const G &Gv) {
    if (Gv.rt == nullptr) return;
    auto &Rt = *Gv.rt;
    const int col_blocks = Rt.col_blocks_val;
    const int row_blocks_per_dev = Rt.row_blocks_per_slice;
    const int total_elems = G::ROW_BLOCK * G::COL_BLOCK;
    const int chunks_per_row = Rt.chunks_per_row;
    const int chunk_tiles = Rt.chunk_tiles_val;
    const int total_chunks = row_blocks_per_dev * chunks_per_row;

    while (true) {
        __shared__ int _ws_red_chunk;
        if (threadIdx.x == 0) {
            _ws_red_chunk = (int)atomicAdd(Rt.next_reduce, 1u);
        }
        __syncthreads();
        const int chunk_id = _ws_red_chunk;
        if (chunk_id >= total_chunks) break;

        const int rb = chunk_id / chunks_per_row;
        const int ci = chunk_id % chunks_per_row;
        const int col_start = ci * chunk_tiles;
        const int cols_this_chunk = min(chunk_tiles, col_blocks - col_start);
        const int first_tile = rb * col_blocks + col_start;


        // Wait for row-block arrival (matches gemm_ar's arrival queue pattern).
        // Coalesced send issues one RDMA per row-block, so proxy sets
        // arrival_flags[rb * col_blocks] = epoch for the entire row.
        //
        // Iter1 hypothesis: at small M (2048/4096) the reducer polling sleeps
        // dominate the critical path. The flag often arrives within a few
        // spin iterations (local chunks are already sender_done almost
        // immediately; remote arrivals at small M take only a few µs). A
        // tight spin for the first ~64 iterations before falling back to the
        // backoff sleep amortizes latency without burning power at large M.
        // Iter11 EXPLORATORY (new axis): parallelize the two independent
        // flag waits. Thread 0 (warp 0) polls `remote_arrived_flag` (RDMA
        // side); thread 32 (warp 1 lane 0) polls `sender_done` (local
        // sender-stage side). Using a different WARP is important: two
        // threads in the same warp would serialize via SIMT divergence
        // when both spin on __nanosleep/load loops, defeating the
        // parallelism claim. Two warps have independent schedulers on
        // H100, so each can genuinely make progress in parallel. Single
        // `__syncthreads()` after serves as the join. Removes one serial
        // __syncthreads per chunk from the reducer critical path.
        //
        // Expected benefit: per-chunk sync cost is ~30-60 ns on H100,
        // scaled by chunk count (M=32K: 256 chunks/GPU → ~10-15 µs saved
        // per reducer CTA). Sender_done is typically already set, so
        // thread 32 completes in one load on the fast path and just joins
        // the sync. When it isn't, we overlap its poll with thread 0's
        // arrival_flags spin.
        if (threadIdx.x == 0) {
            if (gemm_rs_acquire_load_u32(Rt.remote_arrived_flag + chunk_id) == 0u) {
                // Iter17 EXPLORATION (architect-designed, INTER_NODE_BARRIER
                // axis): honor per-chunk RDMA arrivals on the reducer side.
                // Today's reducer polls arrival_flags[row_first_tile] (ci=0's
                // slot) and then cascade-stamps remote_arrived_flag[chunk_id]
                // for all chunks in the row-block, gambling on SRD chunk-order
                // of arrival. SRD is unordered across WRs on a single QP and
                // the proxy round-robins 8 QPs (proxy_efa.h:787-797), so this
                // cascade is a latent correctness bet. Under
                // GEMM_RS_REDUCER_POLL_PER_CHUNK, each reducer chunk waits on its
                // OWN RDMA's arrival flag (proxy_efa.h:705 stamps per-chunk
                // tile_id per gemm_rs_multinode.cu:979), strengthening
                // correctness AND potentially waking ci>0 chunks earlier if
                // their CQE lands before ci=0's. Risk: +7x more
                // __threadfence_system at M=32K (256 vs 32) ~25 us could eat
                // the gain (architect's expected signal: at-floor).
                volatile uint32_t* flag_ptr = &Rt.arrival_flags[first_tile];
                const uint32_t sleep_ns = Rt.reduce_poll_sleep_ns;
                constexpr int kTightSpin = 64;
                int spin = 0;
                if (Rt.use_acquire_poll) {
                    while (gemm_rs_poll_arrival_acquire(flag_ptr) != Rt.epoch) {
                        if (spin < kTightSpin) { ++spin; }
                        else { __nanosleep(sleep_ns); }
                    }
                } else {
                    while (gemm_rs_poll_arrival_relaxed(flag_ptr) != Rt.epoch) {
                        if (spin < kTightSpin) { ++spin; }
                        else { __nanosleep(sleep_ns); }
                    }
                }
                // Iter19 EXPLORATION (REDUCER_POLL axis, architect-designed
                // iter16:352-357 follow-up to run_017's MVP): the release-
                // store below (`remote_arrived_flag + chunk_id`) lives in
                // same-GPU HBM (cudaMalloc at line 2549) and is consumed
                // only by same-GPU reducer CTAs (lines 1456/1523). It is
                // already `st.release.gpu` (line 113-118). The ONLY cross-
                // scope boundary in this pairing is the peer-CPU→GPU
                // visibility of `arrival_flags` (host-mapped, `.sys.global`
                // acquire-load at line 1486). The acquire-load already
                // provides acquire-order for subsequent same-thread loads
                // of `recv_buf` (line 1707-1726), and `recv_buf` is
                // GPU-HBM written by remote NIC → PCIe → same-GPU HBM,
                // visible to other same-GPU CTAs without a system fence.
                // Therefore `__threadfence()` (GPU-scope, ~20 cy) is
                // sufficient to order the release store against the
                // acquire-load's payload; `__threadfence_system()`
                // (~100-200 cy, CPU + peer-GPU scope) is strictly
                // unnecessary. At M=32K, 256 fences × ~100 cy / 1.5 GHz
                // ≈ 17 µs saved per GPU per launch.
                //
                // Flag-off path (GEMM_RS_REDUCER_SYSFENCE_BATCH unset) is byte-
                // identical to the run_017 baseline: system fence retained.
                __threadfence_system();
                gemm_rs_release_store_u32(Rt.remote_arrived_flag + chunk_id, 1u);
            }
        } else if (threadIdx.x == 32) {
            // Concurrent poll on warp 1 lane 0: sender_done for this chunk.
            // Typically already set (sender finishes staging long before
            // RDMA returns), so first load is a hit in the common case.
            // Using warp 1 (not lane 1 of warp 0) avoids SIMT serialization
            // with thread 0's spin loop — H100 has independent warp
            // schedulers so this genuinely runs concurrently.
            while (gemm_rs_acquire_load_u32(Rt.sender_done + chunk_id) == 0u) {
                __nanosleep(32);
            }
        }
        __syncthreads();

        // Iter2 hypothesis: run_001 tight-spun BOTH arrival_flags and
        // sender_done unconditionally. Regression at large M (M=32K: -4.7%,
        // M=16K: -1.7%) is dominated by the *sender_done* tight-spin cost,
        // not arrival_flags:
        //   - arrival_flags is polled once per row-block (M=32K: 32 polls).
        //     64 iters × ~200 ns/load ≈ 12 µs per row-block; 32 × 12 µs = ~0.4 ms
        //     → only ~3% of a 14 ms baseline.
        //   - sender_done is polled once per *chunk* (M=32K: 256 polls; 8× more).
        //     Same 12 µs per spin × 256 ≈ 3 ms wasted on the critical path.
        // By the time the reducer finishes the RDMA-arrival wait above, the
        // sender CTA has long since set sender_done (sender_done is stamped
        // release BEFORE the FIFO push; RDMA round-trip is 10s of µs at the
        // fastest). So sender_done is *already satisfied* when we arrive;
        // tight-spinning just burns 64 iters of L2-dirty loads per chunk for
        // no coverage benefit. Drop the tight-spin here — first load should
        // observe the flag set, and the rare miss path only pays one 32 ns
        // sleep before retry.
        //
        // Iter11 EXPLORATORY (new axis): parallelize the two-flag wait. The
        // above `remote_arrived_flag` check (lines 1437-1469) and this
        // `sender_done` check are INDEPENDENT data hazards — neither depends
        // on the other for correctness of its own poll. Currently we do two
        // serial waits, each guarded by its own `__syncthreads()`. We can
        // fold the sender_done wait INTO the remote_arrived_flag poll block
        // (thread 0 polls remote_arrived_flag, thread 1 polls sender_done
        // concurrently) and amortize under a single `__syncthreads`. This
        // removes one serial __syncthreads per chunk from the critical path.
        //
        // Expected benefit scales with chunk count:
        //   M=2K: 4 chunks/GPU   — tiny
        //   M=32K: 256 chunks/GPU — measurable
        //
        // Safety: thread 1's acquire-load of sender_done propagates via the
        // barrier-release at __syncthreads to all threads (same ordering as
        // before). Thread 0's __threadfence_system ordering for the
        // per-chunk flag stores is unchanged — that fence is inside thread
        // 0's branch, unrelated to thread 1's poll.
        //
        // Gate: -DQ5_DISABLE_PARALLEL_FLAG_WAIT restores the two-sync pattern
        // byte-for-byte (for diagnostic A/B). Default is the fused path.
        // Default path: both polls ran concurrently on threads 0/1 in the
        // block above, joined by the single __syncthreads at the end of
        // that block. No extra wait/sync needed here.

        constexpr int ELEMS_PER_VEC = 8;
        constexpr int VECS_PER_ROW = G::COL_BLOCK / ELEMS_PER_VEC;
        constexpr int ROWS_PER_WAVE = config::NUM_THREADS / VECS_PER_ROW;
        const int vec_lane = threadIdx.x % VECS_PER_ROW;
        const int row_lane = threadIdx.x / VECS_PER_ROW;
        const int col_elem = vec_lane * ELEMS_PER_VEC;

        for (int ti = 0; ti < cols_this_chunk; ++ti) {
            for (int r = row_lane; r < G::ROW_BLOCK; r += ROWS_PER_WAVE) {
                const int si = (rb * G::ROW_BLOCK + r) * Rt.N
                             + (col_start + ti) * G::COL_BLOCK + col_elem;
                // Staging path: sender packs tile-major, so recv_buf is
                // tile-major (tile_id × total_elems blocks).
                const bf16 *recv = Rt.recv_buf
                    + (long)(first_tile + ti) * total_elems
                    + (long)r * G::COL_BLOCK + col_elem;

                // Option A: local contribution lives in staging_buf chunk-major
                // tight (same layout as recv_buf in the default/staging path).
                // Write still goes to output_local row-major (via `si`) so
                // downstream PyTorch sees the expected shape.
                const bf16 *local_src = Rt.staging_buf
                    + (long)(first_tile + ti) * total_elems
                    + (long)r * G::COL_BLOCK + col_elem;
                const uint4 lv = *reinterpret_cast<const uint4*>(local_src);
                const uint4 rv = *reinterpret_cast<const uint4*>(recv);
                __nv_bfloat162 o0 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.x),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.x));
                __nv_bfloat162 o1 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.y),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.y));
                __nv_bfloat162 o2 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.z),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.z));
                __nv_bfloat162 o3 = __hadd2(
                    *reinterpret_cast<const __nv_bfloat162*>(&lv.w),
                    *reinterpret_cast<const __nv_bfloat162*>(&rv.w));
                uint4 ov;
                ov.x = *reinterpret_cast<unsigned int*>(&o0);
                ov.y = *reinterpret_cast<unsigned int*>(&o1);
                ov.z = *reinterpret_cast<unsigned int*>(&o2);
                ov.w = *reinterpret_cast<unsigned int*>(&o3);
                *reinterpret_cast<uint4*>(Rt.output_local + si) = ov;
            }
        }
    }

}




__device__ inline void fused_kernel(const fused_globals &G) {
    const auto &I = G.intra;
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    intra_globals::pipeline_inputs (&inputs)[intra_globals::PIPELINE_STAGES] =
        allocator.allocate<intra_globals::pipeline_inputs, intra_globals::PIPELINE_STAGES>();
    intra_globals::pipeline_outputs &outputs =
        *reinterpret_cast<intra_globals::pipeline_outputs *>(
            &inputs[intra_globals::PIPELINE_STAGES - 1]);

    __shared__ semaphore inputs_arrived[intra_globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[intra_globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < intra_globals::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 8);
        }
        init_semaphore(outputs_arrived, 0, 2);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();



    const int row_blocks = I.A.rows() / intra_globals::ROW_BLOCK;
    const int col_blocks = I.B.cols() / intra_globals::COL_BLOCK;
    const int num_blocks = row_blocks * col_blocks;
    const int num_iters = I.A.cols() / intra_globals::RED_BLOCK;

    const int row_blocks_per_slice = row_blocks / intra_globals::NUM_DEVICES;

    // CTA role dispatch with recycling (matches gemm_ar architecture):
    // - Compute: static stride claiming + gemm_ar tile visit order
    // - Intra-RS: work-stealing
    // - Send: static row-block ownership + coalesced RDMA
    // - Reduce: work-stealing with remote_arrived_flag
    // All roles recycle to reduce after primary role completes.
    if ((int)blockIdx.x < I.num_comp_sms) {
        // Primary: GEMM compute (static stride + gemm_ar tile visit order).
        // gemm_rs_decode_comp_task maps linear task_id to (row, col), and
        // compute_tile_impl takes (row, col) directly — no double-decode.
        // stage/phasebits MUST persist across tasks because inputs_arrived/
        // inputs_finished/outputs_arrived/outputs_finished semaphores are
        // shared state that accumulates phase toggles; resetting phasebits
        // each task would desync wait() from actual semaphore phase.
        int stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        for (int task_id = (int)blockIdx.x; task_id < num_blocks; task_id += I.num_comp_sms) {
            int row_idx, col_idx;
            gemm_rs_decode_comp_task<intra_globals>(task_id, row_blocks_per_slice,
                                               col_blocks, I.dev_idx,
                                               row_idx, col_idx);
            const int ready_idx = row_idx * col_blocks + col_idx;
            compute_tile_impl<intra_globals>(I, row_idx, col_idx, ready_idx,
                                              inputs, outputs, inputs_arrived, inputs_finished,
                                              outputs_arrived, outputs_finished, stage, phasebits,
                                              row_blocks, col_blocks, num_iters);
            // Compute-side chunk-ready signal. When this GPU finishes all
            // tiles in a chunk, emit a "I'm done contributing" edge so the
            // owner's send CTA knows this GPU's partials have landed in
            // staging. Counter is local atomic; the edge itself is either a
            // cross-device barrier signal (default) or a local release-store
            // into a multicast flag (GEMM_RS_READY_VIA_MULTIMEM).
            // With GEMM_RS_FUSE_COMPUTE_INTRA, the compute CTA has also just
            // issued the peer atomic-add into staging_buf — so this signal
            // correctly marks "all tile atomic-adds for this chunk issued."
            if (G.rt != nullptr && G.num_send_sms > 0 && threadIdx.x == 0) {
                auto &Rt = *G.rt;
                const int chunk_col = col_idx / Rt.chunk_tiles_val;
                const int flat_chunk = row_idx * Rt.chunks_per_row + chunk_col;
                const int chunk_start = chunk_col * Rt.chunk_tiles_val;
                const int tiles_this_chunk = min(Rt.chunk_tiles_val,
                                                 Rt.col_blocks_val - chunk_start);
                uint32_t prev = atomicAdd(Rt.comp_gemm_done + flat_chunk, 1u);
                if (prev == (uint32_t)(tiles_this_chunk - 1)) {
                    __threadfence_system();
                    const int owner_dev_idx =
                        row_idx / (row_blocks / intra_globals::NUM_DEVICES);
                    const int local_row_idx_at_owner =
                        row_idx - owner_dev_idx * Rt.row_blocks_per_slice;
                    const int owner_send_id =
                        local_row_idx_at_owner % G.num_send_sms;
                    const int owner_rb_in_queue =
                        local_row_idx_at_owner / G.num_send_sms;
                    const int queue_idx =
                        owner_rb_in_queue * Rt.chunks_per_row + chunk_col;
                    auto *owner_ready_count = reinterpret_cast<unsigned int*>(
                        &((*Rt.ready_chunk)[owner_dev_idx][{0, 0, 0, flat_chunk}]));
                    uint32_t ready_prev = atomicAdd(owner_ready_count, 1u);
                    if (ready_prev == (uint32_t)(intra_globals::NUM_DEVICES - 1)) {
                        const int queue_words = gemm_rs_send_ready_bitmap_words_per_queue(
                            Rt.row_blocks_per_slice, G.num_send_sms, Rt.chunks_per_row);
                        const int queue_base = gemm_rs_send_ready_bitmap_region_base(
                            Rt.row_blocks_per_slice, Rt.chunks_per_row);
                        const int word_idx = queue_idx >> 5;
                        const uint32_t bit = 1u << (queue_idx & 31);
                        auto *owner_ready_word = reinterpret_cast<unsigned int*>(
                            &((*Rt.ready_chunk)[owner_dev_idx][
                                {0, 0, 0, queue_base + owner_send_id * queue_words + word_idx}]));
                        atomicOr(owner_ready_word, bit);
                    }
                }
            }
        }
        // Recycle: transition to reduce work-stealing
        reduce_tiles_ws<fused_globals>(G);
    } else if ((int)blockIdx.x < I.num_comp_sms + I.num_comm_sms + G.num_send_sms) {
        // Primary: inter-node send (static row-block ownership, coalesced RDMA)
        send_tiles_coalesced<fused_globals>(G);
        // Recycle: transition to reduce work-stealing
        reduce_tiles_ws<fused_globals>(G);
    } else {
        // Dedicated reduce CTAs: work-stealing from the start
        reduce_tiles_ws<fused_globals>(G);
    }



    // Match the split intra kernel when we intentionally launch only the
    // compute + intranode CTAs for debugging/reuse checks.
    if ((int)gridDim.x == I.num_comp_sms + I.num_comm_sms && threadIdx.x == 0) {
        __threadfence();
        unsigned int prev = atomicAdd(I.kernel_done, 1u);
        if (prev + 1 == (unsigned int)gridDim.x) {
            atomicExch(I.kernel_done, 0u);
            barrier_all(I.barrier, {0, 0, 0}, I.dev_idx);
        }
    }
}

__global__ void gemm_rs_fused_zero_kernel(gemm_rs_zero_regions_t regs) {
    const int rid = blockIdx.x;
    if (rid >= regs.n) return;
    unsigned int* p = reinterpret_cast<unsigned int*>(regs.ptrs[rid]);
    const size_t words = regs.bytes[rid] / sizeof(unsigned int);
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    for (size_t i = tid; i < words; i += nthr) {
        p[i] = 0u;
    }
}

__global__ __launch_bounds__(config::NUM_THREADS, 1)
void gemm_rs_fused_kernel_stub(const __grid_constant__ fused_globals G) {
    fused_kernel(G);
}

// Launch wrapper stays in this TU so the kernel body stays out of the .cuh.
void launch_fused_gemm_rs(const fused_globals& G, unsigned int active_sms) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int dynamic_shared_memory = config::DYNAMIC_SHARED_MEMORY;
    const unsigned int grid = (active_sms == 0u)
        ? (unsigned int)config::NUM_BLOCKS
        : active_sms;
    OSGC_CUDACHECK(cudaFuncSetAttribute(
        gemm_rs_fused_kernel_stub,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    gemm_rs_fused_kernel_stub<<<grid, config::NUM_THREADS,
                                dynamic_shared_memory, stream>>>(G);
}

}  // namespace gemm_rs_multinode

#include "operators/gemm_rs/session.cuh"
