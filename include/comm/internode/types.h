/**
 * @file types.h
 * @brief Shared data types for inter-node communication.
 *
 * No CUDA runtime dependency — safe to include from both nvcc and gcc/g++.
 * Defines the TransferCmd (GPU→CPU command) and ConnectionInfo (TCP bootstrap).
 */
#pragma once

#include <cstdint>

namespace internode {

// ---------------------------------------------------------------------------
// Transfer command: GPU comm CTA pushes this to the D2H FIFO.
// 32 bytes total. The FIFO publishes bytes [8, 32) first, then commits the
// first 8-byte header with a release store so the host can acquire on cmd_type.
// ---------------------------------------------------------------------------

enum class CmdType : uint8_t {
    EMPTY = 0,           // sentinel for unused FIFO slots
    WRITE = 1,           // RDMA write: send tile data to remote node
    FENCE = 2,           // ordering fence (reserved for future use)
    BARRIER_NOTIFY = 3,  // GPU-initiated cross-node barrier: proxy posts RDMA
                         // write of epoch to remote stage_barrier slot
};

#pragma pack(push, 1)
struct TransferCmd {
    CmdType  cmd_type;       // WRITE or FENCE
    uint8_t  dst_rank;       // target node rank (0/1 in the 2-node setup)
    uint16_t tile_id;        // first tile of this transfer (arrival metadata payload base)
    uint32_t bytes;          // transfer size in bytes
    uint32_t local_offset;   // byte offset into RDMA-registered local buffer
    uint32_t remote_offset;  // byte offset into RDMA-registered remote buffer
    uint16_t lane_id;        // logical lane / remote queue for structural routing
    uint8_t  src_view;       // 0 = staging buffer, 1 = C_local direct (DMA-BUF), 2 = C_local strided (multi-SGE gather)
    uint8_t  reserved0;      // keep command size aligned to 8-byte chunks
    uint16_t row_span;       // src_view=2: bytes per row run (dst_cols * sizeof(bf16))
    uint16_t row_count;      // src_view=2: number of rows gathered (<= ROW_BLOCK)
    uint64_t enqueue_device_ns; // GPU globaltimer timestamp just before fifo.push()
};
#pragma pack(pop)

static_assert(sizeof(TransferCmd) == 32, "TransferCmd must be exactly 32 bytes");

__host__ __device__ inline uint8_t unpack_dst_rank(uint8_t packed) {
    return packed;
}

// Arrival queue payload packs the first tile ID plus the number of contiguous
// tiles covered by one RDMA transfer. Zero remains the "not ready" sentinel.
__host__ __device__ inline uint32_t pack_arrival_work(uint32_t first_tile_id, uint32_t num_tiles) {
    if (num_tiles == 0) num_tiles = 1;
    return (uint32_t)((first_tile_id + 1u) | ((num_tiles - 1u) << 24));
}

__host__ __device__ inline uint32_t unpack_arrival_first_tile(uint32_t packed) {
    return (packed & 0x00FFFFFFu) - 1u;
}

__host__ __device__ inline uint32_t unpack_arrival_num_tiles(uint32_t packed) {
    return ((packed >> 24) & 0xFFu) + 1u;
}

// ---------------------------------------------------------------------------
// RDMA connection info: exchanged over TCP to establish RC QP.
// ---------------------------------------------------------------------------

static constexpr int kMaxExchangeQPs = 24;
static constexpr int kMaxRails = 4;

// Per-rail RDMA registration keys exchanged during TCP bootstrap.
// Rail 0 uses the primary fields in ConnectionInfo; rails 1+ use these.
struct RailExchangeInfo {
    uint8_t  gid[16];       // GID for this rail's NIC
    uint32_t data_rkey;     // rkey for remote data buffer on this rail's PD
    uint32_t flags_rkey;    // rkey for remote arrival flags on this rail's PD
    uint32_t tail_rkey;     // rkey for remote tail counters on this rail's PD
    uint32_t barrier_rkey;  // rkey for remote stage-barrier on this rail's PD
};

struct ConnectionInfo {
    uint32_t qp_num;
    uint32_t psn;          // packet sequence number
    uint16_t lid;          // local identifier (IB)
    uint8_t  gid[16];     // global identifier (IB/RoCE)

    // Remote data buffer (GPU HBM, RDMA-registered)
    uint32_t data_rkey;
    uint64_t data_addr;
    uint64_t data_len;

    // Remote arrival flags (host-pinned, RDMA-registered)
    uint32_t flags_rkey;
    uint64_t flags_addr;

    // Remote arrival tail counters (same MR as arrival flags when enabled)
    uint32_t tail_rkey;
    uint64_t tail_addr;

    // Remote stage barrier flags (host-pinned, RDMA-registered)
    uint32_t barrier_rkey;
    uint64_t barrier_addr;

    // Multi-QP: extra QP nums and PSNs (indices 1..num_qps-1)
    int      num_qps;                          // total QPs (1 = single QP)
    uint32_t extra_qp_nums[kMaxExchangeQPs - 1];
    uint32_t extra_psns[kMaxExchangeQPs - 1];

    // Multi-rail: per-QP rail assignment and per-rail exchange info.
    // Rail 0 uses the primary gid / data_rkey / flags_rkey / tail_rkey /
    // barrier_rkey fields above. Rails 1+ use extra_rails[rail - 1].
    // num_rails == 0 or 1 means single-rail (legacy).
    int      num_rails;
    int      qp_to_rail[kMaxExchangeQPs];           // rail index for each QP
    RailExchangeInfo extra_rails[kMaxRails - 1];     // info for rails 1+
};

} // namespace internode
