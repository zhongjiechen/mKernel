/**
 * @file d2h_fifo.cuh
 * @brief Device-to-host command FIFO for GPU-initiated inter-node communication.
 *
 * Implements the mscclpp FIFO pattern from UCCL (ep/include/fifo_device.hpp):
 *   - triggers[]: host-pinned buffer, GPU writes fixed-size TransferCmd records
 *   - head: device memory, GPU claims slots via atomicAdd (fast, no PCIe RT)
 *   - tail: host-pinned, CPU advances after draining; GPU reads for backpressure
 *   - tail_cache: device memory, cached tail to avoid repeated host reads
 *
 * GPU comm CTAs push TransferCmd via D2HFifoDevice::push().
 * CPU proxy thread reads via D2HFifoHost::poll().
 */
#pragma once

#include "types.h"
#include "../../common/cuda_checks.cuh"

#include <cuda_runtime.h>
#include <cstring>



namespace internode {

// Under IBGDA we have at most one D2HFifoDevice per QP (= one SQ each), and
// Q2 configures up to 24 QPs via OSGC_IB_NUM_QPS=24. Bump the bundle capacity
// so the per-QP entries fit. The proxy-based IBVERBS path still only creates
// kMaxProxyThreads=8 proxy threads; extra slots stay zero.
static constexpr int kMaxProxyThreads = 24;


// ---------------------------------------------------------------------------
// Device-side handle — passed into kernel globals, GPU calls push()
// ---------------------------------------------------------------------------

struct D2HFifoDevice {
    TransferCmd* triggers;   // host-pinned buffer (GPU writes via st.release.sys)
    uint64_t*    head;       // device memory (GPU atomicAdd to claim slot)
    uint64_t*    tail;       // host-pinned (CPU writes after drain, GPU reads)
    uint64_t*    tail_cache; // device memory (cached copy of tail)
    int          capacity;   // must be power of 2

    /**
     * Push a TransferCmd to the FIFO. Called by GPU comm CTAs.
     *
     * 1. atomicAdd(head) to claim a slot (device memory — fast).
     * 2. Backpressure: spin if ring full, refreshing tail_cache from host.
     * 3. Pack cmd into 8-byte chunks, write the payload chunks first, then
     *    commit the first 8-byte header with a release store.
     *
     * Returns the slot index (for optional completion tracking).
     */
#ifdef __CUDA_ARCH__
    __device__ __forceinline__ uint64_t push(const TransferCmd& cmd) const {
        // Claim slot
        uint64_t slot = atomicAdd((unsigned long long*)head, 1ULL);

        // Backpressure: wait if ring is full
        uint64_t mask = (uint64_t)(capacity - 1);
        if (slot >= (uint64_t)capacity + *tail_cache) {
            // Refresh tail_cache from host-pinned tail
            uint64_t t;
            do {
                // Volatile load from host-pinned memory
                asm volatile("ld.volatile.global.u64 %0, [%1];"
                             : "=l"(t) : "l"(tail) : "memory");
                *tail_cache = t;
                if (slot < (uint64_t)capacity + t) break;
                __nanosleep(64);
            } while (true);
        }

        // Publish the command in 8-byte chunks. We store bytes [8, sizeof(cmd))
        // first, then commit the first 8-byte header with a release store so
        // the host's acquire load of cmd_type cannot observe a half-written
        // command body.
        static_assert(sizeof(TransferCmd) % sizeof(uint64_t) == 0);
        constexpr int kCmdWords = (int)(sizeof(TransferCmd) / sizeof(uint64_t));
        uint64_t words[kCmdWords];
        memcpy(words, &cmd, sizeof(TransferCmd));

        // Use release.gpu (cheaper than release.sys) — sys-scope release on
        // H100 forces a kernel-wide L2/HBM drain that serializes with
        // hot-spinning compute CTAs and adds ~ms per push. The store still
        // propagates to host-pinned memory through the normal PCIe write path;
        // the proxy only needs acquire ordering on cmd_type before copying the
        // rest of the record.
        TransferCmd* slot_ptr = &triggers[slot & mask];
        char* slot_bytes = reinterpret_cast<char*>(slot_ptr);
#pragma unroll
        for (int wi = 1; wi < kCmdWords; ++wi) {
            asm volatile("st.global.u64 [%0], %1;"
                         :: "l"(slot_bytes + wi * (int)sizeof(uint64_t)), "l"(words[wi])
                         : "memory");
        }
        asm volatile("st.global.release.gpu.u64 [%0], %1;"
                     :: "l"(slot_bytes), "l"(words[0]) : "memory");

        return slot;
    }
#else
    uint64_t push(const TransferCmd&) const { return 0; } // host stub
#endif
};


// Cap on per-bundle TX/RX-QP arrays under EFAGDA v2. Same upper bound as
// kMaxProxyThreads since the reference's per-CTA-QP count fits comfortably.

struct D2HFifoDeviceBundle {
    D2HFifoDevice fifos[kMaxProxyThreads];
    int           num_fifos;
    int           global_num_qps;
    int           logical_queues_per_qp;
    int           qps_per_fifo;

};

inline __host__ __device__ D2HFifoDeviceBundle make_fifo_bundle(
    const D2HFifoDevice& fifo,
    int global_num_qps,
    int logical_queues_per_qp
) {
    D2HFifoDeviceBundle bundle{};
    bundle.fifos[0] = fifo;
    bundle.num_fifos = 1;
    bundle.global_num_qps = global_num_qps > 0 ? global_num_qps : 1;
    bundle.logical_queues_per_qp =
        logical_queues_per_qp > 0 ? logical_queues_per_qp : 1;
    bundle.qps_per_fifo = bundle.global_num_qps;
    return bundle;
}

inline __host__ __device__ D2HFifoDevice gemm_ar_select_fifo_for_lane(
    const D2HFifoDeviceBundle& bundle,
    uint32_t lane_id
) {
    if (bundle.num_fifos <= 1) {
        return bundle.fifos[0];
    }
    const uint32_t global_num_qps =
        bundle.global_num_qps > 0 ? (uint32_t)bundle.global_num_qps : 1u;
    const uint32_t qps_per_fifo =
        bundle.qps_per_fifo > 0 ? (uint32_t)bundle.qps_per_fifo : global_num_qps;
    const uint32_t global_qp = lane_id % global_num_qps;
    uint32_t fifo_idx = global_qp / qps_per_fifo;
    if (fifo_idx >= (uint32_t)bundle.num_fifos) {
        fifo_idx = (uint32_t)bundle.num_fifos - 1u;
    }
    return bundle.fifos[fifo_idx];
}


// ---------------------------------------------------------------------------
// Host-side handle — used by CPU proxy thread
// (Compiled out under IBGDA / EFAGDA: no proxy thread, no FIFO drain.)
// ---------------------------------------------------------------------------

struct D2HFifoHost {
    TransferCmd* triggers;   // same host-pinned buffer as device side
    uint64_t*    tail;       // host pointer, proxy advances this
    int          capacity;
    uint64_t     cpu_head;   // proxy's read cursor (local, not shared)

    /**
     * Poll for the next available command.
     * Returns true if a command was available, fills *out.
     * Non-blocking: returns false immediately if no command ready.
     */
    bool poll(TransferCmd* out) {
        uint64_t mask = (uint64_t)(capacity - 1);
        uint64_t idx = cpu_head & mask;

        // Read the cmd_type with acquire semantics
        CmdType ct = (CmdType)__atomic_load_n(
            (uint8_t*)&triggers[idx].cmd_type, __ATOMIC_ACQUIRE);
        if (ct == CmdType::EMPTY) {
            return false;
        }

        // Full command is ready — copy it out.
        // Use acquire fence to ensure we see the complete record body that the
        // device wrote before publishing cmd_type.
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        *out = triggers[idx];

        // Clear the slot for reuse
        __atomic_store_n(
            (uint8_t*)&triggers[idx].cmd_type,
            (uint8_t)CmdType::EMPTY, __ATOMIC_RELEASE);

        cpu_head++;
        return true;
    }

    /**
     * Count how many contiguous commands are currently visible to the host from
     * the current cpu_head. This is a host-side approximation of FIFO backlog.
     */
    int count_ready(int limit) const {
        if (limit <= 0) return 0;
        const uint64_t mask = (uint64_t)(capacity - 1);
        int ready = 0;
        while (ready < limit) {
            const uint64_t idx = (cpu_head + (uint64_t)ready) & mask;
            CmdType ct = (CmdType)__atomic_load_n(
                (uint8_t*)&triggers[idx].cmd_type, __ATOMIC_ACQUIRE);
            if (ct == CmdType::EMPTY) break;
            ready++;
        }
        return ready;
    }

    /**
     * Advance the tail pointer so GPU can reuse FIFO slots.
     * Call after processing commands and draining CQ completions.
     */
    void advance_tail(uint64_t new_tail) {
        __atomic_store_n(tail, new_tail, __ATOMIC_RELEASE);
    }
};

// ---------------------------------------------------------------------------
// Combined pair + allocation / destruction
// ---------------------------------------------------------------------------

struct D2HFifoPair {
    D2HFifoDevice device;
    D2HFifoHost   host;
};

/**
 * Create a D2H FIFO pair. Allocates:
 *   - triggers[capacity] in host-pinned memory (cudaHostAlloc)
 *   - head in device memory (cudaMalloc)
 *   - tail in host-pinned memory (cudaHostAlloc)
 *   - tail_cache in device memory (cudaMalloc)
 *
 * @param capacity Must be power of 2. Default 1024.
 */
inline D2HFifoPair create_d2h_fifo(int capacity = 1024) {
    // Validate power of 2
    if (capacity <= 0 || (capacity & (capacity - 1)) != 0) {
        fprintf(stderr, "d2h_fifo: capacity must be power of 2, got %d\n", capacity);
        exit(EXIT_FAILURE);
    }

    D2HFifoPair pair{};

    // Triggers: host-pinned, zeroed (EMPTY cmd_type = 0)
    TransferCmd* triggers = nullptr;
    OSGC_CUDACHECK(cudaHostAlloc(&triggers,
                                  capacity * sizeof(TransferCmd),
                                  cudaHostAllocMapped));
    memset(triggers, 0, capacity * sizeof(TransferCmd));

    // Head: device memory, initialized to 0
    uint64_t* head_dev = nullptr;
    OSGC_CUDACHECK(cudaMalloc(&head_dev, sizeof(uint64_t)));
    OSGC_CUDACHECK(cudaMemset(head_dev, 0, sizeof(uint64_t)));

    // Tail: host-pinned, initialized to 0
    uint64_t* tail_host = nullptr;
    OSGC_CUDACHECK(cudaHostAlloc(&tail_host, sizeof(uint64_t),
                                  cudaHostAllocMapped));
    *tail_host = 0;

    // Tail cache: device memory, initialized to 0
    uint64_t* tail_cache_dev = nullptr;
    OSGC_CUDACHECK(cudaMalloc(&tail_cache_dev, sizeof(uint64_t)));
    OSGC_CUDACHECK(cudaMemset(tail_cache_dev, 0, sizeof(uint64_t)));

    // Device handle
    pair.device.triggers   = triggers;  // host-pinned, accessible from GPU
    pair.device.head       = head_dev;
    pair.device.tail       = tail_host; // GPU reads host-pinned tail
    pair.device.tail_cache = tail_cache_dev;
    pair.device.capacity   = capacity;

    // Host handle
    pair.host.triggers  = triggers;
    pair.host.tail      = tail_host;
    pair.host.capacity  = capacity;
    pair.host.cpu_head  = 0;

    return pair;
}

/**
 * Destroy a D2H FIFO pair. Frees all allocated memory.
 */
inline void destroy_d2h_fifo(D2HFifoPair& pair) {
    if (pair.device.tail_cache) cudaFree(pair.device.tail_cache);
    if (pair.device.head)       cudaFree(pair.device.head);
    if (pair.device.tail)       cudaFreeHost((void*)pair.device.tail);
    if (pair.device.triggers)   cudaFreeHost(pair.device.triggers);
    pair = D2HFifoPair{};
}


} // namespace internode
