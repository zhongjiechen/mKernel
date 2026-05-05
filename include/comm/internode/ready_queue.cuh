/**
 * @file ready_queue.cuh
 * @brief GPU-visible ready queue for push-model RDMA notification.
 *
 * Single-producer (CPU recv proxy) / multi-consumer (GPU CTAs) ring buffer.
 * The CPU proxy polls the recv CQ for WRITE_WITH_IMM completions, extracts
 * the work ID from imm_data, and appends it to the queue. GPU CTAs claim
 * work items via atomicAdd on head.
 *
 * Memory layout:
 *   entries[capacity]  — device memory, holds work IDs
 *   tail               — host-mapped (cudaHostAllocMapped), CPU writes, GPU reads
 *   head               — device memory, GPU atomicAdd to claim slots
 *   done_tail          — device memory, GPU reads to know total expected items
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#ifndef OSGC_CUDACHECK
#define OSGC_CUDACHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    } \
} while(0)
#endif

namespace internode {

/**
 * Device-side handle passed into the kernel. All pointers are GPU-accessible.
 */
struct ReadyQueueDevice {
    volatile uint32_t* entries;   // ring buffer of work IDs [capacity]
    volatile uint32_t* tail;      // producer tail (CPU writes, GPU reads)
    uint32_t*          head;      // consumer head (GPU atomicAdd)
    uint32_t           capacity;  // must be power of 2
    uint32_t           total;     // total expected items (set before launch)
};

/**
 * Host-side handle for the CPU recv proxy to append items.
 */
struct ReadyQueueHost {
    uint32_t* entries_host;       // host-mapped alias of device entries
    uint32_t* tail_host;          // host pointer to tail (same pinned alloc)
    uint32_t  local_tail;         // local copy for batch appends
    uint32_t  capacity;
};

/**
 * Combined pair (mirrors D2HFifoPair pattern).
 */
struct ReadyQueuePair {
    ReadyQueueDevice device;
    ReadyQueueHost   host;
};

/**
 * Create a ready queue with the given capacity (must be power of 2).
 *
 * entries: allocated via cudaHostAlloc(Mapped) so both CPU and GPU can access.
 *   CPU proxy writes entries, GPU reads them via the device pointer alias.
 * tail: same — host-mapped pinned memory. CPU increments, GPU polls.
 * head: device memory (fast GPU atomics).
 */
inline ReadyQueuePair create_ready_queue(uint32_t capacity) {
    // Ensure power of 2
    if (capacity == 0) capacity = 256;
    if ((capacity & (capacity - 1)) != 0) {
        // Round up to next power of 2
        uint32_t v = capacity - 1;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
        capacity = v + 1;
    }

    ReadyQueuePair rq{};
    rq.device.capacity = capacity;
    rq.host.capacity = capacity;
    rq.host.local_tail = 0;

    // entries: host-mapped so CPU can write and GPU can read
    uint32_t* entries_host = nullptr;
    OSGC_CUDACHECK(cudaHostAlloc(&entries_host, capacity * sizeof(uint32_t),
                                  cudaHostAllocMapped));
    memset(entries_host, 0, capacity * sizeof(uint32_t));
    rq.host.entries_host = entries_host;

    // Get device pointer alias
    uint32_t* entries_dev = nullptr;
    OSGC_CUDACHECK(cudaHostGetDevicePointer(&entries_dev, entries_host, 0));
    rq.device.entries = reinterpret_cast<volatile uint32_t*>(entries_dev);

    // tail: host-mapped pinned memory
    uint32_t* tail_host = nullptr;
    OSGC_CUDACHECK(cudaHostAlloc(&tail_host, sizeof(uint32_t),
                                  cudaHostAllocMapped));
    *tail_host = 0;
    rq.host.tail_host = tail_host;

    uint32_t* tail_dev = nullptr;
    OSGC_CUDACHECK(cudaHostGetDevicePointer(&tail_dev, tail_host, 0));
    rq.device.tail = reinterpret_cast<volatile uint32_t*>(tail_dev);

    // head: device memory for fast GPU atomics
    OSGC_CUDACHECK(cudaMalloc(&rq.device.head, sizeof(uint32_t)));
    OSGC_CUDACHECK(cudaMemset(rq.device.head, 0, sizeof(uint32_t)));

    rq.device.total = 0;

    return rq;
}

/**
 * Reset the ready queue between epochs. Call while proxy is paused.
 */
inline void reset_ready_queue(ReadyQueuePair& rq) {
    memset(rq.host.entries_host, 0, rq.host.capacity * sizeof(uint32_t));
    *rq.host.tail_host = 0;
    rq.host.local_tail = 0;
    OSGC_CUDACHECK(cudaMemset(rq.device.head, 0, sizeof(uint32_t)));
    rq.device.total = 0;
}

/**
 * Destroy the ready queue and free all memory.
 */
inline void destroy_ready_queue(ReadyQueuePair& rq) {
    if (rq.device.head) cudaFree(rq.device.head);
    if (rq.host.tail_host) cudaFreeHost(rq.host.tail_host);
    if (rq.host.entries_host) cudaFreeHost(rq.host.entries_host);
    memset(&rq, 0, sizeof(rq));
}

// -------------------------------------------------------------------------
// Host-side producer API (called by recv proxy thread)
// -------------------------------------------------------------------------

/**
 * Append a single work ID to the ready queue. Thread-safe for single producer.
 * The tail is incremented with a release store so the GPU sees the entry
 * before the tail advance.
 */
inline void ready_queue_push(ReadyQueueHost& rq, uint32_t work_id) {
    const uint32_t slot = rq.local_tail & (rq.capacity - 1);
    rq.entries_host[slot] = work_id;
    // Release store: ensure entry is visible before tail advance.
    // On x86 a plain store suffices (TSO), but be explicit for portability.
    __atomic_store_n(rq.tail_host, rq.local_tail + 1, __ATOMIC_RELEASE);
    rq.local_tail++;
}

/**
 * Batch-append multiple work IDs, advancing tail once at the end.
 * More efficient than per-item push when draining multiple CQEs.
 */
inline void ready_queue_push_batch(ReadyQueueHost& rq,
                                    const uint32_t* work_ids, int count) {
    for (int i = 0; i < count; i++) {
        const uint32_t slot = (rq.local_tail + i) & (rq.capacity - 1);
        rq.entries_host[slot] = work_ids[i];
    }
    // Single release store after all entries are written
    __atomic_store_n(rq.tail_host, rq.local_tail + count, __ATOMIC_RELEASE);
    rq.local_tail += count;
}

} // namespace internode
