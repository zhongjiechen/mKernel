/**
 * @file proxy_diagnostics.h
 * @brief Diagnostic / timeline structs shared across proxy backends.
 *
 * Extracted so that backends that don't include the full ibverbs `Proxy`
 * class (e.g. EFA/SRD via libibverbs, libfabric) can still participate in
 * the shared diagnostic vocabulary used by host-side pybind modules.
 *
 * proxy.h and any alternative proxy implementation (proxy_efa.h, proxy_fi.h)
 * pulls these definitions in — no proxy class is defined here, so it is
 * safe to include alongside a backend-specific proxy header without
 * triggering name collisions.
 */
#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace internode {

/**
 * Timestamps for diagnosing first-message latency in the proxy loop.
 * Only the FIRST command per epoch is recorded (read-only instrumentation).
 */
struct ProxyTimestamps {
    uint64_t proxy_start_ns;      // when run() resumed after pause
    uint64_t first_cmd_ns;        // when first FIFO command was seen
    uint64_t first_post_ns;       // when first ibv_post_send completed
    uint64_t first_completion_ns; // when first CQ completion arrived
    bool     recorded;            // set to true after first command processed

    ProxyTimestamps() { reset(); }

    void reset() {
        proxy_start_ns      = 0;
        first_cmd_ns        = 0;
        first_post_ns       = 0;
        first_completion_ns = 0;
        recorded            = false;
    }
};

/**
 * Per-proxy-thread aggregate counters used by the pybind diagnostic API.
 * Backends that don't collect a given counter simply leave it at zero.
 */
struct ProxyDiagnostics {
    int      qp_base_idx;
    int      num_qps;
    uint64_t loops;
    uint64_t empty_loops;
    uint64_t full_batches;
    uint64_t partial_batches;
    uint64_t inflight_limited_loops;
    uint64_t batches_posted;
    uint64_t cmds_posted;
    uint64_t cmds_completed;
    uint64_t backlog_samples;
    uint64_t backlog_nonzero_loops;
    uint64_t backlog_gt_batch_loops;
    uint64_t backlog_sum;
    uint64_t backlog_max;
    uint64_t batch_cmd_to_post_count;
    uint64_t batch_cmd_to_post_sum_ns;
    uint64_t batch_cmd_to_post_max_ns;
    uint64_t batch_completion_count;
    uint64_t batch_completion_sum_ns;
    uint64_t batch_completion_max_ns;
    uint64_t enqueue_to_seen_count;
    int64_t  enqueue_to_seen_raw_sum_ns;
    int64_t  enqueue_to_seen_raw_max_ns;
    int64_t  enqueue_to_seen_raw_min_ns;
#ifdef Q2_PROBE_PROXY_TAIL
    // Iter 30 probe: inter-push interval (delta between successive
    // cmd.enqueue_device_ns timestamps observed at the proxy poll site).
    // Bucket boundaries in nanoseconds: <5us, 5-15us, 15-50us, 50-150us, >=150us.
    // Only populated by proxy_efa.h when compiled with Q2_PROBE_PROXY_TAIL.
    uint64_t ipi_bucket_ns[5];
    uint64_t ipi_count;
    uint64_t ipi_sum_ns;
    uint64_t ipi_max_ns;
#endif
    uint64_t loop_gap_max_ns;
    uint64_t loop_gap_over_100us;
    uint64_t loop_gap_over_1ms;
    uint64_t first_cmd_ns;
    uint64_t last_cmd_ns;
    uint64_t first_post_ns;
    uint64_t last_post_ns;
    uint64_t first_completion_ns;
    uint64_t last_completion_ns;

    ProxyDiagnostics() { reset(); }

    void reset() {
        qp_base_idx = 0;
        num_qps = 0;
        loops = 0;
        empty_loops = 0;
        full_batches = 0;
        partial_batches = 0;
        inflight_limited_loops = 0;
        batches_posted = 0;
        cmds_posted = 0;
        cmds_completed = 0;
        backlog_samples = 0;
        backlog_nonzero_loops = 0;
        backlog_gt_batch_loops = 0;
        backlog_sum = 0;
        backlog_max = 0;
        batch_cmd_to_post_count = 0;
        batch_cmd_to_post_sum_ns = 0;
        batch_cmd_to_post_max_ns = 0;
        batch_completion_count = 0;
        batch_completion_sum_ns = 0;
        batch_completion_max_ns = 0;
        enqueue_to_seen_count = 0;
        enqueue_to_seen_raw_sum_ns = 0;
        enqueue_to_seen_raw_max_ns = std::numeric_limits<int64_t>::min();
        enqueue_to_seen_raw_min_ns = std::numeric_limits<int64_t>::max();
#ifdef Q2_PROBE_PROXY_TAIL
        for (int i = 0; i < 5; ++i) ipi_bucket_ns[i] = 0;
        ipi_count = 0;
        ipi_sum_ns = 0;
        ipi_max_ns = 0;
#endif
        loop_gap_max_ns = 0;
        loop_gap_over_100us = 0;
        loop_gap_over_1ms = 0;
        first_cmd_ns = 0;
        last_cmd_ns = 0;
        first_post_ns = 0;
        last_post_ns = 0;
        first_completion_ns = 0;
        last_completion_ns = 0;
    }
};

/**
 * Per-proxy-thread timeline of signaled-post / completion timestamps,
 * captured as vectors so host code can dump them to JSON.
 */
struct ProxyTimeline {
    std::vector<uint64_t> signaled_post_ns;
    std::vector<uint64_t> completion_ns;
};

} // namespace internode
