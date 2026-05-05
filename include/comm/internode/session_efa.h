/**
 * @file session_efa.h
 * @brief EFA session lifecycle for inter-node RDMA communication.
 *
 * Same API as session.h but uses EFA SRD transport instead of RC.
 *
 * This backend uses direct verbs (`efadv`) for EFA while keeping the same
 * GPU-resident buffer model as the other internode backends. It mirrors the
 * CX7 session's multi-knob layout:
 *   - num_qps            : multiple SRD QPs pointing at the same peer AH
 *   - num_proxy_threads  : N host proxy threads, each owning a QP slice
 *   - logical_queues_per_qp : software queue multiplexing per QP
 *
 * The only differences from session.h are the transport-level details (SRD
 * address handles, QP state transitions, device selection) and that the
 * arrival flags live in host-mapped memory (the SRD receive path uses
 * RDMA_WRITE_WITH_IMM CQEs to publish flags from the host proxy).
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "types.h"
#include "rdma_transport_efa.h"
#include "rdma_gpu_mr.cuh"
#include "d2h_fifo.cuh"
#include "arrival.cuh"
#include "ready_queue.cuh"
#include "proxy_efa.h"
#include "proxy_diagnostics.h"


#ifdef Q2_PROBE_PROXY_TAIL
#include <limits>
#endif

namespace internode {

#ifdef Q2_PROBE_PROXY_TAIL
// Iter 30 probe: local helpers to calibrate the GPU %globaltimer → host
// CLOCK_MONOTONIC offset once per session. Same math as session.h:54-83 but
// defined inside the EFA session header (session.h is not included on the
// EFA path; proxy.h's CX7 backend owns the canonical copy). Gated entirely
// behind Q2_PROBE_PROXY_TAIL so canonical builds are byte-identical.
inline uint64_t q2_probe_host_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

__global__ inline void q2_probe_read_globaltimer_kernel(unsigned long long* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        out[0] = t;
    }
}

inline int64_t calibrate_gpu_to_host_offset_ns(int samples = 32) {
    if (samples <= 0) samples = 1;
    unsigned long long* dev_out = nullptr;
    unsigned long long host_out = 0;
    cudaStream_t stream = nullptr;
    OSGC_CUDACHECK(cudaMalloc(&dev_out, sizeof(unsigned long long)));
    OSGC_CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    uint64_t best_window_ns = std::numeric_limits<uint64_t>::max();
    int64_t best_offset_ns = 0;
    for (int i = 0; i < samples; ++i) {
        const uint64_t host_before_ns = q2_probe_host_now_ns();
        q2_probe_read_globaltimer_kernel<<<1, 1, 0, stream>>>(dev_out);
        OSGC_CUDACHECK(cudaGetLastError());
        OSGC_CUDACHECK(cudaMemcpyAsync(
            &host_out, dev_out, sizeof(unsigned long long),
            cudaMemcpyDeviceToHost, stream));
        OSGC_CUDACHECK(cudaStreamSynchronize(stream));
        const uint64_t host_after_ns = q2_probe_host_now_ns();
        const uint64_t window_ns = host_after_ns - host_before_ns;
        if (window_ns < best_window_ns) {
            best_window_ns = window_ns;
            const uint64_t host_mid_ns = host_before_ns + window_ns / 2ULL;
            best_offset_ns = (int64_t)host_mid_ns - (int64_t)host_out;
        }
    }

    OSGC_CUDACHECK(cudaStreamDestroy(stream));
    OSGC_CUDACHECK(cudaFree(dev_out));
    return best_offset_ns;
}
#endif

// Match CX7: two stage-barrier slots per session.
static constexpr int kEfaStageBarrierSlots = 2;
// Cap total QPs per session (mirrors CX7's kMaxQPs/kMaxExchangeQPs).
static constexpr int kEfaMaxQPs = kMaxExchangeQPs;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct SessionConfig {
    int         rank;
    const char* peer_ip;
    int         tcp_port;

    void*       local_gpu_buf;
    size_t      local_gpu_buf_size;

    size_t      recv_buf_size;

    // Zero-copy receive: when non-null, create_session does NOT allocate its
    // own recv_buf via cudaMalloc. Instead it registers this externally-owned
    // GPU buffer (e.g. an IPC-shared peer_tokens slot) on every rail's PD and
    // exchanges its rkey + remote address over TCP. RDMA writes from peers
    // land directly into this buffer — receivers can read it via IPC without
    // an intermediate D2D copy. Caller owns the lifetime of the buffer.
    void*       external_recv_buf       = nullptr;
    int         num_tiles;
    int         fifo_capacity;
    int         device_id;

    int         max_inflight;

    // Optional second source buffer (e.g. `output_local`) registered as a
    // DMA-BUF MR on every rail's PD. When populated + direct_dmabuf_enabled,
    // the GPU kernel can set cmd.src_view = 1 and compute cmd.local_offset
    // into this buffer; the proxy sends directly from HBM without an
    // intermediate pack into the staging buffer. Mirrors Q2's clocal_gpu_buf.
    void*       clocal_gpu_buf          = nullptr;
    size_t      clocal_gpu_buf_size     = 0;
    bool        direct_dmabuf_enabled   = false;

    // Kept for config parity with session.h's CX7 backend so callers that
    // set row_stride_bytes (for src_view=2 strided sends) compile cleanly
    // even though EFA SRD (max_sge=2) doesn't implement strided gather.
    size_t      row_stride_bytes        = 0;

    // Performance knobs — same semantics as session.h.
    int         num_qps                = 1;
    int         logical_queues_per_qp  = 1;
    int         num_proxy_threads      = 1;

    // Unused on SRD but kept for config parity with the ibverbs backend.
    bool        use_write_imm          = false;
    int         ready_queue_cap        = 0;
    bool        use_arrival_queue      = false;

    // Multi-rail: number of EFA NICs to use per GPU session. Each rail is a
    // separate ibv_context with its own PD, CQs, QPs, and MR registrations.
    // QPs are distributed round-robin across rails. Each proxy thread handles
    // QPs from exactly one rail. Default 2 uses all 16 NICs on p5 (2 per GPU).
    int         num_rails              = 2;

    // Whether to pin each proxy thread to the GPU's NUMA node (default true).
    bool        pin_proxy              = true;
};

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

// Per-rail RDMA resources. Each rail is a separate ibv_context (NIC device)
// with its own PD, CQs, QPs, and MR registrations.
struct RailResources {
    ibv_context* ctx  = nullptr;
    ibv_pd*      pd   = nullptr;
    ibv_mr*      local_data_mr  = nullptr;  // same GPU buf, registered on this rail's PD
    ibv_mr*      clocal_data_mr = nullptr;  // direct-DMA-BUF MR for output_local (optional)
    ibv_mr*      recv_buf_mr    = nullptr;  // same recv buf, registered on this rail's PD
    ibv_mr*      arrival_mr     = nullptr;  // arrival flags, registered on this rail's PD
    ibv_mr*      staging_mrs[kMaxProxyThreads] = {};  // per-proxy staging MRs
    ibv_mr*      barrier_mr     = nullptr;  // stage-barrier, registered on this rail's PD
    ibv_ah*      dst_ah         = nullptr;  // AH for remote GID on this rail
};

struct Session {
    // Multi-rail: each rail is a separate NIC device with its own RDMA resources.
    // Rail 0 is the "primary" rail; its ctx/pd/cq are aliased by the legacy
    // single-rail fields below for minimal churn in non-session code.
    int           num_rails;
    RailResources rails[kMaxRails];

    // Legacy aliases — point to rail 0's resources for backwards compatibility.
    ibv_context*  ctx;   // = rails[0].ctx
    ibv_pd*       pd;    // = rails[0].pd

    // Primary (index 0) QP / CQ. Extra QPs live in `extra_qps`; per-proxy CQs
    // live in `proxy_cqs` — proxy_cqs[0] aliases cq.
    ibv_cq*       cq;
    ibv_qp*       qp;
    ibv_qp*       extra_qps[kEfaMaxQPs - 1];
    ibv_cq*       proxy_cqs[kMaxProxyThreads];
    int           qp_rail[kEfaMaxQPs];       // which rail each QP belongs to

    // Per-QP destination QPN (remote QP number).
    uint32_t      dst_qpns[kEfaMaxQPs];

    // Session-wide counters used by create_session + diagnostics.
    int           num_qps;
    int           num_proxy_threads;
    int           logical_queues_per_qp;

    // Receive buffer (allocated on GPU, RDMA-registered for remote writes).
    // Registered on every rail's PD; the per-rail MRs live in rails[r].recv_buf_mr.
    gpu_mr::GpuRdmaBuffer recv_buf;

    // D2H command FIFOs (one per proxy thread / FIFO channel).
    // Under EFAGDA the FIFO + host-side proxy are absent; the same
    // `fifo_bundle` slot holds the EFA SQ/DB handles instead.
    D2HFifoPair         fifos[kMaxProxyThreads];
    D2HFifoDeviceBundle fifo_bundle;

    // Arrival flags + per-proxy metadata staging.
    ArrivalFlags        arrival;
    FlagStaging         flag_stagings[kMaxProxyThreads];
    StageBarrierFlags   stage_barrier;

    // CPU proxy threads.
    Proxy*        proxies[kMaxProxyThreads];
    // Per-proxy fifo pointer storage (used when num_fifos > num_proxy_threads).
    // Each proxy[t] gets a contiguous slice of fifos via per_proxy_fifo_ptrs[t].
    D2HFifoHost*  per_proxy_fifo_ptrs[kMaxProxyThreads][kMaxProxyThreads];


    // Remote connection info (populated after TCP exchange).
    ConnectionInfo remote_info;

    int rank;
    int sq_depth;
    uint32_t epoch;
};

// ---------------------------------------------------------------------------
// Create / Destroy
// ---------------------------------------------------------------------------

inline Session* create_session(const SessionConfig& cfg) {
    Session* s = new Session{};
    s->rank = cfg.rank;
    s->epoch = 1;

    // --- 0. Clamp knobs and derive session-wide counters.
    int num_rails = cfg.num_rails <= 0 ? 1 : cfg.num_rails;
    if (num_rails > kMaxRails) num_rails = kMaxRails;
    // Allow OSGC_EFA_NUM_RAILS env override.
    {
        const char* env = std::getenv("OSGC_EFA_NUM_RAILS");
        if (env && env[0]) {
            int v = std::atoi(env);
            if (v >= 1 && v <= kMaxRails) num_rails = v;
        }
    }
    s->num_rails = num_rails;

    int num_qps = cfg.num_qps <= 0 ? 1 : cfg.num_qps;
    if (num_qps > kEfaMaxQPs) num_qps = kEfaMaxQPs;
    // Can't have more rails than QPs (each rail needs at least one QP).
    if (num_rails > num_qps) num_rails = num_qps;
    s->num_rails = num_rails;
    int num_proxy_threads = cfg.num_proxy_threads <= 0 ? 1 : cfg.num_proxy_threads;
    // With multi-rail, need at least one proxy per rail (each proxy's QPs must
    // share a single CQ, which is tied to one ibv_context / NIC device).
    if (num_proxy_threads < num_rails) num_proxy_threads = num_rails;
    if (num_proxy_threads > kMaxProxyThreads) num_proxy_threads = kMaxProxyThreads;
    if (num_proxy_threads > num_qps) num_proxy_threads = num_qps;
    num_proxy_threads = std::max(1, num_proxy_threads);
    int logical_queues_per_qp = cfg.logical_queues_per_qp <= 0 ? 1 : cfg.logical_queues_per_qp;
    if (logical_queues_per_qp > 16) logical_queues_per_qp = 16;

    s->num_qps = num_qps;
    s->num_proxy_threads = num_proxy_threads;
    s->logical_queues_per_qp = logical_queues_per_qp;

    const int total_logical_queues = num_qps * logical_queues_per_qp;
    const int logical_queue_stride =
        std::max(1, (cfg.num_tiles + total_logical_queues - 1) / total_logical_queues);
    const int total_arrival_slots = logical_queue_stride * total_logical_queues;
    const int qps_per_proxy =
        std::max(1, (num_qps + num_proxy_threads - 1) / num_proxy_threads);

    // Zero-init fixed arrays for safe partial teardown.
    memset(s->extra_qps, 0, sizeof(s->extra_qps));
    memset(s->proxy_cqs, 0, sizeof(s->proxy_cqs));
    memset(s->proxies, 0, sizeof(s->proxies));
    memset(s->flag_stagings, 0, sizeof(s->flag_stagings));
    memset(s->dst_qpns, 0, sizeof(s->dst_qpns));
    memset(s->qp_rail, 0, sizeof(s->qp_rail));
    memset(s->rails, 0, sizeof(s->rails));

    // --- 1. Open EFA devices (PCIe-root-aware, multi-rail).
    auto opened_devs = rdma::open_efa_devices(cfg.device_id, num_rails);
    num_rails = (int)opened_devs.size();
    s->num_rails = num_rails;

    // Initialize per-rail resources: context, PD.
    for (int r = 0; r < num_rails; r++) {
        s->rails[r].ctx = opened_devs[r].ctx;
        s->rails[r].pd  = rdma::alloc_pd(s->rails[r].ctx);
    }
    // Legacy aliases for rail 0.
    s->ctx = s->rails[0].ctx;
    s->pd  = s->rails[0].pd;

    // --- 1a. Block-assign QPs to rails.
    // QP i → rail = i * num_rails / num_qps. This groups the first
    // ceil(num_qps/num_rails) QPs onto rail 0, the next block onto rail 1, etc.
    // Block assignment ensures each proxy's contiguous QP slice lands entirely
    // on one rail, which is required because all QPs in a proxy share a single
    // CQ tied to one ibv_context.
    for (int i = 0; i < num_qps; i++) {
        s->qp_rail[i] = i * num_rails / num_qps;
    }

    // Verify alignment: every proxy's contiguous QP slice must be single-rail.
    for (int t = 0; t < num_proxy_threads; t++) {
        const int qb = t * qps_per_proxy;
        if (qb >= num_qps) break;
        const int first_rail = s->qp_rail[qb];
        const int qe = std::min(qb + qps_per_proxy, num_qps);
        for (int i = qb; i < qe; i++) {
            if (s->qp_rail[i] != first_rail) {
                fprintf(stderr,
                    "session_efa: QP %d (rail %d) in proxy %d (rail %d) — "
                    "misaligned! Adjust num_qps=%d or num_proxy_threads=%d so "
                    "contiguous QP slices don't span rail boundaries.\n",
                    i, s->qp_rail[i], t, first_rail, num_qps, num_proxy_threads);
                // Auto-fix: clamp to 1 rail to avoid crash.
                for (int j = 0; j < num_qps; j++) s->qp_rail[j] = 0;
                num_rails = 1;
                s->num_rails = 1;
                break;
            }
        }
        if (num_rails == 1) break;
    }

    // --- 1b. Per-proxy CQs. Proxy t's rail is determined by its first QP's rail.
    for (int t = 0; t < num_proxy_threads; t++) {
        const int qb = t * qps_per_proxy;
        const int rail = (qb < num_qps) ? s->qp_rail[qb] : 0;
        s->proxy_cqs[t] = rdma::create_cq(s->rails[rail].ctx, 4096);
    }
    s->cq = s->proxy_cqs[0];

    // --- 1c. Create SRD QPs on their assigned rail's ctx/pd, sharing the
    // owning proxy's CQ.
    s->sq_depth = 512;
    for (int i = 0; i < num_qps; i++) {
        const int rail = s->qp_rail[i];
        const int owner_thread = std::min(num_proxy_threads - 1, i / qps_per_proxy);

        int sq_depth_out = s->sq_depth;
        ibv_qp* new_qp = rdma::create_srd_qp(
            s->rails[rail].pd, s->proxy_cqs[owner_thread],
            s->rails[rail].ctx, 512, &sq_depth_out);
        rdma::modify_srd_qp_init(new_qp);
        rdma::modify_srd_qp_rtr(new_qp);
        rdma::modify_srd_qp_rts(new_qp);

        if (i == 0) {
            s->qp = new_qp;
        } else {
            s->extra_qps[i - 1] = new_qp;
        }
    }

    // --- 2. Register the caller-owned GPU send buffer on every rail's PD.
    for (int r = 0; r < num_rails; r++) {
        s->rails[r].local_data_mr = gpu_mr::register_gpu_buffer(
            s->rails[r].pd, cfg.local_gpu_buf, cfg.local_gpu_buf_size);
    }

    // --- 2b. Optional: register the direct-send GPU source buffer (e.g.
    // output_local) as a DMA-BUF MR on every rail's PD. Skipping the staging
    // pack + gather on the send critical path. Hard-fails if DMA-BUF export
    // is unavailable — callers opt in explicitly.
    if (cfg.direct_dmabuf_enabled && cfg.clocal_gpu_buf != nullptr
        && cfg.clocal_gpu_buf_size > 0) {
        for (int r = 0; r < num_rails; r++) {
            const char* path = nullptr;
            s->rails[r].clocal_data_mr = gpu_mr::register_gpu_buffer_dmabuf_only(
                s->rails[r].pd, cfg.clocal_gpu_buf, cfg.clocal_gpu_buf_size,
                gpu_mr::kDefaultAccess, &path);
            if (s->rails[r].clocal_data_mr == nullptr) {
                fprintf(stderr,
                    "session_efa: direct-DMA-BUF registration failed on rail %d "
                    "(ptr=%p bytes=%zu). EFA driver may not support DMA-BUF or "
                    "the buffer is not exportable. Disable the direct path.\n",
                    r, cfg.clocal_gpu_buf, cfg.clocal_gpu_buf_size);
                exit(EXIT_FAILURE);
            }
        }
    }

    // --- 3. GPU receive buffer.
    // Two modes:
    //   (a) Default — allocate via cudaMalloc and register on every rail's PD.
    //   (b) external_recv_buf set — register the caller-owned buffer instead.
    //       Used by Q3's zero-copy inter-recv where peer_tokens is the RDMA
    //       target, eliminating the recv_buf -> peer_tokens D2D copy. Caller
    //       owns the buffer and its lifetime.
    if (cfg.external_recv_buf != nullptr) {
        s->recv_buf.gpu_ptr = cfg.external_recv_buf;
        s->recv_buf.size = cfg.recv_buf_size;
        s->recv_buf.mr = nullptr;  // not owned: free_buffer becomes a no-op
                                   // for the gpu_ptr; per-rail MRs still freed.
        s->rails[0].recv_buf_mr = gpu_mr::register_gpu_buffer(
            s->rails[0].pd, s->recv_buf.gpu_ptr, s->recv_buf.size);
        for (int r = 1; r < num_rails; r++) {
            s->rails[r].recv_buf_mr = gpu_mr::register_gpu_buffer(
                s->rails[r].pd, s->recv_buf.gpu_ptr, s->recv_buf.size);
        }
    } else {
        // Use rail 0 for the allocation + primary MR.
        s->recv_buf = gpu_mr::alloc_and_register(
            s->rails[0].pd, cfg.recv_buf_size, cfg.device_id);
        s->rails[0].recv_buf_mr = s->recv_buf.mr;
        for (int r = 1; r < num_rails; r++) {
            s->rails[r].recv_buf_mr = gpu_mr::register_gpu_buffer(
                s->rails[r].pd, s->recv_buf.gpu_ptr, s->recv_buf.size);
        }
    }

    // --- 4. D2H FIFOs.
    // Default: one FIFO per proxy thread (num_fifos = num_proxy_threads).
    // OSGC_FIFO_PER_QP=1 enables one-fifo-per-QP mode (num_fifos = num_qps),
    // which decouples FIFO count from thread count and reduces GPU-side
    // atomicAdd contention on the FIFO head pointer (each QP gets its own
    // head). Each proxy thread then round-robins through its slice of fifos.
    bool fifo_per_qp = false;
    if (const char* e = std::getenv("OSGC_FIFO_PER_QP")) {
        fifo_per_qp = (std::atoi(e) != 0);
    }
    int num_fifos = fifo_per_qp ? num_qps : num_proxy_threads;
    if (num_fifos > kMaxProxyThreads) num_fifos = kMaxProxyThreads;  // clamp to bundle.fifos[] size
    const int fifos_per_proxy = num_fifos / num_proxy_threads;
    const int fifo_cap = cfg.fifo_capacity > 0 ? cfg.fifo_capacity : 1024;
    s->fifo_bundle = D2HFifoDeviceBundle{};
    s->fifo_bundle.num_fifos = num_fifos;
    s->fifo_bundle.global_num_qps = num_qps;
    s->fifo_bundle.logical_queues_per_qp = logical_queues_per_qp;
    // qps_per_fifo follows num_fifos: when fifo-per-qp, each fifo owns 1 QP.
    s->fifo_bundle.qps_per_fifo = fifo_per_qp ? 1 : qps_per_proxy;
    for (int t = 0; t < num_fifos; t++) {
        s->fifos[t] = create_d2h_fifo(fifo_cap);
        s->fifo_bundle.fifos[t] = s->fifos[t].device;
    }

    // --- 5. Arrival flags. Register on every rail's PD so the peer can target
    // any of our NICs and the rkey will be valid on the receiving device.
    const size_t arrival_bytes =
        ((size_t)total_arrival_slots + (size_t)total_logical_queues) * sizeof(uint32_t);
    s->arrival = create_mapped_arrival_flags(total_arrival_slots, total_logical_queues);
    for (int r = 0; r < num_rails; r++) {
        s->rails[r].arrival_mr = rdma::reg_mr(
            s->rails[r].pd, (void*)s->arrival.host_ptr, arrival_bytes,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    }
    s->arrival.mr = s->rails[0].arrival_mr;

    // --- 6. Per-proxy flag staging. Register each staging buffer on its proxy's
    // rail PD (only the local NIC needs lkey access for the SGE source).
    for (int t = 0; t < num_proxy_threads; t++) {
        const int qb = t * qps_per_proxy;
        const int rail = (qb < num_qps) ? s->qp_rail[qb] : 0;
        s->flag_stagings[t] = create_flag_staging(8192);
        s->flag_stagings[t].mr = rdma::reg_mr(
            s->rails[rail].pd, s->flag_stagings[t].host_ptr,
            s->flag_stagings[t].count * sizeof(uint32_t),
            IBV_ACCESS_LOCAL_WRITE);
        s->rails[rail].staging_mrs[t] = s->flag_stagings[t].mr;
    }

    // --- 6b. Stage-barrier flags: register on every rail's PD.
    s->stage_barrier = create_stage_barrier_flags(kEfaStageBarrierSlots);
    for (int r = 0; r < num_rails; r++) {
        s->rails[r].barrier_mr = rdma::reg_mr(
            s->rails[r].pd, (void*)s->stage_barrier.host_ptr,
            kEfaStageBarrierSlots * sizeof(uint32_t),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    }
    s->stage_barrier.mr = s->rails[0].barrier_mr;

    // --- 7. TCP exchange — pack primary QP + extras + multi-rail info.
    ConnectionInfo local_info{};
    rdma::fill_local_info(local_info, s->qp, s->rails[0].ctx);
    // Rail 0 rkeys (primary fields).
    local_info.data_rkey    = s->rails[0].recv_buf_mr->rkey;
    local_info.data_addr    = (uint64_t)s->recv_buf.gpu_ptr;
    local_info.data_len     = s->recv_buf.size;
    local_info.flags_rkey   = s->rails[0].arrival_mr->rkey;
    local_info.flags_addr   = (uint64_t)s->arrival.host_ptr;
    local_info.tail_rkey    = s->rails[0].arrival_mr->rkey;
    local_info.tail_addr    = (uint64_t)(s->arrival.host_ptr + total_arrival_slots);
    local_info.barrier_rkey = s->rails[0].barrier_mr->rkey;
    local_info.barrier_addr = (uint64_t)s->stage_barrier.host_ptr;
    local_info.num_qps = num_qps;
    for (int i = 1; i < num_qps; i++) {
        local_info.extra_qp_nums[i - 1] = s->extra_qps[i - 1]->qp_num;
        local_info.extra_psns[i - 1] = 0;  // PSN unused by SRD
    }

    // Multi-rail exchange fields: per-QP rail assignment + per-rail GIDs/rkeys.
    local_info.num_rails = num_rails;
    for (int i = 0; i < num_qps; i++) {
        local_info.qp_to_rail[i] = s->qp_rail[i];
    }
    for (int r = 1; r < num_rails; r++) {
        auto& ri = local_info.extra_rails[r - 1];
        // GID for this rail's NIC.
        ibv_gid gid{};
        ibv_query_gid(s->rails[r].ctx, 1, 0, &gid);
        memcpy(ri.gid, &gid, 16);
        ri.data_rkey    = s->rails[r].recv_buf_mr->rkey;
        ri.flags_rkey   = s->rails[r].arrival_mr->rkey;
        ri.tail_rkey    = s->rails[r].arrival_mr->rkey;
        ri.barrier_rkey = s->rails[r].barrier_mr->rkey;
    }

    const bool is_server = (cfg.rank == 0);
    s->remote_info = rdma::exchange_info_tcp(local_info, cfg.peer_ip,
                                              cfg.tcp_port, is_server);

    // --- 8. Build per-QP (dst_ah, dst_qpn) from exchanged remote info.
    // Each local QP i is paired with remote QP i. The remote QP lives on
    // remote rail qp_to_rail[i], so we use that rail's GID for the AH and
    // that rail's rkeys for data/flag writes.
    s->dst_qpns[0] = s->remote_info.qp_num;
    for (int i = 1; i < num_qps; i++) {
        s->dst_qpns[i] = s->remote_info.extra_qp_nums[i - 1];
    }

    // Create one AH per rail. Each AH targets the remote rail's GID and is
    // created on the local rail's PD. All QPs on the same rail share the AH
    // (they all send to the same remote NIC via the same local NIC).
    for (int r = 0; r < num_rails; r++) {
        // Remote rail r's GID. In symmetric topology both sides use the same
        // block QP-to-rail assignment, so local rail r pairs with remote rail r.
        const uint8_t* remote_gid = (r == 0)
            ? s->remote_info.gid
            : (s->remote_info.num_rails > 1)
                ? s->remote_info.extra_rails[r - 1].gid
                : s->remote_info.gid;
        s->rails[r].dst_ah = rdma::create_ah(s->rails[r].pd, remote_gid);
    }

#ifdef Q2_PROBE_PROXY_TAIL
    // Iter 30 probe: calibrate the GPU %globaltimer → host CLOCK_MONOTONIC
    // offset once per session. Used by proxy_efa.h's account_polled_cmd_for_probe_
    // helper to convert the kernel's `cmd.enqueue_device_ns` (GPU domain)
    // into the host domain for enqueue→seen delta computation. Matches the
    // CX7 backend pattern (session.h:414, pcfg.gpu_to_host_offset_ns line 451).
    const int64_t gpu_to_host_offset_ns = calibrate_gpu_to_host_offset_ns(64);
#endif

    // --- 9. Spawn proxy threads. Each proxy handles QPs on its assigned rail.
    for (int t = 0; t < num_proxy_threads; t++) {
        const int qp_base = t * qps_per_proxy;
        const int proxy_rail = (qp_base < num_qps) ? s->qp_rail[qp_base] : 0;
        const int local_qps = std::min(qps_per_proxy, num_qps - qp_base);
        if (local_qps <= 0) break;

        // Determine remote rkeys for this proxy's rail. The proxy posts RDMA
        // WRITEs to the remote peer's NIC on the remote rail paired with the
        // local rail. Remote rail = qp_to_rail[qp_base] from the exchange.
        const int remote_rail = (s->remote_info.num_rails > 1)
            ? s->remote_info.qp_to_rail[qp_base]
            : 0;
        const uint32_t remote_data_rkey = (remote_rail == 0)
            ? s->remote_info.data_rkey
            : s->remote_info.extra_rails[remote_rail - 1].data_rkey;
        const uint32_t remote_flags_rkey = (remote_rail == 0)
            ? s->remote_info.flags_rkey
            : s->remote_info.extra_rails[remote_rail - 1].flags_rkey;
        const uint32_t remote_tail_rkey = (remote_rail == 0)
            ? s->remote_info.tail_rkey
            : s->remote_info.extra_rails[remote_rail - 1].tail_rkey;
        const uint32_t remote_barrier_rkey = (remote_rail == 0)
            ? s->remote_info.barrier_rkey
            : s->remote_info.extra_rails[remote_rail - 1].barrier_rkey;

        ProxyConfig pcfg{};
        pcfg.fifo = &s->fifos[t].host;
        // Multi-fifo: hand this proxy a slice of fifos[fifos_per_proxy*t ..
        // fifos_per_proxy*(t+1)). Single-fifo mode: num_fifos==1 and run()
        // uses pcfg.fifo (legacy path). Storage is per-proxy in s->per_proxy_fifo_ptrs.
        if (num_fifos > num_proxy_threads) {
            D2HFifoHost** slot = s->per_proxy_fifo_ptrs[t];
            for (int k = 0; k < fifos_per_proxy; ++k) {
                slot[k] = &s->fifos[t * fifos_per_proxy + k].host;
            }
            pcfg.fifos = slot;
            pcfg.num_fifos = fifos_per_proxy;
            pcfg.fifo = slot[0];
        }
        pcfg.qp   = (qp_base == 0) ? s->qp : s->extra_qps[qp_base - 1];
        pcfg.cq   = s->proxy_cqs[t];

        pcfg.dst_ah = s->rails[proxy_rail].dst_ah;
        for (int i = 0; i < local_qps; i++) {
            pcfg.dst_qpns[i] = s->dst_qpns[qp_base + i];
        }
        memset(pcfg.extra_qps, 0, sizeof(pcfg.extra_qps));
        for (int i = 1; i < local_qps; i++) {
            const int global_qp = qp_base + i;
            pcfg.extra_qps[i - 1] = s->extra_qps[global_qp - 1];
        }
        pcfg.num_qps          = local_qps;
        pcfg.qp_base_idx      = qp_base;
        pcfg.global_num_qps   = num_qps;
        pcfg.logical_queues_per_qp = logical_queues_per_qp;

        // Per-rail local lkey and remote rkeys.
        pcfg.local_data_addr  = (uint64_t)cfg.local_gpu_buf;
        pcfg.local_data_lkey  = s->rails[proxy_rail].local_data_mr->lkey;
        pcfg.remote_data_addr = s->remote_info.data_addr;
        pcfg.remote_data_rkey = remote_data_rkey;

        // Direct-DMA-BUF (src_view=1) path: proxy posts SGEs sourced from the
        // caller-owned output buffer instead of staging. Only populated if the
        // session config opted in AND per-rail registration succeeded.
        if (cfg.direct_dmabuf_enabled && cfg.clocal_gpu_buf != nullptr
            && s->rails[proxy_rail].clocal_data_mr != nullptr) {
            pcfg.clocal_data_addr    = (uint64_t)cfg.clocal_gpu_buf;
            pcfg.clocal_data_lkey    = s->rails[proxy_rail].clocal_data_mr->lkey;
            pcfg.clocal_data_bytes   = cfg.clocal_gpu_buf_size;
            pcfg.direct_dmabuf_enabled = true;
        }

        pcfg.local_arrival_host_ptr = s->arrival.host_ptr;
        pcfg.local_arrival_count    = s->arrival.count;

        pcfg.flag_staging      = &s->flag_stagings[t];
        pcfg.remote_flags_addr = s->remote_info.flags_addr;
        pcfg.remote_flags_rkey = remote_flags_rkey;

        pcfg.remote_queue_stride = (uint32_t)logical_queue_stride;
        pcfg.remote_tail_addr    = s->remote_info.tail_addr;
        pcfg.remote_tail_rkey    = remote_tail_rkey;
        {
            const char* tail_env = std::getenv("Q2_SENDER_PUBLISHED_TAIL");
            pcfg.enable_remote_tail =
                (tail_env != nullptr && tail_env[0] == '1');
        }
        // R1 Commit 1: plumbing for the adaptive post/poll controller inside
        // Proxy::run(). Default 0 → byte-identical to the pre-R1 static
        // BATCH_SIZE/ACCUMULATE_SPINS path. When 1, the controller state
        // declared in Proxy is still unread until Commit 2/3; setting the
        // flag today only exercises the plumbing.
        {
            const char* pipe_env = std::getenv("Q2_PROXY_PIPELINE");
            pcfg.pipeline_enabled =
                (pipe_env != nullptr && pipe_env[0] == '1');
            if (t == 0 && cfg.rank == 0) {
                fprintf(stderr,
                        "internode_efa::Session Q2_PROXY_PIPELINE=%d\n",
                        (int)pcfg.pipeline_enabled);
            }
        }

        pcfg.remote_barrier_addr = s->remote_info.barrier_addr;
        pcfg.remote_barrier_rkey = remote_barrier_rkey;

        pcfg.epoch        = s->epoch;
        pcfg.max_inflight = cfg.max_inflight > 0 ? cfg.max_inflight : 512;
        pcfg.sq_depth     = s->sq_depth;
        pcfg.device_id    = cfg.device_id;
        pcfg.pin_proxy    = cfg.pin_proxy;

#ifdef Q2_PROBE_PROXY_TAIL
        // Iter 30 probe: pass the pre-calibrated GPU→host offset to each proxy
        // so account_polled_cmd_for_probe_ can compute enqueue_to_seen deltas
        // in the host clock domain.
        pcfg.gpu_to_host_offset_ns = gpu_to_host_offset_ns;
#endif

        s->proxies[t] = new Proxy(pcfg);
        s->proxies[t]->start();
    }

    if (cfg.rank == 0) {
        fprintf(stderr,
                "internode_efa::Session created (rank=%d, peer=%s:%d, qps=%d, "
                "rails=%d, proxies=%d, logical_q_per_qp=%d)\n",
                cfg.rank, cfg.peer_ip, cfg.tcp_port,
                num_qps, num_rails, num_proxy_threads, logical_queues_per_qp);
    }

    return s;
}

/**
 * Wait for all proxies to finish processing their FIFO commands and complete
 * all in-flight RDMA writes. Call after cudaDeviceSynchronize() so no new FIFO
 * pushes will arrive.
 */
inline void drain_proxy(Session* s) {
    uint64_t fifo_heads[kMaxProxyThreads]{};
    for (int t = 0; t < s->num_proxy_threads; t++) {
        cudaError_t err = cudaMemcpy(
            &fifo_heads[t], s->fifos[t].device.head,
            sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "session_efa: cudaMemcpy(head) failed: %s\n",
                    cudaGetErrorString(err));
            std::abort();
        }
    }

    auto last_log = std::chrono::steady_clock::now();
    for (int i = 0; i < 50000; i++) {  // 5s soft budget, then log+continue
        bool drained = true;
        for (int t = 0; t < s->num_proxy_threads; t++) {
            if (s->fifos[t].host.cpu_head < fifo_heads[t] ||
                s->proxies[t]->inflight() != 0) {
                drained = false;
                break;
            }
        }
        if (drained) return;

        auto now = std::chrono::steady_clock::now();
        if (now - last_log >= std::chrono::seconds(1)) {
            fprintf(stderr, "session_efa: waiting for proxy drain");
            for (int t = 0; t < s->num_proxy_threads; t++) {
                fprintf(stderr, " [t=%d cpu_head=%lu target=%lu inflight=%d]",
                        t,
                        (unsigned long)s->fifos[t].host.cpu_head,
                        (unsigned long)fifo_heads[t],
                        s->proxies[t]->inflight());
            }
            fprintf(stderr, "\n");
            last_log = now;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

inline void destroy_session(Session* s) {
    if (!s) return;

    // Ensure proxies have processed all FIFO commands before shutdown.
    if (s->proxies[0]) {
        cudaDeviceSynchronize();
        drain_proxy(s);
        for (int t = 0; t < s->num_proxy_threads; t++) {
            if (!s->proxies[t]) continue;
            s->proxies[t]->stop();
            delete s->proxies[t];
            s->proxies[t] = nullptr;
        }
    }

    // Deregister per-rail MRs (arrival, barrier, recv_buf, local_data).
    // External recv_buf mode: recv_buf.mr is nullptr (caller-owned buffer),
    // so all per-rail recv_buf MRs must be dereg'd here (free_buffer below
    // is a no-op for external buffers since gpu_ptr is also cleared).
    const bool external_recv_buf = (s->recv_buf.mr == nullptr);
    for (int r = 0; r < s->num_rails; r++) {
        if (s->rails[r].arrival_mr) rdma::dereg_mr(s->rails[r].arrival_mr);
        if (s->rails[r].barrier_mr) rdma::dereg_mr(s->rails[r].barrier_mr);
        if (s->rails[r].local_data_mr) rdma::dereg_mr(s->rails[r].local_data_mr);
        if (s->rails[r].clocal_data_mr) rdma::dereg_mr(s->rails[r].clocal_data_mr);
        // owned mode: rail 0's MR is freed by gpu_mr::free_buffer below.
        // external mode: dereg rail 0 too (free_buffer won't touch it).
        const bool dereg_recv = external_recv_buf || (r > 0);
        if (dereg_recv && s->rails[r].recv_buf_mr) rdma::dereg_mr(s->rails[r].recv_buf_mr);
    }
    // Clear the pointers that gpu_mr::free_buffer and arrival/barrier teardown
    // use so they don't double-free the rail 0 MRs.
    s->arrival.mr = nullptr;
    s->stage_barrier.mr = nullptr;

    // Deregister per-proxy staging MRs (already dereg'd above is fine — the MR
    // pointer in flag_stagings[t].mr is the same object as rails[r].staging_mrs[t]).
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (s->flag_stagings[t].mr) rdma::dereg_mr(s->flag_stagings[t].mr);
        s->flag_stagings[t].mr = nullptr;
    }

    // Free buffers. External recv_buf mode: clear gpu_ptr so free_buffer
    // doesn't cudaFree the caller-owned buffer.
    if (external_recv_buf) {
        s->recv_buf.gpu_ptr = nullptr;
    }
    gpu_mr::free_buffer(s->recv_buf);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        destroy_d2h_fifo(s->fifos[t]);
    }
    destroy_arrival_flags(s->arrival);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        destroy_flag_staging(s->flag_stagings[t]);
    }
    destroy_stage_barrier_flags(s->stage_barrier);

    // Destroy per-rail AHs.
    for (int r = 0; r < s->num_rails; r++) {
        if (s->rails[r].dst_ah) rdma::destroy_ah(s->rails[r].dst_ah);
    }

    // Destroy extra QPs in reverse order.
    for (int i = s->num_qps - 1; i >= 1; i--) {
        if (s->extra_qps[i - 1]) rdma::destroy_qp(s->extra_qps[i - 1]);
    }
    if (s->qp) rdma::destroy_qp(s->qp);

    // Destroy per-proxy CQs.
    for (int t = s->num_proxy_threads - 1; t >= 0; t--) {
        if (s->proxy_cqs[t]) rdma::destroy_cq(s->proxy_cqs[t]);
    }

    // Destroy per-rail PDs and close device contexts.
    for (int r = 0; r < s->num_rails; r++) {
        if (s->rails[r].pd)  rdma::dealloc_pd(s->rails[r].pd);
        if (s->rails[r].ctx) rdma::close_device(s->rails[r].ctx);
    }

    delete s;
}

// ---------------------------------------------------------------------------
// Helpers for kernel globals
// ---------------------------------------------------------------------------

inline D2HFifoDeviceBundle get_fifo_device_handle(const Session* s) {
    return s->fifo_bundle;
}

inline uint32_t* get_arrival_device_ptr(const Session* s) {
    return s->arrival.device_ptr;
}

/**
 * Device pointer to the per-queue tail counters colocated with the arrival
 * flags array. The EFA SRD backend does not publish tails yet, so this returns
 * the trailing tail slots the arrival buffer carries — kernels reading the
 * tail will simply observe zeros.
 */
inline uint32_t* get_arrival_tail_device_ptr(const Session* s) {
    return s ? s->arrival.tail_device_ptr : nullptr;
}

inline void* get_recv_buf_ptr(const Session* s) {
    return s->recv_buf.gpu_ptr;
}

inline int get_num_qps(const Session* s) {
    return s ? s->num_qps : 1;
}

/**
 * Device pointer to the stage-barrier flags — the same host-pinned/device-
 * aliased memory the peer writes tokens into. Kernels spin on this directly.
 */
inline uint32_t* get_stage_barrier_device_ptr(const Session* s) {
    return s ? s->stage_barrier.device_ptr : nullptr;
}

/** Device-side ready-queue handle — SRD does not use it. */
inline ReadyQueueDevice get_ready_queue_device(const Session* /*s*/) {
    return ReadyQueueDevice{};
}

/** Set total expected ready-queue entries — no-op on SRD. */
inline void set_ready_queue_total(Session* /*s*/, uint32_t /*total*/) {}

/** Per-proxy-thread diagnostic counters. (EFAGDA: empty — no proxy threads.) */
inline std::vector<ProxyDiagnostics> get_proxy_diagnostics(const Session* s) {
    std::vector<ProxyDiagnostics> out;
    if (!s) return out;
    out.reserve((size_t)s->num_proxy_threads);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (!s->proxies[t]) continue;
        out.push_back(s->proxies[t]->get_diagnostics());
    }
    return out;
}

/** Per-proxy-thread signaled-post / completion timelines. */
inline std::vector<ProxyTimeline> get_proxy_timelines(const Session* s) {
    std::vector<ProxyTimeline> out;
    if (!s) return out;
    out.reserve((size_t)s->num_proxy_threads);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (!s->proxies[t]) continue;
        out.push_back(s->proxies[t]->get_timeline());
    }
    return out;
}

inline ProxyTimestamps get_proxy_timestamps(const Session* s) {
    return s && s->proxies[0] ? s->proxies[0]->get_timestamps() : ProxyTimestamps{};
}

/**
 * Phase 1 of the epoch transition — mirrors session.h::prepare_epoch.
 * Waits for GPU + all proxies to quiesce, pauses them, drains CQs, resets
 * inflight and timestamps.
 */
inline void prepare_epoch(Session* s) {
    static const bool log_timing = ([]() {
        const char* e = std::getenv("Q2_EPOCH_TIMING");
        return e && e[0] == '1';
    })();
    auto t0 = std::chrono::steady_clock::now();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    drain_proxy(s);
    auto t2 = std::chrono::steady_clock::now();
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->pause();
    }
    auto t3 = std::chrono::steady_clock::now();
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->drain_cq();
        s->proxies[t]->reset_inflight();
        s->proxies[t]->reset_timestamps();
    }
    auto t4 = std::chrono::steady_clock::now();
    if (log_timing) {
        auto us = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
        };
        fprintf(stderr,
                "[EPOCH_TIMING rank=%d] prepare_epoch: cudaSync=%ldus drain_proxy=%ldus "
                "pause=%ldus drain_cq+reset=%ldus total=%ldus\n",
                s->rank, us(t0, t1), us(t1, t2), us(t2, t3), us(t3, t4), us(t0, t4));
    }
}

/**
 * Phase 2 of the epoch transition — mirrors session.h::commit_epoch.
 * Updates epoch, resets arrivals/stage-barrier/FIFOs, resumes all proxies.
 *
 * When OSGC_COMMIT_EPOCH_SKIP_ARRIVAL_RESET=1, skips arrival/stage-barrier
 * resets and FIFO reinit. This is safe under Q2_STEADY_STATE_BENCH because:
 *   - arrival flags: kernel resets them on-GPU via q2_iter_end_reset_arrival_flags
 *     before the epilogue barrier, so they're clean by the time the next iter starts.
 *   - stage_barrier: kernel uses monotonic (>= epoch) checks and never clears,
 *     so host-side zeroing is unnecessary and actually harmful — it races with
 *     incoming BARRIER_NOTIFY RDMA WRITEs from a faster peer that has already
 *     entered its next epoch's kernel.
 *   - FIFOs: poll() clears consumed slots to EMPTY; drain_proxy (in prepare_epoch)
 *     ensures all slots are consumed before we reach commit_epoch.
 */
inline void commit_epoch(Session* s, uint32_t epoch) {
    static const bool log_timing = ([]() {
        const char* e = std::getenv("Q2_EPOCH_TIMING");
        return e && e[0] == '1';
    })();
    auto t0 = std::chrono::steady_clock::now();
    // Read every call — Python sets this env var AFTER the first set_epoch(1)
    // call (line 629 of benchmark_gemm_ar_multinode.py), so a static-const
    // cache would latch false and never pick up the later mutation.
    const char* skip_env = getenv("OSGC_COMMIT_EPOCH_SKIP_ARRIVAL_RESET");
    const bool skip_reset = (skip_env && skip_env[0] == '1');

    s->epoch = epoch;
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->set_epoch(epoch);
    }

    if (!skip_reset) {
        reset_arrival_flags(s->arrival);
        reset_stage_barrier_flags(s->stage_barrier);

        for (int t = 0; t < s->num_proxy_threads; t++) {
            cudaMemset(s->fifos[t].device.head, 0, sizeof(uint64_t));
            cudaMemset(s->fifos[t].device.tail_cache, 0, sizeof(uint64_t));
            *s->fifos[t].host.tail = 0;
            s->fifos[t].host.cpu_head = 0;
            memset(s->fifos[t].host.triggers, 0,
                   s->fifos[t].host.capacity * sizeof(TransferCmd));
        }

        cudaDeviceSynchronize();
    }

    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->resume();
    }
    auto t1 = std::chrono::steady_clock::now();
    if (log_timing) {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        fprintf(stderr,
                "[EPOCH_TIMING rank=%d] commit_epoch: %ldus (skip_reset=%d)\n",
                s->rank, us, (int)skip_reset);
    }
}

/**
 * Host-side stage barrier — send `token` to the peer's barrier slot via SRD,
 * then spin on our own slot until the peer does the same. Mirrors
 * session.h::stage_barrier. Uses proxy 0 / QP 0 on both sides so ordering
 * with the peer is deterministic.
 */
inline void stage_barrier(Session* s, int slot, uint32_t token) {
    if (slot < 0 || slot >= s->stage_barrier.count) {
        fprintf(stderr, "stage_barrier: invalid slot %d\n", slot);
        exit(EXIT_FAILURE);
    }


    cudaDeviceSynchronize();
    drain_proxy(s);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->pause();
        s->proxies[t]->drain_cq();
        s->proxies[t]->reset_inflight();
    }

    s->proxies[0]->post_stage_barrier(slot, token);
    s->proxies[0]->drain_cq();
    s->proxies[0]->reset_inflight();

    for (int i = 0; i < 50000; i++) {
        if (s->stage_barrier.host_ptr[slot] == token) {
            for (int t = 0; t < s->num_proxy_threads; t++) {
                s->proxies[t]->resume();
            }
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->resume();
    }
    fprintf(stderr, "stage_barrier timeout: slot=%d token=%u observed=%u\n",
            slot, token, s->stage_barrier.host_ptr[slot]);
    exit(EXIT_FAILURE);
}

inline bool should_prime_first_launch(Session* /*s*/) {
    const char* env = std::getenv("Q5_PRIME_FIRST_LAUNCH");
    if (env && env[0]) {
        return env[0] == '1';
    }
    const char* fci = std::getenv("Q5_FUSE_COMPUTE_INTRA");
    return fci && fci[0] == '1';
}

inline int prime_first_launch_rounds() {
    const char* env = std::getenv("Q5_PRIME_FIRST_LAUNCH_ROUNDS");
    if (env && env[0]) {
        return std::max(1, std::atoi(env));
    }
    // Four host-injected dummy writes are enough to make the first real
    // fused-compute-intra launch behave like the warmed path at medium sizes.
    return 4;
}

inline void host_publish_transfer_cmd(D2HFifoHost* fifo, uint64_t slot, const TransferCmd& cmd) {
    const uint64_t mask = (uint64_t)(fifo->capacity - 1);
    TransferCmd* dst = &fifo->triggers[slot & mask];
    const char* src_bytes = reinterpret_cast<const char*>(&cmd);
    char* dst_bytes = reinterpret_cast<char*>(dst);
    std::memcpy(dst_bytes + 1, src_bytes + 1, sizeof(TransferCmd) - 1);
    __atomic_store_n(
        reinterpret_cast<uint8_t*>(&dst->cmd_type),
        static_cast<uint8_t>(cmd.cmd_type),
        __ATOMIC_RELEASE);
}

inline void prime_first_launch_transport(Session* s) {
    if (!s || !should_prime_first_launch(s) || s->num_qps <= 0 || s->arrival.tail_count <= 0) {
        return;
    }

    const int qps_per_proxy =
        std::max(1, (s->num_qps + s->num_proxy_threads - 1) / s->num_proxy_threads);
    const uint32_t dummy_tile = (uint32_t)s->arrival.count;  // first tail slot
    const uint32_t tile_bytes = 128u * 256u * 2u;
    const uint32_t prime_bytes = (uint32_t)std::max<uint64_t>(
        4u, std::min<uint64_t>(tile_bytes, s->recv_buf.size / (size_t)std::max(1, s->num_qps)));
    if (dummy_tile > 0xFFFFu) {
        fprintf(stderr,
                "session_efa: skipping first-launch prime; dummy tile %u exceeds TransferCmd tile_id\n",
                dummy_tile);
        return;
    }

    const int rounds = prime_first_launch_rounds();
    for (int round = 0; round < rounds; round++) {
        uint64_t target_heads[kMaxProxyThreads]{};
        int injected[kMaxProxyThreads]{};
        for (int t = 0; t < s->num_proxy_threads; t++) {
            target_heads[t] = s->fifos[t].host.cpu_head;
        }

        for (int global_qp = 0; global_qp < s->num_qps; global_qp++) {
            const int proxy_idx = std::min(s->num_proxy_threads - 1, global_qp / qps_per_proxy);
            D2HFifoHost* fifo = &s->fifos[proxy_idx].host;
            TransferCmd cmd{};
            cmd.cmd_type = CmdType::WRITE;
            cmd.dst_rank = (uint8_t)(s->rank == 0 ? 1 : 0);
            cmd.tile_id = (uint16_t)dummy_tile;
            cmd.bytes = prime_bytes;
            cmd.local_offset = (uint32_t)global_qp * prime_bytes;
            cmd.remote_offset = (uint32_t)global_qp * prime_bytes;
            cmd.lane_id = (uint16_t)global_qp;
            cmd.src_view = 0;
            cmd.reserved0 = 0;
            cmd.row_span = 0;
            cmd.row_count = 0;
            cmd.enqueue_device_ns = 0;
            host_publish_transfer_cmd(fifo, target_heads[proxy_idx], cmd);
            target_heads[proxy_idx]++;
            injected[proxy_idx]++;
        }

        auto last_log = std::chrono::steady_clock::now();
        bool done = false;
        for (int i = 0; i < 50000; i++) {
            done = true;
            for (int t = 0; t < s->num_proxy_threads; t++) {
                if (injected[t] == 0) continue;
                if (s->fifos[t].host.cpu_head < target_heads[t] || s->proxies[t]->inflight() != 0) {
                    done = false;
                    break;
                }
            }
            if (done) break;

            auto now = std::chrono::steady_clock::now();
            if (now - last_log >= std::chrono::seconds(1)) {
                fprintf(stderr, "session_efa: waiting for first-launch transport prime round %d/%d",
                        round + 1, rounds);
                for (int t = 0; t < s->num_proxy_threads; t++) {
                    if (injected[t] == 0) continue;
                    fprintf(stderr, " [t=%d cpu_head=%lu target=%lu inflight=%d]",
                            t,
                            (unsigned long)s->fifos[t].host.cpu_head,
                            (unsigned long)target_heads[t],
                            s->proxies[t]->inflight());
                }
                fprintf(stderr, "\n");
                last_log = now;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (!done) {
            fprintf(stderr, "session_efa: first-launch transport prime timed out in round %d/%d\n",
                    round + 1, rounds);
            break;
        }

        for (int t = 0; t < s->num_proxy_threads; t++) {
            if (injected[t] == 0) continue;
            const uint64_t new_head = target_heads[t];
            OSGC_CUDACHECK(cudaMemcpy(
                s->fifos[t].device.head, &new_head, sizeof(uint64_t),
                cudaMemcpyHostToDevice));
            OSGC_CUDACHECK(cudaMemcpy(
                s->fifos[t].device.tail_cache, &new_head, sizeof(uint64_t),
                cudaMemcpyHostToDevice));
        }
    }

    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (!s->proxies[t]) continue;
        s->proxies[t]->reset_timestamps();
    }
}

/** Legacy single-call epoch transition kept for existing callers. */
inline void set_epoch(Session* s, uint32_t epoch) {
    prepare_epoch(s);
    commit_epoch(s, epoch);
    if (epoch == 1u) {
        prime_first_launch_transport(s);
    }
}

} // namespace internode
