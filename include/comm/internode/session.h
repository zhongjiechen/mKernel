/**
 * @file session.h
 * @brief Session lifecycle for inter-node RDMA communication.
 *
 * One-call setup and teardown. Wires together all internode modules:
 * RDMA transport, GPU MR registration, D2H FIFO, arrival flags, proxy thread.
 *
 * Usage from Python extension host code:
 *   auto* s = internode::create_session({rank, peer_ip, port, gpu_buf, ...});
 *   // ... launch kernel with s->fifo.device, s->arrival.device_ptr ...
 *   internode::destroy_session(s);
 */
#pragma once

#include <chrono>
#include <thread>
#include "types.h"
#include "rdma_transport.h"
#include "rdma_gpu_mr.cuh"
#include "d2h_fifo.cuh"
#include "arrival.cuh"
#include "proxy.h"
#include "ready_queue.cuh"
#if defined(INTERNODE_BACKEND_IBGDA)
#include "ibgda_attach.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace internode {

inline uint64_t q2_host_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

__global__ inline void q2_read_globaltimer_kernel(unsigned long long* out) {
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
        const uint64_t host_before_ns = q2_host_now_ns();
        q2_read_globaltimer_kernel<<<1, 1, 0, stream>>>(dev_out);
        OSGC_CUDACHECK(cudaGetLastError());
        OSGC_CUDACHECK(cudaMemcpyAsync(
            &host_out, dev_out, sizeof(unsigned long long),
            cudaMemcpyDeviceToHost, stream));
        OSGC_CUDACHECK(cudaStreamSynchronize(stream));
        const uint64_t host_after_ns = q2_host_now_ns();
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

// ---------------------------------------------------------------------------
// PCIe-topology-aware NIC selection
// ---------------------------------------------------------------------------
//
// Picks the mlx5 device that shares the same PCIe root complex (NUMA node)
// as the given CUDA device. Falls back to mlx5_{device_id} then mlx5_0 if
// auto-detection fails.
//
// Strategy:
//   1. Read /sys/bus/pci/devices/<gpu_pci>/numa_node for the GPU's NUMA node
//   2. For each /sys/class/infiniband/mlx5_N, read its numa_node
//   3. Match. If multiple match, pick by GPU index modulo number of matches.
inline std::string select_nic_for_device(int device_id) {
    auto read_numa_node = [](const std::string& path) -> int {
        std::ifstream f(path);
        if (!f) return -1;
        int n = -1;
        f >> n;
        return n;
    };

    // 1. Get GPU PCIe BDF from CUDA
    int gpu_domain = 0, gpu_bus = 0, gpu_dev = 0;
    if (cudaDeviceGetAttribute(&gpu_domain, cudaDevAttrPciDomainId, device_id) != cudaSuccess ||
        cudaDeviceGetAttribute(&gpu_bus, cudaDevAttrPciBusId, device_id) != cudaSuccess ||
        cudaDeviceGetAttribute(&gpu_dev, cudaDevAttrPciDeviceId, device_id) != cudaSuccess) {
        return "";
    }
    char gpu_pci[32];
    snprintf(gpu_pci, sizeof(gpu_pci), "%04x:%02x:%02x.0", gpu_domain, gpu_bus, gpu_dev);
    std::string gpu_numa_path = std::string("/sys/bus/pci/devices/") + gpu_pci + "/numa_node";
    int gpu_numa = read_numa_node(gpu_numa_path);

    // 2. Enumerate mlx5 devices and read their NUMA node
    struct NicInfo {
        std::string name;
        int numa_node;
    };
    std::vector<NicInfo> nics;
    DIR* dir = opendir("/sys/class/infiniband");
    if (dir) {
        struct dirent* ent;
        while ((ent = readdir(dir)) != nullptr) {
            std::string name = ent->d_name;
            if (name.rfind("mlx5_", 0) != 0) continue;  // skip non-mlx5
            // Check if port 1 is active (state=4)
            std::string state_path = "/sys/class/infiniband/" + name + "/ports/1/state";
            std::ifstream sf(state_path);
            std::string state_line;
            if (sf) std::getline(sf, state_line);
            if (state_line.find("ACTIVE") == std::string::npos) continue;  // skip inactive
            // Read its NUMA node from the underlying PCI device
            std::string numa_path = "/sys/class/infiniband/" + name + "/device/numa_node";
            int numa = read_numa_node(numa_path);
            nics.push_back({name, numa});
        }
        closedir(dir);
    }

    if (nics.empty()) return "";

    // 3. Filter to NICs matching GPU NUMA, then pick by device_id stride
    std::vector<NicInfo> match;
    for (const auto& n : nics) {
        if (n.numa_node == gpu_numa || gpu_numa < 0 || n.numa_node < 0) {
            match.push_back(n);
        }
    }
    if (match.empty()) match = nics;  // fall back to all

    // Sort by name for determinism
    std::sort(match.begin(), match.end(),
              [](const NicInfo& a, const NicInfo& b) { return a.name < b.name; });

    // Round-robin GPUs across same-NUMA NICs
    return match[device_id % match.size()].name;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct SessionConfig {
    int         rank;                // 0 or 1 (our node rank)
    const char* peer_ip;             // remote node IP (e.g., "38.123.21.6")
    int         tcp_port;            // port for TCP bootstrap (e.g., 18515)

    void*       local_gpu_buf;       // existing GPU buffer to register for RDMA reads
    size_t      local_gpu_buf_size;  // size of local_gpu_buf

    // Optional second buffer to register as a DMA-BUF MR (for direct-from-
    // C_local sends that bypass the staging copy). Required when
    // direct_dmabuf_enabled is true. Must point to a VMM-backed allocation
    // (TKParallelTensor::data_). Registration uses ibv_reg_dmabuf_mr only —
    // no nvidia_peermem / ibv_reg_mr fallback.
    void*       clocal_gpu_buf = nullptr;
    size_t      clocal_gpu_buf_size = 0;
    bool        direct_dmabuf_enabled = false;
    // Row stride in bytes for strided direct-gather sends (src_view=2).
    // Typically N * sizeof(bf16) for Q2 GEMM output. Used by the proxy to
    // compute per-row SGE addresses inside C_local.
    size_t      row_stride_bytes = 0;

    size_t      recv_buf_size;       // size of receive buffer to allocate on GPU
    int         num_tiles;           // number of arrival flag slots
    int         fifo_capacity;       // D2H FIFO capacity (default 1024, must be power of 2)
    int         device_id;           // CUDA device index

    int         max_inflight;        // proxy max outstanding WRs (default 128)

    int         num_qps;             // number of QPs to create (0 or 1 = single QP, max 24)
    int         logical_queues_per_qp = 1; // software queues mapped onto each QP
    int         num_proxy_threads = 1;     // host proxy threads / FIFO channels

    const char* nic_name;            // IB device name (e.g., "mlx5_4"); NULL = auto-select based on device_id

    bool        pin_proxy = true;    // pin proxy thread to GPU's NUMA node (default true)
};

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

/** Maximum number of QPs per session (QP 0 + up to 23 extra). */
static constexpr int kMaxQPs = 24;
static constexpr int kStageBarrierSlots = 2;

struct Session {
    // RDMA resources (QP 0 / CQ 0 — used by proxy today)
    ibv_context*  ctx;
    ibv_pd*       pd;
    ibv_cq*       cq;       // CQ 0
    ibv_qp*       qp;       // QP 0

    // Extra QPs/CQs for future multi-QP support (indices 1..num_qps-1).
    // QPs are indexed by global QP id. CQs are stored per proxy thread and QPs
    // within a proxy thread share the same CQ.
    ibv_qp*       extra_qps[kMaxQPs - 1];  // extra_qps[0] = QP 1, etc.
    ibv_cq*       proxy_cqs[kMaxProxyThreads]; // proxy_cqs[0] aliases cq
    int           num_qps;                  // total QP count (1..kMaxQPs)

    // Local data buffer MR (registered over caller's existing GPU buffer)
    ibv_mr*       local_data_mr;

    // Optional DMA-BUF MR over the C_local (TKParallelTensor data_) buffer.
    // Non-null only when SessionConfig::direct_dmabuf_enabled was set and the
    // strict DMA-BUF registration succeeded.
    ibv_mr*       clocal_data_mr = nullptr;

    // Receive buffer (allocated on GPU, RDMA-registered for remote writes)
    gpu_mr::GpuRdmaBuffer recv_buf;

    // D2H command FIFOs (IBVERBS backend) — one per proxy thread / FIFO channel.
    // Under IBGDA these slots are unused (kept in the struct for binary/layout
    // stability; the device handles live in fifo_bundle, populated by the
    // ibgda runtime instead of create_d2h_fifo).
#if !defined(INTERNODE_BACKEND_IBGDA)
    D2HFifoPair   fifos[kMaxProxyThreads];
#endif
    D2HFifoDeviceBundle fifo_bundle;
    int           num_proxy_threads;

    // Arrival flags + per-proxy metadata staging
    ArrivalFlags  arrival;
    FlagStaging   flag_stagings[kMaxProxyThreads];
    StageBarrierFlags stage_barrier;

    // CPU proxy threads (IBVERBS). Under IBGDA there is no CPU proxy; a
    // lightweight CQ reaper thread lives inside `ibgda_rt` instead.
#if !defined(INTERNODE_BACKEND_IBGDA)
    Proxy*        proxies[kMaxProxyThreads];
#else
    IbgdaRuntime* ibgda_rt = nullptr;
#endif

    // Remote connection info (populated after TCP exchange)
    ConnectionInfo remote_info;

    // Session config (for reference)
    int rank;
    uint32_t epoch;
};

// ---------------------------------------------------------------------------
// Create / Destroy
// ---------------------------------------------------------------------------

/**
 * Create an inter-node RDMA session.
 *
 * Steps:
 *   1. Open IB device, create PD, CQ, RC QP
 *   2. Register local GPU buffer for RDMA (so NIC can read for sends)
 *   3. Allocate + register receive buffer on GPU (remote writes land here)
 *   4. Create D2H FIFO (host-pinned triggers, device-memory head)
 *   5. Create arrival flags (host-pinned, RDMA-registered)
 *   6. Create flag staging (host-pinned, RDMA-registered)
 *   7. Fill local ConnectionInfo, exchange with peer over TCP
 *   8. Transition QP: INIT → RTR → RTS
 *   9. Spawn CPU proxy thread
 */
inline Session* create_session(const SessionConfig& cfg) {
    Session* s = new Session{};
    s->rank = cfg.rank;
    s->epoch = 1;

    // Determine effective QP count (0 treated as 1, clamped to kMaxQPs)
    int effective_num_qps = (cfg.num_qps <= 0) ? 1 : cfg.num_qps;
    if (effective_num_qps > kMaxQPs) effective_num_qps = kMaxQPs;
    s->num_qps = effective_num_qps;
    int requested_proxy_threads = cfg.num_proxy_threads <= 0 ? 1 : cfg.num_proxy_threads;
    if (requested_proxy_threads > kMaxProxyThreads) requested_proxy_threads = kMaxProxyThreads;
    if (requested_proxy_threads > s->num_qps) requested_proxy_threads = s->num_qps;
    s->num_proxy_threads = std::max(1, requested_proxy_threads);
    int logical_queues_per_qp = cfg.logical_queues_per_qp;
    if (logical_queues_per_qp <= 0) logical_queues_per_qp = 1;
    if (logical_queues_per_qp > 16) logical_queues_per_qp = 16;
    const int total_logical_queues = s->num_qps * logical_queues_per_qp;
    const int logical_queue_stride =
        std::max(1, (cfg.num_tiles + total_logical_queues - 1) / total_logical_queues);
    const int total_arrival_slots = logical_queue_stride * total_logical_queues;
    const int qps_per_proxy =
        std::max(1, (s->num_qps + s->num_proxy_threads - 1) / s->num_proxy_threads);

    // Zero-init extra QP/CQ arrays
    memset(s->extra_qps, 0, sizeof(s->extra_qps));
    memset(s->proxy_cqs, 0, sizeof(s->proxy_cqs));
#if !defined(INTERNODE_BACKEND_IBGDA)
    memset(s->proxies, 0, sizeof(s->proxies));
#endif
    memset(s->flag_stagings, 0, sizeof(s->flag_stagings));

    // --- 1. RDMA resources (QP 0 / CQ 0) ---
    // NIC selection priority:
    //   1. Explicit cfg.nic_name (highest priority)
    //   2. PCIe-topology-aware auto-select (matches GPU's NUMA node)
    //   3. mlx5_{device_id} fallback
    //   4. mlx5_0 default
    std::string auto_nic;
    char nic_buf[32];
    const char* nic = cfg.nic_name;
    if (!nic || nic[0] == '\0') {
        auto_nic = select_nic_for_device(cfg.device_id);
        if (!auto_nic.empty()) {
            nic = auto_nic.c_str();
        } else {
            snprintf(nic_buf, sizeof(nic_buf), "mlx5_%d", cfg.device_id);
            nic = nic_buf;
        }
    }
    fprintf(stderr, "session: dev=%d using NIC=%s\n", cfg.device_id, nic);
    s->ctx = rdma::open_device(nic);
    s->pd  = rdma::alloc_pd(s->ctx);
    s->cq  = rdma::create_cq(s->ctx, 4096);
    s->proxy_cqs[0] = s->cq;
    for (int t = 1; t < s->num_proxy_threads; t++) {
        s->proxy_cqs[t] = rdma::create_cq(s->ctx, 4096);
    }
    // QP SGE capacity: 2 default; for DMA-BUF strided gather we need up to 32
    // to pack ROW_BLOCK=128 rows into 4 WRs. Clamped to device cap.
    int qp_max_send_sge = 2;
    if (cfg.direct_dmabuf_enabled) {
        ibv_device_attr dev_attr{};
        if (ibv_query_device(s->ctx, &dev_attr) == 0) {
            qp_max_send_sge = std::min(32, dev_attr.max_sge);
            if (qp_max_send_sge < 8) {
                fprintf(stderr,
                        "session: direct_dmabuf_enabled but device max_sge=%d < 8 — "
                        "cannot support strided gather. Exiting.\n", dev_attr.max_sge);
                exit(EXIT_FAILURE);
            }
        } else {
            qp_max_send_sge = 32;  // hopeful default on mlx5
        }
        fprintf(stderr, "session: q2 dmabuf: max_send_sge=%d row_stride=%zu\n",
                qp_max_send_sge, cfg.row_stride_bytes);
    }
    // SQ depth: IBVERBS proxy batches WRs via ibv_post_send so 2048 is plenty.
    // IBGDA has no per-WQE CQE tracking on the hot path, so the SQ must hold
    // one full iteration's WQEs: 2 per tile (data + flag), divided across QPs.
    int qp_max_send_wr = 2048;
#if defined(INTERNODE_BACKEND_IBGDA)
    {
        // Reserve 25% slack above the worst-case tile count per QP.
        const int tiles_total = std::max(1, cfg.num_tiles);
        const int per_qp = (tiles_total + s->num_qps - 1) / s->num_qps;
        const int needed = per_qp * 2 + per_qp / 4 + 64;
        // libmlx5 rounds up to the nearest power of two internally; clamp to
        // device max_qp_wr to avoid create_qp failures.
        ibv_device_attr da{};
        int max_wr = 32768;  // hopeful cap on mlx5
        if (ibv_query_device(s->ctx, &da) == 0 && da.max_qp_wr > 0) {
            max_wr = da.max_qp_wr;
        }
        qp_max_send_wr = std::min(max_wr, std::max(qp_max_send_wr, needed));
        fprintf(stderr, "session: IBGDA SQ depth per QP=%d (tiles=%d num_qps=%d)\n",
                qp_max_send_wr, tiles_total, s->num_qps);
    }
#endif
    s->qp  = rdma::create_rc_qp(s->pd, s->cq, qp_max_send_wr, qp_max_send_sge);
    rdma::modify_qp_init(s->qp);

    // --- 1b. Extra QPs (indices 1..num_qps-1), sharing one CQ per proxy thread ---
    for (int i = 1; i < s->num_qps; i++) {
        const int owner_thread = std::min(s->num_proxy_threads - 1, i / qps_per_proxy);
        s->extra_qps[i - 1] = rdma::create_rc_qp(s->pd, s->proxy_cqs[owner_thread], qp_max_send_wr, qp_max_send_sge);
        rdma::modify_qp_init(s->extra_qps[i - 1]);
    }

    // --- 2. Register local GPU buffer ---
    // Try DMA-BUF first (works with cudaMalloc and VMM-backed TKParallelTensor,
    // required when Option A routes the staging buffer through a VMM alloc),
    // fall back to nvidia_peermem for cudaMalloc ptrs. Matches session_efa.h.
    // Historical note: raw ibv_reg_mr here fails with "Bad address" when given
    // a VMM pointer, which broke Q5_INTRA_RS_DIRECT_STAGING on CX7 until this
    // call was routed through gpu_mr::register_gpu_buffer (2026-04-23).
    s->local_data_mr = gpu_mr::register_gpu_buffer(
        s->pd, cfg.local_gpu_buf, cfg.local_gpu_buf_size,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
        | IBV_ACCESS_RELAXED_ORDERING);
    if (!s->local_data_mr) {
        fprintf(stderr, "session: register_gpu_buffer(local_gpu_buf) failed: %s\n",
                strerror(errno));
        exit(EXIT_FAILURE);
    }

    // --- 2b. Optional: DMA-BUF-only registration of C_local (direct-send MR) ---
    if (cfg.direct_dmabuf_enabled) {
        if (cfg.clocal_gpu_buf == nullptr || cfg.clocal_gpu_buf_size == 0) {
            fprintf(stderr,
                    "session: direct_dmabuf_enabled but clocal_gpu_buf not provided\n");
            exit(EXIT_FAILURE);
        }
        const char* dmabuf_path = nullptr;
        s->clocal_data_mr = gpu_mr::register_gpu_buffer_dmabuf_only(
            s->pd, cfg.clocal_gpu_buf, cfg.clocal_gpu_buf_size,
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
            | IBV_ACCESS_RELAXED_ORDERING,
            &dmabuf_path);
        if (!s->clocal_data_mr) {
            fprintf(stderr,
                    "session: DMA-BUF-only registration of C_local failed "
                    "(ptr=%p bytes=%zu). No peermem fallback permitted — exiting.\n",
                    cfg.clocal_gpu_buf, cfg.clocal_gpu_buf_size);
            exit(EXIT_FAILURE);
        }
        fprintf(stderr,
                "session: q2 dmabuf: C_local ptr=%p bytes=%zu path=%s lkey=0x%x rkey=0x%x\n",
                cfg.clocal_gpu_buf, cfg.clocal_gpu_buf_size,
                dmabuf_path ? dmabuf_path : "?",
                s->clocal_data_mr->lkey, s->clocal_data_mr->rkey);
    }

    // --- 3. Allocate + register receive buffer ---
    s->recv_buf = gpu_mr::alloc_and_register(
        s->pd, cfg.recv_buf_size, cfg.device_id);

    // --- 4. D2H FIFO ---
#if !defined(INTERNODE_BACKEND_IBGDA)
    int fifo_cap = cfg.fifo_capacity > 0 ? cfg.fifo_capacity : 1024;
    s->fifo_bundle = D2HFifoDeviceBundle{};
    s->fifo_bundle.num_fifos = s->num_proxy_threads;
    s->fifo_bundle.global_num_qps = s->num_qps;
    s->fifo_bundle.logical_queues_per_qp = logical_queues_per_qp;
    s->fifo_bundle.qps_per_fifo = qps_per_proxy;
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->fifos[t] = create_d2h_fifo(fifo_cap);
        s->fifo_bundle.fifos[t] = s->fifos[t].device;
    }
#else
    // IBGDA: D2H FIFO does not exist. The bundle is sized one slot per QP;
    // each slot is filled by the IBGDA runtime after QP transitions (step 9').
    s->fifo_bundle = D2HFifoDeviceBundle{};
    s->fifo_bundle.num_fifos = s->num_qps;
    s->fifo_bundle.global_num_qps = s->num_qps;
    s->fifo_bundle.logical_queues_per_qp = logical_queues_per_qp;
    s->fifo_bundle.qps_per_fifo = 1;
#endif

    // --- 5. Arrival flags ---
    s->arrival = create_arrival_flags(total_arrival_slots, total_logical_queues);
    // Register arrival flags for RDMA (remote writes 4-byte epoch here)
    s->arrival.mr = rdma::reg_mr(s->pd, (void*)s->arrival.device_ptr,
                                  ((size_t)total_arrival_slots + (size_t)total_logical_queues) * sizeof(uint32_t),
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                                  | IBV_ACCESS_RELAXED_ORDERING);

    // --- 6. Per-proxy flag staging (IBVERBS only) ---
#if !defined(INTERNODE_BACKEND_IBGDA)
    for (int t = 0; t < s->num_proxy_threads; t++) {
        // +2 extra slots reserved for strided-direct single-cmd posts
        // (flag and tail scratch, read via IBV_SEND_INLINE).
        s->flag_stagings[t] = create_flag_staging(Proxy::BATCH_SIZE * 2 + 2);
        s->flag_stagings[t].mr = rdma::reg_mr(
            s->pd, s->flag_stagings[t].host_ptr,
            s->flag_stagings[t].count * sizeof(uint32_t),
            IBV_ACCESS_LOCAL_WRITE);
    }
#endif

    // --- 6b. Stage barrier flags ---
    s->stage_barrier = create_stage_barrier_flags(kStageBarrierSlots);
    s->stage_barrier.mr = rdma::reg_mr(s->pd, (void*)s->stage_barrier.host_ptr,
                                       kStageBarrierSlots * sizeof(uint32_t),
                                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                                       | IBV_ACCESS_RELAXED_ORDERING);

    // --- 7. TCP exchange ---
    ConnectionInfo local_info{};
    rdma::fill_local_info(local_info, s->qp, s->ctx);

    // Fill buffer info
    local_info.data_rkey = s->recv_buf.mr->rkey;
    local_info.data_addr = (uint64_t)s->recv_buf.gpu_ptr;
    local_info.data_len  = s->recv_buf.size;
    local_info.flags_rkey = s->arrival.mr->rkey;
    local_info.flags_addr = (uint64_t)s->arrival.device_ptr;
    local_info.tail_rkey = s->arrival.mr->rkey;
    local_info.tail_addr = (uint64_t)s->arrival.tail_device_ptr;
    local_info.barrier_rkey = s->stage_barrier.mr->rkey;
    local_info.barrier_addr = (uint64_t)s->stage_barrier.host_ptr;

    // Multi-QP: fill extra QP info for TCP exchange
    local_info.num_qps = s->num_qps;
    for (int i = 1; i < s->num_qps; i++) {
        local_info.extra_qp_nums[i - 1] = s->extra_qps[i - 1]->qp_num;
        local_info.extra_psns[i - 1] = 0;  // PSN 0 for extra QPs
    }

    bool is_server = (cfg.rank == 0);
    s->remote_info = rdma::exchange_info_tcp(local_info, cfg.peer_ip,
                                              cfg.tcp_port, is_server);

    // --- 8. QP transitions ---
    rdma::modify_qp_rtr(s->qp, s->remote_info);
    rdma::modify_qp_rts(s->qp, local_info.psn);

    // Connect extra QPs (RTR/RTS) using the exchanged QP nums
    for (int i = 1; i < s->num_qps; i++) {
        // Build a ConnectionInfo for this extra QP pair
        ConnectionInfo extra_remote = s->remote_info;
        extra_remote.qp_num = s->remote_info.extra_qp_nums[i - 1];
        extra_remote.psn = s->remote_info.extra_psns[i - 1];
        rdma::modify_qp_rtr(s->extra_qps[i - 1], extra_remote);
        rdma::modify_qp_rts(s->extra_qps[i - 1], local_info.extra_psns[i - 1]);
    }

#if !defined(INTERNODE_BACKEND_IBGDA)
    // --- 9. Proxy thread (IBVERBS only) ---
    const int64_t gpu_to_host_offset_ns = calibrate_gpu_to_host_offset_ns(64);
    ProxyConfig pcfg{};
    for (int t = 0; t < s->num_proxy_threads; t++) {
        const int qp_base = t * qps_per_proxy;
        const int local_qps = std::min(qps_per_proxy, s->num_qps - qp_base);
        if (local_qps <= 0) break;
        ProxyConfig pcfg{};
        pcfg.fifo = &s->fifos[t].host;
        pcfg.qp = (qp_base == 0) ? s->qp : s->extra_qps[qp_base - 1];
        pcfg.cq = s->proxy_cqs[t];
        pcfg.num_qps = local_qps;
        pcfg.qp_base_idx = qp_base;
        pcfg.global_num_qps = s->num_qps;
        memset(pcfg.extra_qps, 0, sizeof(pcfg.extra_qps));
        for (int i = 1; i < local_qps; i++) {
            const int global_qp = qp_base + i;
            pcfg.extra_qps[i - 1] = s->extra_qps[global_qp - 1];
        }
        pcfg.local_data_addr = (uint64_t)cfg.local_gpu_buf;
        pcfg.local_data_lkey = s->local_data_mr->lkey;
        if (s->clocal_data_mr != nullptr) {
            pcfg.clocal_data_addr = (uint64_t)cfg.clocal_gpu_buf;
            pcfg.clocal_data_lkey = s->clocal_data_mr->lkey;
            pcfg.clocal_data_bytes = cfg.clocal_gpu_buf_size;
            pcfg.direct_dmabuf_enabled = true;
            pcfg.row_stride_bytes = cfg.row_stride_bytes;
            pcfg.max_send_sge = (uint32_t)qp_max_send_sge;
        }
        pcfg.remote_data_addr = s->remote_info.data_addr;
        pcfg.remote_data_rkey = s->remote_info.data_rkey;
        pcfg.flag_staging = &s->flag_stagings[t];
        pcfg.remote_flags_addr = s->remote_info.flags_addr;
        pcfg.remote_flags_rkey = s->remote_info.flags_rkey;
        pcfg.remote_tail_addr = s->remote_info.tail_addr;
        pcfg.remote_tail_rkey = s->remote_info.tail_rkey;
        pcfg.use_arrival_queue   = cfg.use_arrival_queue;
        pcfg.remote_queue_stride = (uint32_t)logical_queue_stride;
        pcfg.logical_queues_per_qp = logical_queues_per_qp;
        pcfg.enable_remote_tail = (std::getenv("Q2_SENDER_PUBLISHED_TAIL") != nullptr
                                   && std::getenv("Q2_SENDER_PUBLISHED_TAIL")[0] == '1');
        pcfg.remote_barrier_addr = s->remote_info.barrier_addr;
        pcfg.remote_barrier_rkey = s->remote_info.barrier_rkey;
        pcfg.epoch = s->epoch;
        pcfg.max_inflight = cfg.max_inflight > 0 ? cfg.max_inflight : 512;
        pcfg.device_id = cfg.device_id;
        pcfg.pin_proxy = cfg.pin_proxy;
        pcfg.gpu_to_host_offset_ns = gpu_to_host_offset_ns;

        s->proxies[t] = new Proxy(pcfg);
        s->proxies[t]->start();
    }
#else
    // --- 9'. IBGDA runtime: attach each QP, upload peer table, start reaper ---
    {
        s->ibgda_rt = new IbgdaRuntime{};
        s->ibgda_rt->num_qps = s->num_qps;
        s->ibgda_rt->qps     = new IbgdaQpState[s->num_qps];
        // Attach QP 0 on its CQ, QPs 1..N-1 on the primary CQ too
        // (each gets its own dbrec/SQ/BF regardless of CQ sharing).
        auto qp_of = [&](int i) -> ibv_qp* {
            return (i == 0) ? s->qp : s->extra_qps[i - 1];
        };
        auto cq_of = [&](int i) -> ibv_cq* {
            // Match IBVERBS proxy routing: ceil(num_qps / num_proxy_threads)
            // QPs per CQ, one CQ per proxy thread.
            int owner = std::min(s->num_proxy_threads - 1, i / qps_per_proxy);
            return s->proxy_cqs[owner];
        };
        for (int i = 0; i < s->num_qps; ++i) {
            ibv_qp* qp = qp_of(i);
            ibv_cq* cq = cq_of(i);
            if (ibgda_attach_qp(s->ibgda_rt->qps[i], qp, cq, s->pd) != 0) {
                fprintf(stderr, "session: ibgda_attach_qp %d failed — aborting\n", i);
                exit(EXIT_FAILURE);
            }
        }
        // Peer table on device.
        s->ibgda_rt->d_peers = ibgda_upload_peer_table(s->remote_info, cfg.rank);
        s->ibgda_rt->num_peers = kIbgdaMaxPeers;
        // Populate fifo_bundle.fifos[i] for each QP.
        const uint64_t local_base = (uint64_t)cfg.local_gpu_buf;
        const uint32_t lkey_data  = s->local_data_mr->lkey;
        for (int i = 0; i < s->num_qps; ++i) {
            ibgda_fill_device_handle(s->fifo_bundle.fifos[i],
                                     s->ibgda_rt->qps[i],
                                     local_base, lkey_data,
                                     s->ibgda_rt->d_peers);
        }
        ibgda_runtime_init_done(*s->ibgda_rt);  // no reaper thread
    }
#endif

    if (cfg.rank == 0) {
        fprintf(stderr, "internode::Session created (rank=%d, peer=%s:%d)\n",
                cfg.rank, cfg.peer_ip, cfg.tcp_port);
    }

    return s;
}

/**
 * Wait for the proxy to finish processing all FIFO commands and complete
 * all in-flight RDMA writes. Call after cudaDeviceSynchronize() to ensure
 * no new FIFO pushes will arrive.
 */
inline void drain_proxy(Session* s) {
#if !defined(INTERNODE_BACKEND_IBGDA)
    uint64_t fifo_heads[kMaxProxyThreads]{};
    for (int t = 0; t < s->num_proxy_threads; t++) {
        cudaMemcpy(&fifo_heads[t], s->fifos[t].device.head, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    // Wait until proxy consumed all FIFO commands and all WRs completed.
    for (int i = 0; i < 50000; i++) {  // 5s timeout
        bool drained = true;
        for (int t = 0; t < s->num_proxy_threads; t++) {
            if (s->fifos[t].host.cpu_head < fifo_heads[t] || s->proxies[t]->inflight() != 0) {
                drained = false;
                break;
            }
        }
        if (drained)
            break;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
#else
    // IBGDA: host polls each CQ once to flush pending completions and surface
    // any WQE errors. No per-WQE counter — SQ is sized so push() never
    // overruns within an iteration.
    if (!s->ibgda_rt) return;
    // Read each QP's pi; use as the expected CQE count so we block until all
    // outstanding signaled WQEs retire.
    std::vector<uint64_t> need(s->ibgda_rt->num_qps, 0);
    for (int i = 0; i < s->ibgda_rt->num_qps; ++i) {
        IbgdaQpState& q = s->ibgda_rt->qps[i];
        if (!q.d_pi_counter) continue;
        uint64_t pi = 0;
        cudaMemcpy(&pi, q.d_pi_counter, sizeof(pi), cudaMemcpyDeviceToHost);
        need[i] = pi;
    }
    ibgda_drain_cqs_host(*s->ibgda_rt, need.data(), /*max_spin_ms=*/5000);
#endif
}

/**
 * Destroy an inter-node session. Stops proxy, deregisters MRs, frees everything.
 */
inline void destroy_session(Session* s) {
    if (!s) return;

#if !defined(INTERNODE_BACKEND_IBGDA)
    // Ensure proxy has processed all FIFO commands and RDMA writes before shutdown.
    // Without this, the remote node may never receive some tile arrivals.
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
#else
    // IBGDA: drain all CQEs, then tear down attachments (no reaper thread).
    if (s->ibgda_rt) {
        cudaDeviceSynchronize();
        std::vector<uint64_t> need(s->ibgda_rt->num_qps, 0);
        for (int i = 0; i < s->ibgda_rt->num_qps; ++i) {
            IbgdaQpState& q = s->ibgda_rt->qps[i];
            if (!q.d_pi_counter) continue;
            uint64_t pi = 0;
            cudaMemcpy(&pi, q.d_pi_counter, sizeof(pi), cudaMemcpyDeviceToHost);
            need[i] = pi;
        }
        ibgda_drain_cqs_host(*s->ibgda_rt, need.data(), /*max_spin_ms=*/5000);
        ibgda_runtime_teardown(*s->ibgda_rt);
        delete s->ibgda_rt; s->ibgda_rt = nullptr;
    }
#endif

    // Deregister MRs
    if (s->arrival.mr)     rdma::dereg_mr(s->arrival.mr);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (s->flag_stagings[t].mr) rdma::dereg_mr(s->flag_stagings[t].mr);
    }
    if (s->stage_barrier.mr) rdma::dereg_mr(s->stage_barrier.mr);
    if (s->clocal_data_mr) rdma::dereg_mr(s->clocal_data_mr);
    if (s->local_data_mr)  rdma::dereg_mr(s->local_data_mr);

    // Free buffers
    gpu_mr::free_buffer(s->recv_buf);
#if !defined(INTERNODE_BACKEND_IBGDA)
    for (int t = 0; t < s->num_proxy_threads; t++) {
        destroy_d2h_fifo(s->fifos[t]);
    }
#endif
    destroy_arrival_flags(s->arrival);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        destroy_flag_staging(s->flag_stagings[t]);
    }
    destroy_stage_barrier_flags(s->stage_barrier);

    // Destroy extra QPs (indices 1..num_qps-1, in reverse order)
    for (int i = s->num_qps - 1; i >= 1; i--) {
        rdma::destroy_qp(s->extra_qps[i - 1]);
    }

    // Destroy per-proxy shared CQs.
    for (int t = s->num_proxy_threads - 1; t >= 1; t--) {
        rdma::destroy_cq(s->proxy_cqs[t]);
    }

    // Destroy primary RDMA resources (QP 0 / CQ 0)
    rdma::destroy_qp(s->qp);
    rdma::destroy_cq(s->cq);
    rdma::dealloc_pd(s->pd);
    rdma::close_device(s->ctx);

    delete s;
}

// ---------------------------------------------------------------------------
// Helpers for kernel globals
// ---------------------------------------------------------------------------

/** Get the device-side FIFO handle to pass into kernel globals. */
inline D2HFifoDeviceBundle get_fifo_device_handle(const Session* s) {
    return s->fifo_bundle;
}

/** Get the device-accessible arrival flags pointer for kernel globals. */
inline uint32_t* get_arrival_device_ptr(const Session* s) {
    return s->arrival.device_ptr;
}

inline uint32_t* get_arrival_tail_device_ptr(const Session* s) {
    return s ? s->arrival.tail_device_ptr : nullptr;
}

/** Get the receive buffer device pointer. */
inline void* get_recv_buf_ptr(const Session* s) {
    return s->recv_buf.gpu_ptr;
}

inline int get_num_qps(const Session* s) {
    return s ? s->num_qps : 1;
}

inline uint32_t* get_stage_barrier_device_ptr(const Session* s) {
    return s ? s->stage_barrier.device_ptr : nullptr;
}

/** Get the device-side ready queue handle (stub — returns empty handle). */
inline ReadyQueueDevice get_ready_queue_device(const Session* /*s*/) {
    return ReadyQueueDevice{};
}

/** Set the total expected items for the ready queue (stub — no-op). */
inline void set_ready_queue_total(Session* /*s*/, uint32_t /*total*/) {}

/**
 * Phase 1 of the epoch transition: wait for GPU work and proxy RDMA writes
 * to drain, then pause the proxy so no new FIFO commands are consumed.
 *
 * This allows the caller to perform a cross-rank barrier while both nodes are
 * fully quiesced before either one resets arrival flags or FIFO state.
 */
inline void prepare_epoch(Session* s) {
    // Wait for GPU to finish any pending work — no new FIFO pushes after this.
    cudaDeviceSynchronize();

#if !defined(INTERNODE_BACKEND_IBGDA)
    // Wait for proxy to consume all FIFO commands and drain all RDMA writes.
    drain_proxy(s);

    // Pause proxy so it stops polling the FIFO during reset.
    // The proxy finishes its current loop iteration before acknowledging.
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->pause();
    }

    // Drain any CQ completions for WRs the proxy posted in its final
    // loop iteration (between the drain check and pause acknowledgment).
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->drain_cq();
    }

    // Reset proxy inflight counter (guaranteed 0 after drain_cq)
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->reset_inflight();
    }

    // Reset latency timestamps for the next epoch while the proxy is paused.
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->reset_timestamps();
    }
#else
    // IBGDA: host polls each CQ once to flush pending completions. No per-WQE
    // counter; SQ is sized so push() never overruns within an iteration.
    if (s->ibgda_rt) {
        std::vector<uint64_t> need(s->ibgda_rt->num_qps, 0);
        for (int i = 0; i < s->ibgda_rt->num_qps; ++i) {
            IbgdaQpState& q = s->ibgda_rt->qps[i];
            if (!q.d_pi_counter) continue;
            uint64_t pi = 0;
            cudaMemcpy(&pi, q.d_pi_counter, sizeof(pi), cudaMemcpyDeviceToHost);
            need[i] = pi;
        }
        ibgda_drain_cqs_host(*s->ibgda_rt, need.data(), /*max_spin_ms=*/5000);
    }
#endif
}

/**
 * Phase 2 of the epoch transition: update epoch metadata, reset arrival flags
 * and FIFO state, then resume the proxy after the reset is visible.
 *
 * Caller must invoke prepare_epoch() first and typically synchronize ranks
 * between the two phases.
 */
inline void commit_epoch(Session* s, uint32_t epoch) {
    // Update epoch on session and proxy
    s->epoch = epoch;
#if !defined(INTERNODE_BACKEND_IBGDA)
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->set_epoch(epoch);
    }
#endif

    // Reset arrival flags.
    // NOTE: we intentionally do NOT reset stage_barrier here. The slot is
    // monotonically written (proxy's BARRIER_NOTIFY handler writes cfg_.epoch;
    // host-side stage_barrier() writes monotonically-increasing stage tokens).
    // Resetting it races with in-flight RDMA writes from a peer that has already
    // advanced to the new epoch — the reset can clobber an arriving token and
    // cause a lost-signal deadlock in steady-state benchmarking, because the
    // sender only issues one BARRIER_NOTIFY per iter and never retransmits.
    // Kernel waiters use >= comparison against the current epoch, so stale
    // larger values are harmless (they satisfy the condition immediately, which
    // is correct — the peer is ahead).
    //
    // reset_arrival_flags: also removed from the default path for the same
    // clobber reason — by the time this host memset fires, the peer (which
    // finished its iter boundary earlier) may have already RDMA-written
    // next-iter flag values into our host-pinned arrival array; memset here
    // would zero them and deadlock our next-iter kernel which spins on
    // `flag_val != 0`. Callers that need a reset must do it stream-ordered
    // on the GPU at iter-end BEFORE the cross-node barrier (see Q2 kernels'
    // q2_iter_end_reset_arrival_flags), or call reset_arrival_flags()
    // explicitly in a regime where no in-flight peer writes can race.
    if (std::getenv("OSGC_COMMIT_EPOCH_SKIP_ARRIVAL_RESET") == nullptr) {
        reset_arrival_flags(s->arrival);
    }

#if !defined(INTERNODE_BACKEND_IBGDA)
    // Reset FIFO: zero head (device mem), zero tail (host mem), clear trigger slots
    for (int t = 0; t < s->num_proxy_threads; t++) {
        cudaMemset(s->fifos[t].device.head, 0, sizeof(uint64_t));
        cudaMemset(s->fifos[t].device.tail_cache, 0, sizeof(uint64_t));
        *s->fifos[t].host.tail = 0;
        s->fifos[t].host.cpu_head = 0;
        memset(s->fifos[t].host.triggers, 0,
               s->fifos[t].host.capacity * sizeof(TransferCmd));
    }

    cudaDeviceSynchronize();

    // Resume proxy — it will start polling the now-clean FIFO
    for (int t = 0; t < s->num_proxy_threads; t++) {
        s->proxies[t]->resume();
    }
#else
    // IBGDA: nothing FIFO-like to reset. The per-QP pi/ci counters should
    // already match (prepare_epoch drained to ci == pi). Leave them —
    // monotonic counters wrap naturally at 2^64.
    (void)s;
    cudaDeviceSynchronize();
#endif
}

/**
 * Host-side stage barrier backed by a 4-byte proxy-issued RDMA write.
 *
 * This mirrors the intranode ring-stage barrier at launch boundaries:
 * 1. wait for local kernels and RDMA traffic to quiesce
 * 2. pause the proxy so we can safely use the same QP from the host thread
 * 3. send our stage token to the peer's barrier slot
 * 4. wait until the peer sends the same token to our local barrier slot
 * 5. resume the proxy for the next stage
 */
inline void stage_barrier(Session* s, int slot, uint32_t token) {
    if (slot < 0 || slot >= s->stage_barrier.count) {
        fprintf(stderr, "stage_barrier: invalid slot %d\n", slot);
        exit(EXIT_FAILURE);
    }

#if !defined(INTERNODE_BACKEND_IBGDA)
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
#else
    // IBGDA: drain any GPU-side WQEs, then use host-side ibv_post_send on QP 0
    // to publish the barrier token. The SQ is quiescent after
    // cudaDeviceSynchronize + reaper catches up, so ibv_post_send on the
    // shared QP is safe. After the send completes we poll our arrival slot.
    // NOTE: libmlx5 will advance its internal SQ PI by 1 here, which means
    // on the next GPU push, our d_pi_counter and libmlx5's PI disagree.
    // To keep them in sync we bump d_pi_counter and dbrec_ring_cursor by 1
    // after the post — so device arithmetic "slot = pi++" still matches.
    cudaDeviceSynchronize();
    if (s->ibgda_rt) {
        std::vector<uint64_t> need(s->ibgda_rt->num_qps, 0);
        for (int i = 0; i < s->ibgda_rt->num_qps; ++i) {
            IbgdaQpState& q = s->ibgda_rt->qps[i];
            if (!q.d_pi_counter) continue;
            uint64_t pi = 0;
            cudaMemcpy(&pi, q.d_pi_counter, sizeof(pi), cudaMemcpyDeviceToHost);
            need[i] = pi;
        }
        ibgda_drain_cqs_host(*s->ibgda_rt, need.data(), /*max_spin_ms=*/5000);
    }
    // Build a host-side 4-byte RDMA_WRITE of `token` into the peer's
    // stage_barrier slot using ibv_post_send on QP 0.
    static uint32_t token_scratch;
    token_scratch = token;
    // Register a tiny MR for the token if we haven't already (leaked at
    // session teardown — acceptable for barrier-rate operations).
    static ibv_mr* token_mr = nullptr;
    static Session* token_mr_session = nullptr;
    if (token_mr_session != s) {
        if (token_mr) ibv_dereg_mr(token_mr);
        token_mr = ibv_reg_mr(s->pd, &token_scratch, sizeof(token_scratch),
                              IBV_ACCESS_LOCAL_WRITE);
        token_mr_session = s;
        if (!token_mr) {
            fprintf(stderr, "stage_barrier(IBGDA): ibv_reg_mr failed: %s\n",
                    strerror(errno));
            exit(EXIT_FAILURE);
        }
    }
    ibv_sge sge{};
    sge.addr   = (uintptr_t)&token_scratch;
    sge.length = sizeof(token_scratch);
    sge.lkey   = token_mr->lkey;
    ibv_send_wr wr{}, *bad = nullptr;
    wr.opcode             = IBV_WR_RDMA_WRITE;
    wr.send_flags         = IBV_SEND_SIGNALED;
    wr.sg_list            = &sge;
    wr.num_sge            = 1;
    wr.wr.rdma.remote_addr = s->remote_info.barrier_addr + slot * sizeof(uint32_t);
    wr.wr.rdma.rkey        = s->remote_info.barrier_rkey;
    if (ibv_post_send(s->qp, &wr, &bad) != 0) {
        fprintf(stderr, "stage_barrier(IBGDA): ibv_post_send failed: %s\n",
                strerror(errno));
        exit(EXIT_FAILURE);
    }
    // Poll CQ for our completion (take ownership of just this one CQE).
    ibv_wc wc{};
    for (int spin = 0; spin < 500000; ++spin) {
        int n = ibv_poll_cq(s->cq, 1, &wc);
        if (n == 1) {
            if (wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "stage_barrier(IBGDA): CQE status=%d\n",
                        (int)wc.status);
                exit(EXIT_FAILURE);
            }
            // Account for the host-posted SQ slot in our device counters
            // so the GPU's next push() computes the correct pi. (No ci
            // counter under the reaper-less design; retired_shadow for the
            // drain tracking is already bumped by ibv_poll_cq above.)
            if (s->ibgda_rt && s->ibgda_rt->num_qps > 0) {
                IbgdaQpState& q = s->ibgda_rt->qps[0];
                uint64_t pi = 0;
                cudaMemcpy(&pi, q.d_pi_counter, sizeof(pi), cudaMemcpyDeviceToHost);
                pi += 1;
                cudaMemcpy(q.d_pi_counter, &pi, sizeof(pi), cudaMemcpyHostToDevice);
                cudaMemcpy(q.d_dbrec_ring_cursor, &pi, sizeof(pi), cudaMemcpyHostToDevice);
                q.retired_shadow += 1;
            }
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(2));
    }
    for (int i = 0; i < 50000; i++) {
        if (s->stage_barrier.host_ptr[slot] == token) return;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    fprintf(stderr, "stage_barrier(IBGDA) timeout: slot=%d token=%u observed=%u\n",
            slot, token, s->stage_barrier.host_ptr[slot]);
    exit(EXIT_FAILURE);
#endif
}

/**
 * Legacy single-call epoch transition kept for existing callers.
 */
inline void set_epoch(Session* s, uint32_t epoch) {
    prepare_epoch(s);
    commit_epoch(s, epoch);
}

/** Get proxy first-message latency timestamps for diagnostics. */
inline ProxyTimestamps get_proxy_timestamps(const Session* s) {
#if !defined(INTERNODE_BACKEND_IBGDA)
    return s->proxies[0]->get_timestamps();
#else
    (void)s;
    return ProxyTimestamps{};
#endif
}

inline std::vector<ProxyDiagnostics> get_proxy_diagnostics(const Session* s) {
    std::vector<ProxyDiagnostics> out;
    if (!s) return out;
#if !defined(INTERNODE_BACKEND_IBGDA)
    out.reserve((size_t)s->num_proxy_threads);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (s->proxies[t] == nullptr) continue;
        out.push_back(s->proxies[t]->get_diagnostics());
    }
#endif
    return out;
}

inline std::vector<ProxyTimeline> get_proxy_timelines(const Session* s) {
    std::vector<ProxyTimeline> out;
    if (!s) return out;
#if !defined(INTERNODE_BACKEND_IBGDA)
    out.reserve((size_t)s->num_proxy_threads);
    for (int t = 0; t < s->num_proxy_threads; t++) {
        if (s->proxies[t] == nullptr) continue;
        out.push_back(s->proxies[t]->get_timeline());
    }
#endif
    return out;
}

} // namespace internode
