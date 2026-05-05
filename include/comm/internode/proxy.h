/**
 * @file proxy.h
 * @brief CPU proxy thread for inter-node RDMA communication.
 *
 * Tight-polling loop: reads TransferCmd from D2H FIFO, posts chained
 * RDMA writes (tile data + arrival flag), polls send CQ, advances FIFO tail.
 *
 * Reference: ~/nfs/ziming/uccl/ep/src/proxy.cpp
 */
#pragma once

#include "types.h"
#include "d2h_fifo.cuh"
#include "arrival.cuh"
#include "rdma_transport.h"
#include "proxy_diagnostics.h"

// The entire CPU-proxy class is inert under IBGDA (GPU posts WQEs directly).
// proxy_diagnostics.h already defines ProxyTimestamps/ProxyDiagnostics/
// ProxyTimeline, so session.h's ifdef-guarded accessors can return empty
// instances under IBGDA without any forward-declaration help.
#if !defined(INTERNODE_BACKEND_IBGDA)

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <limits>
#include <fstream>
#include <string>
#include <thread>
#include <deque>
#include <pthread.h>
#include <sched.h>
#include <cuda_runtime.h>

namespace internode {

struct ProxyConfig {
    D2HFifoHost*  fifo;             // command source (GPU pushes, proxy reads)

    ibv_qp*       qp;              // RC QP 0 to remote node
    ibv_cq*       cq;              // send CQ 0 (for completion polling)

    // Multi-QP support: extra QPs for structural lane->QP routing. All QPs in
    // this proxy thread share cfg_.cq.
    int           num_qps;         // total QPs (1 = single QP mode)
    ibv_qp*       extra_qps[kMaxExchangeQPs - 1];   // QPs 1..num_qps-1
    int           qp_base_idx;     // global QP index of cfg_.qp
    int           global_num_qps;  // total QPs across all proxy threads
    int64_t       gpu_to_host_offset_ns; // host_monotonic_ns - gpu_globaltimer_ns

    // Local data buffer (GPU HBM, RDMA-registered)
    uint64_t      local_data_addr;  // base address
    uint32_t      local_data_lkey;  // MR lkey

    // Optional second MR for direct-from-C_local sends (DMA-BUF only).
    // Populated when Q2_DIRECT_DMABUF_SEND is enabled. If
    // direct_dmabuf_enabled == false, the staging path is the only source.
    uint64_t      clocal_data_addr = 0;
    uint32_t      clocal_data_lkey = 0;
    size_t        clocal_data_bytes = 0;
    bool          direct_dmabuf_enabled = false;
    size_t        row_stride_bytes = 0;
    uint32_t      max_send_sge = 2;

    // Remote data buffer (peer's GPU HBM)
    uint64_t      remote_data_addr; // base address
    uint32_t      remote_data_rkey; // MR rkey

    // Flag staging (host-pinned, holds epoch value for flag RDMA write)
    FlagStaging*  flag_staging;

    // Remote arrival flags (peer's host-pinned memory)
    uint64_t      remote_flags_addr;
    uint32_t      remote_flags_rkey;
    uint64_t      remote_tail_addr;
    uint32_t      remote_tail_rkey;
    uint32_t      remote_queue_stride; // slots per remote logical queue (Q2_ARRIVAL_QUEUE)
    int           logical_queues_per_qp; // number of software queues mapped onto each QP
    bool          enable_remote_tail = false; // publish queue tail after metadata write

    // Remote stage barrier flags (for stage_barrier RDMA write)
    uint64_t      remote_barrier_addr;
    uint32_t      remote_barrier_rkey;

    uint32_t      epoch;           // current epoch value written to remote flags
    int           max_inflight;    // max outstanding RDMA WRs (default 128)

    int           device_id;       // CUDA device index (for NUMA-aware CPU pinning)
    bool          pin_proxy;       // pin proxy thread to GPU's NUMA node (default true)
};

class Proxy {
public:
    /** Maximum number of FIFO commands batched into one ibv_post_send call. */
    static constexpr int BATCH_SIZE = 8;
    static constexpr uint64_t kWrCountMask = 0xffffffffULL;

    explicit Proxy(const ProxyConfig& cfg)
        : cfg_(cfg), running_(false), paused_(false), ack_paused_(false),
          inflight_(0), qp_rr_idx_(0) {
        if (cfg_.max_inflight <= 0) cfg_.max_inflight = 512;
        if (cfg_.num_qps <= 0) cfg_.num_qps = 1;
        if (cfg_.global_num_qps <= 0) cfg_.global_num_qps = cfg_.num_qps;
        if (cfg_.logical_queues_per_qp <= 0) cfg_.logical_queues_per_qp = 1;
        if (cfg_.logical_queues_per_qp > 16) cfg_.logical_queues_per_qp = 16;
        // Each WRITE command posts either two WRs (data + arrival flag) or
        // three WRs when sender-published remote tails are enabled
        // (data + arrival flag + remote tail counter).
        // together in a single ibv_post_send. The QP send queue therefore
        // consumes kWrsPerChunk slots per inflight chunk, while inflight_
        // counts logical chunks.
        // To avoid `ibv_post_send: Cannot allocate memory` (SQ overflow) we
        // must throttle inflight chunks to (SQ_depth / kWrsPerChunk) per QP. The QP is
        // currently created with sq_depth=2048 in session.h, so cap chunks at
        // 1024 per QP in the 2-WR mode and proportionally less in 3-WR mode.
        // We leave a small safety margin for in-flight CQEs that
        // have not yet been polled.
        constexpr int kQpSqDepth = 2048;          // matches create_rc_qp default
        const int kWrsPerChunk = cfg_.enable_remote_tail ? 3 : 2;
        constexpr int kSafetyMargin = 64;         // headroom for unpolled CQEs
        int per_qp_chunk_cap = (kQpSqDepth / kWrsPerChunk) - kSafetyMargin;
        if (per_qp_chunk_cap < 1) per_qp_chunk_cap = 1;
        int requested = cfg_.max_inflight;
        if (requested > per_qp_chunk_cap) requested = per_qp_chunk_cap;
        // Effective inflight scales with QP count
        effective_max_inflight_ = requested * cfg_.num_qps;
        memset(sender_seq_, 0, sizeof(sender_seq_));
        diag_.qp_base_idx = cfg_.qp_base_idx;
        diag_.num_qps = cfg_.num_qps;
    }

    /** Spawn the proxy thread. Optionally pin it to the GPU's NUMA node. */
    void start() {
        running_.store(true, std::memory_order_release);
        thread_ = std::thread(&Proxy::run, this);

        if (cfg_.pin_proxy) {
            pin_thread_to_gpu_numa(cfg_.device_id);
        }
    }

    /** Signal shutdown and join the proxy thread. */
    void stop() {
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

    /**
     * Pause the proxy thread. Blocks until the proxy acknowledges
     * that it has stopped polling. Safe to reset FIFO state after
     * this returns.
     */
    void pause() {
        paused_.store(true, std::memory_order_release);
        while (!ack_paused_.load(std::memory_order_acquire)) {
            // spin until proxy acknowledges pause
        }
    }

    /**
     * Resume the proxy thread after a pause. The proxy will begin
     * polling the FIFO again on the next loop iteration.
     */
    void resume() {
        ack_paused_.store(false, std::memory_order_release);
        paused_.store(false, std::memory_order_release);
    }

    /** Update epoch (call between kernel launches, while proxy is paused). */
    void set_epoch(uint32_t epoch) {
        cfg_.epoch = epoch;
        memset(sender_seq_, 0, sizeof(sender_seq_));
    }

    /** Post a 4-byte RDMA write to the peer's stage barrier slot.
     *
     * Uses IBV_SEND_INLINE so the 4-byte token is copied into the WQE
     * at ibv_post_send time — the NIC does NOT later DMA-read any host
     * memory for the payload. This is critical: previously we pointed
     * the SGE at cfg_.flag_staging->host_ptr[0], but that same buffer
     * is reused by subsequent WRITE batches (see Step 2, line ~749/779
     * in the run loop) which can clobber the token BEFORE the NIC DMAs
     * the barrier WR, causing the peer to spin on a stale/zero value
     * forever. INLINE avoids the DMA-read race entirely.
     */
    void post_stage_barrier(int slot, uint32_t token) {
        barrier_token_ = token;
        ibv_sge sge{};
        sge.addr   = (uint64_t)&barrier_token_;
        sge.length = sizeof(uint32_t);
        sge.lkey   = 0; // ignored for INLINE sends

        ibv_send_wr wr{};
        wr.wr_id   = encode_wr_id(/*local_qp=*/0, /*count=*/1);
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode  = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = cfg_.remote_barrier_addr + (uint64_t)slot * sizeof(uint32_t);
        wr.wr.rdma.rkey        = cfg_.remote_barrier_rkey;
        wr.next = nullptr;

        ibv_send_wr* bad = nullptr;
        int ret = ibv_post_send(cfg_.qp, &wr, &bad);
        if (ret != 0) {
            fprintf(stderr, "proxy: post_stage_barrier failed: %s\n", strerror(ret));
            return;
        }
        inflight_.fetch_add(1, std::memory_order_release);
    }

    /**
     * Post a single strided-direct WR chain: gather row_count rows from
     * C_local (up to max_send_sge rows per WR), chained with arrival flag
     * and optional tail WR. Returns true on successful ibv_post_send.
     *
     * Used when cmd.src_view == 2 (direct DMA-BUF strided). The chain is
     * self-signaling (last WR has SIGNAL) and carries wr_id count=1.
     */
    bool post_strided_direct(const TransferCmd& cmd, int batch_qp,
                              uint32_t total_logical_queues) {
        constexpr uint32_t kTileBytes = 128u * 256u * 2u;
        constexpr int kMaxRows = 128;           // ROW_BLOCK
        constexpr int kMaxDataWrs = 8;          // supports up to 256 rows at max_sge=32
        ibv_send_wr wrs[kMaxDataWrs + 2];
        ibv_sge sges[kMaxRows + 2];
        const uint32_t stride = (uint32_t)cfg_.row_stride_bytes;
        const uint32_t rows = cmd.row_count;
        const uint32_t span = cmd.row_span;
        const uint32_t max_sge = cfg_.max_send_sge > 0 ? cfg_.max_send_sge : 1;
        if (stride == 0 || rows == 0 || span == 0 || rows > kMaxRows) return false;
        const uint64_t base = cfg_.clocal_data_addr + cmd.local_offset;
        const uint64_t remote_base = cfg_.remote_data_addr + cmd.remote_offset;
        const uint32_t n_data_wrs = (rows + max_sge - 1u) / max_sge;
        if ((int)n_data_wrs > kMaxDataWrs) return false;

        uint32_t sge_cursor = 0;
        uint32_t row_cursor = 0;
        uint64_t remote_cursor = remote_base;
        for (uint32_t w = 0; w < n_data_wrs; w++) {
            uint32_t n = std::min(rows - row_cursor, max_sge);
            for (uint32_t k = 0; k < n; k++) {
                sges[sge_cursor + k].addr = base + (uint64_t)(row_cursor + k) * stride;
                sges[sge_cursor + k].length = span;
                sges[sge_cursor + k].lkey = cfg_.clocal_data_lkey;
            }
            ibv_send_wr& wr = wrs[w];
            memset(&wr, 0, sizeof(wr));
            wr.sg_list = &sges[sge_cursor];
            wr.num_sge = (int)n;
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.wr.rdma.remote_addr = remote_cursor;
            wr.wr.rdma.rkey = cfg_.remote_data_rkey;
            wr.send_flags = 0;
            wr.next = &wrs[w + 1];
            sge_cursor += n;
            row_cursor += n;
            remote_cursor += (uint64_t)n * span;
        }
        // Flag WR (INLINE). Layout matches the batched path: under
        // Q2_ARRIVAL_QUEUE we write pack_arrival_work(first_tile, run_tiles)
        // into the logical-queue slot; without it, we write the shared epoch
        // value directly into remote_flags[tile_id] (flat array, one slot per
        // tile) so a kernel that polls arrival_flags[first_tile]==epoch (Q5)
        // makes progress.
        ibv_send_wr& flag_wr = wrs[n_data_wrs];
        memset(&flag_wr, 0, sizeof(flag_wr));
#ifdef Q2_ARRIVAL_QUEUE
        const uint32_t flag_slot = (uint32_t)(BATCH_SIZE * 2);
        const uint32_t run_tiles = (cmd.bytes + kTileBytes - 1u) / kTileBytes;
        cfg_.flag_staging->host_ptr[flag_slot] =
            pack_arrival_work((uint32_t)cmd.tile_id, run_tiles);
        sges[sge_cursor].addr = (uint64_t)(cfg_.flag_staging->host_ptr + flag_slot);
#else
        cfg_.flag_staging->host_ptr[0] = cfg_.epoch;
        sges[sge_cursor].addr = (uint64_t)cfg_.flag_staging->host_ptr;
#endif
        sges[sge_cursor].length = sizeof(uint32_t);
        sges[sge_cursor].lkey = cfg_.flag_staging->mr->lkey;
        flag_wr.sg_list = &sges[sge_cursor++];
        flag_wr.num_sge = 1;
        flag_wr.opcode = IBV_WR_RDMA_WRITE;
#ifdef Q2_ARRIVAL_QUEUE
        const uint32_t logical_q =
            total_logical_queues > 0 ? (cmd.lane_id % total_logical_queues) : 0u;
        const uint32_t q_slot = sender_seq_[logical_q]++;
        flag_wr.wr.rdma.remote_addr = cfg_.remote_flags_addr +
            (uint64_t)(logical_q * cfg_.remote_queue_stride + q_slot) * sizeof(uint32_t);
#else
        flag_wr.wr.rdma.remote_addr = cfg_.remote_flags_addr +
            (uint64_t)cmd.tile_id * sizeof(uint32_t);
#endif
        flag_wr.wr.rdma.rkey = cfg_.remote_flags_rkey;
        flag_wr.send_flags = IBV_SEND_INLINE;

#ifdef Q2_ARRIVAL_QUEUE
        if (cfg_.enable_remote_tail) {
            flag_wr.next = &wrs[n_data_wrs + 1];
            ibv_send_wr& tail_wr = wrs[n_data_wrs + 1];
            memset(&tail_wr, 0, sizeof(tail_wr));
            const uint32_t tail_slot = (uint32_t)(BATCH_SIZE * 2 + 1);
            cfg_.flag_staging->host_ptr[tail_slot] = q_slot + 1u;
            sges[sge_cursor].addr = (uint64_t)(cfg_.flag_staging->host_ptr + tail_slot);
            sges[sge_cursor].length = sizeof(uint32_t);
            sges[sge_cursor].lkey = cfg_.flag_staging->mr->lkey;
            tail_wr.sg_list = &sges[sge_cursor++];
            tail_wr.num_sge = 1;
            tail_wr.opcode = IBV_WR_RDMA_WRITE;
            tail_wr.wr.rdma.remote_addr =
                cfg_.remote_tail_addr + (uint64_t)logical_q * sizeof(uint32_t);
            tail_wr.wr.rdma.rkey = cfg_.remote_tail_rkey;
            tail_wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
            tail_wr.wr_id = encode_wr_id(batch_qp, 1);
            tail_wr.next = nullptr;
        } else
#endif
        {
            flag_wr.send_flags |= IBV_SEND_SIGNALED;
            flag_wr.wr_id = encode_wr_id(batch_qp, 1);
            flag_wr.next = nullptr;
        }

        ibv_qp* post_qp = (batch_qp == 0) ? cfg_.qp : cfg_.extra_qps[batch_qp - 1];
        ibv_send_wr* bad = nullptr;
        int ret = ibv_post_send(post_qp, &wrs[0], &bad);
        if (ret != 0) {
            fprintf(stderr, "proxy: strided post failed (rows=%u span=%u wrs=%u): %s\n",
                    rows, span, n_data_wrs, strerror(ret));
            return false;
        }
        if (__builtin_expect(!strided_logged_, false)) {
            fprintf(stderr, "proxy: q2 dmabuf strided: rows=%u span=%u wrs=%u max_sge=%u\n",
                    rows, span, n_data_wrs, max_sge);
            strided_logged_ = true;
        }
        return true;
    }

    /** Get current inflight WR count (approximate, for drain polling). */
    int inflight() const { return inflight_.load(std::memory_order_acquire); }

    /** Reset the inflight counter (call while proxy is paused). */
    void reset_inflight() { inflight_.store(0, std::memory_order_release); }

    /**
     * Drain all remaining CQ completions. Call ONLY while proxy is paused.
     * This ensures no orphaned completions carry over to the next epoch.
     * Returns the number of completions drained.
     */
    int drain_cq() {
        ibv_wc wc[32];
        int total = 0;
        // Poll CQ until empty or inflight reaches 0
        while (inflight_.load(std::memory_order_acquire) > 0) {
            int ne = rdma::poll_cq(cfg_.cq, 32, wc);
            if (ne <= 0) {
                // CQ empty but inflight > 0: WRs still in NIC, spin briefly
                std::this_thread::yield();
                continue;
            }
            for (int i = 0; i < ne; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "proxy drain_cq: CQ error: wr_id=%lu status=%d (%s)\n",
                            wc[i].wr_id, wc[i].status,
                            ibv_wc_status_str(wc[i].status));
                }
                const int local_qp = decode_wr_local_qp(wc[i].wr_id);
                const int wr_count = decode_wr_count(wc[i].wr_id);
                if (local_qp >= 0 && local_qp < kMaxExchangeQPs &&
                    !batch_post_ns_[local_qp].empty()) {
                    batch_post_ns_[local_qp].pop_front();
                }
                inflight_.fetch_sub(wr_count, std::memory_order_acq_rel);
                total += wr_count;
            }
        }
        return total;
    }

    /** Return a snapshot of the first-message latency timestamps. */
    ProxyTimestamps get_timestamps() const { return ts_; }

    /** Return host-side proxy diagnostics for the current epoch. */
    ProxyDiagnostics get_diagnostics() const { return diag_; }

    /** Return signaled-post and CQE timestamps for the current epoch. */
    ProxyTimeline get_timeline() const {
        ProxyTimeline out;
        out.signaled_post_ns = signaled_post_ns_;
        out.completion_ns = completion_ns_;
        return out;
    }

    /** Reset timestamps (call while proxy is paused, e.g. from set_epoch). */
    void reset_timestamps() {
        ts_.reset();
        diag_.reset();
        diag_.qp_base_idx = cfg_.qp_base_idx;
        diag_.num_qps = cfg_.num_qps;
        for (int q = 0; q < kMaxExchangeQPs; q++) {
            batch_post_ns_[q].clear();
        }
        signaled_post_ns_.clear();
        completion_ns_.clear();
    }

    ~Proxy() { stop(); }

private:
    ProxyConfig cfg_;
    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    std::atomic<bool> ack_paused_;
    std::thread thread_;
    std::atomic<int> inflight_;
    int qp_rr_idx_;               // round-robin QP index
    int effective_max_inflight_;   // max_inflight * num_qps
    ProxyTimestamps ts_;
    uint32_t sender_seq_[kMaxExchangeQPs * 16];     // next remote slot per logical queue
    // Backing storage for stage_barrier's IBV_SEND_INLINE payload. The NIC
    // copies the 4 bytes into the WQE at post time, so this value only needs
    // to be valid for the duration of the ibv_post_send call.
    uint32_t barrier_token_ = 0;
    TransferCmd pending_cmd_{};
    bool has_pending_cmd_ = false;
    bool strided_logged_ = false;

    // Pre-allocated WR/SGE templates — initialized once, only dynamic fields updated per batch.
    ibv_send_wr wrs_[BATCH_SIZE * 3];
    ibv_sge     sges_[BATCH_SIZE * 3];
    bool        wrs_initialized_ = false;
    ProxyDiagnostics diag_;
    std::deque<uint64_t> batch_post_ns_[kMaxExchangeQPs];
    std::vector<uint64_t> signaled_post_ns_;
    std::vector<uint64_t> completion_ns_;

    // Diagnostics: track how the proxy spends its time
    uint64_t diag_total_loops_ = 0;
    uint64_t diag_empty_loops_ = 0;      // FIFO empty
    uint64_t diag_inflight_limited_ = 0; // had FIFO data but inflight >= max
    uint64_t diag_full_batches_ = 0;     // got BATCH_SIZE commands
    uint64_t diag_partial_batches_ = 0;  // got >0 but < BATCH_SIZE

    /**
     * Pin the proxy thread to a CPU core on the same NUMA node as the given GPU.
     * Best-effort: logs a warning and skips pinning if NUMA detection fails.
     */
    void pin_thread_to_gpu_numa(int device_id) {
        // 1. Get GPU PCIe BDF to find its NUMA node
        int domain = 0, bus = 0, dev = 0;
        if (cudaDeviceGetAttribute(&domain, cudaDevAttrPciDomainId, device_id) != cudaSuccess ||
            cudaDeviceGetAttribute(&bus, cudaDevAttrPciBusId, device_id) != cudaSuccess ||
            cudaDeviceGetAttribute(&dev, cudaDevAttrPciDeviceId, device_id) != cudaSuccess) {
            fprintf(stderr, "proxy: pin_proxy: cannot read GPU PCI attrs, skipping\n");
            return;
        }
        char pci[32];
        snprintf(pci, sizeof(pci), "%04x:%02x:%02x.0", domain, bus, dev);
        std::string numa_path = std::string("/sys/bus/pci/devices/") + pci + "/numa_node";
        std::ifstream nf(numa_path);
        int numa = -1;
        if (nf) nf >> numa;
        if (numa < 0) {
            fprintf(stderr, "proxy: pin_proxy: NUMA node unknown for GPU %d, skipping\n", device_id);
            return;
        }

        // 2. Read cpulist for this NUMA node (e.g. "0-15,128-143")
        std::string cpu_path = "/sys/devices/system/node/node" + std::to_string(numa) + "/cpulist";
        std::ifstream cf(cpu_path);
        std::string cpulist;
        if (cf) std::getline(cf, cpulist);
        if (cpulist.empty()) {
            fprintf(stderr, "proxy: pin_proxy: empty cpulist for NUMA %d, skipping\n", numa);
            return;
        }

        // 3. Parse cpulist into a cpu_set_t
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        // Parse comma-separated ranges like "0-15,128-143"
        size_t pos = 0;
        while (pos < cpulist.size()) {
            size_t comma = cpulist.find(',', pos);
            std::string token = cpulist.substr(pos, comma == std::string::npos ? comma : comma - pos);
            size_t dash = token.find('-');
            if (dash != std::string::npos) {
                int lo = std::stoi(token.substr(0, dash));
                int hi = std::stoi(token.substr(dash + 1));
                for (int c = lo; c <= hi; c++) CPU_SET(c, &cpuset);
            } else {
                CPU_SET(std::stoi(token), &cpuset);
            }
            pos = (comma == std::string::npos) ? cpulist.size() : comma + 1;
        }

        // 4. Apply affinity
        int ret = pthread_setaffinity_np(thread_.native_handle(), sizeof(cpu_set_t), &cpuset);
        if (ret != 0) {
            fprintf(stderr, "proxy: pin_proxy: pthread_setaffinity_np failed: %s\n", strerror(ret));
        } else {
            fprintf(stderr, "proxy: pinned proxy thread to NUMA %d cpulist=%s (GPU %d)\n",
                    numa, cpulist.c_str(), device_id);
        }
    }

    /** One-time initialization of WR/SGE templates with constant fields. */
    void init_wr_templates() {
        memset(wrs_, 0, sizeof(wrs_));
        memset(sges_, 0, sizeof(sges_));
        for (int i = 0; i < BATCH_SIZE; i++) {
            const int di = i * 3;      // data WR index
            const int fi = i * 3 + 1;  // flag WR index
            const int ti = i * 3 + 2;  // tail WR index

            // Data WR template
            sges_[di].lkey   = cfg_.local_data_lkey;
            wrs_[di].sg_list = &sges_[di];
            wrs_[di].num_sge = 1;
            wrs_[di].opcode  = IBV_WR_RDMA_WRITE;
            wrs_[di].wr.rdma.rkey = cfg_.remote_data_rkey;
            wrs_[di].next    = &wrs_[fi];

            // Flag WR template
            sges_[fi].addr   = (uint64_t)cfg_.flag_staging->host_ptr;
            sges_[fi].length = sizeof(uint32_t);
            sges_[fi].lkey   = cfg_.flag_staging->mr->lkey;
            wrs_[fi].sg_list = &sges_[fi];
            wrs_[fi].num_sge = 1;
            wrs_[fi].opcode  = IBV_WR_RDMA_WRITE;
            wrs_[fi].wr.rdma.rkey = cfg_.remote_flags_rkey;

            // Optional tail WR template
            sges_[ti].addr   = (uint64_t)(cfg_.flag_staging->host_ptr + BATCH_SIZE);
            sges_[ti].length = sizeof(uint32_t);
            sges_[ti].lkey   = cfg_.flag_staging->mr->lkey;
            wrs_[ti].sg_list = &sges_[ti];
            wrs_[ti].num_sge = 1;
            wrs_[ti].opcode  = IBV_WR_RDMA_WRITE;
            wrs_[ti].wr.rdma.rkey = cfg_.remote_tail_rkey;
        }
        wrs_initialized_ = true;
    }

    /** Read CLOCK_MONOTONIC in nanoseconds. */
    static uint64_t now_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }

    static uint64_t encode_wr_id(int local_qp, int wr_count) {
        return ((uint64_t)(uint32_t)local_qp << 32) | (uint64_t)(uint32_t)wr_count;
    }

    static int decode_wr_local_qp(uint64_t wr_id) {
        return (int)((wr_id >> 32) & 0xffffffffULL);
    }

    static int decode_wr_count(uint64_t wr_id) {
        return (int)(wr_id & kWrCountMask);
    }

    /**
     * Main proxy loop (batched):
     *   1. Collect up to BATCH_SIZE commands from D2H FIFO
     *   2. Build a chained WR list (data+flag per cmd), post in one ibv_post_send
     *   3. Poll send CQ to track completions and manage backpressure
     *   4. Advance FIFO tail to free slots for GPU reuse
     *
     * Batching amortises ibv_post_send overhead: for 128 tiles with BATCH_SIZE=8
     * we issue ~16 posts instead of 128.  Only the last WR in each batch is
     * signaled, so CQ pressure is also reduced.
     *
     * The signaled WR's wr_id encodes the batch size so the CQ poll knows how
     * many logical commands each CQE retires.
     */
    void run() {
        ibv_wc wc[32];
        const uint32_t total_logical_queues =
            (uint32_t)(cfg_.global_num_qps * cfg_.logical_queues_per_qp);
        constexpr int kBacklogProbeCap = 128;
        uint64_t prev_loop_ns = 0;

        while (running_.load(std::memory_order_relaxed)) {
            const uint64_t loop_ns = now_ns();
            if (prev_loop_ns != 0 && loop_ns >= prev_loop_ns) {
                const uint64_t gap_ns = loop_ns - prev_loop_ns;
                if (gap_ns > diag_.loop_gap_max_ns) diag_.loop_gap_max_ns = gap_ns;
                if (gap_ns >= 100000ULL) diag_.loop_gap_over_100us++;
                if (gap_ns >= 1000000ULL) diag_.loop_gap_over_1ms++;
            }
            prev_loop_ns = loop_ns;

            // --- Pause gate: spin here while paused, acknowledge to caller ---
            if (paused_.load(std::memory_order_acquire)) {
                ack_paused_.store(true, std::memory_order_release);
                while (paused_.load(std::memory_order_acquire)) {
                    // Spin until resume() clears paused_
                }
                ack_paused_.store(false, std::memory_order_release);
                // Record proxy resume timestamp for first-message latency analysis
                if (!ts_.recorded) {
                    ts_.proxy_start_ns = now_ns();
                }
                continue;  // re-check running_ and start fresh
            }

            // --- Step 1: Collect a batch of commands from FIFO ---
            TransferCmd batch[BATCH_SIZE];
            int count = 0;
            int batch_qp = qp_rr_idx_;

            int pre_inflight = inflight_.load(std::memory_order_relaxed);
#ifdef Q2_ARRIVAL_QUEUE
            if (has_pending_cmd_ && pre_inflight < effective_max_inflight_) {
                batch[0] = pending_cmd_;
                has_pending_cmd_ = false;
                count = 1;
                const int global_qp =
                    (cfg_.global_num_qps > 0) ? (batch[0].lane_id % cfg_.global_num_qps) : 0;
                batch_qp = global_qp - cfg_.qp_base_idx;
            }
            while (count < BATCH_SIZE && pre_inflight + count < effective_max_inflight_) {
                TransferCmd cmd{};
                if (!cfg_.fifo->poll(&cmd)) break;
                const uint64_t host_seen_ns = now_ns();
                if (cmd.enqueue_device_ns != 0) {
                    int64_t delta_ns =
                        (int64_t)host_seen_ns - (int64_t)cmd.enqueue_device_ns -
                        cfg_.gpu_to_host_offset_ns;
                    if (delta_ns < 0) delta_ns = 0;
                    diag_.enqueue_to_seen_count++;
                    diag_.enqueue_to_seen_raw_sum_ns += delta_ns;
                    if (delta_ns > diag_.enqueue_to_seen_raw_max_ns) {
                        diag_.enqueue_to_seen_raw_max_ns = delta_ns;
                    }
                    if (delta_ns < diag_.enqueue_to_seen_raw_min_ns) {
                        diag_.enqueue_to_seen_raw_min_ns = delta_ns;
                    }
                }
                if (cmd.cmd_type == CmdType::BARRIER_NOTIFY) {
                    post_stage_barrier(/*slot=*/0, cfg_.epoch);
                    continue;
                }
                if (cmd.cmd_type != CmdType::WRITE) continue;
                const int global_qp =
                    (cfg_.global_num_qps > 0) ? (cmd.lane_id % cfg_.global_num_qps) : 0;
                const int cmd_qp = global_qp - cfg_.qp_base_idx;
                if (cmd_qp < 0 || cmd_qp >= cfg_.num_qps) {
                    fprintf(stderr,
                            "proxy: lane %u routed to global_qp=%d outside local range [%d, %d)\n",
                            (unsigned)cmd.lane_id, global_qp, cfg_.qp_base_idx,
                            cfg_.qp_base_idx + cfg_.num_qps);
                    continue;
                }
                if (count == 0) batch_qp = cmd_qp;
                if (cmd_qp != batch_qp) {
                    pending_cmd_ = cmd;
                    has_pending_cmd_ = true;
                    break;
                }
                batch[count++] = cmd;
            }
#else
            while (count < BATCH_SIZE &&
                   pre_inflight + count < effective_max_inflight_ &&
                   cfg_.fifo->poll(&batch[count])) {
                const uint64_t host_seen_ns = now_ns();
                if (batch[count].enqueue_device_ns != 0) {
                    int64_t delta_ns =
                        (int64_t)host_seen_ns - (int64_t)batch[count].enqueue_device_ns -
                        cfg_.gpu_to_host_offset_ns;
                    if (delta_ns < 0) delta_ns = 0;
                    diag_.enqueue_to_seen_count++;
                    diag_.enqueue_to_seen_raw_sum_ns += delta_ns;
                    if (delta_ns > diag_.enqueue_to_seen_raw_max_ns) {
                        diag_.enqueue_to_seen_raw_max_ns = delta_ns;
                    }
                    if (delta_ns < diag_.enqueue_to_seen_raw_min_ns) {
                        diag_.enqueue_to_seen_raw_min_ns = delta_ns;
                    }
                }
                if (batch[count].cmd_type == CmdType::BARRIER_NOTIFY) {
                    post_stage_barrier(/*slot=*/0, cfg_.epoch);
                } else if (batch[count].cmd_type == CmdType::WRITE) {
                    count++;
                }
            }
#endif

            int visible_backlog = count;
            if (has_pending_cmd_) visible_backlog++;
            if (visible_backlog < kBacklogProbeCap) {
                visible_backlog += cfg_.fifo->count_ready(kBacklogProbeCap - visible_backlog);
            }

            diag_total_loops_++;
            diag_.loops++;
            diag_.backlog_samples++;
            diag_.backlog_sum += (uint64_t)visible_backlog;
            if ((uint64_t)visible_backlog > diag_.backlog_max) {
                diag_.backlog_max = (uint64_t)visible_backlog;
            }
            if (visible_backlog > 0) diag_.backlog_nonzero_loops++;
            if (visible_backlog > BATCH_SIZE) diag_.backlog_gt_batch_loops++;
            if (count == 0) {
                diag_empty_loops_++;
                diag_.empty_loops++;
            } else if (count == BATCH_SIZE) {
                diag_full_batches_++;
                diag_.full_batches++;
            } else {
                // Partial batch — why? FIFO empty or inflight limit?
                if (pre_inflight + count >= effective_max_inflight_) {
                    diag_inflight_limited_++;
                    diag_.inflight_limited_loops++;
                } else {
                    diag_partial_batches_++;
                    diag_.partial_batches++;
                }
            }

            // --- Step 1b: Pre-filter strided-direct cmds (src_view=2) and
            // post each as its own WR chain via post_strided_direct(). The
            // batched template below assumes fixed 3-WR-per-cmd layout and
            // can't express variable multi-SGE gather, so strided cmds are
            // handled one at a time. Remaining cmds are compacted in-place.
            int strided_posted = 0;
            if (count > 0 && cfg_.direct_dmabuf_enabled) {
                int new_count = 0;
                for (int i = 0; i < count; i++) {
                    if (batch[i].src_view == 2) {
                        if (post_strided_direct(batch[i], batch_qp, total_logical_queues)) {
                            strided_posted++;
                        }
                    } else {
                        if (new_count != i) batch[new_count] = batch[i];
                        new_count++;
                    }
                }
                count = new_count;
            }

            // --- Step 2: Build chained WR list and post in one call ---
            if (count > 0) {
                // Record timestamp when first FIFO command is seen this epoch
                if (!ts_.recorded) {
                    ts_.first_cmd_ns = now_ns();
                }
                const uint64_t cmd_ns = now_ns();
                if (diag_.first_cmd_ns == 0) diag_.first_cmd_ns = cmd_ns;
                diag_.last_cmd_ns = cmd_ns;

                // One-time init of WR/SGE templates
                if (__builtin_expect(!wrs_initialized_, false)) {
                    init_wr_templates();
                }

        // Ensure metadata staging is current before we build the WRs.
        cfg_.flag_staging->host_ptr[0] = cfg_.epoch;

                // Chained data+flag: one data WR plus one compact arrival
                // metadata WR per command.
                // WR[di]: RDMA WRITE full data to recv_buf
                // WR[fi]: RDMA WRITE packed first_tile_id/run_len metadata to
                // arrival_flags[logical_q][slot]. IB ordering guarantees the
                // metadata is visible only after data.
                // Only update dynamic fields — templates set constant fields once.
                for (int i = 0; i < count; i++) {
                    const TransferCmd& cmd = batch[i];
                    const int di = i * 3;
                    const int fi = i * 3 + 1;
                    const int ti = i * 3 + 2;
                    bool is_last = (i == count - 1);

                    // Data WR: update address + length + remote addr.
                    // If direct-DMABUF is enabled and this command tags
                    // src_view=1, post from the C_local MR instead of staging.
                    if (cfg_.direct_dmabuf_enabled && cmd.src_view == 1) {
                        sges_[di].addr = cfg_.clocal_data_addr + cmd.local_offset;
                        sges_[di].lkey = cfg_.clocal_data_lkey;
                    } else {
                        sges_[di].addr = cfg_.local_data_addr + cmd.local_offset;
                        sges_[di].lkey = cfg_.local_data_lkey;
                    }
                    sges_[di].length = cmd.bytes;
                    wrs_[di].wr.rdma.remote_addr = cfg_.remote_data_addr + cmd.remote_offset;

                    // Metadata WR: update remote readiness addr + signaling + chain
                    // IBV_SEND_INLINE embeds the 4-byte flag in the WQE itself,
                    // avoiding a PCIe DMA read of the staging buffer (~500ns savings).
                    wrs_[fi].wr_id      = is_last ? encode_wr_id(batch_qp, count) : 0;
                    wrs_[fi].send_flags = IBV_SEND_INLINE |
                                          (is_last ? IBV_SEND_SIGNALED : 0);
#ifdef Q2_ARRIVAL_QUEUE
                    constexpr uint32_t kTileBytes = 128u * 256u * 2u;
                    const uint32_t run_tiles = (cmd.bytes + kTileBytes - 1u) / kTileBytes;
                    cfg_.flag_staging->host_ptr[i] =
                        pack_arrival_work((uint32_t)cmd.tile_id, run_tiles);
                    sges_[fi].addr = (uint64_t)(cfg_.flag_staging->host_ptr + i);
                    const uint32_t logical_q =
                        total_logical_queues > 0 ? (cmd.lane_id % total_logical_queues) : 0u;
                    const uint32_t q_slot = sender_seq_[logical_q]++;
                    wrs_[fi].wr.rdma.remote_addr = cfg_.remote_flags_addr +
                                                    (uint64_t)(logical_q * cfg_.remote_queue_stride + q_slot) * sizeof(uint32_t);
                    if (cfg_.enable_remote_tail) {
                        cfg_.flag_staging->host_ptr[BATCH_SIZE + i] = q_slot + 1u;
                        sges_[ti].addr = (uint64_t)(cfg_.flag_staging->host_ptr + BATCH_SIZE + i);
                        wrs_[fi].wr_id = 0;
                        wrs_[fi].send_flags = IBV_SEND_INLINE;
                        wrs_[fi].next = &wrs_[ti];
                        wrs_[ti].wr_id = is_last ? encode_wr_id(batch_qp, count) : 0;
                        wrs_[ti].send_flags = IBV_SEND_INLINE |
                                              (is_last ? IBV_SEND_SIGNALED : 0);
                        wrs_[ti].wr.rdma.remote_addr =
                            cfg_.remote_tail_addr + (uint64_t)logical_q * sizeof(uint32_t);
                        wrs_[ti].next = is_last ? nullptr : &wrs_[(i + 1) * 3];
                    } else {
                        wrs_[fi].wr_id      = is_last ? encode_wr_id(batch_qp, count) : 0;
                        wrs_[fi].send_flags = IBV_SEND_INLINE |
                                              (is_last ? IBV_SEND_SIGNALED : 0);
                        wrs_[fi].next = is_last ? nullptr : &wrs_[(i + 1) * 3];
                    }
#else
                    sges_[fi].addr = (uint64_t)cfg_.flag_staging->host_ptr;
                    wrs_[fi].wr.rdma.remote_addr = cfg_.remote_flags_addr +
                                                    cmd.tile_id * sizeof(uint32_t);
                    wrs_[fi].wr_id      = is_last ? encode_wr_id(batch_qp, count) : 0;
                    wrs_[fi].send_flags = IBV_SEND_INLINE |
                                          (is_last ? IBV_SEND_SIGNALED : 0);
                    wrs_[fi].next = is_last ? nullptr : &wrs_[(i + 1) * 3];
#endif
                }

                ibv_qp* post_qp = (batch_qp == 0) ? cfg_.qp
                    : cfg_.extra_qps[batch_qp - 1];
                ibv_send_wr* bad = nullptr;
                int ret = ibv_post_send(post_qp, &wrs_[0], &bad);
                if (ret != 0) {
                    fprintf(stderr,
                            "proxy: ibv_post_send failed (%d WRs) on local QP%d (global QP%d): %s\n",
                            count, batch_qp, cfg_.qp_base_idx + batch_qp, strerror(ret));
                } else {
                    const uint64_t post_ns = now_ns();
                    const uint64_t cmd_to_post_ns =
                        (post_ns >= cmd_ns) ? (post_ns - cmd_ns) : 0;
                    if (!ts_.recorded) {
                        ts_.first_post_ns = post_ns;
                    }
                    if (diag_.first_post_ns == 0) diag_.first_post_ns = post_ns;
                    diag_.last_post_ns = post_ns;
                    diag_.batch_cmd_to_post_count++;
                    diag_.batch_cmd_to_post_sum_ns += cmd_to_post_ns;
                    if (cmd_to_post_ns > diag_.batch_cmd_to_post_max_ns) {
                        diag_.batch_cmd_to_post_max_ns = cmd_to_post_ns;
                    }
                    diag_.batches_posted++;
                    diag_.cmds_posted += (uint64_t)count;
                    batch_post_ns_[batch_qp].push_back(post_ns);
                    signaled_post_ns_.push_back(post_ns);
                    inflight_.fetch_add(count, std::memory_order_release);
                }
#ifndef Q2_ARRIVAL_QUEUE
                // Advance round-robin to next QP
                qp_rr_idx_ = (qp_rr_idx_ + 1) % cfg_.num_qps;
#endif
            }
            if (strided_posted > 0) {
                inflight_.fetch_add(strided_posted, std::memory_order_release);
                diag_.cmds_posted += (uint64_t)strided_posted;
            }

            // --- Step 3: Poll ALL CQs for completions ---
            {
                int ne = rdma::poll_cq(cfg_.cq, 32, wc);
                if (ne > 0 && !ts_.recorded) {
                    ts_.first_completion_ns = now_ns();
                    ts_.recorded = true;
                }
                if (ne > 0) {
                    const uint64_t comp_ns = now_ns();
                    if (diag_.first_completion_ns == 0) diag_.first_completion_ns = comp_ns;
                    diag_.last_completion_ns = comp_ns;
                }
                for (int i = 0; i < ne; i++) {
                    const int local_qp = decode_wr_local_qp(wc[i].wr_id);
                    const int wr_count = decode_wr_count(wc[i].wr_id);
                    completion_ns_.push_back(diag_.last_completion_ns);
                    if (wc[i].status != IBV_WC_SUCCESS) {
                        fprintf(stderr, "proxy: CQ error: wr_id=%lu local_qp=%d status=%d (%s)\n",
                                wc[i].wr_id, local_qp, wc[i].status,
                                ibv_wc_status_str(wc[i].status));
                    }
                    if (local_qp >= 0 && local_qp < kMaxExchangeQPs &&
                        !batch_post_ns_[local_qp].empty()) {
                        const uint64_t lat_ns =
                            diag_.last_completion_ns - batch_post_ns_[local_qp].front();
                        batch_post_ns_[local_qp].pop_front();
                        diag_.batch_completion_count++;
                        diag_.batch_completion_sum_ns += lat_ns;
                        if (lat_ns > diag_.batch_completion_max_ns) {
                            diag_.batch_completion_max_ns = lat_ns;
                        }
                    }
                    diag_.cmds_completed += (uint64_t)wr_count;
                    inflight_.fetch_sub(wr_count, std::memory_order_acq_rel);
                }
            }

            // --- Step 4: Advance FIFO tail ---
            if (count > 0) {
                cfg_.fifo->advance_tail(cfg_.fifo->cpu_head);
            }
        }

        // Drain remaining inflight WRs before exit
        while (inflight_.load(std::memory_order_acquire) > 0) {
            int ne = rdma::poll_cq(cfg_.cq, 32, wc);
            for (int i = 0; i < ne; i++) {
                inflight_.fetch_sub(decode_wr_count(wc[i].wr_id), std::memory_order_acq_rel);
            }
        }

        // Print diagnostics
        if (diag_total_loops_ > 0) {
            fprintf(stderr, "proxy diag: loops=%lu empty=%lu(%.0f%%) full_batch=%lu partial=%lu inflight_limited=%lu\n",
                    diag_total_loops_, diag_empty_loops_,
                    100.0 * diag_empty_loops_ / diag_total_loops_,
                    diag_full_batches_, diag_partial_batches_, diag_inflight_limited_);
        }
    }
};

} // namespace internode

#endif  // !INTERNODE_BACKEND_IBGDA
