#pragma once

// Session management + pybind module for gemm_ar_multinode.
// Included from src/gemm_ar.cu after the kernel namespace closes.
//
// Note: the kernel namespace `gemm_ar_multinode` is opened+closed entirely
// inside src/gemm_ar.cu; this header lives at top-level (no namespace
// gymnastics) and only defines the pybind handlers and PYBIND11_MODULE.

// ============================================================================
// Session management
// ============================================================================
#include "comm/internode/session_select.h"
#include <cuda_runtime.h>
#include <limits>
#include <torch/csrc/utils/pybind.h>

namespace {

uint64_t gemm_ar_host_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

__global__ void gemm_ar_read_globaltimer_kernel(unsigned long long* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        out[0] = t;
    }
}

pybind11::dict get_globaltimer_calibration_py(int samples = 64) {
    if (samples <= 0) samples = 1;
    unsigned long long* dev_out = nullptr;
    unsigned long long host_out = 0;
    cudaStream_t stream = nullptr;
    OSGC_CUDACHECK(cudaMalloc(&dev_out, sizeof(unsigned long long)));
    OSGC_CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    uint64_t best_host_before_ns = 0;
    uint64_t best_host_after_ns = 0;
    uint64_t best_host_mid_ns = 0;
    uint64_t best_gpu_ns = 0;
    uint64_t best_window_ns = std::numeric_limits<uint64_t>::max();

    for (int i = 0; i < samples; ++i) {
        const uint64_t host_before_ns = gemm_ar_host_now_ns();
        gemm_ar_read_globaltimer_kernel<<<1, 1, 0, stream>>>(dev_out);
        OSGC_CUDACHECK(cudaGetLastError());
        OSGC_CUDACHECK(cudaMemcpyAsync(
            &host_out, dev_out, sizeof(unsigned long long),
            cudaMemcpyDeviceToHost, stream));
        OSGC_CUDACHECK(cudaStreamSynchronize(stream));
        const uint64_t host_after_ns = gemm_ar_host_now_ns();
        const uint64_t window_ns = host_after_ns - host_before_ns;
        if (window_ns < best_window_ns) {
            best_window_ns = window_ns;
            best_host_before_ns = host_before_ns;
            best_host_after_ns = host_after_ns;
            best_host_mid_ns = host_before_ns + window_ns / 2ULL;
            best_gpu_ns = (uint64_t)host_out;
        }
    }

    OSGC_CUDACHECK(cudaStreamDestroy(stream));
    OSGC_CUDACHECK(cudaFree(dev_out));

    pybind11::dict d;
    d["host_before_ns"] = best_host_before_ns;
    d["host_after_ns"] = best_host_after_ns;
    d["host_mid_ns"] = best_host_mid_ns;
    d["gpu_ns"] = best_gpu_ns;
    d["window_ns"] = best_window_ns;
    d["offset_ns"] = (int64_t)best_host_mid_ns - (int64_t)best_gpu_ns;
    return d;
}

} // namespace

static internode::Session* g_session = nullptr;
static std::vector<std::string> g_peer_ips_storage;
static std::vector<const char*> g_peer_ips_cstr;
static std::vector<int>         g_peer_ports_storage;
void create_session_py(int rank, const std::string& peer_ip, int tcp_port,
                       int64_t send_buf_ptr, int64_t send_buf_size,
                       int64_t recv_buf_size, int num_tiles,
                       int fifo_capacity, int device_id,
                       int64_t clocal_buf_ptr = 0,
                       int64_t clocal_buf_size = 0,
                       int64_t row_stride_bytes = 0,
                       std::vector<std::string> peer_ips = {},
                       std::vector<int> peer_tcp_ports = {}) {
    if (g_session) { internode::destroy_session(g_session); g_session = nullptr; }
    internode::SessionConfig cfg{}; cfg.rank = rank; cfg.peer_ip = peer_ip.c_str();
    cfg.tcp_port = tcp_port; cfg.local_gpu_buf = reinterpret_cast<void*>(send_buf_ptr);
    if (!peer_ips.empty()) {
        g_peer_ips_storage   = std::move(peer_ips);
        g_peer_ports_storage = std::move(peer_tcp_ports);
        if (g_peer_ports_storage.empty()) {
            g_peer_ports_storage.assign(g_peer_ips_storage.size(), tcp_port);
        }
        g_peer_ips_cstr.resize(g_peer_ips_storage.size());
        for (size_t i = 0; i < g_peer_ips_storage.size(); ++i) {
            g_peer_ips_cstr[i] = g_peer_ips_storage[i].c_str();
        }
        cfg.num_peers      = (int)g_peer_ips_storage.size();
        cfg.peer_ips       = g_peer_ips_cstr.data();
        cfg.peer_tcp_ports = g_peer_ports_storage.data();
    }
    cfg.local_gpu_buf_size = (size_t)send_buf_size; cfg.recv_buf_size = (size_t)recv_buf_size;
    cfg.num_tiles = num_tiles; cfg.fifo_capacity = fifo_capacity;
    cfg.device_id = device_id; cfg.max_inflight = 256;
    (void)clocal_buf_ptr; (void)clocal_buf_size; (void)row_stride_bytes;
    // EFA/libfabric backends support these extended session fields
    cfg.use_write_imm = false;
    cfg.ready_queue_cap = 0;
    cfg.use_arrival_queue = true;
    cfg.num_qps = 4;
    if (const char* env_num_qps = std::getenv("OSGC_INTERNODE_NUM_QPS")) {
        cfg.num_qps = std::atoi(env_num_qps);
    } else if (const char* env_num_qps = std::getenv("OSGC_IB_NUM_QPS")) {
        cfg.num_qps = std::atoi(env_num_qps);
    } else if (const char* env_num_qps = std::getenv("OSGC_EFA_NUM_QPS")) {
        cfg.num_qps = std::atoi(env_num_qps);
    }
    cfg.logical_queues_per_qp = 1;
    if (const char* env_logical = std::getenv("GEMM_AR_LOGICAL_QUEUES_PER_QP")) {
        cfg.logical_queues_per_qp = std::atoi(env_logical);
    } else if (const char* env_logical = std::getenv("OSGC_INTERNODE_LOGICAL_QUEUES_PER_QP")) {
        cfg.logical_queues_per_qp = std::atoi(env_logical);
    }
    cfg.num_proxy_threads = 1;
    if (const char* env_proxy_threads = std::getenv("OSGC_PROXY_THREADS")) {
        cfg.num_proxy_threads = std::atoi(env_proxy_threads);
    } else if (const char* env_proxy_threads = std::getenv("GEMM_AR_PROXY_THREADS")) {
        cfg.num_proxy_threads = std::atoi(env_proxy_threads);
    }
    g_session = internode::create_session(cfg);
}
void destroy_session_py() { if (g_session) { internode::destroy_session(g_session); g_session = nullptr; } }
void set_epoch_py(int epoch) { if (g_session) internode::set_epoch(g_session, (uint32_t)epoch); }
pybind11::list get_proxy_diagnostics_py() {
    pybind11::list out;
    for (const auto& diag : internode::get_proxy_diagnostics(g_session)) {
        pybind11::dict d;
        d["qp_base_idx"] = diag.qp_base_idx;
        d["num_qps"] = diag.num_qps;
        d["loops"] = diag.loops;
        d["empty_loops"] = diag.empty_loops;
        d["full_batches"] = diag.full_batches;
        d["partial_batches"] = diag.partial_batches;
        d["inflight_limited_loops"] = diag.inflight_limited_loops;
        d["batches_posted"] = diag.batches_posted;
        d["cmds_posted"] = diag.cmds_posted;
        d["cmds_completed"] = diag.cmds_completed;
        d["backlog_samples"] = diag.backlog_samples;
        d["backlog_nonzero_loops"] = diag.backlog_nonzero_loops;
        d["backlog_gt_batch_loops"] = diag.backlog_gt_batch_loops;
        d["backlog_sum"] = diag.backlog_sum;
        d["backlog_max"] = diag.backlog_max;
        d["batch_cmd_to_post_count"] = diag.batch_cmd_to_post_count;
        d["batch_cmd_to_post_sum_ns"] = diag.batch_cmd_to_post_sum_ns;
        d["batch_cmd_to_post_max_ns"] = diag.batch_cmd_to_post_max_ns;
        d["batch_completion_count"] = diag.batch_completion_count;
        d["batch_completion_sum_ns"] = diag.batch_completion_sum_ns;
        d["batch_completion_max_ns"] = diag.batch_completion_max_ns;
        d["enqueue_to_seen_count"] = diag.enqueue_to_seen_count;
        d["enqueue_to_seen_raw_sum_ns"] = diag.enqueue_to_seen_raw_sum_ns;
        d["enqueue_to_seen_raw_max_ns"] = diag.enqueue_to_seen_raw_max_ns;
        d["enqueue_to_seen_raw_min_ns"] = diag.enqueue_to_seen_raw_min_ns;
        d["loop_gap_max_ns"] = diag.loop_gap_max_ns;
        d["loop_gap_over_100us"] = diag.loop_gap_over_100us;
        d["loop_gap_over_1ms"] = diag.loop_gap_over_1ms;
        d["first_cmd_ns"] = diag.first_cmd_ns;
        d["last_cmd_ns"] = diag.last_cmd_ns;
        d["first_post_ns"] = diag.first_post_ns;
        d["last_post_ns"] = diag.last_post_ns;
        d["first_completion_ns"] = diag.first_completion_ns;
        d["last_completion_ns"] = diag.last_completion_ns;
        out.append(d);
    }
    return out;
}
pybind11::list get_proxy_timelines_py() {
    pybind11::list out;
    for (const auto& timeline : internode::get_proxy_timelines(g_session)) {
        pybind11::dict d;
        pybind11::list posted;
        for (uint64_t ts : timeline.signaled_post_ns) posted.append(ts);
        pybind11::list completed;
        for (uint64_t ts : timeline.completion_ns) completed.append(ts);
        d["signaled_post_ns"] = posted;
        d["completion_ns"] = completed;
        out.append(d);
    }
    return out;
}
std::tuple<int64_t,int64_t,int64_t,int64_t,int> get_fifo_handles_py() {
    auto h = internode::get_fifo_device_handle(g_session);
    // GPU-initiated backends (EFAGDA/IBGDA) always carry a bundle; there's
    // no host-pinned FIFO tuple to flatten. Force the bundle-pointer path.
    if (h.num_fifos > 1) {
        return {(int64_t)(&g_session->fifo_bundle), 0, 0, 0, -h.num_fifos};
    }
    auto fd = h.fifos[0];
    return {(int64_t)fd.triggers,(int64_t)fd.head,(int64_t)fd.tail,(int64_t)fd.tail_cache,fd.capacity};
}
int64_t get_arrival_flags_ptr_py() { return (int64_t)internode::get_arrival_device_ptr(g_session); }
int64_t get_arrival_tails_ptr_py() { return (int64_t)internode::get_arrival_tail_device_ptr(g_session); }
void reset_arrival_flags_py() { if (g_session) internode::reset_arrival_flags(g_session->arrival); }
int64_t get_recv_buf_ptr_py() { return (int64_t)internode::get_recv_buf_ptr(g_session); }
int get_num_qps_py() { return internode::get_num_qps(g_session); }

int64_t get_barrier_device_ptr_py() { return (int64_t)internode::get_stage_barrier_device_ptr(g_session); }
std::tuple<int64_t,int64_t,int64_t,int> get_ready_queue_handles_py() {
    auto h = internode::get_ready_queue_device(g_session);
    return {(int64_t)h.entries, (int64_t)h.tail, (int64_t)h.head, (int)h.capacity};
}
void set_ready_queue_total_py(int total) {
    if (g_session) internode::set_ready_queue_total(g_session, (uint32_t)total);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("create_session", &create_session_py,
          pybind11::arg("rank"), pybind11::arg("peer_ip"), pybind11::arg("tcp_port"),
          pybind11::arg("send_buf_ptr"), pybind11::arg("send_buf_size"),
          pybind11::arg("recv_buf_size"), pybind11::arg("num_tiles"),
          pybind11::arg("fifo_capacity"), pybind11::arg("device_id"),
          pybind11::arg("clocal_buf_ptr") = 0,
          pybind11::arg("clocal_buf_size") = 0,
          pybind11::arg("row_stride_bytes") = 0,
          pybind11::arg("peer_ips") = std::vector<std::string>{},
          pybind11::arg("peer_tcp_ports") = std::vector<int>{});
    m.def("destroy_session", &destroy_session_py);
    m.def("set_epoch", &set_epoch_py);
    m.def("get_proxy_diagnostics", &get_proxy_diagnostics_py);
    m.def("get_proxy_timelines", &get_proxy_timelines_py);
    m.def("get_globaltimer_calibration", &get_globaltimer_calibration_py,
          pybind11::arg("samples") = 64);
    m.def("get_fifo_handles", &get_fifo_handles_py);
    m.def("get_arrival_flags_ptr", &get_arrival_flags_ptr_py);
    m.def("get_arrival_tails_ptr", &get_arrival_tails_ptr_py);
    m.def("reset_arrival_flags", &reset_arrival_flags_py);
    m.def("get_recv_buf_ptr", &get_recv_buf_ptr_py);
    m.def("get_num_qps", &get_num_qps_py);
    m.def("get_barrier_device_ptr", &get_barrier_device_ptr_py);
    m.def("get_ready_queue_handles", &get_ready_queue_handles_py);
    m.def("set_ready_queue_total", &set_ready_queue_total_py);
    m.def("gemm_ar_multinode", &gemm_ar_multinode::entrypoint,
          pybind11::arg("A"), pybind11::arg("B"),
          pybind11::arg("C"), pybind11::arg("barrier"),
          pybind11::arg("C_final"),
          pybind11::arg("staging_buf_ptr"), pybind11::arg("recv_buf_ptr"),
          pybind11::arg("fifo_triggers"), pybind11::arg("fifo_head"),
          pybind11::arg("fifo_tail"), pybind11::arg("fifo_tail_cache"),
          pybind11::arg("fifo_capacity"),
          pybind11::arg("arrival_flags_ptr"),
          pybind11::arg("epoch"),
          pybind11::arg("node_idx"),
          pybind11::arg("num_intra_comm_sms"),
          pybind11::arg("num_inter_comm_sms"),
          pybind11::arg("ar_done_ptr"),
          pybind11::arg("arrival_tails_ptr") = (int64_t)0,
          pybind11::arg("scratch_ints") = 0,
          pybind11::arg("num_qps") = 1,
          pybind11::arg("num_remote_queues") = 1,
          pybind11::arg("num_allocated_remote_queues") = 1,
          pybind11::arg("rq_entries_ptr") = (int64_t)0,
          pybind11::arg("rq_tail_ptr") = (int64_t)0,
          pybind11::arg("rq_head_ptr") = (int64_t)0,
          pybind11::arg("rq_capacity") = 0,
          pybind11::arg("rq_total") = 0,
          pybind11::arg("cross_node_barrier_ptr") = (int64_t)0,
          pybind11::arg("trace_slot") = -1,
          pybind11::arg("use_acquire_poll") = false,
          pybind11::arg("num_nodes") = 2);
}
// -- END inlined from gemm_ar_multinode_module.cuh
