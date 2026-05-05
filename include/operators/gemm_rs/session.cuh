#pragma once

// Session management + pybind module for gemm_rs_multinode.
// Included from src/gemm_rs.cu after the kernel namespace closes.

// ============================================================================
// Session management (same as v1)
// ============================================================================
#include "comm/internode/session_select.h"

static internode::Session* g_session = nullptr;

void create_session_py(int rank, const std::string& peer_ip, int tcp_port,
                       int64_t send_buf_ptr, int64_t send_buf_size,
                       int64_t recv_buf_size, int num_tiles,
                       int fifo_capacity, int device_id,
                       int64_t output_buf_ptr = 0, int64_t output_buf_size = 0,
                       int output_n = 0) {
    if (g_session) { internode::destroy_session(g_session); g_session = nullptr; }
    internode::SessionConfig cfg{};
    cfg.rank = rank;
    cfg.peer_ip = peer_ip.c_str();
    cfg.tcp_port = tcp_port;
    cfg.local_gpu_buf = reinterpret_cast<void*>(send_buf_ptr);
    cfg.local_gpu_buf_size = (size_t)send_buf_size;
    cfg.recv_buf_size = (size_t)recv_buf_size;
    cfg.num_tiles = num_tiles;
    cfg.fifo_capacity = fifo_capacity;
    cfg.device_id = device_id;

    // GEMM_RS_DIRECT_DMABUF_SEND: register output_local as a DMA-BUF MR on every
    // rail's PD so the send path can skip the pack + gather step. Session
    // creation hard-fails if the buffer can't be DMA-BUF-exported (no peermem
    // fallback — we want the direct path or nothing).
    cfg.max_inflight = 256;
    // Round 26: default to 4 QPs per endpoint to match the gemm_ar round-13 win
    // (51e19e8). Multi-NIC striping is also enabled by default via the
    // round-18 cluster default OSGC_EFA_NUM_NICS=2 in session_fi.h.
    cfg.num_qps = 4;
    if (const char* env_num_qps = std::getenv("OSGC_EFA_NUM_QPS")) {
        cfg.num_qps = std::atoi(env_num_qps);
    }
    // gemm_rs proxy-thread count: default 1 (bumped to num_rails=2 inside session
    // setup). Multi-proxy spreads send-cmd dispatch across host threads —
    // matters when the per-proxy dispatch rate approaches NIC latency floor
    // (~2 µs). Overridable with OSGC_PROXY_THREADS (gemm_ar uses the same env).
    cfg.num_proxy_threads = 1;
    if (const char* env_proxy = std::getenv("OSGC_PROXY_THREADS")) {
        cfg.num_proxy_threads = std::atoi(env_proxy);
    } else if (const char* env_proxy = std::getenv("GEMM_RS_PROXY_THREADS")) {
        cfg.num_proxy_threads = std::atoi(env_proxy);
    }
    cfg.logical_queues_per_qp = 1;
    if (const char* env_lq = std::getenv("OSGC_INTERNODE_LOGICAL_QUEUES_PER_QP")) {
        cfg.logical_queues_per_qp = std::atoi(env_lq);
    } else if (const char* env_lq = std::getenv("GEMM_RS_LOGICAL_QUEUES_PER_QP")) {
        cfg.logical_queues_per_qp = std::atoi(env_lq);
    }
    g_session = internode::create_session(cfg);
}

void destroy_session_py() {
    if (g_session) { internode::destroy_session(g_session); g_session = nullptr; }
}
void set_epoch_py(int epoch) {
    if (g_session) internode::set_epoch(g_session, (uint32_t)epoch);
}
std::tuple<int64_t, int64_t, int64_t, int64_t, int> get_fifo_handles_py() {
    auto h = internode::get_fifo_device_handle(g_session);
    // GPU-initiated backends (EFAGDA/IBGDA) always carry a bundle; there's
    // no host-pinned FIFO tuple to flatten. Force the bundle-pointer path.
    if (h.num_fifos > 1) {
        return {(int64_t)(&g_session->fifo_bundle), 0, 0, 0, -h.num_fifos};
    }
    auto fd = h.fifos[0];
    return std::make_tuple(
        (int64_t)fd.triggers, (int64_t)fd.head, (int64_t)fd.tail,
        (int64_t)fd.tail_cache, fd.capacity);
}
int64_t get_arrival_flags_ptr_py() { return (int64_t)internode::get_arrival_device_ptr(g_session); }
int64_t get_recv_buf_ptr_py() { return (int64_t)internode::get_recv_buf_ptr(g_session); }

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("create_session", &create_session_py,
          pybind11::arg("rank"),
          pybind11::arg("peer_ip"),
          pybind11::arg("tcp_port"),
          pybind11::arg("send_buf_ptr"),
          pybind11::arg("send_buf_size"),
          pybind11::arg("recv_buf_size"),
          pybind11::arg("num_tiles"),
          pybind11::arg("fifo_capacity"),
          pybind11::arg("device_id"),
          pybind11::arg("output_buf_ptr") = 0,
          pybind11::arg("output_buf_size") = 0,
          pybind11::arg("output_n") = 0);
    m.def("destroy_session", &destroy_session_py);
    m.def("set_epoch", &set_epoch_py);
    m.def("get_fifo_handles", &get_fifo_handles_py);
    m.def("get_arrival_flags_ptr", &get_arrival_flags_ptr_py);
    m.def("get_recv_buf_ptr", &get_recv_buf_ptr_py);
    m.def("gemm_rs_fused", &gemm_rs_multinode::entrypoint_fused,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("workspace"),
          pybind11::arg("output"),
          pybind11::arg("barrier"),
          pybind11::arg("ready"),
          pybind11::arg("recv_buf_ptr"),
          pybind11::arg("staging_buf_ptr"),
          pybind11::arg("fifo_triggers"),
          pybind11::arg("fifo_head"),
          pybind11::arg("fifo_tail"),
          pybind11::arg("fifo_tail_cache"),
          pybind11::arg("fifo_capacity"),
          pybind11::arg("arrival_flags_ptr"),
          pybind11::arg("epoch"),
          pybind11::arg("node_idx"),
          pybind11::arg("num_comp_sms"),
          pybind11::arg("num_intra_comm"),
          pybind11::arg("num_send_sms"),
          pybind11::arg("num_reduce_sms"),
          pybind11::arg("use_acquire_poll") = (int64_t)0,
          pybind11::arg("reduce_poll_sleep_ns") = (int64_t)100,
          pybind11::arg("ready_chunk"),
          pybind11::arg("staging") = pybind11::none());
}
