#pragma once

// Session management + pybind module for ag_gemm_multinode.
// Included from src/ag_gemm.cu after the kernel namespace closes.
// ============================================================================
// Session management
// ============================================================================
#include "comm/internode/session_select.h"

static internode::Session* g_session = nullptr;

// Stable storage for the multi-peer string/int arrays that SessionConfig
// references (it stores raw pointers, not values). Lifetime extends until
// the next create_session_py call replaces the session.
static std::vector<std::string> g_peer_ips_storage;
static std::vector<const char*> g_peer_ips_cstr;
static std::vector<int>         g_peer_ports_storage;

void create_session_py(int rank, const std::string& peer_ip, int tcp_port,
                       int64_t send_buf_ptr, int64_t send_buf_size,
                       int64_t recv_buf_size, int num_tiles,
                       int fifo_capacity, int device_id,
                       int64_t clocal_buf_ptr = 0,
                       int64_t clocal_buf_size = 0,
                       std::vector<std::string> peer_ips = {},
                       std::vector<int> peer_tcp_ports = {}) {
    if (g_session) {
        internode::destroy_session(g_session);
        g_session = nullptr;
    }
    internode::SessionConfig cfg{};
    cfg.rank = rank;
    cfg.peer_ip = peer_ip.c_str();
    cfg.tcp_port = tcp_port;
    // Multi-peer path: when peer_ips is non-empty, populate the SessionConfig
    // multi-peer fields. The legacy peer_ip / tcp_port fields are still set
    // above for code paths that haven't migrated; create_session() prefers
    // the multi-peer fields when num_peers > 0.
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
    cfg.local_gpu_buf = reinterpret_cast<void*>(send_buf_ptr);
    cfg.local_gpu_buf_size = (size_t)send_buf_size;
    cfg.recv_buf_size = (size_t)recv_buf_size;
    cfg.num_tiles = num_tiles;
    cfg.fifo_capacity = fifo_capacity;
    cfg.device_id = device_id;
    cfg.max_inflight = 256;
    if (clocal_buf_ptr == 0 || clocal_buf_size == 0) {
        fprintf(stderr, "create_session_py: AG1 direct sends require clocal_buf_ptr/size\n");
        std::exit(EXIT_FAILURE);
    }
    cfg.clocal_gpu_buf = reinterpret_cast<void*>(clocal_buf_ptr);
    cfg.clocal_gpu_buf_size = (size_t)clocal_buf_size;
    cfg.direct_dmabuf_enabled = true;
    cfg.row_stride_bytes = 0;
    // Gen-2 Family B (run_11..13): bumped default 4 → 16 QPs per NIC. Under
    // EARLY_SEND all per-row WRs post at t=0 as a burst; more QPs let the
    // proxy drain the burst in parallel. QP=16 lifted 4K 0.77× → 0.79× and
    // 8K 1.07× → 1.11× (geomean 1.028 → 1.058). QP=32 showed no further
    // gain, so 16 is the knee. Still respects OSGC_EFA_NUM_QPS env override.
    cfg.num_qps = 16;
    if (const char* env_num_qps = std::getenv("OSGC_EFA_NUM_QPS")) {
        cfg.num_qps = std::atoi(env_num_qps);
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

int64_t get_arrival_flags_ptr_py() {
    return (int64_t)internode::get_arrival_device_ptr(g_session);
}

int64_t get_recv_buf_ptr_py() {
    return (int64_t)internode::get_recv_buf_ptr(g_session);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("create_session", &create_session_py,
          pybind11::arg("rank"), pybind11::arg("peer_ip"), pybind11::arg("tcp_port"),
          pybind11::arg("send_buf_ptr"), pybind11::arg("send_buf_size"),
          pybind11::arg("recv_buf_size"), pybind11::arg("num_tiles"),
          pybind11::arg("fifo_capacity"), pybind11::arg("device_id"),
          pybind11::arg("clocal_buf_ptr") = 0,
          pybind11::arg("clocal_buf_size") = 0,
          pybind11::arg("peer_ips") = std::vector<std::string>{},
          pybind11::arg("peer_tcp_ports") = std::vector<int>{});
    m.def("destroy_session", &destroy_session_py);
    m.def("set_epoch", &set_epoch_py);
    m.def("get_fifo_handles", &get_fifo_handles_py);
    m.def("get_arrival_flags_ptr", &get_arrival_flags_ptr_py);
    m.def("get_recv_buf_ptr", &get_recv_buf_ptr_py);
    m.def("ag_gemm_multinode", &ag_gemm_multinode::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("barrier"),
          pybind11::arg("recv_buf_ptr"),
          pybind11::arg("fifo_triggers"),
          pybind11::arg("fifo_head"),
          pybind11::arg("fifo_tail"),
          pybind11::arg("fifo_tail_cache"),
          pybind11::arg("fifo_capacity"),
          pybind11::arg("arrival_flags_ptr"),
          pybind11::arg("epoch"),
          pybind11::arg("node_idx"),
          pybind11::arg("num_comm_sms"),
          pybind11::arg("a_half_bytes"),
          pybind11::arg("A_recv"),
          pybind11::arg("active_sms") = 132,
          pybind11::arg("num_intra_comm_override") = 0,
          pybind11::arg("num_nodes") = 2);
}
