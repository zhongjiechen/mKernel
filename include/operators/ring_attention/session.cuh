#pragma once

// Session management + pybind module for ring_attn_multinode.
// Included from src/ring_attention.cu after the kernel namespace closes.
// ============================================================================
// Session management + pybind module
// ============================================================================
#include "comm/internode/session_select.h"
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>

static internode::Session* g_session = nullptr;
static std::vector<std::string> g_peer_ips_storage;
static std::vector<const char*> g_peer_ips_cstr;
static std::vector<int>         g_peer_ports_storage;

void create_session_py(int rank, const std::string& peer_ip, int tcp_port,
                       int64_t send_buf_ptr, int64_t send_buf_size,
                       int64_t recv_buf_size, int num_tiles,
                       int fifo_capacity, int device_id,
                       int64_t k0_buf_ptr,
                       int64_t k0_buf_size,
                       int64_t v0_buf_ptr,
                       int64_t v0_buf_size,
                       std::vector<std::string> peer_ips = {},
                       std::vector<int> peer_tcp_ports = {}) {
    if (g_session) { internode::destroy_session(g_session); g_session = nullptr; }
    internode::SessionConfig cfg{};
    cfg.rank = rank;
    cfg.peer_ip = peer_ip.c_str();
    cfg.tcp_port = tcp_port;
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
    // Zero-copy send: K0 (src_view=0) and V0 (src_view=1) registered as
    // DMA-BUF MRs. Proxy posts single-SGE WRs straight from the VMM tensors
    // — no pack copy. send_buf_ptr/size are kept in the signature for
    // backwards compatibility but unused.
    if (k0_buf_ptr == 0 || v0_buf_ptr == 0) {
        fprintf(stderr, "create_session_py: ring_attention zero-copy send requires k0/v0 ptrs\n");
        std::exit(EXIT_FAILURE);
    }
    (void)send_buf_ptr; (void)send_buf_size;
    cfg.local_gpu_buf = reinterpret_cast<void*>(k0_buf_ptr);
    cfg.local_gpu_buf_size = (size_t)k0_buf_size;
    cfg.clocal_gpu_buf = reinterpret_cast<void*>(v0_buf_ptr);
    cfg.clocal_gpu_buf_size = (size_t)v0_buf_size;
    cfg.direct_dmabuf_enabled = true;
    cfg.recv_buf_size = (size_t)recv_buf_size;
    cfg.num_tiles = num_tiles;
    cfg.fifo_capacity = fifo_capacity;
    cfg.device_id = device_id;
    cfg.max_inflight = 2048;
    // Round 25: default to 4 QPs per endpoint to match the gemm_ar round-13 win
    // (51e19e8). Multi-NIC striping is also enabled by default via the
    // round-18 cluster default OSGC_EFA_NUM_NICS=2 in session_fi.h.
    cfg.num_qps = 4;
    if (const char* env_num_qps = std::getenv("OSGC_EFA_NUM_QPS")) {
        cfg.num_qps = std::atoi(env_num_qps);
    }
    g_session = internode::create_session(cfg);
}
void destroy_session_py() { if (g_session) { internode::destroy_session(g_session); g_session = nullptr; } }
void set_epoch_py(int epoch) { if (g_session) internode::set_epoch(g_session, (uint32_t)epoch); }
std::tuple<int64_t, int64_t, int64_t, int64_t, int> get_fifo_handles_py() {
    auto h = internode::get_fifo_device_handle(g_session);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("create_session", &create_session_py,
          pybind11::arg("rank"), pybind11::arg("peer_ip"), pybind11::arg("tcp_port"),
          pybind11::arg("send_buf_ptr"), pybind11::arg("send_buf_size"),
          pybind11::arg("recv_buf_size"), pybind11::arg("num_tiles"),
          pybind11::arg("fifo_capacity"), pybind11::arg("device_id"),
          pybind11::arg("k0_buf_ptr"), pybind11::arg("k0_buf_size"),
          pybind11::arg("v0_buf_ptr"), pybind11::arg("v0_buf_size"),
          pybind11::arg("peer_ips") = std::vector<std::string>{},
          pybind11::arg("peer_tcp_ports") = std::vector<int>{});
    m.def("destroy_session", &destroy_session_py);
    m.def("set_epoch", &set_epoch_py);
    m.def("get_fifo_handles", &get_fifo_handles_py);
    m.def("get_arrival_flags_ptr", &get_arrival_flags_ptr_py);
    m.def("get_recv_buf_ptr", &get_recv_buf_ptr_py);
    m.def("ring_attn_multinode", &ring_attn_multinode::entrypoint,
          pybind11::arg("Q"),
          pybind11::arg("K0"),
          pybind11::arg("K1"),
          pybind11::arg("V0"),
          pybind11::arg("V1"),
          pybind11::arg("L"),
          pybind11::arg("L_block"),
          pybind11::arg("O"),
          pybind11::arg("O_block"),
          pybind11::arg("barrier"),
          pybind11::arg("send_buf_ptr"),
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
          pybind11::arg("num_send_sms"),
          pybind11::arg("num_copy_sms"),
          pybind11::arg("num_nodes") = 2);
}
