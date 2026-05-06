#pragma once

// Session management + pybind module for ag_gemm_multinode.
// Included from src/ag_gemm.cu after the kernel namespace closes.
// ============================================================================
// Session management
// ============================================================================
#include "comm/internode/session_py.cuh"

static internode::Session* g_session = nullptr;

void create_session_py(int rank, const std::string& peer_ip, int tcp_port,
                       int64_t send_buf_ptr, int64_t send_buf_size,
                       int64_t recv_buf_size, int num_tiles,
                       int fifo_capacity, int device_id,
                       int64_t clocal_buf_ptr = 0,
                       int64_t clocal_buf_size = 0) {
    internode::py::destroy_session(g_session);
    internode::SessionConfig cfg = internode::py::make_base_config(
        rank, peer_ip.c_str(), tcp_port,
        send_buf_ptr, send_buf_size, recv_buf_size,
        num_tiles, fifo_capacity, device_id);
    cfg.max_inflight = 256;
    if (clocal_buf_ptr == 0 || clocal_buf_size == 0) {
        fprintf(stderr, "create_session_py: AG1 direct sends require clocal_buf_ptr/size\n");
        std::exit(EXIT_FAILURE);
    }
    cfg.clocal_gpu_buf = reinterpret_cast<void*>(clocal_buf_ptr);
    cfg.clocal_gpu_buf_size = (size_t)clocal_buf_size;
    cfg.direct_dmabuf_enabled = true;
    cfg.row_stride_bytes = 0;
    // Early-send posts per-row WRs in a burst, so use multiple QPs by default.
    // MKERNEL_EFA_NUM_QPS can still override this.
    cfg.num_qps = 16;
    if (const char* env_num_qps = std::getenv("MKERNEL_EFA_NUM_QPS")) {
        cfg.num_qps = std::atoi(env_num_qps);
    }
    g_session = internode::create_session(cfg);
}

void destroy_session_py() {
    internode::py::destroy_session(g_session);
}

void set_epoch_py(int epoch) {
    internode::py::set_epoch(g_session, epoch);
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int> get_fifo_handles_py() {
    return internode::py::get_fifo_handles(g_session);
}

int64_t get_arrival_flags_ptr_py() {
    return internode::py::get_arrival_flags_ptr(g_session);
}

int64_t get_recv_buf_ptr_py() {
    return internode::py::get_recv_buf_ptr(g_session);
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
          pybind11::arg("clocal_buf_size") = 0);
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
          pybind11::arg("num_intra_comm_override") = 0);
}
