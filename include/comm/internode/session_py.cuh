#pragma once

#include "session_select.h"
#include "../device_clock.cuh"
#include "../../common/cuda_checks.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <tuple>
#include <torch/csrc/utils/pybind.h>

namespace internode::py {

using FifoHandleTuple = std::tuple<int64_t, int64_t, int64_t, int64_t, int>;

inline SessionConfig make_base_config(
    int rank,
    const char* peer_ip,
    int tcp_port,
    int64_t send_buf_ptr,
    int64_t send_buf_size,
    int64_t recv_buf_size,
    int num_tiles,
    int fifo_capacity,
    int device_id
) {
    SessionConfig cfg{};
    cfg.rank = rank;
    cfg.peer_ip = peer_ip;
    cfg.tcp_port = tcp_port;
    cfg.local_gpu_buf = reinterpret_cast<void*>(send_buf_ptr);
    cfg.local_gpu_buf_size = static_cast<size_t>(send_buf_size);
    cfg.recv_buf_size = static_cast<size_t>(recv_buf_size);
    cfg.num_tiles = num_tiles;
    cfg.fifo_capacity = fifo_capacity;
    cfg.device_id = device_id;
    return cfg;
}

inline void destroy_session(Session*& session) {
    if (session) {
        internode::destroy_session(session);
        session = nullptr;
    }
}

inline void set_epoch(Session* session, int epoch) {
    if (session) internode::set_epoch(session, static_cast<uint32_t>(epoch));
}

inline FifoHandleTuple get_fifo_handles(Session* session) {
    auto h = internode::get_fifo_device_handle(session);
    // Multi-FIFO / GPU-initiated backends pass a bundle pointer in slot 0 and
    // encode the bundle width as a negative capacity sentinel.
    if (h.num_fifos > 1) {
        return {
            reinterpret_cast<int64_t>(&session->fifo_bundle),
            0,
            0,
            0,
            -h.num_fifos
        };
    }
    auto fd = h.fifos[0];
    return {
        reinterpret_cast<int64_t>(fd.triggers),
        reinterpret_cast<int64_t>(fd.head),
        reinterpret_cast<int64_t>(fd.tail),
        reinterpret_cast<int64_t>(fd.tail_cache),
        fd.capacity
    };
}

inline int64_t get_arrival_flags_ptr(Session* session) {
    return reinterpret_cast<int64_t>(internode::get_arrival_device_ptr(session));
}

inline int64_t get_recv_buf_ptr(Session* session) {
    return reinterpret_cast<int64_t>(internode::get_recv_buf_ptr(session));
}

inline uint64_t host_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

static __global__ void read_globaltimer_kernel(unsigned long long* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = comm::globaltimer();
    }
}

inline pybind11::dict get_globaltimer_calibration(int samples = 64) {
    if (samples <= 0) samples = 1;
    unsigned long long* dev_out = nullptr;
    unsigned long long host_out = 0;
    cudaStream_t stream = nullptr;
    MKERNEL_CUDACHECK(cudaMalloc(&dev_out, sizeof(unsigned long long)));
    MKERNEL_CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    uint64_t best_host_before_ns = 0;
    uint64_t best_host_after_ns = 0;
    uint64_t best_host_mid_ns = 0;
    uint64_t best_gpu_ns = 0;
    uint64_t best_window_ns = std::numeric_limits<uint64_t>::max();

    for (int i = 0; i < samples; ++i) {
        const uint64_t host_before_ns = host_now_ns();
        read_globaltimer_kernel<<<1, 1, 0, stream>>>(dev_out);
        MKERNEL_CUDACHECK(cudaGetLastError());
        MKERNEL_CUDACHECK(cudaMemcpyAsync(
            &host_out, dev_out, sizeof(unsigned long long),
            cudaMemcpyDeviceToHost, stream));
        MKERNEL_CUDACHECK(cudaStreamSynchronize(stream));
        const uint64_t host_after_ns = host_now_ns();
        const uint64_t window_ns = host_after_ns - host_before_ns;
        if (window_ns < best_window_ns) {
            best_window_ns = window_ns;
            best_host_before_ns = host_before_ns;
            best_host_after_ns = host_after_ns;
            best_host_mid_ns = host_before_ns + window_ns / 2ULL;
            best_gpu_ns = static_cast<uint64_t>(host_out);
        }
    }

    MKERNEL_CUDACHECK(cudaStreamDestroy(stream));
    MKERNEL_CUDACHECK(cudaFree(dev_out));

    pybind11::dict d;
    d["host_before_ns"] = best_host_before_ns;
    d["host_after_ns"] = best_host_after_ns;
    d["host_mid_ns"] = best_host_mid_ns;
    d["gpu_ns"] = best_gpu_ns;
    d["window_ns"] = best_window_ns;
    d["offset_ns"] = static_cast<int64_t>(best_host_mid_ns) - static_cast<int64_t>(best_gpu_ns);
    return d;
}

} // namespace internode::py
