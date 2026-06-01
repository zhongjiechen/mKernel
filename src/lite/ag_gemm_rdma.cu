#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <mma.h>

#include "comm/global_u64.cuh"
#include "comm/internode/arrival.cuh"
#include "comm/internode/d2h_fifo.cuh"
#include "comm/internode/session_py.cuh"
#include "comm/internode/session_select.h"
#ifndef CUDACHECK
#define CUDACHECK(x) MKERNEL_CUDACHECK(x)
#endif
#include "dist/parallel_buffer.cuh"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>

namespace {

internode::Session* g_session = nullptr;
std::string g_peer_ip_storage;
std::string g_nic_name_storage;

constexpr int kTileM = 32;
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kWmmaK = 8;
constexpr int kWarpsN = 4;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kASmemLd = kTileK + 4;
constexpr int kBSmemLd = kTileN + 4;
constexpr int kMaxLocalWorld = 16;

struct CopyStreamPool {
    cudaStream_t streams[kMaxLocalWorld]{};
    cudaStream_t rdma_stream{};
    cudaEvent_t start_event{};
    cudaEvent_t rdma_done_event{};
    cudaEvent_t done_events[kMaxLocalWorld]{};
    bool initialized = false;
};

CopyStreamPool& copy_stream_pool() {
    static thread_local CopyStreamPool pool;
    if (!pool.initialized) {
        for (int i = 0; i < kMaxLocalWorld; ++i) {
            MKERNEL_CUDACHECK(cudaStreamCreateWithFlags(&pool.streams[i], cudaStreamNonBlocking));
            MKERNEL_CUDACHECK(cudaEventCreateWithFlags(&pool.done_events[i], cudaEventDisableTiming));
        }
        MKERNEL_CUDACHECK(cudaStreamCreateWithFlags(&pool.rdma_stream, cudaStreamNonBlocking));
        MKERNEL_CUDACHECK(cudaEventCreateWithFlags(&pool.start_event, cudaEventDisableTiming));
        MKERNEL_CUDACHECK(cudaEventCreateWithFlags(&pool.rdma_done_event, cudaEventDisableTiming));
        pool.initialized = true;
    }
    return pool;
}

__global__ void post_ag_chunks_kernel(
    internode::D2HFifoDeviceBundle fifo_bundle,
    int peer_rank,
    int num_chunks,
    int local_rows,
    int k,
    int chunk_rows) {
    const int chunk = blockIdx.x;
    if (chunk >= num_chunks || threadIdx.x != 0) return;

    const int row0 = chunk * chunk_rows;
    const int rows = min(chunk_rows, local_rows - row0);
    if (rows <= 0) return;

    internode::TransferCmd cmd{};
    cmd.cmd_type = internode::CmdType::WRITE;
    cmd.dst_rank = static_cast<uint8_t>(peer_rank);
    cmd.tile_id = static_cast<uint16_t>(chunk);
    cmd.bytes = static_cast<uint32_t>(static_cast<uint64_t>(rows) * k * sizeof(float));
    cmd.local_offset = static_cast<uint32_t>(static_cast<uint64_t>(row0) * k * sizeof(float));
    cmd.remote_offset = cmd.local_offset;
    cmd.lane_id = static_cast<uint16_t>(chunk);
    cmd.src_view = 0;
    cmd.enqueue_device_ns = comm::globaltimer();

    internode::D2HFifoDevice fifo =
        internode::q2_select_fifo_for_lane(fifo_bundle, static_cast<uint32_t>(chunk));
    fifo.push(cmd);
}

__global__ void set_index_flag_kernel(uint32_t* flags, int index, uint32_t epoch) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __threadfence_system();
        flags[index] = epoch;
    }
}

__global__ void wait_all_flags_kernel(volatile uint32_t* flags, int count, uint32_t epoch) {
    const int idx = threadIdx.x;
    if (idx >= count) return;
    while (flags[idx] < epoch) {
        __nanosleep(100);
    }
}

__global__ void wait_rdma_chunks_kernel(
    volatile uint32_t* __restrict__ rdma_flags,
    int num_chunks,
    uint32_t epoch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        internode::wait_arrival(rdma_flags, chunk, epoch);
    }
}

__global__ void full_tf32_gemm_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c_out,
    int m,
    int n,
    int k) {
    const int row0 = blockIdx.y * kTileM;
    const int col0 = blockIdx.x * kTileN;
    if (row0 >= m || col0 >= n) return;

    extern __shared__ float smem[];
    float* smem_a = smem;
    float* smem_b = smem_a + kTileM * kASmemLd;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int warp_m = warp_id / kWarpsN;
    const int warp_n = warp_id - warp_m * kWarpsN;
    const int warp_row = warp_m * 16;
    const int warp_col = warp_n * 16;

    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, kWmmaK, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, kWmmaK, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, kWmmaK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < k; k0 += kTileK) {
        for (int idx = tid; idx < kTileM * kTileK; idx += blockDim.x) {
            const int r = idx / kTileK;
            const int kk = idx - r * kTileK;
            const int g_row = row0 + r;
            const int g_k = k0 + kk;
            smem_a[r * kASmemLd + kk] =
                (g_row < m && g_k < k) ? a[static_cast<int64_t>(g_row) * k + g_k] : 0.0f;
        }
        for (int idx = tid; idx < kTileK * kTileN; idx += blockDim.x) {
            const int kk = idx / kTileN;
            const int col = idx - kk * kTileN;
            const int g_k = k0 + kk;
            const int g_col = col0 + col;
            smem_b[kk * kBSmemLd + col] =
                (g_k < k && g_col < n) ? b[static_cast<int64_t>(g_k) * n + g_col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kTileK; kk += kWmmaK) {
            wmma::load_matrix_sync(a_frag, smem_a + warp_row * kASmemLd + kk, kASmemLd);
            wmma::load_matrix_sync(b_frag, smem_b + kk * kBSmemLd + warp_col, kBSmemLd);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    const int out_row = row0 + warp_row;
    const int out_col = col0 + warp_col;
    if (out_row + 16 <= m && out_col + 16 <= n) {
        wmma::store_matrix_sync(c_out + static_cast<int64_t>(out_row) * n + out_col,
                                c_frag,
                                n,
                                wmma::mem_row_major);
    }
}

void check_session() {
    TORCH_CHECK(g_session != nullptr, "RDMA session has not been created");
}

int64_t tensor_nbytes(const torch::Tensor& t) {
    return static_cast<int64_t>(t.numel()) * static_cast<int64_t>(t.element_size());
}

}  // namespace

void create_session(
    int node_rank,
    const std::string& peer_ip,
    int tcp_port,
    torch::Tensor send_buf,
    torch::Tensor recv_buf,
    int num_tiles,
    int fifo_capacity,
    int device_id,
    int num_qps,
    int num_proxy_threads,
    int max_inflight,
    const std::string& nic_name) {
    TORCH_CHECK(send_buf.is_cuda() && send_buf.is_contiguous(), "send_buf must be contiguous CUDA");
    TORCH_CHECK(send_buf.scalar_type() == torch::kFloat32, "basic TP path expects float32 send_buf");
    TORCH_CHECK(recv_buf.is_cuda() && recv_buf.is_contiguous(), "recv_buf must be contiguous CUDA");
    TORCH_CHECK(recv_buf.scalar_type() == torch::kFloat32, "basic TP path expects float32 recv_buf");
    internode::py::destroy_session(g_session);

    g_peer_ip_storage = peer_ip;
    g_nic_name_storage = nic_name;

    internode::SessionConfig cfg{};
    cfg.rank = node_rank;
    cfg.peer_ip = g_peer_ip_storage.c_str();
    cfg.tcp_port = tcp_port;
    cfg.local_gpu_buf = reinterpret_cast<void*>(send_buf.data_ptr());
    cfg.local_gpu_buf_size = static_cast<size_t>(tensor_nbytes(send_buf));
    cfg.recv_buf_size = static_cast<size_t>(tensor_nbytes(recv_buf));
    cfg.external_recv_buf = reinterpret_cast<void*>(recv_buf.data_ptr());
    cfg.num_tiles = num_tiles;
    cfg.fifo_capacity = fifo_capacity;
    cfg.device_id = device_id;
    cfg.max_inflight = max_inflight > 0 ? max_inflight : 512;
    cfg.num_qps = num_qps > 0 ? num_qps : 1;
    cfg.num_proxy_threads = num_proxy_threads > 0 ? num_proxy_threads : 1;
    cfg.nic_name = g_nic_name_storage.empty() ? nullptr : g_nic_name_storage.c_str();
    cfg.pin_proxy = true;

    g_session = internode::create_session(cfg);
    TORCH_CHECK(g_session != nullptr, "internode::create_session failed");
}

void destroy_session() {
    internode::py::destroy_session(g_session);
}

void set_epoch(int epoch) {
    check_session();
    internode::set_epoch(g_session, static_cast<uint32_t>(epoch));
}

void fast_set_epoch(int epoch) {
    check_session();
    g_session->epoch = static_cast<uint32_t>(epoch);
    for (int t = 0; t < g_session->num_proxy_threads; ++t) {
        if (g_session->proxies[t] != nullptr) {
            g_session->proxies[t]->set_epoch(static_cast<uint32_t>(epoch));
        }
    }
}

void reset_arrival_flags() {
    check_session();
    const size_t words =
        static_cast<size_t>(g_session->arrival.count + g_session->arrival.tail_count);
    MKERNEL_CUDACHECK(cudaMemsetAsync(
        g_session->arrival.device_ptr,
        0,
        words * sizeof(uint32_t),
        at::cuda::getCurrentCUDAStream()));
}

void local_barrier(dist::ParallelBuffer& buffer) {
    dist::ParallelBuffer::brokers_.at({buffer.local_rank_, buffer.local_world_size_}).sync();
}

void post_ag_chunks(
    torch::Tensor a_local,
    int node_rank,
    int local_rows,
    int k,
    int chunk_rows) {
    check_session();
    TORCH_CHECK(a_local.is_cuda() && a_local.is_contiguous(), "a_local must be contiguous CUDA");
    TORCH_CHECK(a_local.scalar_type() == torch::kFloat32, "basic TP path expects float32 A");
    TORCH_CHECK(local_rows > 0 && k > 0 && chunk_rows > 0, "invalid shape/chunk_rows");
    const c10::cuda::CUDAGuard device_guard(a_local.device());
    const int num_chunks = (local_rows + chunk_rows - 1) / chunk_rows;
    const int peer_rank = 1 - node_rank;
    internode::D2HFifoDeviceBundle fifo = internode::get_fifo_device_handle(g_session);
    post_ag_chunks_kernel<<<num_chunks, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        fifo,
        peer_rank,
        num_chunks,
        local_rows,
        k,
        chunk_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wait_ipc_flags(dist::ParallelBuffer& ready_flags, int count, int epoch) {
    TORCH_CHECK(ready_flags.data_.is_cuda() && ready_flags.data_.is_contiguous(),
                "ready_flags must be contiguous CUDA");
    TORCH_CHECK(ready_flags.data_.scalar_type() == torch::kInt32, "ready_flags must be int32");
    TORCH_CHECK(count > 0 && count <= ready_flags.data_.numel(), "invalid flag count");
    const c10::cuda::CUDAGuard device_guard(ready_flags.data_.device());
    wait_all_flags_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<volatile uint32_t*>(ready_flags.data_.data_ptr<int32_t>()),
        count,
        static_cast<uint32_t>(epoch));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wait_rdma_chunks_push_pair_a_to_full_signal(
    torch::Tensor a_local,
    torch::Tensor a_remote,
    dist::ParallelBuffer& a_full_buffer,
    dist::ParallelBuffer& ready_flags,
    int local_rank,
    int local_world_size,
    int node_rank,
    int64_t rows,
    int64_t k,
    int num_chunks,
    int epoch) {
    check_session();
    TORCH_CHECK(local_world_size > 0 && local_world_size <= kMaxLocalWorld,
                "local_world_size out of supported range");
    TORCH_CHECK(a_local.is_cuda() && a_local.is_contiguous(), "a_local must be contiguous CUDA");
    TORCH_CHECK(a_remote.is_cuda() && a_remote.is_contiguous(), "a_remote must be contiguous CUDA");
    TORCH_CHECK(a_full_buffer.data_.is_cuda() && a_full_buffer.data_.is_contiguous(),
                "a_full_buffer must be contiguous CUDA");
    TORCH_CHECK(ready_flags.data_.is_cuda() && ready_flags.data_.is_contiguous(),
                "ready_flags must be contiguous CUDA");
    TORCH_CHECK(a_local.scalar_type() == torch::kFloat32 &&
                a_remote.scalar_type() == torch::kFloat32 &&
                a_full_buffer.data_.scalar_type() == torch::kFloat32,
                "basic TP path expects float32 A tensors");
    TORCH_CHECK(ready_flags.data_.scalar_type() == torch::kInt32, "ready_flags must be int32");
    TORCH_CHECK(local_rank == a_full_buffer.local_rank_, "local_rank mismatch");
    TORCH_CHECK(local_world_size == a_full_buffer.local_world_size_, "local_world_size mismatch");
    TORCH_CHECK(node_rank == 0 || node_rank == 1, "node_rank must be 0 or 1");
    TORCH_CHECK(rows > 0 && k > 0 && num_chunks > 0, "invalid rows/k/num_chunks");
    TORCH_CHECK(a_full_buffer.data_.dim() == 2 &&
                a_full_buffer.data_.size(0) == 2 * local_world_size * rows &&
                a_full_buffer.data_.size(1) == k, "a_full_buffer shape mismatch");
    TORCH_CHECK(ready_flags.data_.numel() >= 2 * local_world_size,
                "ready_flags must have at least 2 * local_world_size entries");

    const c10::cuda::CUDAGuard device_guard(a_full_buffer.data_.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(k) * sizeof(float);
    const int src_device = a_local.device().index();
    const char* local_src = reinterpret_cast<const char*>(a_local.data_ptr());
    const char* remote_src = reinterpret_cast<const char*>(a_remote.data_ptr());
    const int64_t elems = rows * k;
    const int64_t local_out =
        (static_cast<int64_t>(node_rank) * local_world_size + local_rank) * elems;
    const int64_t remote_out =
        (static_cast<int64_t>(1 - node_rank) * local_world_size + local_rank) * elems;

    CopyStreamPool& pool = copy_stream_pool();
    MKERNEL_CUDACHECK(cudaEventRecord(pool.start_event, stream));
    for (int peer = 0; peer < local_world_size; ++peer) {
        char* dst = reinterpret_cast<char*>(a_full_buffer.raw_ptrs_.at(static_cast<size_t>(peer)));
        cudaStream_t copy_stream = pool.streams[peer];
        MKERNEL_CUDACHECK(cudaStreamWaitEvent(copy_stream, pool.start_event, 0));
        MKERNEL_CUDACHECK(cudaMemcpyPeerAsync(
            dst + static_cast<size_t>(local_out) * sizeof(float),
            peer,
            local_src,
            src_device,
            bytes,
            copy_stream));
    }

    auto* rdma_flags = reinterpret_cast<volatile uint32_t*>(
        internode::get_arrival_device_ptr(g_session));
    wait_rdma_chunks_kernel<<<1, 1, 0, stream>>>(
        rdma_flags,
        num_chunks,
        static_cast<uint32_t>(epoch));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    MKERNEL_CUDACHECK(cudaEventRecord(pool.rdma_done_event, stream));

    for (int peer = 0; peer < local_world_size; ++peer) {
        char* dst = reinterpret_cast<char*>(a_full_buffer.raw_ptrs_.at(static_cast<size_t>(peer)));
        cudaStream_t copy_stream = pool.streams[peer];
        MKERNEL_CUDACHECK(cudaStreamWaitEvent(copy_stream, pool.rdma_done_event, 0));
        MKERNEL_CUDACHECK(cudaMemcpyPeerAsync(
            dst + static_cast<size_t>(remote_out) * sizeof(float),
            peer,
            remote_src,
            src_device,
            bytes,
            copy_stream));
        MKERNEL_CUDACHECK(cudaEventRecord(pool.done_events[peer], copy_stream));
    }
    for (int peer = 0; peer < local_world_size; ++peer) {
        MKERNEL_CUDACHECK(cudaStreamWaitEvent(stream, pool.done_events[peer], 0));
    }

    auto* local_ready = reinterpret_cast<uint32_t*>(ready_flags.data_.data_ptr<int32_t>());
    set_index_flag_kernel<<<1, 1, 0, stream>>>(
        local_ready,
        local_rank,
        static_cast<uint32_t>(epoch));
    set_index_flag_kernel<<<1, 1, 0, stream>>>(
        local_ready,
        local_world_size + local_rank,
        static_cast<uint32_t>(epoch));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int peer = 0; peer < local_world_size; ++peer) {
        if (peer == local_rank) continue;
        auto* peer_flags = reinterpret_cast<uint32_t*>(
            ready_flags.raw_ptrs_.at(static_cast<size_t>(peer)));
        MKERNEL_CUDACHECK(cudaMemcpyPeerAsync(
            peer_flags + local_rank,
            peer,
            local_ready + local_rank,
            src_device,
            sizeof(uint32_t),
            stream));
        MKERNEL_CUDACHECK(cudaMemcpyPeerAsync(
            peer_flags + local_world_size + local_rank,
            peer,
            local_ready + local_world_size + local_rank,
            src_device,
            sizeof(uint32_t),
            stream));
    }
}

void full_tf32_gemm(torch::Tensor a_full, torch::Tensor b, torch::Tensor c_out) {
    TORCH_CHECK(a_full.is_cuda() && b.is_cuda() && c_out.is_cuda(),
                "a_full/b/c_out must be CUDA");
    TORCH_CHECK(a_full.scalar_type() == torch::kFloat32 &&
                b.scalar_type() == torch::kFloat32 &&
                c_out.scalar_type() == torch::kFloat32,
                "full_tf32_gemm expects float32 tensors");
    TORCH_CHECK(a_full.is_contiguous() && b.is_contiguous() && c_out.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(a_full.dim() == 2 && b.dim() == 2 && c_out.dim() == 2,
                "all tensors must be 2D");
    TORCH_CHECK(a_full.size(1) == b.size(0), "K mismatch");
    TORCH_CHECK(c_out.size(0) == a_full.size(0) && c_out.size(1) == b.size(1),
                "C shape mismatch");
    TORCH_CHECK((a_full.size(0) % kTileM) == 0 && (b.size(1) % 16) == 0 &&
                (a_full.size(1) % kWmmaK) == 0,
                "full_tf32_gemm requires M divisible by 32, N divisible by 16, K divisible by 8");
    TORCH_CHECK(a_full.size(0) <= INT_MAX && a_full.size(1) <= INT_MAX && b.size(1) <= INT_MAX,
                "shape too large");

    const c10::cuda::CUDAGuard device_guard(b.device());
    const int m = static_cast<int>(a_full.size(0));
    const int n = static_cast<int>(b.size(1));
    const int k = static_cast<int>(a_full.size(1));
    const size_t smem_bytes =
        (static_cast<size_t>(kTileM) * kASmemLd +
         static_cast<size_t>(kTileK) * kBSmemLd) * sizeof(float);
    dim3 block(kThreads);
    dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
    full_tf32_gemm_kernel<<<grid, block, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        a_full.data_ptr<float>(),
        b.data_ptr<float>(),
        c_out.data_ptr<float>(),
        m,
        n,
        k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void push_full_tf32_gemm(
    torch::Tensor a_local,
    torch::Tensor a_remote,
    dist::ParallelBuffer& a_full_buffer,
    dist::ParallelBuffer& ready_flags,
    torch::Tensor b,
    torch::Tensor c_out,
    int node_rank,
    int local_rank,
    int local_world_size,
    int64_t local_rows,
    int64_t k,
    int chunk_rows,
    int num_chunks,
    int epoch) {
    post_ag_chunks(a_local, node_rank, static_cast<int>(local_rows), static_cast<int>(k), chunk_rows);
    wait_rdma_chunks_push_pair_a_to_full_signal(
        a_local,
        a_remote,
        a_full_buffer,
        ready_flags,
        local_rank,
        local_world_size,
        node_rank,
        local_rows,
        k,
        num_chunks,
        epoch);
    wait_ipc_flags(ready_flags, 2 * local_world_size, epoch);
    full_tf32_gemm(a_full_buffer.data_, b, c_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("create_session", &create_session,
          pybind11::arg("node_rank"),
          pybind11::arg("peer_ip"),
          pybind11::arg("tcp_port"),
          pybind11::arg("send_buf"),
          pybind11::arg("recv_buf"),
          pybind11::arg("num_tiles"),
          pybind11::arg("fifo_capacity"),
          pybind11::arg("device_id"),
          pybind11::arg("num_qps") = 1,
          pybind11::arg("num_proxy_threads") = 1,
          pybind11::arg("max_inflight") = 512,
          pybind11::arg("nic_name") = std::string{});
    m.def("destroy_session", &destroy_session);
    m.def("set_epoch", &set_epoch);
    m.def("fast_set_epoch", &fast_set_epoch);
    m.def("reset_arrival_flags", &reset_arrival_flags);
    m.def("local_barrier", &local_barrier);
    m.def("push_full_tf32_gemm", &push_full_tf32_gemm);
}
