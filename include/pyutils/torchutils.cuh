/**
 * @file torchutils.cuh
 * @brief PyTorch/CUDA interop utilities: kernel launch helpers, tensor validation
 *        macros, and ParallelTensor integration for ThunderKittens operators.
 */
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Tensor.h>

#include "../common/types.cuh"
#include "../memory/tk_ops_group_group.cuh"
#include "../comm/comm.cuh"
#include "parallel_tensor.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace kittens {
namespace py {

template <typename Config>
concept has_min_blocks_per_sm = requires { std::integral_constant<int, int(Config::MIN_BLOCKS_PER_SM)>{}; };

template <typename Config>
consteval int min_blocks_per_sm() {
    if constexpr(has_min_blocks_per_sm<Config>)
        return Config::MIN_BLOCKS_PER_SM;
    else
        return 1;
}

template <typename Config, typename Globals, auto Kernel>
__global__
__launch_bounds__(Config::NUM_THREADS, min_blocks_per_sm<Config>())
void global_kernel_unclustered(const __grid_constant__ Globals G) {
    Kernel(G);
}

template <typename Config, typename Globals, auto Kernel>
__global__
__launch_bounds__(Config::NUM_THREADS, min_blocks_per_sm<Config>())
__cluster_dims__(Config::CLUSTER_SIZE)
void global_kernel_clustered(const __grid_constant__ Globals G) {
    Kernel(G);
}

template <typename Layout, bool TypeCheck = true>
static inline void tensor_check(const at::Tensor &t) {
    TORCH_CHECK(t.is_cuda(), "Tensor must be on CUDA device")
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous")
    TORCH_CHECK(t.dim() <= 4, "Expected Tensor.dim() <= 4");

    if constexpr (!TypeCheck) {
        return;
    } else if constexpr (std::is_same_v<typename Layout::dtype, char>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Char, "Tensor has invalid dtype (expected int8)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, short>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Short, "Tensor has invalid dtype (expected int16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, int>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Int, "Tensor has invalid dtype (expected int32)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, long>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Long, "Tensor has invalid dtype (expected int64)");
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e4m3>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e4m3fn, "Tensor has invalid dtype (expected fp8e4m3)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e5m2>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e5m2, "Tensor has invalid dtype (expected fp8e5m2)");
#endif
#ifdef KITTENS_BLACKWELL
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp8e8m0>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e8m0fnu || t.dtype() == at::ScalarType::Byte, "Tensor has invalid dtype (expected fp8e8m0)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::fp4e2m1_2>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Tensor has invalid dtype (expected fp4_2)");
#endif
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::bf16>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::BFloat16, "Tensor has invalid dtype (expected bfloat16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, ::kittens::half>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Half, "Tensor has invalid dtype (expected float16)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, float>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Float, "Tensor has invalid dtype (expected float32)");
    } else if constexpr (std::is_same_v<typename Layout::dtype, double>) {
        TORCH_CHECK(t.dtype() == at::ScalarType::Double, "Tensor has invalid dtype (expected float64)");
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

static inline void _device_check(const at::Tensor& first, const at::Tensor& second) {
    TORCH_CHECK(first.device() == second.device(), "All tensors must be on the same device");
}

template <typename T1, typename... Ts>
static inline void device_check(const T1& first, const Ts&... rest) {
    (_device_check(first, rest), ...);
}

static inline void _parallel_tensor_check(const TKParallelTensor& first, const TKParallelTensor& second) {
    TORCH_CHECK(first.local_rank_ == second.local_rank_, "All parallel tensors must have the same local_rank");
    TORCH_CHECK(first.local_world_size_ == second.local_world_size_, "All parallel tensors must have the same local_world_size");
}

template <typename T1, typename... Ts>
static inline void parallel_tensor_check(const T1& first, const Ts&... rest) {
    (_parallel_tensor_check(first, rest), ...);
}

template <typename Config>
concept static_grid = requires { Config::NUM_BLOCKS; };

template <typename Config>
concept static_block = requires { Config::NUM_THREADS; };

template <typename Config>
concept static_dynamic_shared_memory = requires { Config::DYNAMIC_SHARED_MEMORY; };

template <typename Config, typename Globals, auto Kernel>
static inline void launch_kernel(const Globals &G) {
    dim3 grid;
    if constexpr (static_grid<Config>)
        grid = dim3{Config::NUM_BLOCKS, 1, 1};
    else
        grid = G.grid();

    dim3 block;
    if constexpr (static_block<Config>)
        block = dim3{Config::NUM_THREADS, 1, 1};
    else
        block = G.block();

    int dynamic_shared_memory;
    if constexpr (static_dynamic_shared_memory<Config>)
        dynamic_shared_memory = static_cast<int>(Config::DYNAMIC_SHARED_MEMORY);
    else
        dynamic_shared_memory = G.dynamic_shared_memory();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if constexpr (Config::CLUSTER_SIZE <= 1) {
        CUDACHECK(cudaFuncSetAttribute(global_kernel_unclustered<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
        global_kernel_unclustered<Config, Globals, Kernel><<<grid, block, dynamic_shared_memory, stream>>>(G);
    } else {
#if defined(KITTENS_HOPPER)
        static_assert(Config::CLUSTER_SIZE <= 8, "Cluster size must be less than or equal to 8 for Hopper");
#elif defined(KITTENS_BLACKWELL)
        static_assert(Config::CLUSTER_SIZE <= 16, "Cluster size must be less than or equal to 16 for Blackwell");
        if constexpr (Config::CLUSTER_SIZE > 8)
            CUDACHECK(cudaFuncSetAttribute(global_kernel_clustered<Config, Globals, Kernel>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
#endif
        CUDACHECK(cudaFuncSetAttribute(global_kernel_clustered<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
        global_kernel_clustered<Config, Globals, Kernel><<<grid, block, dynamic_shared_memory, stream>>>(G);
    }
}

// Overload with explicit grid size override (e.g. for multi-tenant active_sms).
template <typename Config, typename Globals, auto Kernel>
static inline void launch_kernel(const Globals &G, unsigned int num_blocks) {
    dim3 grid{num_blocks, 1, 1};

    dim3 block;
    if constexpr (static_block<Config>)
        block = dim3{Config::NUM_THREADS, 1, 1};
    else
        block = G.block();

    int dynamic_shared_memory;
    if constexpr (static_dynamic_shared_memory<Config>)
        dynamic_shared_memory = static_cast<int>(Config::DYNAMIC_SHARED_MEMORY);
    else
        dynamic_shared_memory = G.dynamic_shared_memory();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if constexpr (Config::CLUSTER_SIZE <= 1) {
        CUDACHECK(cudaFuncSetAttribute(global_kernel_unclustered<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
        global_kernel_unclustered<Config, Globals, Kernel><<<grid, block, dynamic_shared_memory, stream>>>(G);
    } else {
#if defined(KITTENS_HOPPER)
        static_assert(Config::CLUSTER_SIZE <= 8, "Cluster size must be less than or equal to 8 for Hopper");
#elif defined(KITTENS_BLACKWELL)
        static_assert(Config::CLUSTER_SIZE <= 16, "Cluster size must be less than or equal to 16 for Blackwell");
        if constexpr (Config::CLUSTER_SIZE > 8)
            CUDACHECK(cudaFuncSetAttribute(global_kernel_clustered<Config, Globals, Kernel>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
#endif
        CUDACHECK(cudaFuncSetAttribute(global_kernel_clustered<Config, Globals, Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_memory));
        global_kernel_clustered<Config, Globals, Kernel><<<grid, block, dynamic_shared_memory, stream>>>(G);
    }
}

} // namespace py
} // namespace kittens
