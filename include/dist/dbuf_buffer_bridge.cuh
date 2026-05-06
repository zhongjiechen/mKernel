/**
 * @file
 * @brief Construct dist::local_tensor / dist::distributed_tensor descriptors from release-owned buffers.
 */
#pragma once

#include "distributed_buffer.cuh"
#include "parallel_buffer.cuh"

#include <ATen/core/Tensor.h>
#include <array>

namespace dist {

template<typename LocalTensor>
__host__ inline LocalTensor local_tensor_from_tensor(const at::Tensor& t) {
    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < (int)t.dim(); ++i) {
        shape[4 - t.dim() + i] = (int)t.size(i);
    }
    return make_local_tensor<LocalTensor>(reinterpret_cast<uint64_t>(t.data_ptr()),
                                          shape[0], shape[1], shape[2], shape[3]);
}

template<typename LocalTensor>
__host__ inline LocalTensor local_tensor_from_tensor(const at::Tensor& t, int B, int D, int R, int C) {
    return make_local_tensor<LocalTensor>(reinterpret_cast<uint64_t>(t.data_ptr()), B, D, R, C);
}

template<typename DistributedTensor>
__host__ inline DistributedTensor distributed_tensor_from_buffer(ParallelBuffer& t) {
    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < (int)t.data_.dim(); ++i) {
        shape[4 - t.data_.dim() + i] = (int)t.data_.size(i);
    }

    if constexpr (DistributedTensor::multicast) {
        return make_distributed_tensor<DistributedTensor>(
            reinterpret_cast<uint64_t>(t.multicast_ptr_),
            reinterpret_cast<uint64_t*>(t.raw_ptrs_.data()),
            shape[0], shape[1], shape[2], shape[3]);
    } else {
        return make_distributed_tensor<DistributedTensor>(
            reinterpret_cast<uint64_t*>(t.raw_ptrs_.data()),
            shape[0], shape[1], shape[2], shape[3]);
    }
}

template<typename DistributedTensor>
__host__ inline DistributedTensor distributed_tensor_from_buffer(ParallelBuffer& t, int B, int D, int R, int C) {
    if constexpr (DistributedTensor::multicast) {
        return make_distributed_tensor<DistributedTensor>(
            reinterpret_cast<uint64_t>(t.multicast_ptr_),
            reinterpret_cast<uint64_t*>(t.raw_ptrs_.data()),
            B, D, R, C);
    } else {
        return make_distributed_tensor<DistributedTensor>(
            reinterpret_cast<uint64_t*>(t.raw_ptrs_.data()),
            B, D, R, C);
    }
}

template<typename LocalTensor>
__host__ inline LocalTensor gl_from_tensor(const at::Tensor& t) {
    return local_tensor_from_tensor<LocalTensor>(t);
}

template<typename LocalTensor>
__host__ inline LocalTensor gl_from_tensor(const at::Tensor& t, int B, int D, int R, int C) {
    return local_tensor_from_tensor<LocalTensor>(t, B, D, R, C);
}

template<typename DistributedTensor>
__host__ inline DistributedTensor dbuf_from_buffer(ParallelBuffer& t) {
    return distributed_tensor_from_buffer<DistributedTensor>(t);
}

template<typename DistributedTensor>
__host__ inline DistributedTensor dbuf_from_buffer(ParallelBuffer& t, int B, int D, int R, int C) {
    return distributed_tensor_from_buffer<DistributedTensor>(t, B, D, R, C);
}

}  // namespace dist
