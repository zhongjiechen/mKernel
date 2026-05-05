/**
 * @file
 * @brief Construct dist::gl / dist::dbuf descriptors from release-owned buffers.
 */
#pragma once

#include "../common/types.cuh"
#include "dbuf.cuh"
#include "parallel_buffer.cuh"

#include <ATen/core/Tensor.h>
#include <array>

namespace dist {

template<typename GL>
__host__ inline GL gl_from_tensor(const at::Tensor& t) {
    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < (int)t.dim(); ++i) {
        shape[4 - t.dim() + i] = (int)t.size(i);
    }
    return make_gl<GL>(reinterpret_cast<uint64_t>(t.data_ptr()),
                       shape[0], shape[1], shape[2], shape[3]);
}

template<typename GL>
__host__ inline GL gl_from_tensor(const at::Tensor& t, int B, int D, int R, int C) {
    return make_gl<GL>(reinterpret_cast<uint64_t>(t.data_ptr()), B, D, R, C);
}

template<typename DBUF>
__host__ inline DBUF dbuf_from_buffer(ParallelBuffer& t) {
    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < (int)t.data_.dim(); ++i) {
        shape[4 - t.data_.dim() + i] = (int)t.data_.size(i);
    }

    if constexpr (DBUF::multicast) {
        return make_dbuf<DBUF>(reinterpret_cast<uint64_t>(t.multicast_ptr_),
                               reinterpret_cast<uint64_t*>(t.raw_ptrs_.data()),
                               shape[0], shape[1], shape[2], shape[3]);
    } else {
        return make_dbuf<DBUF>(reinterpret_cast<uint64_t*>(t.raw_ptrs_.data()),
                               shape[0], shape[1], shape[2], shape[3]);
    }
}

template<typename DBUF>
__host__ inline DBUF dbuf_from_buffer(ParallelBuffer& t, int B, int D, int R, int C) {
    if constexpr (DBUF::multicast) {
        return make_dbuf<DBUF>(reinterpret_cast<uint64_t>(t.multicast_ptr_),
                               reinterpret_cast<uint64_t*>(t.raw_ptrs_.data()),
                               B, D, R, C);
    } else {
        return make_dbuf<DBUF>(reinterpret_cast<uint64_t*>(t.raw_ptrs_.data()),
                               B, D, R, C);
    }
}

}  // namespace dist
