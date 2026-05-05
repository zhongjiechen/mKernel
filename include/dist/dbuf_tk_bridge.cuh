/**
 * @file
 * @brief Thin TK adapter for constructing dist::dbuf / dist::gl from
 *        TKParallelTensor / at::Tensor. Kept separate from `dbuf.cuh` so
 *        non-TK call sites only include `dbuf.cuh` + `gl.cuh`.
 *
 * Produces `dist::dbuf` directly from TKParallelTensor metadata — no
 * intermediate pgl.
 */

#pragma once

#include "dbuf.cuh"
#include "../pyutils/parallel_tensor.cuh"
#include <ATen/core/Tensor.h>
#include <array>

namespace dist {

/**
 * @brief Build a dist::gl from an at::Tensor (single-device).
 */
template<typename GL>
__host__ inline GL gl_from_tensor(const at::Tensor& t) {
    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < (int)t.dim(); ++i) shape[4 - t.dim() + i] = (int)t.size(i);
    return make_gl<GL>(reinterpret_cast<uint64_t>(t.data_ptr()),
                       shape[0], shape[1], shape[2], shape[3]);
}

template<typename GL>
__host__ inline GL gl_from_tensor(const at::Tensor& t, int B, int D, int R, int C) {
    return make_gl<GL>(reinterpret_cast<uint64_t>(t.data_ptr()), B, D, R, C);
}

/**
 * @brief Build a dist::dbuf (intra-node) from a TKParallelTensor.
 *
 * Inter-node fields (channels[]) are left default-initialized — they're
 * populated separately by a backend-specific `bind_inter_*()` call when
 * NUM_CHANNELS > 0.
 */
template<typename DBUF>
__host__ inline DBUF dbuf_from_tkpt(kittens::py::TKParallelTensor& t) {
    std::array<int, 4> shape = {1, 1, 1, 1};
    for (int i = 0; i < (int)t.data_.dim(); ++i) shape[4 - t.data_.dim() + i] = (int)t.data_.size(i);

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
__host__ inline DBUF dbuf_from_tkpt(kittens::py::TKParallelTensor& t,
                                    int B, int D, int R, int C) {
    if constexpr (DBUF::multicast) {
        return make_dbuf<DBUF>(reinterpret_cast<uint64_t>(t.multicast_ptr_),
                               reinterpret_cast<uint64_t*>(t.raw_ptrs_.data()),
                               B, D, R, C);
    } else {
        return make_dbuf<DBUF>(reinterpret_cast<uint64_t*>(t.raw_ptrs_.data()),
                               B, D, R, C);
    }
}

/* Back-compat aliases for the M1 names. */
template<typename DBUF> __host__ inline DBUF make_dbuf_intra(kittens::py::TKParallelTensor& t) {
    return dbuf_from_tkpt<DBUF>(t);
}
template<typename DBUF> __host__ inline DBUF make_dbuf_intra(kittens::py::TKParallelTensor& t,
                                                              int B, int D, int R, int C) {
    return dbuf_from_tkpt<DBUF>(t, B, D, R, C);
}

} // namespace dist
