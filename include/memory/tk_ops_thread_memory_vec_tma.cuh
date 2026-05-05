/**
 * @file
 * @brief Minimal vector TMA operations used by the release kernels.
 */

#pragma once

#include "../common/tk_common_common.cuh"
#include "../common/types.cuh"
#include "tk_ops_tma_layout_concepts.cuh"
#include "tk_ops_thread_util_util.cuh"

namespace kittens {
namespace detail {
namespace tma {

template<typename SV, int D=16> struct find_vector_divider {
    static constexpr int value = (SV::length % (16*D) == 0 && (SV::length < 256 || ((16*D)*sizeof(typename SV::dtype)) % 128 == 0)) ?
        16*D : find_vector_divider<SV, D-1>::value;
};
template<typename SV> struct find_vector_divider<SV, 1> { static constexpr int value = 16; };
template<typename SV> constexpr int sv_tma_dim1 = find_vector_divider<SV>::value;
template<typename SV> constexpr int sv_tma_dim2 = (SV::length / sv_tma_dim1<SV>);

template<cache_policy policy>
__device__ static inline void vec_store_async_tma_internal(uint64_t tma_ptr, uint32_t src_i_ptr, coord<> tma_coord) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    } else {
        asm volatile (
            "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5}], [%1], %6;"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}

template<cache_policy policy>
__device__ static inline void vec_load_async_tma_internal(uint64_t tma_ptr, uint32_t dst_i_ptr, uint32_t mbar_ptr, coord<> tma_coord) {
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6}], [%2];"
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    } else {
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}

} // namespace tma
} // namespace detail

namespace tma {

template<cache_policy policy, ducks::sv::all SV, tma::detail::tma_layout_for<SV, -1> GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i * ::kittens::detail::tma::sv_tma_dim1<SV> * sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    store_commit_group();
}

template<ducks::sv::all SV, tma::detail::tma_layout_for<SV, -1> GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const GL &dst, const SV &src, const COORD &idx) {
    store_async<cache_policy::NORMAL, SV, GL, COORD>(dst, src, idx);
}

template<cache_policy policy, ducks::sv::all SV, tma::detail::tma_layout_for<SV, -1> GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i * ::kittens::detail::tma::sv_tma_dim1<SV> * sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_load_async_tma_internal<policy>(tma_ptr, dst_i_ptr, mbar_ptr, tma_coord);
    }
}

template<ducks::sv::all SV, tma::detail::tma_layout_for<SV, -1> GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar) {
    load_async<cache_policy::NORMAL, SV, GL, COORD>(dst, src, idx, bar);
}

} // namespace tma
} // namespace kittens
