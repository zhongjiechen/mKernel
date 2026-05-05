#pragma once

#include "../common/tk_common_common.cuh"
#include "../common/types.cuh"

namespace kittens {
namespace detail {
namespace wgmma {

// templated wrapper for PTX
template<typename T_D, typename T_AB, int cols, int trans_a, int trans_b, int inv=1>
struct base {
    template<int scale_b=1> __device__ static inline void rt_st(
        rt<T_D, 16, cols, ducks::rt_layout::row> &dst,
        const rt<T_AB, 16, cols, ducks::rt_layout::row> & a_rt,
        const uint64_t b_st_desc,
        int scale_d = 1
    );
    template<int scale_b=1> __device__ static inline void st_st(
        rt<T_D, 16, cols, ducks::rt_layout::row> &dst,
        const uint64_t a_st_desc,
        const uint64_t b_st_desc,
        int scale_d = 1
    );
};

#include "tk_ops_group_mma_base_64x64.impl"
#include "tk_ops_group_mma_base_64x128.impl"
#include "tk_ops_group_mma_base_64x256.impl"

} // namespace wgmma
} // namespace detail
} // namespace kittens
