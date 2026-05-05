/**
 * @file
 * @brief Minimal warpgroup tile TMA wrappers used by the release kernels.
 */

template<ducks::st::all ST, ::kittens::tma::detail::tma_layout_for<ST, dim::ROW> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
    }
}

template<ducks::st::all ST, ::kittens::tma::detail::tma_layout_for<ST, dim::ROW> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_add_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
    }
}

template<ducks::st::all ST, ::kittens::tma::detail::tma_layout_for<ST, dim::ROW> GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    if(laneid() == 0) {
        ::kittens::tma::load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar);
    }
}
