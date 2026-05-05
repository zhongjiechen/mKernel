/**
 * @file
 * @brief Minimal register-tile conversions used by the release kernels.
 */

template<typename T, typename U, ducks::rt_layout::all layout>
__device__ static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src) {
    using T2 = typename base_types::packing<T>::packed_type;
    using U2 = typename base_types::packing<U>::packed_type;
    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
    }
}

template<typename T, typename U, int rows, int cols, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T, rows, cols, layout> &dst, const rt<U, rows, cols, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

template<int subtile_rows, ducks::rt::all RT>
__device__ static inline rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout> &subtile_inplace(RT &src, int idx) {
    KITTENS_CHECK_WARP
    using T = typename RT::T;
    static_assert(RT::height % (subtile_rows / TILE_ROW_DIM<T>) == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout>&>(
        src.tiles[idx * (subtile_rows / TILE_ROW_DIM<T>)]
    );
}
