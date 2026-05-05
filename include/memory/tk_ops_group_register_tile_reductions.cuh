/**
 * @file
 * @brief Minimal row reductions used by ring_attention.cu.
 */

template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::height);

    using dtype = V::dtype;
    const int leader = threadIdx.x & 0x1C;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        dtype accum_top_row    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[2]);
        dtype accum_bottom_row = op::template op<dtype>(src.tiles[i][0].data[1], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < src.width; j++) {
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile; k += 2) {
                accum_top_row    = op::template op<dtype>(accum_top_row,    src.tiles[i][j].data[k + 0]);
                accum_bottom_row = op::template op<dtype>(accum_bottom_row, src.tiles[i][j].data[k + 1]);
            }
        }
        dtype accum_packed;
        accum_packed.x = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_top_row.x, accum_top_row.y);
        accum_packed.y = op::template op<typename base_types::packing<dtype>::unpacked_type>(accum_bottom_row.x, accum_bottom_row.y);
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 2));
        accum_packed = op::template op<dtype>(accum_packed, packed_shfl_down_sync(MASK_ALL, accum_packed, 1));
        accum_packed = packed_shfl_sync(MASK_ALL, accum_packed, leader);
        if(reset) {
            row_accum[i][0] = accum_packed;
        } else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_packed);
        }
    }
}

template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::height);

    using dtype = V::dtype;
    const int leader = threadIdx.x & 0x3;
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        dtype accum_top_rows    = op::template op<dtype>(src.tiles[i][0].data[0], src.tiles[i][0].data[1]);
        dtype accum_bottom_rows = op::template op<dtype>(src.tiles[i][0].data[2], src.tiles[i][0].data[3]);
        #pragma unroll
        for(int j = 1; j < src.width; j++) {
            #pragma unroll
            for(int k = 0; k < src.packed_per_tile / 2; k++) {
                accum_top_rows    = op::template op<dtype>(accum_top_rows,    src.tiles[i][j].data[k + 0]);
                accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, src.tiles[i][j].data[k + 2]);
            }
        }
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 16));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 8));
        accum_top_rows = op::template op<dtype>(accum_top_rows, packed_shfl_down_sync(MASK_ALL, accum_top_rows, 4));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 16));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 8));
        accum_bottom_rows = op::template op<dtype>(accum_bottom_rows, packed_shfl_down_sync(MASK_ALL, accum_bottom_rows, 4));
        accum_top_rows    = packed_shfl_sync(MASK_ALL, accum_top_rows,    leader);
        accum_bottom_rows = packed_shfl_sync(MASK_ALL, accum_bottom_rows, leader);
        if(reset) {
            row_accum[i][0] = accum_top_rows;
            row_accum[i][1] = accum_bottom_rows;
        } else {
            row_accum[i][0] = op::template op<dtype>(src_accum[i][0], accum_top_rows);
            row_accum[i][1] = op::template op<dtype>(src_accum[i][1], accum_bottom_rows);
        }
    }
}

template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}

template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}
