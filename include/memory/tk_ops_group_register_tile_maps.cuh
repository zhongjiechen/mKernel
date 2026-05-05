/**
 * @file
 * @brief Minimal register-tile maps used by the release kernels.
 */

template<typename op, ducks::rt::all T>
__device__ static inline void unary_map(T &dst, const T &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k]);
            }
        }
    }
}

template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename T::dtype &param) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(src.tiles[i][j].data[k], param);
            }
        }
    }
}

template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    bin_map<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}

template<typename op, ducks::rt::all T>
__device__ static inline void bin_map(T &dst, const T &lhs, const T &rhs) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst.tiles[i][j].data[k] = op::template op<typename T::dtype>(lhs.tiles[i][j].data[k], rhs.tiles[i][j].data[k]);
            }
        }
    }
}

template<typename op, ducks::rt::row_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(V::outer_dim == T::height);

    using dtype = T::dtype;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        dtype packed_top_row    = base_types::packing<dtype>::pack(row_values[i][0].x);
        dtype packed_bottom_row = base_types::packing<dtype>::pack(row_values[i][0].y);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k += 2) {
                dst.tiles[i][j].data[k + 0] = op::template op<dtype>(src.tiles[i][j].data[k + 0], packed_top_row);
                dst.tiles[i][j].data[k + 1] = op::template op<dtype>(src.tiles[i][j].data[k + 1], packed_bottom_row);
            }
        }
    }
}

template<typename op, ducks::rt::col_layout T, ducks::rv::all V>
__device__ static inline void row_map(T &dst, const T &src, const V &row_values) {
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>);
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    static_assert(V::outer_dim == T::height);

    using dtype = T::dtype;
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile / 2; k++) {
                dst.tiles[i][j].data[k + 0] = op::template op<dtype>(src.tiles[i][j].data[k + 0], row_values[i][0]);
                dst.tiles[i][j].data[k + 2] = op::template op<dtype>(src.tiles[i][j].data[k + 2], row_values[i][1]);
            }
        }
    }
}

template<ducks::rt::all T>
__device__ static inline void zero(T &dst) {
    unary_map<base_ops::zero, T>(dst, dst);
}

template<ducks::rt::all T>
__device__ static inline void exp2(T &dst, const T &src) {
    unary_map<base_ops::exp2, T>(dst, src);
}

template<ducks::rt::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sum, T>(dst, lhs, rhs);
}

template<ducks::rt::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::sub, T>(dst, lhs, rhs);
}

template<ducks::rt::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_map<base_ops::mul, T>(dst, lhs, rhs);
}

template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void sub_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::sub, T, V>(dst, src, row_values);
}

template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void mul_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::mul, T, V>(dst, src, row_values);
}

template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void div_row(T &dst, const T &src, const V &row_values) {
    row_map<base_ops::div, T, V>(dst, src, row_values);
}
