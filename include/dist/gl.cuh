/**
 * @file
 * @brief dist::gl — per-device global-memory layout descriptor.
 *
 * Standalone replacement for `kittens::gl<>`. Owns a raw device pointer,
 * compile-time-or-runtime shape (B/D/R/C), and a TMA-descriptor cache keyed
 * by (TileType, axis).
 *
 * TK touch points (intentional, scoped):
 *   - Tile metadata types (`kittens::st<...>`, `kittens::sv<...>`) are accepted
 *     as `TileTypes...` template args — POD metadata (rows/cols/swizzle/dtype).
 *     `dist::detail::tma_descriptor<ST>` derives the TMA axis from the duck
 *     concept (st => 2, sv => -1).
 *   - `dist::detail::create_tensor_map<TILE, axis>` calls the CUDA driver
 *     `cuTensorMapEncodeTiled` directly. No `kittens::detail::tma::*` call.
 *
 * No inheritance from `kittens::gl`. Layout, accessors, descriptor cache, and
 * encoder are all defined in this header / its sibling.
 */

#pragma once

#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <string>

#include "tma_encode.cuh"                    // dist::detail::create_tensor_map
#include "../common/tk_types_shared_st.cuh"  // kittens::st<> + ducks::st concept (POD tile metadata)
#include "../common/tk_types_shared_sv.cuh"  // kittens::sv<> + ducks::sv concept
#include "../common/tk_types_global_util.cuh" // kittens::coord<> indexing helper

namespace dist {

/* ----------   Compile-time / runtime dimension wrappers  ---------- */

namespace detail {

template<int v> struct ct_dim {
    static constexpr size_t value = v;
    __host__ __device__ inline ct_dim(std::nullptr_t) {}
    __host__ __device__ inline constexpr operator size_t() const { return v; }
};
struct rt_dim {
    size_t value;
    __host__ __device__ inline rt_dim(size_t v) : value(v) {}
    __host__ __device__ inline operator size_t() const { return value; }
};
template<int d> using dim_t  = std::conditional_t<(d == -1), rt_dim, ct_dim<d>>;
template<int d> using arg_t  = std::conditional_t<(d == -1), size_t, std::nullptr_t>;

template<int N> __host__ __device__ inline auto make_arg(int v) {
    if constexpr (N > 0) return nullptr;
    else                 return (size_t)v;
}

/* ----------   TMA descriptor (axis resolution from tile type)  ---------- */

template<typename ST>
struct tma_descriptor {
    using T = ST;
    static constexpr int axis = (kittens::ducks::sv::all<ST> ? -1 : 2);
};

/* ----------   TMA descriptor cache (variadic recursion)  ---------- */

template<typename... Args>
struct tma_dict {
    __host__ tma_dict() {}
    template<typename DT> __host__ tma_dict(DT*, int, int, int, int) {}
    __host__ __device__ tma_dict(const tma_dict&) {}
    template<typename U, int A> __device__ inline const CUtensorMap* get() const { return nullptr; }
};

template<typename ST, typename... Rest>
struct tma_dict<ST, Rest...> {
    using DESC = tma_descriptor<ST>;
    using TILE = typename DESC::T;
    static constexpr int AXIS = DESC::axis;

    CUtensorMap          desc;
    tma_dict<Rest...>    rest;

    __host__ tma_dict() {}
    __host__ tma_dict(typename TILE::dtype* data, int b, int d, int r, int c)
        : rest(data, b, d, r, c) {
        ::dist::detail::create_tensor_map<TILE, AXIS>(&desc, data, b, d, r, c);
    }
    __host__ __device__ tma_dict(const tma_dict& other)
        : desc(other.desc), rest(other.rest) {}

    template<typename U, int A> __device__ inline const CUtensorMap* get() const {
        if constexpr (std::is_same_v<TILE, U> && AXIS == A) { return &desc; }
        else                                                { return rest.template get<U, A>(); }
    }
};

} // namespace detail


/* ----------   Per-device global layout  ---------- */

/**
 * @brief Per-device global tensor layout. Standalone — does not inherit from
 * `kittens::gl`.
 *
 * @tparam _T          element type (bf16, fp16, fp32, int, ...)
 * @tparam B,D,R,C     dimensions: positive = compile-time, -1 = runtime.
 * @tparam TMA_Types   `kittens::st<>` tile metadata types to pre-build descriptors for.
 */
template<typename _T, int B, int D, int R, int C, typename... TMA_Types>
struct gl {
    using dtype = _T;
    using T     = _T;

    T* raw_ptr;
    static constexpr int __b__ = B, __d__ = D, __r__ = R, __c__ = C;

    detail::dim_t<B> batch_internal;
    detail::dim_t<D> depth_internal;
    detail::dim_t<R> rows_internal;
    detail::dim_t<C> cols_internal;

    detail::tma_dict<TMA_Types...> tma_descs;

    /* ----- ctor ----- */
    __host__ inline gl(T* data,
                       detail::arg_t<B> b_arg,
                       detail::arg_t<D> d_arg,
                       detail::arg_t<R> r_arg,
                       detail::arg_t<C> c_arg)
        : raw_ptr(data),
          batch_internal(b_arg), depth_internal(d_arg),
          rows_internal(r_arg),  cols_internal(c_arg),
          tma_descs(data,
                    static_cast<int>(static_cast<size_t>(batch_internal)),
                    static_cast<int>(static_cast<size_t>(depth_internal)),
                    static_cast<int>(static_cast<size_t>(rows_internal)),
                    static_cast<int>(static_cast<size_t>(cols_internal))) {}

    __host__ __device__ inline gl(const gl& o)
        : raw_ptr(o.raw_ptr),
          batch_internal(o.batch_internal),
          depth_internal(o.depth_internal),
          rows_internal(o.rows_internal),
          cols_internal(o.cols_internal),
          tma_descs(o.tma_descs) {}

    /* ----- shape accessors (TK-compatible API) ----- */
    template<int X = B> __device__ __host__ static constexpr std::enable_if_t<(X > 0), int> batch() { return X; }
    template<int X = B> __device__ __host__ std::enable_if_t<(X == -1), int> batch() const { return (int)(size_t)batch_internal; }
    template<int X = D> __device__ __host__ static constexpr std::enable_if_t<(X > 0), int> depth() { return X; }
    template<int X = D> __device__ __host__ std::enable_if_t<(X == -1), int> depth() const { return (int)(size_t)depth_internal; }
    template<int X = R> __device__ __host__ static constexpr std::enable_if_t<(X > 0), int> rows()  { return X; }
    template<int X = R> __device__ __host__ std::enable_if_t<(X == -1), int> rows()  const { return (int)(size_t)rows_internal; }
    template<int X = C> __device__ __host__ static constexpr std::enable_if_t<(X > 0), int> cols()  { return X; }
    template<int X = C> __device__ __host__ std::enable_if_t<(X == -1), int> cols()  const { return (int)(size_t)cols_internal; }

    __device__ __host__ inline size_t numel() const {
        return (size_t)batch() * depth() * rows() * cols();
    }

    template<int axis> __device__ inline size_t shape() const {
        if constexpr      (axis == 0) return (size_t)batch();
        else if constexpr (axis == 1) return (size_t)depth();
        else if constexpr (axis == 2) return (size_t)rows();
        else                          return (size_t)cols();
    }
    template<int axis> __device__ inline size_t stride() const {
        if constexpr      (axis == 0) return (size_t)depth() * rows() * cols();
        else if constexpr (axis == 1) return (size_t)rows() * cols();
        else if constexpr (axis == 2) return (size_t)cols();
        else                          return 1;
    }

    /* ----- linear access (TK-compatible) ----- */
    __device__ inline T& operator[](const kittens::coord<kittens::ducks::default_type>& idx) const {
        return raw_ptr[(((size_t)idx.b * depth() + idx.d) * rows() + idx.r) * cols() + idx.c];
    }

    /* ----- TMA descriptor lookup ----- */
    template<typename U, int axis> __device__ inline const CUtensorMap* get_tma() const {
        return tma_descs.template get<U, axis>();
    }
    template<typename U, int axis = 2> __device__ inline void prefetch_tma() const {
        const CUtensorMap* d = tma_descs.template get<U, axis>();
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(d)) : "memory");
    }
};

/* ----------   Host construction helper  ---------- */

template<typename GL>
__host__ inline GL make_gl(uint64_t data, int b, int d, int r, int c) {
    return GL(reinterpret_cast<typename GL::dtype*>(data),
              detail::make_arg<GL::__b__>(b),
              detail::make_arg<GL::__d__>(d),
              detail::make_arg<GL::__r__>(r),
              detail::make_arg<GL::__c__>(c));
}

} // namespace dist
