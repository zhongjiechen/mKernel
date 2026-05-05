/**
 * @file
 * @brief dist::detail::create_tensor_map — host-side TMA descriptor encoder.
 *
 * Standalone replacement for `kittens::detail::tma::create_tensor_map`.
 * Reads tile metadata (rows/cols/swizzle/dtype) from the `kittens::st<>` /
 * `kittens::sv<>` POD tile-type template arg and calls
 * `cuTensorMapEncodeTiled` directly.
 *
 * The kittens shared-tile types remain the metadata source — this header just
 * removes the call into `kittens::detail::tma::*` so dist:: owns the encoder.
 */

#pragma once

#include <cuda.h>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>

#include "../common/tk_types_shared_st.cuh"
#include "../common/tk_types_shared_sv.cuh"

namespace dist {
namespace detail {

namespace tma_encode_internal {

template<typename DT>
constexpr CUtensorMapDataType pick_format() {
    if constexpr (std::is_same_v<DT, kittens::bf16>)        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if constexpr (std::is_same_v<DT, kittens::half>)   return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    else if constexpr (std::is_same_v<DT, float>)           return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    else if constexpr (std::is_same_v<DT, kittens::fp8e4m3>) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    else if constexpr (std::is_same_v<DT, kittens::fp8e5m2>) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    else                                                     return CUtensorMapDataType(-1);
}

inline std::string format_tma_error(const char* what, const char* err_str,
                                    int batch, int depth, int rows, int cols) {
    std::ostringstream oss;
    oss << "dist::create_tensor_map " << what << ": "
        << (err_str ? err_str : "unknown")
        << "  batch=" << batch << " depth=" << depth
        << " rows=" << rows << " cols=" << cols;
    return oss.str();
}

} // namespace tma_encode_internal

/**
 * @brief Build a CUtensorMap for shared-tile type ST + axis.
 *
 * Mirrors the layout/swizzle setup that the TMA hardware expects. The
 * algorithm follows the TMA programming guide: shape/stride arrays in
 * elements (and bytes for stride), with swizzle adding a leading
 * "swizzle_elements" dimension.
 *
 * @tparam ST    `kittens::st<...>` (or `kittens::sv<...>` with axis=-1)
 * @tparam axis  TMA axis: 0/1/2 for st, -1 for sv. Default-2 for st.
 */
/* ----- SV (vector) helpers ----- */

template<typename SV, int D = 16>
struct sv_find_vector_divider {
    static constexpr int value = (SV::length % (16 * D) == 0)
        ? 16 * D : sv_find_vector_divider<SV, D - 1>::value;
};
template<typename SV> struct sv_find_vector_divider<SV, 1> { static constexpr int value = 16; };
template<typename SV> constexpr int sv_inner_dim = sv_find_vector_divider<SV>::value;

/* ----- Tile (st) overload, axis 0/1/2 ----- */

template<typename ST, int axis,
         typename = std::enable_if_t<kittens::ducks::st::all<ST>>>
__host__ static inline void create_tensor_map(
    CUtensorMap *tma_map, const typename ST::dtype *src,
    int batch, int depth, int rows, int cols
) {
    using dtype = typename ST::dtype;
    static_assert(axis == 0 || axis == 1 || axis == 2,
                  "create_tensor_map<ST>: tile axis must be 0/1/2");

    constexpr uint32_t tma_dim = ST::swizzle ? 5 : 4;
    void* global_addr = (void*)(src);

    constexpr CUtensorMapDataType     tma_format      = tma_encode_internal::pick_format<dtype>();
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = ST::swizzle ? (
        ST::swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
        ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        CU_TENSOR_MAP_SWIZZLE_NONE
    ) : CU_TENSOR_MAP_SWIZZLE_NONE;

    uint64_t gmem_shape [5] = {0, 0, 0, 0, 0};
    uint64_t gmem_stride[4] = {0, 0, 0, 0};
    uint32_t smem_shape [5] = {0, 0, 0, 0, 0};
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    constexpr uint64_t shared_tile_height = ST::rows;
    constexpr uint64_t shared_tile_width  = ST::cols;
    constexpr int swizzle_elements        = ST::swizzle_bytes / sizeof(dtype);

    if constexpr (ST::swizzle) {
        if constexpr (axis == 2) {
            gmem_shape[0] = swizzle_elements;
            gmem_shape[1] = (uint64_t)rows;
            gmem_shape[2] = ((uint64_t)cols + swizzle_elements - 1) / swizzle_elements;
            gmem_shape[3] = (uint64_t)depth;
            gmem_shape[4] = (uint64_t)batch;
            gmem_stride[0] = (uint64_t)cols * sizeof(dtype);
            gmem_stride[1] = ST::swizzle_bytes;
            gmem_stride[2] = (uint64_t)rows * cols * sizeof(dtype);
            gmem_stride[3] = (uint64_t)depth * rows * cols * sizeof(dtype);
        } else if constexpr (axis == 1) {
            gmem_shape[0] = swizzle_elements;
            gmem_shape[1] = (uint64_t)depth;
            gmem_shape[2] = ((uint64_t)cols + swizzle_elements - 1) / swizzle_elements;
            gmem_shape[3] = (uint64_t)rows;
            gmem_shape[4] = (uint64_t)batch;
            gmem_stride[0] = (uint64_t)rows * cols * sizeof(dtype);
            gmem_stride[1] = ST::swizzle_bytes;
            gmem_stride[2] = (uint64_t)cols * sizeof(dtype);
            gmem_stride[3] = (uint64_t)depth * rows * cols * sizeof(dtype);
        } else {  // axis == 0
            gmem_shape[0] = swizzle_elements;
            gmem_shape[1] = (uint64_t)batch;
            gmem_shape[2] = ((uint64_t)cols + swizzle_elements - 1) / swizzle_elements;
            gmem_shape[3] = (uint64_t)rows;
            gmem_shape[4] = (uint64_t)depth;
            gmem_stride[0] = (uint64_t)depth * rows * cols * sizeof(dtype);
            gmem_stride[1] = ST::swizzle_bytes;
            gmem_stride[2] = (uint64_t)cols * sizeof(dtype);
            gmem_stride[3] = (uint64_t)rows * cols * sizeof(dtype);
        }
        smem_shape[0] = swizzle_elements;
        smem_shape[1] = shared_tile_height;
        smem_shape[2] = shared_tile_width / swizzle_elements;
        smem_shape[3] = 1;
        smem_shape[4] = 1;
    } else {
        static_assert(axis == 2, "Non-swizzled tiles only support axis 2.");
        gmem_shape[0] = (uint64_t)cols;
        gmem_shape[1] = (uint64_t)rows;
        gmem_shape[2] = (uint64_t)depth;
        gmem_shape[3] = (uint64_t)batch;
        gmem_stride[0] = (uint64_t)cols * sizeof(dtype);
        gmem_stride[1] = (uint64_t)rows * cols * sizeof(dtype);
        gmem_stride[2] = (uint64_t)depth * rows * cols * sizeof(dtype);
        smem_shape[0] = shared_tile_width;
        smem_shape[1] = shared_tile_height;
        smem_shape[2] = 1;
        smem_shape[3] = 1;
    }

    // Alignment + bound assertions (must match TMA hardware requirements).
    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);
    assert(gmem_stride[0] % 16 == 0);
    assert(gmem_stride[1] % 16 == 0);
    assert(gmem_stride[2] % 16 == 0);
    assert(gmem_stride[3] % 16 == 0);
    assert(smem_shape[0] <= 256);
    assert(smem_shape[1] <= 256);
    assert(smem_shape[2] <= 256);
    assert((smem_shape[0] * sizeof(dtype)) % 16 == 0);
    if constexpr (tma_swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        assert(smem_shape[0] * sizeof(dtype) <= ST::swizzle_bytes);
    }

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, tma_format, tma_dim, global_addr,
        gmem_shape, gmem_stride, smem_shape, smem_stride,
        tma_interleave, tma_swizzle, tma_l2Promotion, tma_oobFill);

    if (result != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(result, &err_str);
        throw std::runtime_error(
            tma_encode_internal::format_tma_error("tile", err_str, batch, depth, rows, cols));
    }
}

/* ----- SV (vector) overload, axis = -1 ----- */

template<typename SV, int axis,
         typename = std::enable_if_t<kittens::ducks::sv::all<SV>>,
         typename = void>
__host__ static inline void create_tensor_map(
    CUtensorMap *tma_map, const typename SV::dtype *src,
    int batch, int depth, int rows, int cols
) {
    using dtype = typename SV::dtype;
    static_assert(axis == -1, "create_tensor_map<SV>: vector axis must be -1");
    static_assert(SV::length <= 256 || (SV::length * sizeof(dtype)) % 128 == 0);

    constexpr uint32_t tma_dim = 4;
    void* global_addr = (void*)(src);

    constexpr CUtensorMapDataType     tma_format      = tma_encode_internal::pick_format<dtype>();
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_NONE;

    constexpr uint64_t dim1 = sv_inner_dim<SV>;

    uint64_t gmem_shape [4] = {(uint64_t)cols, (uint64_t)rows,
                               (uint64_t)depth, (uint64_t)batch};
    uint64_t gmem_stride[3] = {(uint64_t)cols * sizeof(dtype),
                               (uint64_t)cols * rows * sizeof(dtype),
                               (uint64_t)cols * rows * depth * sizeof(dtype)};
    uint32_t smem_shape [4] = {(uint32_t)dim1, 1, 1, 1};
    uint32_t smem_stride[4] = {1, 1, 1, 1};

    assert((reinterpret_cast<uint64_t>(global_addr) & 0b1111) == 0);
    assert(smem_shape[0] <= 256);

    CUresult result = cuTensorMapEncodeTiled(
        tma_map, tma_format, tma_dim, global_addr,
        gmem_shape, gmem_stride, smem_shape, smem_stride,
        tma_interleave, tma_swizzle, tma_l2Promotion, tma_oobFill);

    if (result != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(result, &err_str);
        throw std::runtime_error(
            tma_encode_internal::format_tma_error("vector", err_str, batch, depth, rows, cols));
    }
}

} // namespace detail
} // namespace dist
