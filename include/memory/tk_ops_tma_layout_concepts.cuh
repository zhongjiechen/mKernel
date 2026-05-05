/**
 * @file
 * @brief Structural concepts for TMA-capable global layouts.
 */

#pragma once

#include "../common/types.cuh"

#include <concepts>
#include <cstddef>

namespace kittens {
namespace tma {
namespace detail {

template<typename Layout, typename TmaType, int axis>
concept tma_layout_for = requires(const Layout& layout) {
    typename Layout::dtype;
    { layout.template get_tma<TmaType, axis>() } -> std::convertible_to<const CUtensorMap*>;
    { layout.template shape<0>() } -> std::convertible_to<size_t>;
    { layout.template stride<0>() } -> std::convertible_to<size_t>;
};

} // namespace detail
} // namespace tma
} // namespace kittens
