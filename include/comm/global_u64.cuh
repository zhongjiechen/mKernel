/**
 * @file
 * @brief Single-instruction 64-bit global memory helpers.
 */
#pragma once

#include <cstdint>

namespace comm {
namespace global_u64 {

template <typename PtrT>
__device__ inline uint64_t volatile_load(PtrT* ptr) {
    uint64_t v;
    asm volatile("ld.volatile.global.u64 %0, [%1];"
                 : "=l"(v)
                 : "l"(reinterpret_cast<const uint64_t*>(ptr))
                 : "memory");
    return v;
}

template <typename PtrT>
__device__ inline void store(PtrT* ptr, uint64_t v) {
    asm volatile("st.global.u64 [%0], %1;"
                 :
                 : "l"(reinterpret_cast<uint64_t*>(ptr)), "l"(v)
                 : "memory");
}

template <typename PtrT>
__device__ inline void release_store_gpu(PtrT* ptr, uint64_t v) {
    asm volatile("st.global.release.gpu.u64 [%0], %1;"
                 :
                 : "l"(reinterpret_cast<uint64_t*>(ptr)), "l"(v)
                 : "memory");
}

} // namespace global_u64
} // namespace comm
