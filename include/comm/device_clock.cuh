/**
 * @file
 * @brief Device-side clock reads shared by communication kernels.
 */
#pragma once

namespace comm {

__device__ inline unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

} // namespace comm
