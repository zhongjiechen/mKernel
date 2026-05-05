/**
 * @file
 * @brief An aggregate header for all group-scope MMA operations.
 */

// All compilation targets can use the warp-scope MMA operations.

// Hopper and Blackwell both support warpgroup-scope MMA (WGMMA) operations.
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include "tk_ops_group_mma_warpgroup.cuh"
#endif

// Blackwell has its own MMA operations (Tensor Core Generation 5).
#ifdef KITTENS_BLACKWELL
#endif
