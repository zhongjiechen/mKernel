/**
 * @file
 * @brief An aggregate header for all group-scope MMA operations.
 */

// All compilation targets can use the warp-scope MMA operations.

// Hopper and Blackwell both support warpgroup-scope MMA (WGMMA) operations.
#include "tk_ops_group_mma_warpgroup.cuh"

// Blackwell has its own MMA operations (Tensor Core Generation 5).
