/**
 * @file
 * @brief An aggregate header of all single-threaded operations defined by ThunderKittens
 */

#pragma once

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include "tk_ops_thread_memory_tile_tma.cuh"
#include "tk_ops_thread_memory_vec_tma.cuh"
#endif
#include "tk_ops_thread_util_util.cuh"
