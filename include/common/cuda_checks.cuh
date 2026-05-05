/**
 * @file
 * @brief Shared CUDA runtime/driver error-check helpers used by communication modules.
 *
 * This file is not a transport primitive itself; it provides consistent failure
 * handling for IPC, VMM, multicast, and multimem wrappers.
 */
#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

// Adapted from ThunderKittens (MIT): include/common/util.cuh
#define OSGC_CUCHECK(cmd) do {                                 \
    CUresult err__ = (cmd);                                    \
    if (err__ != CUDA_SUCCESS) {                               \
        const char *err_str__ = nullptr;                       \
        cuGetErrorString(err__, &err_str__);                   \
        std::fprintf(stderr, "CUDA driver error %s:%d '%s'\n", \
            __FILE__, __LINE__, err_str__ ? err_str__ : "");   \
        std::exit(EXIT_FAILURE);                               \
    }                                                          \
} while (0)

// Adapted from ThunderKittens (MIT): include/common/util.cuh
#define OSGC_CUDACHECK(cmd) do {                                \
    cudaError_t err__ = (cmd);                                  \
    if (err__ != cudaSuccess) {                                 \
        std::fprintf(stderr, "CUDA runtime error %s:%d '%s'\n", \
            __FILE__, __LINE__, cudaGetErrorString(err__));     \
        std::exit(EXIT_FAILURE);                                \
    }                                                           \
} while (0)
