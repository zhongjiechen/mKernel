/**
 * @file
 * @brief Umbrella include for communication primitives.
 *
 * Pulls in IPC, VMM/multicast, multimem, and atomic helpers.
 */
#pragma once

#include "atomic_u32.cuh"
#include "device_clock.cuh"
#include "global_u64.cuh"
#include "ipc.cuh"
#include "multimem.cuh"
#include "vmm.cuh"
