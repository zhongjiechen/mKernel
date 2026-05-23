/**
 * @file
 * @brief CUDA VMM allocation, mapping, and multicast helpers.
 *
 * This file wraps CUDA driver VMM APIs for:
 * - shareable allocation handles (FD export/import path),
 * - mapping/unmapping and access control across local devices,
 * - multicast object setup/binding for multi-GPU communication patterns.
 */
#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common/cuda_checks.cuh"

// Adapted from ThunderKittens (MIT): include/types/system/vmm.cuh
namespace comm::vmm {

// Intra-node shareable handle type for cuMemExportToShareableHandle.
static constexpr CUmemAllocationHandleType kHandleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

using Handle = CUmemGenericAllocationHandle;

__host__ inline void alloc(
    Handle *handle,
    size_t *allocated_size,
    size_t size,
    int device_id) {
    CUmemAllocationProp prop = {};
    prop.location.id = device_id;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = kHandleType;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;

    size_t granularity = 0;
    MKERNEL_CUCHECK(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    *allocated_size = ((size + granularity - 1) / granularity) * granularity;

    MKERNEL_CUCHECK(cuMemCreate(handle, *allocated_size, &prop, 0));
}

__host__ inline void map(void **ptr, const Handle &handle, size_t size) {
    CUdeviceptr device_ptr = 0;
    MKERNEL_CUCHECK(cuMemAddressReserve(&device_ptr, size, 0, 0, 0));
    MKERNEL_CUCHECK(cuMemMap(device_ptr, size, 0, handle, 0));
    *ptr = reinterpret_cast<void *>(device_ptr);
}

// Required for multicast access patterns: same virtual address on all processes.
__host__ inline void map_at(void **ptr, void *requested_addr, const Handle &handle, size_t size) {
    CUdeviceptr device_ptr = reinterpret_cast<CUdeviceptr>(requested_addr);
    MKERNEL_CUCHECK(cuMemAddressReserve(&device_ptr, size, 0, device_ptr, 0));
    MKERNEL_CUCHECK(cuMemMap(device_ptr, size, 0, handle, 0));
    *ptr = reinterpret_cast<void *>(device_ptr);
}

__host__ inline void set_access(void *ptr, size_t size, int num_devices) {
    std::vector<CUmemAccessDesc> descs(num_devices);
    for (int i = 0; i < num_devices; ++i) {
        descs[i].location.id = i;
        descs[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        descs[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    MKERNEL_CUCHECK(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(ptr), size, descs.data(), num_devices));
}

__host__ inline void retain_handle(Handle *handle, void *ptr) {
    MKERNEL_CUCHECK(cuMemRetainAllocationHandle(handle, ptr));
}

__host__ inline void unmap(void *ptr, size_t size) {
    MKERNEL_CUCHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr), size));
    MKERNEL_CUCHECK(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), size));
}

__host__ inline void release(Handle &handle) {
    MKERNEL_CUCHECK(cuMemRelease(handle));
}

__host__ inline void alloc_map_set_access(
    void **ptr,
    size_t *allocated_size,
    size_t size,
    int device_id,
    int num_devices) {
    Handle handle;
    alloc(&handle, allocated_size, size, device_id);
    map(ptr, handle, *allocated_size);
    set_access(*ptr, *allocated_size, num_devices);
    release(handle);
}

__host__ inline void multicast_check(int device_id) {
    CUdevice device;
    MKERNEL_CUCHECK(cuDeviceGet(&device, device_id));

    int multicast_supported = 0;
    MKERNEL_CUCHECK(cuDeviceGetAttribute(
        &multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device));

    if (!multicast_supported) {
        throw std::runtime_error("Device does not support multicast");
    }
}

__host__ inline void multicast_create_handle(
    Handle *handle,
    size_t *allocated_size,
    size_t size,
    int num_devices) {
    if (num_devices <= 1) {
        throw std::runtime_error("Multicast requires at least 2 devices");
    }

    CUmulticastObjectProp prop = {};
    prop.numDevices = num_devices;
    prop.handleTypes = kHandleType;

    size_t granularity = 0;
    MKERNEL_CUCHECK(cuMulticastGetGranularity(
        &granularity, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    *allocated_size = ((size + granularity - 1) / granularity) * granularity;
    prop.size = *allocated_size;

    MKERNEL_CUCHECK(cuMulticastCreate(handle, &prop));
}

__host__ inline void multicast_bind_device(const Handle &handle, int device_id) {
    CUdevice device;
    MKERNEL_CUCHECK(cuDeviceGet(&device, device_id));
    MKERNEL_CUCHECK(cuMulticastAddDevice(handle, device));
}

__host__ inline void multicast_bind_memory(
    const Handle &multicast_handle,
    const Handle &memory_handle,
    size_t size) {
    CUresult mc_bind_result = cuMulticastBindMem(multicast_handle, 0, memory_handle, 0, size, 0);
    if (mc_bind_result != CUDA_SUCCESS) {
        const char *err_str = nullptr;
        const char *err_name = nullptr;
        cuGetErrorString(mc_bind_result, &err_str);
        cuGetErrorName(mc_bind_result, &err_name);
        int dev = -1;
        cudaGetDevice(&dev);
        std::fprintf(
            stderr,
            "cuMulticastBindMem failed: dev=%d multicast_handle=0x%llx memory_handle=0x%llx size=%zu err=%s (%s)\n",
            dev,
            static_cast<unsigned long long>(multicast_handle),
            static_cast<unsigned long long>(memory_handle),
            size,
            err_name ? err_name : "unknown",
            err_str ? err_str : "unknown"
        );
        std::abort();
    }
}

__host__ inline void multicast_bind_address(
    const Handle &multicast_handle,
    void *ptr,
    size_t size) {
    Handle memory_handle;
    retain_handle(&memory_handle, ptr);
    multicast_bind_memory(multicast_handle, memory_handle, size);
    release(memory_handle);
}

__host__ inline void multicast_unbind_device(const Handle &handle, size_t size, int device_id) {
    CUdevice device;
    MKERNEL_CUCHECK(cuDeviceGet(&device, device_id));
    MKERNEL_CUCHECK(cuMulticastUnbind(handle, device, 0, size));
}

} // namespace comm::vmm
