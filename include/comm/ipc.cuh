/**
 * @file
 * @brief IPC handle export/import helpers for GPU memory.
 *
 * Supports legacy CUDA IPC handles (`cudaIpcMemHandle_t`) and VMM
 * POSIX-FD-based handles.
 */
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common/cuda_checks.cuh"
#include "vmm.cuh"

// Adapted from ThunderKittens (MIT): include/types/system/ipc.cuh
namespace comm::ipc {

namespace handle {
struct identifier {};

template <typename T>
concept all = requires { typename T::identifier; } &&
              std::is_same_v<typename T::identifier, identifier>;
} // namespace handle

enum class Flavor {
    kLegacy = 0,
    kVmmFd = 1,
};

template <Flavor flavor_v>
struct Handle;

template <>
struct Handle<Flavor::kLegacy> {
    using identifier = handle::identifier;
    static constexpr Flavor flavor = Flavor::kLegacy;
    cudaIpcMemHandle_t value{};
};

template <>
struct Handle<Flavor::kVmmFd> {
    using identifier = handle::identifier;
    static constexpr Flavor flavor = Flavor::kVmmFd;
    int value = -1;
};

__host__ inline void check_support(int device_id) {
    CUdevice device;
    MKERNEL_CUCHECK(cuDeviceGet(&device, device_id));

    int ipc_supported = 0;
    MKERNEL_CUDACHECK(cudaDeviceGetAttribute(&ipc_supported, cudaDevAttrIpcEventSupport, device_id));

    int fd_handle_supported = 0;
    MKERNEL_CUCHECK(cuDeviceGetAttribute(
        &fd_handle_supported,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
        device));

    if (!ipc_supported || !fd_handle_supported) {
        throw std::runtime_error("CUDA IPC is not supported on this device");
    }
}

template <handle::all IpcHandle>
__host__ inline void export_handle(IpcHandle *ipc_handle, void *ptr) {
    if constexpr (IpcHandle::flavor == Flavor::kLegacy) {
        MKERNEL_CUDACHECK(cudaIpcGetMemHandle(&ipc_handle->value, ptr));
    } else if constexpr (IpcHandle::flavor == Flavor::kVmmFd) {
        CUmemGenericAllocationHandle mem_handle;
        vmm::retain_handle(&mem_handle, ptr);
        // The exported FD must be closed by the importer.
        MKERNEL_CUCHECK(cuMemExportToShareableHandle(
            &ipc_handle->value, mem_handle, vmm::kHandleType, 0));
        vmm::release(mem_handle);
    } else {
        static_assert(IpcHandle::flavor == Flavor::kLegacy || IpcHandle::flavor == Flavor::kVmmFd);
    }
}

template <handle::all IpcHandle>
__host__ inline void export_handle(IpcHandle *ipc_handle, CUmemGenericAllocationHandle &mem_handle) {
    if constexpr (IpcHandle::flavor == Flavor::kVmmFd) {
        MKERNEL_CUCHECK(cuMemExportToShareableHandle(
            &ipc_handle->value, mem_handle, vmm::kHandleType, 0));
    } else {
        throw std::runtime_error("Can only export allocation handles using VMM FD flavor");
    }
}

template <handle::all IpcHandle>
__host__ inline void import_handle(
    void **ptr,
    IpcHandle &ipc_handle,
    size_t size,
    int local_world_size) {
    if constexpr (IpcHandle::flavor == Flavor::kLegacy) {
        MKERNEL_CUDACHECK(cudaIpcOpenMemHandle(ptr, ipc_handle.value, cudaIpcMemLazyEnablePeerAccess));
    } else if constexpr (IpcHandle::flavor == Flavor::kVmmFd) {
        CUmemGenericAllocationHandle mem_handle;
        MKERNEL_CUCHECK(cuMemImportFromShareableHandle(
            &mem_handle,
            reinterpret_cast<void *>(static_cast<uintptr_t>(ipc_handle.value)),
            vmm::kHandleType));
        vmm::map(ptr, mem_handle, size);
        vmm::set_access(*ptr, size, local_world_size);
        vmm::release(mem_handle);
        close(ipc_handle.value);
        ipc_handle.value = -1;
    } else {
        static_assert(IpcHandle::flavor == Flavor::kLegacy || IpcHandle::flavor == Flavor::kVmmFd);
    }
}

template <handle::all IpcHandle>
__host__ inline void import_handle(
    CUmemGenericAllocationHandle *mem_handle,
    IpcHandle &ipc_handle) {
    if constexpr (IpcHandle::flavor == Flavor::kVmmFd) {
        MKERNEL_CUCHECK(cuMemImportFromShareableHandle(
            mem_handle,
            reinterpret_cast<void *>(static_cast<uintptr_t>(ipc_handle.value)),
            vmm::kHandleType));
        close(ipc_handle.value);
        ipc_handle.value = -1;
    } else {
        throw std::runtime_error("Can only import allocation handles using VMM FD flavor");
    }
}

template <Flavor flavor_v>
__host__ inline void free_imported_mapping(void *ptr, size_t size) {
    if constexpr (flavor_v == Flavor::kLegacy) {
        MKERNEL_CUDACHECK(cudaIpcCloseMemHandle(ptr));
    } else if constexpr (flavor_v == Flavor::kVmmFd) {
        vmm::unmap(ptr, size);
    } else {
        static_assert(flavor_v == Flavor::kLegacy || flavor_v == Flavor::kVmmFd);
    }
}

} // namespace comm::ipc
