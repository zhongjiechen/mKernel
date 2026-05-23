/**
 * @file rdma_gpu_mr.cuh
 * @brief GPU buffer RDMA memory registration for GDR (GPU Direct RDMA).
 *
 * Registers GPU HBM buffers so the NIC can read/write them directly.
 * Tries ibv_reg_dmabuf_mr first (via CUDA VMM FD export), falls back
 * to ibv_reg_mr with nvidia_peermem.
 */
#pragma once

#include "rdma_transport.h"
#include "../../common/cuda_checks.cuh"
#include "../vmm.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <cstdio>
#include <cstdlib>

namespace internode {
namespace gpu_mr {

// IBV_ACCESS_RELAXED_ORDERING lets the NIC/PCIe fabric reorder writes to this
// MR, avoiding per-TLP fencing at the GPU memory controller.  Critical for
// GPUDirect RDMA into HBM that overlaps with GEMM/SM traffic.  The flag is in
// the OPTIONAL range so ibv_reg_mr silently ignores it if the HCA doesn't
// support it.
static constexpr int kDefaultAccess =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
    | IBV_ACCESS_RELAXED_ORDERING;

/**
 * Register an existing GPU buffer for RDMA.
 *
 * Strategy:
 *   1. Try ibv_reg_dmabuf_mr via CUDA VMM FD export (preferred on GH200)
 *   2. Fall back to ibv_reg_mr with nvidia_peermem
 *
 * @param pd  Protection domain.
 * @param gpu_ptr  Device pointer (from cudaMalloc or comm::vmm::alloc).
 * @param bytes  Buffer size.
 * @param access  RDMA access flags.
 * @return Registered MR. Caller must ibv_dereg_mr when done.
 */
inline ibv_mr* register_gpu_buffer(ibv_pd* pd, void* gpu_ptr, size_t bytes,
                                    int access = kDefaultAccess) {
    ibv_mr* mr = nullptr;

    // --- Path 1a: DMA-BUF via cuMemGetHandleForAddressRange (CUDA 12.4+) ---
    // Works with any GPU allocation (cudaMalloc, VMM, etc.)
    {
        int fd = -1;
        CUresult cu_err = cuMemGetHandleForAddressRange(
            &fd, (CUdeviceptr)gpu_ptr, bytes,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        if (cu_err == CUDA_SUCCESS && fd >= 0) {
            mr = ibv_reg_dmabuf_mr(pd, 0, bytes, (uint64_t)gpu_ptr, fd, access);
            close(fd);
            if (mr) {
                return mr;
            }
            fprintf(stderr, "rdma_gpu_mr: ibv_reg_dmabuf_mr (addr_range) failed: %s\n",
                    strerror(errno));
        }
    }

    // --- Path 1b: DMA-BUF via cuMemRetainAllocationHandle (VMM only) ---
    {
        CUmemGenericAllocationHandle alloc_handle;
        CUresult cu_err = cuMemRetainAllocationHandle(&alloc_handle, gpu_ptr);
        if (cu_err == CUDA_SUCCESS) {
            int fd = -1;
            cu_err = cuMemExportToShareableHandle(&fd, alloc_handle,
                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
            if (cu_err == CUDA_SUCCESS && fd >= 0) {
                mr = ibv_reg_dmabuf_mr(pd, 0, bytes, (uint64_t)gpu_ptr, fd, access);
                close(fd);
                if (mr) {
                    return mr;
                }
                fprintf(stderr, "rdma_gpu_mr: ibv_reg_dmabuf_mr (retain) failed: %s\n",
                        strerror(errno));
            }
        }
    }

    // --- Path 2: nvidia_peermem fallback ---
    mr = ibv_reg_mr(pd, gpu_ptr, bytes, access);
    if (!mr) {
        fprintf(stderr, "rdma_gpu_mr: ibv_reg_mr(gpu) failed: %s\n"
                "Is nvidia_peermem loaded? (lsmod | grep nvidia_peermem)\n",
                strerror(errno));
        exit(EXIT_FAILURE);
    }
    return mr;
}

/**
 * Register an existing GPU buffer for RDMA via **DMA-BUF only**.
 *
 * Differs from register_gpu_buffer in two ways:
 *   1. Returns nullptr on failure instead of falling back to ibv_reg_mr/peermem.
 *   2. Reports which DMA-BUF export path succeeded via *path_out (addr_range|retain).
 *
 * Callers that require a verified DMA-BUF registration (e.g. for direct GPU
 * sends of VMM-allocated buffers without a peermem kernel module) must use
 * this helper — a nullptr return is a hard failure, not a hint to retry with
 * ibv_reg_mr.
 */
inline ibv_mr* register_gpu_buffer_dmabuf_only(
    ibv_pd* pd, void* gpu_ptr, size_t bytes,
    int access = kDefaultAccess, const char** path_out = nullptr) {
    if (path_out) *path_out = "UNAVAILABLE";

    // --- Path 1a: DMA-BUF via cuMemGetHandleForAddressRange (CUDA 12.4+) ---
    {
        int fd = -1;
        CUresult cu_err = cuMemGetHandleForAddressRange(
            &fd, (CUdeviceptr)gpu_ptr, bytes,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        if (cu_err == CUDA_SUCCESS && fd >= 0) {
            ibv_mr* mr = ibv_reg_dmabuf_mr(pd, 0, bytes, (uint64_t)gpu_ptr, fd, access);
            close(fd);
            if (mr) {
                if (path_out) *path_out = "addr_range";
                return mr;
            }
            fprintf(stderr,
                    "rdma_gpu_mr: ibv_reg_dmabuf_mr (addr_range) failed: %s\n",
                    strerror(errno));
        } else if (cu_err != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(cu_err, &err_str);
            fprintf(stderr,
                    "rdma_gpu_mr: cuMemGetHandleForAddressRange failed: %s\n",
                    err_str ? err_str : "unknown");
        }
    }

    // --- Path 1b: DMA-BUF via cuMemRetainAllocationHandle (VMM only) ---
    {
        CUmemGenericAllocationHandle alloc_handle;
        CUresult cu_err = cuMemRetainAllocationHandle(&alloc_handle, gpu_ptr);
        if (cu_err == CUDA_SUCCESS) {
            int fd = -1;
            cu_err = cuMemExportToShareableHandle(
                &fd, alloc_handle,
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
            if (cu_err == CUDA_SUCCESS && fd >= 0) {
                ibv_mr* mr = ibv_reg_dmabuf_mr(pd, 0, bytes, (uint64_t)gpu_ptr, fd, access);
                close(fd);
                if (mr) {
                    if (path_out) *path_out = "retain";
                    return mr;
                }
                fprintf(stderr,
                        "rdma_gpu_mr: ibv_reg_dmabuf_mr (retain) failed: %s\n",
                        strerror(errno));
            }
        }
    }

    fprintf(stderr,
            "rdma_gpu_mr: DMA-BUF registration unavailable for ptr=%p bytes=%zu\n",
            gpu_ptr, bytes);
    return nullptr;
}

/**
 * Allocate a GPU buffer via comm::vmm and register for RDMA in one step.
 */
struct GpuRdmaBuffer {
    void*   gpu_ptr;
    size_t  size;
    ibv_mr* mr;
};

inline GpuRdmaBuffer alloc_and_register(ibv_pd* pd, size_t bytes,
                                         int device_id,
                                         int access = kDefaultAccess) {
    GpuRdmaBuffer buf{};

    // Use cudaMalloc for simplicity — works reliably with nvidia_peermem.
    // VMM allocation + ibv_reg_dmabuf_mr is preferred but may fail on some
    // platforms; cudaMalloc + ibv_reg_mr via nvidia_peermem is the safe path.
    int prev_device;
    cudaGetDevice(&prev_device);
    if (prev_device != device_id) cudaSetDevice(device_id);

    MKERNEL_CUDACHECK(cudaMalloc(&buf.gpu_ptr, bytes));
    MKERNEL_CUDACHECK(cudaMemset(buf.gpu_ptr, 0, bytes));
    buf.size = bytes;

    if (prev_device != device_id) cudaSetDevice(prev_device);

    // Register for RDMA (tries dmabuf first, then nvidia_peermem)
    buf.mr = register_gpu_buffer(pd, buf.gpu_ptr, bytes, access);

    return buf;
}

inline void free_buffer(GpuRdmaBuffer& buf) {
    if (buf.mr) rdma::dereg_mr(buf.mr);
    if (buf.gpu_ptr) cudaFree(buf.gpu_ptr);
    buf = GpuRdmaBuffer{};
}

} // namespace gpu_mr
} // namespace internode
