"""Lite kernels for L4/L40-class GPUs."""

from .cuda_loader import load_ag_gemm_rdma_extension

__all__ = ["load_ag_gemm_rdma_extension"]
