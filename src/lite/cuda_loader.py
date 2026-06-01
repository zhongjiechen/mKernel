"""JIT loader for the lite no-NCCL AG+GEMM RDMA extension."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _default_arch_list() -> str:
    caps = set()
    for idx in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(idx)
        caps.add(f"{major}.{minor}")
    return ";".join(sorted(caps)) if caps else ""


def load_ag_gemm_rdma_extension(*, verbose: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to load the lite AG+GEMM RDMA extension")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        arch_list = _default_arch_list()
        if arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    root = Path(__file__).resolve().parents[2]
    src = Path(__file__).with_name("ag_gemm_rdma.cu")
    return load(
        name="mkernel_lite_ag_gemm_rdma",
        sources=[str(src)],
        extra_include_paths=[str(root / "include")],
        extra_cflags=["-O3", "-std=c++20", "-DINTERNODE_BACKEND_IBVERBS"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "--use_fast_math",
            "-DINTERNODE_BACKEND_IBVERBS",
        ],
        extra_ldflags=["-libverbs", "-lcuda"],
        verbose=verbose,
    )
