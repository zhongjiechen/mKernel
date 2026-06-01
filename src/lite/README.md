# Lite L4 AllGather+GEMM

This directory contains the L4/L40-friendly AllGather+GEMM prototype for
tensor-parallel inference. The measured fused path does **not** call NCCL or
cuBLAS.

## What changed

- Added a full 2-node x 4-GPU benchmark:
  `bench/lite_ag_gemm_rdma_full_bench.py`.
- Added `src/lite/ag_gemm_rdma.cu`, an RDMA + WMMA extension for FP32 inputs
  with TF32 tensor-core compute on L4.
- Added correctness checking with `--check-correctness`.

## TP inference semantics

The tensor-parallel target is column-parallel AG+GEMM:

```text
A_full = all_gather(A_local)
C_i = A_full @ B_i
```

Each rank owns a different local column shard `B_i` and keeps the local output
column shard `C_i`. The benchmark seeds `B_i` with `100 + rank` so correctness
checks catch implementations that incorrectly assume identical `B` on every
rank.

## Current basic fused path

The baseline L4 path is `--compute-backend push-wmma`:

1. Each GPU posts RDMA `TransferCmd`s through the existing mKernel D2H FIFO.
2. A CPU proxy performs RDMA writes of local `A` chunks to the paired remote GPU.
3. The proxy writes arrival flags after each RDMA chunk.
4. Each paired GPU pushes local and received remote `A` shards into every local
   rank's `A_full` staging buffer via CUDA IPC/P2P copies.
5. Each rank waits for `A_full` readiness flags and launches a hand-written WMMA
   GEMM with its own `B_i`.

This is TP-correct and avoids cross-node A fanout, but it is still a basic
implementation: it materializes `A_full` before GEMM rather than overlapping
remote A chunk arrival with GEMM tiles.

## Run example

Use matching cuBLAS/cuBLASLt libraries for the baseline. For conda/pip CUDA
packages, this usually means putting the environment-provided CUDA libraries
before the system CUDA toolkit:

```bash
export CUBLAS_DIR="$CONDA_PREFIX/lib/python*/site-packages/nvidia/cublas/lib"
export CUDART_DIR="$CONDA_PREFIX/lib/python*/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH=$CUBLAS_DIR:$CUDART_DIR:/usr/local/cuda/lib64
```

Launch rank 0 and rank 1 on the two L4 nodes with matching ports:

```bash
NODE0_IP=10.10.55.1 NODE1_IP=10.10.55.2 CUDA_VISIBLE_DEVICES=0,1,2,3 \
"${PYTHON:-python3}" -m torch.distributed.run --nproc_per_node=4 --nnodes=2 \
  --node_rank=0 --master_addr=10.10.55.1 --master_port=29931 \
  bench/lite_ag_gemm_rdma_full_bench.py \
  --mode bench --check-correctness \
  --shapes 512,1024,4096,8192 --chunk-rows 64 \
  --fast-epoch --tcp-port 32231
```

Only the basic FP32/TF32 TP path is kept here.
