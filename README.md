# mKernel — multi-GPU, multi-node fused kernels

Five fused **mKernel**s: AG + GEMM, GEMM + AR, Dispatch + GEMM, Ring Attention, GEMM + RS.

## Highlights

- **mKernel — one persistent kernel per collective+compute pair.** Each mKernel is a single CUDA launch that runs end-to-end across all 16 GPUs; no per-stage relaunches, no host-side orchestration in the inner loop.
- **Multi-GPU + multi-node, in one kernel.** Intra-node NVLink (8 GPUs) and inter-node EFA (2 nodes) live inside the same kernel — the GEMM, the intra-collective, and the inter-node transfer are interleaved at tile granularity, not stitched at the host.
- **Fine-grained intra-kernel overlapping.** Compute and communication overlap *within* a single kernel at tile granularity — producer CTAs release tiles via on-chip flags the moment they're ready, and consumer CTAs (intra-comm / inter-send / inter-reduce) pick them up immediately, so GEMM math, NVLink traffic, and EFA traffic all run concurrently instead of in serialized phases.
- **Persistent kernel with SM specialization.** All 132 SMs on each H200 are claimed at launch and stay resident; CTAs self-assign roles (compute / intra-comm / inter-send / inter-reduce) by `blockIdx.x`. Producers and consumers communicate through on-chip flags using PTX `ld.acquire` / `st.release`, so the GPU schedules itself instead of round-tripping to the host.
- **GPU-driven networking, built from scratch.** Using libfabric/libibverbs proxy (`include/comm/internode/`). The GPU itself posts sends and consumes arrivals. 

## Quick start

```sh
make all                              # build all 5 .so's (both nodes need to do this)
bash bench/run_2node.sh all bench     # 2-node, all kernels, default shapes
make plots                            # regenerate the figures below
```

## Comparison results

| Kernel | Plot |
|---|---|
| AllGather + GEMM | ![ag_gemm](plots/ag_gemm_efa.png) |
| GEMM + AllReduce | ![gemm_ar](plots/gemm_ar_efa.png) |
| MoE Dispatch + GEMM | ![dispatch_gemm](plots/dispatch_gemm_efa.png) |
| Ring Attention | ![ring_attention](plots/ring_attention_efa.png) |
| GEMM + ReduceScatter | ![gemm_rs](plots/gemm_rs_efa.png) |

