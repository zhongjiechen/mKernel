#!/bin/bash
set -euo pipefail

# Reproducible 4-node H200x launcher for mKernel benchmark sweeps.
# Usage:
#   bash bench/run_h200x4_mkernel.sh [kernel|all] [bench|check]
#
# Optional overrides:
#   PY — must match the PyTorch used to build mKernel .so.
#   TORCHRUN — same environment as PY.
#   TIMEOUT=900
#   WARMUP=3 ITERS=10                              # release timing
#   WARMUP=0 ITERS=1 MKERNEL_BENCH_LEGACY_SYNC=1   # smoke/debug mode only
#   AG_GEMM_INTERNODE_COLLECTIVE=ring|direct|auto  # ring is default for N>2
#     (unset behaves like auto). bench/run.sh then sets NCCL iface defaults.
#
# Ops note (confirmed policy): per-GPU utilization >99% for ≥15s with no
# progress ⇒ treat the job as stuck (e.g. ring wait / proxy); kill torchrun
# and triage — do not assume it will recover.
#
# Launch policy: set NODE0_IP / NODE{i}_SSH to four **GPU-idle** nodes
# (no SGLang, training, or other jobs). Run this script from node 0.

KERNEL=${1:-all}
MODE=${2:-bench}

HERE=$(cd "$(dirname "$0")" && pwd)

export MKERNEL_TOPOLOGY=h200x4
export MKERNEL_DIST_BACKEND=${MKERNEL_DIST_BACKEND:-gloo}
export PY=${PY:-python3}
export TORCHRUN=${TORCHRUN:-torchrun}
export TIMEOUT=${TIMEOUT:-900}
export NODE1_LAUNCH_SLEEP=${NODE1_LAUNCH_SLEEP:-2}
export CLEANUP_SETTLE_SLEEP=${CLEANUP_SETTLE_SLEEP:-1}

export NODE0_IP=${NODE0_IP:?Set NODE0_IP to node 0's data-plane IP}
export NODE1_IP=${NODE1_IP:?Set NODE1_IP to node 1's data-plane IP}
export NODE2_IP=${NODE2_IP:?Set NODE2_IP to node 2's data-plane IP}
export NODE3_IP=${NODE3_IP:?Set NODE3_IP to node 3's data-plane IP}
export NODE1_SSH=${NODE1_SSH:?Set NODE1_SSH to the SSH target for node 1}
export NODE2_SSH=${NODE2_SSH:?Set NODE2_SSH to the SSH target for node 2}
export NODE3_SSH=${NODE3_SSH:?Set NODE3_SSH to the SSH target for node 3}
export NODE1_SSH_PORT=${NODE1_SSH_PORT:-2222}
export NODE2_SSH_PORT=${NODE2_SSH_PORT:-2222}
export NODE3_SSH_PORT=${NODE3_SSH_PORT:-2222}

# bench/run.sh (h200x4) overrides NCCL_SOCKET_IFNAME to bond0,…,bond7 for RoCE;
# GLOO stays on eth0. Values here are only fallbacks before run.sh runs.
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_ADDR_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_IB_GID_INDEX=3
export NCCL_IB_MERGE_NICS=0
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Keep torchrun/RDMA listener ports below Linux's default ephemeral range
# (32768-60999) so per-GPU RDMA binds do not race outgoing sockets.
PORT_BASE=${PORT_BASE:-28000}

H200X4_DISPATCH_GEMM_SHAPES=${H200X4_DISPATCH_GEMM_SHAPES:-8192,16384,32768,65536,131072}
H200X4_GEMM_RS_SHAPES=${H200X4_GEMM_RS_SHAPES:-4096,8192,16384,32768,65536}
H200X4_AG_GEMM_SHAPES=${H200X4_AG_GEMM_SHAPES:-8192,16384,32768,49152}
H200X4_GEMM_AR_SHAPES=${H200X4_GEMM_AR_SHAPES:-8192,12288,16384,20480,22528}
H200X4_RING_ATTENTION_SHAPES=${H200X4_RING_ATTENTION_SHAPES:-768,1536,3072,6144,12288}

run_kernel() {
    local kernel=$1
    local master_port=$2
    local tcp_port=$3
    local shapes=$4
    # Empty internode_qps means do not pass MKERNEL_INTERNODE_NUM_QPS into
    # run.sh so bench/run.sh can apply h200x4 default (16). Previously we always
    # forced 4, which blocked that default and could starve the proxy.
    local internode_qps=
    local channelize=${MKERNEL_CHANNELIZE_GPU_PEERS:-1}
    if [[ "$kernel" == "dispatch_gemm" ]]; then
        # Dispatch routes by peer_slot * 8 + local_gpu, so 4 nodes need
        # 3 peers * 8 GPUs = 24 channelized QPs per GPU.
        internode_qps=${MKERNEL_INTERNODE_NUM_QPS:-24}
        channelize=${MKERNEL_CHANNELIZE_GPU_PEERS:-1}
    elif [[ "$kernel" == "gemm_ar" ]]; then
        # Leave split selection to gemm_ar_bench.py so H200x4 can use
        # shape-specific defaults; explicit user overrides still propagate.
        export MKERNEL_MAX_INFLIGHT=${MKERNEL_MAX_INFLIGHT:-64}
    else
        # ag_gemm, gemm_rs, ring_attention: honor user MKERNEL_INTERNODE_NUM_QPS
        # if set; otherwise omit so run.sh sets 16 for h200x3/h200x4.
        internode_qps=${MKERNEL_INTERNODE_NUM_QPS:-}
    fi

    echo "==== mKernel h200x4 ${kernel} shapes=${shapes:-default} ===="
    if [[ "$MODE" == "bench" ]]; then
        rm -f "$HERE/results/${kernel}_h200x4.json"
    fi
    if [[ -n "$internode_qps" ]]; then
        MKERNEL_INTERNODE_NUM_QPS=$internode_qps \
        MKERNEL_CHANNELIZE_GPU_PEERS=$channelize \
        MPORT=$master_port TCP_PORT=$tcp_port bash "$HERE/run.sh" "$kernel" "$MODE" 4 "$shapes"
    else
        MKERNEL_CHANNELIZE_GPU_PEERS=$channelize \
        MPORT=$master_port TCP_PORT=$tcp_port bash "$HERE/run.sh" "$kernel" "$MODE" 4 "$shapes"
    fi
}

case "$KERNEL" in
    all)
        run_kernel dispatch_gemm  $((PORT_BASE + 0))  $((PORT_BASE + 1000)) "$H200X4_DISPATCH_GEMM_SHAPES"
        run_kernel gemm_rs        $((PORT_BASE + 20)) $((PORT_BASE + 1020)) "$H200X4_GEMM_RS_SHAPES"
        run_kernel ag_gemm        $((PORT_BASE + 40)) $((PORT_BASE + 1040)) "$H200X4_AG_GEMM_SHAPES"
        run_kernel gemm_ar        $((PORT_BASE + 60)) $((PORT_BASE + 1060)) "$H200X4_GEMM_AR_SHAPES"
        run_kernel ring_attention $((PORT_BASE + 80)) $((PORT_BASE + 1080)) "$H200X4_RING_ATTENTION_SHAPES"
        ;;
    dispatch_gemm)
        run_kernel dispatch_gemm $((PORT_BASE + 0)) $((PORT_BASE + 1000)) "${SHAPES:-$H200X4_DISPATCH_GEMM_SHAPES}"
        ;;
    gemm_rs)
        run_kernel gemm_rs $((PORT_BASE + 20)) $((PORT_BASE + 1020)) "${SHAPES:-$H200X4_GEMM_RS_SHAPES}"
        ;;
    ag_gemm)
        run_kernel ag_gemm $((PORT_BASE + 40)) $((PORT_BASE + 1040)) "${SHAPES:-$H200X4_AG_GEMM_SHAPES}"
        ;;
    gemm_ar)
        run_kernel gemm_ar $((PORT_BASE + 60)) $((PORT_BASE + 1060)) "${SHAPES:-$H200X4_GEMM_AR_SHAPES}"
        ;;
    ring_attention)
        run_kernel ring_attention $((PORT_BASE + 80)) $((PORT_BASE + 1080)) "${SHAPES:-$H200X4_RING_ATTENTION_SHAPES}"
        ;;
    *)
        echo "unknown kernel: $KERNEL" >&2
        exit 2
        ;;
esac
