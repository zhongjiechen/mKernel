#!/bin/bash
set -euo pipefail

# Reproducible H200 launcher for mKernel benchmark sweeps.
# Usage:
#   NUM_NODES=<n> bash bench/run_h200_mkernel.sh [kernel|all] [bench|check]

KERNEL=${1:-all}
MODE=${2:-bench}

HERE=$(cd "$(dirname "$0")" && pwd)

export NUM_NODES=${NUM_NODES:-4}
export MKERNEL_H200=1
export MKERNEL_DIST_BACKEND=${MKERNEL_DIST_BACKEND:-gloo}
export PY=${PY:-python3}
export TORCHRUN=${TORCHRUN:-torchrun}
export TIMEOUT=${TIMEOUT:-900}
export NODE1_LAUNCH_SLEEP=${NODE1_LAUNCH_SLEEP:-2}
export CLEANUP_SETTLE_SLEEP=${CLEANUP_SETTLE_SLEEP:-1}
export RESULT_SUFFIX=${RESULT_SUFFIX:-h200_n${NUM_NODES}}

: "${NODE0_IP:?Set NODE0_IP to node 0 data-plane IP}"
export NODE0_IP
for ((i = 1; i < NUM_NODES; i++)); do
    ip_var="NODE${i}_IP"
    ssh_var="NODE${i}_SSH"
    port_var="NODE${i}_SSH_PORT"
    if [[ -z "${!ip_var:-}" ]]; then
        echo "Set ${ip_var} to node ${i}'s data-plane IP" >&2
        exit 1
    fi
    if [[ -z "${!ssh_var:-}" ]]; then
        echo "Set ${ssh_var} to the SSH target for node ${i}" >&2
        exit 1
    fi
    export "$ip_var" "$ssh_var"
    port_value="${!port_var:-}"
    if [[ -z "$port_value" ]]; then
        port_value=2222
    fi
    printf -v "$port_var" "%s" "$port_value"
    export "$port_var"
done

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_IB_ADDR_FAMILY=${NCCL_IB_ADDR_FAMILY:-AF_INET}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-0}
export NCCL_NET_MERGE_LEVEL=${NCCL_NET_MERGE_LEVEL:-LOC}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PORT_BASE=${PORT_BASE:-28000}

if [[ -z "${H200_DISPATCH_GEMM_SHAPES:-}" ]]; then
    H200_DISPATCH_GEMM_SHAPES=8192,16384,32768,65536,131072
fi
if [[ -z "${H200_GEMM_RS_SHAPES:-}" ]]; then
    H200_GEMM_RS_SHAPES=4096,8192,16384,32768,65536
fi
if [[ -z "${H200_AG_GEMM_SHAPES:-}" ]]; then
    H200_AG_GEMM_SHAPES=8192,16384,32768,49152
fi
if [[ -z "${H200_GEMM_AR_SHAPES:-}" ]]; then
    H200_GEMM_AR_SHAPES=8192,12288,16384,20480,22528
fi
if [[ -z "${H200_RING_ATTENTION_SHAPES:-}" ]]; then
    H200_RING_ATTENTION_SHAPES=768,1536,3072,6144,12288
fi

run_kernel() {
    local kernel=$1
    local master_port=$2
    local tcp_port=$3
    local shapes=$4
    local channelize=${MKERNEL_CHANNELIZE_GPU_PEERS:-1}
    local internode_qps=

    if [[ "$kernel" == "dispatch_gemm" ]]; then
        internode_qps=${MKERNEL_INTERNODE_NUM_QPS:-$(((NUM_NODES - 1) * 8))}
    elif [[ "$kernel" == "gemm_ar" ]]; then
        export MKERNEL_MAX_INFLIGHT=${MKERNEL_MAX_INFLIGHT:-64}
    else
        internode_qps=${MKERNEL_INTERNODE_NUM_QPS:-}
    fi

    echo "==== mKernel H200 n=${NUM_NODES} ${kernel} shapes=${shapes:-default} ===="
    if [[ "$MODE" == "bench" ]]; then
        rm -f "$HERE/results/${kernel}_${RESULT_SUFFIX}.json"
    fi

    if [[ -n "$internode_qps" ]]; then
        MKERNEL_INTERNODE_NUM_QPS=$internode_qps \
        MKERNEL_CHANNELIZE_GPU_PEERS=$channelize \
        MPORT=$master_port TCP_PORT=$tcp_port \
        bash "$HERE/run.sh" "$kernel" "$MODE" "$NUM_NODES" "$shapes"
    else
        MKERNEL_CHANNELIZE_GPU_PEERS=$channelize \
        MPORT=$master_port TCP_PORT=$tcp_port \
        bash "$HERE/run.sh" "$kernel" "$MODE" "$NUM_NODES" "$shapes"
    fi
}

case "$KERNEL" in
    all)
        run_kernel dispatch_gemm  $((PORT_BASE + 0))  $((PORT_BASE + 1000)) "$H200_DISPATCH_GEMM_SHAPES"
        run_kernel gemm_rs        $((PORT_BASE + 20)) $((PORT_BASE + 1020)) "$H200_GEMM_RS_SHAPES"
        run_kernel ag_gemm        $((PORT_BASE + 40)) $((PORT_BASE + 1040)) "$H200_AG_GEMM_SHAPES"
        run_kernel gemm_ar        $((PORT_BASE + 60)) $((PORT_BASE + 1060)) "$H200_GEMM_AR_SHAPES"
        run_kernel ring_attention $((PORT_BASE + 80)) $((PORT_BASE + 1080)) "$H200_RING_ATTENTION_SHAPES"
        ;;
    dispatch_gemm)
        run_kernel dispatch_gemm $((PORT_BASE + 0)) $((PORT_BASE + 1000)) "${SHAPES:-$H200_DISPATCH_GEMM_SHAPES}"
        ;;
    gemm_rs)
        run_kernel gemm_rs $((PORT_BASE + 20)) $((PORT_BASE + 1020)) "${SHAPES:-$H200_GEMM_RS_SHAPES}"
        ;;
    ag_gemm)
        run_kernel ag_gemm $((PORT_BASE + 40)) $((PORT_BASE + 1040)) "${SHAPES:-$H200_AG_GEMM_SHAPES}"
        ;;
    gemm_ar)
        run_kernel gemm_ar $((PORT_BASE + 60)) $((PORT_BASE + 1060)) "${SHAPES:-$H200_GEMM_AR_SHAPES}"
        ;;
    ring_attention)
        run_kernel ring_attention $((PORT_BASE + 80)) $((PORT_BASE + 1080)) "${SHAPES:-$H200_RING_ATTENTION_SHAPES}"
        ;;
    *)
        echo "unknown kernel: $KERNEL" >&2
        exit 2
        ;;
esac
