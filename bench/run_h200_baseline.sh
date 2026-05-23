#!/bin/bash
set -euo pipefail

# Reproducible H200 launcher for torch/cuBLAS + NCCL baselines.
# Usage:
#   NUM_NODES=<n> bash bench/run_h200_baseline.sh [kernel|all]
#
# Optional overrides:
#   BASELINE_NET=ib|socket     # default: ib
#   SHAPES=8192,16384          # only for one kernel
#   WARMUP=3 ITERS=10
#   NPROC_PER_NODE=8
#   MASTER_PORT=27200
#   PY=python3
#
# For socket smoke fallback runs, set BASELINE_NET=socket.

KERNEL=${1:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")

PY=${PY:-python3}
MASTER_PORT=${MASTER_PORT:-27200}
BASELINE_NET=${BASELINE_NET:-ib}
TIMEOUT=${TIMEOUT:-3000}
NUM_NODES=${NUM_NODES:-4}
TAG=${TAG:-nccl_h200_n${NUM_NODES}}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
DEFAULT_NCCL_SOCKET_IFNAME=bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7
DEFAULT_NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7

: "${NODE0_IP:?Set NODE0_IP to node 0 data-plane IP}"
for ((i=1; i<NUM_NODES; i++)); do
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
    printf -v "$port_var" '%s' "$port_value"
    export "$port_var"
done
MASTER_ADDR=${MASTER_ADDR:-$NODE0_IP}

BASE_ENV=(
    "PATH=${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
    "HOME=${HOME:-/root}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    "BASELINE_NET=$BASELINE_NET"
    "MASTER_ADDR=$MASTER_ADDR"
    "MASTER_PORT=$MASTER_PORT"
    "NUM_NODES=$NUM_NODES"
    "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
    "NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}"
    "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}"
    "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}"
    "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}"
    "BASELINE_RING_ATTN_IMPL=${BASELINE_RING_ATTN_IMPL:-flash_all_gather}"
)
for ((i=0; i<NUM_NODES; i++)); do
    ip_var="NODE${i}_IP"
    BASE_ENV+=("${ip_var}=${!ip_var}")
done
if [[ -n "${NCCL_ALGO:-}" ]]; then
    BASE_ENV+=("NCCL_ALGO=$NCCL_ALGO")
fi
if [[ -n "${NCCL_PROTO:-}" ]]; then
    BASE_ENV+=("NCCL_PROTO=$NCCL_PROTO")
fi
if [[ -n "${NCCL_OOB_NET_ENABLE:-}" ]]; then
    BASE_ENV+=("NCCL_OOB_NET_ENABLE=$NCCL_OOB_NET_ENABLE")
fi
if [[ -n "${NCCL_OOB_NET_IFNAME:-}" ]]; then
    BASE_ENV+=("NCCL_OOB_NET_IFNAME=$NCCL_OOB_NET_IFNAME")
fi

if [[ "$BASELINE_NET" == "socket" ]]; then
    BASE_ENV+=(
        "NCCL_IB_DISABLE=1"
        "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}"
        "NCCL_IB_ADDR_FAMILY="
        "NCCL_IB_HCA="
        "NCCL_IB_GID_INDEX="
    )
else
    ib_addr_family=${MKERNEL_BASELINE_NCCL_IB_ADDR_FAMILY:-AF_INET}
    if [[ "$ib_addr_family" == "unset" ]]; then
        ib_addr_family=
    fi
    BASE_ENV+=(
        "NCCL_IB_DISABLE=0"
        # NCCL's IB bootstrap must use the RoCE-facing bond interfaces, not eth0.
        "NCCL_SOCKET_IFNAME=${MKERNEL_BASELINE_NCCL_SOCKET_IFNAME:-$DEFAULT_NCCL_SOCKET_IFNAME}"
        "NCCL_IB_HCA=${MKERNEL_BASELINE_NCCL_IB_HCA:-$DEFAULT_NCCL_IB_HCA}"
        "NCCL_IB_MERGE_NICS=${MKERNEL_BASELINE_NCCL_IB_MERGE_NICS:-0}"
        "NCCL_NET_MERGE_LEVEL=${MKERNEL_BASELINE_NCCL_NET_MERGE_LEVEL:-LOC}"
    )
    ib_gid_index=${MKERNEL_BASELINE_NCCL_IB_GID_INDEX:-3}
    if [[ "$ib_gid_index" != "unset" ]]; then
        BASE_ENV+=("NCCL_IB_GID_INDEX=$ib_gid_index")
    fi
    if [[ -n "$ib_addr_family" ]]; then
        BASE_ENV+=("NCCL_IB_ADDR_FAMILY=$ib_addr_family")
    fi
    net_gdr_level=${MKERNEL_BASELINE_NCCL_NET_GDR_LEVEL:-${NCCL_NET_GDR_LEVEL:-}}
    if [[ -n "$net_gdr_level" ]]; then
        BASE_ENV+=("NCCL_NET_GDR_LEVEL=$net_gdr_level")
    fi
fi

EXTRA_ARGS=("$KERNEL" "--tag" "$TAG")
if [[ -n "${SHAPES:-}" ]]; then
    if [[ "$KERNEL" == "all" ]]; then
        echo "SHAPES can only be used with one kernel" >&2
        exit 2
    fi
    EXTRA_ARGS+=("--shapes" "$SHAPES")
fi
if [[ -n "${WARMUP:-}" ]]; then
    EXTRA_ARGS+=("--warmup" "$WARMUP")
fi
if [[ -n "${ITERS:-}" ]]; then
    EXTRA_ARGS+=("--iters" "$ITERS")
fi

if [[ "${CLEAN_RESULTS:-1}" == "1" ]]; then
    if [[ "$KERNEL" == "all" ]]; then
        rm -f "$HERE/results/"*_nccl_h200_n"${NUM_NODES}".json
    else
        rm -f "$HERE/results/${KERNEL}_${TAG}.json"
    fi
fi

env_prefix="env -i ${BASE_ENV[*]}"

cleanup() {
    pkill -9 -f '[n]ccl_baseline_bench.py|[t]orch.distributed.run|[t]orchrun' 2>/dev/null || true
    for ((i=1; i<NUM_NODES; i++)); do
        ssh_var="NODE${i}_SSH"
        port_var="NODE${i}_SSH_PORT"
        ssh -p "${!port_var}" -o BatchMode=yes \
            "${!ssh_var}" "pkill -9 -f '[n]ccl_baseline_bench.py|[t]orch.distributed.run|[t]orchrun' 2>/dev/null || true" \
            2>/dev/null || true
    done
    sleep 3
}

trap cleanup EXIT
cleanup

declare -a peer_pids=()
for ((rank=1; rank<NUM_NODES; rank++)); do
    ssh_var="NODE${rank}_SSH"
    port_var="NODE${rank}_SSH_PORT"
    host=${!ssh_var}
    port=${!port_var}
    timeout "$TIMEOUT" ssh -p "$port" \
        -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$host" \
        "cd '$REPO' && $env_prefix '$PY' -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --nnodes=$NUM_NODES --node_rank=$rank --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT bench/nccl_baseline_bench.py ${EXTRA_ARGS[*]}" \
        > "/tmp/nccl_h200_n${NUM_NODES}_node${rank}.log" 2>&1 &
    peer_pids+=("$!")
done

sleep 2
timeout "$TIMEOUT" bash -c \
    "cd '$REPO' && $env_prefix '$PY' -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --nnodes=$NUM_NODES --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT bench/nccl_baseline_bench.py ${EXTRA_ARGS[*]}"
rc0=$?

rc_sum=$rc0
for pid in "${peer_pids[@]}"; do
    wait "$pid" || rc_sum=$((rc_sum + $?))
done

cleanup
exit "$rc_sum"
