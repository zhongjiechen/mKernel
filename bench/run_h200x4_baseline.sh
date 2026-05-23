#!/bin/bash
set -euo pipefail

# Reproducible 4-node H200x launcher for torch/cuBLAS + NCCL baselines.
# Usage:
#   bash bench/run_h200x4_baseline.sh [kernel|all]
#
# Optional overrides:
#   BASELINE_NET=ib|socket     # default: ib
#   SHAPES=8192,16384          # only for one kernel
#   WARMUP=3 ITERS=10
#   NPROC_PER_NODE=8
#   MASTER_PORT=27200
#   PY=python3
#
# For reproducible socket smoke fallback runs, use:
#   bash bench/run_h200x4_baseline_socket_smoke_fill.sh

KERNEL=${1:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")

PY=${PY:-python3}
MASTER_PORT=${MASTER_PORT:-27200}
BASELINE_NET=${BASELINE_NET:-ib}
TIMEOUT=${TIMEOUT:-3000}
TAG=${TAG:-nccl_h200x4}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
DEFAULT_NCCL_SOCKET_IFNAME=bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7
DEFAULT_NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7

NODE0_IP=${NODE0_IP:?Set NODE0_IP to node 0's data-plane IP}
NODE1_IP=${NODE1_IP:?Set NODE1_IP to node 1's data-plane IP}
NODE2_IP=${NODE2_IP:?Set NODE2_IP to node 2's data-plane IP}
NODE3_IP=${NODE3_IP:?Set NODE3_IP to node 3's data-plane IP}
NODE1_SSH=${NODE1_SSH:?Set NODE1_SSH to the SSH target for node 1}
NODE2_SSH=${NODE2_SSH:?Set NODE2_SSH to the SSH target for node 2}
NODE3_SSH=${NODE3_SSH:?Set NODE3_SSH to the SSH target for node 3}
MASTER_ADDR=${MASTER_ADDR:-$NODE0_IP}
NODE1_SSH_PORT=${NODE1_SSH_PORT:-2222}
NODE2_SSH_PORT=${NODE2_SSH_PORT:-2222}
NODE3_SSH_PORT=${NODE3_SSH_PORT:-2222}

BASE_ENV=(
    "PATH=${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
    "HOME=${HOME:-/root}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    "BASELINE_NET=$BASELINE_NET"
    "MASTER_ADDR=$MASTER_ADDR"
    "MASTER_PORT=$MASTER_PORT"
    "NUM_NODES=4"
    "NODE0_IP=$NODE0_IP"
    "NODE1_IP=$NODE1_IP"
    "NODE2_IP=$NODE2_IP"
    "NODE3_IP=$NODE3_IP"
    "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
    "NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}"
    "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}"
    "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}"
    "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}"
    "BASELINE_RING_ATTN_IMPL=${BASELINE_RING_ATTN_IMPL:-flash_all_gather}"
)
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
        rm -f "$HERE/results/"*_nccl_h200x4.json
    else
        rm -f "$HERE/results/${KERNEL}_${TAG}.json"
    fi
fi

env_prefix="env -i ${BASE_ENV[*]}"

cleanup() {
    pkill -9 -f '[n]ccl_baseline_bench.py|[t]orch.distributed.run|[t]orchrun' 2>/dev/null || true
    for item in "1 $NODE1_SSH $NODE1_SSH_PORT" "2 $NODE2_SSH $NODE2_SSH_PORT" "3 $NODE3_SSH $NODE3_SSH_PORT"; do
        set -- $item
        ssh -p "$3" -o BatchMode=yes \
            "$2" "pkill -9 -f '[n]ccl_baseline_bench.py|[t]orch.distributed.run|[t]orchrun' 2>/dev/null || true" \
            2>/dev/null || true
    done
    sleep 3
}

trap cleanup EXIT
cleanup

declare -a peer_pids=()
for item in "1 $NODE1_SSH $NODE1_SSH_PORT" "2 $NODE2_SSH $NODE2_SSH_PORT" "3 $NODE3_SSH $NODE3_SSH_PORT"; do
    set -- $item
    rank=$1
    host=$2
    port=$3
    timeout "$TIMEOUT" ssh -p "$port" \
        -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$host" \
        "cd '$REPO' && $env_prefix '$PY' -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --nnodes=4 --node_rank=$rank --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT bench/nccl_baseline_bench.py ${EXTRA_ARGS[*]}" \
        > "/tmp/nccl_h200x4_node${rank}.log" 2>&1 &
    peer_pids+=("$!")
done

sleep 2
timeout "$TIMEOUT" bash -c \
    "cd '$REPO' && $env_prefix '$PY' -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --nnodes=4 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT bench/nccl_baseline_bench.py ${EXTRA_ARGS[*]}"
rc0=$?

rc_sum=$rc0
for pid in "${peer_pids[@]}"; do
    wait "$pid" || rc_sum=$((rc_sum + $?))
done

cleanup
exit "$rc_sum"
