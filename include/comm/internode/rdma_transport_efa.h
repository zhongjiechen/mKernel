/**
 * @file rdma_transport_efa.h
 * @brief EFA (Elastic Fabric Adapter) extensions for inter-node RDMA communication.
 *
 * Includes rdma_transport.h for shared primitives (PD, CQ, MR, TCP bootstrap)
 * and adds EFA-specific functions: SRD QP creation, address handles, device discovery.
 */
#pragma once

#include "rdma_transport.h"

#include <cuda_runtime.h>
#include <infiniband/efadv.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <vector>

namespace internode {
namespace rdma {

// ---------------------------------------------------------------------------
// QKEY for SRD (must match on both sides)
// ---------------------------------------------------------------------------

static constexpr uint32_t QKEY = 0x11111111;

inline int parse_positive_env_int(const char* name, int default_value) {
    const char* env = std::getenv(name);
    if (!env || !env[0]) return default_value;
    char* end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env || (end && *end != '\0') || value <= 0 || value > INT32_MAX) {
        fprintf(stderr, "rdma_efa: ignoring invalid %s=%s\n", name, env);
        return default_value;
    }
    return static_cast<int>(value);
}

inline int query_efa_max_sq_wr(ibv_context* ctx) {
    efadv_device_attr attr{};
    int ret = efadv_query_device(ctx, &attr, sizeof(attr));
    if (ret != 0) {
        fprintf(stderr, "rdma_efa: efadv_query_device failed (ret=%d)\n", ret);
        return 0;
    }
    return static_cast<int>(attr.max_sq_wr);
}

inline int read_numa_node(const std::string& path) {
    std::ifstream f(path);
    if (!f) return -1;
    int numa = -1;
    f >> numa;
    return numa;
}

inline std::string normalize_pci_bdf(const char* bdf) {
    unsigned dom = 0, bus = 0, dev = 0, func = 0;
    if (std::sscanf(bdf, "%x:%x:%x.%x", &dom, &bus, &dev, &func) == 4) {
        char out[32];
        std::snprintf(out, sizeof(out), "%04x:%02x:%02x.%x", dom, bus, dev, func);
        return std::string(out);
    }
    return std::string(bdf ? bdf : "");
}

inline std::vector<std::string> parse_efa_domain_map_env() {
    std::vector<std::string> names;
    const char* env = std::getenv("MKERNEL_EFA_DOMAIN_MAP");
    if (!env || !env[0]) return names;

    std::stringstream ss(env);
    std::string item;
    while (std::getline(ss, item, ',')) {
        while (!item.empty() && item.front() == ' ') item.erase(item.begin());
        while (!item.empty() && item.back() == ' ') item.pop_back();
        if (item.size() >= 4 && item.compare(item.size() - 4, 4, "-rdm") == 0) {
            item.erase(item.size() - 4);
        }
        if (!item.empty()) names.push_back(item);
    }
    return names;
}

// ---------------------------------------------------------------------------
// PCIe Topology Helpers
// ---------------------------------------------------------------------------

/**
 * Get the PCIe root port identifier for a sysfs device path.
 * Returns a string like "pci0000:44/0000:44:00.0" that uniquely identifies
 * the root port — devices under the same root share a PCIe switch and avoid
 * cross-switch traffic.
 */
inline std::string get_pcie_root_id(const std::string& sysfs_device_path) {
    char resolved[PATH_MAX];
    if (!realpath(sysfs_device_path.c_str(), resolved)) return "";
    std::string path(resolved);
    const std::string prefix = "/sys/devices/";
    if (path.compare(0, prefix.size(), prefix) != 0) return "";
    std::string rel = path.substr(prefix.size());
    // Root ID = first two path components: pciDOMAIN:BUS / DOMAIN:BUS:DEV.FUNC
    size_t slash1 = rel.find('/');
    if (slash1 == std::string::npos) return rel;
    size_t slash2 = rel.find('/', slash1 + 1);
    if (slash2 == std::string::npos) return rel;
    return rel.substr(0, slash2);
}

// ---------------------------------------------------------------------------
// EFA Device Discovery
// ---------------------------------------------------------------------------

struct EfaDeviceEntry {
    std::string name;
    ibv_device* dev;
    int numa_node;
    std::string pcie_root;
};

/**
 * Enumerate all EFA devices with their topology info.
 * Returns a sorted vector of (name, ibv_device*, numa_node, pcie_root).
 * Caller must NOT call ibv_free_device_list — the returned ibv_device pointers
 * remain valid only while the ibv_get_device_list result is live.
 */
inline std::vector<EfaDeviceEntry> enumerate_efa_devices(
        ibv_device** dev_list, int num_devices) {
    std::vector<EfaDeviceEntry> efa_devs;
    for (int i = 0; i < num_devices; i++) {
        const char* name = ibv_get_device_name(dev_list[i]);
        if (strncmp(name, "rdmap", 5) == 0) {
            EfaDeviceEntry e;
            e.name = name;
            e.dev  = dev_list[i];
            e.numa_node = read_numa_node(
                "/sys/class/infiniband/" + e.name + "/device/numa_node");
            e.pcie_root = get_pcie_root_id(
                "/sys/class/infiniband/" + e.name + "/device");
            efa_devs.push_back(std::move(e));
        }
    }
    std::sort(efa_devs.begin(), efa_devs.end(),
              [](const EfaDeviceEntry& a, const EfaDeviceEntry& b) {
                  return a.name < b.name;
              });
    return efa_devs;
}

/**
 * Get GPU's PCIe root ID and NUMA node.
 */
inline void get_gpu_topology(int local_rank, std::string& gpu_root, int& gpu_numa) {
    gpu_root.clear();
    gpu_numa = -1;
    char gpu_pci[32] = {};
    if (cudaDeviceGetPCIBusId(gpu_pci, sizeof(gpu_pci), local_rank) == cudaSuccess) {
        const std::string norm_bdf = normalize_pci_bdf(gpu_pci);
        gpu_root = get_pcie_root_id("/sys/bus/pci/devices/" + norm_bdf);
        gpu_numa = read_numa_node("/sys/bus/pci/devices/" + norm_bdf + "/numa_node");
    }
}

/**
 * Select NICs matching the GPU's topology, using progressively looser criteria:
 *   1. Same PCIe root port (tightest — GPU and NIC share a PCIe switch)
 *   2. Same NUMA node (fallback — correct memory domain, may cross PCIe switch)
 *   3. All available NICs (last resort)
 */
inline std::vector<EfaDeviceEntry> select_topology_local_nics(
        const std::vector<EfaDeviceEntry>& all_nics,
        const std::string& gpu_root, int gpu_numa) {
    // Tier 1: same PCIe root port
    if (!gpu_root.empty()) {
        std::vector<EfaDeviceEntry> matches;
        for (const auto& nic : all_nics) {
            if (nic.pcie_root == gpu_root) matches.push_back(nic);
        }
        if (!matches.empty()) return matches;
    }
    // Tier 2: same NUMA node
    if (gpu_numa >= 0) {
        std::vector<EfaDeviceEntry> matches;
        for (const auto& nic : all_nics) {
            if (nic.numa_node < 0 || nic.numa_node == gpu_numa) matches.push_back(nic);
        }
        if (!matches.empty()) return matches;
    }
    // Tier 3: all devices
    return all_nics;
}

/**
 * Count how many GPUs share the same PCIe root as local_rank's GPU.
 * Used to interleave NIC assignments across GPUs on the same root.
 */
inline int count_gpus_on_same_root(int local_rank, const std::string& gpu_root, int total_gpus) {
    if (gpu_root.empty()) return 1;
    int count = 0;
    int rank_within_root = 0;
    for (int r = 0; r < total_gpus; r++) {
        std::string root;
        int numa;
        get_gpu_topology(r, root, numa);
        if (root == gpu_root) {
            if (r < local_rank) rank_within_root++;
            count++;
        }
    }
    return count > 0 ? count : 1;
}

/**
 * Find the index-within-root for local_rank: how many GPUs with smaller rank
 * share the same PCIe root. Used to interleave NIC assignments.
 */
inline int rank_within_pcie_root(int local_rank, const std::string& gpu_root, int total_gpus) {
    if (gpu_root.empty()) return local_rank;
    int idx = 0;
    for (int r = 0; r < local_rank; r++) {
        std::string root;
        int numa;
        get_gpu_topology(r, root, numa);
        if (root == gpu_root) idx++;
    }
    return idx;
}

/**
 * Find the best EFA device for a given GPU (single-NIC, backwards compatible).
 * Uses PCIe-root-port matching for tightest locality, falling back to NUMA.
 */
inline ibv_context* open_efa_device(int local_rank = 0) {
    int num_devices = 0;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);
    RDMA_CHECK(dev_list && num_devices > 0, "no IB/EFA devices found");

    auto efa_devs = enumerate_efa_devices(dev_list, num_devices);

    if (efa_devs.empty()) {
        fprintf(stderr, "rdma_efa: no EFA devices found, trying first available\n");
        ibv_context* ctx = ibv_open_device(dev_list[0]);
        ibv_free_device_list(dev_list);
        RDMA_CHECK(ctx != nullptr, "ibv_open_device failed");
        return ctx;
    }

    std::string selected_name;
    const std::vector<std::string> env_names = parse_efa_domain_map_env();
    if (!env_names.empty()) {
        selected_name = env_names[local_rank % (int)env_names.size()];
    } else {
        std::string gpu_root;
        int gpu_numa = -1;
        get_gpu_topology(local_rank, gpu_root, gpu_numa);

        auto matches = select_topology_local_nics(efa_devs, gpu_root, gpu_numa);
        // Interleave across GPUs sharing the same PCIe root so each GPU
        // picks a distinct NIC.
        int total_gpus = 8;
        {
            char* vis = std::getenv("CUDA_VISIBLE_DEVICES");
            int dev_count = 0;
            if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count > 0)
                total_gpus = dev_count;
        }
        int gpus_on_root = count_gpus_on_same_root(local_rank, gpu_root, total_gpus);
        int rank_in_root = rank_within_pcie_root(local_rank, gpu_root, total_gpus);
        // With N NICs and G GPUs on the same root, GPU j gets NIC at index j
        // (stride 1). This gives each GPU a unique NIC from the root-local set.
        int nic_idx = rank_in_root % (int)matches.size();
        selected_name = matches[nic_idx].name;
    }

    int idx = -1;
    for (int i = 0; i < (int)efa_devs.size(); ++i) {
        if (efa_devs[i].name == selected_name) {
            idx = i;
            break;
        }
    }
    if (idx < 0) idx = local_rank % (int)efa_devs.size();
    fprintf(stderr, "rdma_efa: local_rank=%d using EFA device %s (%d/%d available, root=%s)\n",
            local_rank, efa_devs[idx].name.c_str(), idx + 1, (int)efa_devs.size(),
            efa_devs[idx].pcie_root.c_str());

    ibv_context* ctx = ibv_open_device(efa_devs[idx].dev);
    ibv_free_device_list(dev_list);
    RDMA_CHECK(ctx != nullptr, "ibv_open_device failed");
    return ctx;
}

// ---------------------------------------------------------------------------
// Multi-NIC Device Discovery (for multi-rail)
// ---------------------------------------------------------------------------

struct EfaOpenedDevice {
    ibv_context* ctx;
    std::string name;
};

/**
 * Open multiple EFA devices for a GPU, using PCIe-root-port locality.
 *
 * Returns up to `count` opened device contexts on the same PCIe root as the
 * GPU identified by `local_rank`. NICs are interleaved across GPUs sharing
 * the same root so that no two GPUs pick the same NIC.
 *
 * Example on p5 with 4 NICs and 2 GPUs per root, count=2:
 *   GPU 0 (rank_in_root=0) → NIC indices 0,2
 *   GPU 1 (rank_in_root=1) → NIC indices 1,3
 * All 4 NICs are used, no overlap.
 *
 * @param local_rank  GPU device index (from CUDA_VISIBLE_DEVICES ordering).
 * @param count       Desired number of NICs. Clamped to available root-local NICs.
 * @return Vector of opened device contexts. Caller owns the ibv_contexts.
 */
inline std::vector<EfaOpenedDevice> open_efa_devices(int local_rank, int count) {
    int num_devices = 0;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);
    RDMA_CHECK(dev_list && num_devices > 0, "no IB/EFA devices found");

    auto efa_devs = enumerate_efa_devices(dev_list, num_devices);
    if (efa_devs.empty()) {
        // Fallback: open the first available device
        ibv_context* ctx = ibv_open_device(dev_list[0]);
        ibv_free_device_list(dev_list);
        RDMA_CHECK(ctx != nullptr, "ibv_open_device failed");
        return {{ctx, ibv_get_device_name(dev_list[0])}};
    }

    std::string gpu_root;
    int gpu_numa = -1;
    get_gpu_topology(local_rank, gpu_root, gpu_numa);

    auto matches = select_topology_local_nics(efa_devs, gpu_root, gpu_numa);

    int total_gpus = 8;
    {
        int dev_count = 0;
        if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count > 0)
            total_gpus = dev_count;
    }
    int gpus_on_root = count_gpus_on_same_root(local_rank, gpu_root, total_gpus);
    int rank_in_root = rank_within_pcie_root(local_rank, gpu_root, total_gpus);

    // How many NICs this GPU can use without overlapping with other GPUs on
    // the same root: nics_per_gpu = floor(total_root_nics / gpus_on_root).
    int nics_per_gpu = (int)matches.size() / std::max(1, gpus_on_root);
    if (nics_per_gpu < 1) nics_per_gpu = 1;
    int actual_count = std::min(count, nics_per_gpu);
    actual_count = std::min(actual_count, (int)matches.size());

    // Assign NICs: GPU j gets NIC indices [j, j + gpus_on_root, j + 2*gpus_on_root, ...]
    // This interleaves across GPUs so adjacent GPUs pick non-overlapping NICs.
    std::vector<EfaOpenedDevice> result;
    for (int r = 0; r < actual_count; r++) {
        int nic_idx = (rank_in_root + r * gpus_on_root) % (int)matches.size();
        const auto& selected = matches[nic_idx];

        // Find in the full efa_devs list to get the ibv_device*
        ibv_device* target_dev = nullptr;
        for (const auto& ed : efa_devs) {
            if (ed.name == selected.name) {
                target_dev = ed.dev;
                break;
            }
        }
        if (!target_dev) continue;

        ibv_context* ctx = ibv_open_device(target_dev);
        RDMA_CHECK(ctx != nullptr, "ibv_open_device failed");
        fprintf(stderr, "rdma_efa: local_rank=%d rail=%d using EFA device %s (root=%s)\n",
                local_rank, r, selected.name.c_str(), selected.pcie_root.c_str());
        result.push_back({ctx, selected.name});
    }

    ibv_free_device_list(dev_list);
    RDMA_CHECK(!result.empty(), "open_efa_devices: no devices opened");
    return result;
}

// ---------------------------------------------------------------------------
// SRD Queue Pair (EFA-specific)
// ---------------------------------------------------------------------------

inline ibv_qp* create_srd_qp(ibv_pd* pd, ibv_cq* send_cq, ibv_context* ctx,
                             int sq_depth = 512, int* actual_sq_depth_out = nullptr) {
    ibv_qp_init_attr_ex qp_attr_ex{};
    efadv_qp_init_attr efa_attr{};
    const int env_sq_depth = parse_positive_env_int("MKERNEL_EFA_SQ_DEPTH", sq_depth);
    const int provider_max_sq_wr = query_efa_max_sq_wr(ctx);
    int requested_sq_depth = env_sq_depth;
    if (provider_max_sq_wr > 0) {
        requested_sq_depth = std::min(requested_sq_depth, provider_max_sq_wr);
    }

    qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    qp_attr_ex.send_ops_flags =
        IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;
    qp_attr_ex.cap.max_send_wr = requested_sq_depth;
    qp_attr_ex.cap.max_recv_wr = 1;
    qp_attr_ex.cap.max_send_sge = 1;
    qp_attr_ex.cap.max_recv_sge = 1;
    // We need inline data >= 8 bytes so post_remote_flag_batch can embed the
    // 4-byte packed (tile_id, run_tiles) flag (and the 4-byte tail publish)
    // directly in the WQE. Without inline, every flag WR sources its 4 bytes
    // from a shared host_ptr staging slot, racing against the proxy's next
    // batch overwriting that slot before the NIC DMAs it.
    qp_attr_ex.cap.max_inline_data = 32;
    qp_attr_ex.pd = pd;
    qp_attr_ex.qp_context = ctx;
    qp_attr_ex.sq_sig_all = 0;
    qp_attr_ex.send_cq = send_cq;
    qp_attr_ex.recv_cq = send_cq;
    qp_attr_ex.qp_type = IBV_QPT_DRIVER;

    efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
    efa_attr.flags = EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

    ibv_qp* qp = efadv_create_qp_ex(ctx, &qp_attr_ex, &efa_attr,
                                      sizeof(efadv_qp_init_attr));
    RDMA_CHECK(qp != nullptr, "efadv_create_qp_ex(SRD) failed");
    if (actual_sq_depth_out) {
        *actual_sq_depth_out = requested_sq_depth;
    }
    fprintf(stderr,
            "rdma_efa: SRD QP created requested sq=%d (default=%d env=%d provider_max=%d)\n",
            requested_sq_depth, sq_depth, env_sq_depth, provider_max_sq_wr);
    return qp;
}

inline void modify_srd_qp_init(ibv_qp* qp) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qkey = QKEY;
    int ret = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
    RDMA_CHECK(ret == 0, "modify_srd_qp_init failed");
}

inline void modify_srd_qp_rtr(ibv_qp* qp) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    int ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
    RDMA_CHECK(ret == 0, "modify_srd_qp_rtr failed");
}

inline void modify_srd_qp_rts(ibv_qp* qp) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    int ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN);
    RDMA_CHECK(ret == 0, "modify_srd_qp_rts failed");
}

// ---------------------------------------------------------------------------
// Address Handle (for SRD sends)
// ---------------------------------------------------------------------------

inline ibv_ah* create_ah(ibv_pd* pd, const uint8_t* remote_gid) {
    ibv_ah_attr ah_attr{};
    ah_attr.is_global = 1;
    ah_attr.port_num = 1;
    ah_attr.grh.sgid_index = 0;
    memcpy(&ah_attr.grh.dgid, remote_gid, 16);
    ah_attr.grh.flow_label = 0;
    ah_attr.grh.hop_limit = 255;
    ah_attr.grh.traffic_class = 0;

    ibv_ah* ah = ibv_create_ah(pd, &ah_attr);
    RDMA_CHECK(ah != nullptr, "ibv_create_ah failed");
    return ah;
}

inline void destroy_ah(ibv_ah* ah) { if (ah) ibv_destroy_ah(ah); }

} // namespace rdma
} // namespace internode
