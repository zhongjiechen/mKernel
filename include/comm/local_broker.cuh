/**
 * @file local_broker.cuh
 * @brief Utility for multiprocess data exchange and synchronization.
 * 
 * This file provides the LocalBroker class, which enables efficient inter-process
 * communication and synchronization using POSIX shared memory, semaphores, and sockets.
 * The broker is designed to work in multi-GPU environments where processes need to
 * exchange data and synchronize execution across different local ranks.
 * 
 * @note This implementation relies on POSIX IPC mechanisms and is intended for
 *       Unix-like systems. All processes must be running on the same node.
 */

#pragma once

#include <cstdlib>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <semaphore.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #error "LocalBroker is not supported on Windows"
#endif

namespace comm {

namespace detail {
namespace local_broker {

static constexpr int MAX_LOCAL_WORLD_SIZE = 72;
static constexpr int VAULT_SIZE_PER_RANK = 64; // sizeof(cudaIpcMemHandle_t)

struct LocalBrokerVault {
    static constexpr int INIT_CODE = 0x44495354; // "DIST"
    int init;
    int barrier;
    int sense;
    uint8_t data[MAX_LOCAL_WORLD_SIZE * VAULT_SIZE_PER_RANK];
};

static constexpr int SHM_SIZE = (sizeof(LocalBrokerVault) + 4095) / 4096 * 4096;

__host__ inline static void init_sync(
    int local_rank,
    volatile LocalBrokerVault *vault
) {
    if (local_rank == 0) {
        // initialize barrier resources
        vault->barrier = 0;
        vault->sense = 0;
        __sync_synchronize(); // make previous writes visible
        vault->init = LocalBrokerVault::INIT_CODE;
    } else {
        while (vault->init != LocalBrokerVault::INIT_CODE) usleep(1);
        __sync_synchronize(); // see leader's previous writes
    }
}

__host__ inline static void sync(
    int local_world_size,
    volatile LocalBrokerVault *vault
) {
    if (vault->init != LocalBrokerVault::INIT_CODE)
        throw std::runtime_error("LocalBroker: LocalBrokerVault not initialized");

    // Phase 1
    int arrived = __sync_add_and_fetch(&vault->barrier, 1);
    if (arrived == local_world_size) vault->sense = 1;
    while (!vault->sense) usleep(1);

    // Make previous writes visible
    __sync_synchronize();

    // Phase 2
    arrived = __sync_add_and_fetch(&vault->barrier, -1);
    if (arrived == 0) vault->sense = 0;
    while (vault->sense) usleep(1);
}

__host__ inline void *create_shm(const char *key, size_t size) {
    int shm_fd;
    shm_fd = shm_open(key, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (shm_fd < 0 && errno == EEXIST) {
        // Previous runs can leave stale broker segments behind after abrupt exits.
        // Rank 0 owns creation, so proactively reclaim once and retry.
        shm_unlink(key);
        shm_fd = shm_open(key, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    }

    if (shm_fd < 0) {
        if (errno == EEXIST)
            throw std::runtime_error("LocalBroker: Named shared memory already exists");
        throw std::runtime_error("LocalBroker: Failed to create shared memory");
    }

    if (ftruncate(shm_fd, size) != 0) {
        shm_unlink(key);
        close(shm_fd);
        throw std::runtime_error("LocalBroker: Failed to truncate shared memory");
    }

    void *addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);
    if (addr == MAP_FAILED) {
        shm_unlink(key);
        throw std::runtime_error("LocalBroker: Failed to map to shared memory");
    }

    return addr;
}

__host__ inline void *open_shm(const char *key, size_t size) {
    int shm_fd;
    while (true) {
        shm_fd = shm_open(key, O_RDWR | O_CLOEXEC, 0);
        if (shm_fd >= 0)
            break;
        if (errno != ENOENT) 
            throw std::runtime_error("LocalBroker: Failed to open shared memory");
        usleep(1);
    }

    struct stat shm_st;
    do {
        if (fstat(shm_fd, &shm_st) != 0) {
            shm_unlink(key);
            close(shm_fd);
            throw std::runtime_error("LocalBroker: Failed to open shared memory stats");
        }
        usleep(1);
    } while ((size_t)shm_st.st_size < size);

    void *addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);
    if (addr == MAP_FAILED) {
        shm_unlink(key);
        throw std::runtime_error("LocalBroker: Failed to map to shared memory");
    }

    return addr;
}

__host__ inline void unlink_shm(const char *key) {
    shm_unlink(key);
}

__host__ inline void unmap_shm(void *addr, size_t size) {
    munmap(addr, size);
}

__host__ inline int create_socket(const char *key, int local_rank) {
    int sock_fd;
    if ((sock_fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0)) < 0)
        throw std::runtime_error("LocalBroker: Socket creation error");

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;

    char unique_key[64];
    int n = snprintf(unique_key, sizeof(unique_key), "%s%d", key, local_rank);
    if (n < 0 || n >= (int)sizeof(unique_key)) {
        close(sock_fd);
        throw std::runtime_error("LocalBroker: Socket name too long"); 
    }

    size_t len = strnlen(unique_key, sizeof(addr.sun_path));
    if (len > (sizeof(addr.sun_path) - 1)) {
        close(sock_fd);
        throw std::runtime_error("LocalBroker: Socket name too long");
    }
    strcpy(addr.sun_path, unique_key);
    unlink(unique_key);

    if (bind(sock_fd, (struct sockaddr *)&addr, SUN_LEN(&addr)) < 0) {
        close(sock_fd);
        throw std::runtime_error("LocalBroker: Failed to bind socket");
    }

    return sock_fd;
}

__host__ inline void send_fd(
    int sock_fd,
    int data_fd,
    const char *dst_key,
    int dst_local_rank,
    int src_local_rank
) {
    union {
      struct cmsghdr cm;
      char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int));
    control_un.control = reinterpret_cast<char *>(malloc(sizeof_control));
    if (!control_un.control) {
        close(sock_fd);
        close(data_fd);
        throw std::runtime_error("LocalBroker: Failed to allocate a control buffer");
    }
  
    struct msghdr msg {};
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;
  
    struct cmsghdr *cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memmove(CMSG_DATA(cmptr), &data_fd, sizeof(data_fd));

    struct sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    char dst_unique_key[64];
    int n = snprintf(dst_unique_key, sizeof(dst_unique_key), "%s%d", dst_key, dst_local_rank);
    if (n < 0 || n >= (int)sizeof(dst_unique_key)) { 
        free(control_un.control);
        close(sock_fd);
        close(data_fd);
        throw std::runtime_error("LocalBroker: dst path too long"); 
    }
    strcpy(addr.sun_path, dst_unique_key);
    msg.msg_name = (void *)&addr;
    msg.msg_namelen = sizeof(struct sockaddr_un);
  
    int payload = src_local_rank;
    struct iovec iov[1];
    iov[0].iov_base = &payload;
    iov[0].iov_len  = sizeof(payload);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
  
    while (true) {
        ssize_t sent = sendmsg(sock_fd, &msg, 0);
        if (sent <= 0) {
            if (errno == EINTR) continue;
            close(sock_fd);
            close(data_fd);
            free(control_un.control);
            throw std::runtime_error("LocalBroker: Failed to send FD over socket");
        }
        break;
    }

    free(control_un.control);
}

__host__ inline void recv_fd(int sock_fd, int *data_fd, int *src_local_rank) {
    union {
      struct cmsghdr cm;
      char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int));
    control_un.control = reinterpret_cast<char *>(malloc(sizeof_control));
    if (!control_un.control) {
        close(sock_fd);
        throw std::runtime_error("LocalBroker: Failed to allocate a control buffer");
    }

    struct msghdr msg {};
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;

    int payload = -1;
    struct iovec iov[1];
    iov[0].iov_base = &payload;
    iov[0].iov_len  = sizeof(payload);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    while (true) {
        ssize_t received = recvmsg(sock_fd, &msg, 0);
        if (received < 0 && errno == EINTR) {
            msg.msg_controllen = sizeof_control;
            msg.msg_iovlen = 1;
            continue;
        }
        if (received < static_cast<ssize_t>(sizeof(*data_fd))) {
            free(control_un.control);
            close(sock_fd);
            throw std::runtime_error("LocalBroker: Failed to receive data over socket");
        }
        break;
    }

    if (msg.msg_flags & MSG_CTRUNC) {
        free(control_un.control);
        close(sock_fd);
        throw std::runtime_error("LocalBroker: Control data truncated");
    }

    struct cmsghdr *cmptr = CMSG_FIRSTHDR(&msg);
    if (!cmptr ||
        cmptr->cmsg_len != CMSG_LEN(sizeof(int)) ||
        cmptr->cmsg_level != SOL_SOCKET ||
        cmptr->cmsg_type != SCM_RIGHTS) {
        free(control_un.control);
        close(sock_fd);
        throw std::runtime_error("LocalBroker: Failed to receive data over socket");
    }

    memmove(data_fd, CMSG_DATA(cmptr), sizeof(*data_fd));
    free(control_un.control);
    *src_local_rank = payload;
}

__host__ inline void unlink_socket(const char *key, int local_rank) {
    char unique_key[64];
    int n = snprintf(unique_key, sizeof(unique_key), "%s%d", key, local_rank);
    if (n < 0 || n >= (int)sizeof(unique_key))
        throw std::runtime_error("LocalBroker: Socket name too long");
    unlink(unique_key);
}

__host__ inline void close_socket(int sock_fd) {
    close(sock_fd);
}

} // namespace local_broker
} // namespace detail

/**
    @brief LocalBroker utility for multiprocess data exchange.

    Note that the code relies on POSIX sockets/shared memory/semaphores for 
    inter-process communication and synchronization.

    The main functions meant to be used by the user are:

        LocalBroker broker(local_rank, local_world_size);
        broker.exchange_data(dst, src, size); // exchange data between all processes
        broker.exchange_fds(dst, src_fd); // exchange file descriptors between all processes
        broker.broadcast_fd(dst, src_fd, src_rank); // broadcast file descriptor from src_rank to all processes
        broker.sync(); // wait until all processes reach here
 */
struct LocalBroker {
    // Keys can be overridden via env DIST_BROKER_KEY to avoid socket/shm conflicts between runs
    static constexpr int KEY_MAX = 96;
    char shm_key_[KEY_MAX];
    char sock_key_[KEY_MAX];

    int local_rank_;
    int local_world_size_;

    void *shm_raw_;
    volatile detail::local_broker::LocalBrokerVault *shm_;
    int sock_;

    __host__ inline LocalBroker(int local_rank, int local_world_size)
        : local_rank_(local_rank), 
          local_world_size_(local_world_size),
          shm_raw_(nullptr),
          shm_(nullptr),
          sock_(-1) {
        if (local_rank_ < 0)
            throw std::runtime_error("LocalBroker: Local rank must be non-negative");
        if (local_rank_ >= local_world_size_)
            throw std::runtime_error("LocalBroker: Local rank is greater than local world size");
        if (local_world_size_ > detail::local_broker::MAX_LOCAL_WORLD_SIZE)
            throw std::runtime_error("LocalBroker: Local world size is greater than MAX_LOCAL_WORLD_SIZE");

        const char *base = std::getenv("DIST_BROKER_KEY");
        if (base && base[0] != '\0') {
            (void)snprintf(shm_key_, KEY_MAX, "/%s_shm", base);
            (void)snprintf(sock_key_, KEY_MAX, "/tmp/%s.sock", base);
        } else {
            (void)snprintf(shm_key_, KEY_MAX, "/dist_broker_shm");
            (void)snprintf(sock_key_, KEY_MAX, "/tmp/dist_broker.sock");
        }

        if (local_rank_ == 0) {
            shm_raw_ = detail::local_broker::create_shm(shm_key_, sizeof(detail::local_broker::LocalBrokerVault));
            shm_ = reinterpret_cast<volatile detail::local_broker::LocalBrokerVault *>(shm_raw_);
            memset(shm_raw_, 0, sizeof(detail::local_broker::LocalBrokerVault));
        } else {
            shm_raw_ = detail::local_broker::open_shm(shm_key_, sizeof(detail::local_broker::LocalBrokerVault));
            shm_ = reinterpret_cast<volatile detail::local_broker::LocalBrokerVault *>(shm_raw_);
        }
        detail::local_broker::init_sync(local_rank_, shm_);
        detail::local_broker::sync(local_world_size_, shm_);

        if (local_rank_ == 0)
            detail::local_broker::unlink_shm(shm_key_);
        detail::local_broker::sync(local_world_size_, shm_);

        sock_ = detail::local_broker::create_socket(sock_key_, local_rank_);
        detail::local_broker::sync(local_world_size_, shm_);
    }

    LocalBroker(const LocalBroker&) = delete;
    LocalBroker& operator=(const LocalBroker&) = delete;

    __host__ inline LocalBroker(LocalBroker&& other) noexcept
        : local_rank_(other.local_rank_),
          local_world_size_(other.local_world_size_),
          shm_raw_(other.shm_raw_),
          shm_(other.shm_),
          sock_(other.sock_) {
        other.local_rank_ = -1;
        other.local_world_size_ = -1;
        other.shm_raw_ = nullptr;
        other.shm_ = nullptr;
        other.sock_ = -1;
    }

    __host__ inline void destroy() {
        if (shm_raw_) {
            detail::local_broker::unmap_shm(shm_raw_, sizeof(detail::local_broker::LocalBrokerVault));
            shm_raw_ = nullptr;
            shm_ = nullptr;
        }
        if (sock_ >= 0) {
            detail::local_broker::unlink_socket(sock_key_, local_rank_);
            detail::local_broker::close_socket(sock_);
            sock_ = -1;
        }
        local_rank_ = -1;
        local_world_size_ = -1;
    }

    __host__ inline LocalBroker& operator=(LocalBroker&& other) noexcept {
        if (this != &other) {
            destroy();
            local_rank_ = other.local_rank_;
            local_world_size_ = other.local_world_size_;
            shm_raw_ = other.shm_raw_;
            shm_ = other.shm_;
            sock_ = other.sock_;
            other.local_rank_ = -1;
            other.local_world_size_ = -1;
            other.shm_raw_ = nullptr;
            other.shm_ = nullptr;
            other.sock_ = -1;
        }
        return *this;
    }

    __host__ inline ~LocalBroker() {
        destroy();
    }

    __host__ inline void sync(int num_ranks = -1) {
        if (num_ranks == -1)
            num_ranks = local_world_size_;
        else if (num_ranks < 0 || num_ranks > local_world_size_)
            throw std::runtime_error("LocalBroker: Invalid number of ranks");

        detail::local_broker::sync(num_ranks, shm_);
    }

    __host__ inline void exchange_data(void *dst_, const void *src_, size_t size) {
        if (size > detail::local_broker::VAULT_SIZE_PER_RANK)
            throw std::runtime_error("LocalBroker: Size is greater than VAULT_SIZE_PER_RANK");

        uint8_t *dst = reinterpret_cast<uint8_t *>(dst_);
        const uint8_t *src = reinterpret_cast<const uint8_t *>(src_);

        // Exchange data
        sync(); // ensure all processes enter together
        memcpy(const_cast<uint8_t *>(shm_->data) + local_rank_ * detail::local_broker::VAULT_SIZE_PER_RANK, src, size);
        sync(); // ensure all processes exit together
    
        // Pack and copy back to destination
        for (int i = 0; i < local_world_size_; i++)
            memcpy(dst + i * size, const_cast<uint8_t *>(shm_->data) + i * detail::local_broker::VAULT_SIZE_PER_RANK, size);
    }

    __host__ inline void exchange_fds(int *dst, const int data_fd) {
        if (dst == nullptr)
            throw std::runtime_error("LocalBroker: dst is null");
        if (data_fd < 0)
            throw std::runtime_error("LocalBroker: source fd is negative");

        // Initialize dst buffer
        for (int i = 0; i < local_world_size_; ++i)
            dst[i] = -1;

        // Ensure all processes enter together
        sync();

        if (local_rank_ == 0) {
            // Rank 0 receives all FDs from and distributes them to other ranks
            dst[0] = data_fd;
            for (int i = 0; i < local_world_size_ - 1; i++) {
                int received_fd;
                int src_local_rank;
                detail::local_broker::recv_fd(sock_, &received_fd, &src_local_rank);
                if (received_fd < 0)
                    throw std::runtime_error("LocalBroker: Failed to receive FD over socket");
                if (src_local_rank == local_rank_)
                    throw std::runtime_error("LocalBroker: Invalid source rank");
                dst[src_local_rank] = received_fd;
            }
            for (int dst_local_rank = 1; dst_local_rank < local_world_size_; dst_local_rank++) {
                for (int src_local_rank = 0; src_local_rank < local_world_size_; src_local_rank++) {
                    if (dst_local_rank == src_local_rank)
                        continue;
                    detail::local_broker::send_fd(sock_, dst[src_local_rank], sock_key_, dst_local_rank, src_local_rank);
                }
            }
            close(dst[0]); // no longer needed
            dst[0] = -1;
        } else {
            // The rest sends its FD to and receives the other FDs from rank 0
            detail::local_broker::send_fd(sock_, data_fd, sock_key_, 0, local_rank_);
            close(data_fd); // no longer needed
            for (int i = 0; i < local_world_size_ - 1; i++) {
                int received_fd;
                int src_local_rank;
                detail::local_broker::recv_fd(sock_, &received_fd, &src_local_rank);
                if (received_fd < 0)
                    throw std::runtime_error("LocalBroker: Failed to receive FD over socket");
                if (src_local_rank == local_rank_)
                    throw std::runtime_error("LocalBroker: Invalid source rank");
                dst[src_local_rank] = received_fd;
            }
        }

        // Ensure all processes exit together
        sync();
    }

    __host__ inline void broadcast_fd(int *dst, const int data_fd, const int src_local_rank) {
        if (src_local_rank < 0 || src_local_rank >= local_world_size_)
            throw std::runtime_error("LocalBroker: Invalid source rank");

        // Ensure all processes enter together
        sync();

        if (local_rank_ == src_local_rank) {
            if (data_fd < 0)
                throw std::runtime_error("LocalBroker: Source rank has invalid FD");
            for (int dst_local_rank = 0; dst_local_rank < local_world_size_; dst_local_rank++) {
                if (dst_local_rank == src_local_rank)
                    continue;
                detail::local_broker::send_fd(sock_, data_fd, sock_key_, dst_local_rank, src_local_rank);
            }
            close(data_fd); // no longer needed
        } else {
            if (!dst)
                throw std::runtime_error("LocalBroker: Destination rank has invalid buffer");
            int _src_local_rank;
            detail::local_broker::recv_fd(sock_, dst, &_src_local_rank);
            if (*dst < 0)
                throw std::runtime_error("LocalBroker: Failed to receive valid FD over socket");
            if (_src_local_rank != src_local_rank)
                throw std::runtime_error("LocalBroker: Invalid source rank");
        }

        // Ensure all processes exit together
        sync();
    }
};

} // namespace comm
