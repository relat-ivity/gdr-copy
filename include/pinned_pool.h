/**
 * pinned_pool.h  —  pool of RDMA-registered pinned host buffers
 *
 * ibv_reg_mr on host memory is cheap (~1 µs) but cudaHostAlloc with the
 * RDMA access flag + ibv_reg_mr together can be amortised across many ops
 * by pre-allocating a slab.
 *
 * For H2D: we use the host buffer as the "local" side of an RDMA WRITE.
 *   The NIC DMAs from pinned host → GPU BAR1.
 * For D2H: we use the host buffer as the "local" sink of an RDMA READ.
 *   The NIC pulls from GPU BAR1 → pinned host.
 *
 * The pool hands out slots of SLOT_SIZE bytes.  Large transfers are chunked.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <infiniband/verbs.h>
#include <cuda_runtime.h>

static constexpr size_t POOL_SLOT_SIZE  = 4ULL << 20;   // 4 MiB per slot
static constexpr size_t POOL_NUM_SLOTS  = 8;            // 32 MiB total

struct PinnedSlot {
    void*          host_ptr = nullptr;
    struct ibv_mr* mr       = nullptr;
    size_t         size     = 0;
};

class PinnedPool {
public:
    PinnedPool(struct ibv_pd* pd, size_t slot_size = POOL_SLOT_SIZE,
               size_t n_slots = POOL_NUM_SLOTS)
        : slot_size_(slot_size)
    {
        slots_.reserve(n_slots);
        for (size_t i = 0; i < n_slots; i++) {
            void* ptr = nullptr;
            // cudaHostAlloc with cudaHostAllocPortable so the pinning is
            // valid across CUDA contexts.
            cudaError_t ce = cudaHostAlloc(&ptr, slot_size,
                                           cudaHostAllocPortable);
            if (ce != cudaSuccess)
                throw std::runtime_error("cudaHostAlloc failed");

            struct ibv_mr* mr = ibv_reg_mr(pd, ptr, slot_size,
                                           IBV_ACCESS_LOCAL_WRITE |
                                           IBV_ACCESS_REMOTE_READ |
                                           IBV_ACCESS_REMOTE_WRITE);
            if (!mr) {
                cudaFreeHost(ptr);
                throw std::runtime_error("ibv_reg_mr on pinned host buf failed");
            }
            slots_.push_back({ptr, mr, slot_size});
            free_.push(i);
        }
    }

    ~PinnedPool() {
        for (auto& s : slots_) {
            if (s.mr)       ibv_dereg_mr(s.mr);
            if (s.host_ptr) cudaFreeHost(s.host_ptr);
        }
    }

    // Acquire a slot (blocking spin — pool should be large enough).
    PinnedSlot* acquire() {
        while (true) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!free_.empty()) {
                size_t idx = free_.front();
                free_.pop();
                return &slots_[idx];
            }
            // busy-wait; in practice this won't spin because ops are
            // serialised per-channel
        }
    }

    void release(PinnedSlot* s) {
        std::lock_guard<std::mutex> lk(mtx_);
        size_t idx = s - slots_.data();
        free_.push(idx);
    }

    size_t slot_size() const { return slot_size_; }

private:
    size_t                  slot_size_;
    std::vector<PinnedSlot> slots_;
    std::queue<size_t>      free_;
    std::mutex              mtx_;
};
