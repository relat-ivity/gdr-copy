/**
 * gdr_copy.h  —  GPUDirect RDMA Copy Library
 *
 * Drop-in replacement for cudaMemcpy on paths where the GPU and NIC
 * share the same PCIe switch (e.g. H20 + ConnectX-7 under one PLX).
 *
 * Motivation
 * ----------
 * Standard cudaMemcpy H2D/D2H crosses the CPU and involves kernel-mode
 * transitions (UVM fault handling, DMA engine scheduling). For small IOs
 * the round-trip through the OS dominates.  With GPUDirect RDMA the NIC
 * can read/write GPU BAR1 memory directly over PCIe with no CPU in the
 * critical path, cutting latency from ~10 µs to ~1-3 µs and allowing
 * the NIC to arbitrate between collective-comm and KVCache traffic
 * (see DualPath, NSDI'25).
 *
 * Architecture
 * ------------
 *   GDRCopyLib (singleton factory)
 *     └─ GDRCopyChannel  (one per GPU × NIC pair)
 *          ├─ RC QP pair (loopback, one for H2D one for D2H)
 *          ├─ fixed MR windows (reuse one host MR + one GPU MR)
 *          └─ PinnedPool (pre-pinned host bounce buffers)
 *
 * Usage
 * -----
 *   #include "gdr_copy.h"
 *
 *   // one-time init
 *   auto ch = GDRCopyLib::open(0, "mlx5_0");
 *
 *   // replace cudaMemcpy
 *   ch->memcpy(d_ptr, h_ptr, bytes, GDR_H2D);
 *   ch->memcpy(h_ptr, d_ptr, bytes, GDR_D2H);
 *
 *   // optional diagnostics
 *   GDRStats s = ch->stats();
 *   printf("last op: %.2f µs\n", s.last_latency_us);
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>

// ── copy direction (mirrors cudaMemcpyKind values for easy search-replace) ──
enum GDRCopyKind {
    GDR_H2D = 1,   // host   → GPU  (cudaMemcpyHostToDevice)
    GDR_D2H = 2,   // GPU    → host (cudaMemcpyDeviceToHost)
    GDR_D2D = 3,   // GPU    → GPU  (same device, falls back to cuMemcpy)
};

// ── per-operation statistics ─────────────────────────────────────────────────
struct GDRStats {
    double last_latency_us   = 0.0;   // wall time of last memcpy (µs)
    double avg_latency_us    = 0.0;   // running average (µs)
    uint64_t total_bytes     = 0;     // total bytes transferred
    uint64_t total_ops       = 0;     // total memcpy calls
    uint64_t rdma_ops        = 0;     // ops served by RDMA path
    uint64_t fallback_ops    = 0;     // ops that fell back to cudaMemcpy
};

// ── opaque channel (one per GPU × NIC) ───────────────────────────────────────
class GDRCopyChannel {
public:
    virtual ~GDRCopyChannel() = default;

    /**
     * memcpy — submit-only API (same submit semantics as memcpy_async).
     *
     * @param dst   destination pointer (GPU VA for H2D, host ptr for D2H)
     * @param src   source pointer
     * @param bytes number of bytes to transfer
     * @param kind  GDR_H2D / GDR_D2H / GDR_D2D
     * @return      0 on successful submission, negative errno-style on failure
     */
    virtual int memcpy(void* dst, const void* src,
                       size_t bytes, GDRCopyKind kind) = 0;

    /**
     * memcpy_async — post the transfer and return immediately.
     * Completion order is transport-driven and may be out-of-order.
     */
    virtual int memcpy_async(void* dst, const void* src,
                             size_t bytes, GDRCopyKind kind) = 0;

    /**
     * memcpy_async_tagged — submit one async request and return request metadata.
     * req_id identifies the request in completion path; expected_wcs is the
     * number of CQEs expected for this request (>=1).
     */
    virtual int memcpy_async_tagged(void* dst, const void* src,
                                    size_t bytes, GDRCopyKind kind,
                                    uint64_t* req_id, int* expected_wcs) = 0;

    /**
     * poll_wc — non-blocking polling of one completion token.
     * Returns 0 and sets req_id when one request is fully completed.
     * Returns -EAGAIN when no request has completed yet.
     */
    virtual int poll_wc(uint64_t* req_id) = 0;

    /**
     * Non-blocking progress check for pending async operations.
     * Returns 0 when one request completion is observed.
     * Returns -EAGAIN when no request has completed yet.
     */
    virtual int sync() = 0;

    /** Return accumulated statistics. */
    virtual GDRStats stats() const = 0;

    /** Reset statistics counters. */
    virtual void reset_stats() = 0;

    /** Return the GPU device index this channel was opened on. */
    virtual int gpu_id() const = 0;

    /** Return the NIC device name (e.g. "mlx5_0"). */
    virtual const std::string& nic_name() const = 0;
};

// ── library entry point ───────────────────────────────────────────────────────
class GDRCopyLib {
public:
    /**
     * open — create (or retrieve cached) channel for (gpu_id, nic_name).
     *
     * @param gpu_id   CUDA device ordinal (0-based)
     * @param nic_name RDMA device name reported by ibv_devinfo (e.g. "mlx5_0")
     * @param use_odp  enable On-Demand Paging (slower first-touch, no BAR1 pin)
     *                 set false (default) for lowest latency on H20
     * @throws std::runtime_error if GPU or NIC cannot be opened, or if
     *         GPUDirect is not supported by the driver stack.
     */
    static std::shared_ptr<GDRCopyChannel>
    open(int gpu_id, const std::string& nic_name, bool use_odp = false);

    /**
     * probe — check whether RDMA path is available without opening a channel.
     * Returns true only when:
     *   - nvidia-peermem / nv_peer_mem kernel module is loaded
     *   - GPU BAR1 is large enough to register the address range
     *   - NIC supports RDMA_RW capability
     */
    static bool probe(int gpu_id, const std::string& nic_name);

    /** Close all cached channels (called automatically at exit). */
    static void shutdown();

private:
    GDRCopyLib() = delete;
};
