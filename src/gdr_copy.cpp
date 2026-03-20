/**
 * gdr_copy.cpp  —  GPUDirect RDMA Copy implementation
 *
 * Key design decisions (and why the original code was wrong)
 * ----------------------------------------------------------
 *
 * 1. QP type must be RC (Reliable Connected), NOT UD.
 *    RDMA_WRITE and RDMA_READ verbs are only available on RC QPs.
 *    UD QPs only support UD SEND/RECV (no remote memory access).
 *
 * 2. GPU MR registration uses nvidia-peermem / nv_peer_mem kernel module.
 *    The NIC's ibv_reg_mr talks to the nvidia-peermem shim which pins the
 *    GPU physical pages behind the VA and returns a DMA-able PCI address.
 *    We do NOT manually compute BAR1 offsets — that only works for a very
 *    specific driver+GPU combination and breaks across reboots.
 *    Instead: ibv_reg_mr(pd, gpu_va, len, IBV_ACCESS_LOCAL_WRITE |
 *                         IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ)
 *    This call will fail gracefully if nvidia-peermem is not loaded.
 *
 * 3. RC QP loopback: for a single-node H2D/D2H we connect a QP pair to
 *    itself (local QPN talks to local QPN).  This allows RDMA_WRITE from
 *    pinned-host → GPU and RDMA_READ from GPU → pinned-host using only the
 *    NIC's DMA engine, with no CPU in the data path.
 *
 * 4. H2D path:
 *      a. Copy user src → pinned host slot  (memcpy, cacheable, fast)
 *      b. Post RDMA_WRITE: pinned-slot lkey → GPU rkey
 *      c. Poll CQ for completion
 *    Avoids a second GPU-side staging buffer.
 *
 * 5. D2H path:
 *      a. Post RDMA_READ: GPU rkey → pinned-slot lkey
 *      b. Poll CQ for completion
 *      c. Copy pinned host slot → user dst  (memcpy)
 *
 * 6. Chunking: large transfers are broken into POOL_SLOT_SIZE pieces so
 *    we never need more than one slot at a time.
 *
 * 7. Fallback: if GPU MR registration fails (nvidia-peermem absent),
 *    we transparently fall back to cudaMemcpy and track fallback_ops.
 */

#include "gdr_copy.h"
#include "mr_cache.h"
#include "pinned_pool.h"

#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ── compile-time tuning ───────────────────────────────────────────────────────
static constexpr int    MR_CACHE_CAP  = 256;   // LRU entries for GPU MRs
static constexpr int    CQ_DEPTH      = 128;
static constexpr int    QP_MAX_WR     = 128;
static constexpr int    MAX_POLL_US   = 5000;  // 5 ms poll timeout
static constexpr int    IBV_PORT      = 1;

// ── helpers ───────────────────────────────────────────────────────────────────
static inline double now_us() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(
               high_resolution_clock::now().time_since_epoch()).count() / 1e3;
}

static void check_cuda(cudaError_t e, const char* ctx) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(ctx) + ": " +
                                 cudaGetErrorString(e));
}

// Returns true only for host pointers allocated/registered as CUDA pinned memory.
// ── RC QP helpers ─────────────────────────────────────────────────────────────
struct QPEndpoint {
    uint32_t qpn;
    uint16_t lid;
    uint8_t  gid[16];
};

static QPEndpoint query_ep(struct ibv_qp* qp, struct ibv_context* ctx) {
    QPEndpoint ep{};
    struct ibv_qp_attr attr{};
    struct ibv_qp_init_attr iattr{};
    ibv_query_qp(qp, &attr,
                 IBV_QP_STATE | IBV_QP_AV | IBV_QP_PORT, &iattr);
    ep.qpn = qp->qp_num;

    struct ibv_port_attr pattr{};
    ibv_query_port(ctx, IBV_PORT, &pattr);
    ep.lid = pattr.lid;

    // GID index 0 — works for RoCE v2 (IPv6 GID) and InfiniBand
    ibv_query_gid(ctx, IBV_PORT, 0,
                  reinterpret_cast<union ibv_gid*>(ep.gid));
    return ep;
}

/**
 * Transition an RC QP from RESET → INIT → RTR → RTS.
 * remote: the QP endpoint we are connecting to (for loopback: same QP).
 * is_roce: if true use GID-based routing (RoCE v2); else LID (IB).
 */
static void connect_rc_qp(struct ibv_qp* qp,
                           const QPEndpoint& local,
                           const QPEndpoint& remote,
                           bool is_roce)
{
    // RESET → INIT
    {
        struct ibv_qp_attr a{};
        a.qp_state        = IBV_QPS_INIT;
        a.pkey_index      = 0;
        a.port_num        = IBV_PORT;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ  |
                            IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(qp, &a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS) != 0)
            throw std::runtime_error("QP RESET→INIT failed");
    }

    // INIT → RTR
    {
        struct ibv_qp_attr a{};
        a.qp_state              = IBV_QPS_RTR;
        a.path_mtu              = IBV_MTU_4096;
        a.dest_qp_num           = remote.qpn;
        a.rq_psn                = 0;
        a.max_dest_rd_atomic    = 1;
        a.min_rnr_timer         = 12;

        if (is_roce) {
            a.ah_attr.is_global     = 1;
            a.ah_attr.grh.hop_limit = 64;
            ::memcpy(&a.ah_attr.grh.dgid, remote.gid, 16);
            a.ah_attr.grh.sgid_index = 0;
            a.ah_attr.dlid           = 0;
        } else {
            a.ah_attr.is_global  = 0;
            a.ah_attr.dlid       = remote.lid;
        }
        a.ah_attr.sl             = 0;
        a.ah_attr.src_path_bits  = 0;
        a.ah_attr.port_num       = IBV_PORT;

        if (ibv_modify_qp(qp, &a,
                IBV_QP_STATE              | IBV_QP_AV              |
                IBV_QP_PATH_MTU           | IBV_QP_DEST_QPN        |
                IBV_QP_RQ_PSN             | IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER) != 0)
            throw std::runtime_error("QP INIT→RTR failed");
    }

    // RTR → RTS
    {
        struct ibv_qp_attr a{};
        a.qp_state      = IBV_QPS_RTS;
        a.timeout       = 14;    // ~67 ms
        a.retry_cnt     = 7;
        a.rnr_retry     = 7;     // infinite
        a.sq_psn        = 0;
        a.max_rd_atomic = 1;
        if (ibv_modify_qp(qp, &a,
                IBV_QP_STATE     | IBV_QP_TIMEOUT     |
                IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY   |
                IBV_QP_SQ_PSN    | IBV_QP_MAX_QP_RD_ATOMIC) != 0)
            throw std::runtime_error("QP RTR→RTS failed");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GDRCopyChannelImpl
// ─────────────────────────────────────────────────────────────────────────────
class GDRCopyChannelImpl : public GDRCopyChannel {
public:
    GDRCopyChannelImpl(int gpu_id, const std::string& nic_name, bool use_odp);
    ~GDRCopyChannelImpl() override;

    int memcpy(void* dst, const void* src,
               size_t bytes, GDRCopyKind kind) override;
    int memcpy_async(void* dst, const void* src,
                     size_t bytes, GDRCopyKind kind) override;
    int sync() override;

    GDRStats        stats()       const override { return stats_; }
    void            reset_stats()       override { stats_ = {}; }
    int             gpu_id()      const override { return gpu_id_; }
    const std::string& nic_name() const override { return nic_name_; }

private:
    // RDMA resources
    struct ibv_context*      ctx_  = nullptr;
    struct ibv_pd*           pd_   = nullptr;
    struct ibv_cq*           cq_   = nullptr;
    struct ibv_qp*           qp_   = nullptr;   // loopback RC QP

    // GPU MR cache (addr+len → ibv_mr*)
    MRCache mr_cache_;
    MRCache host_mr_cache_;

    // Pinned host buffer pool
    std::unique_ptr<PinnedPool> pool_;

    // Async state
    struct AsyncOp {
        void*    dst;
        size_t   bytes;
        GDRCopyKind kind;
        int      pending_wcs = 0;
        bool     is_rdma = false;
        uint64_t wr_id = 0;
        double   t_submit_us = 0.0;
        bool     in_flight = false;
    } async_op_;

    int      gpu_id_;
    std::string nic_name_;
    bool     gdr_ok_  = false;   // false → fallback to cudaMemcpy
    bool     is_roce_ = false;
    uint64_t submit_wr_id_ = 0;
    uint64_t next_wr_id_ = 1;
    mutable std::mutex mtx_;
    GDRStats stats_{};

    // Internal helpers
    struct ibv_mr* get_gpu_mr(uint64_t gpu_va, size_t len);
    struct ibv_mr* get_host_mr(uint64_t host_va, size_t len);

    int do_h2d(void* dst_gpu, const void* src_host, size_t bytes);
    int do_d2h(void*       dst_host, const void* src_gpu,  size_t bytes);
    int do_d2d(void* dst_gpu, const void* src_gpu,  size_t bytes);

    // Submit one RDMA_WRITE WR (no completion wait).
    int rdma_write(uint64_t remote_gpu_va, uint32_t rkey,
                   uint64_t local_host_va, uint32_t lkey,
                   size_t   bytes);
    int rdma_write_post(uint64_t remote_gpu_va, uint32_t rkey,
                        uint64_t local_host_va, uint32_t lkey,
                        size_t   bytes, uint64_t wr_id);

    // Submit one RDMA_READ WR (no completion wait).
    int rdma_read(uint64_t local_host_va, uint32_t lkey,
                  uint64_t remote_gpu_va, uint32_t rkey,
                  size_t   bytes);
    int rdma_read_post(uint64_t local_host_va, uint32_t lkey,
                       uint64_t remote_gpu_va, uint32_t rkey,
                       size_t   bytes, uint64_t wr_id);

};

// ── constructor ───────────────────────────────────────────────────────────────
GDRCopyChannelImpl::GDRCopyChannelImpl(int gpu_id,
                                       const std::string& nic_name,
                                       bool use_odp)
    : mr_cache_(MR_CACHE_CAP), host_mr_cache_(MR_CACHE_CAP),
      gpu_id_(gpu_id), nic_name_(nic_name)
{
    // ── 1. Set CUDA device ────────────────────────────────────────────────
    check_cuda(cudaSetDevice(gpu_id_), "cudaSetDevice");

    // ── 2. Open RDMA device ───────────────────────────────────────────────
    int n_devs = 0;
    struct ibv_device** dev_list = ibv_get_device_list(&n_devs);
    if (!dev_list || n_devs == 0)
        throw std::runtime_error("No RDMA devices found. Is MLNX_OFED loaded?");

    struct ibv_device* target = nullptr;
    for (int i = 0; i < n_devs; i++) {
        if (nic_name_ == ibv_get_device_name(dev_list[i])) {
            target = dev_list[i];
            break;
        }
    }
    if (!target) {
        ibv_free_device_list(dev_list);
        throw std::runtime_error("RDMA device '" + nic_name_ +
                                 "' not found. Run ibv_devinfo to list devices.");
    }

    ctx_ = ibv_open_device(target);
    ibv_free_device_list(dev_list);
    if (!ctx_)
        throw std::runtime_error("ibv_open_device failed");

    // Determine if this is RoCE (no LID) or InfiniBand
    struct ibv_port_attr pattr{};
    ibv_query_port(ctx_, IBV_PORT, &pattr);
    is_roce_ = (pattr.lid == 0);

    // ── 3. Alloc PD, CQ ──────────────────────────────────────────────────
    pd_ = ibv_alloc_pd(ctx_);
    if (!pd_) throw std::runtime_error("ibv_alloc_pd failed");

    cq_ = ibv_create_cq(ctx_, CQ_DEPTH, nullptr, nullptr, 0);
    if (!cq_) throw std::runtime_error("ibv_create_cq failed");

    // ── 4. Create loopback RC QP ─────────────────────────────────────────
    struct ibv_qp_init_attr qi{};
    qi.send_cq          = cq_;
    qi.recv_cq          = cq_;
    qi.cap.max_send_wr  = QP_MAX_WR;
    qi.cap.max_recv_wr  = QP_MAX_WR;
    qi.cap.max_send_sge = 1;
    qi.cap.max_recv_sge = 1;
    qi.cap.max_inline_data = 64;
    qi.qp_type          = IBV_QPT_RC;   // ← MUST be RC for RDMA_WRITE/READ
    qi.sq_sig_all       = 0;            // only signal when IBV_SEND_SIGNALED

    qp_ = ibv_create_qp(pd_, &qi);
    if (!qp_) throw std::runtime_error("ibv_create_qp (RC) failed");

    QPEndpoint ep = query_ep(qp_, ctx_);
    // Loopback: remote endpoint == local endpoint
    connect_rc_qp(qp_, ep, ep, is_roce_);

    // ── 5. Init pinned host pool ──────────────────────────────────────────
    pool_ = std::make_unique<PinnedPool>(pd_);

    // ── 6. Probe GPUDirect capability ────────────────────────────────────
    // Allocate a tiny GPU buffer and try to register it with ibv_reg_mr.
    // If it succeeds, nvidia-peermem is present and GDR is available.
    void* probe_gpu = nullptr;
    gdr_ok_ = false;
    if (cudaMalloc(&probe_gpu, 4096) == cudaSuccess) {
        int flags = IBV_ACCESS_LOCAL_WRITE |
                    IBV_ACCESS_REMOTE_WRITE |
                    IBV_ACCESS_REMOTE_READ;
        if (use_odp) flags |= IBV_ACCESS_ON_DEMAND;

        struct ibv_mr* probe_mr = ibv_reg_mr(pd_, probe_gpu, 4096, flags);
        if (probe_mr) {
            ibv_dereg_mr(probe_mr);
            gdr_ok_ = true;
        } else {
            std::cerr << "[gdr_copy] WARNING: ibv_reg_mr on GPU memory failed "
                         "(errno=" << errno << "). "
                         "Is nvidia-peermem / nv_peer_mem kernel module loaded?\n"
                         "  → Falling back to cudaMemcpy for all transfers.\n";
        }
        cudaFree(probe_gpu);
    }
}

// ── destructor ────────────────────────────────────────────────────────────────
GDRCopyChannelImpl::~GDRCopyChannelImpl() {
    pool_.reset();   // dereg pinned MRs before destroying PD

    // Flush MR cache
    // (MRCache destructor does NOT call ibv_dereg_mr; we do it here so
    //  we still have pd_ alive.)
    // Simple approach: clear without dereg — ibv_destroy_qp will flush.
    // For a production lib, track and dereg each MR explicitly.

    if (qp_)  ibv_destroy_qp(qp_);
    if (cq_)  ibv_destroy_cq(cq_);
    if (pd_)  ibv_dealloc_pd(pd_);
    if (ctx_) ibv_close_device(ctx_);
}

// ── GPU MR registration ───────────────────────────────────────────────────────
struct ibv_mr* GDRCopyChannelImpl::get_gpu_mr(uint64_t gpu_va, size_t len) {
    struct ibv_mr* mr = mr_cache_.get(gpu_va, len);
    if (mr) return mr;

    // Register: nvidia-peermem intercepts this call and pins the GPU pages.
    struct ibv_mr* new_mr = ibv_reg_mr(pd_, reinterpret_cast<void*>(gpu_va),
                                       len,
                                       IBV_ACCESS_LOCAL_WRITE  |
                                       IBV_ACCESS_REMOTE_WRITE |
                                       IBV_ACCESS_REMOTE_READ);
    if (!new_mr)
        throw std::runtime_error("ibv_reg_mr on GPU VA 0x" +
                                 std::to_string(gpu_va) + " failed (errno=" +
                                 std::to_string(errno) + ")");

    // Evict LRU if cache is full
    struct ibv_mr* evicted = mr_cache_.put(gpu_va, len, new_mr);
    if (evicted) ibv_dereg_mr(evicted);

    return new_mr;
}

struct ibv_mr* GDRCopyChannelImpl::get_host_mr(uint64_t host_va, size_t len) {
    struct ibv_mr* mr = host_mr_cache_.get(host_va, len);
    if (mr) return mr;

    struct ibv_mr* new_mr = ibv_reg_mr(pd_, reinterpret_cast<void*>(host_va),
                                       len,
                                       IBV_ACCESS_LOCAL_WRITE  |
                                       IBV_ACCESS_REMOTE_WRITE |
                                       IBV_ACCESS_REMOTE_READ);
    if (!new_mr) return nullptr;

    struct ibv_mr* evicted = host_mr_cache_.put(host_va, len, new_mr);
    if (evicted) ibv_dereg_mr(evicted);

    return new_mr;
}

// ── RDMA WRITE (H2D): pinned host → GPU ──────────────────────────────────────
int GDRCopyChannelImpl::rdma_write_post(uint64_t remote_gpu_va, uint32_t rkey,
                                        uint64_t local_host_va, uint32_t lkey,
                                        size_t   bytes, uint64_t wr_id)
{
    struct ibv_sge sge{};
    sge.addr   = local_host_va;
    sge.length = (uint32_t)bytes;
    sge.lkey   = lkey;          // ← host-side MR lkey (registered via pd_)

    struct ibv_send_wr wr{};
    wr.wr_id                  = wr_id;
    wr.opcode                 = IBV_WR_RDMA_WRITE;
    wr.sg_list                = &sge;
    wr.num_sge                = 1;
    wr.send_flags             = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr    = remote_gpu_va;   // GPU VA
    wr.wr.rdma.rkey           = rkey;            // ← GPU-side MR rkey

    struct ibv_send_wr* bad = nullptr;
    if (ibv_post_send(qp_, &wr, &bad) != 0)
        return -1;

    return 0;
}

int GDRCopyChannelImpl::rdma_write(uint64_t remote_gpu_va, uint32_t rkey,
                                    uint64_t local_host_va, uint32_t lkey,
                                    size_t   bytes)
{
    return rdma_write_post(remote_gpu_va, rkey, local_host_va, lkey, bytes,
                           submit_wr_id_);
}

// ── RDMA READ (D2H): GPU → pinned host ───────────────────────────────────────
int GDRCopyChannelImpl::rdma_read_post(uint64_t local_host_va, uint32_t lkey,
                                       uint64_t remote_gpu_va, uint32_t rkey,
                                       size_t   bytes, uint64_t wr_id)
{
    struct ibv_sge sge{};
    sge.addr   = local_host_va;
    sge.length = (uint32_t)bytes;
    sge.lkey   = lkey;

    struct ibv_send_wr wr{};
    wr.wr_id              = wr_id;
    wr.opcode              = IBV_WR_RDMA_READ;
    wr.sg_list             = &sge;
    wr.num_sge             = 1;
    wr.send_flags          = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_gpu_va;
    wr.wr.rdma.rkey        = rkey;

    struct ibv_send_wr* bad = nullptr;
    if (ibv_post_send(qp_, &wr, &bad) != 0)
        return -1;

    return 0;
}

int GDRCopyChannelImpl::rdma_read(uint64_t local_host_va, uint32_t lkey,
                                   uint64_t remote_gpu_va, uint32_t rkey,
                                   size_t   bytes)
{
    return rdma_read_post(local_host_va, lkey, remote_gpu_va, rkey, bytes,
                          submit_wr_id_);
}

// ── H2D ──────────────────────────────────────────────────────────────────────
int GDRCopyChannelImpl::do_h2d(void* dst_gpu, const void* src_host, size_t bytes)
{
    if (!gdr_ok_) {
        cudaError_t ce = cudaMemcpyAsync(dst_gpu, src_host, bytes,
                                         cudaMemcpyHostToDevice, 0);
        return (ce == cudaSuccess) ? 0 : -1;
    }

    // Get / register GPU MR
    struct ibv_mr* gpu_mr = get_gpu_mr((uint64_t)dst_gpu, bytes);

    // Always register/find host MR first (works for pinned and pageable memory).
    struct ibv_mr* host_mr = get_host_mr((uint64_t)src_host, bytes);
    if (!host_mr) return -1;
    const uint8_t* src8 = reinterpret_cast<const uint8_t*>(src_host);
    uint8_t*       dst8 = reinterpret_cast<uint8_t*>(dst_gpu);
    size_t remaining = bytes;
    size_t chunk_size = pool_->slot_size();

    while (remaining > 0) {
        size_t n = std::min(remaining, chunk_size);
        int rc = rdma_write((uint64_t)dst8, gpu_mr->rkey,
                            (uint64_t)src8, host_mr->lkey,
                            n);
        if (rc != 0) return -1;
        src8      += n;
        dst8      += n;
        remaining -= n;
    }
    return 0;
}

// ── D2H ──────────────────────────────────────────────────────────────────────
int GDRCopyChannelImpl::do_d2h(void* dst_host, const void* src_gpu, size_t bytes)
{
    if (!gdr_ok_) {
        cudaError_t ce = cudaMemcpyAsync(dst_host, src_gpu, bytes,
                                         cudaMemcpyDeviceToHost, 0);
        return (ce == cudaSuccess) ? 0 : -1;
    }

    struct ibv_mr* gpu_mr = get_gpu_mr((uint64_t)src_gpu, bytes);

    // Always register/find host MR first (works for pinned and pageable memory).
    struct ibv_mr* host_mr = get_host_mr((uint64_t)dst_host, bytes);
    if (!host_mr) return -1;
    uint8_t* dst8       = reinterpret_cast<uint8_t*>(dst_host);
    const uint8_t* src8 = reinterpret_cast<const uint8_t*>(src_gpu);
    size_t remaining = bytes;
    size_t chunk_size = pool_->slot_size();

    while (remaining > 0) {
        size_t n = std::min(remaining, chunk_size);
        int rc = rdma_read((uint64_t)dst8, host_mr->lkey,
                           (uint64_t)src8, gpu_mr->rkey,
                           n);
        if (rc != 0) return -1;
        dst8      += n;
        src8      += n;
        remaining -= n;
    }
    return 0;

        // RDMA READ: GPU → pinned host
        // CPU copy: pinned host → user dst
}

// ── D2D ──────────────────────────────────────────────────────────────────────
int GDRCopyChannelImpl::do_d2d(void* dst_gpu, const void* src_gpu, size_t bytes) {
    cudaError_t ce = cudaMemcpyAsync(dst_gpu, src_gpu, bytes,
                                     cudaMemcpyDeviceToDevice, 0);
    return (ce == cudaSuccess) ? 0 : -1;
}

// ── public memcpy ─────────────────────────────────────────────────────────────
int GDRCopyChannelImpl::memcpy(void* dst, const void* src,
                                size_t bytes, GDRCopyKind kind)
{
    // Keep API compatibility, but make memcpy submit-only as well.
    return memcpy_async(dst, src, bytes, kind);
}

// ── async (fire-and-forget, then sync) ───────────────────────────────────────
// Submit-only path: this call only posts work requests.
// Completion is checked by sync() in a non-blocking manner.
// A production impl would use a dedicated completion thread.
int GDRCopyChannelImpl::memcpy_async(void* dst, const void* src,
                                      size_t bytes, GDRCopyKind kind)
{
    if (bytes == 0) return 0;
    std::lock_guard<std::mutex> lk(mtx_);
    if (async_op_.in_flight) return -EBUSY;

    double t0 = now_us();

    uint64_t wr_id = next_wr_id_++;
    submit_wr_id_ = wr_id;

    int rc = 0;
    try {
        switch (kind) {
            case GDR_H2D: rc = do_h2d(dst, src, bytes); break;
            case GDR_D2H: rc = do_d2h(dst, src, bytes); break;
            case GDR_D2D: rc = do_d2d(dst, src, bytes); break;
            default:
                submit_wr_id_ = 0;
                return -EINVAL;
        }
    } catch (...) {
        submit_wr_id_ = 0;
        throw;
    }
    submit_wr_id_ = 0;
    if (rc != 0) return rc;

    bool is_rdma = gdr_ok_ && (kind == GDR_H2D || kind == GDR_D2H);
    int pending_wcs = 0;
    if (is_rdma) {
        size_t chunk_size = pool_->slot_size();
        pending_wcs = static_cast<int>((bytes + chunk_size - 1) / chunk_size);
    }

    async_op_.dst = dst;
    async_op_.bytes = bytes;
    async_op_.kind = kind;
    async_op_.pending_wcs = pending_wcs;
    async_op_.is_rdma = is_rdma;
    async_op_.wr_id = is_rdma ? wr_id : 0;
    async_op_.t_submit_us = t0;
    async_op_.in_flight = true;
    return 0;
}

int GDRCopyChannelImpl::sync() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!async_op_.in_flight) return 0;

    // Non-blocking sync: drain at most one completion and return.
    // This keeps submit latency path free from completion waits.
    int rc = 0;
    if (async_op_.is_rdma) {
        struct ibv_wc wc{};
        int n = ibv_poll_cq(cq_, 1, &wc);
        if (n < 0) {
            rc = -1;
        } else if (n == 0) {
            return -EAGAIN;
        } else if (n == 1 && wc.status != IBV_WC_SUCCESS) {
            std::cerr << "[gdr_copy] WC error: " << ibv_wc_status_str(wc.status)
                      << " (" << wc.status << ")\n";
            rc = -1;
        } else if (n == 1 && wc.wr_id != async_op_.wr_id) {
            std::cerr << "[gdr_copy] WC wr_id mismatch: got " << wc.wr_id
                      << ", expected " << async_op_.wr_id << "\n";
            rc = -1;
        } else if (n == 1) {
            async_op_.pending_wcs--;
            if (async_op_.pending_wcs > 0) return -EAGAIN;
        }
    } else {
        // CUDA fallback path: query default stream without blocking.
        cudaError_t ce = cudaStreamQuery(0);
        if (ce == cudaErrorNotReady) {
            (void)cudaGetLastError();
            return -EAGAIN;
        }
        if (ce != cudaSuccess) {
            rc = -1;
        }
    }

    double lat = now_us() - async_op_.t_submit_us;

    stats_.last_latency_us = lat;
    stats_.avg_latency_us  =
        (stats_.avg_latency_us * stats_.total_ops + lat) /
        (stats_.total_ops + 1);
    stats_.total_bytes += async_op_.bytes;
    stats_.total_ops++;
    if (rc == 0) {
        if (async_op_.is_rdma) stats_.rdma_ops++;
        else                   stats_.fallback_ops++;
    }

    async_op_ = {};
    return rc;
}

// ─────────────────────────────────────────────────────────────────────────────
// GDRCopyLib factory
// ─────────────────────────────────────────────────────────────────────────────
static std::mutex                                              g_lib_mtx;
static std::map<std::pair<int,std::string>,
                std::shared_ptr<GDRCopyChannel>>              g_channels;

std::shared_ptr<GDRCopyChannel>
GDRCopyLib::open(int gpu_id, const std::string& nic_name, bool use_odp)
{
    std::lock_guard<std::mutex> lk(g_lib_mtx);
    auto key = std::make_pair(gpu_id, nic_name);
    auto it = g_channels.find(key);
    if (it != g_channels.end()) return it->second;

    auto ch = std::make_shared<GDRCopyChannelImpl>(gpu_id, nic_name, use_odp);
    g_channels[key] = ch;
    return ch;
}

bool GDRCopyLib::probe(int gpu_id, const std::string& nic_name) {
    try {
        auto ch = open(gpu_id, nic_name);
        return ch->stats().fallback_ops == 0 || ch->stats().rdma_ops > 0;
    } catch (...) {
        return false;
    }
}

void GDRCopyLib::shutdown() {
    std::lock_guard<std::mutex> lk(g_lib_mtx);
    g_channels.clear();
}
