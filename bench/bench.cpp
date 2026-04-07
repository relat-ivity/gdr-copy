/**
 * bench.cpp - GDR Copy vs NIXL-UCX timing benchmark
 *
 * Usage:
 *   sudo ./build/bench [gpu_id] [nic_name]
 *   sudo ./build/bench 0 mlx5_0
 *
 * Output:
 *   For each transfer size x direction, prints:
 *     - submit (issue) latency
 *     - batch bandwidth
 *     - median latency (us)
 *     - p99 latency (us)
 *     - GB/s only for bandwidth tables
 *   for both GDR RDMA path and NIXL-UCX baseline.
 *
 * Why sudo?
 *   Accessing PCIe config space for GPUDirect registration may require
 *   CAP_NET_ADMIN or CAP_SYS_RAWIO on some distros. Alternatively, set
 *   /proc/sys/kernel/perf_event_paranoid appropriately.
 */

#include "gdr_copy.h"

#include <cuda_runtime.h>
#include <nixl.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <numeric>

// Timing helpers.
static double now_us() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(
               high_resolution_clock::now().time_since_epoch()).count() / 1e3;
}

// Aggregate statistics for one metric set.
struct BenchResult {
    double median_us;
    double p99_us;
    double bw_GBs;
};

struct BenchPair {
    BenchResult issue;
    BenchResult transfer;
};

struct DirectionRow {
    size_t bytes = 0;
    BenchPair gdr{};
    BenchPair nixl{};
};

static BenchResult analyse(std::vector<double>& samples, size_t bytes) {
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    BenchResult r{};
    r.median_us = samples[n / 2];
    r.p99_us    = samples[(size_t)(n * 0.99)];
    double avg  = std::accumulate(samples.begin(), samples.end(), 0.0) / n;
    r.bw_GBs    = (bytes / 1e9) / (avg / 1e6);   // GB/s
    return r;
}

static const char* kNixlBackendName = "UCX";

static nixlBasicDesc make_nixl_basic_desc(void* ptr, size_t bytes, uint64_t dev_id) {
    return nixlBasicDesc(reinterpret_cast<uintptr_t>(ptr), bytes, dev_id);
}

static nixlBlobDesc make_nixl_blob_desc(void* ptr, size_t bytes, uint64_t dev_id) {
    return nixlBlobDesc(reinterpret_cast<uintptr_t>(ptr), bytes, dev_id);
}

class NixlLoopbackPair {
public:
    NixlLoopbackPair(int gpu_id, const std::string& nic_name)
        : initiator_name_("bench-init-" + std::to_string(gpu_id)),
          target_name_("bench-target-" + std::to_string(gpu_id)),
          initiator_(initiator_name_, make_agent_config()),
          target_(target_name_, make_agent_config()) {
        nixl_mem_list_t initiator_mems;
        nixl_mem_list_t target_mems;
        nixl_b_params_t initiator_params;
        nixl_b_params_t target_params;

        check_status(initiator_.getPluginParams(kNixlBackendName, initiator_mems, initiator_params),
                     "nixl getPluginParams initiator");
        check_status(target_.getPluginParams(kNixlBackendName, target_mems, target_params),
                     "nixl getPluginParams target");

        configure_ucx_params(initiator_params, nic_name);
        configure_ucx_params(target_params, nic_name);

        check_status(initiator_.createBackend(kNixlBackendName, initiator_params, initiator_backend_),
                     "nixl createBackend initiator");
        check_status(target_.createBackend(kNixlBackendName, target_params, target_backend_),
                     "nixl createBackend target");

        initiator_args_.backends.push_back(initiator_backend_);
        target_args_.backends.push_back(target_backend_);
    }

    ~NixlLoopbackPair() {
        if (remote_md_loaded_) {
            initiator_.invalidateRemoteMD(target_name_);
        }
    }

    // Register the local/remote buffers once, then exchange the target metadata in-process.
    // The benchmark keeps createXferReq outside the measured region, so issue timing only
    // reflects NIXL postXferReq overhead.
    void register_buffers(const nixl_reg_dlist_t& initiator_regs,
                          const nixl_reg_dlist_t& target_regs) {
        check_status(initiator_.registerMem(initiator_regs, &initiator_args_),
                     "nixl registerMem initiator");
        check_status(target_.registerMem(target_regs, &target_args_),
                     "nixl registerMem target");

        nixl_blob_t target_md;
        std::string loaded_name;
        check_status(target_.getLocalMD(target_md), "nixl getLocalMD target");
        check_status(initiator_.loadRemoteMD(target_md, loaded_name), "nixl loadRemoteMD initiator");
        remote_md_loaded_ = true;
    }

    void deregister_buffers(const nixl_reg_dlist_t& initiator_regs,
                            const nixl_reg_dlist_t& target_regs) {
        if (remote_md_loaded_) {
            initiator_.invalidateRemoteMD(target_name_);
            remote_md_loaded_ = false;
        }
        check_status(initiator_.deregisterMem(initiator_regs, &initiator_args_),
                     "nixl deregisterMem initiator");
        check_status(target_.deregisterMem(target_regs, &target_args_),
                     "nixl deregisterMem target");
    }

    nixlXferReqH* create_request(const nixl_xfer_op_t& op,
                                 const nixl_xfer_dlist_t& local_descs,
                                 const nixl_xfer_dlist_t& remote_descs) const {
        nixlXferReqH* req = nullptr;
        check_status(initiator_.createXferReq(op, local_descs, remote_descs, target_name_, req,
                                              &initiator_args_),
                     "nixl createXferReq");
        return req;
    }

    nixl_status_t post_request(nixlXferReqH* req) const {
        return initiator_.postXferReq(req, &initiator_args_);
    }

    nixl_status_t wait_request(nixlXferReqH* req) const {
        nixl_status_t st = initiator_.getXferStatus(req);
        while (st == NIXL_IN_PROG) {
            std::this_thread::yield();
            st = initiator_.getXferStatus(req);
        }
        return st;
    }

    void release_request(nixlXferReqH* req) const {
        if (!req)
            return;
        check_status(initiator_.releaseXferReq(req), "nixl releaseXferReq");
    }

private:
    static nixlAgentConfig make_agent_config() {
        nixlAgentConfig cfg;
        cfg.useProgThread = true;
        cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        return cfg;
    }

    static void configure_ucx_params(nixl_b_params_t& params, const std::string& nic_name) {
        params["device_list"] = nic_name;
        params["ucx_devices"] = nic_name;
    }

    static void check_status(nixl_status_t status, const char* what) {
        if (status == NIXL_SUCCESS)
            return;
        fprintf(stderr, "[issue] %s failed: rc=%d (%s)\n",
                what, (int)status, nixlEnumStrings::statusStr(status).c_str());
        std::exit(2);
    }

    std::string initiator_name_;
    std::string target_name_;
    nixlAgent initiator_;
    nixlAgent target_;
    nixlBackendH* initiator_backend_ = nullptr;
    nixlBackendH* target_backend_ = nullptr;
    nixl_opt_args_t initiator_args_{};
    nixl_opt_args_t target_args_{};
    bool remote_md_loaded_ = false;
};

// Batch issue + batch sync; only total time is measured for bandwidth.
static BenchPair run_gdr_timings(std::shared_ptr<GDRCopyChannel> ch,
                                 void* dst, const void* src,
                                 size_t bytes, GDRCopyKind kind,
                                 int warmup, int iters)
{
    ch->reset_stats();
    std::vector<double> issue_samples;
    issue_samples.reserve(iters);

    auto run_batch = [&](int count, bool measure) -> double {
        if (count <= 0) return 0.0;

        double t0 = 0.0;
        if (measure) t0 = now_us();

        // Keep issuing as much as possible; if SQ/CQ fills up, recycle one completion
        // and then continue issuing.
        int issued = 0;
        int done = 0;
        while (issued < count) {
            double t_issue0 = 0.0;
            if (measure) t_issue0 = now_us();
            int rc = ch->memcpy_async(dst, src, bytes, kind);
            if (rc == 0) {
                if (measure) {
                    double t_issue1 = now_us();
                    issue_samples.push_back(t_issue1 - t_issue0);
                }
                ++issued;
                continue;
            }
            if (rc == -EBUSY) {
                while (done < issued) {
                    int sc = ch->sync();
                    if (sc == 0) {
                        ++done;
                        break;
                    }
                    if (sc == -EAGAIN) continue;
                    fprintf(stderr, "[issue] gdr sync failed: rc=%d kind=%d bytes=%zu done=%d/%d\n",
                            sc, (int)kind, bytes, done, count);
                    std::exit(2);
                }
                continue;
            }
            fprintf(stderr, "[issue] gdr memcpy_async failed: rc=%d kind=%d bytes=%zu i=%d/%d\n",
                    rc, (int)kind, bytes, issued, count);
            std::exit(2);
        }

        // Drain all remaining completions for this batch.
        while (done < count) {
            int sc = ch->sync();
            if (sc == 0) {
                ++done;
                continue;
            }
            if (sc == -EAGAIN) continue;
            fprintf(stderr, "[issue] gdr sync failed: rc=%d kind=%d bytes=%zu done=%d/%d\n",
                    sc, (int)kind, bytes, done, count);
            std::exit(2);
        }

        if (!measure) return 0.0;
        return now_us() - t0;
    };

    run_batch(warmup, false);
    double total_us = run_batch(iters, true);

    BenchPair out{};
    out.issue = analyse(issue_samples, bytes);
    out.transfer.median_us = total_us;
    out.transfer.p99_us = total_us;
    out.transfer.bw_GBs = (iters > 0 && total_us > 0.0)
                            ? ((double)bytes * (double)iters / 1e9) / (total_us / 1e6)
                            : 0.0;
    return out;
}

static void format_size(size_t bytes, char* out, size_t out_len) {
    if (bytes < (1 << 10))      snprintf(out, out_len, "%zuB", bytes);
    else if (bytes < (1 << 20)) snprintf(out, out_len, "%zuKiB", bytes >> 10);
    else                        snprintf(out, out_len, "%zuMiB", bytes >> 20);
}

static void print_latency_table(const char* title,
                                const std::vector<DirectionRow>& rows,
                                bool issue_table)
{
    printf("\n--- %s ---\n", title);
    if (issue_table) {
        printf("%-12s | %-21s | %-21s\n",
               "Size", "   GDR (median / p99)  ", "   NIXL (median / p99)");
        printf("%-12s-+-%-21s-+-%-21s\n",
               "------------", "-----------------------", "-----------------------");
    } else {
        printf("%-12s | %-18s | %-18s\n",
               "Size", "    GDR (BW)     ", "    NIXL (BW)    ");
        printf("%-12s-+-%-18s-+-%-18s\n",
               "------------", "------------------", "------------------");
    }

    for (const auto& row : rows) {
        const BenchResult& g = issue_table ? row.gdr.issue : row.gdr.transfer;
        const BenchResult& c = issue_table ? row.nixl.issue : row.nixl.transfer;
        char size_str[32];
        format_size(row.bytes, size_str, sizeof(size_str));

        if (issue_table) {
            printf("%-12s | %7.2f us / %7.2f us | %7.2f us / %7.2f us\n",
                   size_str,
                   g.median_us, g.p99_us,
                   c.median_us, c.p99_us);
        } else {
            printf("%-12s | %8.2f GB/s      | %8.2f GB/s      \n",
                   size_str,
                   g.bw_GBs, c.bw_GBs);
        }
    }
}

static BenchPair run_nixl_timings(size_t bytes, cudaMemcpyKind kind,
                                  int warmup, int iters,
                                  int gpu_id, const std::string& nic_name)
{
    std::vector<double> issue_samples;
    issue_samples.reserve(iters);

    cudaError_t ce = cudaSetDevice(gpu_id);
    if (ce != cudaSuccess) {
        fprintf(stderr, "[nixl] cudaSetDevice failed: %s\n", cudaGetErrorString(ce));
        std::exit(2);
    }

    void* host_buf = nullptr;
    void* gpu_buf  = nullptr;
    nixl_mem_t   host_mem, gpu_mem;
    nixl_xfer_op_t op;

    switch (kind) {
        case cudaMemcpyHostToDevice:
            host_mem = DRAM_SEG; gpu_mem = VRAM_SEG; op = NIXL_WRITE;
            ce = cudaHostAlloc(&host_buf, bytes, cudaHostAllocPortable);
            if (ce == cudaSuccess) ce = cudaMalloc(&gpu_buf, bytes);
            if (ce == cudaSuccess) std::memset(host_buf, 0xA5, bytes);
            if (ce == cudaSuccess) ce = cudaMemset(gpu_buf, 0, bytes);
            break;
        case cudaMemcpyDeviceToHost:
            host_mem = DRAM_SEG; gpu_mem = VRAM_SEG; op = NIXL_READ;
            ce = cudaHostAlloc(&host_buf, bytes, cudaHostAllocPortable);
            if (ce == cudaSuccess) ce = cudaMalloc(&gpu_buf, bytes);
            if (ce == cudaSuccess) std::memset(host_buf, 0, bytes);
            if (ce == cudaSuccess) ce = cudaMemset(gpu_buf, 0x5A, bytes);
            break;
        default:
            fprintf(stderr, "[nixl] unsupported cudaMemcpyKind=%d\n", (int)kind);
            std::exit(2);
    }
    if (ce != cudaSuccess) {
        fprintf(stderr, "[nixl] buffer alloc failed: %s\n", cudaGetErrorString(ce));
        std::exit(2);
    }

    nixl_reg_dlist_t initiator_regs(host_mem);
    nixl_reg_dlist_t target_regs(gpu_mem);
    initiator_regs.addDesc(make_nixl_blob_desc(host_buf, bytes, 0));
    target_regs.addDesc(make_nixl_blob_desc(gpu_buf, bytes, (uint64_t)gpu_id));

    NixlLoopbackPair pair(gpu_id, nic_name);
    pair.register_buffers(initiator_regs, target_regs);

    nixl_xfer_dlist_t local_descs(host_mem);
    nixl_xfer_dlist_t remote_descs(gpu_mem);
    local_descs.addDesc(make_nixl_basic_desc(host_buf, bytes, 0));
    remote_descs.addDesc(make_nixl_basic_desc(gpu_buf, bytes, (uint64_t)gpu_id));

    // Pre-create handles; reused across warmup and measured phases.
    const int n_handles = std::max(warmup, iters);
    std::vector<nixlXferReqH*> handles(n_handles, nullptr);
    for (int i = 0; i < n_handles; ++i)
        handles[i] = pair.create_request(op, local_descs, remote_descs);

    // Warmup: sequential post+wait, no measurement.
    for (int i = 0; i < warmup; ++i) {
        pair.post_request(handles[i]);
        pair.wait_request(handles[i]);
    }

    // Measured: post all (record per-post issue latency), then wait all.
    // Total time covers the full post→wait window for bandwidth.
    double t0 = now_us();
    for (int i = 0; i < iters; ++i) {
        double ti = now_us();
        nixl_status_t rc = pair.post_request(handles[i]);
        issue_samples.push_back(now_us() - ti);
        if (rc != NIXL_SUCCESS && rc != NIXL_IN_PROG) {
            fprintf(stderr, "[nixl] postXferReq failed iter=%d rc=%d\n", i, (int)rc);
            std::exit(2);
        }
    }
    for (int i = 0; i < iters; ++i) {
        nixl_status_t rc = pair.wait_request(handles[i]);
        if (rc != NIXL_SUCCESS) {
            fprintf(stderr, "[nixl] wait failed iter=%d rc=%d\n", i, (int)rc);
            std::exit(2);
        }
    }
    double total_us = now_us() - t0;

    for (auto* h : handles) pair.release_request(h);
    pair.deregister_buffers(initiator_regs, target_regs);
    cudaFreeHost(host_buf);
    cudaFree(gpu_buf);

    BenchPair out{};
    out.issue = analyse(issue_samples, bytes);
    out.transfer.median_us = total_us;
    out.transfer.p99_us    = total_us;
    out.transfer.bw_GBs    = (iters > 0 && total_us > 0.0)
                               ? ((double)bytes * iters / 1e9) / (total_us / 1e6)
                               : 0.0;
    return out;
}

// Main benchmark entry.
int main(int argc, char** argv)
{
    int         gpu_id   = (argc > 1) ? std::atoi(argv[1]) : 0;
    std::string  nic_name = (argc > 2) ? argv[2]            : "mlx5_0";

    // Force UCX to use GPUDirect (gdr_copy) path: NIC DMA engine reads/writes
    // GPU memory directly via nvidia-peermem, bypassing CPU on the data path.
    // Must be set before any UCX/NIXL context is created.
    // Force UCX to use RC RDMA with GPUDirect (nvidia-peermem):
    // NIC DMA engine writes directly into GPU memory — host→NIC→GPU path.
    // This mirrors the ibverbs RDMA_WRITE(pinned_host lkey → GPU rkey) in gdr_copy.cpp.
    // Do not force UCX to pure IB transports only. NIXL validates CUDA support
    // from the UCX context memory_types bitmask, and a too-narrow UCX_TLS such as
    // "rc_x" filters CUDA memory components out of the context. That makes VRAM
    // appear as host memory even when UCX was built with CUDA support.
    //
    // Keep the default broad enough to retain CUDA memory registration support.
    // gdr_copy is not forced here because it is optional and absent on many
    // systems unless gdrcopy is installed and UCX was built against it.
    setenv("UCX_TLS",                "rc_x,cuda", 0);
    setenv("UCX_IB_GPU_DIRECT_RDMA", "yes",            0);
    setenv("UCX_RNDV_THRESH",        "0",              0);
    setenv("UCX_ZCOPY_THRESH",       "0",              0);

    printf("=================================================================\n");
    printf("  GDR Copy Benchmark  -  GPU %d  NIC %s\n", gpu_id, nic_name.c_str());
    printf("=================================================================\n\n");

    // CUDA setup.
    cudaSetDevice(gpu_id);
    struct cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, gpu_id);
    printf("GPU: %s  (PCIe gen%d x%d)\n\n",
           prop.name, prop.pciBusID, prop.pciDeviceID);

    // Open GDR channel.
    std::shared_ptr<GDRCopyChannel> ch;
    try {
        ch = GDRCopyLib::open(gpu_id, nic_name);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to open GDR channel: %s\n", e.what());
        return 1;
    }

    GDRStats s = ch->stats();
    bool gdr_active = (s.fallback_ops == 0);
    printf("GPUDirect RDMA path: %s\n\n", gdr_active ? "ACTIVE" : "FALLBACK (cudaMemcpy)");
    printf("Benchmark mode: NIXL-UCX issue latency + wait-all bandwidth\n\n");

    // Transfer sizes to sweep.
    std::vector<size_t> sizes;
    for (size_t s = 4096; s <= 64ULL << 20; s *= 4)
        sizes.push_back(s);

    static const int WARMUP = 100;
    static const int ITERS  = 1000;

    std::vector<DirectionRow> h2d_rows;
    h2d_rows.reserve(sizes.size());

    for (size_t bytes : sizes) {
        void* h_src = nullptr;
        void* d_dst = nullptr;
        cudaHostAlloc(&h_src, bytes, cudaHostAllocPortable);
        cudaMalloc(&d_dst, bytes);
        cudaMemset(d_dst, 0, bytes);
        memset(h_src, 0xA5, bytes);

        BenchPair gdr  = run_gdr_timings(ch, d_dst, h_src, bytes, GDR_H2D, WARMUP, ITERS);
        BenchPair nixl = run_nixl_timings(bytes, cudaMemcpyHostToDevice,
                                          WARMUP, ITERS, gpu_id, nic_name);

        h2d_rows.push_back(DirectionRow{bytes, gdr, nixl});

        cudaFreeHost(h_src);
        cudaFree(d_dst);
    }

    print_latency_table("Host->Device Issue Latency", h2d_rows, true);
    print_latency_table("Host->Device Bandwidth", h2d_rows, false);

    // Reopen channel before D2H sweep to avoid stale GPU MR reuse.
    GDRCopyLib::shutdown();
    try {
        ch = GDRCopyLib::open(gpu_id, nic_name);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to reopen GDR channel for D2H issue bench: %s\n", e.what());
        return 1;
    }

    std::vector<DirectionRow> d2h_rows;
    d2h_rows.reserve(sizes.size());

    for (size_t bytes : sizes) {
        void* d_src = nullptr;
        void* h_dst = nullptr;
        cudaMalloc(&d_src, bytes);
        cudaHostAlloc(&h_dst, bytes, cudaHostAllocPortable);
        cudaMemset(d_src, 0x5A, bytes);
        memset(h_dst, 0, bytes);

        BenchPair gdr  = run_gdr_timings(ch, h_dst, d_src, bytes, GDR_D2H, WARMUP, ITERS);
        BenchPair nixl = run_nixl_timings(bytes, cudaMemcpyDeviceToHost,
                                          WARMUP, ITERS, gpu_id, nic_name);

        d2h_rows.push_back(DirectionRow{bytes, gdr, nixl});

        cudaFree(d_src);
        cudaFreeHost(h_dst);
    }

    print_latency_table("Device->Host Issue Latency", d2h_rows, true);
    print_latency_table("Device->Host Bandwidth", d2h_rows, false);

    // Summary.
    GDRStats final_s = ch->stats();
    printf("\n=================================================================\n");
    printf("Total ops: %lu  (RDMA: %lu  Fallback: %lu)\n",
           final_s.total_ops, final_s.rdma_ops, final_s.fallback_ops);
    printf("Total bytes: %.2f GiB\n", final_s.total_bytes / (double)(1ULL<<30));
    printf("=================================================================\n");

    GDRCopyLib::shutdown();
    return 0;
}
