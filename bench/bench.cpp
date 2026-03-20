/**
 * bench.cpp  —  GDR Copy vs cudaMemcpy async timing benchmark
 *
 * Usage:
 *   sudo ./build/bench [gpu_id] [nic_name]
 *   sudo ./build/bench 0 mlx5_0
 *
 * Output:
 *   For each transfer size × direction, prints:
 *     - submit (issue) latency
 *     - transfer-completion latency (issue return -> completion observed)
 *     - median latency (µs)
 *     - p99 latency (µs)
 *     - rate-derived GB/s for each latency bucket
 *   for both GDR RDMA path and cudaMemcpyAsync baseline.
 *
 * Why sudo?
 *   Accessing PCIe config space for GPUDirect registration may require
 *   CAP_NET_ADMIN or CAP_SYS_RAWIO on some distros. Alternatively, set
 *   /proc/sys/kernel/perf_event_paranoid appropriately.
 */

#include "gdr_copy.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>

// ── timing ────────────────────────────────────────────────────────────────────
static double now_us() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(
               high_resolution_clock::now().time_since_epoch()).count() / 1e3;
}

// ── statistics ────────────────────────────────────────────────────────────────
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
    BenchPair cuda{};
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

// Measure both submit latency and transfer-completion latency.
static BenchPair run_gdr_timings(std::shared_ptr<GDRCopyChannel> ch,
                                 void* dst, const void* src,
                                 size_t bytes, GDRCopyKind kind,
                                 int warmup, int iters)
{
    ch->reset_stats();
    std::vector<double> issue_samples;
    std::vector<double> transfer_samples;
    issue_samples.reserve(iters);
    transfer_samples.reserve(iters);

    for (int i = 0; i < warmup + iters; i++) {
        double t0 = now_us();
        int rc = ch->memcpy_async(dst, src, bytes, kind);
        double t1 = now_us();
        if (rc != 0) {
            fprintf(stderr, "[issue] gdr memcpy_async failed: rc=%d kind=%d bytes=%zu\n",
                    rc, (int)kind, bytes);
            std::exit(2);
        }
        while (true) {
            int sc = ch->sync();
            if (sc == 0) break;
            if (sc == -EAGAIN) continue;
            fprintf(stderr, "[issue] gdr sync failed: rc=%d kind=%d bytes=%zu\n",
                    sc, (int)kind, bytes);
            std::exit(2);
        }
        double t2 = now_us();
        if (i >= warmup) {
            issue_samples.push_back(t1 - t0);
            transfer_samples.push_back(t2 - t1);
        }
    }
    BenchPair out{};
    out.issue = analyse(issue_samples, bytes);
    out.transfer = analyse(transfer_samples, bytes);
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
    printf("%-12s | %-28s | %-28s\n",
           "Size", "  GDR (median / p99 / BW)", "  CUDA (median / p99 / BW)");
    printf("%-12s-+-%-28s-+-%-28s\n",
           "------------", "----------------------------", "----------------------------");

    for (const auto& row : rows) {
        const BenchResult& g = issue_table ? row.gdr.issue : row.gdr.transfer;
        const BenchResult& c = issue_table ? row.cuda.issue : row.cuda.transfer;
        char size_str[32];
        format_size(row.bytes, size_str, sizeof(size_str));

        printf("%-12s | %7.2f µs / %7.2f µs / %5.2f GB/s | "
               "%7.2f µs / %7.2f µs / %5.2f GB/s\n",
               size_str,
               g.median_us, g.p99_us, g.bw_GBs,
               c.median_us, c.p99_us, c.bw_GBs);
    }
}

static BenchPair run_cuda_timings(void* dst, const void* src,
                                  size_t bytes, cudaMemcpyKind kind,
                                  int warmup, int iters, cudaStream_t stream)
{
    std::vector<double> issue_samples;
    std::vector<double> transfer_samples;
    issue_samples.reserve(iters);
    transfer_samples.reserve(iters);

    for (int i = 0; i < warmup + iters; i++) {
        double t0 = now_us();
        cudaError_t ce = cudaMemcpyAsync(dst, src, bytes, kind, stream);
        double t1 = now_us();
        if (ce != cudaSuccess) {
            fprintf(stderr, "[issue] cudaMemcpyAsync failed: %s\n", cudaGetErrorString(ce));
            std::exit(2);
        }
        ce = cudaStreamSynchronize(stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[issue] cudaStreamSynchronize failed: %s\n", cudaGetErrorString(ce));
            std::exit(2);
        }
        double t2 = now_us();
        if (i >= warmup) {
            issue_samples.push_back(t1 - t0);
            transfer_samples.push_back(t2 - t1);
        }
    }
    BenchPair out{};
    out.issue = analyse(issue_samples, bytes);
    out.transfer = analyse(transfer_samples, bytes);
    return out;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    int         gpu_id   = (argc > 1) ? std::atoi(argv[1]) : 0;
    std::string  nic_name = (argc > 2) ? argv[2]            : "mlx5_0";

    printf("=================================================================\n");
    printf("  GDR Copy Benchmark  —  GPU %d  NIC %s\n", gpu_id, nic_name.c_str());
    printf("=================================================================\n\n");

    // ── CUDA setup ────────────────────────────────────────────────────────
    cudaSetDevice(gpu_id);
    struct cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, gpu_id);
    printf("GPU: %s  (PCIe gen%d x%d)\n\n",
           prop.name, prop.pciBusID, prop.pciDeviceID);

    // ── Open GDR channel ──────────────────────────────────────────────────
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
    printf("Benchmark mode: issue + transfer timing (async submit/completion split)\n\n");

    // ── Transfer sizes to sweep ───────────────────────────────────────────
    std::vector<size_t> sizes;
    for (size_t s = 4096; s <= 64ULL << 20; s *= 4)
        sizes.push_back(s);

    static const int WARMUP = 10;
    static const int ITERS  = 100;

    // Async benchmark: split timing into issue and transfer-completion parts.
    cudaStream_t issue_stream{};
    cudaStreamCreate(&issue_stream);

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
        BenchPair cuda = run_cuda_timings(d_dst, h_src, bytes, cudaMemcpyHostToDevice,
                                          WARMUP, ITERS, issue_stream);

        h2d_rows.push_back(DirectionRow{bytes, gdr, cuda});

        cudaFreeHost(h_src);
        cudaFree(d_dst);
    }

    print_latency_table("Host->Device Issue Latency", h2d_rows, true);
    print_latency_table("Host->Device Transfer Latency", h2d_rows, false);

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
        BenchPair cuda = run_cuda_timings(h_dst, d_src, bytes, cudaMemcpyDeviceToHost,
                                          WARMUP, ITERS, issue_stream);

        d2h_rows.push_back(DirectionRow{bytes, gdr, cuda});

        cudaFree(d_src);
        cudaFreeHost(h_dst);
    }

    print_latency_table("Device->Host Issue Latency", d2h_rows, true);
    print_latency_table("Device->Host Transfer Latency", d2h_rows, false);

    cudaStreamDestroy(issue_stream);

    // ── Summary ───────────────────────────────────────────────────────────
    GDRStats final_s = ch->stats();
    printf("\n=================================================================\n");
    printf("Total ops: %lu  (RDMA: %lu  Fallback: %lu)\n",
           final_s.total_ops, final_s.rdma_ops, final_s.fallback_ops);
    printf("Total bytes: %.2f GiB\n", final_s.total_bytes / (double)(1ULL<<30));
    printf("=================================================================\n");

    GDRCopyLib::shutdown();
    return 0;
}
