/**
 * bench.cpp  —  GDR Copy vs cudaMemcpy latency/bandwidth benchmark
 *
 * Usage:
 *   sudo ./build/bench [gpu_id] [nic_name]
 *   sudo ./build/bench 0 mlx5_0
 *
 * Output:
 *   For each transfer size × direction, prints:
 *     - median latency (µs)
 *     - p99 latency (µs)
 *     - bandwidth (GB/s)
 *   for both GDR RDMA path and cudaMemcpy baseline.
 *
 * Why sudo?
 *   Accessing PCIe config space for GPUDirect registration may require
 *   CAP_NET_ADMIN or CAP_SYS_RAWIO on some distros. Alternatively, set
 *   /proc/sys/kernel/perf_event_paranoid appropriately.
 */

#include "gdr_copy.h"

#include <cuda_runtime.h>
#include <algorithm>
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

// ── run one benchmark cell ────────────────────────────────────────────────────
static BenchResult run_gdr(std::shared_ptr<GDRCopyChannel> ch,
                            void* dst, const void* src,
                            size_t bytes, GDRCopyKind kind,
                            int warmup, int iters)
{
    ch->reset_stats();
    std::vector<double> samples;
    samples.reserve(iters);

    for (int i = 0; i < warmup + iters; i++) {
        double t0 = now_us();
        ch->memcpy(dst, src, bytes, kind);
        double dt = now_us() - t0;
        if (i >= warmup) samples.push_back(dt);
    }
    return analyse(samples, bytes);
}

static BenchResult run_cuda(void* dst, const void* src,
                             size_t bytes, cudaMemcpyKind kind,
                             int warmup, int iters)
{
    std::vector<double> samples;
    samples.reserve(iters);

    for (int i = 0; i < warmup + iters; i++) {
        double t0 = now_us();
        cudaMemcpy(dst, src, bytes, kind);
        double dt = now_us() - t0;
        if (i >= warmup) samples.push_back(dt);
    }
    return analyse(samples, bytes);
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

    // ── Transfer sizes to sweep ───────────────────────────────────────────
    std::vector<size_t> sizes;
    for (size_t s = 4096; s <= 64ULL << 20; s *= 4)
        sizes.push_back(s);

    static const int WARMUP = 10;
    static const int ITERS  = 100;

    // ── H2D benchmark ─────────────────────────────────────────────────────
    printf("--- Host→Device (H2D) ---\n");
    printf("%-12s | %-28s | %-28s\n",
           "Size", "  GDR (median / p99 / BW)", "  CUDA (median / p99 / BW)");
    printf("%-12s-+-%-28s-+-%-28s\n",
           "------------", "----------------------------", "----------------------------");

    for (size_t bytes : sizes) {
        void* h_src = nullptr;
        void* d_dst = nullptr;
        h_src = std::malloc(bytes);
        if (!h_src) {
            fprintf(stderr, "malloc failed for H2D host buffer (%zu bytes)\n", bytes);
            return 1;
        }
        cudaMalloc(&d_dst, bytes);
        cudaMemset(d_dst, 0, bytes);
        memset(h_src, 0xAB, bytes);

        BenchResult gdr  = run_gdr(ch, d_dst, h_src, bytes, GDR_H2D, WARMUP, ITERS);
        BenchResult cuda = run_cuda(d_dst, h_src, bytes, cudaMemcpyHostToDevice, WARMUP, ITERS);

        char size_str[32];
        if (bytes < (1 << 10))      snprintf(size_str, sizeof(size_str), "%zuB",    bytes);
        else if (bytes < (1 << 20)) snprintf(size_str, sizeof(size_str), "%zuKiB",  bytes >> 10);
        else                        snprintf(size_str, sizeof(size_str), "%zuMiB",  bytes >> 20);

        printf("%-12s | %7.2f µs / %7.2f µs / %5.2f GB/s | "
               "%7.2f µs / %7.2f µs / %5.2f GB/s\n",
               size_str,
               gdr.median_us,  gdr.p99_us,  gdr.bw_GBs,
               cuda.median_us, cuda.p99_us, cuda.bw_GBs);

        std::free(h_src);
        cudaFree(d_dst);
    }

    // ── D2H benchmark ─────────────────────────────────────────────────────
    GDRCopyLib::shutdown();
    try {
        ch = GDRCopyLib::open(gpu_id, nic_name);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to reopen GDR channel for D2H: %s\n", e.what());
        return 1;
    }

    printf("\n--- Device→Host (D2H) ---\n");
    printf("%-12s | %-28s | %-28s\n",
           "Size", "  GDR (median / p99 / BW)", "  CUDA (median / p99 / BW)");
    printf("%-12s-+-%-28s-+-%-28s\n",
           "------------", "----------------------------", "----------------------------");

    for (size_t bytes : sizes) {
        void* d_src = nullptr;
        void* h_dst = nullptr;
        cudaMalloc(&d_src, bytes);
        h_dst = std::malloc(bytes);
        if (!h_dst) {
            fprintf(stderr, "malloc failed for D2H host buffer (%zu bytes)\n", bytes);
            cudaFree(d_src);
            return 1;
        }
        cudaMemset(d_src, 0xCD, bytes);
        memset(h_dst, 0, bytes);

        BenchResult gdr  = run_gdr(ch, h_dst, d_src, bytes, GDR_D2H, WARMUP, ITERS);
        BenchResult cuda = run_cuda(h_dst, d_src, bytes, cudaMemcpyDeviceToHost, WARMUP, ITERS);

        char size_str[32];
        if (bytes < (1 << 10))      snprintf(size_str, sizeof(size_str), "%zuB",    bytes);
        else if (bytes < (1 << 20)) snprintf(size_str, sizeof(size_str), "%zuKiB",  bytes >> 10);
        else                        snprintf(size_str, sizeof(size_str), "%zuMiB",  bytes >> 20);

        printf("%-12s | %7.2f µs / %7.2f µs / %5.2f GB/s | "
               "%7.2f µs / %7.2f µs / %5.2f GB/s\n",
               size_str,
               gdr.median_us,  gdr.p99_us,  gdr.bw_GBs,
               cuda.median_us, cuda.p99_us, cuda.bw_GBs);

        cudaFree(d_src);
        std::free(h_dst);
    }

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
