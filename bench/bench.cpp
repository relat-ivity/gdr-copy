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
 *     - GB/s only for transfer-completion tables
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

// Batch issue + batch sync；只测总时间并换算带宽。
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

        // 尽可能连续下发；若 SQ/CQ 满则先回收一部分完成再继续下发。
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
                // 发生背压：先等至少一个完成，给后续下发腾位置。
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

        // 把剩余未完成的请求全部 drain 掉。
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
               "Size", "   GDR (median / p99)  ", "   CUDA (median / p99)");
        printf("%-12s-+-%-21s-+-%-21s\n",
               "------------", "-----------------------", "-----------------------");
    } else {
        printf("%-12s | %-18s | %-18s\n",
               "Size", "    GDR (BW)     ", "    CUDA (BW)    ");
        printf("%-12s-+-%-18s-+-%-18s\n",
               "------------", "------------------", "------------------");
    }

    for (const auto& row : rows) {
        const BenchResult& g = issue_table ? row.gdr.issue : row.gdr.transfer;
        const BenchResult& c = issue_table ? row.cuda.issue : row.cuda.transfer;
        char size_str[32];
        format_size(row.bytes, size_str, sizeof(size_str));

        if (issue_table) {
            printf("%-12s | %7.2f µs / %7.2f µs | %7.2f µs / %7.2f µs\n",
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

static BenchPair run_cuda_timings(void* dst, const void* src,
                                  size_t bytes, cudaMemcpyKind kind,
                                  int warmup, int iters, cudaStream_t stream)
{
    std::vector<double> issue_samples;
    issue_samples.reserve(iters);

    // 先测纯下发耗时：每次下发后立刻同步，避免队列背压把等待时间算进 issue。
    for (int i = 0; i < warmup + iters; ++i) {
        double t0 = now_us();
        cudaError_t ce = cudaMemcpyAsync(dst, src, bytes, kind, stream);
        double t1 = now_us();
        if (ce != cudaSuccess) {
            fprintf(stderr, "[issue] cudaMemcpyAsync failed: %s i=%d/%d\n",
                    cudaGetErrorString(ce), i, warmup + iters);
            std::exit(2);
        }
        ce = cudaStreamSynchronize(stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[issue] cudaStreamSynchronize failed: %s\n", cudaGetErrorString(ce));
            std::exit(2);
        }
        if (i >= warmup) issue_samples.push_back(t1 - t0);
    }

    auto run_batch = [&](int count, bool measure) -> double {
        if (count <= 0) return 0.0;

        double t0 = 0.0;
        if (measure) t0 = now_us();

        // 一次性下发本批次全部异步 memcpy
        for (int i = 0; i < count; ++i) {
            cudaError_t ce = cudaMemcpyAsync(dst, src, bytes, kind, stream);
            if (ce != cudaSuccess) {
                fprintf(stderr, "[issue] cudaMemcpyAsync failed: %s i=%d/%d\n",
                        cudaGetErrorString(ce), i, count);
                std::exit(2);
            }
        }

        // 一次 sync 等待全部完成
        cudaError_t ce = cudaStreamSynchronize(stream);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[issue] cudaStreamSynchronize failed: %s\n", cudaGetErrorString(ce));
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

static void prime_gdr_window(std::shared_ptr<GDRCopyChannel> ch,
                             void* dst, const void* src,
                             size_t bytes, GDRCopyKind kind)
{
    // 先用最大块做一次非统计提交，让 channel 把后续要复用的 host/GPU buffer
    // 注册成固定 MR window。后面的尺寸 sweep 都只使用这块 buffer 的前缀。
    int rc = ch->memcpy_async(dst, src, bytes, kind);
    if (rc != 0) {
        fprintf(stderr, "[prime] gdr memcpy_async failed: rc=%d kind=%d bytes=%zu\n",
                rc, (int)kind, bytes);
        std::exit(2);
    }

    while (true) {
        int sc = ch->sync();
        if (sc == 0)
            break;
        if (sc == -EAGAIN)
            continue;
        fprintf(stderr, "[prime] gdr sync failed: rc=%d kind=%d bytes=%zu\n",
                sc, (int)kind, bytes);
        std::exit(2);
    }
    ch->reset_stats();
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
    printf("Benchmark mode: async issue latency (isolated) + sync-drain bandwidth\n\n");

    // ── Transfer sizes to sweep ───────────────────────────────────────────
    std::vector<size_t> sizes;
    for (size_t s = 4096; s <= 64ULL << 20; s *= 4)
        sizes.push_back(s);
    const size_t max_bytes = sizes.back();

    static const int WARMUP = 100;
    static const int ITERS  = 10000;

    // 输出异步下发时延和带宽。
    cudaStream_t issue_stream{};
    cudaStreamCreate(&issue_stream);

    std::vector<DirectionRow> h2d_rows;
    h2d_rows.reserve(sizes.size());

    void* h_src = nullptr;
    void* d_dst = nullptr;
    cudaHostAlloc(&h_src, max_bytes, cudaHostAllocPortable);
    cudaMalloc(&d_dst, max_bytes);
    cudaMemset(d_dst, 0, max_bytes);
    memset(h_src, 0xA5, max_bytes);

    prime_gdr_window(ch, d_dst, h_src, max_bytes, GDR_H2D);

    for (size_t bytes : sizes) {
        BenchPair gdr  = run_gdr_timings(ch, d_dst, h_src, bytes, GDR_H2D, WARMUP, ITERS);
        BenchPair cuda = run_cuda_timings(d_dst, h_src, bytes, cudaMemcpyHostToDevice,
                                          WARMUP, ITERS, issue_stream);

        h2d_rows.push_back(DirectionRow{bytes, gdr, cuda});
    }

    cudaFreeHost(h_src);
    cudaFree(d_dst);

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

    void* d_src = nullptr;
    void* h_dst = nullptr;
    cudaMalloc(&d_src, max_bytes);
    cudaHostAlloc(&h_dst, max_bytes, cudaHostAllocPortable);
    cudaMemset(d_src, 0x5A, max_bytes);
    memset(h_dst, 0, max_bytes);

    prime_gdr_window(ch, h_dst, d_src, max_bytes, GDR_D2H);

    for (size_t bytes : sizes) {
        BenchPair gdr  = run_gdr_timings(ch, h_dst, d_src, bytes, GDR_D2H, WARMUP, ITERS);
        BenchPair cuda = run_cuda_timings(h_dst, d_src, bytes, cudaMemcpyDeviceToHost,
                                          WARMUP, ITERS, issue_stream);

        d2h_rows.push_back(DirectionRow{bytes, gdr, cuda});
    }

    cudaFree(d_src);
    cudaFreeHost(h_dst);

    print_latency_table("Device->Host Issue Latency", d2h_rows, true);
    print_latency_table("Device->Host Bandwidth", d2h_rows, false);

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
