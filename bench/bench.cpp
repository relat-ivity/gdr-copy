/**
 * bench.cpp  —  GDR Copy vs cudaMemcpy async timing benchmark
 *
 * Usage:
 *   sudo ./build/bench [gpu_id] [nic_name] [threads]
 *   sudo ./build/bench 0 mlx5_0
 *   sudo ./build/bench 0 mlx5_0 8
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
#include <thread>

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
    if (issue_table) {
        printf("%-12s | %-21s | %-21s\n",
               "Size", "   GDR (median / p99)  ", "   CUDA (median / p99)");
        printf("%-12s-+-%-21s-+-%-21s\n",
               "------------", "-----------------------", "-----------------------");
    } else {
        printf("%-12s | %-28s | %-28s\n",
               "Size", "      GDR (median / p99 / BW)       ", "      CUDA (median / p99 / BW)");
        printf("%-12s-+-%-28s-+-%-28s\n",
               "------------", "------------------------------------", "------------------------------------");
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
            printf("%-12s | %7.2f µs / %7.2f µs / %5.2f GB/s | "
                   "%7.2f µs / %7.2f µs / %5.2f GB/s\n",
                   size_str,
                   g.median_us, g.p99_us, g.bw_GBs,
                   c.median_us, c.p99_us, c.bw_GBs);
        }
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

static BenchResult weighted_average_results(const std::vector<BenchResult>& items,
                                            const std::vector<int>& weights,
                                            bool include_bw)
{
    if (items.size() != weights.size()) {
        fprintf(stderr, "weighted_average_results size mismatch: items=%zu weights=%zu\n",
                items.size(), weights.size());
        std::exit(2);
    }
    double sum_w = 0.0;
    BenchResult out{};
    for (size_t i = 0; i < items.size(); ++i) {
        if (weights[i] <= 0) continue;
        const double w = (double)weights[i];
        out.median_us += items[i].median_us * w;
        out.p99_us    += items[i].p99_us * w;
        if (include_bw)
            out.bw_GBs += items[i].bw_GBs * w;
        sum_w += w;
    }
    if (sum_w <= 0.0) {
        fprintf(stderr, "weighted_average_results received no positive weights\n");
        std::exit(2);
    }
    out.median_us /= sum_w;
    out.p99_us    /= sum_w;
    if (include_bw)
        out.bw_GBs /= sum_w;
    else
        out.bw_GBs = 0.0;
    return out;
}

static BenchPair merge_thread_pairs(const std::vector<BenchPair>& pairs,
                                    const std::vector<int>& iters_per_lane)
{
    std::vector<BenchResult> issues;
    std::vector<BenchResult> transfers;
    issues.reserve(pairs.size());
    transfers.reserve(pairs.size());
    for (const auto& p : pairs) {
        issues.push_back(p.issue);
        transfers.push_back(p.transfer);
    }
    BenchPair out{};
    out.issue = weighted_average_results(issues, iters_per_lane, false);
    out.transfer = weighted_average_results(transfers, iters_per_lane, true);
    return out;
}

static BenchPair run_gdr_parallel_timings(int gpu_id,
                                          const std::vector<std::shared_ptr<GDRCopyChannel>>& channels,
                                          const std::vector<void*>& dsts,
                                          const std::vector<void*>& srcs,
                                          size_t bytes, GDRCopyKind kind,
                                          int warmup,
                                          const std::vector<int>& iters_per_lane)
{
    const size_t lanes = channels.size();
    if (lanes == 0 || dsts.size() != lanes || srcs.size() != lanes || iters_per_lane.size() != lanes) {
        fprintf(stderr, "Invalid GDR threaded setup: lanes=%zu dst=%zu src=%zu\n",
                lanes, dsts.size(), srcs.size());
        std::exit(2);
    }

    std::vector<BenchPair> lane_results(lanes);
    std::vector<std::thread> workers;
    workers.reserve(lanes);

    for (size_t lane = 0; lane < lanes; ++lane) {
        if (iters_per_lane[lane] <= 0)
            continue;
        workers.emplace_back([&, lane]() {
            cudaError_t ce = cudaSetDevice(gpu_id);
            if (ce != cudaSuccess) {
                fprintf(stderr, "cudaSetDevice failed in GDR worker lane=%zu: %s\n",
                        lane, cudaGetErrorString(ce));
                std::exit(2);
            }
            lane_results[lane] = run_gdr_timings(channels[lane], dsts[lane], srcs[lane],
                                                 bytes, kind, warmup, iters_per_lane[lane]);
        });
    }

    for (auto& t : workers) t.join();
    return merge_thread_pairs(lane_results, iters_per_lane);
}

static BenchPair run_cuda_parallel_timings(int gpu_id,
                                           const std::vector<void*>& dsts,
                                           const std::vector<void*>& srcs,
                                           size_t bytes, cudaMemcpyKind kind,
                                           int warmup,
                                           const std::vector<int>& iters_per_lane,
                                           const std::vector<cudaStream_t>& streams)
{
    const size_t lanes = streams.size();
    if (lanes == 0 || dsts.size() != lanes || srcs.size() != lanes || iters_per_lane.size() != lanes) {
        fprintf(stderr, "Invalid CUDA threaded setup: lanes=%zu dst=%zu src=%zu\n",
                lanes, dsts.size(), srcs.size());
        std::exit(2);
    }

    std::vector<BenchPair> lane_results(lanes);
    std::vector<std::thread> workers;
    workers.reserve(lanes);

    for (size_t lane = 0; lane < lanes; ++lane) {
        if (iters_per_lane[lane] <= 0)
            continue;
        workers.emplace_back([&, lane]() {
            cudaError_t ce = cudaSetDevice(gpu_id);
            if (ce != cudaSuccess) {
                fprintf(stderr, "cudaSetDevice failed in CUDA worker lane=%zu: %s\n",
                        lane, cudaGetErrorString(ce));
                std::exit(2);
            }
            lane_results[lane] = run_cuda_timings(dsts[lane], srcs[lane], bytes, kind,
                                                  warmup, iters_per_lane[lane], streams[lane]);
        });
    }

    for (auto& t : workers) t.join();
    return merge_thread_pairs(lane_results, iters_per_lane);
}

static std::vector<std::shared_ptr<GDRCopyChannel>>
open_channel_pool(int gpu_id, const std::string& nic_name, int lanes)
{
    std::vector<std::shared_ptr<GDRCopyChannel>> channels;
    channels.reserve((size_t)lanes);
    for (int i = 0; i < lanes; ++i) {
        channels.push_back(GDRCopyLib::open(gpu_id, nic_name));
        // open() is cached by (gpu,nic); clear cache so next lane gets a distinct channel.
        GDRCopyLib::shutdown();
    }
    return channels;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    int         gpu_id   = (argc > 1) ? std::atoi(argv[1]) : 0;
    std::string  nic_name = (argc > 2) ? argv[2]            : "mlx5_0";
    int         threads   = (argc > 3) ? std::atoi(argv[3]) : 8;
    if (threads < 1) threads = 1;
    static const int WARMUP = 10;
    static const int TOTAL_ITERS = 100;
    int active_lanes = std::min(threads, TOTAL_ITERS);
    if (active_lanes < 1) active_lanes = 1;

    std::vector<int> iters_per_lane((size_t)active_lanes, TOTAL_ITERS / active_lanes);
    for (int i = 0; i < (TOTAL_ITERS % active_lanes); ++i)
        iters_per_lane[(size_t)i] += 1;

    printf("=================================================================\n");
    printf("  GDR Copy Benchmark  —  GPU %d  NIC %s\n", gpu_id, nic_name.c_str());
    printf("=================================================================\n\n");

    // ── CUDA setup ────────────────────────────────────────────────────────
    cudaSetDevice(gpu_id);
    struct cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, gpu_id);
    printf("GPU: %s  (PCIe gen%d x%d)\n\n",
           prop.name, prop.pciBusID, prop.pciDeviceID);

    // ── Open GDR channel pool (one channel per active worker lane) ───────
    std::vector<std::shared_ptr<GDRCopyChannel>> channels;
    try {
        channels = open_channel_pool(gpu_id, nic_name, active_lanes);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to open GDR channel pool: %s\n", e.what());
        return 1;
    }
    if (channels.empty()) {
        fprintf(stderr, "No GDR channels created\n");
        return 1;
    }

    GDRStats s = channels[0]->stats();
    bool gdr_active = (s.fallback_ops == 0);
    printf("GPUDirect RDMA path: %s\n\n", gdr_active ? "ACTIVE" : "FALLBACK (cudaMemcpy)");
    printf("Benchmark mode: threaded lanes (requested=%d, active=%d), each lane has its own channel/stream/buffer\n",
           threads, active_lanes);
    printf("Per-lane warmup=%d, total measured samples=%d (evenly split)\n\n",
           WARMUP, TOTAL_ITERS);

    // ── Transfer sizes to sweep ───────────────────────────────────────────
    std::vector<size_t> sizes;
    for (size_t s = 4096; s <= 64ULL << 20; s *= 4)
        sizes.push_back(s);

    // Async benchmark: one stream per worker lane.
    std::vector<cudaStream_t> issue_streams((size_t)active_lanes, nullptr);
    for (int lane = 0; lane < active_lanes; ++lane) {
        cudaError_t ce = cudaStreamCreate(&issue_streams[(size_t)lane]);
        if (ce != cudaSuccess) {
            fprintf(stderr, "cudaStreamCreate failed lane=%d: %s\n",
                    lane, cudaGetErrorString(ce));
            return 1;
        }
    }

    std::vector<DirectionRow> h2d_rows;
    h2d_rows.reserve(sizes.size());

    for (size_t bytes : sizes) {
        std::vector<void*> h_src((size_t)active_lanes, nullptr);
        std::vector<void*> d_dst((size_t)active_lanes, nullptr);
        for (int lane = 0; lane < active_lanes; ++lane) {
            cudaHostAlloc(&h_src[(size_t)lane], bytes, cudaHostAllocPortable);
            cudaMalloc(&d_dst[(size_t)lane], bytes);
            cudaMemset(d_dst[(size_t)lane], 0, bytes);
            memset(h_src[(size_t)lane], 0xA5, bytes);
        }

        BenchPair gdr  = run_gdr_parallel_timings(gpu_id, channels, d_dst, h_src, bytes, GDR_H2D,
                                                  WARMUP, iters_per_lane);
        BenchPair cuda = run_cuda_parallel_timings(gpu_id, d_dst, h_src, bytes, cudaMemcpyHostToDevice,
                                                   WARMUP, iters_per_lane, issue_streams);

        h2d_rows.push_back(DirectionRow{bytes, gdr, cuda});

        for (int lane = 0; lane < active_lanes; ++lane) {
            cudaFreeHost(h_src[(size_t)lane]);
            cudaFree(d_dst[(size_t)lane]);
        }
    }

    print_latency_table("Host->Device Issue Latency", h2d_rows, true);
    print_latency_table("Host->Device Transfer Latency", h2d_rows, false);

    // Reopen channel before D2H sweep to avoid stale GPU MR reuse.
    for (auto& ch : channels) ch.reset();
    channels.clear();
    GDRCopyLib::shutdown();
    try {
        channels = open_channel_pool(gpu_id, nic_name, active_lanes);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to reopen GDR channel for D2H issue bench: %s\n", e.what());
        return 1;
    }

    std::vector<DirectionRow> d2h_rows;
    d2h_rows.reserve(sizes.size());

    for (size_t bytes : sizes) {
        std::vector<void*> d_src((size_t)active_lanes, nullptr);
        std::vector<void*> h_dst((size_t)active_lanes, nullptr);
        for (int lane = 0; lane < active_lanes; ++lane) {
            cudaMalloc(&d_src[(size_t)lane], bytes);
            cudaHostAlloc(&h_dst[(size_t)lane], bytes, cudaHostAllocPortable);
            cudaMemset(d_src[(size_t)lane], 0x5A, bytes);
            memset(h_dst[(size_t)lane], 0, bytes);
        }

        BenchPair gdr  = run_gdr_parallel_timings(gpu_id, channels, h_dst, d_src, bytes, GDR_D2H,
                                                  WARMUP, iters_per_lane);
        BenchPair cuda = run_cuda_parallel_timings(gpu_id, h_dst, d_src, bytes, cudaMemcpyDeviceToHost,
                                                   WARMUP, iters_per_lane, issue_streams);

        d2h_rows.push_back(DirectionRow{bytes, gdr, cuda});

        for (int lane = 0; lane < active_lanes; ++lane) {
            cudaFree(d_src[(size_t)lane]);
            cudaFreeHost(h_dst[(size_t)lane]);
        }
    }

    print_latency_table("Device->Host Issue Latency", d2h_rows, true);
    print_latency_table("Device->Host Transfer Latency", d2h_rows, false);

    for (auto s_handle : issue_streams) {
        cudaStreamDestroy(s_handle);
    }

    // ── Summary ───────────────────────────────────────────────────────────
    GDRStats final_s{};
    for (const auto& ch : channels) {
        GDRStats lane_s = ch->stats();
        final_s.total_ops    += lane_s.total_ops;
        final_s.rdma_ops     += lane_s.rdma_ops;
        final_s.fallback_ops += lane_s.fallback_ops;
        final_s.total_bytes  += lane_s.total_bytes;
    }
    printf("\n=================================================================\n");
    printf("Total ops: %lu  (RDMA: %lu  Fallback: %lu)\n",
           final_s.total_ops, final_s.rdma_ops, final_s.fallback_ops);
    printf("Total bytes: %.2f GiB\n", final_s.total_bytes / (double)(1ULL<<30));
    printf("=================================================================\n");

    GDRCopyLib::shutdown();
    return 0;
}
