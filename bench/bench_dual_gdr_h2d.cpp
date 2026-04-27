/**
 * bench_dual_gdr_h2d.cpp - dual-GPU GDR H2D aggregate bandwidth benchmark
 *
 * Usage:
 *   sudo ./build/bench_dual_gdr_h2d <gpu0> <gpu1> <nic_name>
 *   sudo ./build/bench_dual_gdr_h2d 4 5 mlx5_4
 *
 * This benchmark uses the same NIC for two GPUs at the same time. Each GPU has
 * its own GDRCopyChannel, host source buffer, and device destination buffer.
 * The reported total bandwidth is:
 *
 *   bytes * ITERS * 2 / shared_wall_clock_time
 */

#include "gdr_copy.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static constexpr int WARMUP = 100;
static constexpr int ITERS  = 1000;

static double now_us() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(
               high_resolution_clock::now().time_since_epoch()).count() / 1e3;
}

static void die_cuda(cudaError_t rc, const char* what, int gpu_id, size_t bytes) {
    if (rc == cudaSuccess)
        return;
    fprintf(stderr, "[dual] %s failed: gpu=%d bytes=%zu err=%s\n",
            what, gpu_id, bytes, cudaGetErrorString(rc));
    std::exit(2);
}

static void format_size(size_t bytes, char* out, size_t out_len) {
    if (bytes < (1 << 10))      snprintf(out, out_len, "%zuB", bytes);
    else if (bytes < (1 << 20)) snprintf(out, out_len, "%zuKiB", bytes >> 10);
    else                        snprintf(out, out_len, "%zuMiB", bytes >> 20);
}

// Submit count async H2D requests on one channel and drain all completions.
// This is intentionally the same backpressure model as the normal bench:
// keep posting until SQ/CQ is full, recycle one completion, then continue.
static double run_gdr_h2d_batch(const std::shared_ptr<GDRCopyChannel>& ch,
                                void* dst, const void* src,
                                size_t bytes, int count,
                                bool measure, int gpu_id)
{
    if (count <= 0)
        return 0.0;

    double t0 = 0.0;
    if (measure)
        t0 = now_us();

    int issued = 0;
    int done = 0;
    while (issued < count) {
        int rc = ch->memcpy_async(dst, src, bytes, GDR_H2D);
        if (rc == 0) {
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
                if (sc == -EAGAIN)
                    continue;
                fprintf(stderr,
                        "[dual] gdr sync failed: gpu=%d rc=%d bytes=%zu done=%d issued=%d\n",
                        gpu_id, sc, bytes, done, issued);
                std::exit(2);
            }
            continue;
        }
        fprintf(stderr,
                "[dual] gdr memcpy_async failed: gpu=%d rc=%d bytes=%zu issued=%d/%d\n",
                gpu_id, rc, bytes, issued, count);
        std::exit(2);
    }

    while (done < count) {
        int sc = ch->sync();
        if (sc == 0) {
            ++done;
            continue;
        }
        if (sc == -EAGAIN)
            continue;
        fprintf(stderr,
                "[dual] gdr sync failed: gpu=%d rc=%d bytes=%zu done=%d/%d\n",
                gpu_id, sc, bytes, done, count);
        std::exit(2);
    }

    return measure ? (now_us() - t0) : 0.0;
}

static void prime_gdr_h2d_window(const std::shared_ptr<GDRCopyChannel>& ch,
                                 void* dst, const void* src,
                                 size_t bytes, int gpu_id)
{
    int rc = ch->memcpy_async(dst, src, bytes, GDR_H2D);
    if (rc != 0) {
        fprintf(stderr,
                "[dual] gdr prime memcpy_async failed: gpu=%d rc=%d bytes=%zu\n",
                gpu_id, rc, bytes);
        std::exit(2);
    }

    while (true) {
        int sc = ch->sync();
        if (sc == 0)
            break;
        if (sc == -EAGAIN)
            continue;
        fprintf(stderr,
                "[dual] gdr prime sync failed: gpu=%d rc=%d bytes=%zu\n",
                gpu_id, sc, bytes);
        std::exit(2);
    }
    ch->reset_stats();
}

struct GpuLane {
    int gpu_id = -1;
    std::shared_ptr<GDRCopyChannel> channel;
    void* host_src = nullptr;
    void* device_dst = nullptr;
    double measured_us = 0.0;
};

static void allocate_lane_buffers(GpuLane& lane, size_t bytes) {
    die_cuda(cudaSetDevice(lane.gpu_id), "cudaSetDevice", lane.gpu_id, bytes);
    die_cuda(cudaHostAlloc(&lane.host_src, bytes, cudaHostAllocPortable),
             "cudaHostAlloc", lane.gpu_id, bytes);
    die_cuda(cudaMalloc(&lane.device_dst, bytes), "cudaMalloc", lane.gpu_id, bytes);
    std::memset(lane.host_src, 0xA5, bytes);
    die_cuda(cudaMemset(lane.device_dst, 0, bytes), "cudaMemset", lane.gpu_id, bytes);
    die_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", lane.gpu_id, bytes);
}

static void free_lane_buffers(GpuLane& lane) {
    cudaSetDevice(lane.gpu_id);
    if (lane.device_dst) {
        cudaFree(lane.device_dst);
        lane.device_dst = nullptr;
    }
    if (lane.host_src) {
        cudaFreeHost(lane.host_src);
        lane.host_src = nullptr;
    }
}

static void print_usage(const char* argv0) {
    fprintf(stderr, "Usage: %s <gpu0> <gpu1> <nic_name>\n", argv0);
    fprintf(stderr, "Example: %s 4 5 mlx5_4\n", argv0);
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    int gpu0 = std::atoi(argv[1]);
    int gpu1 = std::atoi(argv[2]);
    std::string nic_name = argv[3];

    if (gpu0 == gpu1) {
        fprintf(stderr, "[dual] gpu0 and gpu1 must be different: gpu=%d\n", gpu0);
        return 1;
    }

    printf("=================================================================\n");
    printf("  Dual-GPU GDR H2D Benchmark  -  GPUs %d,%d  NIC %s\n",
           gpu0, gpu1, nic_name.c_str());
    printf("=================================================================\n\n");

    cudaDeviceProp prop0{};
    cudaDeviceProp prop1{};
    die_cuda(cudaGetDeviceProperties(&prop0, gpu0), "cudaGetDeviceProperties", gpu0, 0);
    die_cuda(cudaGetDeviceProperties(&prop1, gpu1), "cudaGetDeviceProperties", gpu1, 0);
    printf("GPU%d: %s\n", gpu0, prop0.name);
    printf("GPU%d: %s\n\n", gpu1, prop1.name);

    GpuLane lanes[2];
    lanes[0].gpu_id = gpu0;
    lanes[1].gpu_id = gpu1;

    try {
        lanes[0].channel = GDRCopyLib::open(gpu0, nic_name);
        lanes[1].channel = GDRCopyLib::open(gpu1, nic_name);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to open GDR channel: %s\n", e.what());
        return 1;
    }

    printf("Benchmark mode: dual GPU concurrent GDR H2D, bandwidth by shared total time\n");
    printf("Warmup=%d  Iters=%d\n\n", WARMUP, ITERS);

    std::vector<size_t> sizes;
    for (size_t s = 4096; s <= 64ULL << 20; s *= 4)
        sizes.push_back(s);

    printf("%-12s | %-16s | %-16s | %-16s\n",
           "Size", "GPU0 BW", "GPU1 BW", "Total BW");
    printf("%-12s-+-%-16s-+-%-16s-+-%-16s\n",
           "------------", "----------------", "----------------", "----------------");

    uint64_t total_ops = 0;
    uint64_t total_bytes = 0;

    const size_t max_bytes = sizes.back();
    allocate_lane_buffers(lanes[0], max_bytes);
    allocate_lane_buffers(lanes[1], max_bytes);
    prime_gdr_h2d_window(lanes[0].channel, lanes[0].device_dst, lanes[0].host_src,
                         max_bytes, lanes[0].gpu_id);
    prime_gdr_h2d_window(lanes[1].channel, lanes[1].device_dst, lanes[1].host_src,
                         max_bytes, lanes[1].gpu_id);

    for (size_t bytes : sizes) {

        std::atomic<int> ready{0};
        std::atomic<int> finished{0};
        std::atomic<bool> start{false};

        auto worker = [&](int lane_index) {
            GpuLane& lane = lanes[lane_index];
            die_cuda(cudaSetDevice(lane.gpu_id), "cudaSetDevice", lane.gpu_id, bytes);

            run_gdr_h2d_batch(lane.channel, lane.device_dst, lane.host_src,
                              bytes, WARMUP, false, lane.gpu_id);

            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();

            lane.measured_us = run_gdr_h2d_batch(lane.channel, lane.device_dst, lane.host_src,
                                                 bytes, ITERS, true, lane.gpu_id);
            finished.fetch_add(1, std::memory_order_release);
        };

        std::thread t0(worker, 0);
        std::thread t1(worker, 1);

        while (ready.load(std::memory_order_acquire) != 2)
            std::this_thread::yield();

        double t_start = now_us();
        start.store(true, std::memory_order_release);
        while (finished.load(std::memory_order_acquire) != 2)
            std::this_thread::yield();
        double shared_us = now_us() - t_start;

        t0.join();
        t1.join();

        double gpu0_bw = ((double)bytes * (double)ITERS / 1e9) /
                         (lanes[0].measured_us / 1e6);
        double gpu1_bw = ((double)bytes * (double)ITERS / 1e9) /
                         (lanes[1].measured_us / 1e6);
        double total_bw = ((double)bytes * (double)ITERS * 2.0 / 1e9) /
                          (shared_us / 1e6);

        char size_str[32];
        format_size(bytes, size_str, sizeof(size_str));
        printf("%-12s | %8.2f GB/s   | %8.2f GB/s   | %8.2f GB/s\n",
               size_str, gpu0_bw, gpu1_bw, total_bw);

        total_ops += (uint64_t)ITERS * 2ULL;
        total_bytes += (uint64_t)bytes * (uint64_t)ITERS * 2ULL;
    }

    free_lane_buffers(lanes[0]);
    free_lane_buffers(lanes[1]);

    printf("\n=================================================================\n");
    printf("Total ops: %lu\n", (unsigned long)total_ops);
    printf("Total bytes: %.2f GiB\n", total_bytes / (double)(1ULL << 30));
    printf("=================================================================\n");

    GDRCopyLib::shutdown();
    return 0;
}
