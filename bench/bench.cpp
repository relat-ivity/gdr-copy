/**
 * bench.cpp  —  GDR Copy vs cudaMemcpy async timing benchmark
 *
 * Usage:
 *   sudo ./build/bench [gpu_id] [nic_name] [pipeline_depth]
 *   sudo ./build/bench 0 mlx5_0
 *   sudo ./build/bench 0 mlx5_0 16
 *
 * Output:
 *   For each transfer size × direction, prints:
 *     - submit (issue) latency
 *     - transfer service latency (queue-excluded per request)
 *     - median latency (µs)
 *     - p99 latency (µs)
 *     - GB/s only for transfer/service tables
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
#include <deque>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

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

struct GdrCompletion {
    uint64_t req_id = 0;
    double observed_us = 0.0;
};

class GlobalGdrCqPoller {
public:
    explicit GlobalGdrCqPoller(const std::shared_ptr<GDRCopyChannel>& ch)
        : ch_(ch) {
        worker_ = std::thread([this]() { run(); });
    }

    ~GlobalGdrCqPoller() {
        stop();
    }

    void stop() {
        bool expected = false;
        if (!stopped_.compare_exchange_strong(expected, true,
                                              std::memory_order_relaxed)) {
            return;
        }
        stop_flag_.store(true, std::memory_order_relaxed);
        if (worker_.joinable()) worker_.join();
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stopped_done_ = true;
        }
        cv_.notify_all();
    }

    int fatal_rc() const {
        return fatal_rc_.load(std::memory_order_relaxed);
    }

    bool try_pop(GdrCompletion* out) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (q_.empty()) return false;
        *out = q_.front();
        q_.pop_front();
        return true;
    }

    int wait_pop(GdrCompletion* out) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [&]() {
            return fatal_rc_.load(std::memory_order_relaxed) != 0 ||
                   !q_.empty() ||
                   stopped_done_;
        });
        if (!q_.empty()) {
            *out = q_.front();
            q_.pop_front();
            return 0;
        }
        int frc = fatal_rc_.load(std::memory_order_relaxed);
        if (frc != 0) return frc;
        return -EAGAIN;
    }

private:
    void run() {
        while (!stop_flag_.load(std::memory_order_relaxed)) {
            uint64_t req_id = 0;
            int rc = ch_->poll_wc(&req_id);
            if (rc == -EAGAIN) {
                std::this_thread::yield();
                continue;
            }
            if (rc != 0) {
                fatal_rc_.store(rc, std::memory_order_relaxed);
                cv_.notify_all();
                return;
            }
            GdrCompletion c{req_id, now_us()};
            {
                std::lock_guard<std::mutex> lk(mtx_);
                q_.push_back(c);
            }
            cv_.notify_all();
        }
    }

    std::shared_ptr<GDRCopyChannel> ch_;
    std::thread worker_;
    std::deque<GdrCompletion> q_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<int> fatal_rc_{0};
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> stopped_{false};
    bool stopped_done_ = false;
};

// Measure request-level latency:
// transfer sample uses queue-excluded service time approximation on one channel:
//   service = completion_time - max(submit_done_time, previous_completed_time).
static BenchPair run_gdr_timings(std::shared_ptr<GDRCopyChannel> ch,
                                 GlobalGdrCqPoller& cq_poller,
                                 const std::vector<void*>& dst_slots,
                                 const std::vector<void*>& src_slots,
                                 size_t bytes, GDRCopyKind kind,
                                 int warmup, int iters)
{
    const int depth = (int)dst_slots.size();
    if (depth <= 0 || src_slots.size() != dst_slots.size()) {
        fprintf(stderr, "[issue] invalid slot setup: dst=%zu src=%zu\n",
                dst_slots.size(), src_slots.size());
        std::exit(2);
    }

    std::vector<double> issue_samples;
    std::vector<double> transfer_samples;
    issue_samples.reserve(iters);
    transfer_samples.reserve(iters);
    std::vector<char> slot_busy((size_t)depth, 0);
    struct PendingReq {
        double submit_done_us = 0.0;
        bool record = false;
        int slot = -1;
        int remaining_wcs = 0;
    };
    struct EarlyWc {
        int count = 0;
        double last_seen_us = 0.0;
    };
    std::unordered_map<uint64_t, PendingReq> pending;
    pending.reserve((size_t)(warmup + iters) * 2);
    std::unordered_map<uint64_t, EarlyWc> early_wcs;
    early_wcs.reserve((size_t)(warmup + iters) * 2);

    const int total = warmup + iters;
    int issued = 0;
    int completed = 0;
    int next_slot_hint = 0;
    double measure_begin_us = -1.0;
    double measure_end_us = -1.0;
    double service_tail_us = -1.0;
    int fatal_rc = 0;

    auto pick_free_slot = [&]() -> int {
        for (int probe = 0; probe < depth; ++probe) {
            int cand = (next_slot_hint + probe) % depth;
            if (!slot_busy[(size_t)cand]) {
                next_slot_hint = (cand + 1) % depth;
                return cand;
            }
        }
        return -1;
    };

    auto finalize_req = [&](const PendingReq& req, double done_us) {
        slot_busy[(size_t)req.slot] = 0;
        if (req.record) {
            double service_start = req.submit_done_us;
            if (service_tail_us > service_start) service_start = service_tail_us;
            double service_us = done_us - service_start;
            if (service_us < 0.0) service_us = 0.0;
            transfer_samples.push_back(service_us);
            measure_end_us = done_us;
        }
        if (done_us > service_tail_us) service_tail_us = done_us;
        completed++;
    };

    auto on_completion = [&](uint64_t req_id, double done_us) {
        auto it = pending.find(req_id);
        if (it == pending.end()) {
            EarlyWc& ew = early_wcs[req_id];
            ew.count += 1;
            ew.last_seen_us = done_us;
            return;
        }

        PendingReq& req = it->second;
        req.remaining_wcs--;
        if (req.remaining_wcs <= 0) {
            PendingReq done_req = req;
            pending.erase(it);
            finalize_req(done_req, done_us);
        }
    };

    while (issued < total) {
        int frc = cq_poller.fatal_rc();
        if (frc != 0) {
            fatal_rc = frc;
            break;
        }

        GdrCompletion c{};
        while (cq_poller.try_pop(&c)) {
            on_completion(c.req_id, c.observed_us);
        }

        int slot = pick_free_slot();
        if (slot < 0) {
            int wrc = cq_poller.wait_pop(&c);
            if (wrc == 0) {
                on_completion(c.req_id, c.observed_us);
                continue;
            }
            if (wrc == -EAGAIN) continue;
            fatal_rc = wrc;
            break;
        }
        slot_busy[(size_t)slot] = 1;

        double t0 = now_us();
        uint64_t req_id = 0;
        int expected_wcs = 0;
        int rc = ch->memcpy_async_tagged(dst_slots[(size_t)slot], src_slots[(size_t)slot], bytes, kind,
                                         &req_id, &expected_wcs);
        double t1 = now_us();

        if (rc == -EBUSY) {
            slot_busy[(size_t)slot] = 0;
            int wrc = cq_poller.wait_pop(&c);
            if (wrc == 0) {
                on_completion(c.req_id, c.observed_us);
                continue;
            }
            if (wrc == -EAGAIN) continue;
            fatal_rc = wrc;
            break;
        }
        if (rc != 0) {
            slot_busy[(size_t)slot] = 0;
            fatal_rc = rc;
            break;
        }

        const bool record = (issued >= warmup);
        if (record) issue_samples.push_back(t1 - t0);
        if (record && measure_begin_us < 0.0) measure_begin_us = t1;
        if (expected_wcs <= 0) expected_wcs = 1;

        PendingReq req{t1, record, slot, expected_wcs};
        auto ew = early_wcs.find(req_id);
        if (ew != early_wcs.end()) {
            req.remaining_wcs -= ew->second.count;
            double done_us = ew->second.last_seen_us;
            early_wcs.erase(ew);
            if (req.remaining_wcs <= 0) {
                finalize_req(req, done_us);
            } else {
                pending[req_id] = req;
            }
        } else {
            pending[req_id] = req;
        }

        issued++;
    }

    GdrCompletion c{};
    while (fatal_rc == 0 && completed < issued) {
        while (cq_poller.try_pop(&c)) {
            on_completion(c.req_id, c.observed_us);
        }
        if (completed >= issued) break;
        int wrc = cq_poller.wait_pop(&c);
        if (wrc == 0) {
            on_completion(c.req_id, c.observed_us);
            continue;
        }
        if (wrc == -EAGAIN) continue;
        fatal_rc = wrc;
        break;
    }

    if (fatal_rc != 0) {
        fprintf(stderr, "[issue] gdr pipeline failed: rc=%d kind=%d bytes=%zu\n",
                fatal_rc, (int)kind, bytes);
        std::exit(2);
    }
    if (issued != total || completed != total) {
        fprintf(stderr,
                "[issue] gdr pipeline count mismatch: issued=%d completed=%d total=%d kind=%d bytes=%zu\n",
                issued, completed, total, (int)kind, bytes);
        std::exit(2);
    }

    BenchPair out{};
    out.issue = analyse(issue_samples, bytes);
    out.transfer = analyse(transfer_samples, bytes);
    if (measure_begin_us > 0.0 && measure_end_us > measure_begin_us) {
        const double span_us = measure_end_us - measure_begin_us;
        out.transfer.bw_GBs = ((double)bytes * (double)iters / 1e9) / (span_us / 1e6);
    }
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

static BenchPair run_cuda_timings(const std::vector<void*>& dst_slots,
                                  const std::vector<void*>& src_slots,
                                  size_t bytes, cudaMemcpyKind kind,
                                  int warmup, int iters, cudaStream_t stream)
{
    const int depth = (int)dst_slots.size();
    if (depth <= 0 || src_slots.size() != dst_slots.size()) {
        fprintf(stderr, "[issue] invalid CUDA slot setup: dst=%zu src=%zu\n",
                dst_slots.size(), src_slots.size());
        std::exit(2);
    }

    std::vector<double> issue_samples;
    std::vector<double> transfer_samples;
    issue_samples.reserve(iters);
    transfer_samples.reserve(iters);
    std::vector<char> slot_busy((size_t)depth, 0);
    std::vector<cudaEvent_t> slot_start_events((size_t)depth, nullptr);
    std::vector<cudaEvent_t> slot_done_events((size_t)depth, nullptr);
    for (int s = 0; s < depth; ++s) {
        cudaError_t ce = cudaEventCreate(&slot_start_events[(size_t)s]);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[issue] cudaEventCreate(start) failed: slot=%d err=%s\n",
                    s, cudaGetErrorString(ce));
            std::exit(2);
        }
        ce = cudaEventCreate(&slot_done_events[(size_t)s]);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[issue] cudaEventCreate(done) failed: slot=%d err=%s\n",
                    s, cudaGetErrorString(ce));
            std::exit(2);
        }
    }
    struct PendingCudaReq {
        double submit_done_us = 0.0;
        bool record = false;
        int slot = -1;
    };
    std::unordered_map<uint64_t, PendingCudaReq> pending;
    pending.reserve((size_t)(warmup + iters) * 2);
    uint64_t next_req_id = 1;
    const int total = warmup + iters;
    int issued = 0;
    int next_slot_hint = 0;
    double measure_begin_us = -1.0;
    double measure_end_us = -1.0;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<int> completed{0};
    std::atomic<int> fatal_rc{0};

    auto has_free_slot = [&]() -> bool {
        for (int i = 0; i < depth; ++i) {
            if (!slot_busy[(size_t)i]) return true;
        }
        return false;
    };

    auto pick_free_slot = [&]() -> int {
        for (int probe = 0; probe < depth; ++probe) {
            int cand = (next_slot_hint + probe) % depth;
            if (!slot_busy[(size_t)cand]) {
                next_slot_hint = (cand + 1) % depth;
                return cand;
            }
        }
        return -1;
    };

    std::thread cq_thread([&]() {
        while (completed.load(std::memory_order_relaxed) < total &&
               fatal_rc.load(std::memory_order_relaxed) == 0) {
            std::vector<std::pair<uint64_t, int>> snapshot;
            {
                std::lock_guard<std::mutex> lk(mtx);
                snapshot.reserve(pending.size());
                for (const auto& kv : pending) {
                    snapshot.push_back({kv.first, kv.second.slot});
                }
            }
            if (snapshot.empty()) {
                std::this_thread::yield();
                continue;
            }

            uint64_t ready_req_id = 0;
            cudaError_t ready_ce = cudaErrorNotReady;
            for (const auto& item : snapshot) {
                cudaError_t ce = cudaEventQuery(slot_done_events[(size_t)item.second]);
                if (ce == cudaErrorNotReady) {
                    (void)cudaGetLastError();
                    continue;
                }
                if (ce != cudaSuccess) {
                    fatal_rc.store((int)ce, std::memory_order_relaxed);
                    cv.notify_all();
                    return;
                }
                ready_req_id = item.first;
                ready_ce = ce;
                break;
            }
            if (ready_ce == cudaErrorNotReady || ready_req_id == 0) {
                std::this_thread::yield();
                continue;
            }

            PendingCudaReq req{};
            {
                std::lock_guard<std::mutex> lk(mtx);
                auto it = pending.find(ready_req_id);
                if (it == pending.end()) continue;
                req = it->second;
                pending.erase(it);
            }

            double t2 = now_us();
            if (req.record) {
                float ms = 0.0f;
                cudaError_t ce = cudaEventElapsedTime(&ms,
                                                      slot_start_events[(size_t)req.slot],
                                                      slot_done_events[(size_t)req.slot]);
                if (ce != cudaSuccess) {
                    {
                        std::lock_guard<std::mutex> lk(mtx);
                        slot_busy[(size_t)req.slot] = 0;
                    }
                    fatal_rc.store((int)ce, std::memory_order_relaxed);
                    cv.notify_all();
                    return;
                }
                transfer_samples.push_back((double)ms * 1000.0);
                measure_end_us = t2;
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                slot_busy[(size_t)req.slot] = 0;
            }
            completed.fetch_add(1, std::memory_order_relaxed);
            cv.notify_all();
        }
    });

    while (issued < total) {
        if (fatal_rc.load(std::memory_order_relaxed) != 0) break;

        int slot = -1;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&]() {
                return fatal_rc.load(std::memory_order_relaxed) != 0 ||
                       has_free_slot();
            });
            if (fatal_rc.load(std::memory_order_relaxed) != 0) break;

            slot = pick_free_slot();
            if (slot < 0) continue;
            slot_busy[(size_t)slot] = 1;  // reserve before submit
        }

        cudaError_t ce = cudaEventRecord(slot_start_events[(size_t)slot], stream);
        if (ce != cudaSuccess) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                slot_busy[(size_t)slot] = 0;
            }
            fatal_rc.store((int)ce, std::memory_order_relaxed);
            cv.notify_all();
            break;
        }
        double t0 = now_us();
        ce = cudaMemcpyAsync(dst_slots[(size_t)slot], src_slots[(size_t)slot], bytes, kind, stream);
        double t1 = now_us();
        if (ce != cudaSuccess) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                slot_busy[(size_t)slot] = 0;
            }
            fatal_rc.store((int)ce, std::memory_order_relaxed);
            cv.notify_all();
            break;
        }
        ce = cudaEventRecord(slot_done_events[(size_t)slot], stream);
        if (ce != cudaSuccess) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                slot_busy[(size_t)slot] = 0;
            }
            fatal_rc.store((int)ce, std::memory_order_relaxed);
            cv.notify_all();
            break;
        }

        const bool record = (issued >= warmup);
        if (record) issue_samples.push_back(t1 - t0);
        if (record && measure_begin_us < 0.0) measure_begin_us = t1;
        uint64_t req_id = next_req_id++;
        {
            std::lock_guard<std::mutex> lk(mtx);
            pending[req_id] = PendingCudaReq{t1, record, slot};
        }
        ++issued;
    }

    {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&]() {
            return fatal_rc.load(std::memory_order_relaxed) != 0 ||
                   completed.load(std::memory_order_relaxed) >= total;
        });
    }

    if (cq_thread.joinable()) cq_thread.join();

    for (auto& evt : slot_start_events) {
        cudaEventDestroy(evt);
    }
    for (auto& evt : slot_done_events) {
        cudaEventDestroy(evt);
    }

    int frc = fatal_rc.load(std::memory_order_relaxed);
    if (frc != 0) {
        fprintf(stderr, "[issue] cuda pipeline failed: rc=%d(%s) bytes=%zu\n",
                frc, cudaGetErrorString((cudaError_t)frc), bytes);
        std::exit(2);
    }

    BenchPair out{};
    out.issue = analyse(issue_samples, bytes);
    out.transfer = analyse(transfer_samples, bytes);
    if (measure_begin_us > 0.0 && measure_end_us > measure_begin_us) {
        const double span_us = measure_end_us - measure_begin_us;
        out.transfer.bw_GBs = ((double)bytes * (double)iters / 1e9) / (span_us / 1e6);
    }
    return out;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    int         gpu_id   = (argc > 1) ? std::atoi(argv[1]) : 0;
    std::string  nic_name = (argc > 2) ? argv[2]            : "mlx5_0";
    int pipeline_depth    = (argc > 3) ? std::atoi(argv[3]) : 16;
    if (pipeline_depth < 1) pipeline_depth = 1;

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
    printf("Benchmark mode: pipelined issue + global CQ listener thread (wr_id match, depth=%d)\n\n",
           pipeline_depth);
    ch->reset_stats();
    std::unique_ptr<GlobalGdrCqPoller> gdr_cq_poller =
        std::make_unique<GlobalGdrCqPoller>(ch);

    // ── Transfer sizes to sweep ───────────────────────────────────────────
    std::vector<size_t> sizes;
    for (size_t s = 4096; s <= 64ULL << 20; s *= 4)
        sizes.push_back(s);

    static const int WARMUP = 100;
    static const int ITERS  = 1000;

    // Async benchmark: split timing into issue and transfer-service parts.
    cudaStream_t issue_stream{};
    cudaStreamCreate(&issue_stream);

    std::vector<DirectionRow> h2d_rows;
    h2d_rows.reserve(sizes.size());

    for (size_t bytes : sizes) {
        std::vector<void*> h_src_slots((size_t)pipeline_depth, nullptr);
        std::vector<void*> d_dst_slots((size_t)pipeline_depth, nullptr);
        for (int slot = 0; slot < pipeline_depth; ++slot) {
            cudaHostAlloc(&h_src_slots[(size_t)slot], bytes, cudaHostAllocPortable);
            cudaMalloc(&d_dst_slots[(size_t)slot], bytes);
            cudaMemset(d_dst_slots[(size_t)slot], 0, bytes);
            memset(h_src_slots[(size_t)slot], (slot & 1) ? 0xA5 : 0x5A, bytes);
        }

        BenchPair gdr  = run_gdr_timings(ch, *gdr_cq_poller, d_dst_slots, h_src_slots, bytes, GDR_H2D, WARMUP, ITERS);
        BenchPair cuda = run_cuda_timings(d_dst_slots, h_src_slots, bytes, cudaMemcpyHostToDevice,
                                          WARMUP, ITERS, issue_stream);

        h2d_rows.push_back(DirectionRow{bytes, gdr, cuda});

        for (int slot = 0; slot < pipeline_depth; ++slot) {
            cudaFreeHost(h_src_slots[(size_t)slot]);
            cudaFree(d_dst_slots[(size_t)slot]);
        }
    }

    print_latency_table("Host->Device Issue Latency", h2d_rows, true);
    print_latency_table("Host->Device Transfer Service Latency", h2d_rows, false);

    std::vector<DirectionRow> d2h_rows;
    d2h_rows.reserve(sizes.size());

    for (size_t bytes : sizes) {
        std::vector<void*> d_src_slots((size_t)pipeline_depth, nullptr);
        std::vector<void*> h_dst_slots((size_t)pipeline_depth, nullptr);
        for (int slot = 0; slot < pipeline_depth; ++slot) {
            cudaMalloc(&d_src_slots[(size_t)slot], bytes);
            cudaHostAlloc(&h_dst_slots[(size_t)slot], bytes, cudaHostAllocPortable);
            cudaMemset(d_src_slots[(size_t)slot], (slot & 1) ? 0x5A : 0xA5, bytes);
            memset(h_dst_slots[(size_t)slot], 0, bytes);
        }

        BenchPair gdr  = run_gdr_timings(ch, *gdr_cq_poller, h_dst_slots, d_src_slots, bytes, GDR_D2H, WARMUP, ITERS);
        BenchPair cuda = run_cuda_timings(h_dst_slots, d_src_slots, bytes, cudaMemcpyDeviceToHost,
                                          WARMUP, ITERS, issue_stream);

        d2h_rows.push_back(DirectionRow{bytes, gdr, cuda});

        for (int slot = 0; slot < pipeline_depth; ++slot) {
            cudaFree(d_src_slots[(size_t)slot]);
            cudaFreeHost(h_dst_slots[(size_t)slot]);
        }
    }

    print_latency_table("Device->Host Issue Latency", d2h_rows, true);
    print_latency_table("Device->Host Transfer Service Latency", d2h_rows, false);

    cudaStreamDestroy(issue_stream);

    // ── Summary ───────────────────────────────────────────────────────────
    GDRStats final_s = ch->stats();
    printf("\n=================================================================\n");
    printf("Total ops: %lu  (RDMA: %lu  Fallback: %lu)\n",
           final_s.total_ops, final_s.rdma_ops, final_s.fallback_ops);
    printf("Total bytes: %.2f GiB\n", final_s.total_bytes / (double)(1ULL<<30));
    printf("=================================================================\n");

    gdr_cq_poller->stop();
    gdr_cq_poller.reset();
    GDRCopyLib::shutdown();
    return 0;
}
