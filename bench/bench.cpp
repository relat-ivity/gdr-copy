/**
 * bench.cpp  —  GDR Copy vs cudaMemcpy async timing benchmark
 *
 * Usage:
 *   sudo ./build/bench [gpu_id] [nic_name] [parallel_reqs]
 *   sudo ./build/bench 0 mlx5_0
 *   sudo ./build/bench 0 mlx5_0 8
 *
 * Output:
 *   For each transfer size × direction, prints:
 *     - submit (issue) latency
 *     - transfer latency (request submit-done -> completion observed)
 *     - median latency (µs)
 *     - p99 latency (µs)
 *     - GB/s only for transfer tables
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>

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

struct ReqKey {
    size_t lane = 0;
    uint64_t req_id = 0;
    bool operator==(const ReqKey& o) const {
        return lane == o.lane && req_id == o.req_id;
    }
};

struct ReqKeyHash {
    size_t operator()(const ReqKey& k) const {
        size_t h1 = std::hash<size_t>{}(k.lane);
        size_t h2 = std::hash<uint64_t>{}(k.req_id);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

struct GdrRunState {
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<double> transfer_samples;
    std::vector<size_t> completed_lanes;
    int completed = 0;
    int fatal_rc = 0;
    double measure_end_us = -1.0;
};

// One global CQ polling thread; it polls all channel CQs and handles completion.
class GdrCqListener {
public:
    explicit GdrCqListener(std::vector<std::shared_ptr<GDRCopyChannel>> channels)
        : channels_(std::move(channels)), th_([this]() { this->loop(); }) {}

    ~GdrCqListener() { stop(); }

    int register_request(size_t lane, uint64_t req_id,
                         GdrRunState& run,
                         double submit_done_us,
                         bool record,
                         int expected_wcs)
    {
        if (lane >= channels_.size()) return -EINVAL;
        if (expected_wcs <= 0) expected_wcs = 1;

        std::lock_guard<std::mutex> lk(mtx_);
        if (stop_) return -ESHUTDOWN;
        if (listener_rc_ != 0) return listener_rc_;

        ReqKey key{lane, req_id};
        if (pending_.find(key) != pending_.end()) return -EEXIST;

        int early = 0;
        auto eit = early_wc_.find(key);
        if (eit != early_wc_.end()) {
            early = eit->second;
            early_wc_.erase(eit);
        }

        PendingReq req{};
        req.run = &run;
        req.lane = lane;
        req.submit_done_us = submit_done_us;
        req.record = record;
        req.remaining_wcs = expected_wcs - early;
        if (req.remaining_wcs <= 0) {
            ready_.push_back(req);
        } else {
            pending_[key] = req;
        }
        return 0;
    }

    int wait_until_completed(GdrRunState& run, int target_completed) {
        std::unique_lock<std::mutex> lk(run.mtx);
        run.cv.wait(lk, [&]() {
            return run.fatal_rc != 0 ||
                   run.completed >= target_completed;
        });
        return run.fatal_rc;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (stop_) return;
            stop_ = true;
        }
        if (th_.joinable()) th_.join();
    }

private:
    struct PendingReq {
        GdrRunState* run = nullptr;
        size_t lane = 0;
        double submit_done_us = 0.0;
        bool record = false;
        int remaining_wcs = 0;
    };

    void complete_req(const PendingReq& req, double done_us) {
        if (!req.run) return;
        {
            std::lock_guard<std::mutex> run_lk(req.run->mtx);
            if (req.record) {
                req.run->transfer_samples.push_back(done_us - req.submit_done_us);
                req.run->measure_end_us = done_us;
            }
            req.run->completed++;
            req.run->completed_lanes.push_back(req.lane);
        }
        req.run->cv.notify_all();
    }

    void mark_all_failed(int rc) {
        std::vector<GdrRunState*> runs;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (listener_rc_ == 0) listener_rc_ = rc;
            runs.reserve(pending_.size() + ready_.size());
            for (const auto& kv : pending_) {
                if (kv.second.run) runs.push_back(kv.second.run);
            }
            for (const auto& req : ready_) {
                if (req.run) runs.push_back(req.run);
            }
        }
        for (GdrRunState* run : runs) {
            std::lock_guard<std::mutex> run_lk(run->mtx);
            if (run->fatal_rc == 0) run->fatal_rc = rc;
            run->cv.notify_all();
        }
    }

    bool pop_ready(PendingReq* out) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (ready_.empty()) return false;
        *out = ready_.back();
        ready_.pop_back();
        return true;
    }

    void loop() {
        while (true) {
            {
                std::lock_guard<std::mutex> lk(mtx_);
                if (stop_) return;
            }

            PendingReq ready_req{};
            if (pop_ready(&ready_req)) {
                complete_req(ready_req, now_us());
                continue;
            }

            bool progressed = false;
            for (size_t lane = 0; lane < channels_.size(); ++lane) {
                uint64_t req_id = 0;
                int rc = channels_[lane]->poll_wc(&req_id);
                if (rc == -EAGAIN) continue;
                if (rc != 0) {
                    mark_all_failed(rc);
                    return;
                }

                progressed = true;
                const double done_us = now_us();
                ReqKey key{lane, req_id};
                PendingReq req{};
                bool done = false;
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    auto it = pending_.find(key);
                    if (it == pending_.end()) {
                        early_wc_[key]++;
                    } else {
                        it->second.remaining_wcs--;
                        if (it->second.remaining_wcs <= 0) {
                            req = it->second;
                            pending_.erase(it);
                            done = true;
                        }
                    }
                }
                if (done) complete_req(req, done_us);
            }

            if (!progressed) std::this_thread::yield();
        }
    }

    std::vector<std::shared_ptr<GDRCopyChannel>> channels_;
    mutable std::mutex mtx_;
    std::unordered_map<ReqKey, PendingReq, ReqKeyHash> pending_;
    std::unordered_map<ReqKey, int, ReqKeyHash> early_wc_;
    std::vector<PendingReq> ready_;
    bool stop_ = false;
    int listener_rc_ = 0;
    std::thread th_;
};

// Main thread: submit + record submit_done time only.
// CQ thread: poll all CQs, match (lane, req_id), and record transfer timing.
static BenchPair run_gdr_timings(
    const std::vector<std::shared_ptr<GDRCopyChannel>>& channels,
    GdrCqListener& cq_listener,
    const std::vector<void*>& dst_lanes,
    const std::vector<void*>& src_lanes,
    size_t bytes, GDRCopyKind kind,
    int warmup, int iters)
{
    const size_t lanes = channels.size();
    if (lanes == 0 || dst_lanes.size() != lanes || src_lanes.size() != lanes) {
        fprintf(stderr, "[issue] invalid gdr lane setup: channels=%zu dst=%zu src=%zu\n",
                channels.size(), dst_lanes.size(), src_lanes.size());
        std::exit(2);
    }

    std::vector<double> issue_samples;
    issue_samples.reserve(iters);
    GdrRunState run{};
    run.transfer_samples.reserve(iters);
    run.completed_lanes.reserve((size_t)(warmup + iters));

    const int total = warmup + iters;
    double measure_begin_us = -1.0;

    // Lane state in main thread: one lane can carry at most one in-flight request.
    // A lane becomes available only after CQ thread reports completion for that lane.
    std::vector<char> lane_busy(lanes, 0);

    // Reclaim completed lanes from CQ thread notifications.
    auto reclaim_completed_lanes = [&]() -> int {
        std::lock_guard<std::mutex> lk(run.mtx);
        if (run.fatal_rc != 0) return run.fatal_rc;
        for (size_t lane : run.completed_lanes) {
            if (lane < lane_busy.size()) lane_busy[lane] = 0;
        }
        run.completed_lanes.clear();
        return 0;
    };

    int issued = 0;
    size_t rr_lane = 0;
    while (issued < total) {
        int frc = reclaim_completed_lanes();
        if (frc != 0) {
            fprintf(stderr, "[issue] gdr completion failed: rc=%d kind=%d bytes=%zu\n",
                    frc, (int)kind, bytes);
            std::exit(2);
        }

        // Pick next free lane in round-robin order to avoid lane starvation.
        size_t lane = lanes;
        for (size_t probe = 0; probe < lanes; ++probe) {
            const size_t cand = (rr_lane + probe) % lanes;
            if (!lane_busy[cand]) {
                lane = cand;
                rr_lane = (cand + 1) % lanes;
                break;
            }
        }
        if (lane == lanes) {
            // All lanes busy: wait until CQ thread returns at least one completion.
            std::unique_lock<std::mutex> lk(run.mtx);
            run.cv.wait(lk, [&]() {
                return run.fatal_rc != 0 || !run.completed_lanes.empty();
            });
            continue;
        }

        lane_busy[lane] = 1;
        const double t0 = now_us();
        uint64_t req_id = 0;
        int expected_wcs = 0;
        int rc = channels[lane]->memcpy_async_tagged(
            dst_lanes[lane], src_lanes[lane], bytes, kind, &req_id, &expected_wcs);
        const double t1 = now_us();

        if (rc == -EBUSY) {
            lane_busy[lane] = 0;
            std::this_thread::yield();
            continue;
        }
        if (rc != 0) {
            fprintf(stderr, "[issue] gdr submit failed: rc=%d lane=%zu kind=%d bytes=%zu\n",
                    rc, lane, (int)kind, bytes);
            std::exit(2);
        }

        const bool record = (issued >= warmup);
        if (record) {
            // Issue latency: submit call return time minus submit start time.
            issue_samples.push_back(t1 - t0);
            if (measure_begin_us < 0.0) measure_begin_us = t1;
        }

        // Register (lane, req_id) so CQ thread can map WC -> request timing.
        rc = cq_listener.register_request(lane, req_id, run, t1, record, expected_wcs);
        if (rc != 0) {
            fprintf(stderr,
                    "[issue] gdr register failed: rc=%d lane=%zu kind=%d bytes=%zu req_id=%llu\n",
                    rc, lane, (int)kind, bytes, (unsigned long long)req_id);
            std::exit(2);
        }

        issued++;
    }

    // Wait until all submitted requests are completed by CQ thread.
    int rc = cq_listener.wait_until_completed(run, total);
    if (rc != 0) {
        fprintf(stderr, "[issue] gdr completion failed: rc=%d kind=%d bytes=%zu\n",
                rc, (int)kind, bytes);
        std::exit(2);
    }

    std::vector<double> transfer_samples;
    double measure_end_us = -1.0;
    {
        std::lock_guard<std::mutex> lk(run.mtx);
        transfer_samples = run.transfer_samples;
        measure_end_us = run.measure_end_us;
    }

    BenchPair out{};
    out.issue = analyse(issue_samples, bytes);
    out.transfer = analyse(transfer_samples, bytes);
    if (measure_begin_us > 0.0 && measure_end_us > measure_begin_us) {
        // Throughput window: first measured submit-done -> last measured completion.
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

static BenchPair run_cuda_timings(
    const std::vector<void*>& dst_lanes,
    const std::vector<void*>& src_lanes,
    size_t bytes, cudaMemcpyKind kind,
    int warmup, int iters,
    const std::vector<cudaStream_t>& streams)
{
    const size_t lanes = dst_lanes.size();
    if (lanes == 0 || src_lanes.size() != lanes || streams.size() != lanes) {
        fprintf(stderr, "[issue] invalid cuda lane setup: dst=%zu src=%zu streams=%zu\n",
                dst_lanes.size(), src_lanes.size(), streams.size());
        std::exit(2);
    }

    std::vector<double> issue_samples;
    std::vector<double> transfer_samples;
    issue_samples.reserve(iters);
    transfer_samples.reserve(iters);

    // One completion event per lane; each lane keeps one in-flight request.
    std::vector<cudaEvent_t> done_events(lanes, nullptr);
    for (size_t lane = 0; lane < lanes; ++lane) {
        cudaError_t ce = cudaEventCreateWithFlags(&done_events[lane], cudaEventDisableTiming);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[issue] cudaEventCreate failed: lane=%zu err=%s\n",
                    lane, cudaGetErrorString(ce));
            std::exit(2);
        }
    }

    struct LanePending {
        bool active = false;
        double submit_done_us = 0.0;
        bool record = false;
    };

    const int total = warmup + iters;
    double measure_begin_us = -1.0;
    double measure_end_us = -1.0;

    std::mutex mtx;
    std::condition_variable cv;
    std::vector<LanePending> pending(lanes);
    std::vector<size_t> completed_lanes;
    completed_lanes.reserve((size_t)total);
    std::vector<char> lane_busy(lanes, 0);

    int completed = 0;
    int fatal_rc = 0;
    bool stop_collector = false;

    std::thread collector([&]() {
        while (true) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (stop_collector || fatal_rc != 0) return;
            }

            bool progressed = false;
            for (size_t lane = 0; lane < lanes; ++lane) {
                bool active = false;
                double submit_done_us = 0.0;
                bool record = false;
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    if (pending[lane].active) {
                        active = true;
                        submit_done_us = pending[lane].submit_done_us;
                        record = pending[lane].record;
                    }
                }
                if (!active) continue;

                // CUDA completion side: query lane event and finalize when ready.
                cudaError_t q = cudaEventQuery(done_events[lane]);
                if (q == cudaErrorNotReady) {
                    (void)cudaGetLastError();
                    continue;
                }
                if (q != cudaSuccess) {
                    std::lock_guard<std::mutex> lk(mtx);
                    if (fatal_rc == 0) fatal_rc = (int)q;
                    cv.notify_all();
                    return;
                }

                progressed = true;
                const double done_us = now_us();
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    if (!pending[lane].active) continue;
                    if (record) {
                        // Transfer latency: submit-done -> event completion observed.
                        transfer_samples.push_back(done_us - submit_done_us);
                        measure_end_us = done_us;
                    }
                    pending[lane].active = false;
                    completed++;
                    completed_lanes.push_back(lane);
                }
                cv.notify_all();
            }

            if (!progressed) std::this_thread::yield();
        }
    });

    auto reclaim_completed_lanes = [&]() -> int {
        std::lock_guard<std::mutex> lk(mtx);
        if (fatal_rc != 0) return fatal_rc;
        for (size_t lane : completed_lanes) {
            if (lane < lane_busy.size()) lane_busy[lane] = 0;
        }
        completed_lanes.clear();
        return 0;
    };

    int issued = 0;
    size_t rr_lane = 0;
    while (issued < total) {
        int frc = reclaim_completed_lanes();
        if (frc != 0) break;

        // Same scheduling policy as GDR path: round-robin across free lanes.
        size_t lane = lanes;
        for (size_t probe = 0; probe < lanes; ++probe) {
            const size_t cand = (rr_lane + probe) % lanes;
            if (!lane_busy[cand]) {
                lane = cand;
                rr_lane = (cand + 1) % lanes;
                break;
            }
        }
        if (lane == lanes) {
            // All lanes are in-flight; wait until collector releases one.
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&]() {
                return fatal_rc != 0 || !completed_lanes.empty();
            });
            continue;
        }

        lane_busy[lane] = 1;
        const double t0 = now_us();
        cudaError_t ce = cudaMemcpyAsync(dst_lanes[lane], src_lanes[lane], bytes, kind, streams[lane]);
        const double t1 = now_us();
        if (ce != cudaSuccess) {
            lane_busy[lane] = 0;
            std::lock_guard<std::mutex> lk(mtx);
            if (fatal_rc == 0) fatal_rc = (int)ce;
            cv.notify_all();
            break;
        }
        ce = cudaEventRecord(done_events[lane], streams[lane]);
        if (ce != cudaSuccess) {
            lane_busy[lane] = 0;
            std::lock_guard<std::mutex> lk(mtx);
            if (fatal_rc == 0) fatal_rc = (int)ce;
            cv.notify_all();
            break;
        }

        const bool record = (issued >= warmup);
        if (record) {
            // Issue latency: cudaMemcpyAsync call return time.
            issue_samples.push_back(t1 - t0);
            if (measure_begin_us < 0.0) measure_begin_us = t1;
        }

        {
            std::lock_guard<std::mutex> lk(mtx);
            pending[lane].active = true;
            pending[lane].submit_done_us = t1;
            pending[lane].record = record;
        }
        issued++;
    }

    {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&]() { return fatal_rc != 0 || completed >= issued; });
        stop_collector = true;
        cv.notify_all();
    }
    if (collector.joinable()) collector.join();
    for (auto& evt : done_events) {
        if (evt) cudaEventDestroy(evt);
    }

    if (fatal_rc != 0) {
        fprintf(stderr, "[issue] cuda pipeline failed: rc=%d(%s) bytes=%zu\n",
                fatal_rc, cudaGetErrorString((cudaError_t)fatal_rc), bytes);
        std::exit(2);
    }

    BenchPair out{};
    out.issue = analyse(issue_samples, bytes);
    out.transfer = analyse(transfer_samples, bytes);
    if (measure_begin_us > 0.0 && measure_end_us > measure_begin_us) {
        // Throughput window aligned with GDR path for fair comparison.
        const double span_us = measure_end_us - measure_begin_us;
        out.transfer.bw_GBs = ((double)bytes * (double)iters / 1e9) / (span_us / 1e6);
    }
    return out;
}

static std::vector<std::shared_ptr<GDRCopyChannel>>
open_parallel_channels(int gpu_id, const std::string& nic_name, int lanes) {
    std::vector<std::shared_ptr<GDRCopyChannel>> channels;
    channels.reserve((size_t)lanes);

    GDRCopyLib::shutdown();
    for (int i = 0; i < lanes; ++i) {
        auto ch = GDRCopyLib::open(gpu_id, nic_name);
        channels.push_back(ch);
        // Clear global cache so next open() creates another independent channel/QP.
        GDRCopyLib::shutdown();
    }
    return channels;
}

static GDRStats aggregate_stats(const std::vector<std::shared_ptr<GDRCopyChannel>>& channels) {
    GDRStats out{};
    for (const auto& ch : channels) {
        GDRStats s = ch->stats();
        out.total_bytes  += s.total_bytes;
        out.total_ops    += s.total_ops;
        out.rdma_ops     += s.rdma_ops;
        out.fallback_ops += s.fallback_ops;
    }
    return out;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    int         gpu_id   = (argc > 1) ? std::atoi(argv[1]) : 0;
    std::string nic_name = (argc > 2) ? argv[2]            : "mlx5_0";
    int parallel_reqs    = (argc > 3) ? std::atoi(argv[3]) : 1;
    if (parallel_reqs < 1) parallel_reqs = 1;

    printf("=================================================================\n");
    printf("  GDR Copy Benchmark  —  GPU %d  NIC %s\n", gpu_id, nic_name.c_str());
    printf("=================================================================\n\n");

    // ── CUDA setup ────────────────────────────────────────────────────────
    cudaSetDevice(gpu_id);
    struct cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, gpu_id);
    printf("GPU: %s  (PCIe gen%d x%d)\n\n",
           prop.name, prop.pciBusID, prop.pciDeviceID);

    // ── Open N independent channels (one QP per channel) ─────────────────
    std::vector<std::shared_ptr<GDRCopyChannel>> gdr_channels;
    try {
        gdr_channels = open_parallel_channels(gpu_id, nic_name, parallel_reqs);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to open GDR channels: %s\n", e.what());
        return 1;
    }
    if (gdr_channels.empty()) {
        fprintf(stderr, "Failed to create any GDR channel.\n");
        return 1;
    }

    GDRStats s = gdr_channels[0]->stats();
    bool gdr_active = (s.fallback_ops == 0);
    printf("GPUDirect RDMA path: %s\n\n", gdr_active ? "ACTIVE" : "FALLBACK (cudaMemcpy)");
    printf("Benchmark mode: fixed parallel in-flight (%d), one QP per lane, global CQ listener\n\n",
           parallel_reqs);

    for (auto& ch : gdr_channels) ch->reset_stats();
    GdrCqListener gdr_cq_listener(gdr_channels);

    // ── Transfer sizes to sweep ───────────────────────────────────────────
    std::vector<size_t> sizes;
    for (size_t ssz = 4096; ssz <= 64ULL << 20; ssz *= 4)
        sizes.push_back(ssz);

    static const int WARMUP = 100;
    static const int ITERS  = 10000;

    std::vector<cudaStream_t> issue_streams((size_t)parallel_reqs, nullptr);
    for (int lane = 0; lane < parallel_reqs; ++lane) {
        cudaError_t ce = cudaStreamCreate(&issue_streams[(size_t)lane]);
        if (ce != cudaSuccess) {
            fprintf(stderr, "[issue] cudaStreamCreate failed: lane=%d err=%s\n",
                    lane, cudaGetErrorString(ce));
            std::exit(2);
        }
    }

    std::vector<DirectionRow> h2d_rows;
    h2d_rows.reserve(sizes.size());
    for (size_t bytes : sizes) {
        std::vector<void*> h_src_lanes((size_t)parallel_reqs, nullptr);
        std::vector<void*> d_dst_lanes((size_t)parallel_reqs, nullptr);
        for (int lane = 0; lane < parallel_reqs; ++lane) {
            cudaHostAlloc(&h_src_lanes[(size_t)lane], bytes, cudaHostAllocPortable);
            cudaMalloc(&d_dst_lanes[(size_t)lane], bytes);
            cudaMemset(d_dst_lanes[(size_t)lane], 0, bytes);
            memset(h_src_lanes[(size_t)lane], (lane & 1) ? 0xA5 : 0x5A, bytes);
        }

        BenchPair gdr  = run_gdr_timings(
            gdr_channels, gdr_cq_listener, d_dst_lanes, h_src_lanes,
            bytes, GDR_H2D, WARMUP, ITERS);
        BenchPair cuda = run_cuda_timings(
            d_dst_lanes, h_src_lanes, bytes, cudaMemcpyHostToDevice,
            WARMUP, ITERS, issue_streams);

        h2d_rows.push_back(DirectionRow{bytes, gdr, cuda});

        for (int lane = 0; lane < parallel_reqs; ++lane) {
            cudaFreeHost(h_src_lanes[(size_t)lane]);
            cudaFree(d_dst_lanes[(size_t)lane]);
        }
    }

    print_latency_table("Host->Device Issue Latency", h2d_rows, true);
    print_latency_table("Host->Device Transfer Latency", h2d_rows, false);

    std::vector<DirectionRow> d2h_rows;
    d2h_rows.reserve(sizes.size());
    for (size_t bytes : sizes) {
        std::vector<void*> d_src_lanes((size_t)parallel_reqs, nullptr);
        std::vector<void*> h_dst_lanes((size_t)parallel_reqs, nullptr);
        for (int lane = 0; lane < parallel_reqs; ++lane) {
            cudaMalloc(&d_src_lanes[(size_t)lane], bytes);
            cudaHostAlloc(&h_dst_lanes[(size_t)lane], bytes, cudaHostAllocPortable);
            cudaMemset(d_src_lanes[(size_t)lane], (lane & 1) ? 0x5A : 0xA5, bytes);
            memset(h_dst_lanes[(size_t)lane], 0, bytes);
        }

        BenchPair gdr  = run_gdr_timings(
            gdr_channels, gdr_cq_listener, h_dst_lanes, d_src_lanes,
            bytes, GDR_D2H, WARMUP, ITERS);
        BenchPair cuda = run_cuda_timings(
            h_dst_lanes, d_src_lanes, bytes, cudaMemcpyDeviceToHost,
            WARMUP, ITERS, issue_streams);

        d2h_rows.push_back(DirectionRow{bytes, gdr, cuda});

        for (int lane = 0; lane < parallel_reqs; ++lane) {
            cudaFree(d_src_lanes[(size_t)lane]);
            cudaFreeHost(h_dst_lanes[(size_t)lane]);
        }
    }

    print_latency_table("Device->Host Issue Latency", d2h_rows, true);
    print_latency_table("Device->Host Transfer Latency", d2h_rows, false);

    for (auto& stream : issue_streams) {
        if (stream) cudaStreamDestroy(stream);
    }

    GDRStats final_s = aggregate_stats(gdr_channels);
    printf("\n=================================================================\n");
    printf("Total ops: %lu  (RDMA: %lu  Fallback: %lu)\n",
           final_s.total_ops, final_s.rdma_ops, final_s.fallback_ops);
    printf("Total bytes: %.2f GiB\n", final_s.total_bytes / (double)(1ULL << 30));
    printf("=================================================================\n");

    gdr_cq_listener.stop();
    GDRCopyLib::shutdown();
    return 0;
}
