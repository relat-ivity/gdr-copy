/**
 * demo.cpp  —  minimal runnable example of the GDR Copy API
 *
 * This is what a developer integrating the library would write.
 * Shows:
 *   1. One-time channel open
 *   2. H2D transfer (replacing cudaMemcpy HostToDevice)
 *   3. D2H transfer (replacing cudaMemcpy DeviceToHost)
 *   4. Data correctness check
 *   5. Statistics printout
 */

#include "gdr_copy.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

static void die(const char* msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    std::exit(1);
}

int main(int argc, char** argv)
{
    int         gpu_id   = (argc > 1) ? std::atoi(argv[1]) : 0;
    std::string nic_name = (argc > 2) ? argv[2]            : "mlx5_0";

    printf("GDR Copy Demo — GPU %d  NIC %s\n\n", gpu_id, nic_name.c_str());

    // ── 1. One-time init (shared_ptr; safe to hold globally) ──────────────
    //
    //    Before: nothing
    //    After:  RC QP connected loopback, GPUDirect capability probed,
    //            pinned host pool pre-allocated
    //
    std::shared_ptr<GDRCopyChannel> ch;
    try {
        ch = GDRCopyLib::open(gpu_id, nic_name);
    } catch (const std::exception& e) {
        fprintf(stderr, "GDRCopyLib::open failed: %s\n", e.what());
        fprintf(stderr, "Is MLNX_OFED installed? ibv_devinfo will show NIC names.\n");
        return 1;
    }

    // ── 2. Allocate buffers ───────────────────────────────────────────────
    const size_t N     = 1024 * 1024;           // 1M floats
    const size_t bytes = N * sizeof(float);

    float* h_input  = nullptr;
    float* h_output = nullptr;
    float* d_buf    = nullptr;

    cudaHostAlloc(&h_input,  bytes, cudaHostAllocPortable);
    cudaHostAlloc(&h_output, bytes, cudaHostAllocPortable);
    cudaMalloc(&d_buf, bytes);

    if (!h_input || !h_output || !d_buf) die("allocation failed");

    // Fill input
    for (size_t i = 0; i < N; i++) h_input[i] = (float)i * 0.001f;
    memset(h_output, 0, bytes);

    // ── 3. H2D: host → GPU ───────────────────────────────────────────────
    //
    //    Replace:  cudaMemcpy(d_buf, h_input, bytes, cudaMemcpyHostToDevice)
    //    With:     ch->memcpy(d_buf, h_input, bytes, GDR_H2D)
    //
    printf("[H2D] Transferring %.2f MiB ... ", bytes / (1.0 * (1 << 20)));
    fflush(stdout);

    int rc = ch->memcpy(d_buf, h_input, bytes, GDR_H2D);
    if (rc != 0) die("H2D transfer failed");

    printf("done  (%.2f µs)\n", ch->stats().last_latency_us);

    // ── 4. GPU-side work (trivial: just verify data is there) ─────────────
    //    In a real application: launch CUDA kernels on d_buf here.

    // ── 5. D2H: GPU → host ───────────────────────────────────────────────
    //
    //    Replace:  cudaMemcpy(h_output, d_buf, bytes, cudaMemcpyDeviceToHost)
    //    With:     ch->memcpy(h_output, d_buf, bytes, GDR_D2H)
    //
    printf("[D2H] Transferring %.2f MiB ... ", bytes / (1.0 * (1 << 20)));
    fflush(stdout);

    rc = ch->memcpy(h_output, d_buf, bytes, GDR_D2H);
    if (rc != 0) die("D2H transfer failed");

    printf("done  (%.2f µs)\n", ch->stats().last_latency_us);

    // ── 6. Correctness check ──────────────────────────────────────────────
    bool ok = true;
    for (size_t i = 0; i < N; i++) {
        if (std::abs(h_output[i] - h_input[i]) > 1e-6f) {
            fprintf(stderr, "Mismatch at [%zu]: got %f expected %f\n",
                    i, h_output[i], h_input[i]);
            ok = false;
            break;
        }
    }
    printf("\nData verification: %s\n", ok ? "PASSED ✓" : "FAILED ✗");

    // ── 7. Stats ──────────────────────────────────────────────────────────
    GDRStats s = ch->stats();
    printf("\n--- Transfer Statistics ---\n");
    printf("  Total ops   : %lu\n",    s.total_ops);
    printf("  RDMA ops    : %lu\n",    s.rdma_ops);
    printf("  Fallback ops: %lu\n",    s.fallback_ops);
    printf("  Total bytes : %.2f MiB\n", s.total_bytes / (double)(1 << 20));
    printf("  Avg latency : %.2f µs\n",  s.avg_latency_us);

    if (s.fallback_ops > 0)
        printf("\n  NOTE: Some ops used cudaMemcpy fallback.\n"
               "  To use RDMA: ensure nvidia-peermem is loaded:\n"
               "    sudo modprobe nvidia-peermem\n"
               "  and that GPU/NIC share a PCIe switch.\n");

    // ── 8. Cleanup ────────────────────────────────────────────────────────
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_buf);
    GDRCopyLib::shutdown();

    return ok ? 0 : 1;
}
