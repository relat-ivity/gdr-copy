# gdr_copy — GPUDirect RDMA Copy Library

Drop-in replacement for `cudaMemcpy` that routes H2D/D2H transfers
through a NIC co-located on the same PCIe switch as the GPU, eliminating
kernel-mode transitions and CPU-side DMA scheduling.

---

## Background and motivation

Standard `cudaMemcpy` on H2D/D2H:

```
CPU (user mode)
   │  cudaMemcpy()
   ↓
kernel IOCTL  ←── mode switch #1
   │
   ↓  UVM fault / DMA engine programming
PCIe Root Complex
   │
   ↓  PCIe transaction
GPU BAR1
```

With GPUDirect RDMA (NIC and GPU under the same PCIe switch):

```
CPU writes → pinned host buffer (cache-coherent, ~1 ns/word)
NIC DMA reads pinned buffer → GPU BAR1 (bypasses CPU entirely)
```

Latency benefit is largest for small IOs (< 1 MiB) where the IOCTL overhead
of `cudaMemcpy` dominates. The NIC also provides hardware flow control
between collective-comm (NCCL) and KVCache traffic, avoiding PCIe congestion
(as described in DualPath, NSDI'25).

---

## Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | NVIDIA H20 (Hopper) | H20 SXM |
| PCIe | Gen 5 x16 | Gen 5 x16 |
| NIC | ConnectX-6/7 | ConnectX-7 (mlx5_0) |
| Topology | GPU + NIC under same PLX switch | NVSwitch + PCIe 5.0 |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| CUDA | 12.0+ | 12.2 |
| Driver | 525+ | 535.104 |
| MLNX_OFED | 5.9+ | 23.10 |
| Kernel module | `nvidia-peermem` | loaded |

**Key requirement**: `nvidia-peermem` (or `nv_peer_mem`) must be loaded.
Without it, `ibv_reg_mr` cannot pin GPU memory and the library automatically
falls back to `cudaMemcpy`.

```bash
lsmod | grep nvidia_peermem   # must show a line
sudo modprobe nvidia-peermem  # if not loaded
```

---

## Quick start

```bash
# 1. Install dependencies (once per node)
sudo bash scripts/install_deps.sh

# 2. Build
bash scripts/build.sh

# 3. Run correctness demo
sudo ./build/demo 0 mlx5_0

# 4. Run latency/bandwidth benchmark
sudo ./build/bench 0 mlx5_0
```

---

## API reference

### `GDRCopyLib::open`

```cpp
#include "gdr_copy.h"

std::shared_ptr<GDRCopyChannel>
GDRCopyLib::open(int gpu_id, const std::string& nic_name, bool use_odp = false);
```

Opens (or retrieves a cached) channel for the given GPU + NIC pair.
Internally this:
1. Opens the RDMA device context
2. Creates a loopback RC QP (RESET→INIT→RTR→RTS)
3. Pre-allocates a pool of 32 MiB of pinned host buffers
4. Probes GPUDirect availability via a test `ibv_reg_mr` on a tiny GPU buffer

| Parameter | Description |
|-----------|-------------|
| `gpu_id` | CUDA device ordinal (0-based, as in `cudaSetDevice`) |
| `nic_name` | RDMA device name from `ibv_devinfo` (e.g. `"mlx5_0"`) |
| `use_odp` | Enable On-Demand Paging (ODP). Slower first-touch, no pre-pinning. Leave false for lowest latency. |

Throws `std::runtime_error` if the RDMA device is not found or GPU init fails.

---

### `GDRCopyChannel::memcpy`

```cpp
int ch->memcpy(void* dst, const void* src, size_t bytes, GDRCopyKind kind);
```

Synchronous transfer. Blocks until the NIC completes the DMA.

| `kind` | Direction | RDMA verb | Notes |
|--------|-----------|-----------|-------|
| `GDR_H2D` | host → GPU | `RDMA_WRITE` | CPU memcpy to pinned buf, then NIC DMA |
| `GDR_D2H` | GPU → host | `RDMA_READ`  | NIC DMA from GPU, then CPU memcpy to dst |
| `GDR_D2D` | GPU → GPU  | (fallback) | `cudaMemcpy` DeviceToDevice (CE engine is optimal here) |

Returns 0 on success, negative errno-style on failure.

**Migration from `cudaMemcpy`:**

```cpp
// Before
cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost);

// After (one-time init at program start)
auto ch = GDRCopyLib::open(0, "mlx5_0");

// Drop-in replacement
ch->memcpy(d_ptr, h_ptr, bytes, GDR_H2D);
ch->memcpy(h_ptr, d_ptr, bytes, GDR_D2H);
```

---

### `GDRCopyChannel::memcpy_async` / `sync`

```cpp
int ch->memcpy_async(void* dst, const void* src, size_t bytes, GDRCopyKind kind);
int ch->sync();
```

Post a transfer without blocking, then call `sync()` to wait.
Currently implemented as serialized `memcpy` + no-op `sync`.
A future version will use a dedicated completion thread and overlapping WQEs.

---

### `GDRCopyChannel::stats`

```cpp
GDRStats s = ch->stats();
```

```cpp
struct GDRStats {
    double   last_latency_us;   // wall time of last memcpy (µs)
    double   avg_latency_us;    // running mean (µs)
    uint64_t total_bytes;       // cumulative bytes
    uint64_t total_ops;         // cumulative call count
    uint64_t rdma_ops;          // ops served by RDMA path
    uint64_t fallback_ops;      // ops that fell back to cudaMemcpy
};
```

If `fallback_ops > 0` and `rdma_ops == 0`, GPUDirect is not working.
Check `nvidia-peermem` and PCIe topology.

---

### `GDRCopyLib::probe`

```cpp
bool GDRCopyLib::probe(int gpu_id, const std::string& nic_name);
```

Returns `true` if RDMA path is functional. Useful for capability detection
at startup without committing to a full channel.

---

### `GDRCopyLib::shutdown`

```cpp
GDRCopyLib::shutdown();
```

Destroy all cached channels, deregister all MRs, deallocate pinned buffers.
Called automatically at process exit via `std::shared_ptr` refcount.

---

## Internal architecture

```
GDRCopyLib (static factory, mutex-protected channel cache)
    └── GDRCopyChannelImpl (one per gpu_id × nic_name)
         ├── ibv_context / pd / cq
         ├── RC QP (loopback, RESET→INIT→RTR→RTS)
         │    └── connect_rc_qp() — loopback: remote QPN == local QPN
         ├── MRCache (LRU, 256 entries)
         │    └── ibv_reg_mr(gpu_va, len)  ← nvidia-peermem intercepts this
         └── PinnedPool (8 × 4 MiB slots, pre-registered)
              └── cudaHostAlloc + ibv_reg_mr (done once at open())
```

### H2D data path

```
user src (any host ptr)
   │ ::memcpy (L1/L2 cached, ~10 GB/s)
   ↓
PinnedSlot.host_ptr (pinned, registered with NIC lkey)
   │ ibv_post_send(RDMA_WRITE, lkey=slot->lkey, rkey=gpu_mr->rkey)
   ↓  NIC DMA over PCIe (no CPU in critical path)
GPU BAR1 (dst)
```

### D2H data path

```
GPU BAR1 (src, registered via nvidia-peermem, rkey)
   │ ibv_post_send(RDMA_READ, lkey=slot->lkey, rkey=gpu_mr->rkey)
   ↓  NIC DMA over PCIe
PinnedSlot.host_ptr
   │ ::memcpy
   ↓
user dst (any host ptr)
```

---

## Why the NIC must be a loopback RC QP

For a single-node H2D/D2H, both endpoints of the RDMA operation are on the
same host. We connect the QP to itself (remote QPN = local QPN). The RC QP
type is required because:

- `RDMA_WRITE` and `RDMA_READ` are only defined on RC (Reliable Connected) QPs
- UD (Unreliable Datagram) QPs only support `SEND`/`RECV` — no remote memory access
- The original code in the provided snippet used `IBV_QPT_UD`, which would
  cause `ibv_post_send(RDMA_WRITE)` to return `EINVAL` immediately

---

## Performance expectations on H20 + ConnectX-7

| Transfer size | cudaMemcpy (H2D) | GDR RDMA (H2D) | Speedup |
|--------------|-----------------|----------------|---------|
| 4 KiB        | ~12 µs          | ~2.5 µs        | ~5×     |
| 64 KiB       | ~15 µs          | ~4 µs          | ~3.5×   |
| 1 MiB        | ~30 µs          | ~12 µs         | ~2.5×   |
| 16 MiB       | ~200 µs         | ~120 µs        | ~1.7×   |
| 64 MiB       | ~750 µs         | ~500 µs        | ~1.5×   |

Numbers are indicative; actual results depend on PCIe topology, NUMA
placement, and system load. Run `./build/bench` for your exact configuration.

**Where GDR helps most**: KVCache token streaming (typically 4–256 KiB
per decode step) and small gradient scatters, where IOCTL overhead of
`cudaMemcpy` dominates transfer time.

---

## Common issues

### `ibv_reg_mr on GPU VA failed (errno=12)`

Out of memory — either GPU memory is exhausted or `nvidia-peermem` could not
pin the pages. Check `nvidia-smi` for memory usage.

### `ibv_reg_mr on GPU VA failed (errno=22)` (EINVAL)

`nvidia-peermem` is not loaded. Run:
```bash
sudo modprobe nvidia-peermem
lsmod | grep nvidia_peermem
```

### `QP INIT→RTR failed`

Port state is not ACTIVE. Check cable and run:
```bash
ibv_devinfo -v | grep port_state
```

### `poll_cq timeout after 5000 µs`

RDMA operation did not complete. Possible causes:
- QP not in RTS state (check `ibv_query_qp`)
- MTU mismatch between QP attr and port (`ibv_devinfo -v | grep active_mtu`)
- nvidia-peermem unloaded mid-run

### Fallback path active (`fallback_ops > 0`)

The library is silently using `cudaMemcpy`. Common causes:
1. `nvidia-peermem` not loaded
2. GPU and NIC are not behind the same PCIe switch (no peer access)
3. Running inside a container without `--device=/dev/infiniband/...`

Verify topology:
```bash
nvidia-smi topo -m         # should show NIC under same switch as GPU
ibv_devinfo                # confirm NIC is visible
lsmod | grep nvidia        # should include nvidia_peermem
```

---

## License

Apache 2.0
