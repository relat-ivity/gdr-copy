#!/usr/bin/env bash
# install_deps.sh  —  prepare an H20 node to run gdr_copy
#
# Tested on: Ubuntu 22.04, CUDA 12.2, MLNX_OFED 23.10, driver 535+
# Run as root or with sudo.
set -euo pipefail

info()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
fatal() { echo -e "\033[1;31m[FAIL]\033[0m  $*"; exit 1; }

# ── 1. Base packages ──────────────────────────────────────────────────────────
info "Installing build dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build \
    pciutils numactl \
    linux-headers-$(uname -r)

# ── 2. MLNX_OFED ─────────────────────────────────────────────────────────────
# Mellanox/NVIDIA OFED provides libibverbs with GPUDirect support.
# Download from: https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
# Adjust version/OS as needed.
OFED_VER="${OFED_VER:-23.10-2.1.3.1}"
OS_VER=$(lsb_release -rs)          # e.g. "22.04"
ARCH=$(uname -m)                   # x86_64 | aarch64

if command -v ibv_devinfo &>/dev/null; then
    info "MLNX_OFED already installed ($(ofed_info -s 2>/dev/null || echo unknown))"
else
    TARBALL="MLNX_OFED_LINUX-${OFED_VER}-ubuntu${OS_VER}-${ARCH}.tgz"
    URL="https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VER}/${TARBALL}"

    info "Downloading MLNX_OFED ${OFED_VER}..."
    wget -q --show-progress "${URL}" -O /tmp/${TARBALL}

    info "Extracting and installing..."
    tar -C /tmp -xzf /tmp/${TARBALL}
    DIR="/tmp/MLNX_OFED_LINUX-${OFED_VER}-ubuntu${OS_VER}-${ARCH}"
    ${DIR}/mlnxofedinstall \
        --without-fw-update \
        --force \
        --add-kernel-support   # rebuild kernel modules for current kernel
    /etc/init.d/openibd restart || systemctl restart openibd || true
    rm -rf /tmp/${TARBALL} "${DIR}"
fi

# ── 3. nvidia-peermem kernel module ──────────────────────────────────────────
# This is the critical module that allows ibv_reg_mr to pin GPU memory.
# On modern CUDA/driver stacks it ships with the NVIDIA driver.
info "Loading nvidia-peermem..."
if ! lsmod | grep -q nvidia_peermem; then
    modprobe nvidia-peermem 2>/dev/null \
    || modprobe nv_peer_mem  2>/dev/null \
    || warn "nvidia-peermem not found — GDR will fall back to cudaMemcpy.
         Check: ls /lib/modules/$(uname -r)/kernel/drivers/infiniband/
         Install: apt install nvidia-peer-memory-dkms  (or via DKMS)"
fi

if lsmod | grep -q nvidia.peer; then
    info "nvidia-peermem active ✓"
    # Make it persistent
    echo "nvidia-peermem" >> /etc/modules-load.d/gdr_copy.conf 2>/dev/null || true
fi

# ── 4. GPUDirect tuning (PCIe 5.0 / H20 specific) ───────────────────────────
info "Applying PCIe / RDMA tuning..."

# Disable IOMMU passthrough (prevents DMA remapping overhead)
# WARNING: reduces security isolation; use only in trusted ML cluster env.
if grep -q "intel_iommu=on" /proc/cmdline; then
    warn "IOMMU is enabled — may add ~0.5 µs latency. Consider:
         GRUB_CMDLINE_LINUX='intel_iommu=on iommu=pt'
         in /etc/default/grub (iommu=pt enables passthrough, preserving IOMMU
         for PCI isolation while removing DMA remapping overhead)."
fi

# HugePages: reduce TLB pressure on GPU BAR1 MMIO mappings
if [ "$(cat /proc/sys/vm/nr_hugepages)" -lt 512 ]; then
    echo 512 > /proc/sys/vm/nr_hugepages
    info "Set nr_hugepages=512"
fi

# Maximize RDMA send/recv queue sizes
if command -v mlnx_qos &>/dev/null; then
    mlnx_qos -i $(ibv_devinfo | awk '/hca_id/{print $2; exit}') \
             --pfc 1,1,1,1,1,1,1,1 2>/dev/null || true
fi

# ── 5. Verify ─────────────────────────────────────────────────────────────────
info "Verification..."
ibv_devinfo | grep -E "hca_id|port_state|link_layer" \
    || fatal "No RDMA device visible. Check cable and HCA."

nvidia-smi -q | grep -E "Product Name|BAR1" \
    || fatal "No NVIDIA GPU found."

info ""
info "All dependencies installed successfully."
info ""
info "Quick sanity:"
info "  ibv_devinfo          — list RDMA ports"
info "  lsmod | grep peer    — confirm nvidia-peermem"
info "  nvidia-smi -q        — check BAR1 size (H20 should show 128GB+)"
